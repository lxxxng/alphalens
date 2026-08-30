"""
AlphaLens - SEC Chunk Embedding + FAISS Index Builder

Purpose
-------
Convert SEC filing chunks into OpenAI embedding vectors and
store those vectors inside a local FAISS index.


Current pipeline
----------------

filings
    ↓
filing_sections
    ↓
filing_chunks
    ↓
OpenAI embeddings
    ↓
FAISS index


PostgreSQL stores:
    chunk_id
    ticker
    filing metadata
    section metadata
    original chunk text

FAISS stores:
    chunk_id
    embedding vector


Example
-------

PostgreSQL:

    chunk_id = 12562

    ticker = NVDA

    section =
        Risk Factors

    content =
        "Cybersecurity incidents could..."


FAISS:

    ID:
        12562

    vector:
        [0.019, -0.041, 0.006, ...]


Later:

    user question
         ↓
    embed question
         ↓
    FAISS similarity search
         ↓
    chunk IDs
         ↓
    PostgreSQL
         ↓
    original text + metadata
"""

import json
import os
import time

from pathlib import Path

import faiss
import numpy as np

from dotenv import load_dotenv

from openai import (
    APIConnectionError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)

from sqlalchemy import (
    MetaData,
    Table,
    create_engine,
    func,
    select,
    update,
)


# ============================================================
# Environment
# ============================================================

load_dotenv()


# ============================================================
# Embedding Configuration
# ============================================================

# OpenAI embedding model.
#
# We use the small model because:
#
#     - SEC documents are English text
#     - it is inexpensive
#     - it is suitable for semantic search
#
EMBEDDING_MODEL = "text-embedding-3-small"


# ============================================================
# API Batch Configuration
# ============================================================
#
# OpenAI allows multiple input strings in one embedding
# request.
#
# Sending chunks in batches is much more efficient than:
#
#     one HTTP request
#     per chunk
#
# Our chunks are max ~700 tokens.
#
# We additionally limit the approximate number of tokens in
# one API request.
# ============================================================

MAX_BATCH_ITEMS = 256

MAX_BATCH_TOKENS = 180_000


# ============================================================
# Retry Configuration
# ============================================================

MAX_RETRIES = 5

INITIAL_RETRY_SECONDS = 2


# ============================================================
# FAISS Files
# ============================================================

FAISS_DIRECTORY = Path(
    "data/faiss"
)


INDEX_PATH = (
    FAISS_DIRECTORY
    / "sec_chunks.faiss"
)


METADATA_PATH = (
    FAISS_DIRECTORY
    / "sec_chunks.meta.json"
)


# ============================================================
# get_database_engine()
# ============================================================

def get_database_engine():
    """
    Create the PostgreSQL SQLAlchemy engine.

    DATABASE_URL comes from .env.
    """

    database_url = os.getenv(
        "DATABASE_URL"
    )

    if not database_url:

        raise ValueError(
            "DATABASE_URL was not found in .env."
        )

    return create_engine(
        database_url,
        pool_pre_ping=True,
    )


# ============================================================
# get_openai_client()
# ============================================================

def get_openai_client() -> OpenAI:
    """
    Create an authenticated OpenAI API client.

    OPENAI_API_KEY comes from:

        .env

    We never hard-code the API key in Python.
    """

    api_key = os.getenv(
        "OPENAI_API_KEY"
    )

    if not api_key:

        raise ValueError(
            "OPENAI_API_KEY was not found in .env."
        )

    return OpenAI(
        api_key=api_key
    )


# ============================================================
# load_existing_index()
# ============================================================

def load_existing_index():
    """
    Load an existing FAISS index if one already exists.

    Returns
    -------
    faiss.Index or None

    Why?
    ----
    This lets our embedding job resume.

    Example:

        Run 1:
            10,000 vectors indexed
            program stops

        Run 2:
            load those 10,000 vectors
            continue with remaining chunks
    """

    if not INDEX_PATH.exists():

        return None


    # --------------------------------------------------------
    # Require metadata file too.
    # --------------------------------------------------------
    #
    # The metadata tells us which OpenAI model produced the
    # existing vectors.
    #
    # Mixing different embedding models inside the same
    # index would be incorrect.
    # --------------------------------------------------------

    if not METADATA_PATH.exists():

        raise RuntimeError(
            "FAISS index exists but metadata file is missing.\n"
            "Either restore the metadata file or rebuild the "
            "index."
        )


    metadata = json.loads(
        METADATA_PATH.read_text(
            encoding="utf-8"
        )
    )


    existing_model = metadata.get(
        "embedding_model"
    )


    if existing_model != EMBEDDING_MODEL:

        raise RuntimeError(
            "Existing FAISS index uses a different "
            "embedding model.\n"
            f"Existing: {existing_model}\n"
            f"Configured: {EMBEDDING_MODEL}\n\n"
            "Rebuild the FAISS index before changing models."
        )


    index = faiss.read_index(
        str(INDEX_PATH)
    )


    print(
        f"Loaded existing FAISS index: "
        f"{index.ntotal:,} vectors"
    )


    return index


# ============================================================
# get_existing_faiss_ids()
# ============================================================

def get_existing_faiss_ids(
    index,
) -> set[int]:
    """
    Get every PostgreSQL chunk_id already stored in FAISS.

    FAISS IndexIDMap2 stores an internal ID map.

    Example:

        [1, 2, 3, 4, 5, ...]

    These IDs correspond directly to:

        filing_chunks.chunk_id
    """

    if index is None:

        return set()


    if index.ntotal == 0:

        return set()


    # Convert FAISS's internal ID vector into a NumPy array.
    id_array = faiss.vector_to_array(
        index.id_map
    )


    return set(
        int(value)
        for value in id_array
    )


# ============================================================
# get_all_chunks()
# ============================================================

def get_all_chunks(
    engine,
    chunk_table,
):
    """
    Retrieve all SEC chunks that could potentially be embedded.

    We retrieve:

        chunk_id
        content
        token_count

    chunk_id:
        becomes the FAISS vector ID.

    content:
        is sent to OpenAI.

    token_count:
        helps construct safe API batches.
    """

    query = (
        select(
            chunk_table.c.chunk_id,
            chunk_table.c.content,
            chunk_table.c.token_count,
        )
        .order_by(
            chunk_table.c.chunk_id
        )
    )


    with engine.connect() as connection:

        result = connection.execute(
            query
        )


        chunks = (
            result
            .mappings()
            .all()
        )


    return chunks


# ============================================================
# synchronize_database_status()
# ============================================================

def synchronize_database_status(
    engine,
    chunk_table,
    existing_ids: set[int],
):
    """
    Synchronize PostgreSQL status with the FAISS index.

    Why is this useful?
    -------------------

    Consider this rare situation:

        1. vectors successfully written to FAISS
        2. program crashes
        3. PostgreSQL status wasn't updated yet

    FAISS may contain:

        chunk 123

    while PostgreSQL still says:

        PENDING

    On restart we trust the actual FAISS index and update
    PostgreSQL accordingly.

    This prevents duplicate vector insertion.
    """

    if not existing_ids:

        return


    ids = list(
        existing_ids
    )


    # Don't create one enormous SQL IN (...) statement.
    #
    # Process IDs in groups of 1,000.
    update_batch_size = 1000


    with engine.begin() as connection:

        for start in range(
            0,
            len(ids),
            update_batch_size,
        ):

            batch_ids = ids[
                start:
                start + update_batch_size
            ]


            connection.execute(
                update(
                    chunk_table
                )
                .where(
                    chunk_table.c.chunk_id.in_(
                        batch_ids
                    )
                )
                .values(
                    embedding_status="EMBEDDED",
                    embedding_model=EMBEDDING_MODEL,
                    embedding_error=None,
                )
            )


# ============================================================
# validate_index_ids()
# ============================================================

def validate_index_ids(
    database_chunks,
    existing_ids: set[int],
):
    """
    Make sure FAISS does not contain chunk IDs that no longer
    exist in PostgreSQL.

    Why could this happen?
    ----------------------

    Our chunker can be rerun.

    If the chunking algorithm changes, PostgreSQL chunk IDs
    may change.

    An old FAISS index would then refer to obsolete IDs.

    In that case it is safer to rebuild the index instead of
    silently returning the wrong text.
    """

    database_ids = {
        int(chunk["chunk_id"])
        for chunk in database_chunks
    }


    stale_ids = (
        existing_ids
        - database_ids
    )


    if stale_ids:

        raise RuntimeError(
            f"FAISS contains {len(stale_ids)} chunk IDs "
            "that no longer exist in PostgreSQL.\n\n"
            "The chunking data probably changed after the "
            "FAISS index was created.\n"
            "Delete the FAISS index and metadata file and "
            "rebuild the embeddings."
        )


# ============================================================
# get_chunks_to_embed()
# ============================================================

def get_chunks_to_embed(
    database_chunks,
    existing_ids: set[int],
):
    """
    Remove chunks that already exist in FAISS.

    FAISS itself is treated as the source of truth.

    This makes the process restartable.
    """

    return [
        chunk
        for chunk in database_chunks

        if int(chunk["chunk_id"])
        not in existing_ids
    ]


# ============================================================
# build_batches()
# ============================================================

def build_batches(
    chunks,
):
    """
    Group chunks into embedding API requests.

    Two limits are used:

        MAX_BATCH_ITEMS
        MAX_BATCH_TOKENS


    Example
    -------

    Suppose:

        MAX_BATCH_ITEMS = 256

    but the accumulated token count reaches our token limit
    after only 220 chunks.

    Then the batch stops at 220.


    Why?
    ----
    OpenAI's embedding endpoint has an aggregate token limit
    per request.

    Keeping a comfortable margin makes the pipeline safer.
    """

    current_batch = []

    current_tokens = 0


    for chunk in chunks:

        chunk_tokens = int(
            chunk["token_count"]
        )


        would_exceed_items = (
            len(current_batch)
            >= MAX_BATCH_ITEMS
        )


        would_exceed_tokens = (
            current_batch
            and
            current_tokens + chunk_tokens
            > MAX_BATCH_TOKENS
        )


        # ----------------------------------------------------
        # Current batch is full.
        # ----------------------------------------------------

        if (
            would_exceed_items
            or would_exceed_tokens
        ):

            yield current_batch


            current_batch = []

            current_tokens = 0


        current_batch.append(
            chunk
        )


        current_tokens += (
            chunk_tokens
        )


    # Return final partial batch.
    if current_batch:

        yield current_batch


# ============================================================
# create_embeddings()
# ============================================================

def create_embeddings(
    client: OpenAI,
    batch,
) -> np.ndarray:
    """
    Send one batch of chunk text to OpenAI.

    Returns
    -------
    numpy.ndarray

        Shape conceptually:

            number_of_chunks
                  ×
            embedding_dimensions


    Example:

        200 chunks
            ↓
        200 embedding vectors
    """

    texts = [
        chunk["content"]
        for chunk in batch
    ]


    retry_delay = (
        INITIAL_RETRY_SECONDS
    )


    for attempt in range(
        1,
        MAX_RETRIES + 1,
    ):

        try:

            response = (
                client.embeddings.create(
                    model=EMBEDDING_MODEL,
                    input=texts,
                )
            )


            # ------------------------------------------------
            # API response contains an index for each result.
            #
            # Sort explicitly so vectors remain aligned with
            # the input chunk ordering.
            # ------------------------------------------------

            ordered_embeddings = sorted(
                response.data,
                key=lambda item:
                    item.index,
            )


            vectors = np.array(
                [
                    item.embedding
                    for item
                    in ordered_embeddings
                ],
                dtype="float32",
            )


            return vectors


        except (
            RateLimitError,
            APIConnectionError,
            APITimeoutError,
        ) as error:

            if attempt == MAX_RETRIES:

                raise


            print(
                f"    API retry "
                f"{attempt}/{MAX_RETRIES}: "
                f"{error}"
            )


            time.sleep(
                retry_delay
            )


            # Exponential backoff:
            #
            # 2 seconds
            # 4 seconds
            # 8 seconds
            # 16 seconds
            #
            retry_delay *= 2


    # Defensive fallback.
    raise RuntimeError(
        "Embedding request unexpectedly exited "
        "without returning vectors."
    )


# ============================================================
# create_faiss_index()
# ============================================================

def create_faiss_index(
    dimension: int,
):
    """
    Create the FAISS vector index.

    We want COSINE similarity.

    FAISS does not need a separate cosine index.

    Instead:

        normalize vectors to length 1

    then use:

        inner product


    For normalized vectors:

        inner product == cosine similarity


    Base index:
        IndexFlatIP

    Wrapper:
        IndexIDMap2


    Why IndexIDMap2?
    ----------------

    Normal IndexFlatIP internally uses positions:

        0
        1
        2

    We want our own PostgreSQL IDs instead:

        chunk_id 1834
        chunk_id 1835
        chunk_id 1902

    IndexIDMap2 allows:

        vector ↔ PostgreSQL chunk_id
    """

    base_index = (
        faiss.IndexFlatIP(
            dimension
        )
    )


    index = (
        faiss.IndexIDMap2(
            base_index
        )
    )


    return index


# ============================================================
# save_faiss_index()
# ============================================================

def save_faiss_index(
    index,
):
    """
    Safely save FAISS index to disk.

    We first write:

        sec_chunks.tmp.faiss

    then replace:

        sec_chunks.faiss


    This reduces the chance of leaving a corrupt primary index
    if the program stops during disk writing.
    """

    FAISS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )


    temporary_path = (
        FAISS_DIRECTORY
        / "sec_chunks.tmp.faiss"
    )


    faiss.write_index(
        index,
        str(temporary_path),
    )


    os.replace(
        temporary_path,
        INDEX_PATH,
    )


# ============================================================
# save_index_metadata()
# ============================================================

def save_index_metadata(
    index,
):
    """
    Save small metadata describing the FAISS index.

    This is NOT the financial metadata.

    It describes how the vectors themselves were generated.
    """

    metadata = {
        "embedding_model":
            EMBEDDING_MODEL,

        "dimension":
            int(index.d),

        "similarity":
            "cosine",

        "normalized":
            True,

        "vectors":
            int(index.ntotal),
    }


    temporary_path = (
        FAISS_DIRECTORY
        / "sec_chunks.meta.tmp.json"
    )


    temporary_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )


    os.replace(
        temporary_path,
        METADATA_PATH,
    )


# ============================================================
# mark_batch_embedded()
# ============================================================

def mark_batch_embedded(
    engine,
    chunk_table,
    chunk_ids,
):
    """
    Mark a successfully indexed batch as EMBEDDED.
    """

    with engine.begin() as connection:

        connection.execute(
            update(
                chunk_table
            )
            .where(
                chunk_table.c.chunk_id.in_(
                    chunk_ids
                )
            )
            .values(
                embedding_status="EMBEDDED",
                embedding_model=EMBEDDING_MODEL,
                embedded_at=func.now(),
                embedding_error=None,
            )
        )


# ============================================================
# mark_batch_failed()
# ============================================================

def mark_batch_failed(
    engine,
    chunk_table,
    chunk_ids,
    error_message: str,
):
    """
    Record an embedding failure in PostgreSQL.
    """

    with engine.begin() as connection:

        connection.execute(
            update(
                chunk_table
            )
            .where(
                chunk_table.c.chunk_id.in_(
                    chunk_ids
                )
            )
            .values(
                embedding_status="FAILED",
                embedding_model=EMBEDDING_MODEL,
                embedding_error=(
                    error_message[:1000]
                ),
            )
        )


# ============================================================
# run_embedding_pipeline()
# ============================================================

def run_embedding_pipeline():
    """
    Build or resume the AlphaLens SEC FAISS index.

    Flow:

        PostgreSQL filing_chunks
                ↓
        determine missing FAISS IDs
                ↓
        OpenAI embedding batches
                ↓
        normalize vectors
                ↓
        FAISS add_with_ids()
                ↓
        save index
                ↓
        PostgreSQL status update
    """

    engine = (
        get_database_engine()
    )


    client = (
        get_openai_client()
    )


    metadata = MetaData()


    chunk_table = Table(
        "filing_chunks",
        metadata,
        autoload_with=engine,
    )


    # ========================================================
    # Load PostgreSQL chunks
    # ========================================================

    database_chunks = (
        get_all_chunks(
            engine=engine,
            chunk_table=chunk_table,
        )
    )


    print(
        f"\nPostgreSQL chunks found: "
        f"{len(database_chunks):,}"
    )


    # ========================================================
    # Load previous FAISS progress
    # ========================================================

    index = (
        load_existing_index()
    )


    existing_ids = (
        get_existing_faiss_ids(
            index
        )
    )


    # Detect an old/incompatible index.
    validate_index_ids(
        database_chunks=
            database_chunks,

        existing_ids=
            existing_ids,
    )


    # Fix PostgreSQL statuses if a previous run successfully
    # saved vectors before stopping.
    synchronize_database_status(
        engine=engine,
        chunk_table=chunk_table,
        existing_ids=existing_ids,
    )


    # ========================================================
    # Find chunks that still need embeddings
    # ========================================================

    chunks_to_embed = (
        get_chunks_to_embed(
            database_chunks=
                database_chunks,

            existing_ids=
                existing_ids,
        )
    )


    print(
        f"Already indexed: "
        f"{len(existing_ids):,}"
    )


    print(
        f"Remaining: "
        f"{len(chunks_to_embed):,}"
    )


    # Nothing left.
    if not chunks_to_embed:

        print(
            "\nAll chunks are already embedded."
        )

        return


    # ========================================================
    # Build API batches
    # ========================================================

    batches = list(
        build_batches(
            chunks_to_embed
        )
    )


    print(
        f"API batches: "
        f"{len(batches):,}"
    )


    # ========================================================
    # Process batches
    # ========================================================

    for batch_number, batch in enumerate(
        batches,
        start=1,
    ):

        chunk_ids = np.array(
            [
                int(chunk["chunk_id"])
                for chunk in batch
            ],
            dtype="int64",
        )


        batch_token_count = sum(
            int(chunk["token_count"])
            for chunk in batch
        )


        print(
            f"\nBatch "
            f"{batch_number}/{len(batches)} "
            f"| chunks: {len(batch)} "
            f"| ~tokens: {batch_token_count:,}"
        )


        try:

            # ================================================
            # Generate OpenAI vectors
            # ================================================

            vectors = create_embeddings(
                client=client,
                batch=batch,
            )


            # ================================================
            # Create FAISS index on first API response
            # ================================================
            #
            # Instead of hard-coding the embedding dimension,
            # discover it from the actual model response.
            # ================================================

            if index is None:

                dimension = (
                    vectors.shape[1]
                )


                index = create_faiss_index(
                    dimension=dimension
                )


                print(
                    f"Created FAISS index "
                    f"with dimension {dimension}"
                )


            # ================================================
            # Safety check
            # ================================================

            if vectors.shape[1] != index.d:

                raise ValueError(
                    "Embedding dimension does not match "
                    "existing FAISS index."
                )


            # ================================================
            # Normalize vectors
            # ================================================
            #
            # After L2 normalization:
            #
            #     IndexFlatIP
            #
            # behaves as cosine-similarity search.
            # ================================================

            faiss.normalize_L2(
                vectors
            )


            # ================================================
            # Add vectors using PostgreSQL chunk IDs
            # ================================================

            index.add_with_ids(
                vectors,
                chunk_ids,
            )


            # ================================================
            # IMPORTANT ORDER
            # ================================================
            #
            # Save FAISS FIRST.
            #
            # Then mark PostgreSQL rows EMBEDDED.
            #
            # If the program crashes between these two steps,
            # the next run reads the IDs from FAISS and repairs
            # PostgreSQL status automatically.
            # ================================================

            save_faiss_index(
                index
            )


            save_index_metadata(
                index
            )


            mark_batch_embedded(
                engine=engine,
                chunk_table=chunk_table,
                chunk_ids=(
                    chunk_ids.tolist()
                ),
            )


            print(
                f"[OK] FAISS now contains "
                f"{index.ntotal:,} vectors"
            )


        except Exception as error:

            mark_batch_failed(
                engine=engine,
                chunk_table=chunk_table,
                chunk_ids=(
                    chunk_ids.tolist()
                ),
                error_message=str(error),
            )


            print(
                f"[FAILED] Batch "
                f"{batch_number}: "
                f"{error}"
            )


            # Continue with subsequent batches.
            continue


    # ========================================================
    # Final Result
    # ========================================================

    print()

    print(
        "========================================"
    )

    print(
        "EMBEDDING + FAISS COMPLETE"
    )

    print(
        "========================================"
    )


    if index is not None:

        print(
            f"Vectors indexed: "
            f"{index.ntotal:,}"
        )


        print(
            f"FAISS file: "
            f"{INDEX_PATH}"
        )


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":

    run_embedding_pipeline()