"""
AlphaLens - Earnings Transcript Chunk Embedding + FAISS Index Builder

Purpose
-------
Convert earnings transcript chunks into OpenAI embedding vectors and store
those vectors in a dedicated local FAISS index.

PostgreSQL stores transcript text and metadata:

    earnings_transcript_chunks.chunk_id
    earnings_transcript_chunks.ticker
    earnings_transcript_chunks.fiscal_period
    earnings_transcript_chunks.content

FAISS stores:

    chunk_id -> embedding vector

The transcript index is intentionally separate from the SEC filing index so
later retrieval can search transcripts, filings, or both with explicit source
control.
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


load_dotenv()


EMBEDDING_MODEL = "text-embedding-3-small"

MAX_BATCH_ITEMS = 256
MAX_BATCH_TOKENS = 180_000

MAX_RETRIES = 5
INITIAL_RETRY_SECONDS = 2

FAISS_DIRECTORY = Path("data/faiss")
INDEX_PATH = FAISS_DIRECTORY / "transcript_chunks.faiss"
METADATA_PATH = FAISS_DIRECTORY / "transcript_chunks.meta.json"


def get_database_engine():
    """
    Create the PostgreSQL SQLAlchemy engine from DATABASE_URL.
    """

    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise ValueError(
            "DATABASE_URL was not found in .env."
        )

    return create_engine(
        database_url,
        pool_pre_ping=True,
    )


def get_openai_client() -> OpenAI:
    """
    Create an authenticated OpenAI API client.
    """

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY was not found in .env."
        )

    return OpenAI(
        api_key=api_key,
    )


def load_existing_index():
    """
    Load the existing transcript FAISS index when present.
    """

    if not INDEX_PATH.exists():
        return None

    if not METADATA_PATH.exists():
        raise RuntimeError(
            "Transcript FAISS index exists but metadata file is missing."
        )

    metadata = json.loads(
        METADATA_PATH.read_text(
            encoding="utf-8",
        )
    )

    existing_model = metadata.get("embedding_model")

    if existing_model != EMBEDDING_MODEL:
        raise RuntimeError(
            "Existing transcript FAISS index uses a different embedding "
            f"model. Existing: {existing_model}; configured: "
            f"{EMBEDDING_MODEL}."
        )

    index = faiss.read_index(
        str(INDEX_PATH)
    )

    print(
        f"Loaded existing transcript FAISS index: "
        f"{index.ntotal:,} vectors"
    )

    return index


def get_existing_faiss_ids(
    index,
) -> set[int]:
    """
    Return chunk IDs already stored in the FAISS ID map.
    """

    if index is None or index.ntotal == 0:
        return set()

    id_array = faiss.vector_to_array(
        index.id_map
    )

    return {
        int(value)
        for value in id_array
    }


def get_all_chunks(
    engine,
    chunk_table,
):
    """
    Retrieve all transcript chunks that can be embedded.
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

        return (
            result
            .mappings()
            .all()
        )


def synchronize_database_status(
    engine,
    chunk_table,
    existing_ids: set[int],
):
    """
    Mark rows as embedded when their vectors already exist in FAISS.
    """

    if not existing_ids:
        return

    ids = list(existing_ids)
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
                update(chunk_table)
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


def validate_index_ids(
    database_chunks,
    existing_ids: set[int],
):
    """
    Fail if FAISS references chunk IDs that no longer exist in PostgreSQL.
    """

    database_ids = {
        int(chunk["chunk_id"])
        for chunk in database_chunks
    }

    stale_ids = existing_ids - database_ids

    if stale_ids:
        raise RuntimeError(
            f"Transcript FAISS contains {len(stale_ids)} chunk IDs that "
            "no longer exist in PostgreSQL. Delete "
            f"{INDEX_PATH} and {METADATA_PATH}, then rebuild."
        )


def get_chunks_to_embed(
    database_chunks,
    existing_ids: set[int],
):
    """
    Return chunks not already present in FAISS.
    """

    return [
        chunk
        for chunk in database_chunks
        if int(chunk["chunk_id"]) not in existing_ids
    ]


def build_batches(
    chunks,
):
    """
    Group transcript chunks into embedding API batches.
    """

    current_batch = []
    current_tokens = 0

    for chunk in chunks:
        chunk_tokens = int(chunk["token_count"])

        would_exceed_items = (
            len(current_batch) >= MAX_BATCH_ITEMS
        )

        would_exceed_tokens = (
            current_batch
            and current_tokens + chunk_tokens > MAX_BATCH_TOKENS
        )

        if would_exceed_items or would_exceed_tokens:
            yield current_batch
            current_batch = []
            current_tokens = 0

        current_batch.append(chunk)
        current_tokens += chunk_tokens

    if current_batch:
        yield current_batch


def create_embeddings(
    client: OpenAI,
    batch,
) -> np.ndarray:
    """
    Send one batch of transcript chunk text to OpenAI.
    """

    texts = [
        chunk["content"]
        for chunk in batch
    ]

    retry_delay = INITIAL_RETRY_SECONDS

    for attempt in range(
        1,
        MAX_RETRIES + 1,
    ):
        try:
            response = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=texts,
            )

            ordered_embeddings = sorted(
                response.data,
                key=lambda item: item.index,
            )

            return np.array(
                [
                    item.embedding
                    for item in ordered_embeddings
                ],
                dtype="float32",
            )

        except (
            RateLimitError,
            APIConnectionError,
            APITimeoutError,
        ) as error:
            if attempt == MAX_RETRIES:
                raise

            print(
                f"    API retry {attempt}/{MAX_RETRIES}: {error}"
            )

            time.sleep(
                retry_delay
            )

            retry_delay *= 2

    raise RuntimeError(
        "Embedding request unexpectedly exited without returning vectors."
    )


def create_faiss_index(
    dimension: int,
):
    """
    Create a cosine-similarity FAISS index with PostgreSQL chunk IDs.
    """

    base_index = faiss.IndexFlatIP(
        dimension
    )

    return faiss.IndexIDMap2(
        base_index
    )


def save_faiss_index(
    index,
):
    """
    Atomically save the transcript FAISS index.
    """

    FAISS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = FAISS_DIRECTORY / "transcript_chunks.tmp.faiss"

    faiss.write_index(
        index,
        str(temporary_path),
    )

    os.replace(
        temporary_path,
        INDEX_PATH,
    )


def save_index_metadata(
    index,
):
    """
    Save metadata describing the transcript vector index.
    """

    metadata = {
        "embedding_model": EMBEDDING_MODEL,
        "dimension": int(index.d),
        "similarity": "cosine",
        "normalized": True,
        "vectors": int(index.ntotal),
        "source_table": "earnings_transcript_chunks",
    }

    temporary_path = FAISS_DIRECTORY / "transcript_chunks.meta.tmp.json"

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


def mark_batch_embedded(
    engine,
    chunk_table,
    chunk_ids,
):
    """
    Mark a successfully indexed batch as embedded.
    """

    with engine.begin() as connection:
        connection.execute(
            update(chunk_table)
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
            update(chunk_table)
            .where(
                chunk_table.c.chunk_id.in_(
                    chunk_ids
                )
            )
            .values(
                embedding_status="FAILED",
                embedding_model=EMBEDDING_MODEL,
                embedding_error=error_message[:1000],
            )
        )


def run_embedding_pipeline():
    """
    Build or resume the transcript FAISS index.
    """

    engine = get_database_engine()
    client = get_openai_client()
    metadata = MetaData()

    chunk_table = Table(
        "earnings_transcript_chunks",
        metadata,
        autoload_with=engine,
    )

    database_chunks = get_all_chunks(
        engine=engine,
        chunk_table=chunk_table,
    )

    print(
        f"\nPostgreSQL transcript chunks found: "
        f"{len(database_chunks):,}"
    )

    index = load_existing_index()
    existing_ids = get_existing_faiss_ids(index)

    validate_index_ids(
        database_chunks=database_chunks,
        existing_ids=existing_ids,
    )

    synchronize_database_status(
        engine=engine,
        chunk_table=chunk_table,
        existing_ids=existing_ids,
    )

    chunks_to_embed = get_chunks_to_embed(
        database_chunks=database_chunks,
        existing_ids=existing_ids,
    )

    print(
        f"Already indexed: {len(existing_ids):,}"
    )

    print(
        f"Remaining: {len(chunks_to_embed):,}"
    )

    if not chunks_to_embed:
        print(
            "\nAll transcript chunks are already embedded."
        )
        return

    batches = list(
        build_batches(
            chunks_to_embed
        )
    )

    print(
        f"API batches: {len(batches):,}"
    )

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
            f"\nBatch {batch_number}/{len(batches)} "
            f"| chunks: {len(batch)} "
            f"| ~tokens: {batch_token_count:,}"
        )

        try:
            vectors = create_embeddings(
                client=client,
                batch=batch,
            )

            if index is None:
                dimension = vectors.shape[1]
                index = create_faiss_index(
                    dimension=dimension
                )
                print(
                    f"Created transcript FAISS index "
                    f"with dimension {dimension}"
                )

            if vectors.shape[1] != index.d:
                raise ValueError(
                    "Embedding dimension does not match existing "
                    "transcript FAISS index."
                )

            faiss.normalize_L2(
                vectors
            )

            index.add_with_ids(
                vectors,
                chunk_ids,
            )

            save_faiss_index(
                index
            )

            save_index_metadata(
                index
            )

            mark_batch_embedded(
                engine=engine,
                chunk_table=chunk_table,
                chunk_ids=chunk_ids.tolist(),
            )

            print(
                f"[OK] Transcript FAISS now contains "
                f"{index.ntotal:,} vectors"
            )

        except Exception as error:
            mark_batch_failed(
                engine=engine,
                chunk_table=chunk_table,
                chunk_ids=chunk_ids.tolist(),
                error_message=str(error),
            )

            print(
                f"[FAILED] Batch {batch_number}: {error}"
            )

            continue

    print()
    print("========================================")
    print("TRANSCRIPT EMBEDDING + FAISS COMPLETE")
    print("========================================")

    if index is not None:
        print(
            f"Vectors indexed: {index.ntotal:,}"
        )
        print(
            f"FAISS file: {INDEX_PATH}"
        )


if __name__ == "__main__":
    run_embedding_pipeline()
