"""
AlphaLens - SEC Semantic Retriever

Purpose
-------
Search the SEC filing chunks stored in FAISS and return
the corresponding original text + metadata from PostgreSQL.


Current RAG pipeline
--------------------

SEC filings
    ↓
sections
    ↓
chunks
    ↓
OpenAI embeddings
    ↓
FAISS index                         DONE
    ↓
retriever.py                        <- THIS FILE
    ↓
relevant SEC chunks
    ↓
LLM answer generation              NEXT


How retrieval works
-------------------

User question:

    "What cybersecurity risks does NVIDIA face?"

        ↓

1. Convert question into an embedding using the SAME
   embedding model that created the FAISS index.

        ↓

2. Normalize the query vector.

        ↓

3. FAISS compares that vector against all stored SEC
   chunk vectors.

        ↓

4. FAISS returns:

       similarity score
       chunk_id

        ↓

5. chunk_id is used to retrieve the real text and metadata
   from PostgreSQL.

Example:

    FAISS:
        chunk_id = 18291
        score = 0.71

    PostgreSQL:
        ticker = NVDA
        form_type = 10-K
        section = Risk Factors
        filing_date = ...
        content = "Cybersecurity threats may..."
"""

import json
import os

from pathlib import Path

import faiss
import numpy as np

from dotenv import load_dotenv

from openai import OpenAI

from sqlalchemy import (
    MetaData,
    Table,
    create_engine,
    select,
)


# ============================================================
# Environment
# ============================================================

load_dotenv()


# ============================================================
# FAISS Paths
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
# Retrieval Configuration
# ============================================================

# Number of final results returned by default.
DEFAULT_TOP_K = 5


# When metadata filters are used, such as:
#
#     ticker = NVDA
#
# FAISS itself is still searching globally.
#
# We therefore retrieve MORE than top_k from FAISS first,
# then filter those results using PostgreSQL metadata.
#
# Example:
#
#     user wants top 5 NVDA chunks
#
#     retrieve top 100 globally
#         ↓
#     keep NVDA only
#         ↓
#     return first 5
#
INITIAL_FILTER_SEARCH_SIZE = 100


# If 100 results do not contain enough matching filtered
# chunks, increase the search automatically.
MAX_FILTER_SEARCH_SIZE = 5000


# ============================================================
# get_database_engine()
# ============================================================

def get_database_engine():
    """
    Create PostgreSQL connection engine.

    DATABASE_URL is read from .env.
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
    Create authenticated OpenAI client.

    OPENAI_API_KEY is read from .env.
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
# load_index_metadata()
# ============================================================

def load_index_metadata() -> dict:
    """
    Read metadata describing how the FAISS index was created.

    Example:

        {
            "embedding_model":
                "text-embedding-3-small",

            "dimension":
                1536,

            "similarity":
                "cosine",

            "normalized":
                true,

            "vectors":
                32000
        }


    Why read this?
    --------------

    The query MUST be embedded using the same model as the
    SEC chunks.

    We should NOT do:

        documents:
            text-embedding-3-small

        query:
            some different embedding model

    because the vectors would not belong to the same vector
    space and similarity scores would be meaningless.
    """

    if not METADATA_PATH.exists():

        raise FileNotFoundError(
            f"FAISS metadata file not found: "
            f"{METADATA_PATH}"
        )


    metadata = json.loads(
        METADATA_PATH.read_text(
            encoding="utf-8"
        )
    )


    embedding_model = (
        metadata.get(
            "embedding_model"
        )
    )


    if not embedding_model:

        raise ValueError(
            "embedding_model is missing from "
            "FAISS metadata."
        )


    return metadata


# ============================================================
# load_faiss_index()
# ============================================================

def load_faiss_index():
    """
    Load the SEC chunk FAISS index from disk.

    Returns
    -------
    faiss.Index

    The index contains:

        embedding vectors
            +
        PostgreSQL chunk_id values
    """

    if not INDEX_PATH.exists():

        raise FileNotFoundError(
            f"FAISS index not found: "
            f"{INDEX_PATH}"
        )


    index = faiss.read_index(
        str(INDEX_PATH)
    )


    if index.ntotal == 0:

        raise ValueError(
            "FAISS index contains zero vectors."
        )


    return index


# ============================================================
# embed_query()
# ============================================================

def embed_query(
    client: OpenAI,
    query: str,
    embedding_model: str,
) -> np.ndarray:
    """
    Convert a user question into an embedding vector.

    Parameters
    ----------
    client:
        Authenticated OpenAI API client.

    query:
        User's natural-language search question.

    embedding_model:
        SAME model used to embed the SEC chunks.


    Returns
    -------
    numpy.ndarray

        Shape:

            (1, embedding_dimension)


    Example
    -------

    Query:

        "What cybersecurity risks does NVIDIA face?"

                ↓

    OpenAI embedding:

        [
            0.0123,
            -0.0281,
            ...
        ]
    """

    # --------------------------------------------------------
    # Validate question
    # --------------------------------------------------------

    query = query.strip()


    if not query:

        raise ValueError(
            "Search query cannot be empty."
        )


    # --------------------------------------------------------
    # Generate embedding
    # --------------------------------------------------------

    response = (
        client.embeddings.create(
            model=embedding_model,
            input=query,
        )
    )


    embedding = (
        response
        .data[0]
        .embedding
    )


    # --------------------------------------------------------
    # FAISS expects float32 vectors.
    # --------------------------------------------------------

    vector = np.array(
        [
            embedding
        ],
        dtype="float32",
    )


    return vector


# ============================================================
# normalize_query_vector()
# ============================================================

def normalize_query_vector(
    query_vector: np.ndarray,
):
    """
    Normalize query vector for cosine similarity.

    Why?
    ----

    Our FAISS index was created using:

        IndexFlatIP

    where IP means:

        inner product

    But we want:

        cosine similarity


    For normalized vectors:

        inner product
            =
        cosine similarity


    During indexing we already normalized every document
    embedding.

    Therefore we MUST normalize the query embedding too.
    """

    faiss.normalize_L2(
        query_vector
    )


# ============================================================
# search_faiss()
# ============================================================

def search_faiss(
    index,
    query_vector: np.ndarray,
    search_size: int,
):
    """
    Perform nearest-neighbor search in FAISS.

    Parameters
    ----------
    index:
        Loaded FAISS SEC index.

    query_vector:
        Normalized question embedding.

    search_size:
        Number of nearest vectors to retrieve.


    Returns
    -------
    list[dict]

    Example:

        [
            {
                "chunk_id": 18291,
                "score": 0.712
            },

            {
                "chunk_id": 8812,
                "score": 0.693
            }
        ]


    FAISS output
    ------------

    index.search() returns TWO matrices:

        scores
        ids


    Since we search one query:

        scores[0]
        ids[0]

    contain the results for that question.
    """

    # Never ask FAISS for more results than it contains.
    search_size = min(
        search_size,
        index.ntotal,
    )


    scores, ids = index.search(
        query_vector,
        search_size,
    )


    results = []


    for score, chunk_id in zip(
        scores[0],
        ids[0],
    ):

        # FAISS uses -1 when no valid vector exists for a
        # requested result position.
        if chunk_id == -1:

            continue


        results.append(
            {
                "chunk_id":
                    int(chunk_id),

                "score":
                    float(score),
            }
        )


    return results


# ============================================================
# fetch_chunk_metadata()
# ============================================================

def fetch_chunk_metadata(
    engine,
    chunk_table,
    faiss_results,
):
    """
    Retrieve original text + metadata from PostgreSQL.

    FAISS only gives us:

        chunk_id
        similarity score

    PostgreSQL gives us:

        ticker
        filing
        section
        original content
        etc.
    """

    if not faiss_results:

        return []


    chunk_ids = [
        result["chunk_id"]
        for result in faiss_results
    ]


    query = (
        select(
            chunk_table.c.chunk_id,
            chunk_table.c.ticker,
            chunk_table.c.form_type,
            chunk_table.c.filing_date,
            chunk_table.c.accession_number,
            chunk_table.c.section_key,
            chunk_table.c.section_title,
            chunk_table.c.chunk_index,
            chunk_table.c.token_count,
            chunk_table.c.content,
        )
        .where(
            chunk_table.c.chunk_id.in_(
                chunk_ids
            )
        )
    )


    with engine.connect() as connection:

        rows = (
            connection
            .execute(query)
            .mappings()
            .all()
        )


    # --------------------------------------------------------
    # IMPORTANT:
    #
    # SQL WHERE id IN (...) does NOT guarantee the same order
    # as FAISS returned.
    #
    # Therefore first create:
    #
    #     chunk_id -> database row
    #
    # and then rebuild the result list following FAISS ranking.
    # --------------------------------------------------------

    row_lookup = {
        int(row["chunk_id"]):
            row

        for row in rows
    }


    combined_results = []


    for faiss_result in faiss_results:

        chunk_id = (
            faiss_result[
                "chunk_id"
            ]
        )


        database_row = (
            row_lookup.get(
                chunk_id
            )
        )


        if database_row is None:

            # This should normally never happen.
            #
            # It would mean FAISS contains an ID that does
            # not exist in filing_chunks.
            continue


        combined_results.append(
            {
                "chunk_id":
                    chunk_id,

                "score":
                    faiss_result[
                        "score"
                    ],

                "ticker":
                    database_row[
                        "ticker"
                    ],

                "form_type":
                    database_row[
                        "form_type"
                    ],

                "filing_date":
                    database_row[
                        "filing_date"
                    ],

                "accession_number":
                    database_row[
                        "accession_number"
                    ],

                "section_key":
                    database_row[
                        "section_key"
                    ],

                "section_title":
                    database_row[
                        "section_title"
                    ],

                "chunk_index":
                    database_row[
                        "chunk_index"
                    ],

                "token_count":
                    database_row[
                        "token_count"
                    ],

                "content":
                    database_row[
                        "content"
                    ],
            }
        )


    return combined_results


# ============================================================
# apply_metadata_filters()
# ============================================================

def apply_metadata_filters(
    results,
    ticker=None,
    form_type=None,
    section_key=None,
):
    """
    Apply optional PostgreSQL metadata filters.

    Examples
    --------

    Search only NVIDIA:

        ticker="NVDA"


    Search only annual reports:

        form_type="10-K"


    Search only Risk Factors:

        section_key="item_1a_risk_factors"


    They can also be combined:

        ticker="NVDA"
        form_type="10-K"
        section_key="item_1a_risk_factors"
    """

    filtered_results = []


    for result in results:

        # ----------------------------------------------------
        # Ticker filter
        # ----------------------------------------------------

        if (
            ticker is not None
            and
            result["ticker"].upper()
            != ticker.upper()
        ):

            continue


        # ----------------------------------------------------
        # Filing type filter
        # ----------------------------------------------------

        if (
            form_type is not None
            and
            result["form_type"].upper()
            != form_type.upper()
        ):

            continue


        # ----------------------------------------------------
        # Section filter
        # ----------------------------------------------------

        if (
            section_key is not None
            and
            result["section_key"]
            != section_key
        ):

            continue


        filtered_results.append(
            result
        )


    return filtered_results


# ============================================================
# semantic_search()
# ============================================================

def semantic_search(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    ticker: str | None = None,
    form_type: str | None = None,
    section_key: str | None = None,
):
    """
    Search AlphaLens SEC chunks semantically.

    Parameters
    ----------
    query:
        Natural-language search question.

    top_k:
        Number of final chunks to return.

    ticker:
        Optional ticker filter.

        Example:
            NVDA

    form_type:
        Optional filing type.

        Example:
            10-K

    section_key:
        Optional AlphaLens section identifier.

        Example:
            item_1a_risk_factors


    Returns
    -------
    list[dict]


    Example
    -------

    semantic_search(
        query=
            "What cybersecurity risks does NVIDIA face?",

        top_k=5,

        ticker="NVDA",

        section_key=
            "item_1a_risk_factors",
    )
    """

    # ========================================================
    # Validate top_k
    # ========================================================

    if top_k <= 0:

        raise ValueError(
            "top_k must be greater than zero."
        )


    # ========================================================
    # Load infrastructure
    # ========================================================

    engine = (
        get_database_engine()
    )


    client = (
        get_openai_client()
    )


    index_metadata = (
        load_index_metadata()
    )


    index = (
        load_faiss_index()
    )


    # ========================================================
    # Verify index consistency
    # ========================================================

    expected_dimension = int(
        index_metadata[
            "dimension"
        ]
    )


    if index.d != expected_dimension:

        raise ValueError(
            "FAISS index dimension does not match "
            "its metadata file."
        )


    embedding_model = (
        index_metadata[
            "embedding_model"
        ]
    )


    # ========================================================
    # Embed question
    # ========================================================

    query_vector = embed_query(
        client=client,
        query=query,
        embedding_model=embedding_model,
    )


    # Ensure query embedding dimension matches FAISS.
    if query_vector.shape[1] != index.d:

        raise ValueError(
            "Query embedding dimension does not match "
            "FAISS index dimension."
        )


    # ========================================================
    # Normalize
    # ========================================================

    normalize_query_vector(
        query_vector
    )


    # ========================================================
    # Load filing_chunks table
    # ========================================================

    metadata = MetaData()


    chunk_table = Table(
        "filing_chunks",
        metadata,
        autoload_with=engine,
    )


    # ========================================================
    # Decide whether filtering is requested
    # ========================================================

    has_filters = any(
        value is not None

        for value in [
            ticker,
            form_type,
            section_key,
        ]
    )


    # ========================================================
    # No filters
    # ========================================================
    #
    # Easy case:
    #
    #     ask FAISS directly for top_k.
    # ========================================================

    if not has_filters:

        faiss_results = search_faiss(
            index=index,
            query_vector=query_vector,
            search_size=top_k,
        )


        results = fetch_chunk_metadata(
            engine=engine,
            chunk_table=chunk_table,
            faiss_results=faiss_results,
        )


        return results[:top_k]


    # ========================================================
    # Filtered Search
    # ========================================================
    #
    # FAISS does not know that:
    #
    #     chunk 123 = AAPL
    #     chunk 456 = NVDA
    #
    # That metadata lives in PostgreSQL.
    #
    # Therefore:
    #
    # 1. retrieve top 100 semantic matches
    # 2. fetch their metadata
    # 3. keep only requested ticker/form/section
    #
    # If we don't have top_k matches yet:
    #
    #     100 -> 200 -> 400 -> 800 ...
    #
    # until enough matches are found.
    # ========================================================

    search_size = max(
        INITIAL_FILTER_SEARCH_SIZE,
        top_k,
    )


    while True:

        faiss_results = search_faiss(
            index=index,
            query_vector=query_vector,
            search_size=search_size,
        )


        results = fetch_chunk_metadata(
            engine=engine,
            chunk_table=chunk_table,
            faiss_results=faiss_results,
        )


        filtered_results = (
            apply_metadata_filters(
                results=results,
                ticker=ticker,
                form_type=form_type,
                section_key=section_key,
            )
        )


        # We have enough.
        if len(filtered_results) >= top_k:

            return (
                filtered_results[
                    :top_k
                ]
            )


        # We searched the entire FAISS index.
        if search_size >= index.ntotal:

            return filtered_results


        # Stop excessively large filtered searches.
        if (
            search_size
            >= MAX_FILTER_SEARCH_SIZE
        ):

            return filtered_results


        # Double search scope and try again.
        search_size = min(
            search_size * 2,
            MAX_FILTER_SEARCH_SIZE,
            index.ntotal,
        )


# ============================================================
# print_search_results()
# ============================================================

def print_search_results(
    query: str,
    results,
):
    """
    Print semantic-search results in a readable format.

    This is only for development/testing.

    Later FastAPI will return structured JSON instead.
    """

    print()

    print(
        "========================================"
    )

    print(
        "ALPHALENS SEMANTIC SEARCH"
    )

    print(
        "========================================"
    )


    print(
        f"\nQuery:\n{query}"
    )


    print(
        f"\nResults returned: "
        f"{len(results)}"
    )


    for rank, result in enumerate(
        results,
        start=1,
    ):

        print()

        print(
            "----------------------------------------"
        )

        print(
            f"RESULT {rank}"
        )

        print(
            "----------------------------------------"
        )


        print(
            f"Similarity: "
            f"{result['score']:.4f}"
        )


        print(
            f"Chunk ID: "
            f"{result['chunk_id']}"
        )


        print(
            f"Ticker: "
            f"{result['ticker']}"
        )


        print(
            f"Form: "
            f"{result['form_type']}"
        )


        print(
            f"Filing date: "
            f"{result['filing_date']}"
        )


        print(
            f"Section: "
            f"{result['section_title']}"
        )


        print(
            f"Section key: "
            f"{result['section_key']}"
        )


        print(
            f"Chunk index: "
            f"{result['chunk_index']}"
        )


        print(
            f"Tokens: "
            f"{result['token_count']}"
        )


        print(
            "\nCONTENT:"
        )


        # Print the full chunk for now.
        #
        # During retrieval testing we WANT to inspect the
        # complete text to determine whether search is good.
        print(
            result[
                "content"
            ]
        )


# ============================================================
# Development Test
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Test question
    # --------------------------------------------------------

    test_query = (
        "What cybersecurity risks does NVIDIA face?"
    )


    # --------------------------------------------------------
    # First test:
    #
    # Search only NVDA.
    #
    # Don't restrict section yet.
    #
    # This lets us see whether semantic retrieval naturally
    # finds Risk Factors or another relevant section.
    # --------------------------------------------------------

    results = semantic_search(
        query=test_query,
        top_k=5,
        ticker="NVDA",
    )


    print_search_results(
        query=test_query,
        results=results,
    )