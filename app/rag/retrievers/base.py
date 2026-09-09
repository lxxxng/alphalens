"""
Shared FAISS, OpenAI, and PostgreSQL helpers for AlphaLens retrievers.
"""

import json
import os
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sqlalchemy import create_engine


load_dotenv()


# Self-hosted CI can keep large, private index files outside the Actions
# checkout. Local development continues to use data/faiss by default.
FAISS_DIRECTORY = Path(
    os.getenv(
        "ALPHALENS_FAISS_DIRECTORY",
        "data/faiss",
    )
)

SEC_INDEX_PATH = FAISS_DIRECTORY / "sec_chunks.faiss"
SEC_METADATA_PATH = FAISS_DIRECTORY / "sec_chunks.meta.json"

TRANSCRIPT_INDEX_PATH = FAISS_DIRECTORY / "transcript_chunks.faiss"
TRANSCRIPT_METADATA_PATH = FAISS_DIRECTORY / "transcript_chunks.meta.json"

DEFAULT_TOP_K = 5
INITIAL_FILTER_SEARCH_SIZE = 100
MAX_FILTER_SEARCH_SIZE = 5000


def get_database_engine():
    """
    Create PostgreSQL connection engine from DATABASE_URL.
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
    Create authenticated OpenAI client from OPENAI_API_KEY.
    """

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY was not found in .env."
        )

    return OpenAI(
        api_key=api_key,
    )


def load_index_metadata(
    metadata_path: Path,
) -> dict:
    """
    Read metadata describing how one FAISS index was created.
    """

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"FAISS metadata file not found: {metadata_path}"
        )

    metadata = json.loads(
        metadata_path.read_text(
            encoding="utf-8",
        )
    )

    if not metadata.get("embedding_model"):
        raise ValueError(
            "embedding_model is missing from FAISS metadata."
        )

    return metadata


def load_faiss_index(
    index_path: Path,
):
    """
    Load one non-empty FAISS index from disk.
    """

    if not index_path.exists():
        raise FileNotFoundError(
            f"FAISS index not found: {index_path}"
        )

    index = faiss.read_index(
        str(index_path)
    )

    if index.ntotal == 0:
        raise ValueError(
            "FAISS index contains zero vectors."
        )

    return index


def embed_query(
    client: OpenAI,
    query: str,
    embedding_model: str,
) -> np.ndarray:
    """
    Convert a search query into a FAISS-ready embedding vector.
    """

    query = query.strip()

    if not query:
        raise ValueError(
            "Search query cannot be empty."
        )

    response = client.embeddings.create(
        model=embedding_model,
        input=query,
    )

    return np.array(
        [
            response.data[0].embedding,
        ],
        dtype="float32",
    )


def normalize_query_vector(
    query_vector: np.ndarray,
):
    """
    Normalize a query vector for cosine-similarity FAISS indexes.
    """

    faiss.normalize_L2(
        query_vector
    )


def validate_index_dimension(
    index,
    index_metadata: dict,
):
    """
    Ensure a FAISS index matches its metadata file.
    """

    expected_dimension = int(
        index_metadata["dimension"]
    )

    if index.d != expected_dimension:
        raise ValueError(
            "FAISS index dimension does not match its metadata file."
        )


def search_faiss(
    index,
    query_vector: np.ndarray,
    search_size: int,
):
    """
    Search FAISS and return ranked chunk IDs with similarity scores.
    """

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
        if chunk_id == -1:
            continue

        results.append(
            {
                "chunk_id": int(chunk_id),
                "score": float(score),
            }
        )

    return results


def build_query_vector(
    query: str,
    index_metadata: dict,
    index,
) -> np.ndarray:
    """
    Embed and normalize a query for one loaded FAISS index.
    """

    client = get_openai_client()
    query_vector = embed_query(
        client=client,
        query=query,
        embedding_model=index_metadata["embedding_model"],
    )

    if query_vector.shape[1] != index.d:
        raise ValueError(
            "Query embedding dimension does not match FAISS index dimension."
        )

    normalize_query_vector(
        query_vector
    )

    return query_vector


def apply_metadata_filters(
    results,
    ticker: str | None = None,
    form_type: str | None = None,
    section_key: str | None = None,
    fiscal_period: str | None = None,
):
    """
    Apply common metadata filters after FAISS search.
    """

    filtered_results = []

    for result in results:
        if (
            ticker is not None
            and result["ticker"].upper() != ticker.upper()
        ):
            continue

        if (
            form_type is not None
            and result.get("form_type", "").upper() != form_type.upper()
        ):
            continue

        if (
            section_key is not None
            and result.get("section_key") != section_key
        ):
            continue

        if (
            fiscal_period is not None
            and result.get("fiscal_period", "").upper()
            != fiscal_period.upper()
        ):
            continue

        filtered_results.append(
            result
        )

    return filtered_results


def search_with_filters(
    index,
    query_vector: np.ndarray,
    top_k: int,
    has_filters: bool,
    fetch_metadata,
    apply_filters,
):
    """
    Search FAISS directly or with progressively wider post-filtering.
    """

    if not has_filters:
        faiss_results = search_faiss(
            index=index,
            query_vector=query_vector,
            search_size=top_k,
        )

        return fetch_metadata(
            faiss_results,
        )[:top_k]

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

        results = fetch_metadata(
            faiss_results,
        )

        filtered_results = apply_filters(
            results,
        )

        if len(filtered_results) >= top_k:
            return filtered_results[:top_k]

        if search_size >= index.ntotal:
            return filtered_results

        if search_size >= MAX_FILTER_SEARCH_SIZE:
            return filtered_results

        search_size = min(
            search_size * 2,
            MAX_FILTER_SEARCH_SIZE,
            index.ntotal,
        )
