"""
Earnings transcript semantic retrieval.
"""

from sqlalchemy import MetaData, Table, select

from app.rag.retrievers.base import (
    DEFAULT_TOP_K,
    TRANSCRIPT_INDEX_PATH,
    TRANSCRIPT_METADATA_PATH,
    apply_metadata_filters,
    build_query_vector,
    get_database_engine,
    load_faiss_index,
    load_index_metadata,
    search_with_filters,
    validate_index_dimension,
)


def fetch_chunk_metadata(
    engine,
    chunk_table,
    transcript_table,
    faiss_results,
):
    """
    Retrieve transcript chunk text and metadata from PostgreSQL.
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
            chunk_table.c.transcript_id,
            chunk_table.c.ticker,
            chunk_table.c.fiscal_year,
            chunk_table.c.fiscal_quarter,
            chunk_table.c.fiscal_period,
            chunk_table.c.source_provider,
            chunk_table.c.chunk_index,
            chunk_table.c.speaker_names,
            chunk_table.c.token_count,
            chunk_table.c.content,
            transcript_table.c.call_date,
            transcript_table.c.title,
            transcript_table.c.source_url,
        )
        .select_from(
            chunk_table.join(
                transcript_table,
                (
                    chunk_table.c.transcript_id
                    == transcript_table.c.transcript_id
                ),
            )
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

    row_lookup = {
        int(row["chunk_id"]): row
        for row in rows
    }

    combined_results = []

    for faiss_result in faiss_results:
        chunk_id = faiss_result["chunk_id"]
        database_row = row_lookup.get(
            chunk_id
        )

        if database_row is None:
            continue

        combined_results.append(
            {
                "source_type": "transcript",
                "chunk_id": chunk_id,
                "score": faiss_result["score"],
                "transcript_id": database_row["transcript_id"],
                "ticker": database_row["ticker"],
                "fiscal_year": database_row["fiscal_year"],
                "fiscal_quarter": database_row["fiscal_quarter"],
                "fiscal_period": database_row["fiscal_period"],
                "call_date": database_row["call_date"],
                "title": database_row["title"],
                "source_provider": database_row["source_provider"],
                "source_url": database_row["source_url"],
                "speaker_names": database_row["speaker_names"],
                "chunk_index": database_row["chunk_index"],
                "token_count": database_row["token_count"],
                "content": database_row["content"],
            }
        )

    return combined_results


def semantic_search(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    ticker: str | None = None,
    fiscal_period: str | None = None,
):
    """
    Search earnings transcript chunks semantically.
    """

    if top_k <= 0:
        raise ValueError(
            "top_k must be greater than zero."
        )

    engine = get_database_engine()
    index_metadata = load_index_metadata(
        TRANSCRIPT_METADATA_PATH
    )
    index = load_faiss_index(
        TRANSCRIPT_INDEX_PATH
    )

    validate_index_dimension(
        index=index,
        index_metadata=index_metadata,
    )

    query_vector = build_query_vector(
        query=query,
        index_metadata=index_metadata,
        index=index,
    )

    metadata = MetaData()
    chunk_table = Table(
        "earnings_transcript_chunks",
        metadata,
        autoload_with=engine,
    )
    transcript_table = Table(
        "earnings_transcripts",
        metadata,
        autoload_with=engine,
    )

    has_filters = any(
        value is not None
        for value in [
            ticker,
            fiscal_period,
        ]
    )

    def fetch_metadata(faiss_results):
        return fetch_chunk_metadata(
            engine=engine,
            chunk_table=chunk_table,
            transcript_table=transcript_table,
            faiss_results=faiss_results,
        )

    def filter_results(results):
        return apply_metadata_filters(
            results=results,
            ticker=ticker,
            fiscal_period=fiscal_period,
        )

    return search_with_filters(
        index=index,
        query_vector=query_vector,
        top_k=top_k,
        has_filters=has_filters,
        fetch_metadata=fetch_metadata,
        apply_filters=filter_results,
    )
