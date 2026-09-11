"""
SEC filing semantic retrieval.
"""

from sqlalchemy import MetaData, Table, select

from app.rag.retrievers.base import (
    DEFAULT_TOP_K,
    SEC_INDEX_PATH,
    SEC_METADATA_PATH,
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
    faiss_results,
):
    """
    Retrieve SEC filing chunk text and metadata from PostgreSQL.
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
                "source_type": "filing",
                "chunk_id": chunk_id,
                "score": faiss_result["score"],
                "ticker": database_row["ticker"],
                "form_type": database_row["form_type"],
                "filing_date": database_row["filing_date"],
                "accession_number": database_row["accession_number"],
                "section_key": database_row["section_key"],
                "section_title": database_row["section_title"],
                "chunk_index": database_row["chunk_index"],
                "token_count": database_row["token_count"],
                "content": database_row["content"],
            }
        )

    return combined_results


def find_latest_accession(
    engine,
    chunk_table,
    ticker: str | None,
    form_type: str | None,
) -> str | None:
    """Return the newest filing represented by the selected SEC chunks."""

    if not ticker:
        return None

    conditions = [chunk_table.c.ticker == ticker.upper()]

    if form_type:
        conditions.append(chunk_table.c.form_type == form_type.upper())

    query = (
        select(chunk_table.c.accession_number)
        .where(*conditions)
        .order_by(
            chunk_table.c.filing_date.desc(),
            chunk_table.c.accession_number.desc(),
        )
        .limit(1)
    )

    with engine.connect() as connection:
        return connection.execute(query).scalar_one_or_none()


def semantic_search(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    ticker: str | None = None,
    form_type: str | None = None,
    section_key: str | None = None,
    prefer_latest: bool = False,
):
    """
    Search SEC filing chunks semantically.
    """

    if top_k <= 0:
        raise ValueError(
            "top_k must be greater than zero."
        )

    engine = get_database_engine()
    index_metadata = load_index_metadata(
        SEC_METADATA_PATH
    )
    index = load_faiss_index(
        SEC_INDEX_PATH
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
        "filing_chunks",
        metadata,
        autoload_with=engine,
    )
    latest_accession = (
        find_latest_accession(
            engine=engine,
            chunk_table=chunk_table,
            ticker=ticker,
            form_type=form_type,
        )
        if prefer_latest
        else None
    )

    has_filters = any(
        value is not None
        for value in [
            ticker,
            form_type,
            section_key,
            latest_accession,
        ]
    )

    def fetch_metadata(faiss_results):
        return fetch_chunk_metadata(
            engine=engine,
            chunk_table=chunk_table,
            faiss_results=faiss_results,
        )

    def filter_results(results):
        filtered = apply_metadata_filters(
            results=results,
            ticker=ticker,
            form_type=form_type,
            section_key=section_key,
        )

        if latest_accession:
            filtered = [
                result
                for result in filtered
                if result.get("accession_number") == latest_accession
            ]

        return filtered

    return search_with_filters(
        index=index,
        query_vector=query_vector,
        top_k=top_k,
        has_filters=has_filters,
        fetch_metadata=fetch_metadata,
        apply_filters=filter_results,
    )
