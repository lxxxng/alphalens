"""
Metadata lookups used by the AlphaLens research UI.

These queries are small and deterministic, so they belong in PostgreSQL
rather than in the vector retrieval layer.
"""

from sqlalchemy import (
    MetaData,
    Table,
    and_,
    func,
    literal_column,
    select,
)

from app.services.market_context import get_database_engine


def get_available_tickers() -> list[dict]:
    """
    Return tickers known to AlphaLens and the data each ticker has.
    """

    engine = get_database_engine()
    metadata = MetaData()

    companies = Table(
        "companies",
        metadata,
        autoload_with=engine,
    )

    filings = Table(
        "filings",
        metadata,
        autoload_with=engine,
    )

    transcripts = Table(
        "earnings_transcripts",
        metadata,
        autoload_with=engine,
    )

    market_prices = Table(
        "market_prices",
        metadata,
        autoload_with=engine,
    )

    filing_counts = (
        select(
            filings.c.ticker,
            func.count().label("filing_count"),
        )
        .group_by(filings.c.ticker)
        .subquery()
    )

    transcript_counts = (
        select(
            transcripts.c.ticker,
            func.count().label("transcript_count"),
        )
        .group_by(transcripts.c.ticker)
        .subquery()
    )

    price_counts = (
        select(
            market_prices.c.ticker,
            func.count().label("market_price_count"),
        )
        .group_by(market_prices.c.ticker)
        .subquery()
    )

    query = (
        select(
            companies.c.ticker,
            companies.c.company_name,
            func.coalesce(
                filing_counts.c.filing_count,
                literal_column("0"),
            ).label("filing_count"),
            func.coalesce(
                transcript_counts.c.transcript_count,
                literal_column("0"),
            ).label("transcript_count"),
            func.coalesce(
                price_counts.c.market_price_count,
                literal_column("0"),
            ).label("market_price_count"),
        )
        .outerjoin(
            filing_counts,
            filing_counts.c.ticker == companies.c.ticker,
        )
        .outerjoin(
            transcript_counts,
            transcript_counts.c.ticker == companies.c.ticker,
        )
        .outerjoin(
            price_counts,
            price_counts.c.ticker == companies.c.ticker,
        )
        .order_by(companies.c.ticker)
    )

    with engine.connect() as connection:
        rows = connection.execute(query).mappings().all()

    return [
        {
            "ticker": row["ticker"],
            "company_name": row["company_name"],
            "filing_count": int(row["filing_count"]),
            "transcript_count": int(row["transcript_count"]),
            "market_price_count": int(row["market_price_count"]),
        }
        for row in rows
    ]


def get_transcript_periods(
    ticker: str,
) -> list[dict]:
    """
    Return stored earnings-call periods for a ticker, newest first.
    """

    engine = get_database_engine()
    metadata = MetaData()

    transcripts = Table(
        "earnings_transcripts",
        metadata,
        autoload_with=engine,
    )

    query = (
        select(
            transcripts.c.fiscal_period,
            transcripts.c.fiscal_year,
            transcripts.c.fiscal_quarter,
            transcripts.c.call_date,
            transcripts.c.title,
            transcripts.c.turn_count,
            transcripts.c.char_count,
        )
        .where(
            transcripts.c.ticker == ticker.upper()
        )
        .order_by(
            transcripts.c.fiscal_year.desc(),
            transcripts.c.fiscal_quarter.desc(),
            transcripts.c.call_date.desc().nullslast(),
        )
    )

    with engine.connect() as connection:
        rows = connection.execute(query).mappings().all()

    return [
        {
            "fiscal_period": row["fiscal_period"],
            "fiscal_year": row["fiscal_year"],
            "fiscal_quarter": row["fiscal_quarter"],
            "call_date": (
                str(row["call_date"])
                if row["call_date"] is not None
                else None
            ),
            "title": row["title"],
            "turn_count": row["turn_count"],
            "char_count": row["char_count"],
        }
        for row in rows
    ]


def get_filing_types(
    ticker: str | None = None,
) -> list[dict]:
    """
    Return available SEC filing types, optionally filtered by ticker.
    """

    engine = get_database_engine()
    metadata = MetaData()

    filings = Table(
        "filings",
        metadata,
        autoload_with=engine,
    )

    query = select(
        filings.c.form_type,
        func.count().label("filing_count"),
    )

    if ticker:
        query = query.where(
            filings.c.ticker == ticker.upper()
        )

    query = query.group_by(
        filings.c.form_type
    ).order_by(
        filings.c.form_type
    )

    with engine.connect() as connection:
        rows = connection.execute(query).mappings().all()

    return [
        {
            "form_type": row["form_type"],
            "filing_count": int(row["filing_count"]),
        }
        for row in rows
    ]


def get_filing_sections(
    ticker: str | None = None,
    form_type: str | None = None,
) -> list[dict]:
    """
    Return available chunk section filters for SEC filings.
    """

    engine = get_database_engine()
    metadata = MetaData()

    filing_chunks = Table(
        "filing_chunks",
        metadata,
        autoload_with=engine,
    )

    filters = [
        filing_chunks.c.section_key.is_not(None)
    ]

    if ticker:
        filters.append(
            filing_chunks.c.ticker == ticker.upper()
        )

    if form_type:
        filters.append(
            filing_chunks.c.form_type == form_type.upper()
        )

    query = (
        select(
            filing_chunks.c.section_key,
            filing_chunks.c.section_title,
            func.count().label("chunk_count"),
        )
        .where(and_(*filters))
        .group_by(
            filing_chunks.c.section_key,
            filing_chunks.c.section_title,
        )
        .order_by(
            filing_chunks.c.section_key,
            filing_chunks.c.section_title,
        )
    )

    with engine.connect() as connection:
        rows = connection.execute(query).mappings().all()

    return [
        {
            "section_key": row["section_key"],
            "section_title": row["section_title"],
            "chunk_count": int(row["chunk_count"]),
        }
        for row in rows
    ]
