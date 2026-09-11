"""Persistence and cache helpers for generated AlphaLens event briefs."""

import hashlib
import json

from sqlalchemy import (
    MetaData,
    Table,
    delete,
    desc,
    func,
    insert,
    select,
)

from app.services.research_history import get_database_engine


def get_event_briefs_table(engine) -> Table:
    """Reflect the migration-managed event_briefs table."""

    return Table(
        "event_briefs",
        MetaData(),
        autoload_with=engine,
    )


def normalize_brief_tickers(request_data: dict) -> list[str]:
    """Return a stable, deduplicated ticker scope for persistence."""

    candidates = [
        *request_data.get("tickers", []),
        request_data.get("ticker"),
    ]
    tickers = []

    for candidate in candidates:
        if not candidate:
            continue

        ticker = str(candidate).strip().upper()

        if ticker and ticker not in tickers:
            tickers.append(ticker)

    return tickers


def _json_value(value):
    """Convert database dates and timestamps into cache-safe JSON values."""

    return value.isoformat() if hasattr(value, "isoformat") else value


def _latest_transcript(connection, table, ticker, fiscal_period=None):
    statement = (
        select(
            table.c.transcript_id,
            table.c.fiscal_period,
            table.c.call_date,
            table.c.updated_at,
        )
        .where(table.c.ticker == ticker)
        .order_by(
            desc(table.c.fiscal_year),
            desc(table.c.fiscal_quarter),
            desc(table.c.transcript_id),
        )
        .limit(1)
    )

    if fiscal_period:
        statement = statement.where(
            table.c.fiscal_period == fiscal_period
        )

    row = connection.execute(statement).mappings().first()
    return (
        {key: _json_value(value) for key, value in row.items()}
        if row
        else None
    )


def _latest_filing(connection, table, ticker, form_type=None):
    statement = (
        select(
            table.c.accession_number,
            table.c.form_type,
            table.c.filing_date,
            table.c.updated_at,
        )
        .where(table.c.ticker == ticker)
        .order_by(
            desc(table.c.filing_date),
            desc(table.c.accession_number),
        )
        .limit(1)
    )

    if form_type:
        statement = statement.where(
            table.c.form_type == form_type
        )

    row = connection.execute(statement).mappings().first()
    return (
        {key: _json_value(value) for key, value in row.items()}
        if row
        else None
    )


def _sentiment_versions(
    connection,
    table,
    identity_column,
    identity,
) -> list[dict]:
    """Fingerprint every model revision available for one source event."""

    if identity is None:
        return []

    statement = (
        select(
            table.c.model_name,
            table.c.model_revision,
            func.count().label("row_count"),
            func.max(table.c.updated_at).label("updated_at"),
        )
        .where(identity_column == identity)
        .group_by(
            table.c.model_name,
            table.c.model_revision,
        )
        .order_by(
            table.c.model_name,
            table.c.model_revision,
        )
    )
    rows = connection.execute(statement).mappings().all()
    return [
        {key: _json_value(value) for key, value in row.items()}
        for row in rows
    ]


def get_latest_event_fingerprint(request_data: dict) -> dict:
    """Identify the latest event rows represented by a brief request."""

    engine = get_database_engine()
    metadata = MetaData()
    transcripts = Table(
        "earnings_transcripts",
        metadata,
        autoload_with=engine,
    )
    filings = Table(
        "filings",
        metadata,
        autoload_with=engine,
    )
    transcript_sentiment = Table(
        "earnings_transcript_turn_sentiment",
        metadata,
        autoload_with=engine,
    )
    filing_sentiment = Table(
        "filing_chunk_sentiment",
        metadata,
        autoload_with=engine,
    )
    event_type = request_data.get("event_type", "combined")
    fingerprint = {"companies": []}

    with engine.connect() as connection:
        for ticker in normalize_brief_tickers(request_data):
            company = {"ticker": ticker}

            if event_type in {"earnings", "combined"}:
                company["earnings"] = _latest_transcript(
                    connection,
                    transcripts,
                    ticker,
                    request_data.get("fiscal_period"),
                )
                company["earnings_sentiment"] = _sentiment_versions(
                    connection,
                    transcript_sentiment,
                    transcript_sentiment.c.transcript_id,
                    (
                        company["earnings"]["transcript_id"]
                        if company["earnings"]
                        else None
                    ),
                )

            if event_type in {"filing", "combined"}:
                company["filing"] = _latest_filing(
                    connection,
                    filings,
                    ticker,
                    request_data.get("form_type"),
                )
                company["filing_sentiment"] = _sentiment_versions(
                    connection,
                    filing_sentiment,
                    filing_sentiment.c.accession_number,
                    (
                        company["filing"]["accession_number"]
                        if company["filing"]
                        else None
                    ),
                )

            fingerprint["companies"].append(company)

    return fingerprint


def build_event_brief_cache_key(
    request_data: dict,
    event_fingerprint: dict,
    chain_version: str,
) -> str:
    """Hash every input that can change a generated brief's meaning."""

    scope = {
        "tickers": normalize_brief_tickers(request_data),
        "event_type": request_data.get("event_type", "combined"),
        "fiscal_period": request_data.get("fiscal_period"),
        "form_type": request_data.get("form_type"),
        "top_k": request_data.get("top_k", 6),
        "focus": request_data.get("focus"),
        "chain_version": chain_version,
        "event_fingerprint": event_fingerprint,
    }
    payload = json.dumps(
        scope,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _serialize_row(row) -> dict:
    result = dict(row)
    result["created_at"] = row["created_at"].isoformat()
    result["ticker"] = (
        result.get("tickers", [None])[0]
        if result.get("tickers")
        else None
    )
    return result


def find_cached_event_brief(cache_key: str) -> dict | None:
    """Return the newest immutable snapshot matching one cache key."""

    engine = get_database_engine()
    event_briefs = get_event_briefs_table(engine)
    statement = (
        select(event_briefs)
        .where(event_briefs.c.cache_key == cache_key)
        .order_by(
            desc(event_briefs.c.created_at),
            desc(event_briefs.c.brief_id),
        )
        .limit(1)
    )

    with engine.connect() as connection:
        row = connection.execute(statement).mappings().first()

    return _serialize_row(row) if row else None


def save_event_brief(
    request_data: dict,
    result: dict,
    cache_key: str,
    event_fingerprint: dict,
) -> int:
    """Insert one immutable brief snapshot and return its ID."""

    engine = get_database_engine()
    event_briefs = get_event_briefs_table(engine)
    sources = result.get("sources", [])
    tickers = result.get("tickers") or normalize_brief_tickers(request_data)
    statement = (
        insert(event_briefs)
        .values(
            cache_key=cache_key,
            tickers=tickers,
            event_type=result["event_type"],
            fiscal_period=request_data.get("fiscal_period"),
            form_type=request_data.get("form_type"),
            top_k=request_data.get("top_k", 6),
            focus=request_data.get("focus"),
            question=result["question"],
            headline=result["brief"]["headline"],
            brief=result["brief"],
            market_context=result.get("market_context", []),
            sentiment_context=result.get("sentiment_context", {}),
            sources=sources,
            source_count=len(sources),
            event_fingerprint=event_fingerprint,
            chain_version=result["chain_version"],
            model_name=result["model_name"],
        )
        .returning(event_briefs.c.brief_id)
    )

    with engine.begin() as connection:
        return int(connection.execute(statement).scalar_one())


def list_event_briefs(limit: int = 20) -> list[dict]:
    """Return recent lightweight brief summaries, newest first."""

    engine = get_database_engine()
    event_briefs = get_event_briefs_table(engine)
    statement = (
        select(
            event_briefs.c.brief_id,
            event_briefs.c.tickers,
            event_briefs.c.event_type,
            event_briefs.c.headline,
            event_briefs.c.source_count,
            event_briefs.c.chain_version,
            event_briefs.c.model_name,
            event_briefs.c.created_at,
        )
        .order_by(
            desc(event_briefs.c.created_at),
            desc(event_briefs.c.brief_id),
        )
        .limit(limit)
    )

    with engine.connect() as connection:
        rows = connection.execute(statement).mappings().all()

    return [_serialize_row(row) for row in rows]


def get_event_brief(brief_id: int) -> dict | None:
    """Return one complete saved brief without rerunning generation."""

    engine = get_database_engine()
    event_briefs = get_event_briefs_table(engine)
    statement = select(event_briefs).where(
        event_briefs.c.brief_id == brief_id
    )

    with engine.connect() as connection:
        row = connection.execute(statement).mappings().first()

    return _serialize_row(row) if row else None


def delete_event_brief(brief_id: int) -> bool:
    """Delete one saved brief and report whether it existed."""

    engine = get_database_engine()
    event_briefs = get_event_briefs_table(engine)
    statement = delete(event_briefs).where(
        event_briefs.c.brief_id == brief_id
    )

    with engine.begin() as connection:
        result = connection.execute(statement)

    return result.rowcount > 0
