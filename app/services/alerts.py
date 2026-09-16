"""Persistence and ingestion-time detection for watched-company alerts."""

from __future__ import annotations

import os
from datetime import date, datetime

from dotenv import load_dotenv
from sqlalchemy import MetaData, Table, and_, create_engine, desc, func, select, update
from sqlalchemy.dialects.postgresql import insert as postgres_insert


load_dotenv()


class AlertNotFoundError(LookupError):
    """Raised when a requested event alert does not exist."""


def get_database_engine():
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise ValueError("DATABASE_URL was not found in .env.")

    return create_engine(database_url, pool_pre_ping=True)


def _tables(engine) -> dict[str, Table]:
    metadata = MetaData()
    names = ("event_alerts", "filings", "earnings_transcripts")
    return {
        name: Table(name, metadata, autoload_with=engine)
        for name in names
    }


def _iso(value):
    if isinstance(value, (date, datetime)):
        return value.isoformat()

    return value


def _serialize_alert(row) -> dict:
    values = dict(row)

    for key in ("event_date", "read_at", "created_at"):
        values[key] = _iso(values.get(key))

    return values


def create_event_alerts(
    *,
    tickers: list[str],
    since: datetime,
    ingestion_run_id: int | None = None,
) -> dict[str, int]:
    """Insert alerts for source records first created during this run."""

    normalized = sorted({ticker.strip().upper() for ticker in tickers if ticker})

    if not normalized:
        return {"filing": 0, "earnings": 0, "total": 0}

    engine = get_database_engine()
    tables = _tables(engine)
    alerts = tables["event_alerts"]
    filings = tables["filings"]
    transcripts = tables["earnings_transcripts"]

    filing_query = (
        select(
            filings.c.accession_number,
            filings.c.ticker,
            filings.c.form_type,
            filings.c.filing_date,
            filings.c.report_date,
            filings.c.source_url,
        )
        .where(
            and_(
                filings.c.ticker.in_(normalized),
                filings.c.created_at >= since,
            )
        )
    )
    transcript_query = (
        select(
            transcripts.c.transcript_id,
            transcripts.c.ticker,
            transcripts.c.fiscal_period,
            transcripts.c.call_date,
            transcripts.c.title,
        )
        .where(
            and_(
                transcripts.c.ticker.in_(normalized),
                transcripts.c.created_at >= since,
            )
        )
    )

    with engine.connect() as connection:
        filing_rows = connection.execute(filing_query).mappings().all()
        transcript_rows = connection.execute(transcript_query).mappings().all()

    records = []

    for row in filing_rows:
        report_suffix = (
            f" for the period ended {_iso(row['report_date'])}"
            if row["report_date"]
            else ""
        )
        records.append({
            "ticker": row["ticker"],
            "event_type": "filing",
            "source_record_id": row["accession_number"],
            "event_date": row["filing_date"],
            "title": f"{row['ticker']} filed {row['form_type']}",
            "message": f"New SEC filing{report_suffix}.",
            "source_url": row["source_url"],
            "ingestion_run_id": ingestion_run_id,
        })

    for row in transcript_rows:
        records.append({
            "ticker": row["ticker"],
            "event_type": "earnings",
            "source_record_id": str(row["transcript_id"]),
            "event_date": row["call_date"],
            "title": f"{row['ticker']} {row['fiscal_period']} earnings call",
            "message": row["title"] or "New earnings call transcript available.",
            "source_url": f"/transcripts/{row['transcript_id']}",
            "ingestion_run_id": ingestion_run_id,
        })

    if not records:
        return {"filing": 0, "earnings": 0, "total": 0}

    statement = (
        postgres_insert(alerts)
        .values(records)
        .on_conflict_do_nothing(
            index_elements=[alerts.c.event_type, alerts.c.source_record_id]
        )
        .returning(alerts.c.event_type)
    )

    with engine.begin() as connection:
        inserted_types = list(connection.execute(statement).scalars())

    filing_count = inserted_types.count("filing")
    earnings_count = inserted_types.count("earnings")
    return {
        "filing": filing_count,
        "earnings": earnings_count,
        "total": len(inserted_types),
    }


def list_event_alerts(
    *,
    limit: int = 30,
    unread_only: bool = False,
    ticker: str | None = None,
) -> dict:
    """Return recent alerts and the global unread count."""

    engine = get_database_engine()
    alerts = _tables(engine)["event_alerts"]
    filters = []

    if unread_only:
        filters.append(alerts.c.is_read.is_(False))

    if ticker:
        filters.append(alerts.c.ticker == ticker.strip().upper())

    query = select(alerts).order_by(desc(alerts.c.created_at)).limit(limit)

    if filters:
        query = query.where(and_(*filters))

    with engine.connect() as connection:
        rows = connection.execute(query).mappings().all()
        unread_count = int(connection.execute(
            select(func.count())
            .select_from(alerts)
            .where(alerts.c.is_read.is_(False))
        ).scalar_one())

    return {
        "alerts": [_serialize_alert(row) for row in rows],
        "unread_count": unread_count,
    }


def mark_event_alert_read(alert_id: int) -> dict:
    """Mark one alert read and return its current representation."""

    engine = get_database_engine()
    alerts = _tables(engine)["event_alerts"]
    statement = (
        update(alerts)
        .where(alerts.c.alert_id == alert_id)
        .values(is_read=True, read_at=func.now())
        .returning(alerts)
    )

    with engine.begin() as connection:
        row = connection.execute(statement).mappings().first()

    if row is None:
        raise AlertNotFoundError(f"Alert {alert_id} was not found.")

    return _serialize_alert(row)


def mark_all_event_alerts_read() -> int:
    """Mark every currently unread alert read and return the affected count."""

    engine = get_database_engine()
    alerts = _tables(engine)["event_alerts"]

    with engine.begin() as connection:
        result = connection.execute(
            update(alerts)
            .where(alerts.c.is_read.is_(False))
            .values(is_read=True, read_at=func.now())
        )

    return int(result.rowcount or 0)
