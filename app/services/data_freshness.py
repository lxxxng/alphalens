"""Operational freshness checks for market data and scheduled ingestion."""

from __future__ import annotations

import os
from datetime import date, datetime, timezone

from dotenv import load_dotenv
from sqlalchemy import create_engine, text


load_dotenv()


DEFAULT_MARKET_MAX_AGE_DAYS = 5
DEFAULT_SCHEDULER_MAX_AGE_HOURS = 72


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(value) -> datetime | None:
    if value is None:
        return None

    if isinstance(value, date) and not isinstance(value, datetime):
        value = datetime.combine(value, datetime.min.time())

    parsed = value if isinstance(value, datetime) else datetime.fromisoformat(str(value))

    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)

    return parsed.astimezone(timezone.utc)


def _iso(value) -> str | None:
    if value is None:
        return None

    return value.isoformat() if hasattr(value, "isoformat") else str(value)


def _positive_setting(name: str, default: int) -> int:
    raw_value = os.getenv(name, str(default)).strip()

    try:
        value = int(raw_value)
    except ValueError as error:
        raise ValueError(f"{name} must be a positive integer.") from error

    if value < 1:
        raise ValueError(f"{name} must be a positive integer.")

    return value


def _run_payload(row: dict | None, now: datetime) -> dict | None:
    if row is None:
        return None

    completed = _as_utc(row.get("completed_at"))
    started = _as_utc(row.get("started_at"))
    reference = completed or started
    return {
        "run_id": int(row["run_id"]),
        "status": row["status"],
        "trigger_type": row.get("trigger_type"),
        "scope": row.get("scope"),
        "current_stage": row.get("current_stage"),
        "started_at": _iso(started),
        "completed_at": _iso(completed),
        "age_hours": (
            max((now - reference).total_seconds() / 3600, 0.0)
            if reference is not None
            else None
        ),
        "error": row.get("error"),
    }


def assess_data_freshness(
    *,
    market_rows: list[dict],
    latest_run: dict | None,
    latest_success: dict | None,
    source_activity: dict[str, dict],
    now: datetime | None = None,
    market_max_age_days: int = DEFAULT_MARKET_MAX_AGE_DAYS,
    scheduler_max_age_hours: int = DEFAULT_SCHEDULER_MAX_AGE_HOURS,
) -> dict:
    """Apply stable freshness policy to already loaded operational facts."""

    if market_max_age_days < 1 or scheduler_max_age_hours < 1:
        raise ValueError("Freshness thresholds must be positive.")

    current = _as_utc(now or _utc_now())
    market = []

    for row in market_rows:
        latest = row.get("latest_trading_date")
        if isinstance(latest, datetime):
            latest_date = latest.date()
        elif isinstance(latest, date):
            latest_date = latest
        else:
            latest_date = (
                date.fromisoformat(str(latest)[:10])
                if latest is not None
                else None
            )
        age_days = (
            max((current.date() - latest_date).days, 0)
            if latest_date is not None
            else None
        )
        market.append({
            "ticker": row["ticker"],
            "latest_trading_date": _iso(latest_date),
            "age_days": age_days,
            "fresh": age_days is not None and age_days <= market_max_age_days,
        })

    stale_tickers = [row for row in market if not row["fresh"]]
    latest_payload = _run_payload(latest_run, current)
    success_payload = _run_payload(latest_success, current)
    issues = []

    if not market:
        issues.append({
            "code": "market_data_missing",
            "severity": "critical",
            "message": "No market prices exist for watched companies or SPY.",
        })
    elif stale_tickers:
        names = ", ".join(row["ticker"] for row in stale_tickers[:6])
        suffix = "..." if len(stale_tickers) > 6 else ""
        issues.append({
            "code": "market_data_stale",
            "severity": "critical",
            "message": (
                f"Market data exceeds {market_max_age_days} days for "
                f"{names}{suffix}."
            ),
        })

    if success_payload is None:
        issues.append({
            "code": "scheduler_never_succeeded",
            "severity": "critical",
            "message": "No successful scheduled ingestion run was found.",
        })
    elif (
        success_payload["age_hours"] is None
        or success_payload["age_hours"] > scheduler_max_age_hours
    ):
        issues.append({
            "code": "scheduler_stale",
            "severity": "critical",
            "message": (
                "The last successful ingestion run exceeds "
                f"{scheduler_max_age_hours} hours."
            ),
        })

    if latest_payload and latest_payload["status"] == "FAILED":
        issues.append({
            "code": "latest_scheduler_failed",
            "severity": "critical",
            "message": "The latest scheduled ingestion run failed.",
        })

    critical = [issue for issue in issues if issue["severity"] == "critical"]
    latest_market_date = max(
        (
            row["latest_trading_date"]
            for row in market
            if row["latest_trading_date"] is not None
        ),
        default=None,
    )
    return {
        "status": "stale" if critical else "fresh",
        "ready": not critical,
        "checked_at": current.isoformat(),
        "thresholds": {
            "market_max_age_days": market_max_age_days,
            "scheduler_max_age_hours": scheduler_max_age_hours,
        },
        "market": {
            "latest_trading_date": latest_market_date,
            "tracked_tickers": len(market),
            "fresh_tickers": len(market) - len(stale_tickers),
            "stale_tickers": stale_tickers,
            "tickers": market,
        },
        "scheduler": {
            "latest_run": latest_payload,
            "latest_success": success_payload,
        },
        "sources": {
            name: {
                "rows": int(values.get("rows") or 0),
                "last_activity_at": _iso(values.get("last_activity_at")),
            }
            for name, values in source_activity.items()
        },
        "issues": issues,
    }


def _load_freshness_facts(engine) -> dict:
    market_query = text("""
        WITH tracked AS (
            SELECT DISTINCT ticker FROM watchlist_items
            UNION
            SELECT 'SPY'
        )
        SELECT tracked.ticker, MAX(market_prices.trading_date) AS latest_trading_date
        FROM tracked
        LEFT JOIN market_prices ON market_prices.ticker = tracked.ticker
        GROUP BY tracked.ticker
        ORDER BY tracked.ticker
    """)
    latest_run_query = text("""
        SELECT run_id, trigger_type, scope, status, current_stage, error,
               started_at, completed_at
        FROM ingestion_runs
        ORDER BY started_at DESC
        LIMIT 1
    """)
    latest_success_query = text("""
        SELECT run_id, trigger_type, scope, status, current_stage, error,
               started_at, completed_at
        FROM ingestion_runs
        WHERE status = 'SUCCEEDED'
        ORDER BY completed_at DESC
        LIMIT 1
    """)
    source_queries = {
        "sec_filings": text(
            "SELECT COUNT(*) AS rows, MAX(created_at) AS last_activity_at FROM filings"
        ),
        "earnings_transcripts": text(
            "SELECT COUNT(*) AS rows, MAX(created_at) AS last_activity_at "
            "FROM earnings_transcripts"
        ),
        "sentiment": text("""
            SELECT
                (SELECT COUNT(*) FROM earnings_transcript_turn_sentiment)
                + (SELECT COUNT(*) FROM filing_chunk_sentiment) AS rows,
                GREATEST(
                    (SELECT MAX(scored_at) FROM earnings_transcript_turn_sentiment),
                    (SELECT MAX(scored_at) FROM filing_chunk_sentiment)
                ) AS last_activity_at
        """),
        "prospective_predictions": text(
            "SELECT COUNT(*) AS rows, MAX(updated_at) AS last_activity_at "
            "FROM prospective_predictions"
        ),
    }

    with engine.connect() as connection:
        market_rows = [
            dict(row)
            for row in connection.execute(market_query).mappings().all()
        ]
        latest_run = connection.execute(latest_run_query).mappings().first()
        latest_success = connection.execute(
            latest_success_query
        ).mappings().first()
        sources = {
            name: dict(connection.execute(query).mappings().one())
            for name, query in source_queries.items()
        }

    return {
        "market_rows": market_rows,
        "latest_run": dict(latest_run) if latest_run else None,
        "latest_success": dict(latest_success) if latest_success else None,
        "source_activity": sources,
    }


def get_data_freshness(engine=None, *, now: datetime | None = None) -> dict:
    """Load database timestamps and return deployment-readiness evidence."""

    resolved_engine = engine

    if resolved_engine is None:
        database_url = os.getenv("DATABASE_URL", "").strip()

        if not database_url:
            raise ValueError("DATABASE_URL is not configured.")

        resolved_engine = create_engine(database_url, pool_pre_ping=True)

    facts = _load_freshness_facts(resolved_engine)
    return assess_data_freshness(
        **facts,
        now=now,
        market_max_age_days=_positive_setting(
            "ALPHALENS_MARKET_MAX_AGE_DAYS",
            DEFAULT_MARKET_MAX_AGE_DAYS,
        ),
        scheduler_max_age_hours=_positive_setting(
            "ALPHALENS_SCHEDULER_MAX_AGE_HOURS",
            DEFAULT_SCHEDULER_MAX_AGE_HOURS,
        ),
    )
