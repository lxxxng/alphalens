"""Generate, evaluate, and attach briefs for queued event alerts."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy import MetaData, Table, and_, or_, select, update

from app.rag.brief_workflow import BRIEF_CHAIN_VERSION, generate_event_brief
from app.services.brief_quality import evaluate_event_brief
from app.services.event_brief_history import (
    build_event_brief_cache_key,
    find_cached_event_brief,
    get_latest_event_fingerprint,
    save_event_brief,
    update_event_brief_evaluation,
)
from app.services.research_history import get_database_engine


AUTOMATED_BRIEF_FOCUS = (
    "Prioritize what changed, forward guidance, material risks, and specific "
    "items to monitor after this event."
)


def _tables(engine) -> dict[str, Table]:
    metadata = MetaData()
    return {
        name: Table(name, metadata, autoload_with=engine)
        for name in (
            "event_alerts",
            "filings",
            "earnings_transcripts",
        )
    }


def _load_candidates(
    engine,
    alerts: Table,
    tickers: list[str],
    *,
    limit: int,
    max_attempts: int,
) -> list[dict]:
    stale_before = datetime.now(timezone.utc) - timedelta(hours=1)
    retryable = or_(
        alerts.c.brief_status.in_(["PENDING", "FAILED", "REJECTED"]),
        and_(
            alerts.c.brief_status == "GENERATING",
            or_(
                alerts.c.brief_last_attempt_at.is_(None),
                alerts.c.brief_last_attempt_at < stale_before,
            ),
        ),
    )
    statement = (
        select(alerts)
        .where(
            alerts.c.ticker.in_(tickers),
            alerts.c.brief_attempt_count < max_attempts,
            retryable,
        )
        .order_by(alerts.c.created_at, alerts.c.alert_id)
        .limit(limit)
    )

    with engine.connect() as connection:
        return [dict(row) for row in connection.execute(statement).mappings()]


def _claim_alert(engine, alerts: Table, alert: dict) -> bool:
    """Claim one candidate only if another worker has not changed it."""

    with engine.begin() as connection:
        result = connection.execute(
            update(alerts)
            .where(
                alerts.c.alert_id == alert["alert_id"],
                alerts.c.brief_status == alert["brief_status"],
                alerts.c.brief_attempt_count == alert["brief_attempt_count"],
            )
            .values(
                brief_status="GENERATING",
                brief_attempt_count=alerts.c.brief_attempt_count + 1,
                brief_error=None,
                brief_last_attempt_at=datetime.now(timezone.utc),
            )
        )

    return result.rowcount == 1


def _build_request(connection, tables: dict[str, Table], alert: dict) -> dict:
    request = {
        "ticker": alert["ticker"],
        "tickers": [],
        "event_type": alert["event_type"],
        "top_k": 6,
        "focus": AUTOMATED_BRIEF_FOCUS,
        "fiscal_period": None,
        "form_type": None,
        "accession_number": None,
        "transcript_id": None,
    }

    if alert["event_type"] == "filing":
        accession_number = alert["source_record_id"]
        row = connection.execute(
            select(tables["filings"].c.form_type)
            .where(tables["filings"].c.accession_number == accession_number)
        ).mappings().first()

        if row is None:
            raise LookupError(f"Filing {accession_number} was not found.")

        request["form_type"] = row["form_type"]
        request["accession_number"] = accession_number
    else:
        transcript_id = int(alert["source_record_id"])
        row = connection.execute(
            select(tables["earnings_transcripts"].c.fiscal_period)
            .where(
                tables["earnings_transcripts"].c.transcript_id == transcript_id
            )
        ).mappings().first()

        if row is None:
            raise LookupError(f"Transcript {transcript_id} was not found.")

        request["fiscal_period"] = row["fiscal_period"]
        request["transcript_id"] = transcript_id

    return request


def _generate_snapshot(request: dict, *, force_refresh: bool) -> tuple[dict, int, bool]:
    fingerprint = get_latest_event_fingerprint(request)
    cache_key = build_event_brief_cache_key(
        request_data=request,
        event_fingerprint=fingerprint,
        chain_version=BRIEF_CHAIN_VERSION,
    )
    cached = None if force_refresh else find_cached_event_brief(cache_key)

    if cached:
        evaluation = cached.get("quality_evaluation") or evaluate_event_brief(
            cached,
            request,
        )

        if not cached.get("quality_evaluation"):
            update_event_brief_evaluation(cached["brief_id"], evaluation)

        cached["quality_evaluation"] = evaluation
        return cached, int(cached["brief_id"]), True

    result = generate_event_brief(
        ticker=request["ticker"],
        event_type=request["event_type"],
        fiscal_period=request.get("fiscal_period"),
        form_type=request.get("form_type"),
        top_k=request["top_k"],
        focus=request.get("focus"),
        accession_number=request.get("accession_number"),
        transcript_id=request.get("transcript_id"),
    )
    evaluation = evaluate_event_brief(result, request)
    result["quality_evaluation"] = evaluation
    brief_id = save_event_brief(
        request_data=request,
        result=result,
        cache_key=cache_key,
        event_fingerprint=fingerprint,
        generation_source="automatic",
        quality_evaluation=evaluation,
    )
    result["brief_id"] = brief_id
    return result, brief_id, False


def run_automated_event_briefs(
    tickers: list[str],
    *,
    max_briefs: int = 5,
    max_attempts: int = 3,
) -> dict:
    """Process a bounded batch of retryable alert brief jobs."""

    normalized = sorted({ticker.strip().upper() for ticker in tickers if ticker})

    if not normalized or max_briefs <= 0:
        return {
            "candidates": 0,
            "processed": 0,
            "passed": 0,
            "rejected": 0,
            "failed": 0,
            "cached": 0,
            "brief_ids": [],
        }

    engine = get_database_engine()
    tables = _tables(engine)
    alerts = tables["event_alerts"]
    candidates = _load_candidates(
        engine,
        alerts,
        normalized,
        limit=max_briefs,
        max_attempts=max_attempts,
    )
    summary = {
        "candidates": len(candidates),
        "processed": 0,
        "passed": 0,
        "rejected": 0,
        "failed": 0,
        "cached": 0,
        "brief_ids": [],
    }

    for alert in candidates:
        if not _claim_alert(engine, alerts, alert):
            continue

        summary["processed"] += 1

        try:
            with engine.connect() as connection:
                request = _build_request(connection, tables, alert)

            result, brief_id, cached = _generate_snapshot(
                request,
                force_refresh=alert["brief_status"] == "REJECTED",
            )
            evaluation = result["quality_evaluation"]
            status = "PASSED" if evaluation["passed"] else "REJECTED"

            with engine.begin() as connection:
                connection.execute(
                    update(alerts)
                    .where(alerts.c.alert_id == alert["alert_id"])
                    .values(
                        brief_id=brief_id,
                        brief_status=status,
                        brief_evaluation=evaluation,
                        brief_error=None,
                        brief_generated_at=datetime.now(timezone.utc),
                    )
                )

            summary[status.lower()] += 1
            summary["cached"] += int(cached)
            summary["brief_ids"].append(brief_id)
        except Exception as error:
            with engine.begin() as connection:
                connection.execute(
                    update(alerts)
                    .where(alerts.c.alert_id == alert["alert_id"])
                    .values(
                        brief_status="FAILED",
                        brief_error=str(error)[:2000],
                    )
                )

            summary["failed"] += 1

    return summary
