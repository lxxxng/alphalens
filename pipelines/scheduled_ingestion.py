"""Incremental, watchlist-scoped ingestion for unattended AlphaLens runs.

The scheduler deliberately reuses the existing ETL functions. Its extra job is
coordination: resolve ticker scope, avoid overlapping runs, select only missing
chunks, and persist stage-level run history for later monitoring and alerts.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from time import perf_counter, sleep
from typing import Callable

from dotenv import load_dotenv
from sqlalchemy import MetaData, Table, create_engine, text, update


load_dotenv()


INGESTION_LOCK_ID = 8_140_501
DEFAULT_STAGES = (
    "market",
    "sec",
    "transcripts",
    "embeddings",
    "sentiment",
    "briefs",
)


def get_database_engine():
    """Create the PostgreSQL engine shared by run tracking and scope queries."""

    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise ValueError("DATABASE_URL was not found in .env.")

    return create_engine(database_url, pool_pre_ping=True)


def normalize_tickers(tickers: list[str] | None) -> list[str]:
    """Normalize a ticker list while preserving the caller's display order."""

    normalized = []

    for ticker in tickers or []:
        value = str(ticker or "").strip().upper()

        if value and value not in normalized:
            normalized.append(value)

    return normalized


def resolve_tickers(
    engine,
    *,
    scope: str,
    explicit_tickers: list[str] | None = None,
    watchlist_id: int | None = None,
) -> list[str]:
    """Resolve explicit, one-watchlist, all-watchlists, or full-corpus scope."""

    explicit = normalize_tickers(explicit_tickers)

    if explicit:
        return explicit

    if scope == "all":
        query = text(
            "SELECT ticker FROM companies WHERE ticker <> 'SPY' ORDER BY ticker"
        )
        parameters = {}
    elif watchlist_id is not None:
        query = text(
            """
            SELECT ticker
            FROM watchlist_items
            WHERE watchlist_id = :watchlist_id
            ORDER BY added_at, ticker
            """
        )
        parameters = {"watchlist_id": watchlist_id}
    else:
        query = text(
            """
            SELECT ticker
            FROM watchlist_items
            GROUP BY ticker
            ORDER BY MIN(added_at), ticker
            """
        )
        parameters = {}

    with engine.connect() as connection:
        return normalize_tickers(
            list(connection.execute(query, parameters).scalars())
        )


def _ingestion_table(engine) -> Table:
    """Reflect the migration-managed run table with a useful setup error."""

    try:
        return Table(
            "ingestion_runs",
            MetaData(),
            autoload_with=engine,
        )
    except Exception as error:
        raise RuntimeError(
            "ingestion_runs is unavailable. Apply db/sql/014_ingestion_runs.sql."
        ) from error


def _create_run(
    engine,
    table: Table,
    *,
    trigger_type: str,
    scope: str,
    tickers: list[str],
) -> int:
    statement = (
        table.insert()
        .values(
            trigger_type=trigger_type,
            scope=scope,
            tickers=tickers,
            status="RUNNING",
            stage_results=[],
        )
        .returning(table.c.run_id)
    )

    with engine.begin() as connection:
        return int(connection.execute(statement).scalar_one())


def _update_run(engine, table: Table, run_id: int, **values) -> None:
    values["updated_at"] = datetime.now(timezone.utc)

    with engine.begin() as connection:
        connection.execute(
            update(table)
            .where(table.c.run_id == run_id)
            .values(**values)
        )


def _json_result(value):
    """Keep stage results compact and JSON-safe for PostgreSQL JSONB."""

    if value is None:
        return {}

    if isinstance(value, dict):
        return value

    if isinstance(value, (str, int, float, bool)):
        return {"result": value}

    return {"result": str(value)}


def _run_market_stage(tickers: list[str], lookback_days: int) -> dict:
    from pipelines.market_data.run_pipeline import run_market_pipeline

    start_date = (date.today() - timedelta(days=lookback_days)).isoformat()
    rows = run_market_pipeline(tickers=tickers, start_date=start_date)
    return {"rows_processed": rows, "start_date": start_date}


def _run_sec_metadata_stage(tickers: list[str]) -> dict:
    from pipelines.sec.run_pipeline import run_sec_pipeline

    return {"filings_processed": run_sec_pipeline(tickers=tickers)}


def _run_sec_documents_stage(tickers: list[str]) -> dict:
    """Download and parse selected filings; local files make reruns cheap."""

    from pipelines.sec.downloader import (
        create_sec_session,
        download_filing,
        get_database_engine as get_sec_engine,
        get_filing_candidates,
    )
    from pipelines.sec.parser import get_downloaded_filings, parse_filing

    ticker_set = set(tickers)
    engine = get_sec_engine()
    session = create_sec_session()
    filings = []

    for row in get_filing_candidates(engine):
        raw_path = row.get("raw_file_path")

        if (
            row["ticker"] in ticker_set
            and (
                row.get("download_status") != "DOWNLOADED"
                or not raw_path
                or not Path(raw_path).exists()
            )
        ):
            filings.append(row)

    downloaded = 0

    for filing in filings:
        downloaded += bool(
            download_filing(session=session, engine=engine, filing=filing)
        )
        # Keep unattended runs comfortably below SEC fair-access limits.
        sleep(0.2)

    parse_candidates = [
        row
        for row in get_downloaded_filings(engine)
        if (
            row["ticker"] in ticker_set
            and (
                row.get("parse_status") != "PARSED"
                or not row.get("clean_text_path")
                or not Path(row["clean_text_path"]).exists()
            )
        )
    ]
    parsed = sum(
        bool(parse_filing(engine=engine, filing=filing))
        for filing in parse_candidates
    )
    return {
        "download_candidates": len(filings),
        "downloaded_or_present": downloaded,
        "parse_candidates": len(parse_candidates),
        "parsed_or_present": parsed,
    }


def _run_sec_sections_stage(tickers: list[str]) -> dict:
    """Extract sections only for parsed filings with no stored sections."""

    from pipelines.sec.section_extractor import (
        get_database_engine as get_sec_engine,
        get_parsed_filings,
        process_filing,
    )

    engine = get_sec_engine()
    ticker_set = set(tickers)

    with engine.connect() as connection:
        existing = set(connection.execute(text(
            "SELECT DISTINCT accession_number FROM filing_sections"
        )).scalars())

    candidates = [
        row
        for row in get_parsed_filings(engine)
        if row["ticker"] in ticker_set
        and row["accession_number"] not in existing
    ]
    sections = sum(
        process_filing(engine=engine, filing=filing)
        for filing in candidates
    )
    return {"filings_processed": len(candidates), "sections_created": sections}


def _run_sec_chunks_stage(tickers: list[str]) -> dict:
    """Chunk only previously unchunked sections to preserve FAISS IDs."""

    from pipelines.sec.chunker import (
        get_database_engine as get_sec_engine,
        get_sections,
        process_section,
    )

    engine = get_sec_engine()
    chunk_table = Table(
        "filing_chunks",
        MetaData(),
        autoload_with=engine,
    )

    with engine.connect() as connection:
        existing = set(connection.execute(text(
            "SELECT DISTINCT section_id FROM filing_chunks"
        )).scalars())

    ticker_set = set(tickers)
    candidates = [
        row
        for row in get_sections(engine)
        if row["ticker"] in ticker_set and row["section_id"] not in existing
    ]
    chunks = sum(
        process_section(engine, chunk_table, section)
        for section in candidates
    )
    return {"sections_processed": len(candidates), "chunks_created": chunks}


def _run_transcript_extract_stage(
    tickers: list[str],
    max_transcripts_per_ticker: int,
) -> dict:
    from pipelines.transcripts.run_pipeline import run_transcript_pipeline

    loaded = run_transcript_pipeline(
        tickers=tickers,
        max_transcripts_per_ticker=max_transcripts_per_ticker,
    )
    return {"transcripts_processed": loaded}


def _run_transcript_chunks_stage(tickers: list[str]) -> dict:
    """Chunk only calls that do not already have stable chunk IDs."""

    from pipelines.transcripts.chunker import (
        chunk_transcript_content,
        chunk_turns,
        get_database_engine as get_transcript_engine,
        get_transcripts,
        get_turns_for_transcript,
        save_chunks,
    )

    engine = get_transcript_engine()
    metadata = MetaData()
    transcript_table = Table(
        "earnings_transcripts", metadata, autoload_with=engine
    )
    turn_table = Table(
        "earnings_transcript_turns", metadata, autoload_with=engine
    )
    chunk_table = Table(
        "earnings_transcript_chunks", metadata, autoload_with=engine
    )

    with engine.connect() as connection:
        existing = set(connection.execute(text(
            "SELECT DISTINCT transcript_id FROM earnings_transcript_chunks"
        )).scalars())

    ticker_set = set(tickers)
    candidates = [
        row
        for row in get_transcripts(engine, transcript_table)
        if row["ticker"] in ticker_set and row["transcript_id"] not in existing
    ]
    chunks_created = 0

    for transcript in candidates:
        turns = get_turns_for_transcript(
            engine,
            turn_table,
            transcript["transcript_id"],
        )
        chunks = (
            chunk_turns(turns)
            if turns
            else chunk_transcript_content(transcript["content"])
        )
        chunks_created += save_chunks(
            engine,
            chunk_table,
            transcript,
            chunks,
        )

    return {
        "transcripts_processed": len(candidates),
        "chunks_created": chunks_created,
    }


def _run_embedding_stage() -> dict:
    from pipelines.sec.embedder import run_embedding_pipeline as embed_sec
    from pipelines.transcripts.embedder import (
        run_embedding_pipeline as embed_transcripts,
    )

    embed_sec()
    embed_transcripts()
    return {"indexes": ["filings", "transcripts"]}


def _run_sentiment_stage(tickers: list[str]) -> dict:
    from pipelines.sec.sentiment import run_sentiment_backfill as score_sec
    from pipelines.transcripts.sentiment import (
        run_sentiment_backfill as score_transcripts,
    )

    sec = score_sec(tickers=tickers)
    transcripts = score_transcripts(tickers=tickers)

    if sec.get("FAILED", 0) or transcripts.get("FAILED", 0):
        raise RuntimeError(
            "Sentiment scoring left failed rows; rerun with retry-failed enabled."
        )

    return {"sec": sec, "transcripts": transcripts}


def _run_event_alert_stage(
    tickers: list[str],
    *,
    since: datetime,
    ingestion_run_id: int | None,
) -> dict:
    """Create deduplicated alerts for events inserted during this run."""

    from app.services.alerts import create_event_alerts

    return create_event_alerts(
        tickers=tickers,
        since=since,
        ingestion_run_id=ingestion_run_id,
    )


def _run_automated_brief_stage(
    tickers: list[str],
    *,
    max_briefs: int,
) -> dict:
    """Generate bounded, quality-gated briefs for queued event alerts."""

    from app.services.automated_briefs import run_automated_event_briefs

    return run_automated_event_briefs(
        tickers,
        max_briefs=max_briefs,
    )


def _stage_plan(
    tickers: list[str],
    stages: list[str],
    *,
    market_lookback_days: int,
    max_transcripts_per_ticker: int,
    ingestion_run_id: int | None = None,
    run_started_at: datetime | None = None,
    max_auto_briefs: int = 5,
) -> list[tuple[str, Callable[[], dict]]]:
    """Expand user-facing stage groups into ordered, observable work units."""

    requested = set(stages)
    plan = []

    if "market" in requested:
        plan.append((
            "market",
            lambda: _run_market_stage(tickers, market_lookback_days),
        ))

    if "sec" in requested:
        plan.extend((
            ("sec_metadata", lambda: _run_sec_metadata_stage(tickers)),
            ("sec_documents", lambda: _run_sec_documents_stage(tickers)),
            ("sec_sections", lambda: _run_sec_sections_stage(tickers)),
            ("sec_chunks", lambda: _run_sec_chunks_stage(tickers)),
        ))

    if "transcripts" in requested:
        plan.extend((
            (
                "transcript_extract",
                lambda: _run_transcript_extract_stage(
                    tickers,
                    max_transcripts_per_ticker,
                ),
            ),
            ("transcript_chunks", lambda: _run_transcript_chunks_stage(tickers)),
        ))

    if requested.intersection({"sec", "transcripts"}):
        plan.append((
            "event_alerts",
            lambda: _run_event_alert_stage(
                tickers,
                since=run_started_at or datetime.now(timezone.utc),
                ingestion_run_id=ingestion_run_id,
            ),
        ))

    if "embeddings" in requested:
        plan.append(("embeddings", _run_embedding_stage))

    if "sentiment" in requested:
        plan.append(("sentiment", lambda: _run_sentiment_stage(tickers)))

    if "briefs" in requested:
        plan.append((
            "automated_briefs",
            lambda: _run_automated_brief_stage(
                tickers,
                max_briefs=max_auto_briefs,
            ),
        ))

    return plan


def run_scheduled_ingestion(
    *,
    scope: str = "watchlists",
    tickers: list[str] | None = None,
    watchlist_id: int | None = None,
    stages: list[str] | None = None,
    trigger_type: str = "manual",
    market_lookback_days: int = 14,
    max_transcripts_per_ticker: int = 2,
    max_auto_briefs: int = 5,
    dry_run: bool = False,
) -> dict:
    """Run an incremental refresh and persist an inspectable terminal state."""

    engine = get_database_engine()
    selected_tickers = resolve_tickers(
        engine,
        scope=scope,
        explicit_tickers=tickers,
        watchlist_id=watchlist_id,
    )
    selected_stages = list(stages or DEFAULT_STAGES)

    if dry_run:
        return {
            "status": "DRY_RUN",
            "scope": scope,
            "tickers": selected_tickers,
            "stages": selected_stages,
        }

    table = _ingestion_table(engine)
    lock_connection = engine.connect()
    locked = bool(lock_connection.execute(
        text("SELECT pg_try_advisory_lock(:lock_id)"),
        {"lock_id": INGESTION_LOCK_ID},
    ).scalar_one())

    if not locked:
        lock_connection.close()
        run_id = _create_run(
            engine,
            table,
            trigger_type=trigger_type,
            scope=scope,
            tickers=selected_tickers,
        )
        _update_run(
            engine,
            table,
            run_id,
            status="SKIPPED",
            error="Another ingestion run already holds the scheduler lock.",
            completed_at=datetime.now(timezone.utc),
        )
        return {"run_id": run_id, "status": "SKIPPED", "tickers": selected_tickers}

    run_started_at = datetime.now(timezone.utc)
    run_id = _create_run(
        engine,
        table,
        trigger_type=trigger_type,
        scope=scope,
        tickers=selected_tickers,
    )
    results = []

    try:
        if not selected_tickers:
            _update_run(
                engine,
                table,
                run_id,
                status="SKIPPED",
                error="No tickers are present in the selected scope.",
                completed_at=datetime.now(timezone.utc),
            )
            return {"run_id": run_id, "status": "SKIPPED", "tickers": []}

        plan = _stage_plan(
            selected_tickers,
            selected_stages,
            market_lookback_days=max(2, market_lookback_days),
            max_transcripts_per_ticker=max(1, max_transcripts_per_ticker),
            ingestion_run_id=run_id,
            run_started_at=run_started_at,
            max_auto_briefs=max(0, max_auto_briefs),
        )

        print(f"Ingestion run {run_id} | tickers: {', '.join(selected_tickers)}")
        print(f"Stages: {', '.join(name for name, _ in plan)}")

        for stage_name, operation in plan:
            started = perf_counter()
            print(f"\n[START] {stage_name}", flush=True)
            _update_run(engine, table, run_id, current_stage=stage_name)

            if (
                stage_name == "automated_briefs"
                and any(item["status"] == "FAILED" for item in results)
            ):
                blockers = [
                    item["stage"]
                    for item in results
                    if item["status"] == "FAILED"
                ]
                detail = {
                    "reason": "Earlier ingestion stages failed.",
                    "blocking_stages": blockers,
                }
                results.append({
                    "stage": stage_name,
                    "status": "SKIPPED",
                    "duration_ms": round((perf_counter() - started) * 1000, 1),
                    "detail": detail,
                    "error": None,
                })
                _update_run(engine, table, run_id, stage_results=results)
                print(
                    f"[SKIPPED] {stage_name}: failed dependencies "
                    f"({', '.join(blockers)})",
                    flush=True,
                )
                continue

            try:
                detail = _json_result(operation())
                stage_status = "SUCCEEDED"
                stage_error = None
                print(f"[OK] {stage_name}", flush=True)
            except Exception as error:
                detail = {}
                stage_status = "FAILED"
                stage_error = str(error)[:2000]
                print(f"[FAILED] {stage_name}: {error}", flush=True)

            results.append({
                "stage": stage_name,
                "status": stage_status,
                "duration_ms": round((perf_counter() - started) * 1000, 1),
                "detail": detail,
                "error": stage_error,
            })
            _update_run(engine, table, run_id, stage_results=results)

        failed = [item for item in results if item["status"] == "FAILED"]
        final_status = "FAILED" if failed else "SUCCEEDED"
        error = (
            f"{len(failed)} stage(s) failed: "
            + ", ".join(item["stage"] for item in failed)
            if failed
            else None
        )
        _update_run(
            engine,
            table,
            run_id,
            status=final_status,
            current_stage=None,
            error=error,
            completed_at=datetime.now(timezone.utc),
        )
        return {
            "run_id": run_id,
            "status": final_status,
            "tickers": selected_tickers,
            "stage_results": results,
        }
    finally:
        lock_connection.execute(
            text("SELECT pg_advisory_unlock(:lock_id)"),
            {"lock_id": INGESTION_LOCK_ID},
        )
        lock_connection.close()


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Incrementally refresh AlphaLens data for monitored tickers.",
    )
    parser.add_argument("--scope", choices=("watchlists", "all"), default="watchlists")
    parser.add_argument("--watchlist-id", type=int, default=None)
    parser.add_argument("--tickers", nargs="+", default=None)
    parser.add_argument("--stages", nargs="+", choices=DEFAULT_STAGES, default=None)
    parser.add_argument("--trigger-type", default="manual")
    parser.add_argument("--market-lookback-days", type=int, default=14)
    parser.add_argument("--max-transcripts-per-ticker", type=int, default=2)
    parser.add_argument("--max-auto-briefs", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    return parser


if __name__ == "__main__":
    arguments = build_argument_parser().parse_args()
    result = run_scheduled_ingestion(
        scope=arguments.scope,
        tickers=arguments.tickers,
        watchlist_id=arguments.watchlist_id,
        stages=arguments.stages,
        trigger_type=arguments.trigger_type,
        market_lookback_days=arguments.market_lookback_days,
        max_transcripts_per_ticker=arguments.max_transcripts_per_ticker,
        max_auto_briefs=arguments.max_auto_briefs,
        dry_run=arguments.dry_run,
    )
    print("\nIngestion result:")
    print(json.dumps(result, indent=2, default=str))

    if result["status"] == "FAILED":
        raise SystemExit(1)
