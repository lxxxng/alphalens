"""Resumable FinBERT backfill for locally stored transcript turns."""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone

from dotenv import load_dotenv
from sqlalchemy import MetaData, Table, and_, create_engine, or_, select, update
from sqlalchemy.dialects.postgresql import insert

from app.ml.transcript_sentiment import (
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_REVISION,
    FinBERTSentimentScorer,
)


load_dotenv()


def get_database_engine():
    """Create the configured PostgreSQL engine."""

    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise ValueError("DATABASE_URL was not found in .env.")

    return create_engine(database_url, pool_pre_ping=True)


def reflect_sentiment_tables(engine):
    """Reflect sentiment tables once for a complete backfill process."""

    metadata = MetaData()
    turns = Table(
        "earnings_transcript_turns",
        metadata,
        autoload_with=engine,
    )
    transcripts = Table(
        "earnings_transcripts",
        metadata,
        autoload_with=engine,
    )
    sentiment = Table(
        "earnings_transcript_turn_sentiment",
        metadata,
        autoload_with=engine,
    )
    return turns, transcripts, sentiment


def should_skip_turn(row: dict, include_operators: bool = False) -> bool:
    """Exclude empty and operator-only turns from analytical sentiment."""

    if not str(row.get("content") or "").strip():
        return True

    identity = " ".join(
        str(row.get(field) or "")
        for field in ("speaker_name", "speaker_role", "speaker_title")
    ).lower()
    return not include_operators and "operator" in identity


def build_sentiment_record(
    row: dict,
    scorer,
    include_operators: bool = False,
) -> dict:
    """Create a persistable result without coupling tests to PostgreSQL."""

    base = {
        "turn_id": row["turn_id"],
        "transcript_id": row["transcript_id"],
        "model_name": scorer.model_name,
        "model_revision": scorer.model_revision,
        "updated_at": datetime.now(timezone.utc),
    }

    if should_skip_turn(row, include_operators=include_operators):
        return {
            **base,
            "status": "SKIPPED",
            "error": None,
        }

    try:
        result = scorer.score(row["content"])
        return build_scored_record(base, result)
    except Exception as error:
        return {
            **base,
            "status": "FAILED",
            "error": str(error)[:2000],
        }


def build_scored_record(base: dict, result) -> dict:
    """Combine turn identity with one normalized model result."""

    return {
        **base,
        "model_commit": result.model_commit,
        "sentiment_label": result.label,
        "sentiment_score": result.score,
        "confidence": result.confidence,
        "probabilities": result.probabilities,
        "token_count": result.token_count,
        "segment_count": result.segment_count,
        "status": "SCORED",
        "error": None,
        "scored_at": datetime.now(timezone.utc),
    }


def build_sentiment_records(
    rows: list[dict],
    scorer,
    include_operators: bool = False,
) -> list[dict]:
    """Batch score useful turns while preserving skip and failure records."""

    records = [None] * len(rows)
    score_indexes = []

    for index, row in enumerate(rows):
        if should_skip_turn(row, include_operators=include_operators):
            records[index] = build_sentiment_record(
                row,
                scorer,
                include_operators=include_operators,
            )
        else:
            score_indexes.append(index)

    if not score_indexes:
        return records

    try:
        results = scorer.score_many(
            [rows[index]["content"] for index in score_indexes]
        )

        for index, result in zip(score_indexes, results, strict=True):
            row = rows[index]
            base = {
                "turn_id": row["turn_id"],
                "transcript_id": row["transcript_id"],
                "model_name": scorer.model_name,
                "model_revision": scorer.model_revision,
                "updated_at": datetime.now(timezone.utc),
            }
            records[index] = build_scored_record(base, result)
    except Exception:
        # Isolate individual failures so one malformed turn cannot discard a
        # complete inference batch or stop the resumable corpus backfill.
        for index in score_indexes:
            records[index] = build_sentiment_record(
                rows[index],
                scorer,
                include_operators=include_operators,
            )

    return records


def get_pending_turns(
    engine,
    model_name: str,
    model_revision: str,
    tickers: list[str] | None = None,
    transcript_id: int | None = None,
    limit: int | None = None,
    retry_failed: bool = False,
    tables=None,
) -> list[dict]:
    """Return turns without a terminal result for the selected model."""

    turns, transcripts, sentiment = (
        tables or reflect_sentiment_tables(engine)
    )
    existing = sentiment.alias("existing_sentiment")
    query = (
        select(
            turns.c.turn_id,
            turns.c.transcript_id,
            turns.c.turn_index,
            turns.c.speaker_name,
            turns.c.speaker_title,
            turns.c.speaker_role,
            turns.c.content,
            transcripts.c.ticker,
            transcripts.c.fiscal_period,
        )
        .select_from(
            turns.join(
                transcripts,
                transcripts.c.transcript_id == turns.c.transcript_id,
            ).outerjoin(
                existing,
                and_(
                    existing.c.turn_id == turns.c.turn_id,
                    existing.c.model_name == model_name,
                    existing.c.model_revision == model_revision,
                ),
            )
        )
        .order_by(
            transcripts.c.ticker,
            transcripts.c.fiscal_year,
            transcripts.c.fiscal_quarter,
            turns.c.turn_index,
        )
    )
    pending_condition = existing.c.turn_sentiment_id.is_(None)

    if retry_failed:
        pending_condition = or_(
            pending_condition,
            existing.c.status == "FAILED",
        )

    query = query.where(pending_condition)

    if tickers:
        query = query.where(
            transcripts.c.ticker.in_(
                [ticker.upper() for ticker in tickers]
            )
        )

    if transcript_id is not None:
        query = query.where(turns.c.transcript_id == transcript_id)

    if limit is not None:
        query = query.limit(limit)

    with engine.connect() as connection:
        return [
            dict(row)
            for row in connection.execute(query).mappings().all()
        ]


def persist_sentiment_record(engine, record: dict, tables=None):
    """Upsert one model result and expose its latest score on the turn."""

    turns, _, sentiment = tables or reflect_sentiment_tables(engine)
    statement = insert(sentiment).values(**record)
    update_values = {
        key: value
        for key, value in record.items()
        if key not in {"turn_id", "model_name", "model_revision"}
    }
    statement = statement.on_conflict_do_update(
        index_elements=["turn_id", "model_name", "model_revision"],
        set_=update_values,
    )

    with engine.begin() as connection:
        connection.execute(statement)

        if record["status"] == "SCORED":
            connection.execute(
                update(turns)
                .where(turns.c.turn_id == record["turn_id"])
                .values(
                    sentiment_label=record["sentiment_label"],
                    sentiment_score=record["sentiment_score"],
                )
            )


def run_sentiment_backfill(
    tickers: list[str] | None = None,
    transcript_id: int | None = None,
    limit: int | None = None,
    retry_failed: bool = False,
    include_operators: bool = False,
    batch_size: int = 16,
    scorer=None,
) -> dict[str, int]:
    """Score pending turns and return terminal status counts."""

    model_name = os.getenv("FINBERT_MODEL_NAME", DEFAULT_MODEL_NAME)
    model_revision = os.getenv(
        "FINBERT_MODEL_REVISION",
        DEFAULT_MODEL_REVISION,
    )
    scorer = scorer or FinBERTSentimentScorer(
        model_name=model_name,
        model_revision=model_revision,
    )
    engine = get_database_engine()
    tables = reflect_sentiment_tables(engine)
    rows = get_pending_turns(
        engine=engine,
        model_name=scorer.model_name,
        model_revision=scorer.model_revision,
        tickers=tickers,
        transcript_id=transcript_id,
        limit=limit,
        retry_failed=retry_failed,
        tables=tables,
    )
    counts = {"SCORED": 0, "SKIPPED": 0, "FAILED": 0}
    batch_size = max(1, batch_size)

    print(f"Pending transcript turns: {len(rows):,}")
    print(f"Sentiment model: {scorer.model_name}@{scorer.model_revision}")

    processed = 0

    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        records = build_sentiment_records(
            batch,
            scorer,
            include_operators=include_operators,
        )

        for record in records:
            persist_sentiment_record(engine, record, tables=tables)
            counts[record["status"]] += 1
            processed += 1

        if processed % 25 == 0 or processed == len(rows):
            print(
                f"Processed {processed:,}/{len(rows):,} "
                f"(scored={counts['SCORED']:,}, "
                f"skipped={counts['SKIPPED']:,}, "
                f"failed={counts['FAILED']:,})"
            )

    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Score transcript turns with financial FinBERT.",
    )
    parser.add_argument("--tickers", nargs="+", default=None)
    parser.add_argument("--transcript-id", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--include-operators", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    arguments = parser.parse_args()
    summary = run_sentiment_backfill(
        tickers=arguments.tickers,
        transcript_id=arguments.transcript_id,
        limit=arguments.limit,
        retry_failed=arguments.retry_failed,
        include_operators=arguments.include_operators,
        batch_size=max(1, arguments.batch_size),
    )

    if summary["FAILED"]:
        raise SystemExit(1)
