"""Resumable FinBERT backfill for narrative SEC filing chunks."""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone

from dotenv import load_dotenv
from sqlalchemy import MetaData, Table, and_, create_engine, or_, select
from sqlalchemy.dialects.postgresql import insert

from app.ml.transcript_sentiment import (
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_REVISION,
    FILING_SENTIMENT_SECTION_KEYS,
    FinBERTSentimentScorer,
)


load_dotenv()


# These sections contain narrative management or risk language. Table-heavy
# financial statements are intentionally outside the sentiment universe.
DEFAULT_SECTION_KEYS = FILING_SENTIMENT_SECTION_KEYS


def get_database_engine():
    """Create the configured PostgreSQL engine."""

    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise ValueError("DATABASE_URL was not found in .env.")

    return create_engine(database_url, pool_pre_ping=True)


def reflect_sentiment_tables(engine):
    """Reflect SEC sentiment tables once for a complete process."""

    metadata = MetaData()
    chunks = Table("filing_chunks", metadata, autoload_with=engine)
    sentiment = Table(
        "filing_chunk_sentiment",
        metadata,
        autoload_with=engine,
    )
    return chunks, sentiment


def get_pending_chunks(
    engine,
    model_name: str,
    model_revision: str,
    tickers: list[str] | None = None,
    accession_number: str | None = None,
    section_keys: list[str] | tuple[str, ...] = DEFAULT_SECTION_KEYS,
    limit: int | None = None,
    retry_failed: bool = False,
    tables=None,
) -> list[dict]:
    """Return targeted chunks without a terminal model result."""

    chunks, sentiment = tables or reflect_sentiment_tables(engine)
    existing = sentiment.alias("existing_sentiment")
    query = (
        select(
            chunks.c.chunk_id,
            chunks.c.accession_number,
            chunks.c.ticker,
            chunks.c.form_type,
            chunks.c.filing_date,
            chunks.c.section_key,
            chunks.c.section_title,
            chunks.c.chunk_index,
            chunks.c.content,
        )
        .select_from(
            chunks.outerjoin(
                existing,
                and_(
                    existing.c.chunk_id == chunks.c.chunk_id,
                    existing.c.model_name == model_name,
                    existing.c.model_revision == model_revision,
                ),
            )
        )
        .where(chunks.c.section_key.in_(section_keys))
        .order_by(
            chunks.c.ticker,
            chunks.c.filing_date,
            chunks.c.accession_number,
            chunks.c.section_key,
            chunks.c.chunk_index,
        )
    )
    pending = existing.c.chunk_sentiment_id.is_(None)

    if retry_failed:
        pending = or_(pending, existing.c.status == "FAILED")

    query = query.where(pending)

    if tickers:
        query = query.where(
            chunks.c.ticker.in_([ticker.upper() for ticker in tickers])
        )

    if accession_number:
        query = query.where(
            chunks.c.accession_number == accession_number
        )

    if limit is not None:
        query = query.limit(limit)

    with engine.connect() as connection:
        return [
            dict(row)
            for row in connection.execute(query).mappings().all()
        ]


def result_record(row: dict, scorer, result) -> dict:
    """Create one successful persistence record."""

    return {
        "chunk_id": row["chunk_id"],
        "accession_number": row["accession_number"],
        "model_name": scorer.model_name,
        "model_revision": scorer.model_revision,
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
        "updated_at": datetime.now(timezone.utc),
    }


def failure_record(row: dict, scorer, error: Exception) -> dict:
    """Create one retryable failure record."""

    return {
        "chunk_id": row["chunk_id"],
        "accession_number": row["accession_number"],
        "model_name": scorer.model_name,
        "model_revision": scorer.model_revision,
        "status": "FAILED",
        "error": str(error)[:2000],
        "updated_at": datetime.now(timezone.utc),
    }


def build_sentiment_records(rows: list[dict], scorer) -> list[dict]:
    """Batch score chunks, falling back to isolated failure handling."""

    try:
        results = scorer.score_many([row["content"] for row in rows])
        return [
            result_record(row, scorer, result)
            for row, result in zip(rows, results, strict=True)
        ]
    except Exception:
        records = []

        for row in rows:
            try:
                records.append(
                    result_record(row, scorer, scorer.score(row["content"]))
                )
            except Exception as error:
                records.append(failure_record(row, scorer, error))

        return records


def persist_record(engine, record: dict, tables=None):
    """Commit one idempotent chunk result."""

    _, sentiment = tables or reflect_sentiment_tables(engine)
    statement = insert(sentiment).values(**record)
    update_values = {
        key: value
        for key, value in record.items()
        if key not in {"chunk_id", "model_name", "model_revision"}
    }
    statement = statement.on_conflict_do_update(
        index_elements=["chunk_id", "model_name", "model_revision"],
        set_=update_values,
    )

    with engine.begin() as connection:
        connection.execute(statement)


def run_sentiment_backfill(
    tickers: list[str] | None = None,
    accession_number: str | None = None,
    section_keys: list[str] | None = None,
    limit: int | None = None,
    retry_failed: bool = False,
    batch_size: int = 16,
    scorer=None,
) -> dict[str, int]:
    """Score pending SEC narrative chunks and return status counts."""

    scorer = scorer or FinBERTSentimentScorer(
        model_name=os.getenv("FINBERT_MODEL_NAME", DEFAULT_MODEL_NAME),
        model_revision=os.getenv(
            "FINBERT_MODEL_REVISION",
            DEFAULT_MODEL_REVISION,
        ),
    )
    engine = get_database_engine()
    tables = reflect_sentiment_tables(engine)
    rows = get_pending_chunks(
        engine=engine,
        model_name=scorer.model_name,
        model_revision=scorer.model_revision,
        tickers=tickers,
        accession_number=accession_number,
        section_keys=section_keys or DEFAULT_SECTION_KEYS,
        limit=limit,
        retry_failed=retry_failed,
        tables=tables,
    )
    counts = {"SCORED": 0, "FAILED": 0}
    batch_size = max(1, batch_size)
    processed = 0

    print(f"Pending SEC narrative chunks: {len(rows):,}")
    print(f"Sentiment model: {scorer.model_name}@{scorer.model_revision}")

    for start in range(0, len(rows), batch_size):
        records = build_sentiment_records(
            rows[start:start + batch_size],
            scorer,
        )

        for record in records:
            persist_record(engine, record, tables=tables)
            counts[record["status"]] += 1
            processed += 1

        if processed % 25 == 0 or processed == len(rows):
            print(
                f"Processed {processed:,}/{len(rows):,} "
                f"(scored={counts['SCORED']:,}, "
                f"failed={counts['FAILED']:,})"
            )

    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Score narrative SEC filing chunks with FinBERT.",
    )
    parser.add_argument("--tickers", nargs="+", default=None)
    parser.add_argument("--accession-number", default=None)
    parser.add_argument("--section-keys", nargs="+", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    arguments = parser.parse_args()
    summary = run_sentiment_backfill(
        tickers=arguments.tickers,
        accession_number=arguments.accession_number,
        section_keys=arguments.section_keys,
        limit=arguments.limit,
        retry_failed=arguments.retry_failed,
        batch_size=arguments.batch_size,
    )

    if summary["FAILED"]:
        raise SystemExit(1)
