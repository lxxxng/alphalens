"""Coverage-aware sentiment summaries for transcripts and SEC filings."""

from __future__ import annotations

import os
from collections import defaultdict

from dotenv import load_dotenv
from sqlalchemy import MetaData, Table, and_, create_engine, select

from app.ml.transcript_sentiment import (
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_REVISION,
    EXPECTED_LABELS,
    FILING_SENTIMENT_SECTION_KEYS,
)


load_dotenv()


def get_database_engine():
    """Create the configured PostgreSQL engine."""

    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise ValueError("DATABASE_URL was not found in .env.")

    return create_engine(database_url, pool_pre_ping=True)


def model_config() -> tuple[str, str]:
    """Return the exact sentiment model identity used by API summaries."""

    return (
        os.getenv("FINBERT_MODEL_NAME", DEFAULT_MODEL_NAME),
        os.getenv("FINBERT_MODEL_REVISION", DEFAULT_MODEL_REVISION),
    )


def speaker_group(row: dict) -> str:
    """Normalize provider roles into stable analytical groups."""

    identity = " ".join(
        str(row.get(field) or "")
        for field in ("speaker_role", "speaker_name", "speaker_title")
    ).lower()

    if "operator" in identity:
        return "operator"

    if "analyst" in identity:
        return "analyst"

    return "management"


def summarize_sentiment_rows(rows: list[dict]) -> dict:
    """Token-weight scored rows so tiny greetings cannot dominate tone."""

    scored = [row for row in rows if row.get("status") == "SCORED"]

    if not scored:
        return {
            "label": None,
            "score": None,
            "confidence": None,
            "probabilities": None,
            "scored_items": 0,
            "token_count": 0,
        }

    weights = [max(1, int(row.get("token_count") or 1)) for row in scored]
    total_weight = sum(weights)
    probabilities = {label: 0.0 for label in EXPECTED_LABELS}
    weighted_score = 0.0
    weighted_confidence = 0.0

    for row, weight in zip(scored, weights, strict=True):
        weighted_score += float(row["sentiment_score"]) * weight
        weighted_confidence += float(row["confidence"]) * weight

        for label in EXPECTED_LABELS:
            probabilities[label] += (
                float((row.get("probabilities") or {}).get(label, 0.0))
                * weight
            )

    probabilities = {
        label: round(value / total_weight, 6)
        for label, value in probabilities.items()
    }

    return {
        "label": max(probabilities, key=probabilities.get),
        "score": round(weighted_score / total_weight, 6),
        "confidence": round(weighted_confidence / total_weight, 6),
        "probabilities": probabilities,
        "scored_items": len(scored),
        "token_count": total_weight,
    }


def coverage_summary(rows: list[dict]) -> dict:
    """Attach explicit corpus coverage to one sentiment aggregate."""

    aggregate = summarize_sentiment_rows(rows)
    eligible = len(rows)
    scored = aggregate["scored_items"]
    return {
        **aggregate,
        "eligible_items": eligible,
        "coverage": round(scored / eligible, 6) if eligible else 0.0,
    }


def _transcript_rows(
    ticker: str | None = None,
    transcript_id: int | None = None,
) -> list[dict]:
    engine = get_database_engine()
    metadata = MetaData()
    transcripts = Table(
        "earnings_transcripts",
        metadata,
        autoload_with=engine,
    )
    turns = Table(
        "earnings_transcript_turns",
        metadata,
        autoload_with=engine,
    )
    sentiment = Table(
        "earnings_transcript_turn_sentiment",
        metadata,
        autoload_with=engine,
    )
    model_name, model_revision = model_config()
    query = (
        select(
            transcripts.c.transcript_id,
            transcripts.c.ticker,
            transcripts.c.fiscal_period,
            transcripts.c.call_date,
            turns.c.turn_id,
            turns.c.speaker_name,
            turns.c.speaker_title,
            turns.c.speaker_role,
            sentiment.c.status,
            sentiment.c.sentiment_label,
            sentiment.c.sentiment_score,
            sentiment.c.confidence,
            sentiment.c.probabilities,
            sentiment.c.token_count,
        )
        .select_from(
            transcripts.join(
                turns,
                turns.c.transcript_id == transcripts.c.transcript_id,
            ).outerjoin(
                sentiment,
                and_(
                    sentiment.c.turn_id == turns.c.turn_id,
                    sentiment.c.model_name == model_name,
                    sentiment.c.model_revision == model_revision,
                ),
            )
        )
        .order_by(
            transcripts.c.fiscal_year,
            transcripts.c.fiscal_quarter,
            turns.c.turn_index,
        )
    )

    if ticker:
        query = query.where(transcripts.c.ticker == ticker.upper())

    if transcript_id is not None:
        query = query.where(transcripts.c.transcript_id == transcript_id)

    with engine.connect() as connection:
        return [
            dict(row)
            for row in connection.execute(query).mappings().all()
        ]


def build_transcript_summary(rows: list[dict]) -> dict | None:
    """Build management/analyst aggregates for one earnings call."""

    if not rows:
        return None

    eligible = [row for row in rows if speaker_group(row) != "operator"]
    groups = []

    for group_name in ("management", "analyst"):
        group_rows = [
            row for row in eligible if speaker_group(row) == group_name
        ]

        if group_rows:
            groups.append({
                "group": group_name,
                **coverage_summary(group_rows),
            })

    first = rows[0]
    return {
        "transcript_id": first["transcript_id"],
        "ticker": first["ticker"],
        "fiscal_period": first["fiscal_period"],
        "call_date": (
            str(first["call_date"])
            if first.get("call_date") is not None
            else None
        ),
        "overall": coverage_summary(eligible),
        "groups": groups,
    }


def get_transcript_sentiment(transcript_id: int) -> dict | None:
    """Return one call's coverage-aware sentiment summary."""

    rows = _transcript_rows(transcript_id=transcript_id)
    summary = build_transcript_summary(rows)

    if summary is None:
        return None

    model_name, model_revision = model_config()
    return {
        "model_name": model_name,
        "model_revision": model_revision,
        **summary,
    }


def get_transcript_sentiment_timeline(ticker: str) -> dict:
    """Return comparable call summaries in fiscal-period order."""

    rows = _transcript_rows(ticker=ticker)
    grouped = defaultdict(list)

    for row in rows:
        grouped[row["transcript_id"]].append(row)

    calls = [
        build_transcript_summary(group_rows)
        for group_rows in grouped.values()
    ]
    previous_call = None

    for call in calls:
        current_score = call["overall"]["score"]
        previous_score = (
            previous_call["overall"]["score"]
            if previous_call is not None
            else None
        )
        call["score_change"] = (
            round(current_score - previous_score, 6)
            if current_score is not None and previous_score is not None
            else None
        )
        previous_call = call

    model_name, model_revision = model_config()
    return {
        "ticker": ticker.upper(),
        "model_name": model_name,
        "model_revision": model_revision,
        "calls": calls,
    }


def get_filing_sentiment(accession_number: str) -> dict | None:
    """Aggregate one filing across selected narrative sections."""

    engine = get_database_engine()
    metadata = MetaData()
    chunks = Table("filing_chunks", metadata, autoload_with=engine)
    sentiment = Table(
        "filing_chunk_sentiment",
        metadata,
        autoload_with=engine,
    )
    model_name, model_revision = model_config()
    query = (
        select(
            chunks.c.accession_number,
            chunks.c.ticker,
            chunks.c.form_type,
            chunks.c.filing_date,
            chunks.c.section_key,
            chunks.c.section_title,
            sentiment.c.status,
            sentiment.c.sentiment_label,
            sentiment.c.sentiment_score,
            sentiment.c.confidence,
            sentiment.c.probabilities,
            sentiment.c.token_count,
        )
        .select_from(
            chunks.outerjoin(
                sentiment,
                and_(
                    sentiment.c.chunk_id == chunks.c.chunk_id,
                    sentiment.c.model_name == model_name,
                    sentiment.c.model_revision == model_revision,
                ),
            )
        )
        .where(
            chunks.c.accession_number == accession_number,
            chunks.c.section_key.in_(FILING_SENTIMENT_SECTION_KEYS),
        )
        .order_by(chunks.c.section_key, chunks.c.chunk_index)
    )

    with engine.connect() as connection:
        rows = [
            dict(row)
            for row in connection.execute(query).mappings().all()
        ]

    if not rows:
        return None

    section_rows = defaultdict(list)

    for row in rows:
        section_rows[(row["section_key"], row["section_title"])].append(row)

    first = rows[0]
    return {
        "accession_number": first["accession_number"],
        "ticker": first["ticker"],
        "form_type": first["form_type"],
        "filing_date": str(first["filing_date"]),
        "model_name": model_name,
        "model_revision": model_revision,
        "overall": coverage_summary(rows),
        "sections": [
            {
                "section_key": section_key,
                "section_title": section_title,
                **coverage_summary(group_rows),
            }
            for (section_key, section_title), group_rows
            in section_rows.items()
        ],
    }
