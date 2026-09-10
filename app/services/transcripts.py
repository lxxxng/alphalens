"""Read locally stored earnings transcripts for the AlphaLens UI."""

import os

from dotenv import load_dotenv
from sqlalchemy import MetaData, Table, create_engine, select


load_dotenv()


def transcript_viewer_url(transcript_id: int) -> str:
    """Return the local reader URL for one stored transcript."""

    return f"/transcripts/{int(transcript_id)}"


def get_database_engine():
    """Create a PostgreSQL engine from the local AlphaLens configuration."""

    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise ValueError("DATABASE_URL was not found in .env.")

    return create_engine(database_url, pool_pre_ping=True)


def get_transcript_detail(transcript_id: int) -> dict | None:
    """Return one transcript and its ordered speaker turns."""

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
    transcript_query = select(
        transcripts.c.transcript_id,
        transcripts.c.ticker,
        transcripts.c.fiscal_year,
        transcripts.c.fiscal_quarter,
        transcripts.c.fiscal_period,
        transcripts.c.call_date,
        transcripts.c.title,
        transcripts.c.source_provider,
        transcripts.c.content,
        transcripts.c.char_count,
        transcripts.c.turn_count,
    ).where(transcripts.c.transcript_id == transcript_id)
    turns_query = (
        select(
            turns.c.turn_index,
            turns.c.speaker_name,
            turns.c.speaker_title,
            turns.c.speaker_role,
            turns.c.content,
            turns.c.sentiment_label,
            turns.c.sentiment_score,
        )
        .where(turns.c.transcript_id == transcript_id)
        .order_by(turns.c.turn_index)
    )

    with engine.connect() as connection:
        transcript = connection.execute(
            transcript_query
        ).mappings().first()

        if transcript is None:
            return None

        speaker_turns = [
            dict(row)
            for row in connection.execute(turns_query).mappings().all()
        ]

    # Older provider records may have only canonical full text. Returning it
    # as one turn keeps the reader useful without duplicating content normally.
    if not speaker_turns:
        speaker_turns = [
            {
                "turn_index": 0,
                "speaker_name": None,
                "speaker_title": None,
                "speaker_role": None,
                "content": transcript["content"],
                "sentiment_label": None,
                "sentiment_score": None,
            }
        ]

    return {
        "transcript_id": int(transcript["transcript_id"]),
        "ticker": transcript["ticker"],
        "fiscal_year": transcript["fiscal_year"],
        "fiscal_quarter": transcript["fiscal_quarter"],
        "fiscal_period": transcript["fiscal_period"],
        "call_date": (
            str(transcript["call_date"])
            if transcript["call_date"] is not None
            else None
        ),
        "title": transcript["title"],
        "source_provider": transcript["source_provider"],
        "char_count": transcript["char_count"],
        "turn_count": len(speaker_turns),
        "turns": speaker_turns,
    }
