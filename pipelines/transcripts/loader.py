"""
AlphaLens - Earnings Call Transcript Loader

Purpose:
    Stores normalized earnings call transcripts in PostgreSQL.

Pipeline position:

    extractor.py
        |
        v
    Pandas DataFrame
        |
        v
    loader.py
        |
        v
    PostgreSQL
        |-- earnings_transcripts
        |-- earnings_transcript_turns


Why two transcript tables?
--------------------------

earnings_transcripts:
    One row per company + fiscal quarter.

    Example:

        AAPL 2026Q3

    This table stores document-level fields such as:

        ticker
        fiscal_year
        fiscal_quarter
        full transcript text
        source provider

earnings_transcript_turns:
    One row per speaker turn inside the call.

    Example:

        Operator opening remarks
        CEO prepared remarks
        CFO prepared remarks
        Analyst question
        Management answer

    Speaker turns are useful later for:

        - separating prepared remarks from Q&A
        - analyzing management vs analyst tone
        - searching for statements by speaker
        - building cleaner transcript chunks
"""

import os

import pandas as pd

from dotenv import load_dotenv

from sqlalchemy import (
    MetaData,
    Table,
    create_engine,
    delete,
    text,
)

from sqlalchemy.dialects.postgresql import insert


# ============================================================
# Environment
# ============================================================

load_dotenv()


def get_database_engine():
    """
    Create a SQLAlchemy PostgreSQL engine.
    """

    # DATABASE_URL lives in .env so database credentials are not hard-coded.
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise ValueError(
            "DATABASE_URL was not found in .env."
        )

    # pool_pre_ping=True checks connections before reusing them.
    #
    # This helps avoid confusing errors if PostgreSQL was restarted while
    # the Python process was still alive.
    return create_engine(
        database_url,
        pool_pre_ping=True,
    )


def ensure_company_rows(
    engine,
    tickers: list[str],
):
    """
    Make sure transcript tickers exist in companies.

    SEC metadata later fills richer company_name and CIK values.
    """

    # The transcript table has a foreign key:
    #
    #     earnings_transcripts.ticker -> companies.ticker
    #
    # So a company row must exist before a transcript can reference it.
    if not tickers:
        return

    # Reflect the existing companies table from PostgreSQL.
    #
    # This keeps the Python loader aligned with the SQL migration instead
    # of redefining the table manually in two places.
    metadata = MetaData()

    companies_table = Table(
        "companies",
        metadata,
        autoload_with=engine,
    )

    # Insert minimal company rows if transcript ingestion is run before
    # the SEC metadata pipeline.
    #
    # Later, pipelines.sec.loader can fill company_name and cik.
    records = [
        {
            "ticker": ticker,
        }
        for ticker in sorted(set(tickers))
    ]

    # If the company already exists, do nothing.
    #
    # This avoids overwriting richer company metadata that may have come
    # from the SEC pipeline.
    statement = (
        insert(companies_table)
        .on_conflict_do_nothing(
            index_elements=["ticker"]
        )
    )

    with engine.begin() as connection:
        connection.execute(
            statement,
            records,
        )


def validate_transcripts(
    transcripts: pd.DataFrame,
):
    """
    Validate the extractor output before loading.
    """

    # Loading an empty DataFrame usually means:
    #
    #     missing API key
    #     provider rate limit
    #     no available transcripts
    #
    # Raise a clear error instead of silently doing nothing.
    if transcripts.empty:
        raise ValueError(
            "Cannot load transcripts because the DataFrame is empty."
        )

    # These are the fields the extractor promises to produce.
    #
    # Checking them at the boundary makes debugging much easier if the
    # provider response shape changes later.
    required_columns = [
        "ticker",
        "fiscal_year",
        "fiscal_quarter",
        "fiscal_period",
        "source_provider",
        "source_url",
        "content",
        "char_count",
        "turn_count",
        "raw_payload",
        "turns",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in transcripts.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing transcript columns: {missing_columns}"
        )


def build_transcript_record(
    row,
) -> dict:
    """
    Convert one DataFrame row into a transcript table record.
    """

    # pandas may represent missing values as NaN/NaT.
    #
    # SQLAlchemy expects Python None so it can store PostgreSQL NULL.
    call_date = row.get("call_date")

    if pd.isna(call_date):
        call_date = None

    title = row.get("title")

    if pd.isna(title):
        title = None

    # Store the full transcript in the parent table.
    #
    # This gives us one canonical text document even when speaker turns
    # are also stored separately.
    return {
        "ticker": row["ticker"],
        "fiscal_year": int(row["fiscal_year"]),
        "fiscal_quarter": int(row["fiscal_quarter"]),
        "fiscal_period": row["fiscal_period"],
        "call_date": call_date,
        "title": title,
        "source_provider": row["source_provider"],
        "source_url": row["source_url"],
        "content": row["content"],
        "char_count": int(row["char_count"]),
        "turn_count": int(row["turn_count"]),
        "raw_payload": row["raw_payload"],
        "ingest_status": "INGESTED",
        "ingest_error": None,
    }


def build_turn_records(
    transcript_id: int,
    turns: list[dict],
) -> list[dict]:
    """
    Attach transcript_id to normalized speaker turns.
    """

    # Each turn gets the transcript_id foreign key assigned here.
    #
    # The extractor cannot know this ID because PostgreSQL creates it
    # when the transcript row is inserted/upserted.
    records = []

    for turn in turns:
        records.append(
            {
                "transcript_id": transcript_id,
                "turn_index": int(turn["turn_index"]),
                "speaker_name": turn.get("speaker_name"),
                "speaker_title": turn.get("speaker_title"),
                "speaker_role": turn.get("speaker_role"),
                "content": turn["content"],
                "sentiment_label": turn.get("sentiment_label"),
                "sentiment_score": turn.get("sentiment_score"),
                "char_count": int(turn["char_count"]),
            }
        )

    return records


def load_earnings_transcripts(
    transcripts: pd.DataFrame,
) -> int:
    """
    Insert/update transcript records and replace their speaker turns.
    """

    # Validate first so database work only begins after the incoming data
    # has the expected shape.
    validate_transcripts(
        transcripts
    )

    engine = get_database_engine()

    # Make foreign-key parent rows available before inserting transcript
    # records.
    ensure_company_rows(
        engine=engine,
        tickers=transcripts["ticker"].tolist(),
    )

    # Reflect table definitions from PostgreSQL.
    #
    # This follows the same style as the market-data and SEC loaders.
    metadata = MetaData()

    transcript_table = Table(
        "earnings_transcripts",
        metadata,
        autoload_with=engine,
    )

    turn_table = Table(
        "earnings_transcript_turns",
        metadata,
        autoload_with=engine,
    )

    loaded_rows = 0

    # One transaction covers transcript rows and their turns.
    #
    # If something fails halfway through one run, PostgreSQL rolls back the
    # incomplete work instead of leaving mismatched parent/child records.
    with engine.begin() as connection:
        for _, row in transcripts.iterrows():
            transcript_record = build_transcript_record(
                row
            )

            # Insert new transcript rows.
            #
            # If the same ticker/quarter/provider already exists, the
            # ON CONFLICT clause below updates it instead.
            insert_statement = insert(
                transcript_table
            ).values(
                transcript_record
            )

            # Idempotency:
            #
            #     ticker + fiscal_year + fiscal_quarter + provider
            #
            # uniquely identifies one transcript.
            #
            # Rerunning the pipeline refreshes that row rather than
            # creating duplicates.
            upsert_statement = (
                insert_statement
                .on_conflict_do_update(
                    index_elements=[
                        "ticker",
                        "fiscal_year",
                        "fiscal_quarter",
                        "source_provider",
                    ],
                    set_={
                        "fiscal_period":
                            insert_statement.excluded.fiscal_period,
                        "call_date":
                            insert_statement.excluded.call_date,
                        "title":
                            insert_statement.excluded.title,
                        "source_url":
                            insert_statement.excluded.source_url,
                        "content":
                            insert_statement.excluded.content,
                        "char_count":
                            insert_statement.excluded.char_count,
                        "turn_count":
                            insert_statement.excluded.turn_count,
                        "raw_payload":
                            insert_statement.excluded.raw_payload,
                        "ingest_status":
                            insert_statement.excluded.ingest_status,
                        "ingest_error":
                            insert_statement.excluded.ingest_error,
                        "updated_at":
                            text("NOW()"),
                    },
                )
                .returning(
                    transcript_table.c.transcript_id
                )
            )

            # RETURNING transcript_id is important for both INSERT and
            # UPDATE paths. We need that ID to replace the child speaker
            # turns below.
            transcript_id = connection.execute(
                upsert_statement
            ).scalar_one()

            # Replace turns instead of only upserting them.
            #
            # Why?
            #
            # If the provider changes formatting and a transcript goes from
            # 80 turns to 75 turns, old turns 75-79 should disappear.
            connection.execute(
                delete(
                    turn_table
                ).where(
                    turn_table.c.transcript_id == transcript_id
                )
            )

            turn_records = build_turn_records(
                transcript_id=transcript_id,
                turns=row["turns"],
            )

            # Some provider responses may only include full transcript text
            # and no speaker-level structure. In that case the transcript is
            # still useful, and the chunker has a full-text fallback.
            if turn_records:
                connection.execute(
                    insert(turn_table),
                    turn_records,
                )

            loaded_rows += 1

            print(
                f"[OK] Loaded {row['ticker']} "
                f"{row['fiscal_period']} "
                f"| {len(turn_records)} turns"
            )

    print(
        f"\nTranscript rows loaded: {loaded_rows}"
    )

    return loaded_rows


if __name__ == "__main__":
    raise SystemExit(
        "Run python -m pipelines.transcripts.run_pipeline "
        "so extraction and loading happen together."
    )
