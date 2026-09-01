"""
AlphaLens - Earnings Call Transcript Pipeline

Purpose:
    Runs the earnings call transcript ETL workflow.

Pipeline:

    Alpha Vantage
        |
        v
    extractor.py
        |
        v
    normalized Pandas DataFrame
        |
        v
    loader.py
        |
        v
    PostgreSQL


Why keep this separate from chunking?
-------------------------------------

This runner handles extraction + loading only.

Chunking is a separate command:

    python -m pipelines.transcripts.chunker

That mirrors the SEC pipeline:

    metadata/download/parse/extract/chunk/embed

Keeping these stages separate makes reruns easier. For example, if we
improve transcript chunking later, we can rerun only the chunker without
calling Alpha Vantage again.
"""

from pipelines.transcripts.extractor import (
    extract_earnings_transcripts,
)

from pipelines.transcripts.loader import (
    load_earnings_transcripts,
)


def run_transcript_pipeline():
    """
    Execute the earnings transcript ETL workflow.
    """

    print("\n========================================")
    print("STEP 1 - TRANSCRIPT EXTRACT")
    print("========================================")

    # STEP 1:
    #
    # Call the provider API and normalize every valid transcript into one
    # row-like dictionary.
    transcripts = extract_earnings_transcripts()

    print(
        f"\nTranscripts extracted: {len(transcripts)}"
    )

    # If nothing came back, stop before the loader.
    #
    # This avoids a misleading database transaction when the provider
    # returned no usable transcript data.
    if transcripts.empty:
        print(
            "No transcripts were returned. Nothing to load."
        )
        return 0

    print("\n========================================")
    print("STEP 2 - POSTGRESQL LOAD")
    print("========================================")

    # STEP 2:
    #
    # Store transcript rows and speaker turns in PostgreSQL using upserts.
    loaded_rows = load_earnings_transcripts(
        transcripts
    )

    print("\n========================================")
    print("TRANSCRIPT PIPELINE COMPLETE")
    print("========================================")

    print(
        f"Transcripts processed: {loaded_rows}"
    )

    return loaded_rows


if __name__ == "__main__":
    run_transcript_pipeline()
