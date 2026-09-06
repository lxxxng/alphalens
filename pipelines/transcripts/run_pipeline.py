"""
AlphaLens - Earnings Call Transcript Pipeline

Purpose:
    Runs the earnings call transcript ETL workflow.

Pipeline:

    EarningsCalls.dev
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
calling EarningsCalls.dev again.
"""

import argparse

from pipelines.market_data.extractor import TICKERS

from pipelines.transcripts.earningscalls_extractor import (
    SOURCE_PROVIDER,
    extract_earnings_transcripts,
)

from pipelines.transcripts.loader import (
    get_loaded_source_urls,
    load_earnings_transcripts,
)


def run_transcript_pipeline(
    tickers: list[str] | None = None,
    max_transcripts_per_ticker: int | None = None,
    refresh_existing: bool = False,
):
    """
    Execute the earnings transcript ETL workflow.
    """

    if tickers is None:
        tickers = TICKERS

    # Existing source URLs are stable provider call IDs. Supplying them to
    # the extractor avoids spending API quota on completed transcripts.
    loaded_source_urls = (
        set()
        if refresh_existing
        else get_loaded_source_urls(SOURCE_PROVIDER)
    )

    loaded_rows = 0

    # Extract and commit one ticker at a time. If a long backfill stops,
    # completed tickers remain in PostgreSQL and are skipped on rerun.
    for ticker in tickers:
        print("\n========================================")
        print(f"TRANSCRIPT EXTRACT - {ticker}")
        print("========================================")

        transcripts = extract_earnings_transcripts(
            tickers=[ticker],
            loaded_source_urls=loaded_source_urls,
            max_transcripts_per_ticker=max_transcripts_per_ticker,
        )

        if transcripts.empty:
            print(
                f"No new transcripts returned for {ticker}."
            )
            continue

        print("\n========================================")
        print(f"POSTGRESQL LOAD - {ticker}")
        print("========================================")

        loaded_rows += load_earnings_transcripts(
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
    parser = argparse.ArgumentParser(
        description=(
            "Download EarningsCalls.dev transcripts and load PostgreSQL."
        )
    )

    parser.add_argument(
        "--tickers",
        nargs="+",
        default=None,
        help="Optional ticker subset, for example: AAPL MSFT",
    )

    parser.add_argument(
        "--max-transcripts-per-ticker",
        type=int,
        default=None,
        help="Limit downloads per ticker for a small integration test.",
    )

    parser.add_argument(
        "--refresh-existing",
        action="store_true",
        help="Download calls again even when they are already loaded.",
    )

    arguments = parser.parse_args()

    run_transcript_pipeline(
        tickers=arguments.tickers,
        max_transcripts_per_ticker=(
            arguments.max_transcripts_per_ticker
        ),
        refresh_existing=arguments.refresh_existing,
    )
