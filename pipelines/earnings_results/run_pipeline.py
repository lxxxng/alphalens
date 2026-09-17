"""Command-line ETL pipeline for point-in-time earnings results."""

from __future__ import annotations

import argparse

from pipelines.earnings_results.extractor import (
    DEFAULT_RESULT_LIMIT,
    extract_earnings_results,
)
from pipelines.earnings_results.loader import load_earnings_results


def run_earnings_results_pipeline(
    tickers: list[str] | None = None,
    *,
    limit: int = DEFAULT_RESULT_LIMIT,
) -> int:
    """Extract and upsert announced EPS results for the requested companies."""

    print("\n========================================")
    print("EARNINGS RESULTS - EXTRACT")
    print("========================================")
    results = extract_earnings_results(tickers, limit=limit)

    print("\n========================================")
    print("EARNINGS RESULTS - LOAD")
    print("========================================")
    loaded = load_earnings_results(results)
    print(f"Rows processed: {loaded:,}")
    return loaded


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and load historical announced EPS results.",
    )
    parser.add_argument("--tickers", nargs="+", default=None)
    parser.add_argument("--limit", type=int, default=DEFAULT_RESULT_LIMIT)
    arguments = parser.parse_args()
    run_earnings_results_pipeline(
        arguments.tickers,
        limit=arguments.limit,
    )
