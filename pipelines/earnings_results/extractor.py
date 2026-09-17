"""Extract announced EPS results from Yahoo Finance."""

from __future__ import annotations

from collections.abc import Callable, Iterable

import pandas as pd
import yfinance as yf

from pipelines.market_data.extractor import TICKERS


SOURCE_PROVIDER = "yahoo_finance"
DEFAULT_RESULT_LIMIT = 24
RESULT_COLUMNS = (
    "ticker",
    "earnings_date",
    "earnings_timestamp",
    "eps_estimate",
    "reported_eps",
    "eps_surprise",
    "eps_surprise_pct",
    "source_provider",
    "source_url",
    "raw_payload",
)


def normalize_tickers(tickers: Iterable[str] | None) -> list[str]:
    """Normalize a ticker scope while preserving its requested order."""

    normalized = []

    for ticker in tickers or TICKERS:
        value = str(ticker or "").strip().upper()

        if value and value not in normalized:
            normalized.append(value)

    return normalized


def normalize_earnings_dates(
    ticker: str,
    raw_dates: pd.DataFrame | None,
) -> pd.DataFrame:
    """Convert one Yahoo earnings calendar into database-ready rows.

    Yahoo labels surprise as a percentage, so 9.27 means 9.27%. AlphaLens
    stores rates as decimals and converts that value to 0.0927. Rows without
    reported EPS are estimates for future events and are intentionally
    excluded from the point-in-time corpus.
    """

    if raw_dates is None or raw_dates.empty:
        return pd.DataFrame(columns=RESULT_COLUMNS)

    required = {"EPS Estimate", "Reported EPS", "Surprise(%)"}
    missing = required - set(raw_dates.columns)

    if missing:
        raise ValueError(
            "Yahoo earnings dates are missing columns: "
            + ", ".join(sorted(missing))
        )

    result = raw_dates.copy()
    source_timestamps = pd.to_datetime(result.index, errors="coerce")

    # Capture the source-local date before converting timestamps to UTC.
    result["earnings_date"] = [
        value.date() if not pd.isna(value) else None
        for value in source_timestamps
    ]
    result["earnings_timestamp"] = pd.to_datetime(
        result.index,
        errors="coerce",
        utc=True,
    )
    result["eps_estimate"] = pd.to_numeric(
        result["EPS Estimate"],
        errors="coerce",
    )
    result["reported_eps"] = pd.to_numeric(
        result["Reported EPS"],
        errors="coerce",
    )
    yahoo_surprise_pct = pd.to_numeric(
        result["Surprise(%)"],
        errors="coerce",
    )
    result["eps_surprise"] = (
        result["reported_eps"] - result["eps_estimate"]
    )
    result["eps_surprise_pct"] = yahoo_surprise_pct / 100.0
    result["ticker"] = ticker.strip().upper()
    result["source_provider"] = SOURCE_PROVIDER
    result["source_url"] = (
        f"https://finance.yahoo.com/quote/{ticker.strip().upper()}/analysis/"
    )
    result["raw_payload"] = [
        {
            "eps_estimate": _json_number(estimate),
            "reported_eps": _json_number(reported),
            "surprise_percent": _json_number(surprise),
        }
        for estimate, reported, surprise in zip(
            result["eps_estimate"],
            result["reported_eps"],
            yahoo_surprise_pct,
            strict=True,
        )
    ]
    result = result.loc[
        result["earnings_date"].notna()
        & result["earnings_timestamp"].notna()
        & result["reported_eps"].notna(),
        list(RESULT_COLUMNS),
    ]
    return result.sort_values(
        ["ticker", "earnings_timestamp"],
        kind="stable",
    ).drop_duplicates(
        ["ticker", "earnings_date", "source_provider"],
        keep="last",
    ).reset_index(drop=True)


def _json_number(value) -> float | None:
    """Convert pandas numeric values into strict JSON-safe primitives."""

    return None if pd.isna(value) else float(value)


def extract_earnings_results(
    tickers: Iterable[str] | None = None,
    *,
    limit: int = DEFAULT_RESULT_LIMIT,
    ticker_factory: Callable[[str], object] = yf.Ticker,
) -> pd.DataFrame:
    """Download and normalize announced earnings for a ticker scope."""

    if limit < 1:
        raise ValueError("limit must be at least one earnings event.")

    frames = []
    failures = []

    for ticker in normalize_tickers(tickers):
        print(f"Fetching earnings results for {ticker}", flush=True)

        try:
            raw_dates = ticker_factory(ticker).get_earnings_dates(limit=limit)
            normalized = normalize_earnings_dates(ticker, raw_dates)
            frames.append(normalized)
            print(f"  [OK] {len(normalized)} announced results", flush=True)
        except Exception as error:
            failures.append((ticker, str(error)))
            print(f"  [FAILED] {error}", flush=True)

    if not frames or all(frame.empty for frame in frames):
        detail = "; ".join(
            f"{ticker}: {message}" for ticker, message in failures
        )
        raise RuntimeError(
            "No announced earnings results were downloaded."
            + (f" {detail}" if detail else "")
        )

    if failures:
        failed_tickers = ", ".join(ticker for ticker, _ in failures)
        raise RuntimeError(
            "Earnings-result extraction was incomplete for: "
            + failed_tickers
        )

    return pd.concat(frames, ignore_index=True).sort_values(
        ["ticker", "earnings_timestamp"],
        kind="stable",
    ).reset_index(drop=True)

