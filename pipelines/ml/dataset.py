"""Point-in-time event targets for AlphaLens return modeling."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text


load_dotenv()


EVENT_COLUMNS = (
    "event_key",
    "event_source",
    "event_id",
    "ticker",
    "event_date",
    "event_subtype",
    "fiscal_period",
    "form_type",
    "report_date",
)

DEFAULT_MAX_ANCHOR_GAP_DAYS = 7


def get_database_engine(database_url: str | None = None):
    """Create the PostgreSQL engine used to construct modeling targets."""

    resolved_url = database_url or os.getenv("DATABASE_URL")

    if not resolved_url:
        raise ValueError("DATABASE_URL was not found in .env.")

    return create_engine(resolved_url, pool_pre_ping=True)


def load_events(engine) -> pd.DataFrame:
    """Load event identities using dates that were observable to investors."""

    query = text("""
        SELECT
            CONCAT('earnings_call:', transcript_id) AS event_key,
            'earnings_call' AS event_source,
            transcript_id::TEXT AS event_id,
            ticker,
            call_date AS event_date,
            'earnings_call' AS event_subtype,
            fiscal_period,
            NULL::TEXT AS form_type,
            NULL::DATE AS report_date
        FROM earnings_transcripts
        UNION ALL
        SELECT
            CONCAT('sec_filing:', accession_number) AS event_key,
            'sec_filing' AS event_source,
            accession_number AS event_id,
            ticker,
            filing_date AS event_date,
            LOWER(form_type) AS event_subtype,
            NULL::TEXT AS fiscal_period,
            form_type,
            report_date
        FROM filings
        WHERE form_type IN ('10-K', '10-Q')
        ORDER BY event_date, ticker, event_source, event_id
    """)
    return pd.read_sql_query(query, engine)


def load_adjusted_prices(engine) -> pd.DataFrame:
    """Load the adjusted closes needed for stock and benchmark labels."""

    query = text("""
        SELECT ticker, trading_date, adjusted_close
        FROM market_prices
        ORDER BY ticker, trading_date
    """)
    return pd.read_sql_query(query, engine)


def _validate_inputs(
    events: pd.DataFrame,
    prices: pd.DataFrame,
    horizon: int,
    max_anchor_gap_days: int,
) -> None:
    """Reject malformed source data before it can create plausible labels."""

    if horizon < 1:
        raise ValueError("horizon must be at least one trading day.")

    if max_anchor_gap_days < 1:
        raise ValueError("max_anchor_gap_days must be at least one day.")

    missing_event_columns = set(EVENT_COLUMNS) - set(events.columns)
    missing_price_columns = {
        "ticker",
        "trading_date",
        "adjusted_close",
    } - set(prices.columns)

    if missing_event_columns:
        raise ValueError(
            "events is missing columns: "
            + ", ".join(sorted(missing_event_columns))
        )

    if missing_price_columns:
        raise ValueError(
            "prices is missing columns: "
            + ", ".join(sorted(missing_price_columns))
        )

    if events["event_key"].duplicated().any():
        duplicates = events.loc[
            events["event_key"].duplicated(keep=False),
            "event_key",
        ].unique()
        raise ValueError(
            "events contains duplicate event keys: "
            + ", ".join(map(str, duplicates[:5]))
        )

    duplicate_prices = prices.duplicated(
        subset=["ticker", "trading_date"],
        keep=False,
    )
    if duplicate_prices.any():
        raise ValueError("prices contains duplicate ticker/date rows.")


def _price_series_by_ticker(prices: pd.DataFrame) -> dict[str, pd.Series]:
    """Normalize each ticker to one sorted adjusted-close time series."""

    normalized = prices.copy()
    normalized["ticker"] = normalized["ticker"].astype(str).str.upper()
    normalized["trading_date"] = pd.to_datetime(
        normalized["trading_date"],
        errors="coerce",
    )
    normalized["adjusted_close"] = pd.to_numeric(
        normalized["adjusted_close"],
        errors="coerce",
    )
    return {
        ticker: group.set_index("trading_date")["adjusted_close"].sort_index()
        for ticker, group in normalized.groupby("ticker", sort=False)
    }


def build_forward_return_targets(
    events: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    benchmark: str = "SPY",
    horizon: int = 30,
    max_anchor_gap_days: int = DEFAULT_MAX_ANCHOR_GAP_DAYS,
) -> pd.DataFrame:
    """Attach stock, benchmark, and excess returns to every observed event.

    The anchor is the first trading session strictly after the event date.
    This avoids assuming whether a date-only filing or call happened before,
    during, or after market hours. The return starts at that session's close,
    so the target measures subsequent drift rather than the immediate gap.
    """

    _validate_inputs(events, prices, horizon, max_anchor_gap_days)
    benchmark = benchmark.strip().upper()

    if not benchmark:
        raise ValueError("benchmark must not be empty.")

    result = events.copy()
    result["ticker"] = result["ticker"].astype(str).str.upper()
    result["event_date"] = pd.to_datetime(
        result["event_date"],
        errors="coerce",
    )
    series_by_ticker = _price_series_by_ticker(prices)
    benchmark_prices = series_by_ticker.get(benchmark)
    records = []

    for event in result.itertuples(index=False):
        record = {
            "anchor_trading_date": pd.NaT,
            "target_trading_date": pd.NaT,
            "anchor_adjusted_close": float("nan"),
            "target_adjusted_close": float("nan"),
            f"stock_forward_return_{horizon}d": float("nan"),
            "spy_anchor_adjusted_close": float("nan"),
            "spy_target_adjusted_close": float("nan"),
            f"spy_forward_return_{horizon}d": float("nan"),
            f"excess_return_{horizon}d": float("nan"),
            "target_available": False,
            "target_status": "UNAVAILABLE",
            "target_error": None,
        }

        if pd.isna(event.event_date):
            record["target_error"] = "missing_event_date"
            records.append(record)
            continue

        stock_prices = series_by_ticker.get(event.ticker)

        if stock_prices is None or stock_prices.empty:
            record["target_error"] = "missing_ticker_prices"
            records.append(record)
            continue

        anchor_position = int(
            stock_prices.index.searchsorted(event.event_date, side="right")
        )
        target_position = anchor_position + horizon

        if anchor_position >= len(stock_prices):
            record["target_error"] = "missing_post_event_anchor"
            records.append(record)
            continue

        anchor_date = stock_prices.index[anchor_position]
        record["anchor_trading_date"] = anchor_date

        anchor_gap_days = int(
            (anchor_date.normalize() - event.event_date.normalize()).days
        )
        if anchor_gap_days > max_anchor_gap_days:
            record["target_error"] = "missing_near_event_anchor"
            records.append(record)
            continue

        if target_position >= len(stock_prices):
            record["target_error"] = "incomplete_forward_window"
            records.append(record)
            continue

        target_date = stock_prices.index[target_position]
        record["target_trading_date"] = target_date

        if benchmark_prices is None:
            record["target_error"] = "missing_benchmark_prices"
            records.append(record)
            continue

        if (
            anchor_date not in benchmark_prices.index
            or target_date not in benchmark_prices.index
        ):
            record["target_error"] = "missing_benchmark_dates"
            records.append(record)
            continue

        stock_anchor = stock_prices.iloc[anchor_position]
        stock_target = stock_prices.iloc[target_position]
        benchmark_anchor = benchmark_prices.loc[anchor_date]
        benchmark_target = benchmark_prices.loc[target_date]
        price_values = (
            stock_anchor,
            stock_target,
            benchmark_anchor,
            benchmark_target,
        )

        if any(pd.isna(value) or float(value) <= 0 for value in price_values):
            record["target_error"] = "invalid_adjusted_close"
            records.append(record)
            continue

        stock_return = float(stock_target / stock_anchor - 1.0)
        benchmark_return = float(benchmark_target / benchmark_anchor - 1.0)
        record.update({
            "anchor_adjusted_close": float(stock_anchor),
            "target_adjusted_close": float(stock_target),
            f"stock_forward_return_{horizon}d": stock_return,
            "spy_anchor_adjusted_close": float(benchmark_anchor),
            "spy_target_adjusted_close": float(benchmark_target),
            f"spy_forward_return_{horizon}d": benchmark_return,
            f"excess_return_{horizon}d": stock_return - benchmark_return,
            "target_available": True,
            "target_status": "LABELED",
        })
        records.append(record)

    targets = pd.concat(
        [result.reset_index(drop=True), pd.DataFrame(records)],
        axis=1,
    )
    return targets.sort_values(
        ["event_date", "ticker", "event_source", "event_id"],
        kind="stable",
    ).reset_index(drop=True)


def summarize_targets(
    targets: pd.DataFrame,
    horizon: int = 30,
) -> pd.DataFrame:
    """Summarize label coverage and return distributions by event source."""

    stock_column = f"stock_forward_return_{horizon}d"
    benchmark_column = f"spy_forward_return_{horizon}d"
    excess_column = f"excess_return_{horizon}d"
    required = {
        "event_source",
        "event_id",
        "target_available",
        stock_column,
        benchmark_column,
        excess_column,
    }

    if not required.issubset(targets.columns):
        raise ValueError("targets is missing required target columns.")

    rows = []

    for source, group in targets.groupby("event_source", sort=True):
        labeled = group[group["target_available"]].copy()
        events = len(group)
        available = len(labeled)
        rows.append({
            "event_source": source,
            "events": events,
            "targets_available": available,
            "targets_unavailable": events - available,
            "coverage": round(available / events, 4) if events else 0.0,
            "mean_stock_return": labeled[stock_column].mean(),
            "mean_spy_return": labeled[benchmark_column].mean(),
            "mean_excess_return": labeled[excess_column].mean(),
            "median_excess_return": labeled[excess_column].median(),
            "positive_excess_rate": (
                labeled[excess_column].gt(0).mean()
                if available
                else pd.NA
            ),
        })

    return pd.DataFrame(rows)


def validate_target_dataset(
    targets: pd.DataFrame,
    horizon: int = 30,
) -> dict[str, int]:
    """Enforce chronology and completeness invariants for labeled rows."""

    excess_column = f"excess_return_{horizon}d"
    labeled = targets[targets["target_available"]].copy()
    checks = {
        "rows": len(targets),
        "labeled_rows": len(labeled),
        "duplicate_event_keys": int(targets["event_key"].duplicated().sum()),
        "anchor_not_after_event": int(
            (labeled["anchor_trading_date"] <= labeled["event_date"]).sum()
        ),
        "target_not_after_anchor": int(
            (
                labeled["target_trading_date"]
                <= labeled["anchor_trading_date"]
            ).sum()
        ),
        "missing_labeled_targets": int(labeled[excess_column].isna().sum()),
    }
    failures = {
        key: value
        for key, value in checks.items()
        if key not in {"rows", "labeled_rows"} and value
    }

    if failures:
        raise ValueError(f"Target dataset validation failed: {failures}")

    return checks


def build_modeling_targets(
    engine=None,
    horizon: int = 30,
    max_anchor_gap_days: int = DEFAULT_MAX_ANCHOR_GAP_DAYS,
) -> pd.DataFrame:
    """Build and validate the complete PostgreSQL-backed target dataset."""

    resolved_engine = engine or get_database_engine()
    targets = build_forward_return_targets(
        load_events(resolved_engine),
        load_adjusted_prices(resolved_engine),
        benchmark="SPY",
        horizon=horizon,
        max_anchor_gap_days=max_anchor_gap_days,
    )
    validate_target_dataset(targets, horizon=horizon)
    return targets


def main() -> None:
    """Build targets from PowerShell and optionally save a local CSV."""

    parser = argparse.ArgumentParser(
        description="Build point-in-time event return targets.",
    )
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument(
        "--max-anchor-gap-days",
        type=int,
        default=DEFAULT_MAX_ANCHOR_GAP_DAYS,
    )
    parser.add_argument("--output", type=Path, default=None)
    arguments = parser.parse_args()
    targets = build_modeling_targets(
        horizon=arguments.horizon,
        max_anchor_gap_days=arguments.max_anchor_gap_days,
    )
    summary = summarize_targets(targets, horizon=arguments.horizon)

    print("\nTarget coverage")
    print(summary.to_string(index=False))
    print("\nUnavailable reasons")
    print(
        targets.loc[~targets["target_available"], "target_error"]
        .value_counts(dropna=False)
        .to_string()
    )

    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        targets.to_csv(arguments.output, index=False)
        print(f"\nDataset: {arguments.output}")


if __name__ == "__main__":
    main()
