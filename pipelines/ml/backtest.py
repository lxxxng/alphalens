"""Event-driven portfolio backtesting for AlphaLens model predictions."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pipelines.ml.dataset import get_database_engine, load_adjusted_prices
from pipelines.ml.features import build_event_feature_dataset
from pipelines.ml.xgboost_model import run_xgboost_experiment


DEFAULT_TOP_K = 3
DEFAULT_MIN_SIGNALS = 4
DEFAULT_TRANSACTION_COST_BPS = 10.0


@dataclass(frozen=True)
class BacktestResult:
    """Daily strategy paths, inspectable positions, and summary statistics."""

    daily_returns: pd.DataFrame
    weights: pd.DataFrame
    summary: pd.DataFrame


def _normalize_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Validate prediction timestamps before constructing any positions."""

    required = {
        "event_key",
        "ticker",
        "feature_as_of_date",
        "target_trading_date",
        "prediction",
    }
    missing = required - set(predictions.columns)

    if missing:
        raise ValueError(
            "predictions is missing columns: " + ", ".join(sorted(missing))
        )

    normalized = predictions.copy()
    normalized["ticker"] = normalized["ticker"].astype(str).str.upper()

    for column in ("feature_as_of_date", "target_trading_date"):
        normalized[column] = pd.to_datetime(
            normalized[column],
            errors="coerce",
        ).dt.normalize()

    normalized["prediction"] = pd.to_numeric(
        normalized["prediction"],
        errors="coerce",
    )
    normalized = normalized.dropna(subset=[
        "ticker",
        "feature_as_of_date",
        "target_trading_date",
        "prediction",
    ])

    if normalized.empty:
        raise ValueError("predictions has no usable model rows.")

    if (
        normalized["target_trading_date"]
        <= normalized["feature_as_of_date"]
    ).any():
        raise ValueError("Every target date must follow its feature date.")

    if normalized.duplicated("event_key").any():
        raise ValueError("predictions contains duplicate event keys.")

    return normalized.sort_values(
        ["feature_as_of_date", "ticker", "event_key"],
        kind="stable",
    ).reset_index(drop=True)


def _price_return_panel(prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create adjusted-close and next-session return matrices."""

    required = {"ticker", "trading_date", "adjusted_close"}
    missing = required - set(prices.columns)

    if missing:
        raise ValueError(
            "prices is missing columns: " + ", ".join(sorted(missing))
        )

    normalized = prices.copy()
    normalized["ticker"] = normalized["ticker"].astype(str).str.upper()
    normalized["trading_date"] = pd.to_datetime(
        normalized["trading_date"],
        errors="coerce",
    ).dt.normalize()
    normalized["adjusted_close"] = pd.to_numeric(
        normalized["adjusted_close"],
        errors="coerce",
    )
    normalized = normalized.dropna(subset=[
        "ticker",
        "trading_date",
        "adjusted_close",
    ])

    if normalized.duplicated(["ticker", "trading_date"]).any():
        raise ValueError("prices contains duplicate ticker/date rows.")

    close = normalized.pivot(
        index="trading_date",
        columns="ticker",
        values="adjusted_close",
    ).sort_index()

    if "SPY" not in close.columns:
        raise ValueError("SPY prices are required for benchmark comparison.")

    returns = close.shift(-1) / close - 1.0
    return close, returns


def _daily_signal_weights(
    active: pd.DataFrame,
    *,
    top_k: int,
    min_signals: int,
) -> tuple[dict[str, float], dict[str, float], pd.DataFrame]:
    """Rank the latest active company signals into two portfolio variants."""

    if len(active) < min_signals:
        return {}, {}, active.iloc[0:0]

    latest = (
        active.sort_values(
            ["feature_as_of_date", "event_key"],
            kind="stable",
        )
        .drop_duplicates("ticker", keep="last")
        .sort_values(["prediction", "ticker"], kind="stable")
    )

    if len(latest) < min_signals:
        return {}, {}, latest.iloc[0:0]

    side_count = min(top_k, len(latest) // 2)

    if side_count < 1:
        return {}, {}, latest.iloc[0:0]

    short_rows = latest.head(side_count)
    long_rows = latest.tail(side_count)
    long_short = {
        **{
            row.ticker: 0.5 / side_count
            for row in long_rows.itertuples(index=False)
        },
        **{
            row.ticker: -0.5 / side_count
            for row in short_rows.itertuples(index=False)
        },
    }
    long_only = {
        row.ticker: 1.0 / side_count
        for row in long_rows.itertuples(index=False)
    }
    selected = pd.concat([
        long_rows.assign(position="long"),
        short_rows.assign(position="short"),
    ], ignore_index=True)
    return long_short, long_only, selected


def _turnover(
    current: dict[str, float],
    previous: dict[str, float],
) -> float:
    """Measure one-way traded notional when moving between daily weights."""

    tickers = set(current) | set(previous)
    return float(sum(
        abs(current.get(ticker, 0.0) - previous.get(ticker, 0.0))
        for ticker in tickers
    ))


def _weighted_return(
    weights: dict[str, float],
    returns: pd.Series,
) -> float:
    """Calculate portfolio return and fail loudly on unavailable prices."""

    missing = [
        ticker
        for ticker in weights
        if ticker not in returns.index or pd.isna(returns[ticker])
    ]

    if missing:
        raise ValueError(
            "Missing next-session returns for active positions: "
            + ", ".join(sorted(missing))
        )

    return float(sum(
        weight * float(returns[ticker])
        for ticker, weight in weights.items()
    ))


def calculate_performance_metrics(
    returns,
    *,
    strategy: str,
    turnover=None,
) -> dict:
    """Calculate return, risk, drawdown, and trading-intensity statistics."""

    series = pd.Series(returns, dtype=float).dropna()

    if series.empty:
        raise ValueError("returns must contain at least one observation.")

    wealth = (1.0 + series).cumprod()
    drawdown = wealth / wealth.cummax() - 1.0
    periods = len(series)
    total_return = float(wealth.iloc[-1] - 1.0)
    annualized_return = float(
        wealth.iloc[-1] ** (252.0 / periods) - 1.0
    )
    volatility = float(series.std(ddof=0) * math.sqrt(252))
    sharpe = (
        float(series.mean() / series.std(ddof=0) * math.sqrt(252))
        if series.std(ddof=0) > 1e-12
        else float("nan")
    )
    turnover_series = (
        pd.Series(turnover, dtype=float).dropna()
        if turnover is not None
        else pd.Series(dtype=float)
    )
    return {
        "strategy": strategy,
        "trading_days": periods,
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": float(drawdown.min()),
        "positive_day_rate": float((series > 0).mean()),
        "average_daily_turnover": (
            float(turnover_series.mean())
            if not turnover_series.empty
            else float("nan")
        ),
        "total_turnover": (
            float(turnover_series.sum())
            if not turnover_series.empty
            else float("nan")
        ),
    }


def run_event_backtest(
    predictions: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    top_k: int = DEFAULT_TOP_K,
    min_signals: int = DEFAULT_MIN_SIGNALS,
    transaction_cost_bps: float = DEFAULT_TRANSACTION_COST_BPS,
    strategy_name: str = "xgboost",
) -> BacktestResult:
    """Backtest active event forecasts using next-session market returns.

    A signal first enters after its feature date close and expires on its
    target date. Weights selected on date t therefore earn only the adjusted-
    close return from t to the next market session. Test labels never enter
    the portfolio construction path.
    """

    if top_k < 1:
        raise ValueError("top_k must be at least one.")

    if min_signals < 2:
        raise ValueError("min_signals must be at least two.")

    if transaction_cost_bps < 0:
        raise ValueError("transaction_cost_bps cannot be negative.")

    if not strategy_name.strip():
        raise ValueError("strategy_name must not be empty.")

    signals = _normalize_predictions(predictions)
    close, forward_returns = _price_return_panel(prices)
    start = signals["feature_as_of_date"].min()
    end = signals["target_trading_date"].max()
    trading_dates = close.index[(close.index >= start) & (close.index <= end)]

    if len(trading_dates) < 2:
        raise ValueError("Price history does not cover the prediction period.")

    cost_rate = transaction_cost_bps / 10_000.0
    previous_long_short = {}
    previous_long_only = {}
    daily_rows = []
    weight_rows = []

    for date in trading_dates:
        if date not in forward_returns.index or pd.isna(
            forward_returns.at[date, "SPY"]
        ):
            continue

        active = signals.loc[
            (signals["feature_as_of_date"] <= date)
            & (signals["target_trading_date"] > date)
        ]
        long_short, long_only, selected = _daily_signal_weights(
            active,
            top_k=top_k,
            min_signals=min_signals,
        )
        long_short_turnover = _turnover(
            long_short,
            previous_long_short,
        )
        long_only_turnover = _turnover(long_only, previous_long_only)
        market_returns = forward_returns.loc[date]
        long_short_gross = _weighted_return(long_short, market_returns)
        long_only_gross = _weighted_return(long_only, market_returns)
        daily_rows.append({
            "trading_date": date,
            "next_trading_date": close.index[close.index.get_loc(date) + 1],
            "active_signals": int(active["ticker"].nunique()),
            "long_positions": len(long_only),
            "short_positions": len([w for w in long_short.values() if w < 0]),
            "long_short_gross_return": long_short_gross,
            "long_short_turnover": long_short_turnover,
            "long_short_cost": long_short_turnover * cost_rate,
            "long_short_net_return": (
                long_short_gross - long_short_turnover * cost_rate
            ),
            "long_only_gross_return": long_only_gross,
            "long_only_turnover": long_only_turnover,
            "long_only_cost": long_only_turnover * cost_rate,
            "long_only_net_return": (
                long_only_gross - long_only_turnover * cost_rate
            ),
            "spy_return": float(market_returns["SPY"]),
        })

        selected_lookup = selected.set_index("ticker") if not selected.empty else None
        for ticker in sorted(set(long_short) | set(long_only)):
            signal = selected_lookup.loc[ticker]
            weight_rows.append({
                "trading_date": date,
                "ticker": ticker,
                "event_key": signal["event_key"],
                "feature_as_of_date": signal["feature_as_of_date"],
                "target_trading_date": signal["target_trading_date"],
                "prediction": float(signal["prediction"]),
                "position": signal["position"],
                "long_short_weight": long_short.get(ticker, 0.0),
                "long_only_weight": long_only.get(ticker, 0.0),
            })

        previous_long_short = long_short
        previous_long_only = long_only

    daily = pd.DataFrame(daily_rows)

    if daily.empty:
        raise ValueError("The backtest produced no daily observations.")

    summary = pd.DataFrame([
        calculate_performance_metrics(
            daily["long_short_gross_return"],
            strategy=f"{strategy_name}_long_short_gross",
            turnover=daily["long_short_turnover"],
        ),
        calculate_performance_metrics(
            daily["long_short_net_return"],
            strategy=f"{strategy_name}_long_short_net",
            turnover=daily["long_short_turnover"],
        ),
        calculate_performance_metrics(
            daily["long_only_gross_return"],
            strategy=f"{strategy_name}_long_only_gross",
            turnover=daily["long_only_turnover"],
        ),
        calculate_performance_metrics(
            daily["long_only_net_return"],
            strategy=f"{strategy_name}_long_only_net",
            turnover=daily["long_only_turnover"],
        ),
        calculate_performance_metrics(
            daily["spy_return"],
            strategy="spy_buy_and_hold",
        ),
    ])
    return BacktestResult(
        daily_returns=daily,
        weights=pd.DataFrame(weight_rows),
        summary=summary,
    )


def _json_records(frame: pd.DataFrame) -> list[dict]:
    """Serialize metric tables without pandas-specific scalar values."""

    return json.loads(frame.to_json(orient="records"))


def main() -> None:
    """Train the locked model and run its event-driven test backtest."""

    parser = argparse.ArgumentParser(
        description="Backtest AlphaLens test-period XGBoost predictions.",
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--min-signals", type=int, default=DEFAULT_MIN_SIGNALS)
    parser.add_argument(
        "--transaction-cost-bps",
        type=float,
        default=DEFAULT_TRANSACTION_COST_BPS,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/ml/backtest_summary.json"),
    )
    parser.add_argument(
        "--daily",
        type=Path,
        default=Path("data/ml/backtest_daily.csv"),
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("data/ml/backtest_weights.csv"),
    )
    arguments = parser.parse_args()
    engine = get_database_engine()
    dataset = build_event_feature_dataset(engine, horizon=30)
    experiment = run_xgboost_experiment(dataset)
    model_name = next(
        model
        for model in experiment.test_predictions["model"].unique()
        if model.startswith("xgboost_selected_")
    )
    predictions = experiment.test_predictions.loc[
        experiment.test_predictions["model"] == model_name
    ]
    result = run_event_backtest(
        predictions,
        load_adjusted_prices(engine),
        top_k=arguments.top_k,
        min_signals=arguments.min_signals,
        transaction_cost_bps=arguments.transaction_cost_bps,
    )

    print("\nBacktest assumptions")
    print({
        "model": model_name,
        "top_k": arguments.top_k,
        "min_signals": arguments.min_signals,
        "transaction_cost_bps": arguments.transaction_cost_bps,
    })
    print("\nPerformance")
    print(result.summary.to_string(index=False))

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps({
            "assumptions": {
                "model": model_name,
                "top_k": arguments.top_k,
                "min_signals": arguments.min_signals,
                "transaction_cost_bps": arguments.transaction_cost_bps,
            },
            "performance": _json_records(result.summary),
        }, indent=2),
        encoding="utf-8",
    )
    arguments.daily.parent.mkdir(parents=True, exist_ok=True)
    result.daily_returns.to_csv(arguments.daily, index=False)
    arguments.weights.parent.mkdir(parents=True, exist_ok=True)
    result.weights.to_csv(arguments.weights, index=False)
    print(f"\nSummary: {arguments.output}")
    print(f"Daily returns: {arguments.daily}")
    print(f"Weights: {arguments.weights}")


if __name__ == "__main__":
    main()
