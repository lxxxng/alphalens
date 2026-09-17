"""Cost-aware backtest of the selected 10-session OOF return forecasts."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from pipelines.ml.backtest import (
    DEFAULT_MIN_SIGNALS,
    DEFAULT_TOP_K,
    DEFAULT_TRANSACTION_COST_BPS,
    calculate_performance_metrics,
    run_event_backtest,
)
from pipelines.ml.dataset import get_database_engine, load_adjusted_prices
from pipelines.ml.model_benchmark import (
    DEFAULT_CANDIDATES,
    DEFAULT_FOLD_WINDOWS,
    BenchmarkResult,
    CandidateSpec,
    run_walk_forward_benchmark,
)
from pipelines.ml.target_study import (
    build_horizon_feature_datasets,
    target_column,
)


SELECTED_HORIZON = 10
SELECTED_MODEL = "elasticnet_a01_l50"
STRATEGY_NAME = "elasticnet_10d_oof"
OOF_CANDIDATE_NAMES = {"historical_mean", SELECTED_MODEL}
OOF_CANDIDATES = tuple(
    candidate
    for candidate in DEFAULT_CANDIDATES
    if candidate.name in OOF_CANDIDATE_NAMES
)


@dataclass
class OOFBacktestResult:
    """Forecast evidence and cost-aware strategy results from the same folds."""

    benchmark: BenchmarkResult
    daily_returns: pd.DataFrame
    weights: pd.DataFrame
    performance: pd.DataFrame
    fold_performance: pd.DataFrame
    forecast_comparison: pd.DataFrame
    event_diagnostics: pd.DataFrame
    decision: dict


def _assign_folds(
    daily_returns: pd.DataFrame,
    benchmark: BenchmarkResult,
) -> pd.DataFrame:
    """Attach every portfolio day to its original OOF validation window."""

    daily = daily_returns.copy()
    daily["fold"] = pd.Series(pd.NA, index=daily.index, dtype="string")

    for fold in benchmark.folds:
        mask = (
            (daily["trading_date"] >= fold.validation_start)
            & (daily["trading_date"] < fold.validation_end)
        )
        daily.loc[mask, "fold"] = fold.name

    if daily["fold"].isna().any():
        missing = daily.loc[daily["fold"].isna(), "trading_date"]
        raise ValueError(
            "Portfolio dates fall outside the OOF folds: "
            + ", ".join(missing.dt.date.astype(str).head(5))
        )

    return daily


def _fold_performance(daily: pd.DataFrame) -> pd.DataFrame:
    """Measure net strategy stability rather than relying on one pooled path."""

    series_by_strategy = {
        f"{STRATEGY_NAME}_long_short_net": (
            "long_short_net_return",
            "long_short_turnover",
        ),
        f"{STRATEGY_NAME}_long_only_net": (
            "long_only_net_return",
            "long_only_turnover",
        ),
        "spy_buy_and_hold": ("spy_return", None),
    }
    rows = []

    for fold, group in daily.groupby("fold", sort=False):
        for strategy, (return_column, turnover_column) in (
            series_by_strategy.items()
        ):
            metrics = calculate_performance_metrics(
                group[return_column],
                strategy=strategy,
                turnover=(
                    group[turnover_column]
                    if turnover_column is not None
                    else None
                ),
            )
            metrics.update({
                "fold": fold,
                "active_days": int((group["active_signals"] > 0).sum()),
                "invested_days": int((group["long_positions"] > 0).sum()),
            })
            rows.append(metrics)

    return pd.DataFrame(rows)[[
        "fold",
        "strategy",
        "trading_days",
        "active_days",
        "invested_days",
        "total_return",
        "annualized_return",
        "annualized_volatility",
        "sharpe",
        "max_drawdown",
        "positive_day_rate",
        "average_daily_turnover",
        "total_turnover",
    ]]


def _forecast_comparison(
    benchmark: BenchmarkResult,
    model_name: str,
) -> pd.DataFrame:
    """Keep the constant historical mean as a forecast, not a fake rank trade."""

    comparison = benchmark.summary.loc[
        benchmark.summary["model"].isin(["historical_mean", model_name])
    ].copy()

    if len(comparison) != 2:
        raise ValueError(
            "OOF benchmark requires the selected model and historical_mean."
        )

    return comparison.sort_values("mean_mae", kind="stable").reset_index(
        drop=True
    )


def _event_diagnostics(
    benchmark: BenchmarkResult,
    predictions: pd.DataFrame,
    model_name: str,
) -> pd.DataFrame:
    """Expose event hit rate and rank quality for every validation fold."""

    selected_metrics = benchmark.fold_metrics.loc[
        benchmark.fold_metrics["model"] == model_name,
        ["fold", "rows", "mae", "rmse", "directional_accuracy", "spearman_ic"],
    ].copy()
    observed_hits = (
        predictions.groupby("fold", sort=False)["direction_correct"]
        .mean()
        .rename("event_hit_rate")
        .reset_index()
    )
    diagnostics = selected_metrics.merge(
        observed_hits,
        on="fold",
        validate="one_to_one",
    )

    if not (
        diagnostics["directional_accuracy"]
        .sub(diagnostics["event_hit_rate"])
        .abs()
        .lt(1e-12)
        .all()
    ):
        raise ValueError("Fold hit rates do not match OOF prediction rows.")

    return diagnostics


def backtest_oof_benchmark(
    benchmark: BenchmarkResult,
    prices: pd.DataFrame,
    *,
    model_name: str = SELECTED_MODEL,
    top_k: int = DEFAULT_TOP_K,
    min_signals: int = DEFAULT_MIN_SIGNALS,
    transaction_cost_bps: float = DEFAULT_TRANSACTION_COST_BPS,
) -> OOFBacktestResult:
    """Backtest one model using only predictions made out of sample by fold."""

    if benchmark.target_column != target_column(SELECTED_HORIZON):
        raise ValueError("OOF strategy backtest requires the 10-session target.")

    predictions = benchmark.oof_predictions.loc[
        benchmark.oof_predictions["model"] == model_name
    ].copy()

    if predictions.empty:
        raise ValueError(f"OOF predictions do not contain model: {model_name}")

    expected_folds = {fold.name for fold in benchmark.folds}
    observed_folds = set(predictions["fold"])

    if observed_folds != expected_folds:
        raise ValueError("Selected OOF predictions do not cover every fold.")

    portfolio = run_event_backtest(
        predictions,
        prices,
        top_k=top_k,
        min_signals=min_signals,
        transaction_cost_bps=transaction_cost_bps,
        strategy_name=STRATEGY_NAME,
    )
    daily = _assign_folds(portfolio.daily_returns, benchmark)
    weights = portfolio.weights.merge(
        predictions[["event_key", "fold"]],
        on="event_key",
        how="left",
        validate="many_to_one",
    )
    forecast_comparison = _forecast_comparison(benchmark, model_name)
    diagnostics = _event_diagnostics(benchmark, predictions, model_name)
    long_short_net = portfolio.summary.loc[
        portfolio.summary["strategy"]
        == f"{STRATEGY_NAME}_long_short_net"
    ].iloc[0]
    decision = {
        "status": "research_only",
        "production_eligible": False,
        "selected_horizon": SELECTED_HORIZON,
        "selected_model": model_name,
        "oof_event_rows": len(predictions),
        "event_hit_rate": float(predictions["direction_correct"].mean()),
        "net_long_short_sharpe": (
            None
            if pd.isna(long_short_net["sharpe"])
            else float(long_short_net["sharpe"])
        ),
        "reason": (
            "The horizon and model were selected after inspecting these "
            "historical folds. This backtest measures economic plausibility "
            "but cannot replace a genuinely future holdout."
        ),
        "historical_mean_strategy_note": (
            "Historical-mean predictions are constant within each fold, so "
            "they are retained as a forecast baseline and are not converted "
            "into arbitrary top/bottom ticker ranks."
        ),
    }
    return OOFBacktestResult(
        benchmark=benchmark,
        daily_returns=daily,
        weights=weights,
        performance=portfolio.summary,
        fold_performance=_fold_performance(daily),
        forecast_comparison=forecast_comparison,
        event_diagnostics=diagnostics,
        decision=decision,
    )


def run_oof_strategy_backtest(
    dataset: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    fold_windows: tuple[tuple[str, str], ...] = DEFAULT_FOLD_WINDOWS,
    candidates: tuple[CandidateSpec, ...] = OOF_CANDIDATES,
    include_topics: bool = True,
    top_k: int = DEFAULT_TOP_K,
    min_signals: int = DEFAULT_MIN_SIGNALS,
    transaction_cost_bps: float = DEFAULT_TRANSACTION_COST_BPS,
) -> OOFBacktestResult:
    """Refit the locked candidate in each fold, then backtest its OOF rows."""

    benchmark = run_walk_forward_benchmark(
        dataset,
        fold_windows=fold_windows,
        candidates=candidates,
        include_topics=include_topics,
        target_column=target_column(SELECTED_HORIZON),
    )
    return backtest_oof_benchmark(
        benchmark,
        prices,
        top_k=top_k,
        min_signals=min_signals,
        transaction_cost_bps=transaction_cost_bps,
    )


def _json_records(frame: pd.DataFrame) -> list[dict]:
    """Serialize report tables without pandas-specific scalar values."""

    return json.loads(frame.to_json(orient="records", date_format="iso"))


def oof_backtest_to_json(result: OOFBacktestResult) -> dict:
    """Build the persisted audit report for the strategy experiment."""

    return {
        "decision": result.decision,
        "assumptions": {
            "horizon_sessions": SELECTED_HORIZON,
            "model": SELECTED_MODEL,
            "signal_source": "purged_walk_forward_oof_predictions",
            "top_k": DEFAULT_TOP_K,
            "min_signals": DEFAULT_MIN_SIGNALS,
            "transaction_cost_bps_one_way": DEFAULT_TRANSACTION_COST_BPS,
        },
        "forecast_comparison": _json_records(result.forecast_comparison),
        "performance": _json_records(result.performance),
        "fold_performance": _json_records(result.fold_performance),
        "event_diagnostics": _json_records(result.event_diagnostics),
    }


def main() -> None:
    """Reproduce the selected 10-session OOF strategy experiment."""

    parser = argparse.ArgumentParser(
        description="Backtest the selected 10-session OOF Elastic Net.",
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
        default=Path("data/ml/oof_backtest_summary.json"),
    )
    parser.add_argument(
        "--daily",
        type=Path,
        default=Path("data/ml/oof_backtest_daily.csv"),
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("data/ml/oof_backtest_weights.csv"),
    )
    arguments = parser.parse_args()
    engine = get_database_engine()
    dataset = build_horizon_feature_datasets(
        engine,
        horizons=(SELECTED_HORIZON,),
    )[SELECTED_HORIZON]
    result = run_oof_strategy_backtest(
        dataset,
        load_adjusted_prices(engine),
        top_k=arguments.top_k,
        min_signals=arguments.min_signals,
        transaction_cost_bps=arguments.transaction_cost_bps,
    )

    print("\nOOF forecast comparison")
    print(result.forecast_comparison.to_string(index=False))
    print("\nPooled strategy performance")
    print(result.performance.to_string(index=False))
    print("\nNet performance by fold")
    print(result.fold_performance.to_string(index=False))
    print("\nDecision")
    print(json.dumps(result.decision, indent=2))

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    report = oof_backtest_to_json(result)
    report["assumptions"].update({
        "top_k": arguments.top_k,
        "min_signals": arguments.min_signals,
        "transaction_cost_bps_one_way": arguments.transaction_cost_bps,
    })
    arguments.output.write_text(
        json.dumps(report, indent=2),
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
