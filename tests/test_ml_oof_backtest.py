"""Tests for the selected-horizon out-of-fold strategy backtest."""

import unittest

import pandas as pd

from pipelines.ml.model_benchmark import BenchmarkResult, CandidateSpec, WalkForwardFold
from pipelines.ml.oof_backtest import (
    SELECTED_MODEL,
    backtest_oof_benchmark,
    oof_backtest_to_json,
)


def _prices() -> pd.DataFrame:
    dates = pd.bdate_range("2026-01-02", "2026-02-20")
    daily_returns = {
        "A": 0.010,
        "B": 0.004,
        "C": -0.003,
        "D": -0.008,
        "SPY": 0.001,
    }
    rows = []

    for ticker, daily_return in daily_returns.items():
        for index, date in enumerate(dates):
            rows.append({
                "ticker": ticker,
                "trading_date": date,
                "adjusted_close": 100 * (1 + daily_return) ** index,
            })

    return pd.DataFrame(rows)


def _prediction_rows(model: str, family: str) -> list[dict]:
    rows = []

    for fold, feature_date, target_date in (
        ("fold_01", "2026-01-08", "2026-01-20"),
        ("fold_02", "2026-02-05", "2026-02-17"),
    ):
        for ticker, prediction, target in (
            ("A", 0.04, 0.05),
            ("B", 0.02, 0.01),
            ("C", -0.01, -0.02),
            ("D", -0.03, -0.04),
        ):
            effective_prediction = prediction if model == SELECTED_MODEL else 0.0
            rows.append({
                "fold": fold,
                "model": model,
                "family": family,
                "event_key": f"{fold}:{ticker}",
                "ticker": ticker,
                "event_source": "earnings_call",
                "event_date": pd.Timestamp(feature_date),
                "feature_as_of_date": pd.Timestamp(feature_date),
                "target_trading_date": pd.Timestamp(target_date),
                "excess_return_10d": target,
                "prediction": effective_prediction,
                "residual": target - effective_prediction,
                "direction_correct": (
                    (effective_prediction > 0) == (target > 0)
                ),
            })

    return rows


def _benchmark() -> BenchmarkResult:
    selected = CandidateSpec(
        SELECTED_MODEL,
        "elasticnet",
        {"alpha": 0.01, "l1_ratio": 0.5},
    )
    baseline = CandidateSpec("historical_mean", "baseline_mean", {})
    folds = (
        WalkForwardFold(
            "fold_01",
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.Timestamp("2026-01-01"),
            pd.Timestamp("2026-02-01"),
        ),
        WalkForwardFold(
            "fold_02",
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.Timestamp("2026-02-01"),
            pd.Timestamp("2026-03-01"),
        ),
    )
    metric_rows = []

    for fold in ("fold_01", "fold_02"):
        metric_rows.extend([
            {
                "fold": fold,
                "model": SELECTED_MODEL,
                "family": "elasticnet",
                "rows": 4,
                "mae": 0.01,
                "rmse": 0.012,
                "directional_accuracy": 1.0,
                "spearman_ic": 1.0,
            },
            {
                "fold": fold,
                "model": "historical_mean",
                "family": "baseline_mean",
                "rows": 4,
                "mae": 0.03,
                "rmse": 0.035,
                "directional_accuracy": 0.5,
                "spearman_ic": 0.0,
            },
        ])

    summary = pd.DataFrame([
        {
            "model": SELECTED_MODEL,
            "family": "elasticnet",
            "folds": 2,
            "mean_mae": 0.01,
            "pooled_spearman_ic": 1.0,
        },
        {
            "model": "historical_mean",
            "family": "baseline_mean",
            "folds": 2,
            "mean_mae": 0.03,
            "pooled_spearman_ic": 0.0,
        },
    ])
    predictions = pd.DataFrame(
        _prediction_rows(SELECTED_MODEL, "elasticnet")
        + _prediction_rows("historical_mean", "baseline_mean")
    )
    return BenchmarkResult(
        folds=folds,
        candidates=(baseline, selected),
        feature_columns=("feature",),
        target_column="excess_return_10d",
        fold_metrics=pd.DataFrame(metric_rows),
        summary=summary,
        oof_predictions=predictions,
        selection={"selected_model": SELECTED_MODEL},
    )


class OOFBacktestTests(unittest.TestCase):
    def test_backtest_preserves_oof_folds_and_position_boundaries(self):
        result = backtest_oof_benchmark(
            _benchmark(),
            _prices(),
            top_k=1,
            min_signals=4,
            transaction_cost_bps=10,
        )

        self.assertEqual(set(result.daily_returns["fold"]), {"fold_01", "fold_02"})
        self.assertFalse(result.weights["fold"].isna().any())
        self.assertFalse((
            result.weights["trading_date"] >= result.weights["target_trading_date"]
        ).any())
        self.assertEqual(
            set(result.fold_performance["strategy"]),
            {
                "elasticnet_10d_oof_long_short_net",
                "elasticnet_10d_oof_long_only_net",
                "spy_buy_and_hold",
            },
        )

    def test_report_keeps_historical_mean_as_forecast_baseline(self):
        result = backtest_oof_benchmark(
            _benchmark(),
            _prices(),
            top_k=1,
            min_signals=4,
        )
        payload = oof_backtest_to_json(result)

        self.assertEqual(
            set(result.forecast_comparison["model"]),
            {SELECTED_MODEL, "historical_mean"},
        )
        self.assertTrue((result.event_diagnostics["event_hit_rate"] == 1).all())
        self.assertEqual(result.decision["status"], "research_only")
        self.assertFalse(result.decision["production_eligible"])
        self.assertEqual(len(payload["fold_performance"]), 6)

    def test_rejects_a_non_ten_session_benchmark(self):
        benchmark = _benchmark()
        benchmark.target_column = "excess_return_30d"

        with self.assertRaisesRegex(ValueError, "10-session"):
            backtest_oof_benchmark(benchmark, _prices())


if __name__ == "__main__":
    unittest.main()
