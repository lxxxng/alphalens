"""Tests for event-driven portfolio construction and performance metrics."""

import unittest

import numpy as np
import pandas as pd

from pipelines.ml.backtest import (
    calculate_performance_metrics,
    run_event_backtest,
)


def _predictions() -> pd.DataFrame:
    dates = pd.bdate_range("2026-01-02", periods=10)
    return pd.DataFrame([
        {
            "event_key": f"earnings_call:{ticker}",
            "ticker": ticker,
            "feature_as_of_date": dates[2],
            "target_trading_date": dates[7],
            "prediction": prediction,
        }
        for ticker, prediction in (
            ("A", 0.4),
            ("B", 0.2),
            ("C", -0.1),
            ("D", -0.3),
        )
    ])


def _prices() -> pd.DataFrame:
    dates = pd.bdate_range("2026-01-02", periods=10)
    returns = {
        "A": 0.02,
        "B": 0.01,
        "C": -0.005,
        "D": -0.01,
        "SPY": 0.001,
    }
    rows = []

    for ticker, daily_return in returns.items():
        for index, date in enumerate(dates):
            rows.append({
                "ticker": ticker,
                "trading_date": date,
                "adjusted_close": 100 * (1 + daily_return) ** index,
            })

    return pd.DataFrame(rows)


class EventBacktestTests(unittest.TestCase):
    def test_positions_respect_signal_start_and_expiry(self):
        predictions = _predictions()
        result = run_event_backtest(
            predictions,
            _prices(),
            top_k=1,
            min_signals=4,
            transaction_cost_bps=10,
        )
        feature_date = predictions["feature_as_of_date"].min()
        target_date = predictions["target_trading_date"].min()

        self.assertGreaterEqual(result.weights["trading_date"].min(), feature_date)
        self.assertLess(result.weights["trading_date"].max(), target_date)
        self.assertFalse((
            result.weights["trading_date"] >= result.weights["target_trading_date"]
        ).any())

    def test_cross_sectional_weights_are_balanced_and_costed(self):
        result = run_event_backtest(
            _predictions(),
            _prices(),
            top_k=1,
            min_signals=4,
            transaction_cost_bps=10,
        )
        first_date = result.weights["trading_date"].min()
        weights = result.weights.loc[
            result.weights["trading_date"] == first_date
        ].set_index("ticker")
        first_daily = result.daily_returns.loc[
            result.daily_returns["trading_date"] == first_date
        ].iloc[0]

        self.assertAlmostEqual(weights["long_short_weight"].sum(), 0.0)
        self.assertAlmostEqual(weights["long_short_weight"].abs().sum(), 1.0)
        self.assertEqual(weights.loc["A", "long_short_weight"], 0.5)
        self.assertEqual(weights.loc["D", "long_short_weight"], -0.5)
        self.assertAlmostEqual(
            first_daily["long_short_net_return"],
            first_daily["long_short_gross_return"]
            - first_daily["long_short_cost"],
        )
        self.assertGreater(first_daily["long_short_cost"], 0)

    def test_performance_metrics_capture_drawdown(self):
        metrics = calculate_performance_metrics(
            [0.10, -0.20, 0.05],
            strategy="test",
            turnover=[1.0, 0.5, 0.0],
        )

        self.assertEqual(metrics["strategy"], "test")
        self.assertLess(metrics["max_drawdown"], 0)
        self.assertAlmostEqual(metrics["average_daily_turnover"], 0.5)
        self.assertTrue(np.isfinite(metrics["annualized_volatility"]))


if __name__ == "__main__":
    unittest.main()
