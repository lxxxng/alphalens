"""Tests for point-in-time event return target construction."""

import unittest

import pandas as pd

from pipelines.ml.dataset import (
    build_forward_return_targets,
    summarize_targets,
    validate_target_dataset,
)


def _event(event_date="2026-01-05"):
    return pd.DataFrame([{
        "event_key": "earnings_call:1",
        "event_source": "earnings_call",
        "event_id": "1",
        "ticker": "WMT",
        "event_date": pd.Timestamp(event_date),
        "event_subtype": "earnings_call",
        "fiscal_period": "2026Q4",
        "form_type": None,
        "report_date": pd.NaT,
    }])


def _prices(periods=40):
    dates = pd.bdate_range("2026-01-02", periods=periods)
    return pd.concat([
        pd.DataFrame({
            "ticker": "WMT",
            "trading_date": dates,
            "adjusted_close": [100.0 + index for index in range(periods)],
        }),
        pd.DataFrame({
            "ticker": "SPY",
            "trading_date": dates,
            "adjusted_close": [200.0 + index for index in range(periods)],
        }),
    ], ignore_index=True)


class ModelingTargetTests(unittest.TestCase):
    def test_builds_stock_spy_and_excess_returns(self):
        targets = build_forward_return_targets(
            _event(),
            _prices(),
            horizon=30,
        )

        row = targets.iloc[0]
        self.assertEqual(row["anchor_trading_date"], pd.Timestamp("2026-01-06"))
        self.assertEqual(row["target_trading_date"], pd.Timestamp("2026-02-17"))
        self.assertAlmostEqual(row["stock_forward_return_30d"], 30 / 102)
        self.assertAlmostEqual(row["spy_forward_return_30d"], 30 / 202)
        self.assertAlmostEqual(
            row["excess_return_30d"],
            (30 / 102) - (30 / 202),
        )
        self.assertTrue(bool(row["target_available"]))
        self.assertEqual(row["target_status"], "LABELED")

    def test_recent_event_is_retained_as_unavailable(self):
        targets = build_forward_return_targets(
            _event("2026-02-20"),
            _prices(),
            horizon=30,
        )

        row = targets.iloc[0]
        self.assertFalse(bool(row["target_available"]))
        self.assertEqual(row["target_error"], "incomplete_forward_window")

    def test_distant_first_price_is_not_treated_as_event_anchor(self):
        targets = build_forward_return_targets(
            _event("2025-12-01"),
            _prices(),
            horizon=30,
        )

        row = targets.iloc[0]
        self.assertFalse(bool(row["target_available"]))
        self.assertEqual(row["target_error"], "missing_near_event_anchor")

    def test_missing_benchmark_date_prevents_label(self):
        prices = _prices()
        prices = prices.loc[
            ~(
                (prices["ticker"] == "SPY")
                & (prices["trading_date"] == pd.Timestamp("2026-02-17"))
            )
        ]

        targets = build_forward_return_targets(
            _event(),
            prices,
            horizon=30,
        )

        self.assertEqual(targets.loc[0, "target_error"], "missing_benchmark_dates")

    def test_duplicate_market_rows_are_rejected(self):
        prices = _prices()
        prices = pd.concat([prices, prices.iloc[[0]]], ignore_index=True)

        with self.assertRaisesRegex(ValueError, "duplicate ticker/date"):
            build_forward_return_targets(_event(), prices, horizon=30)

    def test_summary_and_validation_report_clean_dataset(self):
        targets = build_forward_return_targets(
            _event(),
            _prices(),
            horizon=30,
        )

        summary = summarize_targets(targets, horizon=30)
        checks = validate_target_dataset(targets, horizon=30)

        self.assertEqual(int(summary.loc[0, "targets_available"]), 1)
        self.assertEqual(float(summary.loc[0, "coverage"]), 1.0)
        self.assertEqual(checks["labeled_rows"], 1)
        self.assertEqual(checks["anchor_not_after_event"], 0)


if __name__ == "__main__":
    unittest.main()
