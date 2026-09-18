"""Tests for frozen prospective scoring and delayed outcome evaluation."""

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from pipelines.ml.features import model_feature_columns
from pipelines.ml.prospective import (
    build_maturation_updates,
    build_prediction_records,
    evaluate_prospective_ledger,
    freeze_prospective_model,
    load_frozen_model,
)


TICKERS = ("WMT", "NVDA", "KO", "NFLX")


def _dataset() -> pd.DataFrame:
    features = model_feature_columns()
    dates = pd.bdate_range("2025-01-02", periods=36)
    rows = []

    for index, feature_date in enumerate(dates):
        values = {
            feature: ((index + offset * 3) % 23) / 23
            for offset, feature in enumerate(features)
        }

        if index % 8 == 0:
            values[features[-1]] = np.nan

        available = index < 28
        target_date = feature_date + pd.Timedelta(days=7) if available else pd.NaT
        target_value = 0.03 * values[features[0]] - 0.01
        rows.append({
            "event_key": f"earnings_call:{index}",
            "ticker": TICKERS[index % len(TICKERS)],
            "event_source": "earnings_call",
            "event_id": str(index),
            "event_date": feature_date - pd.Timedelta(days=1),
            "feature_as_of_date": feature_date,
            "target_trading_date": target_date,
            "target_available": available,
            "target_error": None if available else "incomplete_forward_window",
            "stock_forward_return_10d": (
                target_value + 0.01 if available else np.nan
            ),
            "spy_forward_return_10d": 0.01 if available else np.nan,
            "excess_return_10d": target_value if available else np.nan,
            **values,
        })

    return pd.DataFrame(rows)


def _prices() -> pd.DataFrame:
    dates = pd.bdate_range("2025-03-03", periods=15)
    returns = {
        "WMT": 0.010,
        "NVDA": 0.005,
        "KO": -0.003,
        "NFLX": -0.008,
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


class ProspectiveModelTests(unittest.TestCase):
    def test_frozen_json_replays_and_scores_only_unlabeled_events(self):
        dataset = _dataset()
        generated_at = datetime(2025, 2, 21, tzinfo=timezone.utc)

        with tempfile.TemporaryDirectory() as directory:
            artifact = freeze_prospective_model(
                dataset,
                artifact_directory=Path(directory),
                frozen_at=generated_at,
                version="elasticnet-test",
            )
            loaded = load_frozen_model(Path(directory))
            records = build_prediction_records(
                loaded,
                dataset,
                existing_event_keys={"earnings_call:28"},
                generated_at=generated_at,
            )

        self.assertEqual(artifact["version"], "elasticnet-test")
        self.assertEqual(len(records), 7)
        self.assertNotIn(
            "earnings_call:28",
            {record["event_key"] for record in records},
        )
        self.assertTrue(all(record["status"] == "PENDING" for record in records))
        self.assertTrue(all(record["feature_snapshot"] for record in records))

    def test_maturation_rejects_post_target_predictions(self):
        dataset = _dataset().iloc[-2:].copy()
        dataset["target_available"] = True
        dataset["target_error"] = None
        dataset["target_trading_date"] = pd.Timestamp("2025-03-10")
        dataset["stock_forward_return_10d"] = [0.04, -0.01]
        dataset["spy_forward_return_10d"] = [0.01, 0.01]
        dataset["excess_return_10d"] = [0.03, -0.02]
        pending = pd.DataFrame([
            {
                "prediction_id": 1,
                "event_key": dataset.iloc[0]["event_key"],
                "feature_as_of_date": pd.Timestamp("2025-02-20"),
                "prediction_generated_at": datetime(
                    2025, 3, 1, tzinfo=timezone.utc
                ),
            },
            {
                "prediction_id": 2,
                "event_key": dataset.iloc[1]["event_key"],
                "feature_as_of_date": pd.Timestamp("2025-02-20"),
                "prediction_generated_at": datetime(
                    2025, 3, 10, tzinfo=timezone.utc
                ),
            },
        ])

        updates = build_maturation_updates(pending, dataset)

        self.assertEqual(updates[0]["status"], "MATURED")
        self.assertAlmostEqual(updates[0]["realized_excess_return"], 0.03)
        self.assertEqual(updates[1]["status"], "INVALID")
        self.assertIn("not_recorded", updates[1]["outcome_error"])

    def test_stale_local_prices_do_not_create_prospective_records(self):
        dataset = _dataset().iloc[-1:].copy()

        with tempfile.TemporaryDirectory() as directory:
            artifact = freeze_prospective_model(
                _dataset(),
                artifact_directory=Path(directory),
                frozen_at=datetime(2025, 3, 1, tzinfo=timezone.utc),
                version="elasticnet-stale-test",
            )
            records = build_prediction_records(
                artifact,
                dataset,
                generated_at=datetime(2025, 4, 1, tzinfo=timezone.utc),
            )

        self.assertEqual(records, [])

    def test_evaluation_waits_for_minimum_future_sample(self):
        feature_date = pd.Timestamp("2025-03-05")
        target_date = pd.Timestamp("2025-03-14")
        ledger = pd.DataFrame([
            {
                "event_key": f"future:{ticker}",
                "ticker": ticker,
                "feature_as_of_date": feature_date,
                "target_trading_date": target_date,
                "predicted_excess_return": prediction,
                "baseline_predicted_excess_return": 0.0,
                "realized_excess_return": actual,
                "status": "MATURED",
            }
            for ticker, prediction, actual in (
                ("WMT", 0.04, 0.05),
                ("NVDA", 0.02, 0.03),
                ("KO", -0.01, -0.02),
                ("NFLX", -0.03, -0.04),
            )
        ])

        result = evaluate_prospective_ledger(
            ledger,
            _prices(),
            model_version="elasticnet-test",
            minimum_review_events=5,
        )

        self.assertEqual(result["status"], "collecting")
        self.assertFalse(result["checks"]["minimum_events"])
        self.assertTrue(result["checks"]["mae_beats_frozen_mean"])
        self.assertFalse(result["production_eligible"])
        self.assertEqual(len(result["strategy_performance"]), 5)


if __name__ == "__main__":
    unittest.main()
