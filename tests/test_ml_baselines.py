"""Tests for chronological splits and baseline evaluation."""

import unittest

import numpy as np
import pandas as pd

from pipelines.ml.baselines import (
    build_ridge_pipeline,
    evaluate_predictions,
    purged_time_split,
    validate_purged_split,
)


def _split_dataset():
    rows = [
        ("train", "2024-04-01", "2024-05-15"),
        ("purge_train", "2024-06-15", "2024-07-20"),
        ("validation", "2024-08-01", "2024-09-15"),
        ("purge_validation", "2025-06-15", "2025-07-20"),
        ("test", "2025-08-01", "2025-09-15"),
    ]
    return pd.DataFrame([
        {
            "event_key": name,
            "ticker": "WMT",
            "event_source": "earnings_call",
            "event_id": name,
            "feature_as_of_date": pd.Timestamp(feature_date),
            "target_trading_date": pd.Timestamp(target_date),
            "target_available": True,
            "excess_return_30d": 0.01,
        }
        for name, feature_date, target_date in rows
    ])


class PurgedSplitTests(unittest.TestCase):
    def test_cross_boundary_targets_are_purged(self):
        split = purged_time_split(
            _split_dataset(),
            validation_start="2024-07-01",
            test_start="2025-07-01",
        )

        self.assertEqual(split.train["event_key"].tolist(), ["train"])
        self.assertEqual(
            split.validation["event_key"].tolist(),
            ["validation"],
        )
        self.assertEqual(split.test["event_key"].tolist(), ["test"])
        self.assertEqual(
            set(split.purged["event_key"]),
            {"purge_train", "purge_validation"},
        )
        self.assertEqual(
            validate_purged_split(split)["overlapping_event_keys"],
            0,
        )

    def test_invalid_boundary_order_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "before"):
            purged_time_split(
                _split_dataset(),
                validation_start="2025-07-01",
                test_start="2024-07-01",
            )


class BaselineMetricTests(unittest.TestCase):
    def test_prediction_metrics_are_calculated(self):
        metrics = evaluate_predictions(
            [0.1, -0.1, 0.2, -0.2],
            [0.08, -0.05, 0.1, -0.1],
            model_name="test",
            split_name="validation",
        )

        self.assertEqual(metrics["rows"], 4)
        self.assertEqual(metrics["directional_accuracy"], 1.0)
        self.assertGreater(metrics["pearson"], 0.9)

    def test_ridge_pipeline_handles_missing_values(self):
        features = pd.DataFrame({
            "a": [1.0, 2.0, np.nan, 4.0],
            "b": [np.nan, 1.0, 0.0, 2.0],
        })
        target = pd.Series([0.1, 0.2, -0.1, 0.3])
        model = build_ridge_pipeline(alpha=1.0)

        model.fit(features, target)
        predictions = model.predict(features)

        self.assertEqual(len(predictions), 4)
        self.assertTrue(np.isfinite(predictions).all())


if __name__ == "__main__":
    unittest.main()
