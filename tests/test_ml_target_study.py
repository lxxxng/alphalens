"""Tests for multi-horizon target assembly and governed selection."""

import unittest

import numpy as np
import pandas as pd

from pipelines.ml.features import model_feature_columns
from pipelines.ml.model_benchmark import CandidateSpec
from pipelines.ml.target_study import (
    assemble_horizon_dataset,
    run_target_study,
    target_study_to_json,
)


TEST_FOLDS = (
    ("2022-01-01", "2023-01-01"),
    ("2023-01-01", "2024-01-01"),
)
TEST_CANDIDATES = (
    CandidateSpec("zero_excess", "baseline_zero", {}),
    CandidateSpec("historical_mean", "baseline_mean", {}),
    CandidateSpec("ridge", "ridge", {"alpha": 10.0}),
)


def _horizon_datasets() -> dict[int, pd.DataFrame]:
    """Create identical features with distinct five- and 30-day labels."""

    features = model_feature_columns(include_topics=False)
    dates = pd.date_range("2020-01-01", periods=72, freq="MS")
    common_rows = []

    for index, feature_date in enumerate(dates):
        values = {
            feature: ((index + offset * 2) % 19) / 19
            for offset, feature in enumerate(features)
        }

        if index % 9 == 0:
            values[features[-1]] = np.nan

        common_rows.append({
            "event_key": f"event:{index}",
            "ticker": "WMT" if index % 2 == 0 else "NVDA",
            "event_source": (
                "earnings_call" if index % 2 == 0 else "sec_filing"
            ),
            "event_id": str(index),
            "event_date": feature_date - pd.Timedelta(days=1),
            "feature_as_of_date": feature_date,
            "target_available": True,
            **values,
        })

    datasets = {}

    for horizon in (5, 30):
        frame = pd.DataFrame(common_rows)
        frame["target_trading_date"] = (
            frame["feature_as_of_date"] + pd.Timedelta(days=horizon + 2)
        )
        frame[f"excess_return_{horizon}d"] = (
            (0.04 if horizon == 5 else 0.01) * frame[features[0]]
            - 0.015 * frame[features[1]]
            + (frame.index % 3 - 1) * 0.002
        )
        datasets[horizon] = frame

    return datasets


class TargetStudyTests(unittest.TestCase):
    def test_assembles_requested_target_and_removes_stale_horizon(self):
        base = pd.DataFrame([
            {
                "event_key": "event:1",
                "event_source": "earnings_call",
                "event_id": "1",
                "ticker": "WMT",
                "event_date": pd.Timestamp("2024-01-01"),
                "feature_as_of_date": pd.Timestamp("2024-01-02"),
                "anchor_trading_date": pd.Timestamp("2024-01-02"),
                "excess_return_30d": 0.5,
                "target_available": True,
            }
        ])
        targets = pd.DataFrame([
            {
                "event_key": "event:1",
                "event_source": "earnings_call",
                "event_id": "1",
                "event_date": pd.Timestamp("2024-01-01"),
                "anchor_trading_date": pd.Timestamp("2024-01-02"),
                "target_trading_date": pd.Timestamp("2024-01-09"),
                "target_adjusted_close": 101.0,
                "stock_forward_return_5d": 0.02,
                "spy_target_adjusted_close": 102.0,
                "spy_forward_return_5d": 0.01,
                "excess_return_5d": 0.01,
                "target_available": True,
                "target_status": "LABELED",
                "target_error": None,
            }
        ])

        result = assemble_horizon_dataset(base, targets, horizon=5)

        self.assertIn("excess_return_5d", result.columns)
        self.assertNotIn("excess_return_30d", result.columns)
        self.assertEqual(result.loc[0, "excess_return_5d"], 0.01)

    def test_study_compares_horizons_and_event_sources_without_promotion(self):
        result = run_target_study(
            _horizon_datasets(),
            fold_windows=TEST_FOLDS,
            candidates=TEST_CANDIDATES,
            include_topics=False,
        )
        payload = target_study_to_json(result)

        self.assertEqual(result.horizon_summary["horizon"].tolist(), [5, 30])
        self.assertEqual(
            set(result.source_summary["event_source"]),
            {"earnings_call", "sec_filing"},
        )
        self.assertEqual(set(result.benchmarks), {5, 30})
        self.assertFalse(result.selection["production_eligible"])
        self.assertEqual(set(payload["benchmarks"]), {"5", "30"})
        self.assertEqual(
            result.benchmarks[5].target_column,
            "excess_return_5d",
        )

    def test_mismatched_event_identities_are_rejected(self):
        base = pd.DataFrame({
            "event_key": ["event:1"],
            "feature_as_of_date": [pd.Timestamp("2024-01-02")],
        })
        targets = pd.DataFrame({
            "event_key": ["event:2"],
            "event_source": ["earnings_call"],
            "event_id": ["2"],
            "event_date": [pd.Timestamp("2024-01-01")],
            "anchor_trading_date": [pd.Timestamp("2024-01-02")],
            "target_trading_date": [pd.Timestamp("2024-01-09")],
            "target_adjusted_close": [101.0],
            "stock_forward_return_5d": [0.02],
            "spy_target_adjusted_close": [102.0],
            "spy_forward_return_5d": [0.01],
            "excess_return_5d": [0.01],
            "target_available": [True],
            "target_status": ["LABELED"],
            "target_error": [None],
        })

        with self.assertRaisesRegex(ValueError, "identities"):
            assemble_horizon_dataset(base, targets, horizon=5)


if __name__ == "__main__":
    unittest.main()
