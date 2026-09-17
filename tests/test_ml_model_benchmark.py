"""Tests for leakage-safe walk-forward model-family comparison."""

import unittest

import numpy as np
import pandas as pd

from pipelines.ml.features import model_feature_columns
from pipelines.ml.model_benchmark import (
    CandidateSpec,
    benchmark_to_json,
    make_walk_forward_folds,
    run_walk_forward_benchmark,
    validate_walk_forward_fold,
)


TEST_FOLDS = (
    ("2022-01-01", "2023-01-01"),
    ("2023-01-01", "2024-01-01"),
)

TEST_CANDIDATES = (
    CandidateSpec("zero_excess", "baseline_zero", {}),
    CandidateSpec("historical_mean", "baseline_mean", {}),
    CandidateSpec("ridge", "ridge", {"alpha": 10.0}),
    CandidateSpec(
        "elasticnet",
        "elasticnet",
        {"alpha": 0.001, "l1_ratio": 0.1},
    ),
    CandidateSpec(
        "extra_trees",
        "extra_trees",
        {
            "n_estimators": 10,
            "max_depth": 3,
            "min_samples_leaf": 2,
            "max_features": 0.8,
        },
    ),
    CandidateSpec(
        "xgboost",
        "xgboost",
        {
            "n_estimators": 10,
            "max_depth": 2,
            "learning_rate": 0.1,
            "min_child_weight": 1,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "reg_lambda": 1.0,
            "reg_alpha": 0.0,
        },
    ),
    CandidateSpec(
        "catboost",
        "catboost",
        {
            "iterations": 10,
            "depth": 2,
            "learning_rate": 0.1,
            "l2_leaf_reg": 3.0,
        },
    ),
)


def _benchmark_dataset() -> pd.DataFrame:
    """Create six years of monthly events with every non-topic feature."""

    features = model_feature_columns(include_topics=False)
    dates = pd.date_range("2020-01-01", periods=72, freq="MS")
    rows = []

    for index, feature_date in enumerate(dates):
        values = {
            feature: ((index * (offset + 1)) % 17) / 17
            for offset, feature in enumerate(features)
        }

        if index % 11 == 0:
            values[features[-1]] = np.nan

        rows.append({
            "event_key": f"earnings_call:{index}",
            "ticker": "WMT" if index % 2 == 0 else "NVDA",
            "event_source": "earnings_call",
            "event_id": str(index),
            "event_date": feature_date - pd.Timedelta(days=1),
            "feature_as_of_date": feature_date,
            "target_trading_date": feature_date + pd.Timedelta(days=10),
            "target_available": True,
            "excess_return_30d": (
                0.05 * values[features[0]]
                - 0.03 * values[features[1]]
                + 0.005 * (index % 3 - 1)
            ),
            **values,
        })

    return pd.DataFrame(rows)


class WalkForwardBenchmarkTests(unittest.TestCase):
    def test_folds_expand_and_exclude_future_rows(self):
        dataset = _benchmark_dataset()
        crossing = dataset.loc[
            dataset["feature_as_of_date"] == pd.Timestamp("2021-12-01")
        ].index[0]
        dataset.loc[crossing, "target_trading_date"] = pd.Timestamp(
            "2022-01-10"
        )
        folds = make_walk_forward_folds(dataset, fold_windows=TEST_FOLDS)

        self.assertEqual(len(folds), 2)
        self.assertLess(len(folds[0].train), len(folds[1].train))
        self.assertIn(
            dataset.loc[crossing, "event_key"],
            folds[0].purged["event_key"].tolist(),
        )
        self.assertTrue(
            (folds[0].validation["feature_as_of_date"] < "2023-01-01").all()
        )
        self.assertEqual(
            validate_walk_forward_fold(folds[0])["overlapping_event_keys"],
            0,
        )

    def test_compares_every_model_family_and_serializes_selection(self):
        result = run_walk_forward_benchmark(
            _benchmark_dataset(),
            fold_windows=TEST_FOLDS,
            candidates=TEST_CANDIDATES,
            include_topics=False,
        )
        payload = benchmark_to_json(result)

        self.assertEqual(
            set(result.summary["family"]),
            {candidate.family for candidate in TEST_CANDIDATES},
        )
        self.assertTrue((result.summary["folds"] == 2).all())
        self.assertEqual(
            len(result.fold_metrics),
            len(TEST_FOLDS) * len(TEST_CANDIDATES),
        )
        self.assertNotIn(
            result.selection["selected_family"],
            {"baseline_zero", "baseline_mean"},
        )
        self.assertFalse(result.selection["production_eligible"])
        self.assertEqual(payload["feature_count"], len(
            model_feature_columns(include_topics=False)
        ))
        self.assertEqual(len(payload["folds"]), 2)

    def test_duplicate_candidate_names_are_rejected(self):
        duplicate = (
            CandidateSpec("same", "ridge", {"alpha": 1.0}),
            CandidateSpec("same", "ridge", {"alpha": 10.0}),
        )

        with self.assertRaisesRegex(ValueError, "unique"):
            run_walk_forward_benchmark(
                _benchmark_dataset(),
                fold_windows=TEST_FOLDS,
                candidates=duplicate,
                include_topics=False,
            )


if __name__ == "__main__":
    unittest.main()
