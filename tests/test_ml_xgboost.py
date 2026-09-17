"""Tests for chronological XGBoost selection and serialization."""

import unittest

import numpy as np
import pandas as pd

from pipelines.ml.features import model_feature_columns
from pipelines.ml.xgboost_model import (
    build_xgboost_regressor,
    metrics_to_json,
    run_xgboost_experiment,
)


TEST_PARAMETERS = ({
    "max_depth": 2,
    "learning_rate": 0.1,
    "min_child_weight": 1,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "reg_lambda": 1.0,
    "reg_alpha": 0.0,
},)


def _modeling_dataset() -> pd.DataFrame:
    """Create chronological rows with every approved non-topic feature."""

    features = model_feature_columns(include_topics=False)
    dates = pd.date_range("2022-01-01", periods=48, freq="MS")
    rows = []

    for index, feature_date in enumerate(dates):
        feature_values = {
            feature: ((index + offset) % 11) / 10
            for offset, feature in enumerate(features)
        }
        if index % 7 == 0:
            feature_values[features[-1]] = np.nan

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
                0.04 * feature_values[features[0]]
                - 0.02 * feature_values[features[1]]
            ),
            **feature_values,
        })

    return pd.DataFrame(rows)


class XGBoostModelTests(unittest.TestCase):
    def test_rejects_invalid_tree_count(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            build_xgboost_regressor(TEST_PARAMETERS[0], n_estimators=0)

    def test_runs_purged_selection_and_serializes_metadata(self):
        experiment = run_xgboost_experiment(
            _modeling_dataset(),
            validation_start="2024-01-01",
            test_start="2025-01-01",
            parameter_grid=TEST_PARAMETERS,
            max_estimators=20,
            early_stopping_rounds=5,
            include_topics=False,
        )
        payload = metrics_to_json(experiment)

        self.assertEqual(len(experiment.split.train), 24)
        self.assertEqual(len(experiment.split.validation), 12)
        self.assertEqual(len(experiment.split.test), 12)
        self.assertEqual(experiment.selected_parameters, TEST_PARAMETERS[0])
        self.assertGreaterEqual(experiment.selected_boosting_rounds, 1)
        self.assertIn(
            "xgboost_selected_01",
            experiment.test_metrics["model"].tolist(),
        )
        self.assertEqual(payload["feature_count"], len(
            model_feature_columns(include_topics=False)
        ))
        self.assertEqual(payload["selected_parameters"], TEST_PARAMETERS[0])


if __name__ == "__main__":
    unittest.main()
