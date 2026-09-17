"""Tests for SHAP reconstruction, rankings, and error diagnostics."""

import unittest

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from pipelines.ml.interpretability import analyze_xgboost_model


def _analysis_data() -> tuple[XGBRegressor, pd.DataFrame, tuple[str, ...]]:
    features = ("signal", "noise", "secondary")
    rows = 60
    rng = np.random.default_rng(42)
    values = pd.DataFrame({
        "signal": np.linspace(-1, 1, rows),
        "noise": rng.normal(0, 1, rows),
        "secondary": np.sin(np.linspace(0, 4, rows)),
    })
    target = 0.08 * values["signal"] + 0.01 * values["secondary"]
    model = XGBRegressor(
        n_estimators=30,
        max_depth=2,
        learning_rate=0.1,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=1,
    )
    model.fit(values.iloc[:40], target.iloc[:40])
    test_values = values.iloc[40:].reset_index(drop=True)
    frame = test_values.copy()
    frame["event_key"] = [f"event:{index}" for index in range(len(frame))]
    frame["ticker"] = ["WMT" if index % 2 == 0 else "NVDA" for index in range(len(frame))]
    frame["event_source"] = [
        "earnings_call" if index % 2 == 0 else "sec_filing"
        for index in range(len(frame))
    ]
    frame["event_date"] = pd.date_range("2025-01-01", periods=len(frame))
    frame["feature_as_of_date"] = pd.date_range(
        "2025-01-02",
        periods=len(frame),
    )
    frame["excess_return_30d"] = target.iloc[40:].to_numpy()
    return model, frame, features


class InterpretabilityTests(unittest.TestCase):
    def test_shap_values_reconstruct_predictions(self):
        model, frame, features = _analysis_data()

        report = analyze_xgboost_model(
            model,
            frame,
            features,
            top_local_features=2,
            top_stability_features=2,
        )

        self.assertTrue(report.additivity["passed"])
        self.assertLess(report.additivity["max_absolute_error"], 1e-5)
        self.assertEqual(len(report.local_contributions), len(frame) * 2)
        self.assertEqual(report.global_importance.iloc[0]["feature"], "signal")

    def test_reports_event_time_stability_and_error_slices(self):
        model, frame, features = _analysis_data()

        report = analyze_xgboost_model(model, frame, features)

        self.assertEqual(
            set(report.stability["group_type"]),
            {"event_source", "test_half"},
        )
        self.assertIn("overall", report.error_slices["slice_type"].tolist())
        self.assertIn("ticker", report.error_slices["slice_type"].tolist())
        self.assertEqual(len(report.errors), len(frame))


if __name__ == "__main__":
    unittest.main()
