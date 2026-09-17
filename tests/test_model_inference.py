"""Tests for guarded model status, inference, and HTTP behavior."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from app.main import app
from app.services.model_inference import (
    get_model_status,
    normalize_prediction_tickers,
    predict_latest_events,
)


def _write_registry(directory: Path, status: str = "rejected") -> dict:
    """Create the smallest valid registry metadata package for status tests."""

    version = "xgboost-test-001"
    model_directory = directory / version
    model_directory.mkdir(parents=True)
    manifest = {
        "version": version,
        "model_family": "xgboost",
        "status": status,
        "created_at": "2026-09-17T00:00:00+00:00",
        "target": "excess_return_30d",
        "feature_count": 2,
        "training_window": {"first_feature_date": "2022-01-01"},
        "promotion": {
            "status": status,
            "passed": status == "champion",
            "checks": {
                "test_mae_improvement": {
                    "passed": False,
                    "model_mae": 0.063,
                    "historical_mean_mae": 0.062,
                }
            },
            "reasons": ["Test MAE did not beat the baseline."],
        },
        "backtest": [],
        "interpretability": {"top_features": []},
    }
    manifest_path = model_directory / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    (directory / "registry.json").write_text(
        json.dumps({
            "latest_version": version,
            "champion_version": version if status == "champion" else None,
            "models": [{
                "version": version,
                "status": status,
                "manifest_sha256": digest,
            }],
        }),
        encoding="utf-8",
    )
    return manifest


class _RecordingModel:
    def __init__(self):
        self.columns = None

    def predict(self, frame):
        self.columns = list(frame.columns)
        return np.asarray([0.025, -0.01][:len(frame)])


class ModelInferenceTests(unittest.TestCase):
    def test_status_exposes_rejected_model_without_calling_xgboost(self):
        with tempfile.TemporaryDirectory() as temporary:
            registry = Path(temporary)
            _write_registry(registry)

            result = get_model_status(registry)

        self.assertEqual(result["serving_status"], "blocked")
        self.assertEqual(result["latest_model"]["status"], "rejected")
        self.assertTrue(result["latest_model"]["manifest_verified"])
        self.assertIsNone(result["champion_version"])

    def test_status_fails_closed_when_manifest_was_changed(self):
        with tempfile.TemporaryDirectory() as temporary:
            registry = Path(temporary)
            manifest = _write_registry(registry)
            manifest["target"] = "tampered"
            path = registry / manifest["version"] / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")

            result = get_model_status(registry)

        self.assertEqual(result["serving_status"], "error")
        self.assertIn("checksum", result["message"])

    def test_tickers_are_normalized_deduplicated_and_bounded(self):
        self.assertEqual(
            normalize_prediction_tickers([" wmt ", "NVDA", "wmt"]),
            ["WMT", "NVDA"],
        )

        with self.assertRaisesRegex(ValueError, "At most four"):
            normalize_prediction_tickers(["A", "B", "C", "D", "E"])

    def test_research_preview_scores_latest_event_in_manifest_feature_order(self):
        model = _RecordingModel()
        manifest = {
            "version": "test-model",
            "model_family": "xgboost",
            "target": "excess_return_30d",
            "feature_count": 2,
            "feature_columns": ["feature_b", "feature_a"],
            "promotion": {
                "status": "rejected",
                "passed": False,
                "checks": {},
                "reasons": ["Research only."],
            },
        }
        loaded = SimpleNamespace(manifest=manifest, model=model)
        dataset = pd.DataFrame([
            {
                "ticker": "WMT",
                "event_key": "earnings_call:1",
                "event_source": "earnings_call",
                "event_date": pd.Timestamp("2026-01-01"),
                "feature_as_of_date": pd.Timestamp("2026-01-02"),
                "fiscal_period": "2026Q3",
                "form_type": pd.NA,
                "target_available": True,
                "excess_return_30d": 0.04,
                "feature_a": 1.0,
                "feature_b": 2.0,
            },
            {
                "ticker": "WMT",
                "event_key": "earnings_call:2",
                "event_source": "earnings_call",
                "event_date": pd.Timestamp("2026-04-01"),
                "feature_as_of_date": pd.Timestamp("2026-04-02"),
                "fiscal_period": "2026Q4",
                "form_type": pd.NA,
                "target_available": False,
                "excess_return_30d": np.nan,
                "feature_a": 3.0,
                "feature_b": 4.0,
            },
            {
                "ticker": "NVDA",
                "event_key": "sec_filing:3",
                "event_source": "sec_filing",
                "event_date": pd.Timestamp("2026-03-01"),
                "feature_as_of_date": pd.Timestamp("2026-03-02"),
                "fiscal_period": pd.NA,
                "form_type": "10-K",
                "target_available": True,
                "excess_return_30d": -0.02,
                "feature_a": 5.0,
                "feature_b": 6.0,
            },
        ])

        with patch(
            "pipelines.ml.registry.load_registered_model",
            return_value=loaded,
        ) as load, patch(
            "pipelines.ml.features.build_event_feature_dataset",
            return_value=dataset,
        ):
            result = predict_latest_events(
                ["WMT", "NVDA"],
                research_preview=True,
                registry_directory=Path("unused"),
            )

        load.assert_called_once_with(
            Path("unused"),
            version="latest",
            allow_rejected=True,
        )
        self.assertEqual(model.columns, ["feature_b", "feature_a"])
        self.assertEqual(
            [row["event_key"] for row in result["predictions"]],
            ["earnings_call:2", "sec_filing:3"],
        )
        self.assertIsNone(result["predictions"][0]["form_type"])
        self.assertFalse(result["predictions"][0]["target_available"])
        self.assertEqual(result["mode"], "research_preview")

    def test_prediction_endpoint_returns_conflict_without_champion(self):
        with patch(
            "app.api.models.predict_latest_events",
            side_effect=LookupError("Registry has no champion model."),
        ):
            response = TestClient(app).post(
                "/api/models/predict",
                json={"tickers": ["WMT"]},
            )

        self.assertEqual(response.status_code, 409)
        self.assertIn("champion", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
