"""Tests for the read-only prospective model monitor API."""

import unittest
from unittest.mock import patch

import pandas as pd
from fastapi.testclient import TestClient

from app.main import app
from app.services.prospective_monitor import get_prospective_model_status


ARTIFACT = {
    "version": "elasticnet-10d-test",
    "model_family": "elasticnet",
    "model_name": "elasticnet_a01_l50",
    "created_at": "2026-09-18T00:00:00+00:00",
    "training_cutoff": "2026-09-17",
    "training_rows": 733,
    "horizon_sessions": 10,
    "target": "excess_return_10d",
    "benchmark_ticker": "SPY",
}


class ProspectiveMonitorTests(unittest.TestCase):
    @patch(
        "app.services.prospective_monitor.load_frozen_model",
        side_effect=FileNotFoundError("freeze the model first"),
    )
    def test_missing_artifact_is_a_normal_unavailable_state(self, _load):
        result = get_prospective_model_status(engine=object())

        self.assertFalse(result["available"])
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["progress"]["required"], 40)

    @patch(
        "app.services.prospective_monitor.load_prediction_ledger",
    )
    @patch(
        "app.services.prospective_monitor.load_frozen_model",
        return_value=ARTIFACT,
    )
    def test_status_exposes_bounded_audit_rows_without_features(
        self,
        _load_artifact,
        load_ledger,
    ):
        load_ledger.return_value = pd.DataFrame([
            {
                "prediction_id": 8,
                "ticker": "WMT",
                "event_source": "earnings_call",
                "event_date": pd.Timestamp("2026-09-15"),
                "feature_as_of_date": pd.Timestamp("2026-09-16"),
                "prediction_generated_at": pd.Timestamp(
                    "2026-09-16T23:00:00Z"
                ),
                "predicted_excess_return": 0.012,
                "status": "PENDING",
                "target_trading_date": pd.NaT,
                "realized_excess_return": None,
                "outcome_error": None,
                "feature_snapshot": {"secret_feature": 3.0},
            },
            {
                "prediction_id": 7,
                "ticker": "UNH",
                "event_source": "sec_filing",
                "event_date": pd.Timestamp("2026-08-10"),
                "feature_as_of_date": pd.Timestamp("2026-08-11"),
                "prediction_generated_at": pd.Timestamp(
                    "2026-09-17T08:29:00Z"
                ),
                "predicted_excess_return": -0.0012,
                "status": "INVALID",
                "target_trading_date": pd.NaT,
                "realized_excess_return": None,
                "outcome_error": "local_prices_stale_past_expected_horizon",
                "feature_snapshot": {"secret_feature": 4.0},
            },
        ])

        result = get_prospective_model_status(engine=object(), recent_limit=1)

        self.assertTrue(result["available"])
        self.assertEqual(result["status"], "collecting")
        self.assertEqual(result["counts"]["pending"], 1)
        self.assertEqual(result["progress"]["ratio"], 0.0)
        self.assertEqual(len(result["recent_predictions"]), 1)
        self.assertNotIn("feature_snapshot", result["recent_predictions"][0])

    def test_http_endpoint_returns_monitor_payload(self):
        payload = {
            "available": True,
            "status": "collecting",
            "message": "Waiting for outcomes.",
        }

        with patch(
            "app.api.models.get_prospective_model_status",
            return_value=payload,
        ):
            response = TestClient(app).get("/api/models/prospective/status")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), payload)


if __name__ == "__main__":
    unittest.main()
