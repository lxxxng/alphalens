"""Tests for event-alert detection boundaries and API behavior."""

import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app
from app.services.alerts import AlertNotFoundError, create_event_alerts


ALERT = {
    "alert_id": 11,
    "ticker": "WMT",
    "event_type": "filing",
    "source_record_id": "0000104169-26-000123",
    "event_date": "2026-09-16",
    "title": "WMT filed 10-Q",
    "message": "New SEC filing.",
    "source_url": "https://www.sec.gov/example",
    "ingestion_run_id": 8,
    "is_read": False,
    "read_at": None,
    "created_at": "2026-09-16T06:30:00+00:00",
}


class EventAlertTests(unittest.TestCase):
    def test_empty_ticker_detection_has_no_database_side_effect(self):
        self.assertEqual(
            create_event_alerts(tickers=[], since=None),
            {"filing": 0, "earnings": 0, "total": 0},
        )

    def test_list_endpoint_returns_alerts_and_unread_count(self):
        with patch(
            "app.api.alerts.list_event_alerts",
            return_value={"alerts": [ALERT], "unread_count": 1},
        ):
            response = TestClient(app).get("/api/alerts")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["unread_count"], 1)
        self.assertEqual(response.json()["alerts"][0]["ticker"], "WMT")

    def test_mark_read_endpoint_returns_updated_alert(self):
        updated = {**ALERT, "is_read": True, "read_at": ALERT["created_at"]}

        with patch(
            "app.api.alerts.mark_event_alert_read",
            return_value=updated,
        ) as mark_read:
            response = TestClient(app).patch("/api/alerts/11/read")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["is_read"])
        mark_read.assert_called_once_with(11)

    def test_missing_alert_returns_not_found(self):
        with patch(
            "app.api.alerts.mark_event_alert_read",
            side_effect=AlertNotFoundError("Alert 99 was not found."),
        ):
            response = TestClient(app).patch("/api/alerts/99/read")

        self.assertEqual(response.status_code, 404)

    def test_mark_all_endpoint_returns_updated_count(self):
        with patch(
            "app.api.alerts.mark_all_event_alerts_read",
            return_value=4,
        ):
            response = TestClient(app).post("/api/alerts/read-all")

        self.assertEqual(response.json(), {"updated_count": 4})


if __name__ == "__main__":
    unittest.main()
