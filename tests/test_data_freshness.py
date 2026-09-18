"""Tests for production data-freshness policy and HTTP diagnostics."""

import unittest
from datetime import date, datetime, timedelta, timezone
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app
from app.services.data_freshness import assess_data_freshness


NOW = datetime(2026, 9, 18, 12, 0, tzinfo=timezone.utc)
SOURCES = {
    "sec_filings": {"rows": 300, "last_activity_at": NOW - timedelta(days=20)},
    "earnings_transcripts": {
        "rows": 350,
        "last_activity_at": NOW - timedelta(days=10),
    },
    "sentiment": {"rows": 20_000, "last_activity_at": NOW - timedelta(hours=5)},
    "prospective_predictions": {"rows": 2, "last_activity_at": NOW},
}


def _run(status: str, *, hours_ago: int, run_id: int = 4) -> dict:
    completed = NOW - timedelta(hours=hours_ago)
    return {
        "run_id": run_id,
        "trigger_type": "scheduled",
        "scope": "watchlists",
        "current_stage": None,
        "status": status,
        "error": None if status == "SUCCEEDED" else "market failed",
        "started_at": completed - timedelta(minutes=12),
        "completed_at": completed,
    }


class DataFreshnessTests(unittest.TestCase):
    def test_recent_prices_and_scheduler_are_ready_despite_quiet_filings(self):
        success = _run("SUCCEEDED", hours_ago=18)
        result = assess_data_freshness(
            market_rows=[
                {"ticker": "SPY", "latest_trading_date": date(2026, 9, 16)},
                {"ticker": "WMT", "latest_trading_date": date(2026, 9, 16)},
            ],
            latest_run=success,
            latest_success=success,
            source_activity=SOURCES,
            now=NOW,
        )

        self.assertTrue(result["ready"])
        self.assertEqual(result["status"], "fresh")
        self.assertEqual(result["issues"], [])
        self.assertEqual(result["sources"]["sec_filings"]["rows"], 300)

    def test_stale_market_and_failed_latest_run_block_readiness(self):
        previous_success = _run("SUCCEEDED", hours_ago=100, run_id=3)
        result = assess_data_freshness(
            market_rows=[
                {"ticker": "SPY", "latest_trading_date": date(2026, 8, 24)},
                {"ticker": "WMT", "latest_trading_date": None},
            ],
            latest_run=_run("FAILED", hours_ago=2),
            latest_success=previous_success,
            source_activity=SOURCES,
            now=NOW,
        )

        codes = {issue["code"] for issue in result["issues"]}
        self.assertFalse(result["ready"])
        self.assertEqual(result["status"], "stale")
        self.assertEqual(
            codes,
            {"market_data_stale", "scheduler_stale", "latest_scheduler_failed"},
        )
        self.assertEqual(len(result["market"]["stale_tickers"]), 2)

    def test_freshness_endpoint_returns_service_payload(self):
        payload = {
            "status": "stale",
            "ready": False,
            "issues": [{"code": "market_data_stale"}],
        }

        with patch(
            "app.api.system.get_data_freshness",
            return_value=payload,
        ):
            response = TestClient(app).get("/api/system/data-freshness")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), payload)


if __name__ == "__main__":
    unittest.main()
