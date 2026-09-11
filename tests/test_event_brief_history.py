"""Tests for saved event-brief persistence and cache behavior."""

import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app
from app.services.event_brief_history import (
    build_event_brief_cache_key,
    normalize_brief_tickers,
)


def _generated_brief():
    return {
        "ticker": "WMT",
        "tickers": ["WMT", "COST"],
        "event_type": "combined",
        "question": "Prepare a comparative event brief for WMT, COST.",
        "chain_version": "alphalens-event-brief-lcel-v1",
        "model_name": "gpt-5-mini",
        "brief": {
            "headline": "WMT and COST latest event comparison",
            "executive_summary": "Margins were discussed [S1].",
            "key_developments": ["WMT discussed margins [S1]."],
            "topic_signals": [],
            "market_reaction": [],
            "risks": [],
            "watch_items": [],
            "limitations": [],
        },
        "market_context": [],
        "sentiment_context": {},
        "sources": [],
    }


class EventBriefHistoryTests(unittest.TestCase):
    def test_tickers_are_normalized_and_deduplicated(self):
        result = normalize_brief_tickers({
            "ticker": "wmt",
            "tickers": ["cost", "WMT", "cost"],
        })

        self.assertEqual(result, ["COST", "WMT"])

    def test_cache_key_changes_with_event_fingerprint(self):
        request = {
            "tickers": ["WMT"],
            "event_type": "combined",
            "top_k": 6,
        }
        first = build_event_brief_cache_key(
            request,
            {"companies": [{"ticker": "WMT", "call_id": 1}]},
            "chain-v1",
        )
        second = build_event_brief_cache_key(
            request,
            {"companies": [{"ticker": "WMT", "call_id": 2}]},
            "chain-v1",
        )

        self.assertEqual(len(first), 64)
        self.assertNotEqual(first, second)

    def test_generate_reuses_cached_brief(self):
        cached = {
            **_generated_brief(),
            "brief_id": 31,
            "created_at": "2026-09-11T10:00:00+00:00",
        }

        with (
            patch(
                "app.api.research.get_latest_event_fingerprint",
                return_value={"companies": []},
            ),
            patch(
                "app.api.research.find_cached_event_brief",
                return_value=cached,
            ),
            patch("app.api.research.generate_event_brief") as generate,
        ):
            response = TestClient(app).post(
                "/api/briefs/generate",
                json={"tickers": ["WMT", "COST"]},
            )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["cached"])
        self.assertEqual(response.json()["brief_id"], 31)
        generate.assert_not_called()

    def test_refresh_generates_and_saves_new_snapshot(self):
        generated = _generated_brief()

        with (
            patch(
                "app.api.research.get_latest_event_fingerprint",
                return_value={"companies": []},
            ),
            patch("app.api.research.find_cached_event_brief") as find_cached,
            patch(
                "app.api.research.generate_event_brief",
                return_value=generated,
            ),
            patch(
                "app.api.research.save_event_brief",
                return_value=44,
            ) as save,
        ):
            response = TestClient(app).post(
                "/api/briefs/generate",
                json={
                    "tickers": ["WMT", "COST"],
                    "refresh": True,
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["cached"])
        self.assertEqual(response.json()["brief_id"], 44)
        find_cached.assert_not_called()
        save.assert_called_once()

    def test_history_endpoint_returns_summaries(self):
        summary = {
            "brief_id": 8,
            "tickers": ["WMT"],
            "event_type": "combined",
            "headline": "WMT latest event brief",
            "source_count": 6,
            "chain_version": "chain-v1",
            "model_name": "gpt-5-mini",
            "created_at": "2026-09-11T10:00:00+00:00",
        }

        with patch(
            "app.api.research.list_event_briefs",
            return_value=[summary],
        ):
            response = TestClient(app).get("/api/briefs/history")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["briefs"], [summary])

    def test_saved_brief_not_found_returns_404(self):
        with patch(
            "app.api.research.get_event_brief",
            return_value=None,
        ):
            response = TestClient(app).get("/api/briefs/history/999")

        self.assertEqual(response.status_code, 404)

    def test_saved_brief_detail_returns_original_snapshot(self):
        saved = {
            **_generated_brief(),
            "brief_id": 12,
            "fiscal_period": None,
            "form_type": None,
            "top_k": 6,
            "focus": None,
            "created_at": "2026-09-11T10:00:00+00:00",
        }

        with patch(
            "app.api.research.get_event_brief",
            return_value=saved,
        ):
            response = TestClient(app).get("/api/briefs/history/12")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["cached"])
        self.assertEqual(
            response.json()["brief"]["headline"],
            "WMT and COST latest event comparison",
        )

    def test_saved_brief_delete_returns_confirmation(self):
        with patch(
            "app.api.research.delete_event_brief",
            return_value=True,
        ):
            response = TestClient(app).delete("/api/briefs/history/12")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {"brief_id": 12, "deleted": True},
        )


if __name__ == "__main__":
    unittest.main()
