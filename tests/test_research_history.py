"""Tests for saved research history behavior."""

import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app
from app.services.research_history import collect_result_tickers


class ResearchHistoryTests(unittest.TestCase):
    def test_result_tickers_are_normalized_and_deduplicated(self):
        tickers = collect_result_tickers(
            request_data={"ticker": "wmt"},
            result={
                "market_context": [{"ticker": "WMT"}],
                "sources": [
                    {"ticker": "wmt"},
                    {"ticker": "SPY"},
                ],
            },
        )

        self.assertEqual(tickers, ["WMT", "SPY"])

    def test_successful_research_response_includes_saved_run_id(self):
        generated = {
            "question": "How are margins changing?",
            "answer": "Margins improved [S1].",
            "market_context": [],
            "sources": [],
        }

        with (
            patch(
                "app.api.research.answer_question",
                return_value=generated.copy(),
            ),
            patch(
                "app.api.research.save_research_run",
                return_value=42,
            ),
        ):
            response = TestClient(app).post(
                "/api/research",
                json={
                    "question": "How are margins changing?",
                    "ticker": "WMT",
                    "source_type": "transcripts",
                    "top_k": 3,
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["run_id"], 42)

    def test_history_detail_returns_not_found(self):
        with patch(
            "app.api.research.get_research_run",
            return_value=None,
        ):
            response = TestClient(app).get(
                "/api/research/history/999"
            )

        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
