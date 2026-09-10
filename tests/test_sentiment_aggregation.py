"""Tests for sentiment aggregation and read-only API routes."""

import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app
from app.services.sentiment import (
    build_transcript_summary,
    coverage_summary,
    get_transcript_sentiment_timeline,
    speaker_group,
    summarize_sentiment_rows,
)


def _row(
    role="executive",
    status="SCORED",
    score=0.8,
    confidence=0.8,
    token_count=3,
    probabilities=None,
):
    return {
        "transcript_id": 20,
        "ticker": "WMT",
        "fiscal_period": "2026Q4",
        "call_date": None,
        "speaker_role": role,
        "speaker_name": "Speaker",
        "speaker_title": None,
        "status": status,
        "sentiment_score": score,
        "confidence": confidence,
        "token_count": token_count,
        "probabilities": probabilities or {
            "positive": 0.8,
            "negative": 0.1,
            "neutral": 0.1,
        },
    }


class SentimentAggregationTests(unittest.TestCase):
    def test_speaker_roles_are_normalized(self):
        self.assertEqual(speaker_group(_row("executive")), "management")
        self.assertEqual(speaker_group(_row("analyst")), "analyst")
        self.assertEqual(speaker_group(_row("operator")), "operator")

    def test_aggregate_is_weighted_by_model_token_count(self):
        rows = [
            _row(score=0.8, token_count=3),
            _row(
                score=-0.4,
                confidence=0.7,
                token_count=1,
                probabilities={
                    "positive": 0.1,
                    "negative": 0.7,
                    "neutral": 0.2,
                },
            ),
        ]

        summary = summarize_sentiment_rows(rows)

        self.assertEqual(summary["label"], "positive")
        self.assertAlmostEqual(summary["score"], 0.5)
        self.assertEqual(summary["token_count"], 4)

    def test_coverage_includes_pending_items(self):
        summary = coverage_summary([
            _row(),
            _row(status=None, score=None, confidence=None),
        ])

        self.assertEqual(summary["scored_items"], 1)
        self.assertEqual(summary["eligible_items"], 2)
        self.assertEqual(summary["coverage"], 0.5)

    def test_transcript_summary_excludes_operator_from_coverage(self):
        summary = build_transcript_summary([
            _row("executive"),
            _row("analyst", status=None, score=None, confidence=None),
            _row("operator", status="SKIPPED", score=None, confidence=None),
        ])

        self.assertEqual(summary["overall"]["eligible_items"], 2)
        self.assertEqual(summary["overall"]["coverage"], 0.5)
        self.assertEqual(
            [group["group"] for group in summary["groups"]],
            ["management", "analyst"],
        )

    def test_transcript_sentiment_api(self):
        result = {"transcript_id": 20, "overall": {"coverage": 1.0}}

        with patch(
            "app.api.sentiment.get_transcript_sentiment",
            return_value=result,
        ):
            response = TestClient(app).get("/api/sentiment/transcripts/20")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), result)

    def test_transcript_timeline_api_normalizes_request(self):
        result = {"ticker": "WMT", "calls": []}

        with patch(
            "app.api.sentiment.get_transcript_sentiment_timeline",
            return_value=result,
        ) as timeline:
            response = TestClient(app).get(
                "/api/sentiment/transcripts?ticker=wmt"
            )

        self.assertEqual(response.status_code, 200)
        timeline.assert_called_once_with("wmt")

    def test_timeline_calculates_quarter_over_quarter_change(self):
        first = _row(score=0.2)
        second = _row(score=0.5)
        second.update({"transcript_id": 21, "fiscal_period": "2027Q1"})

        with patch(
            "app.services.sentiment._transcript_rows",
            return_value=[first, second],
        ):
            result = get_transcript_sentiment_timeline("WMT")

        self.assertIsNone(result["calls"][0]["score_change"])
        self.assertEqual(result["calls"][1]["score_change"], 0.3)

    def test_filing_sentiment_api_returns_not_found(self):
        with patch(
            "app.api.sentiment.get_filing_sentiment",
            return_value=None,
        ):
            response = TestClient(app).get(
                "/api/sentiment/filings/missing"
            )

        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
