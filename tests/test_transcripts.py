"""Tests for local transcript links, API responses, and reader routes."""

import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app
from app.rag.generator import build_source_records
from app.services.transcripts import transcript_viewer_url


TRANSCRIPT = {
    "transcript_id": 247,
    "ticker": "WMT",
    "fiscal_year": 2026,
    "fiscal_quarter": 4,
    "fiscal_period": "2026Q4",
    "call_date": "2026-08-20",
    "title": "Walmart Inc. - Earnings",
    "source_provider": "earningscalls_dev",
    "char_count": 58_395,
    "turn_count": 1,
    "turns": [
        {
            "turn_index": 0,
            "speaker_name": "John David Rainey",
            "speaker_title": "CFO",
            "speaker_role": "management",
            "content": "Operating income grew faster than sales.",
            "sentiment_label": None,
            "sentiment_score": None,
        }
    ],
}


class TranscriptTests(unittest.TestCase):
    def test_viewer_url_is_local(self):
        self.assertEqual(transcript_viewer_url(247), "/transcripts/247")

    def test_transcript_api_returns_stored_turns(self):
        with patch(
            "app.api.research.get_transcript_detail",
            return_value=TRANSCRIPT,
        ):
            response = TestClient(app).get("/api/transcripts/247")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["ticker"], "WMT")
        self.assertEqual(len(response.json()["turns"]), 1)

    def test_transcript_api_returns_not_found(self):
        with patch(
            "app.api.research.get_transcript_detail",
            return_value=None,
        ):
            response = TestClient(app).get("/api/transcripts/999999")

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"], "Transcript not found.")

    def test_transcript_page_is_served_for_local_links(self):
        response = TestClient(app).get("/transcripts/247")

        self.assertEqual(response.status_code, 200)
        self.assertIn('id="transcript-title"', response.text)

    def test_source_records_replace_authenticated_provider_url(self):
        sources = build_source_records(
            [
                {
                    "source_type": "transcript",
                    "chunk_id": 11,
                    "chunk_index": 0,
                    "transcript_id": 247,
                    "ticker": "WMT",
                    "score": 0.91,
                    "source_url": (
                        "https://earningscalls.dev/api/v1/transcripts/19930"
                    ),
                },
                {
                    "source_type": "filing",
                    "chunk_id": 12,
                    "chunk_index": 0,
                    "ticker": "WMT",
                    "score": 0.87,
                    "source_url": "https://www.sec.gov/example",
                },
            ]
        )

        self.assertEqual(sources[0]["source_url"], "/transcripts/247")
        self.assertEqual(
            sources[1]["source_url"],
            "https://www.sec.gov/example",
        )


if __name__ == "__main__":
    unittest.main()
