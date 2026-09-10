"""Tests for targeted SEC filing sentiment records."""

import unittest

from app.ml.transcript_sentiment import SentimentResult
from pipelines.sec.sentiment import (
    DEFAULT_SECTION_KEYS,
    build_sentiment_records,
)


def _result(label="positive", score=0.6):
    return SentimentResult(
        label=label,
        score=score,
        confidence=0.8,
        probabilities={
            "positive": 0.8 if label == "positive" else 0.1,
            "negative": 0.8 if label == "negative" else 0.1,
            "neutral": 0.1,
        },
        token_count=20,
        segment_count=1,
        model_name="test-model",
        model_revision="v1",
        model_commit="abc123",
    )


class _BatchScorer:
    model_name = "test-model"
    model_revision = "v1"

    def score_many(self, texts):
        return [_result() for _ in texts]


class _FallbackScorer(_BatchScorer):
    def score_many(self, texts):
        raise RuntimeError("batch failed")

    def score(self, text):
        if "bad" in text:
            raise ValueError("bad chunk")

        return _result()


class SecSentimentTests(unittest.TestCase):
    def test_default_scope_targets_narrative_sections(self):
        self.assertIn("item_7_mda", DEFAULT_SECTION_KEYS)
        self.assertIn("item_1a_risk_factors", DEFAULT_SECTION_KEYS)
        self.assertNotIn("item_8_financial_statements", DEFAULT_SECTION_KEYS)

    def test_batch_results_preserve_chunk_identity(self):
        rows = [
            {"chunk_id": 1, "accession_number": "a", "content": "good"},
            {"chunk_id": 2, "accession_number": "a", "content": "better"},
        ]

        records = build_sentiment_records(rows, _BatchScorer())

        self.assertEqual([record["chunk_id"] for record in records], [1, 2])
        self.assertTrue(all(record["status"] == "SCORED" for record in records))

    def test_failed_batch_is_retried_per_chunk(self):
        rows = [
            {"chunk_id": 1, "accession_number": "a", "content": "good"},
            {"chunk_id": 2, "accession_number": "a", "content": "bad"},
        ]

        records = build_sentiment_records(rows, _FallbackScorer())

        self.assertEqual(records[0]["status"], "SCORED")
        self.assertEqual(records[1]["status"], "FAILED")
        self.assertEqual(records[1]["error"], "bad chunk")


if __name__ == "__main__":
    unittest.main()
