"""Tests for FinBERT scoring and resumable transcript records."""

import unittest

from app.ml.transcript_sentiment import (
    FinBERTSentimentScorer,
    SentimentResult,
)
from pipelines.transcripts.sentiment import (
    build_sentiment_record,
    should_skip_turn,
)


class _Tokenizer:
    model_max_length = 6

    def encode(self, text, add_special_tokens=False, **kwargs):
        return text.split()

    def decode(self, tokens, skip_special_tokens=True):
        return " ".join(tokens)

    def num_special_tokens_to_add(self, pair=False):
        return 2


class _Config:
    _commit_hash = "model-commit"


class _Model:
    config = _Config()


class _Classifier:
    model = _Model()

    def __init__(self):
        self.call_count = 0

    def __call__(self, segments, **kwargs):
        self.call_count += 1
        outputs = []

        for segment in segments:
            if "gain" in segment:
                outputs.append(
                    [
                        {"label": "positive", "score": 0.8},
                        {"label": "negative", "score": 0.1},
                        {"label": "neutral", "score": 0.1},
                    ]
                )
            else:
                outputs.append(
                    [
                        {"label": "positive", "score": 0.1},
                        {"label": "negative", "score": 0.7},
                        {"label": "neutral", "score": 0.2},
                    ]
                )

        return outputs


class _Scorer:
    model_name = "test-model"
    model_revision = "v1"

    def score(self, text):
        return SentimentResult(
            label="positive",
            score=0.6,
            confidence=0.8,
            probabilities={
                "positive": 0.8,
                "negative": 0.2,
                "neutral": 0.0,
            },
            token_count=8,
            segment_count=1,
            model_name=self.model_name,
            model_revision=self.model_revision,
            model_commit="abc123",
        )


class _FailingScorer(_Scorer):
    def score(self, text):
        raise RuntimeError("model unavailable")


class TranscriptSentimentTests(unittest.TestCase):
    def test_long_turn_scores_are_weighted_by_segment_tokens(self):
        scorer = FinBERTSentimentScorer(
            tokenizer=_Tokenizer(),
            classifier=_Classifier(),
        )

        result = scorer.score("gain gain gain gain risk risk")

        self.assertEqual(result.label, "positive")
        self.assertEqual(result.token_count, 6)
        self.assertEqual(result.segment_count, 2)
        self.assertAlmostEqual(result.confidence, 0.56666667)
        self.assertAlmostEqual(result.score, 0.26666667)
        self.assertEqual(result.model_commit, "model-commit")

    def test_multiple_turns_share_one_classifier_call(self):
        classifier = _Classifier()
        scorer = FinBERTSentimentScorer(
            tokenizer=_Tokenizer(),
            classifier=classifier,
        )

        results = scorer.score_many([
            "gain gain",
            "risk risk",
        ])

        self.assertEqual(classifier.call_count, 1)
        self.assertEqual(
            [result.label for result in results],
            ["positive", "negative"],
        )

    def test_empty_turn_is_rejected(self):
        scorer = FinBERTSentimentScorer(
            tokenizer=_Tokenizer(),
            classifier=_Classifier(),
        )

        with self.assertRaisesRegex(ValueError, "content is empty"):
            scorer.score("  ")

    def test_operator_turns_are_skipped_by_default(self):
        row = {
            "content": "Your next question comes from...",
            "speaker_name": "Operator",
        }

        self.assertTrue(should_skip_turn(row))
        self.assertFalse(should_skip_turn(row, include_operators=True))

    def test_scored_record_preserves_reproducibility_metadata(self):
        row = {
            "turn_id": 10,
            "transcript_id": 20,
            "speaker_name": "Chief Financial Officer",
            "content": "Margins improved during the quarter.",
        }

        record = build_sentiment_record(row, _Scorer())

        self.assertEqual(record["status"], "SCORED")
        self.assertEqual(record["sentiment_label"], "positive")
        self.assertEqual(record["sentiment_score"], 0.6)
        self.assertEqual(record["model_commit"], "abc123")

    def test_model_error_becomes_resumable_failed_record(self):
        row = {
            "turn_id": 10,
            "transcript_id": 20,
            "speaker_name": "Chief Financial Officer",
            "content": "Margins improved during the quarter.",
        }

        record = build_sentiment_record(row, _FailingScorer())

        self.assertEqual(record["status"], "FAILED")
        self.assertEqual(record["error"], "model unavailable")


if __name__ == "__main__":
    unittest.main()
