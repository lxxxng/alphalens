"""Financial sentiment scoring for earnings-call speaker turns."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any


DEFAULT_MODEL_NAME = "ProsusAI/finbert"
DEFAULT_MODEL_REVISION = "4556d13015211d73dccd3fdd39d39232506f3e43"
EXPECTED_LABELS = ("positive", "negative", "neutral")
FILING_SENTIMENT_SECTION_KEYS = (
    "item_7_mda",
    "part1_item2_mda",
    "item_1a_risk_factors",
    "part2_item1a_risk_factors",
    "item_7a_market_risk",
    "part1_item3_market_risk",
)


@dataclass(frozen=True)
class SentimentResult:
    """Normalized output persisted for one transcript turn."""

    label: str
    score: float
    confidence: float
    probabilities: dict[str, float]
    token_count: int
    segment_count: int
    model_name: str
    model_revision: str
    model_commit: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class FinBERTSentimentScorer:
    """Lazily load FinBERT and aggregate long-turn segment probabilities."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        model_revision: str = DEFAULT_MODEL_REVISION,
        tokenizer=None,
        classifier=None,
    ):
        self.model_name = model_name
        self.model_revision = model_revision
        self._tokenizer = tokenizer
        self._classifier = classifier

    def _load_backend(self):
        if self._tokenizer is not None and self._classifier is not None:
            return

        try:
            import torch
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
                pipeline,
            )
        except ImportError as error:
            raise RuntimeError(
                "FinBERT dependencies are missing. Install them with "
                "pip install -r requirements-ml.txt."
            ) from error

        # Large Windows hosts can make a small BERT inference slower by
        # oversubscribing CPU workers. Keep the default predictable while
        # allowing dedicated workers to tune it explicitly.
        cpu_threads = max(1, int(os.getenv("FINBERT_CPU_THREADS", "4")))
        torch.set_num_threads(cpu_threads)

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            revision=self.model_revision,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            revision=self.model_revision,
        )
        self._classifier = pipeline(
            "text-classification",
            model=model,
            tokenizer=self._tokenizer,
            device=-1,
        )

    def _token_segments(self, text: str) -> tuple[list[str], list[int]]:
        token_ids = self._tokenizer.encode(
            text,
            add_special_tokens=False,
            verbose=False,
        )

        if not token_ids:
            raise ValueError("Transcript turn produced no model tokens.")

        model_limit = getattr(self._tokenizer, "model_max_length", 512)

        # Some tokenizers use enormous sentinel values for an unknown limit.
        if not isinstance(model_limit, int) or model_limit > 512:
            model_limit = 512

        special_tokens = self._tokenizer.num_special_tokens_to_add(
            pair=False
        )
        segment_size = max(1, model_limit - special_tokens)
        id_segments = [
            token_ids[index:index + segment_size]
            for index in range(0, len(token_ids), segment_size)
        ]
        text_segments = [
            self._tokenizer.decode(
                segment,
                skip_special_tokens=True,
            )
            for segment in id_segments
        ]
        return text_segments, [len(segment) for segment in id_segments]

    @staticmethod
    def _normalize_output(output: list[dict]) -> dict[str, float]:
        probabilities = {label: 0.0 for label in EXPECTED_LABELS}

        for item in output:
            label = str(item.get("label", "")).lower()

            if label in probabilities:
                probabilities[label] = float(item.get("score", 0.0))

        total = sum(probabilities.values())

        if total <= 0:
            raise ValueError("FinBERT returned no recognized label scores.")

        return {
            label: value / total
            for label, value in probabilities.items()
        }

    def _build_result(
        self,
        outputs: list[list[dict]],
        weights: list[int],
    ) -> SentimentResult:
        weighted = {label: 0.0 for label in EXPECTED_LABELS}
        total_weight = sum(weights)

        for output, weight in zip(outputs, weights):
            probabilities = self._normalize_output(output)

            for label in EXPECTED_LABELS:
                weighted[label] += probabilities[label] * weight

        probabilities = {
            label: round(value / total_weight, 8)
            for label, value in weighted.items()
        }
        label = max(probabilities, key=probabilities.get)
        model = getattr(self._classifier, "model", None)
        config = getattr(model, "config", None)

        return SentimentResult(
            label=label,
            score=round(
                probabilities["positive"] - probabilities["negative"],
                8,
            ),
            confidence=probabilities[label],
            probabilities=probabilities,
            token_count=total_weight,
            segment_count=len(weights),
            model_name=self.model_name,
            model_revision=self.model_revision,
            model_commit=getattr(config, "_commit_hash", None),
        )

    def score_many(self, texts: list[str]) -> list[SentimentResult]:
        """Score multiple turns in one model call for efficient backfills."""

        if not texts:
            return []

        if any(not text or not text.strip() for text in texts):
            raise ValueError("Transcript turn content is empty.")

        self._load_backend()
        grouped_segments = [
            self._token_segments(text)
            for text in texts
        ]
        segments = [
            segment
            for text_segments, _ in grouped_segments
            for segment in text_segments
        ]
        outputs = self._classifier(
            segments,
            top_k=None,
            truncation=True,
            batch_size=max(
                1,
                int(os.getenv("FINBERT_INFERENCE_BATCH_SIZE", "16")),
            ),
        )

        if outputs and isinstance(outputs[0], dict):
            outputs = [outputs]

        if len(outputs) != len(segments):
            raise ValueError("FinBERT returned an unexpected segment count.")

        results = []
        offset = 0

        for text_segments, weights in grouped_segments:
            segment_count = len(text_segments)
            turn_outputs = outputs[offset:offset + segment_count]
            results.append(self._build_result(turn_outputs, weights))
            offset += segment_count

        return results

    def score(self, text: str) -> SentimentResult:
        """Score one turn, splitting text that exceeds BERT's token limit."""

        return self.score_many([text])[0]
