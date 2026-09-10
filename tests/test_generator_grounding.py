"""Tests for deterministic answer finalization after model generation."""

import unittest
from unittest.mock import patch

from app.rag.generator import (
    collect_evidence,
    finalize_grounded_answer,
    resolve_question_tickers,
)
from app.rag.company_resolver import resolve_tickers
from app.rag.retrievers.router import should_prefer_latest_transcript


class GeneratorGroundingTests(unittest.TestCase):
    def test_latest_plural_calls_prefer_each_company_latest_period(self):
        self.assertTrue(
            should_prefer_latest_transcript(
                "Compare Walmart and Costco on their latest earnings calls."
            )
        )

    def test_historical_calls_do_not_force_latest_period(self):
        self.assertFalse(
            should_prefer_latest_transcript(
                "Compare Walmart margin trends over time across calls."
            )
        )

    def test_automatic_tickers_follow_question_mention_order(self):
        companies = [
            {"ticker": "WMT", "company_name": "Walmart Inc."},
            {"ticker": "COST", "company_name": "Costco Wholesale Corporation"},
        ]

        with (
            patch("app.rag.company_resolver.get_database_engine"),
            patch(
                "app.rag.company_resolver.get_companies",
                return_value=companies,
            ),
        ):
            tickers = resolve_tickers(
                "Compare Costco with Walmart margins."
            )

        self.assertEqual(tickers, ["COST", "WMT"])

    def test_explicit_ticker_list_is_normalized_and_deduplicated(self):
        with patch("app.rag.generator.resolve_tickers") as resolver:
            tickers = resolve_question_tickers(
                "Compare Walmart and NVIDIA",
                ticker="AAPL",
                tickers=["wmt", "NVDA", "WMT"],
            )

        self.assertEqual(tickers, ["WMT", "NVDA"])
        resolver.assert_not_called()

    def test_singular_ticker_remains_backward_compatible(self):
        tickers = resolve_question_tickers(
            "What did management say?",
            ticker="wmt",
        )

        self.assertEqual(tickers, ["WMT"])

    def test_multi_ticker_evidence_is_retrieved_as_one_balanced_request(self):
        with patch(
            "app.rag.generator.retrieve_evidence",
            return_value=[],
        ) as retrieve:
            collect_evidence(
                question="Compare Walmart and NVIDIA margins",
                tickers=["WMT", "NVDA"],
                source_type="transcripts",
                top_k=3,
            )

        self.assertEqual(
            retrieve.call_args.kwargs["tickers"],
            ["WMT", "NVDA"],
        )

    def test_market_only_answer_cannot_keep_document_citation(self):
        answer = finalize_grounded_answer(
            "NVDA returned 17.29% versus SPY [S1].",
            [],
        )

        self.assertEqual(answer, "NVDA returned 17.29% versus SPY.")

    def test_single_call_answer_gets_exact_fiscal_period(self):
        answer = finalize_grounded_answer(
            "Management said margins improved [S1].",
            [
                {
                    "source_type": "transcript",
                    "ticker": "WMT",
                    "fiscal_period": "2026Q4",
                }
            ],
        )

        self.assertTrue(
            answer.startswith("Evidence period: WMT 2026Q4 earnings call.")
        )
        self.assertIn("[S1]", answer)

    def test_existing_exact_period_is_not_duplicated(self):
        answer = finalize_grounded_answer(
            "On the WMT 2026Q4 call, management discussed margins [S1].",
            [
                {
                    "source_type": "transcript",
                    "ticker": "WMT",
                    "fiscal_period": "2026Q4",
                }
            ],
        )

        self.assertEqual(
            answer,
            "On the WMT 2026Q4 call, management discussed margins [S1].",
        )


if __name__ == "__main__":
    unittest.main()
