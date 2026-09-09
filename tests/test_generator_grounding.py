"""Tests for deterministic answer finalization after model generation."""

import unittest

from app.rag.generator import finalize_grounded_answer


class GeneratorGroundingTests(unittest.TestCase):
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
