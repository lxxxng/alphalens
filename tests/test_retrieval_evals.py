"""Tests for deterministic retrieval graders."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evals.run_retrieval import (
    build_report,
    grade_retrieval_case,
    load_suite,
    run_case,
)


class RetrievalEvalTests(unittest.TestCase):
    def test_runner_forwards_explicit_ticker_list(self):
        case = {
            "id": "multi_ticker_request",
            "question": "Compare Walmart and Costco",
            "request": {
                "tickers": ["WMT", "COST"],
                "source_type": "transcripts",
            },
            "expect": {
                "detected_tickers": ["WMT", "COST"],
            },
        }
        evidence = {
            "detected_tickers": ["WMT", "COST"],
            "market_context": [],
            "retrieved_results": [],
        }

        with patch(
            "evals.run_retrieval.collect_evidence",
            return_value=evidence,
        ) as collect:
            result = run_case(case)

        self.assertTrue(result["passed"])
        self.assertEqual(
            collect.call_args.kwargs["tickers"],
            ["WMT", "COST"],
        )

    def test_grader_requires_evidence_from_each_company(self):
        case = {
            "expect": {
                "required_source_tickers": ["WMT", "COST"],
            }
        }
        evidence = {
            "retrieved_results": [
                {"ticker": "WMT", "source_type": "transcript"},
                {"ticker": "COST", "source_type": "transcript"},
            ],
            "market_context": [],
        }

        checks = grade_retrieval_case(
            case,
            evidence,
            ["transcripts"],
            True,
        )

        self.assertTrue(checks[0]["passed"])

    def test_expected_comparison_limit_error_can_pass(self):
        case = {
            "id": "comparison_limit",
            "question": "Compare five companies",
            "expect": {
                "expected_error_contains": "up to 4 companies",
            },
        }

        with patch(
            "evals.run_retrieval.collect_evidence",
            side_effect=ValueError(
                "AlphaLens supports comparisons between up to 4 companies."
            ),
        ):
            result = run_case(case)

        self.assertTrue(result["passed"])

    def test_grader_accepts_matching_transcript_evidence(self):
        case = {
            "expect": {
                "detected_tickers": ["WMT"],
                "routed_source_types": ["transcripts"],
                "text_search_enabled": True,
                "min_sources": 1,
                "required_source_types": ["transcript"],
                "allowed_fiscal_periods": ["2026Q4"],
                "content_terms_any": ["margin"],
            }
        }
        evidence = {
            "detected_tickers": ["WMT"],
            "market_context": [],
            "retrieved_results": [
                {
                    "source_type": "transcript",
                    "fiscal_period": "2026Q4",
                    "content": "Operating margin improved.",
                }
            ],
        }

        checks = grade_retrieval_case(
            case,
            evidence,
            ["transcripts"],
            True,
        )

        self.assertTrue(all(check["passed"] for check in checks))

    def test_grader_reports_wrong_fiscal_period(self):
        case = {
            "expect": {
                "allowed_fiscal_periods": ["2026Q4"],
            }
        }
        evidence = {
            "retrieved_results": [
                {"fiscal_period": "2026Q3"}
            ],
            "market_context": [],
        }

        checks = grade_retrieval_case(
            case,
            evidence,
            ["transcripts"],
            True,
        )

        self.assertFalse(checks[0]["passed"])

    def test_report_calculates_case_and_check_rates(self):
        results = [
            {
                "passed": True,
                "checks": [{"passed": True}],
            },
            {
                "passed": False,
                "checks": [
                    {"passed": True},
                    {"passed": False},
                ],
            },
        ]

        report = build_report(
            suite={"name": "test", "version": 1},
            results=results,
            cases_path=Path("test-cases.json"),
        )

        self.assertEqual(report["summary"]["case_pass_rate"], 0.5)
        self.assertEqual(
            report["summary"]["check_pass_rate"],
            2 / 3,
        )

    def test_suite_rejects_unknown_expectations(self):
        suite = {
            "name": "test",
            "version": 1,
            "cases": [
                {
                    "id": "typo",
                    "question": "A valid question?",
                    "expect": {"min_soruces": 1},
                }
            ],
        }

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.json"
            path.write_text(
                json.dumps(suite),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError,
                "unsupported expectation",
            ):
                load_suite(path)

if __name__ == "__main__":
    unittest.main()
