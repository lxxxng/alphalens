"""Tests for deterministic retrieval graders."""

import json
import tempfile
import unittest
from pathlib import Path

from evals.run_retrieval import (
    build_report,
    grade_retrieval_case,
    load_suite,
)


class RetrievalEvalTests(unittest.TestCase):
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
