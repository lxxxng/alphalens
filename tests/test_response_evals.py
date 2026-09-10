"""Tests for response-quality suite validation and deterministic graders."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evals.run_responses import (
    build_report,
    citation_labels,
    grade_deterministic,
    load_suite,
    run_case,
)


class ResponseEvalTests(unittest.TestCase):
    def test_runner_forwards_explicit_ticker_list(self):
        case = {
            "id": "multi_ticker_answer",
            "question": "Compare Walmart and Costco",
            "request": {"tickers": ["WMT", "COST"]},
            "expect": {"answer_terms_any": ["comparison"]},
            "rubric": "Answer the comparison.",
        }
        evidence = {
            "question": case["question"],
            "market_context": [],
            "retrieved_results": [],
        }
        judgment = {
            "scores": {
                "groundedness": 5,
                "relevance": 5,
                "completeness": 5,
                "citation_quality": 5,
            },
            "overall_pass": True,
            "reasoning": "Pass.",
            "unsupported_claims": [],
        }

        with (
            patch(
                "evals.run_responses.collect_evidence",
                return_value=evidence,
            ) as collect,
            patch(
                "evals.run_responses.generate_grounded_answer",
                return_value="Comparison complete.",
            ),
        ):
            result = run_case(
                case,
                judge=lambda *_: judgment,
            )

        self.assertTrue(result["passed"])
        self.assertEqual(
            collect.call_args.kwargs["tickers"],
            ["WMT", "COST"],
        )

    def test_multi_company_grader_requires_citations_from_each_ticker(self):
        case = {
            "expect": {
                "required_source_tickers": ["WMT", "COST"],
                "required_cited_source_tickers": ["WMT", "COST"],
                "answer_terms_all": ["WMT", "COST"],
            }
        }
        sources = [
            {"source": "S1", "source_type": "transcript", "ticker": "WMT"},
            {"source": "S2", "source_type": "transcript", "ticker": "COST"},
        ]

        checks = grade_deterministic(
            case,
            "WMT discussed mix [S1]. COST discussed membership [S2].",
            sources,
            [],
        )

        self.assertTrue(all(check["passed"] for check in checks))

    def test_citation_labels_are_unique_and_ordered(self):
        self.assertEqual(
            citation_labels("Claim [S2]. More [S1], repeated [S2]."),
            ["S2", "S1"],
        )

    def test_deterministic_grader_accepts_grounded_answer(self):
        case = {
            "expect": {
                "requires_citations": True,
                "required_source_types": ["transcript"],
                "answer_terms_any": ["margin"],
                "should_abstain": False,
            }
        }
        checks = grade_deterministic(
            case,
            "Management said operating margin improved [S1].",
            [{"source": "S1", "source_type": "transcript"}],
            [],
        )

        self.assertTrue(all(check["passed"] for check in checks))

    def test_deterministic_grader_rejects_unknown_citation(self):
        case = {"expect": {"requires_citations": True}}
        checks = grade_deterministic(
            case,
            "The filing describes the risk [S2].",
            [{"source": "S1", "source_type": "filing"}],
            [],
        )
        checks_by_name = {check["name"]: check for check in checks}

        self.assertFalse(checks_by_name["valid_citation_labels"]["passed"])

    def test_deterministic_grader_accepts_abstention_without_citations(self):
        case = {
            "expect": {
                "requires_citations": False,
                "max_sources": 0,
                "should_abstain": True,
            }
        }
        checks = grade_deterministic(
            case,
            "The retrieval system did not find AlphaLens evidence relevant to this question.",
            [],
            [],
        )

        self.assertTrue(all(check["passed"] for check in checks))

    def test_suite_rejects_unknown_expectation(self):
        suite = {
            "name": "test",
            "version": 1,
            "cases": [
                {
                    "id": "typo",
                    "question": "A valid question?",
                    "rubric": "A valid rubric.",
                    "expect": {"requiers_citations": True},
                }
            ],
        }

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.json"
            path.write_text(json.dumps(suite), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "unsupported expectation"):
                load_suite(path)

    def test_report_averages_judge_dimensions(self):
        results = [
            {
                "passed": True,
                "checks": [{"passed": True}],
                "judge": {
                    "scores": {
                        "groundedness": 5,
                        "relevance": 4,
                        "completeness": 4,
                        "citation_quality": 5,
                    }
                },
            },
            {
                "passed": False,
                "checks": [{"passed": False}],
                "judge": {
                    "scores": {
                        "groundedness": 3,
                        "relevance": 4,
                        "completeness": 2,
                        "citation_quality": 3,
                    }
                },
            },
        ]

        report = build_report(
            {"name": "test", "version": 1},
            results,
            Path("cases.json"),
            4,
        )

        self.assertEqual(report["summary"]["case_pass_rate"], 0.5)
        self.assertEqual(
            report["summary"]["average_judge_scores"]["groundedness"],
            4.0,
        )


if __name__ == "__main__":
    unittest.main()
