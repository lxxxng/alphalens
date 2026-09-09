"""Tests for evaluation report loading and historical summaries."""

import json
import tempfile
import unittest
from pathlib import Path

from app.services.evaluation_reports import load_evaluation_dashboard
from evals.reporting import write_report


def _report(
    suite: str,
    generated_at: str,
    *,
    response: bool = False,
    passed_cases: int = 2,
) -> dict:
    summary = {
        "total_cases": 2,
        "passed_cases": passed_cases,
        "failed_cases": 2 - passed_cases,
        "case_pass_rate": passed_cases / 2,
        "total_checks": 4,
        "passed_checks": passed_cases * 2,
        "check_pass_rate": passed_cases / 2,
    }
    if response:
        summary["average_judge_scores"] = {
            "groundedness": 5.0,
            "relevance": 4.5,
            "completeness": 4.0,
            "citation_quality": 5.0,
        }

    return {
        "suite": suite,
        "version": 1,
        "generated_at": generated_at,
        "summary": summary,
        "results": [],
    }


class EvaluationReportTests(unittest.TestCase):
    def test_writer_preserves_timestamped_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = _report("response", "2026-09-09T02:00:00+00:00", response=True)

            archive_path = write_report(
                report,
                root / "response_report.json",
                archive=True,
            )

            self.assertTrue((root / "response_report.json").is_file())
            self.assertTrue(archive_path.is_file())
            self.assertEqual(json.loads(archive_path.read_text()), report)

    def test_loads_latest_reports_and_identifies_suite_types(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "retrieval_report.json").write_text(
                json.dumps(_report("retrieval", "2026-09-09T01:00:00+00:00")),
                encoding="utf-8",
            )
            (root / "response_report.json").write_text(
                json.dumps(
                    _report(
                        "response",
                        "2026-09-09T02:00:00+00:00",
                        response=True,
                    )
                ),
                encoding="utf-8",
            )

            dashboard = load_evaluation_dashboard(report_directory=root)

        self.assertEqual(dashboard["latest"]["retrieval"]["suite"], "retrieval")
        self.assertEqual(dashboard["history"][0]["suite_type"], "response")
        self.assertEqual(len(dashboard["history"]), 2)

    def test_history_is_bounded_and_latest_is_not_duplicated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            history = root / "history"
            history.mkdir()
            report = _report("retrieval", "2026-09-09T01:00:00+00:00")
            serialized = json.dumps(report)
            (root / "retrieval_report.json").write_text(
                serialized,
                encoding="utf-8",
            )
            (history / "same.json").write_text(serialized, encoding="utf-8")
            (history / "older.json").write_text(
                json.dumps(_report("retrieval", "2026-09-08T01:00:00+00:00")),
                encoding="utf-8",
            )

            dashboard = load_evaluation_dashboard(
                history_limit=1,
                report_directory=root,
            )

        self.assertEqual(len(dashboard["history"]), 1)
        self.assertEqual(
            dashboard["history"][0]["generated_at"],
            "2026-09-09T01:00:00+00:00",
        )

    def test_malformed_latest_report_returns_an_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "retrieval_report.json").write_text("not-json", encoding="utf-8")

            dashboard = load_evaluation_dashboard(report_directory=root)

        self.assertIsNone(dashboard["latest"]["retrieval"])
        self.assertEqual(len(dashboard["errors"]), 1)


if __name__ == "__main__":
    unittest.main()
