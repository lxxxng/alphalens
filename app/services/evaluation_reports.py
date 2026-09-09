"""Read local evaluation reports for the internal quality dashboard."""

import json
import os
from pathlib import Path


LATEST_REPORTS = {
    "retrieval": "retrieval_report.json",
    "response": "response_report.json",
}


def get_report_directory() -> Path:
    """Return the shared report directory used by local runs and CI."""

    return Path(
        os.getenv(
            "ALPHALENS_EVAL_REPORT_DIRECTORY",
            "data/evals",
        )
    )


def read_report(path: Path) -> dict:
    """Read one JSON object and reject malformed report roots."""

    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError("Evaluation report root must be an object.")
    if not isinstance(report.get("summary"), dict):
        raise ValueError("Evaluation report is missing its summary.")
    return report


def report_type(report: dict) -> str:
    """Identify the suite without depending on a specific suite version."""

    summary = report.get("summary", {})
    if "average_judge_scores" in summary:
        return "response"
    return "retrieval"


def summarize_report(report: dict) -> dict:
    """Keep only fields needed to plot and label one historical run."""

    summary = report["summary"]
    return {
        "suite_type": report_type(report),
        "suite": report.get("suite"),
        "version": report.get("version"),
        "generated_at": report.get("generated_at"),
        "total_cases": summary.get("total_cases", 0),
        "passed_cases": summary.get("passed_cases", 0),
        "failed_cases": summary.get("failed_cases", 0),
        "case_pass_rate": summary.get("case_pass_rate", 0),
        "check_pass_rate": summary.get("check_pass_rate", 0),
        "average_judge_scores": summary.get("average_judge_scores"),
        "generation_model": report.get("generation_model"),
        "judge_model": report.get("judge_model"),
    }


def load_evaluation_dashboard(
    history_limit: int = 50,
    report_directory: Path | None = None,
) -> dict:
    """Load latest full reports and bounded historical summaries."""

    directory = report_directory or get_report_directory()
    latest = {}
    errors = []

    for suite_type, filename in LATEST_REPORTS.items():
        path = directory / filename
        if not path.is_file():
            latest[suite_type] = None
            continue

        try:
            latest[suite_type] = read_report(path)
        except Exception as exception:
            latest[suite_type] = None
            errors.append(f"Could not read {filename}: {exception}")

    history = []
    history_directory = directory / "history"
    history_paths = (
        sorted(history_directory.glob("*.json"), reverse=True)
        if history_directory.is_dir()
        else []
    )

    for path in history_paths:
        try:
            history.append(summarize_report(read_report(path)))
        except Exception as exception:
            errors.append(f"Could not read history report {path.name}: {exception}")

        if len(history) >= history_limit:
            break

    # A new installation has latest reports before it has archived runs. Add
    # those latest points so the chart and run ledger are immediately useful.
    seen = {
        (item["suite"], item["generated_at"])
        for item in history
    }
    for report in latest.values():
        if report is None:
            continue

        item = summarize_report(report)
        key = (item["suite"], item["generated_at"])
        if key not in seen:
            history.append(item)
            seen.add(key)

    history.sort(
        key=lambda item: item.get("generated_at") or "",
        reverse=True,
    )
    history = history[:history_limit]

    return {
        "latest": latest,
        "history": history,
        "errors": errors,
    }
