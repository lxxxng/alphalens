"""Shared report writing and archival helpers for AlphaLens evaluations."""

import json
import re
from datetime import datetime
from pathlib import Path


def archive_filename(report: dict, output_path: Path) -> str:
    """Build a Windows-safe, chronologically sortable archive filename."""

    generated_at = datetime.fromisoformat(report["generated_at"])
    timestamp = generated_at.strftime("%Y%m%dT%H%M%S%fZ")
    suite = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(report.get("suite", "eval")))
    return f"{timestamp}_{suite}_{output_path.stem}.json"


def write_report(
    report: dict,
    output_path: Path,
    *,
    archive: bool = False,
) -> Path | None:
    """Write the latest report and optionally preserve a timestamped copy."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(report, indent=2) + "\n"
    output_path.write_text(serialized, encoding="utf-8")

    if not archive:
        return None

    history_directory = output_path.parent / "history"
    history_directory.mkdir(parents=True, exist_ok=True)
    archive_path = history_directory / archive_filename(report, output_path)
    archive_path.write_text(serialized, encoding="utf-8")
    return archive_path
