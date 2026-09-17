"""Deterministic publication checks for generated event briefs."""

from __future__ import annotations

import re
from datetime import datetime, timezone


BRIEF_QUALITY_VERSION = "alphalens-brief-quality-v1"
CITATION_PATTERN = re.compile(r"\[S(\d+)\]")
REQUIRED_SECTIONS = {
    "headline",
    "executive_summary",
    "key_developments",
    "topic_signals",
    "market_reaction",
    "risks",
    "watch_items",
    "limitations",
}


def _add_check(checks: list[dict], name: str, passed: bool, detail: str) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


def _brief_text(brief: dict) -> str:
    values = []

    for value in brief.values():
        if isinstance(value, str):
            values.append(value)
        elif isinstance(value, list):
            values.extend(str(item) for item in value)

    return "\n".join(values)


def evaluate_event_brief(result: dict, request_data: dict) -> dict:
    """Apply a cheap, explainable gate before automation marks a brief ready."""

    brief = result.get("brief") or {}
    sources = result.get("sources") or []
    market_context = result.get("market_context") or []
    requested_tickers = {
        str(value).strip().upper()
        for value in (
            request_data.get("tickers")
            or [request_data.get("ticker")]
        )
        if value
    }
    text = _brief_text(brief)
    citation_labels = {
        f"S{number}" for number in CITATION_PATTERN.findall(text)
    }
    available_labels = {
        str(source.get("source")) for source in sources if source.get("source")
    }
    source_tickers = {
        str(source.get("ticker", "")).upper()
        for source in sources
        if source.get("ticker")
    }
    checks = []

    _add_check(
        checks,
        "structured_sections",
        REQUIRED_SECTIONS.issubset(brief),
        "All required brief sections are present.",
    )
    _add_check(
        checks,
        "substantive_summary",
        len(str(brief.get("executive_summary", "")).strip()) >= 80
        and bool(brief.get("key_developments")),
        "Executive summary has at least 80 characters and key developments.",
    )
    _add_check(
        checks,
        "evidence_present",
        bool(sources),
        "At least one retrieved document source supports the brief.",
    )
    _add_check(
        checks,
        "valid_citations",
        bool(citation_labels) and citation_labels.issubset(available_labels),
        "Citations are present and every label belongs to the evidence bundle.",
    )
    _add_check(
        checks,
        "ticker_coverage",
        bool(requested_tickers) and requested_tickers.issubset(source_tickers),
        "Retrieved evidence covers every requested ticker.",
    )

    accession_number = request_data.get("accession_number")
    transcript_id = request_data.get("transcript_id")
    event_scope_matches = True

    if accession_number:
        event_scope_matches = bool(sources) and all(
            source.get("accession_number") == accession_number
            for source in sources
        )
    elif transcript_id is not None:
        event_scope_matches = bool(sources) and all(
            int(source.get("transcript_id") or 0) == int(transcript_id)
            for source in sources
        )

    _add_check(
        checks,
        "exact_event_scope",
        event_scope_matches,
        "Every document source belongs to the triggering event.",
    )
    _add_check(
        checks,
        "market_context_covered",
        not market_context or bool(brief.get("market_reaction")),
        "Available structured market context is represented in the brief.",
    )

    passed_count = sum(check["passed"] for check in checks)
    return {
        "version": BRIEF_QUALITY_VERSION,
        "passed": passed_count == len(checks),
        "score": round(passed_count / len(checks), 3),
        "passed_checks": passed_count,
        "total_checks": len(checks),
        "checks": checks,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }
