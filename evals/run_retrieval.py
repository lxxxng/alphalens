"""
Run deterministic retrieval evaluations against the live AlphaLens corpus.

The harness calls collect_evidence(), so text cases create query embeddings
but never call the answer-generation model. Results are suitable for local
development and CI release gates.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from app.rag.generator import (
    collect_evidence,
    wants_text_evidence,
)
from app.rag.retrievers.router import resolve_source_types


DEFAULT_CASES_PATH = Path("evals/retrieval_cases.json")
DEFAULT_REPORT_PATH = Path("data/evals/retrieval_report.json")

SUPPORTED_EXPECTATIONS = {
    "allowed_fiscal_periods",
    "allowed_form_types",
    "content_terms_any",
    "detected_tickers",
    "market_tickers",
    "max_sources",
    "min_market_snapshots",
    "min_sources",
    "required_section_keys",
    "required_source_types",
    "routed_source_types",
    "text_search_enabled",
}


def load_suite(path: Path) -> dict:
    """Load and minimally validate a retrieval evaluation suite."""

    data = json.loads(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(data.get("cases"), list):
        raise ValueError(
            "Evaluation file must contain a cases list."
        )

    case_ids = [case.get("id") for case in data["cases"]]

    if any(not case_id for case_id in case_ids):
        raise ValueError(
            "Every evaluation case must have an id."
        )

    if len(case_ids) != len(set(case_ids)):
        raise ValueError(
            "Evaluation case ids must be unique."
        )

    for case in data["cases"]:
        if not str(case.get("question", "")).strip():
            raise ValueError(
                f"Case {case['id']} must have a question."
            )

        expectations = case.get("expect")

        if not isinstance(expectations, dict) or not expectations:
            raise ValueError(
                f"Case {case['id']} must declare expectations."
            )

        unsupported = set(expectations) - SUPPORTED_EXPECTATIONS

        if unsupported:
            raise ValueError(
                f"Case {case['id']} has unsupported expectation(s): "
                + ", ".join(sorted(unsupported))
            )

    return data


def add_check(
    checks: list[dict],
    *,
    name: str,
    passed: bool,
    expected,
    actual,
) -> None:
    """Append one JSON-friendly deterministic grader result."""

    checks.append(
        {
            "name": name,
            "passed": bool(passed),
            "expected": expected,
            "actual": actual,
        }
    )


def grade_retrieval_case(
    case: dict,
    evidence: dict,
    routed_source_types: list[str],
    text_search_enabled: bool,
) -> list[dict]:
    """Grade one retrieval result against its declared expectations."""

    expected = case.get("expect", {})
    sources = evidence.get("retrieved_results", [])
    market_context = evidence.get("market_context", [])
    checks = []

    if "detected_tickers" in expected:
        actual = evidence.get("detected_tickers", [])
        add_check(
            checks,
            name="detected_tickers",
            passed=actual == expected["detected_tickers"],
            expected=expected["detected_tickers"],
            actual=actual,
        )

    if "routed_source_types" in expected:
        add_check(
            checks,
            name="routed_source_types",
            passed=(
                routed_source_types
                == expected["routed_source_types"]
            ),
            expected=expected["routed_source_types"],
            actual=routed_source_types,
        )

    if "text_search_enabled" in expected:
        add_check(
            checks,
            name="text_search_enabled",
            passed=(
                text_search_enabled
                is expected["text_search_enabled"]
            ),
            expected=expected["text_search_enabled"],
            actual=text_search_enabled,
        )

    if "min_sources" in expected:
        minimum = expected["min_sources"]
        add_check(
            checks,
            name="min_sources",
            passed=len(sources) >= minimum,
            expected=f">= {minimum}",
            actual=len(sources),
        )

    if "max_sources" in expected:
        maximum = expected["max_sources"]
        add_check(
            checks,
            name="max_sources",
            passed=len(sources) <= maximum,
            expected=f"<= {maximum}",
            actual=len(sources),
        )

    source_types = sorted(
        {
            source.get("source_type")
            for source in sources
            if source.get("source_type")
        }
    )

    if "required_source_types" in expected:
        required = expected["required_source_types"]
        add_check(
            checks,
            name="required_source_types",
            passed=set(required).issubset(source_types),
            expected=required,
            actual=source_types,
        )

    for expectation_key, source_key, check_name in [
        (
            "allowed_fiscal_periods",
            "fiscal_period",
            "fiscal_periods",
        ),
        (
            "allowed_form_types",
            "form_type",
            "form_types",
        ),
    ]:
        if expectation_key not in expected:
            continue

        allowed = expected[expectation_key]
        actual = sorted(
            {
                source[source_key]
                for source in sources
                if source.get(source_key)
            }
        )
        add_check(
            checks,
            name=check_name,
            passed=bool(actual) and set(actual).issubset(allowed),
            expected=allowed,
            actual=actual,
        )

    if "required_section_keys" in expected:
        required = expected["required_section_keys"]
        actual = sorted(
            {
                source["section_key"]
                for source in sources
                if source.get("section_key")
            }
        )
        add_check(
            checks,
            name="section_keys",
            passed=set(required).issubset(actual),
            expected=required,
            actual=actual,
        )

    if "content_terms_any" in expected:
        terms = [
            term.lower()
            for term in expected["content_terms_any"]
        ]
        combined_content = "\n".join(
            source.get("content") or ""
            for source in sources
        ).lower()
        matched_terms = [
            term
            for term in terms
            if term in combined_content
        ]
        add_check(
            checks,
            name="content_terms_any",
            passed=bool(matched_terms),
            expected=terms,
            actual=matched_terms,
        )

    if "min_market_snapshots" in expected:
        minimum = expected["min_market_snapshots"]
        add_check(
            checks,
            name="min_market_snapshots",
            passed=len(market_context) >= minimum,
            expected=f">= {minimum}",
            actual=len(market_context),
        )

    if "market_tickers" in expected:
        expected_tickers = expected["market_tickers"]
        actual_tickers = [
            snapshot["ticker"]
            for snapshot in market_context
        ]
        add_check(
            checks,
            name="market_tickers",
            passed=actual_tickers == expected_tickers,
            expected=expected_tickers,
            actual=actual_tickers,
        )

    return checks


def summarize_sources(sources: list[dict]) -> list[dict]:
    """Keep report evidence useful without duplicating full chunk text."""

    return [
        {
            "rank": rank,
            "chunk_id": source.get("chunk_id"),
            "ticker": source.get("ticker"),
            "source_type": source.get("source_type"),
            "form_type": source.get("form_type"),
            "section_key": source.get("section_key"),
            "fiscal_period": source.get("fiscal_period"),
            "score": round(float(source.get("score", 0)), 4),
        }
        for rank, source in enumerate(sources, start=1)
    ]


def run_case(case: dict) -> dict:
    """Run retrieval and deterministic graders for one case."""

    started_at = perf_counter()
    request = dict(case.get("request", {}))
    question = case["question"]
    source_type = request.get("source_type", "auto")
    routed_source_types = resolve_source_types(
        question=question,
        source_type=source_type,
    )
    text_search_enabled = wants_text_evidence(
        question=question,
        source_type=source_type,
    )

    try:
        evidence = collect_evidence(
            question=question,
            top_k=request.get("top_k", 5),
            ticker=request.get("ticker"),
            form_type=request.get("form_type"),
            section_key=request.get("section_key"),
            fiscal_period=request.get("fiscal_period"),
            source_type=source_type,
        )
        checks = grade_retrieval_case(
            case=case,
            evidence=evidence,
            routed_source_types=routed_source_types,
            text_search_enabled=text_search_enabled,
        )
        error = None
    except Exception as exception:
        evidence = {
            "detected_tickers": [],
            "market_context": [],
            "retrieved_results": [],
        }
        checks = [
            {
                "name": "execution",
                "passed": False,
                "expected": "successful retrieval",
                "actual": type(exception).__name__,
            }
        ]
        error = str(exception)

    return {
        "id": case["id"],
        "question": question,
        "passed": all(check["passed"] for check in checks),
        "duration_ms": round(
            (perf_counter() - started_at) * 1000,
            1,
        ),
        "checks": checks,
        "detected_tickers": evidence.get("detected_tickers", []),
        "market_tickers": [
            item["ticker"]
            for item in evidence.get("market_context", [])
        ],
        "sources": summarize_sources(
            evidence.get("retrieved_results", [])
        ),
        "error": error,
    }


def build_report(
    suite: dict,
    results: list[dict],
    cases_path: Path,
) -> dict:
    """Build aggregate case and grader pass rates."""

    total_checks = sum(
        len(result["checks"])
        for result in results
    )
    passed_checks = sum(
        check["passed"]
        for result in results
        for check in result["checks"]
    )
    passed_cases = sum(
        result["passed"]
        for result in results
    )
    total_cases = len(results)

    return {
        "suite": suite.get("name"),
        "version": suite.get("version"),
        "cases_path": str(cases_path),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "total_cases": total_cases,
            "passed_cases": passed_cases,
            "failed_cases": total_cases - passed_cases,
            "case_pass_rate": (
                passed_cases / total_cases
                if total_cases
                else 0
            ),
            "total_checks": total_checks,
            "passed_checks": passed_checks,
            "check_pass_rate": (
                passed_checks / total_checks
                if total_checks
                else 0
            ),
        },
        "results": results,
    }


def print_result(result: dict) -> None:
    """Print one compact terminal result with failure diagnostics."""

    label = "PASS" if result["passed"] else "FAIL"
    print(
        f"[{label}] {result['id']} "
        f"({result['duration_ms']:.0f} ms, "
        f"{len(result['sources'])} sources, "
        f"{len(result['market_tickers'])} market)"
    )

    for check in result["checks"]:
        if check["passed"]:
            continue

        print(
            f"  - {check['name']}: "
            f"expected {check['expected']!r}, "
            f"got {check['actual']!r}"
        )

    if result["error"]:
        print(f"  - error: {result['error']}")


def parse_args() -> argparse.Namespace:
    """Parse retrieval-evaluation CLI options."""

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate AlphaLens retrieval without generating answers."
        )
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=DEFAULT_CASES_PATH,
        help="Path to the versioned evaluation JSON file.",
    )
    parser.add_argument(
        "--case-id",
        action="append",
        help="Run one case ID; repeat to select several cases.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Run only the first N selected cases.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Path for the JSON report.",
    )
    parser.add_argument(
        "--fail-under",
        type=float,
        default=1.0,
        help="Required case pass rate from 0 to 1 (default: 1).",
    )
    return parser.parse_args()


def main() -> int:
    """Run selected cases, write the report, and return a CI exit code."""

    args = parse_args()

    if not 0 <= args.fail_under <= 1:
        print("--fail-under must be between 0 and 1.", file=sys.stderr)
        return 2

    try:
        suite = load_suite(args.cases)
    except Exception as exception:
        print(
            f"Could not load evaluation suite: {exception}",
            file=sys.stderr,
        )
        return 2

    cases = suite["cases"]

    if args.case_id:
        selected = set(args.case_id)
        cases = [
            case
            for case in cases
            if case["id"] in selected
        ]
        missing = selected - {
            case["id"]
            for case in cases
        }

        if missing:
            print(
                "Unknown case ID(s): "
                + ", ".join(sorted(missing)),
                file=sys.stderr,
            )
            return 2

    if args.limit is not None:
        if args.limit < 1:
            print("--limit must be at least 1.", file=sys.stderr)
            return 2

        cases = cases[:args.limit]

    results = []

    for case in cases:
        result = run_case(case)
        results.append(result)
        print_result(result)

    report = build_report(
        suite=suite,
        results=results,
        cases_path=args.cases,
    )
    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    args.output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    summary = report["summary"]
    print()
    print(
        "Retrieval eval: "
        f"{summary['passed_cases']}/{summary['total_cases']} cases passed "
        f"({summary['case_pass_rate']:.1%}); "
        f"{summary['passed_checks']}/{summary['total_checks']} checks passed."
    )
    print(f"Report: {args.output}")

    return (
        0
        if summary["case_pass_rate"] >= args.fail_under
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
