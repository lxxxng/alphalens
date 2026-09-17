"""Tests for event-brief quality checks and alert queue processing."""

import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from sqlalchemy import (
    JSON,
    BigInteger,
    Column,
    DateTime,
    Integer,
    MetaData,
    String,
    Table,
    create_engine,
    insert,
    select,
)
from sqlalchemy.pool import StaticPool

from app.rag.retrievers.base import apply_metadata_filters
from app.services.automated_briefs import (
    _generate_snapshot,
    run_automated_event_briefs,
)
from app.services.brief_quality import evaluate_event_brief


def _brief_result():
    return {
        "ticker": "WMT",
        "tickers": ["WMT"],
        "event_type": "earnings",
        "question": "Prepare a WMT event brief.",
        "chain_version": "chain-v1",
        "model_name": "test-model",
        "brief": {
            "headline": "WMT reports its latest quarter",
            "executive_summary": (
                "Walmart discussed margin drivers, operating discipline, and "
                "the outlook for its latest quarter in the supplied call [S1]."
            ),
            "key_developments": ["Management discussed margins [S1]."],
            "topic_signals": [],
            "market_reaction": ["WMT performance was measured against SPY."],
            "risks": [],
            "watch_items": ["Monitor operating margin progression."],
            "limitations": [],
        },
        "market_context": [{"ticker": "WMT"}],
        "sentiment_context": {},
        "sources": [{
            "source": "S1",
            "ticker": "WMT",
            "source_type": "transcript",
            "transcript_id": 91,
        }],
    }


def _sqlite_engine():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    metadata = MetaData()
    Table(
        "event_alerts",
        metadata,
        Column("alert_id", Integer, primary_key=True),
        Column("ticker", String, nullable=False),
        Column("event_type", String, nullable=False),
        Column("source_record_id", String, nullable=False),
        Column("brief_id", BigInteger),
        Column("brief_status", String, nullable=False),
        Column("brief_attempt_count", Integer, nullable=False, default=0),
        Column("brief_error", String),
        Column("brief_evaluation", JSON, nullable=False, default=dict),
        Column("brief_last_attempt_at", DateTime(timezone=True)),
        Column("brief_generated_at", DateTime(timezone=True)),
        Column("created_at", DateTime(timezone=True), nullable=False),
    )
    Table(
        "filings",
        metadata,
        Column("accession_number", String, primary_key=True),
        Column("form_type", String, nullable=False),
    )
    Table(
        "earnings_transcripts",
        metadata,
        Column("transcript_id", BigInteger, primary_key=True),
        Column("fiscal_period", String, nullable=False),
    )
    metadata.create_all(engine)
    return engine, metadata


class AutomatedBriefTests(unittest.TestCase):
    def test_metadata_filter_targets_exact_source_identity(self):
        results = [
            {
                "ticker": "WMT",
                "accession_number": "0001",
                "transcript_id": None,
            },
            {
                "ticker": "WMT",
                "accession_number": "0002",
                "transcript_id": None,
            },
        ]

        filtered = apply_metadata_filters(
            results,
            ticker="WMT",
            accession_number="0002",
        )

        self.assertEqual(filtered, [results[1]])

    def test_snapshot_reuses_cached_brief_without_generation(self):
        cached = {
            **_brief_result(),
            "brief_id": 22,
            "quality_evaluation": {"passed": True},
        }
        request = {
            "ticker": "WMT",
            "event_type": "earnings",
            "transcript_id": 91,
            "top_k": 6,
        }

        with (
            patch(
                "app.services.automated_briefs.get_latest_event_fingerprint",
                return_value={"companies": []},
            ),
            patch(
                "app.services.automated_briefs.find_cached_event_brief",
                return_value=cached,
            ),
            patch(
                "app.services.automated_briefs.generate_event_brief",
            ) as generate,
        ):
            result, brief_id, was_cached = _generate_snapshot(
                request,
                force_refresh=False,
            )

        self.assertIs(result, cached)
        self.assertEqual(brief_id, 22)
        self.assertTrue(was_cached)
        generate.assert_not_called()

    def test_quality_gate_accepts_exact_grounded_event(self):
        evaluation = evaluate_event_brief(
            _brief_result(),
            {
                "ticker": "WMT",
                "event_type": "earnings",
                "transcript_id": 91,
            },
        )

        self.assertTrue(evaluation["passed"])
        self.assertEqual(evaluation["passed_checks"], evaluation["total_checks"])

    def test_quality_gate_rejects_wrong_event_source(self):
        result = _brief_result()
        result["sources"][0]["transcript_id"] = 92

        evaluation = evaluate_event_brief(
            result,
            {
                "ticker": "WMT",
                "event_type": "earnings",
                "transcript_id": 91,
            },
        )
        checks = {check["name"]: check for check in evaluation["checks"]}

        self.assertFalse(evaluation["passed"])
        self.assertFalse(checks["exact_event_scope"]["passed"])

    def test_queue_marks_quality_approved_brief_passed(self):
        engine, metadata = _sqlite_engine()
        alerts = metadata.tables["event_alerts"]
        transcripts = metadata.tables["earnings_transcripts"]

        with engine.begin() as connection:
            connection.execute(insert(transcripts).values(
                transcript_id=91,
                fiscal_period="2026Q4",
            ))
            connection.execute(insert(alerts).values(
                alert_id=1,
                ticker="WMT",
                event_type="earnings",
                source_record_id="91",
                brief_status="PENDING",
                brief_attempt_count=0,
                brief_evaluation={},
                created_at=datetime(2026, 9, 16, 6, tzinfo=timezone.utc),
            ))

        generated = _brief_result()
        generated["quality_evaluation"] = {
            "passed": True,
            "passed_checks": 7,
            "total_checks": 7,
        }

        with (
            patch(
                "app.services.automated_briefs.get_database_engine",
                return_value=engine,
            ),
            patch(
                "app.services.automated_briefs._generate_snapshot",
                return_value=(generated, 44, False),
            ),
        ):
            summary = run_automated_event_briefs(["WMT"])

        with engine.connect() as connection:
            row = connection.execute(select(alerts)).mappings().one()

        self.assertEqual(summary["passed"], 1)
        self.assertEqual(row["brief_status"], "PASSED")
        self.assertEqual(row["brief_id"], 44)
        self.assertEqual(row["brief_attempt_count"], 1)

    def test_queue_preserves_failure_for_retry(self):
        engine, metadata = _sqlite_engine()
        alerts = metadata.tables["event_alerts"]
        filings = metadata.tables["filings"]

        with engine.begin() as connection:
            connection.execute(insert(filings).values(
                accession_number="0001",
                form_type="10-Q",
            ))
            connection.execute(insert(alerts).values(
                alert_id=2,
                ticker="WMT",
                event_type="filing",
                source_record_id="0001",
                brief_status="PENDING",
                brief_attempt_count=0,
                brief_evaluation={},
                created_at=datetime(2026, 9, 16, 6, tzinfo=timezone.utc),
            ))

        with (
            patch(
                "app.services.automated_briefs.get_database_engine",
                return_value=engine,
            ),
            patch(
                "app.services.automated_briefs._generate_snapshot",
                side_effect=RuntimeError("temporary model outage"),
            ),
        ):
            summary = run_automated_event_briefs(["WMT"])

        with engine.connect() as connection:
            row = connection.execute(select(alerts)).mappings().one()

        self.assertEqual(summary["failed"], 1)
        self.assertEqual(row["brief_status"], "FAILED")
        self.assertIn("temporary model outage", row["brief_error"])
        self.assertEqual(row["brief_attempt_count"], 1)


if __name__ == "__main__":
    unittest.main()
