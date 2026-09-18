"""Tests for scheduled ingestion scope and stage orchestration."""

import unittest
from unittest.mock import MagicMock, Mock, patch

from sqlalchemy import Column, Integer, MetaData, String, Table

from pipelines.market_data.extractor import extract_market_data
from pipelines.scheduled_ingestion import (
    DEFAULT_STAGES,
    _mark_orphaned_runs,
    _run_model_monitor_stage,
    _stage_plan,
    normalize_tickers,
    resolve_tickers,
    run_scheduled_ingestion,
)


class ScheduledIngestionTests(unittest.TestCase):
    def test_orphaned_running_runs_are_closed_before_a_new_run(self):
        table = Table(
            "ingestion_runs",
            MetaData(),
            Column("run_id", Integer),
            Column("status", String),
            Column("current_stage", String),
            Column("error", String),
            Column("completed_at", String),
            Column("updated_at", String),
        )
        engine = MagicMock()
        connection = engine.begin.return_value.__enter__.return_value
        connection.execute.return_value.rowcount = 2

        recovered = _mark_orphaned_runs(engine, table)

        self.assertEqual(recovered, 2)
        statement = connection.execute.call_args.args[0]
        self.assertIn("UPDATE ingestion_runs", str(statement))
        self.assertIn("ingestion_runs.status", str(statement))

    @patch("pipelines.market_data.extractor.yf.download")
    def test_market_refresh_scopes_companies_and_keeps_spy(self, download):
        download.return_value = Mock()

        extract_market_data(
            tickers=["wmt", "NVDA", "wmt"],
            start_date="2026-09-01",
        )

        self.assertEqual(
            download.call_args.kwargs["tickers"],
            ["WMT", "NVDA", "SPY"],
        )
        self.assertEqual(
            download.call_args.kwargs["start"],
            "2026-09-01",
        )

    def test_tickers_are_normalized_in_requested_order(self):
        self.assertEqual(
            normalize_tickers([" wmt ", "NVDA", "wmt", ""]),
            ["WMT", "NVDA"],
        )

    def test_explicit_tickers_do_not_query_database_scope(self):
        engine = Mock()

        result = resolve_tickers(
            engine,
            scope="watchlists",
            explicit_tickers=["wmt", " nvda "],
        )

        self.assertEqual(result, ["WMT", "NVDA"])
        engine.connect.assert_not_called()

    def test_default_stage_plan_preserves_dependency_order(self):
        plan = _stage_plan(
            ["WMT"],
            list(DEFAULT_STAGES),
            market_lookback_days=14,
            max_transcripts_per_ticker=2,
        )

        self.assertEqual(
            [name for name, _ in plan],
            [
                "market",
                "earnings_results",
                "sec_metadata",
                "sec_documents",
                "sec_sections",
                "sec_chunks",
                "transcript_extract",
                "transcript_chunks",
                "event_alerts",
                "embeddings",
                "sentiment",
                "model_monitor",
                "automated_briefs",
            ],
        )

    @patch(
        "pipelines.ml.prospective.run_prospective_monitor",
        side_effect=FileNotFoundError("freeze the model first"),
    )
    def test_model_monitor_skips_cleanly_before_one_time_freeze(self, run_monitor):
        result = _run_model_monitor_stage(["WMT"])

        self.assertEqual(result["status"], "SKIPPED")
        self.assertIn("freeze", result["reason"])
        run_monitor.assert_called_once_with(tickers=["WMT"])

    @patch("pipelines.scheduled_ingestion.resolve_tickers")
    @patch("pipelines.scheduled_ingestion.get_database_engine")
    def test_dry_run_has_no_pipeline_or_run_history_side_effects(
        self,
        get_engine,
        resolve,
    ):
        get_engine.return_value = Mock()
        resolve.return_value = ["WMT", "NVDA"]

        result = run_scheduled_ingestion(dry_run=True)

        self.assertEqual(result["status"], "DRY_RUN")
        self.assertEqual(result["tickers"], ["WMT", "NVDA"])
        self.assertEqual(result["stages"], list(DEFAULT_STAGES))


if __name__ == "__main__":
    unittest.main()
