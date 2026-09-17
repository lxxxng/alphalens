"""Tests for point-in-time modeling-data audit helpers."""

import unittest

import pandas as pd

from pipelines.ml.data_audit import (
    assess_event_horizon,
    summarize_event_horizon,
)


class EventHorizonAuditTests(unittest.TestCase):
    def setUp(self):
        self.calendar = pd.DataFrame({
            "ticker": ["WMT"] * 40,
            "trading_date": pd.bdate_range("2026-01-02", periods=40),
        })

    def test_marks_complete_forward_window(self):
        events = pd.DataFrame([{
            "source": "earnings_call",
            "event_id": "1",
            "ticker": "WMT",
            "event_date": pd.Timestamp("2026-01-03"),
        }])

        result = assess_event_horizon(events, self.calendar, horizon=30)

        self.assertEqual(
            result.loc[0, "anchor_trading_date"],
            pd.Timestamp("2026-01-05"),
        )
        self.assertTrue(bool(result.loc[0, "forward_30d_available"]))

    def test_uses_session_after_event_when_event_is_a_trading_day(self):
        events = pd.DataFrame([{
            "source": "sec_filing",
            "event_id": "filing",
            "ticker": "WMT",
            "event_date": pd.Timestamp("2026-01-05"),
        }])

        result = assess_event_horizon(events, self.calendar, horizon=5)

        self.assertEqual(
            result.loc[0, "anchor_trading_date"],
            pd.Timestamp("2026-01-06"),
        )

    def test_marks_recent_and_undated_events_unavailable(self):
        events = pd.DataFrame([
            {
                "source": "earnings_call",
                "event_id": "recent",
                "ticker": "WMT",
                "event_date": self.calendar["trading_date"].iloc[-3],
            },
            {
                "source": "earnings_call",
                "event_id": "undated",
                "ticker": "WMT",
                "event_date": pd.NaT,
            },
        ])

        result = assess_event_horizon(events, self.calendar, horizon=30)

        self.assertEqual(result["forward_30d_available"].tolist(), [False, False])

    def test_rejects_distant_first_market_date(self):
        events = pd.DataFrame([{
            "source": "earnings_call",
            "event_id": "old",
            "ticker": "WMT",
            "event_date": pd.Timestamp("2025-10-01"),
        }])

        result = assess_event_horizon(events, self.calendar, horizon=5)

        self.assertFalse(bool(result.loc[0, "forward_5d_available"]))

    def test_summary_reports_coverage(self):
        assessed = pd.DataFrame([
            {
                "source": "sec_filing",
                "event_id": "a",
                "event_date": pd.Timestamp("2026-01-02"),
                "forward_30d_available": True,
            },
            {
                "source": "sec_filing",
                "event_id": "b",
                "event_date": pd.Timestamp("2026-02-02"),
                "forward_30d_available": False,
            },
        ])

        summary = summarize_event_horizon(assessed, horizon=30)

        self.assertEqual(int(summary.loc[0, "events"]), 2)
        self.assertEqual(int(summary.loc[0, "forward_horizon_available"]), 1)
        self.assertEqual(float(summary.loc[0, "forward_horizon_coverage"]), 0.5)

    def test_rejects_invalid_horizon(self):
        events = pd.DataFrame(columns=[
            "source", "event_id", "ticker", "event_date",
        ])

        with self.assertRaisesRegex(ValueError, "at least one"):
            assess_event_horizon(events, self.calendar, horizon=0)


if __name__ == "__main__":
    unittest.main()
