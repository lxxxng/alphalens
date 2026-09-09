"""Tests for chart-ready market history helpers."""

import unittest
from datetime import date, timedelta

from app.services.market_context import (
    build_market_context_text,
    build_market_price_series,
    build_ticker_snapshot,
    calculate_event_reactions,
    downsample_rows,
    filter_rows_for_period,
)


def _rows(count: int) -> list[dict]:
    return [
        {
            "trading_date": date(2026, 1, 1) + timedelta(days=index),
            "close": 50 + index,
            "adjusted_close": 100 + index,
            "volume": 1_000,
        }
        for index in range(count)
    ]


class MarketHistoryTests(unittest.TestCase):
    def test_series_uses_adjusted_close_and_indexes_from_100(self):
        series = build_market_price_series("WMT", _rows(3))

        self.assertEqual(series["ticker"], "WMT")
        self.assertEqual(series["points"][0]["close"], 100.0)
        self.assertEqual(series["points"][0]["indexed_value"], 100.0)
        self.assertEqual(series["points"][-1]["indexed_value"], 102.0)

    def test_period_filter_is_inclusive(self):
        rows = _rows(5)
        filtered = filter_rows_for_period(
            rows,
            date(2026, 1, 2),
            date(2026, 1, 4),
        )

        self.assertEqual(len(filtered), 3)

    def test_downsampling_preserves_endpoints(self):
        rows = _rows(20)
        sampled = downsample_rows(rows, max_points=5)

        self.assertEqual(len(sampled), 5)
        self.assertEqual(sampled[0], rows[0])
        self.assertEqual(sampled[-1], rows[-1])

    def test_event_reactions_use_forward_trading_sessions(self):
        rows = _rows(8)
        reactions = calculate_event_reactions(
            rows,
            date(2026, 1, 2),
        )

        self.assertEqual(reactions["plot_date"], "2026-01-02")
        self.assertAlmostEqual(
            reactions["reaction_1d"],
            102 / 101 - 1,
        )
        self.assertAlmostEqual(
            reactions["reaction_5d"],
            106 / 101 - 1,
        )

    def test_event_reactions_tolerate_missing_forward_prices(self):
        reactions = calculate_event_reactions(
            _rows(3),
            date(2026, 1, 3),
        )

        self.assertIsNone(reactions["reaction_1d"])
        self.assertIsNone(reactions["reaction_5d"])

    def test_snapshot_and_prompt_include_absolute_benchmark_returns(self):
        benchmark_returns = {
            "1M": 0.01,
            "3M": 0.03,
            "1Y": 0.20,
            "5Y": 0.50,
        }
        snapshot = build_ticker_snapshot(
            "NVDA",
            _rows(400),
            benchmark_returns,
        )

        self.assertEqual(snapshot["benchmark_returns"], benchmark_returns)
        self.assertIn(
            "SPY benchmark returns: 1M 1.00%, 3M 3.00%, "
            "1Y 20.00%, 5Y 50.00%",
            build_market_context_text([snapshot]),
        )


if __name__ == "__main__":
    unittest.main()
