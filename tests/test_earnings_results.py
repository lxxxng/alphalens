"""Tests for point-in-time earnings-result extraction."""

import unittest
from unittest.mock import Mock

import pandas as pd

from pipelines.earnings_results.extractor import (
    extract_earnings_results,
    normalize_earnings_dates,
)


class EarningsResultExtractorTests(unittest.TestCase):
    def test_normalizes_percent_and_excludes_future_estimate(self):
        index = pd.DatetimeIndex([
            "2026-08-20 06:00:00",
            "2026-11-19 07:00:00",
        ], tz="America/New_York", name="Earnings Date")
        raw = pd.DataFrame({
            "EPS Estimate": [0.74, 0.64],
            "Reported EPS": [0.81, float("nan")],
            "Surprise(%)": [9.27, float("nan")],
        }, index=index)

        result = normalize_earnings_dates("wmt", raw)

        self.assertEqual(len(result), 1)
        self.assertEqual(result.loc[0, "ticker"], "WMT")
        self.assertEqual(
            str(result.loc[0, "earnings_date"]),
            "2026-08-20",
        )
        self.assertAlmostEqual(result.loc[0, "eps_surprise"], 0.07)
        self.assertAlmostEqual(result.loc[0, "eps_surprise_pct"], 0.0927)
        self.assertEqual(
            result.loc[0, "earnings_timestamp"].tzname(),
            "UTC",
        )

    def test_rejects_provider_schema_drift(self):
        raw = pd.DataFrame(
            {"Reported EPS": [1.0]},
            index=pd.DatetimeIndex(["2026-01-01"], tz="UTC"),
        )

        with self.assertRaisesRegex(ValueError, "missing columns"):
            normalize_earnings_dates("WMT", raw)

    def test_extracts_normalized_scope_with_injected_client(self):
        raw = pd.DataFrame({
            "EPS Estimate": [1.0],
            "Reported EPS": [1.1],
            "Surprise(%)": [10.0],
        }, index=pd.DatetimeIndex(["2026-01-01"], tz="UTC"))
        client = Mock()
        client.get_earnings_dates.return_value = raw
        factory = Mock(return_value=client)

        result = extract_earnings_results(
            [" wmt ", "WMT"],
            limit=8,
            ticker_factory=factory,
        )

        factory.assert_called_once_with("WMT")
        client.get_earnings_dates.assert_called_once_with(limit=8)
        self.assertEqual(len(result), 1)


if __name__ == "__main__":
    unittest.main()

