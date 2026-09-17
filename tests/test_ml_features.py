"""Tests for leakage-safe market and sentiment feature construction."""

import unittest

import pandas as pd

from pipelines.ml.features import (
    aggregate_sentiment_features,
    attach_earnings_result_features,
    attach_market_features,
    calculate_market_feature_panel,
    model_feature_columns,
)


def _price_history(periods=150):
    dates = pd.bdate_range("2025-01-02", periods=periods)
    frames = []

    for ticker, base, slope in (("WMT", 100.0, 0.4), ("SPY", 200.0, 0.2)):
        frames.append(pd.DataFrame({
            "ticker": ticker,
            "trading_date": dates,
            "adjusted_close": [base + slope * index for index in range(periods)],
            "volume": [1_000_000 + 1_000 * index for index in range(periods)],
        }))

    return pd.concat(frames, ignore_index=True)


class MarketFeatureTests(unittest.TestCase):
    def test_feature_allowlist_excludes_target_and_future_prices(self):
        columns = model_feature_columns(include_topics=True)

        self.assertNotIn("excess_return_30d", columns)
        self.assertNotIn("target_adjusted_close", columns)
        self.assertNotIn("target_trading_date", columns)

    def test_features_use_exact_anchor_session(self):
        prices = _price_history()
        panel = calculate_market_feature_panel(prices)
        anchor = prices.loc[prices["ticker"] == "WMT", "trading_date"].iloc[-1]
        events = pd.DataFrame([{
            "event_key": "earnings_call:1",
            "ticker": "WMT",
            "anchor_trading_date": anchor,
        }])

        result = attach_market_features(events, panel)

        self.assertEqual(result.loc[0, "feature_as_of_date"], anchor)
        self.assertGreater(result.loc[0, "momentum_21d"], 0)
        self.assertIn("relative_momentum_63d", result.columns)

    def test_future_price_changes_do_not_change_anchor_features(self):
        prices = _price_history()
        anchor = pd.Timestamp("2025-06-02")
        original = calculate_market_feature_panel(prices)
        changed = prices.copy()
        changed.loc[
            changed["trading_date"] > anchor,
            "adjusted_close",
        ] *= 10
        changed_panel = calculate_market_feature_panel(changed)
        feature = "momentum_63d"
        original_value = original.loc[
            (original["ticker"] == "WMT")
            & (original["trading_date"] == anchor),
            feature,
        ].iloc[0]
        changed_value = changed_panel.loc[
            (changed_panel["ticker"] == "WMT")
            & (changed_panel["trading_date"] == anchor),
            feature,
        ].iloc[0]

        self.assertAlmostEqual(original_value, changed_value)


class SentimentFeatureTests(unittest.TestCase):
    def test_token_weighted_sentiment_and_speaker_gap(self):
        items = pd.DataFrame([
            {
                "event_key": "earnings_call:1",
                "event_source": "earnings_call",
                "speaker_group": "management",
                "content": "Margins and guidance improved.",
                "status": "SCORED",
                "sentiment_score": 0.8,
                "confidence": 0.9,
                "token_count": 300,
            },
            {
                "event_key": "earnings_call:1",
                "event_source": "earnings_call",
                "speaker_group": "analyst",
                "content": "What risks could pressure margins?",
                "status": "SCORED",
                "sentiment_score": -0.4,
                "confidence": 0.7,
                "token_count": 100,
            },
        ])

        result = aggregate_sentiment_features(items, include_topics=True).iloc[0]

        self.assertAlmostEqual(result["sentiment_score"], 0.5)
        self.assertAlmostEqual(result["management_analyst_gap"], 1.2)
        self.assertAlmostEqual(result["sentiment_coverage"], 1.0)
        self.assertGreater(result["topic_margins_profitability_share"], 0)

    def test_unscored_items_reduce_coverage(self):
        items = pd.DataFrame([
            {
                "event_key": "sec_filing:a",
                "event_source": "sec_filing",
                "speaker_group": "filing",
                "content": "Demand increased.",
                "status": "SCORED",
                "sentiment_score": 0.2,
                "confidence": 0.8,
                "token_count": 50,
            },
            {
                "event_key": "sec_filing:a",
                "event_source": "sec_filing",
                "speaker_group": "filing",
                "content": "Risk discussion.",
                "status": None,
                "sentiment_score": None,
                "confidence": None,
                "token_count": None,
            },
        ])

        result = aggregate_sentiment_features(items, include_topics=False).iloc[0]

        self.assertAlmostEqual(result["sentiment_coverage"], 0.5)
        self.assertEqual(int(result["sentiment_scored_items"]), 1)


class EarningsResultFeatureTests(unittest.TestCase):
    def test_matches_calls_and_calculates_prior_surprise(self):
        events = pd.DataFrame([
            {
                "event_key": "earnings_call:1",
                "event_source": "earnings_call",
                "ticker": "WMT",
                "event_date": "2026-08-20",
                "anchor_trading_date": "2026-08-21",
            },
            {
                "event_key": "sec_filing:a",
                "event_source": "sec_filing",
                "ticker": "WMT",
                "event_date": "2026-08-21",
                "anchor_trading_date": "2026-08-24",
            },
        ])
        results = pd.DataFrame([
            {
                "ticker": "WMT",
                "earnings_date": "2026-05-21",
                "eps_estimate": 0.66,
                "reported_eps": 0.66,
                "eps_surprise": 0.0,
                "eps_surprise_pct": 0.0019,
            },
            {
                "ticker": "WMT",
                "earnings_date": "2026-08-20",
                "eps_estimate": 0.74,
                "reported_eps": 0.81,
                "eps_surprise": 0.07,
                "eps_surprise_pct": 0.0927,
            },
        ])

        matched = attach_earnings_result_features(events, results)

        self.assertAlmostEqual(matched.loc[0, "eps_surprise_pct"], 0.0927)
        self.assertAlmostEqual(
            matched.loc[0, "prior_eps_surprise_pct"],
            0.0019,
        )
        self.assertAlmostEqual(
            matched.loc[0, "eps_surprise_change"],
            0.0908,
        )
        self.assertEqual(matched.loc[0, "eps_surprise_direction"], 1.0)
        self.assertTrue(pd.isna(matched.loc[1, "reported_eps"]))

    def test_does_not_use_result_after_feature_anchor(self):
        events = pd.DataFrame([{
            "event_key": "earnings_call:1",
            "event_source": "earnings_call",
            "ticker": "WMT",
            "event_date": "2026-08-20",
            "anchor_trading_date": "2026-08-21",
        }])
        results = pd.DataFrame([{
            "ticker": "WMT",
            "earnings_date": "2026-08-22",
            "eps_estimate": 1.0,
            "reported_eps": 1.1,
            "eps_surprise": 0.1,
            "eps_surprise_pct": 0.1,
        }])

        matched = attach_earnings_result_features(events, results)

        self.assertTrue(pd.isna(matched.loc[0, "reported_eps"]))


if __name__ == "__main__":
    unittest.main()
