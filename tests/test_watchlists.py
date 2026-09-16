"""Tests for watchlist normalization and API behavior."""

import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app
from app.services.watchlists import (
    WatchlistNotFoundError,
    normalize_watchlist_tickers,
)


SUMMARY = {
    "watchlist_id": 7,
    "name": "Core Watchlist",
    "item_count": 1,
    "created_at": "2026-09-11T10:00:00+00:00",
    "updated_at": "2026-09-11T10:00:00+00:00",
}

DETAIL = {
    **SUMMARY,
    "items": [
        {
            "ticker": "WMT",
            "company_name": "Walmart Inc.",
            "added_at": "2026-09-11T10:00:00+00:00",
            "latest_price": 120.5,
            "latest_price_date": "2026-08-24",
            "daily_return": 0.012,
            "sentiment_label": "positive",
            "sentiment_score": 0.48,
            "sentiment_period": "2026Q4",
            "latest_event": {
                "event_type": "filing",
                "date": "2026-05-29",
                "label": "10-Q",
                "source_url": "https://www.sec.gov/example",
            },
        }
    ],
}


class WatchlistTests(unittest.TestCase):
    def test_tickers_are_normalized_and_deduplicated(self):
        self.assertEqual(
            normalize_watchlist_tickers([" wmt ", "NVDA", "wmt", ""]),
            ["WMT", "NVDA"],
        )

    def test_list_endpoint_returns_summaries(self):
        with patch(
            "app.api.watchlists.list_watchlists",
            return_value=[SUMMARY],
        ):
            response = TestClient(app).get("/api/watchlists")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["watchlists"], [SUMMARY])

    def test_create_endpoint_returns_created_watchlist(self):
        with patch(
            "app.api.watchlists.create_watchlist",
            return_value=SUMMARY,
        ) as create:
            response = TestClient(app).post(
                "/api/watchlists",
                json={"name": "Core Watchlist"},
            )

        self.assertEqual(response.status_code, 201)
        create.assert_called_once_with("Core Watchlist")

    def test_detail_endpoint_returns_local_signals(self):
        with patch(
            "app.api.watchlists.get_watchlist",
            return_value=DETAIL,
        ):
            response = TestClient(app).get("/api/watchlists/7")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["items"][0]["ticker"], "WMT")
        self.assertEqual(response.json()["items"][0]["sentiment_score"], 0.48)

    def test_add_items_forwards_all_tickers(self):
        with patch(
            "app.api.watchlists.add_watchlist_items",
            return_value=DETAIL,
        ) as add:
            response = TestClient(app).post(
                "/api/watchlists/7/items",
                json={"tickers": ["WMT", "NVDA"]},
            )

        self.assertEqual(response.status_code, 200)
        add.assert_called_once_with(7, ["WMT", "NVDA"])

    def test_missing_watchlist_returns_not_found(self):
        with patch(
            "app.api.watchlists.get_watchlist",
            side_effect=WatchlistNotFoundError(
                "Watchlist 99 was not found."
            ),
        ):
            response = TestClient(app).get("/api/watchlists/99")

        self.assertEqual(response.status_code, 404)

    def test_delete_endpoint_returns_confirmation(self):
        with patch(
            "app.api.watchlists.delete_watchlist",
            return_value=True,
        ):
            response = TestClient(app).delete("/api/watchlists/7")

        self.assertEqual(
            response.json(),
            {"watchlist_id": 7, "deleted": True},
        )


if __name__ == "__main__":
    unittest.main()
