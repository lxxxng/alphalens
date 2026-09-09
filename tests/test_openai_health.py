"""Tests for the OpenAI connection diagnostic."""

import os
import unittest
from unittest.mock import patch

from app.services.openai_health import check_openai_connection


class _WorkingModels:
    def list(self):
        return []


class _WorkingClient:
    def __init__(self, **kwargs):
        self.models = _WorkingModels()


class OpenAIHealthTests(unittest.TestCase):
    def test_missing_api_key_is_reported_without_a_request(self):
        with patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "YOUR_OPENAI_API_KEY"},
        ):
            result = check_openai_connection()

        self.assertEqual(result["status"], "not_configured")
        self.assertFalse(result["configured"])
        self.assertFalse(result["reachable"])
        self.assertEqual(result["error_type"], "missing_api_key")

    def test_working_client_is_reported_as_connected(self):
        with patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "test-key"},
        ):
            result = check_openai_connection(
                client_factory=_WorkingClient,
            )

        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["configured"])
        self.assertTrue(result["reachable"])
        self.assertIsNone(result["error_type"])


if __name__ == "__main__":
    unittest.main()
