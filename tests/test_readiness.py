"""Tests for liveness and deployment-readiness behavior."""

import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app


class ReadinessTests(unittest.TestCase):
    def test_liveness_does_not_require_external_dependencies(self):
        response = TestClient(app).get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_readiness_returns_503_when_dependency_is_missing(self):
        result = {
            "status": "not_ready",
            "service": "alphalens",
            "checks": {"database": {"ready": False}},
        }

        with patch("app.main.check_readiness", return_value=result):
            response = TestClient(app).get("/ready")

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json(), result)

    def test_readiness_returns_200_when_dependencies_are_usable(self):
        result = {
            "status": "ready",
            "service": "alphalens",
            "checks": {
                "database": {"ready": True},
                "faiss": {"ready": True, "missing_files": []},
            },
        }

        with patch("app.main.check_readiness", return_value=result):
            response = TestClient(app).get("/ready")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), result)


if __name__ == "__main__":
    unittest.main()
