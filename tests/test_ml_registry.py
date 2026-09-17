"""Tests for versioned model registration and integrity verification."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from pipelines.ml.features import model_feature_columns
from pipelines.ml.registry import (
    load_registered_model,
    register_model,
    verify_registered_model,
)
from pipelines.ml.xgboost_model import run_xgboost_experiment


TEST_PARAMETERS = ({
    "max_depth": 2,
    "learning_rate": 0.1,
    "min_child_weight": 1,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "reg_lambda": 1.0,
    "reg_alpha": 0.0,
},)


def _experiment():
    """Train a tiny chronological experiment suitable for registry tests."""

    features = model_feature_columns(include_topics=False)
    dates = pd.date_range("2022-01-01", periods=48, freq="MS")
    rows = []

    for index, feature_date in enumerate(dates):
        feature_values = {
            feature: ((index + offset) % 13) / 10
            for offset, feature in enumerate(features)
        }
        if index % 8 == 0:
            feature_values[features[-1]] = np.nan

        rows.append({
            "event_key": f"earnings_call:{index}",
            "ticker": "WMT" if index % 2 == 0 else "NVDA",
            "event_source": "earnings_call",
            "event_id": str(index),
            "event_date": feature_date - pd.Timedelta(days=1),
            "feature_as_of_date": feature_date,
            "target_trading_date": feature_date + pd.Timedelta(days=10),
            "target_available": True,
            "excess_return_30d": (
                0.03 * feature_values[features[0]]
                - 0.02 * feature_values[features[1]]
            ),
            **feature_values,
        })

    return run_xgboost_experiment(
        pd.DataFrame(rows),
        validation_start="2024-01-01",
        test_start="2025-01-01",
        parameter_grid=TEST_PARAMETERS,
        max_estimators=20,
        early_stopping_rounds=5,
        include_topics=False,
    )


class ModelRegistryTests(unittest.TestCase):
    def test_registers_verifies_and_loads_explicit_rejected_model(self):
        experiment = _experiment()

        with tempfile.TemporaryDirectory() as temporary:
            registry = Path(temporary)
            manifest = register_model(
                experiment,
                registry_directory=registry,
                version="test-model-001",
            )
            directory = registry / manifest["version"]
            verification = verify_registered_model(directory)
            index = json.loads(
                (registry / "registry.json").read_text(encoding="utf-8")
            )

            self.assertEqual(manifest["promotion"]["status"], "rejected")
            self.assertTrue(verification["passed"])
            self.assertEqual(verification["reference_rows"], 5)
            self.assertEqual(index["latest_version"], "test-model-001")
            self.assertIsNone(index["champion_version"])

            with self.assertRaises(PermissionError):
                load_registered_model(registry, version="latest")

            loaded = load_registered_model(
                registry,
                version="latest",
                allow_rejected=True,
            )
            self.assertEqual(loaded.version, "test-model-001")

    def test_checksum_tampering_is_detected(self):
        with tempfile.TemporaryDirectory() as temporary:
            registry = Path(temporary)
            manifest = register_model(
                _experiment(),
                registry_directory=registry,
                version="test-model-002",
            )
            directory = registry / manifest["version"]
            card = directory / "model_card.md"
            card.write_text(
                card.read_text(encoding="utf-8") + "\ntampered\n",
                encoding="utf-8",
            )

            verification = verify_registered_model(directory)

            self.assertFalse(verification["passed"])
            self.assertIn(
                "model_card: checksum mismatch",
                verification["errors"],
            )

    def test_manifest_tampering_blocks_registry_loading(self):
        with tempfile.TemporaryDirectory() as temporary:
            registry = Path(temporary)
            manifest = register_model(
                _experiment(),
                registry_directory=registry,
                version="test-model-003",
            )
            manifest_path = registry / manifest["version"] / "manifest.json"
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload["target"] = "tampered_target"
            manifest_path.write_text(
                json.dumps(payload, indent=2),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(RuntimeError, "Manifest checksum"):
                load_registered_model(
                    registry,
                    version="latest",
                    allow_rejected=True,
                )

    def test_rejects_unsafe_version_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "unsupported"):
                register_model(
                    _experiment(),
                    registry_directory=Path(temporary),
                    version="../outside",
                )


if __name__ == "__main__":
    unittest.main()
