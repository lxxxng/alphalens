"""Versioned, integrity-checked model artifacts for AlphaLens."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import xgboost as xgb

from pipelines.ml.backtest import BacktestResult, run_event_backtest
from pipelines.ml.baselines import TARGET_COLUMN, validate_purged_split
from pipelines.ml.dataset import get_database_engine, load_adjusted_prices
from pipelines.ml.features import build_event_feature_dataset
from pipelines.ml.interpretability import (
    InterpretationReport,
    analyze_experiment,
)
from pipelines.ml.xgboost_model import (
    XGBoostExperiment,
    metrics_to_json,
    run_xgboost_experiment,
)


REGISTRY_SCHEMA_VERSION = 1
DEFAULT_REGISTRY_DIRECTORY = Path("data/ml/registry")
REFERENCE_ROW_COUNT = 5
REFERENCE_TOLERANCE = 1e-6


@dataclass(frozen=True)
class RegisteredModel:
    """A verified registry version and its loaded native model."""

    version: str
    directory: Path
    manifest: dict
    model: xgb.XGBRegressor


def _utc_now() -> datetime:
    """Keep timestamp creation injectable through one small boundary."""

    return datetime.now(timezone.utc)


def _sha256(path: Path) -> str:
    """Calculate a streaming SHA-256 digest for one artifact."""

    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def _json_write(path: Path, payload: dict | list) -> None:
    """Write deterministic UTF-8 JSON suitable for hashing and review."""

    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


def _json_safe(value):
    """Convert numpy, pandas, and non-finite values to strict JSON types."""

    if value is None or value is pd.NA:
        return None

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None

    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()

    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]

    return value


def _records(frame: pd.DataFrame) -> list[dict]:
    """Convert a DataFrame into strict JSON-safe record dictionaries."""

    return [
        {key: _json_safe(value) for key, value in row.items()}
        for row in frame.to_dict(orient="records")
    ]


def _git_commit() -> str | None:
    """Return the source revision when registration runs inside Git."""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def dataset_fingerprint(
    frame: pd.DataFrame,
    feature_columns: tuple[str, ...],
) -> str:
    """Fingerprint ordered identities, dates, targets, and model inputs."""

    columns = [
        "event_key",
        "feature_as_of_date",
        "target_trading_date",
        TARGET_COLUMN,
        *feature_columns,
    ]
    missing = set(columns) - set(frame.columns)

    if missing:
        raise ValueError(
            "fingerprint frame is missing columns: "
            + ", ".join(sorted(missing))
        )

    ordered = frame[columns].sort_values("event_key", kind="stable").copy()

    for column in ("feature_as_of_date", "target_trading_date"):
        ordered[column] = pd.to_datetime(
            ordered[column],
            errors="coerce",
        ).astype("string")

    row_hashes = pd.util.hash_pandas_object(ordered, index=False).to_numpy()
    return hashlib.sha256(row_hashes.tobytes()).hexdigest()


def evaluate_promotion(
    experiment: XGBoostExperiment,
    backtest: BacktestResult | None = None,
) -> dict:
    """Apply fixed predictive and economic gates without test-set tuning."""

    metrics = experiment.test_metrics
    model_row = metrics.loc[
        metrics["model"].str.startswith("xgboost_selected_")
    ]
    baseline_row = metrics.loc[metrics["model"] == "historical_mean"]

    if len(model_row) != 1 or len(baseline_row) != 1:
        raise ValueError("Expected one selected XGBoost and historical baseline.")

    model_mae = float(model_row.iloc[0]["mae"])
    baseline_mae = float(baseline_row.iloc[0]["mae"])
    mae_improvement = baseline_mae - model_mae
    predictive_passed = mae_improvement > 0
    economic_passed = None
    net_sharpe = None

    if backtest is not None:
        strategy = backtest.summary.loc[
            backtest.summary["strategy"] == "xgboost_long_short_net"
        ]

        if len(strategy) != 1:
            raise ValueError("Backtest is missing xgboost_long_short_net.")

        net_sharpe = float(strategy.iloc[0]["sharpe"])
        economic_passed = bool(np.isfinite(net_sharpe) and net_sharpe > 0)

    passed = predictive_passed and economic_passed is True
    reasons = []

    if not predictive_passed:
        reasons.append(
            "Selected XGBoost test MAE did not beat historical_mean."
        )

    if economic_passed is False:
        reasons.append("Net long-short test Sharpe was not positive.")

    if economic_passed is None:
        reasons.append("No transaction-cost-aware backtest was supplied.")

    return {
        "status": "champion" if passed else "rejected",
        "passed": passed,
        "checks": {
            "test_mae_improvement": {
                "passed": predictive_passed,
                "model_mae": model_mae,
                "historical_mean_mae": baseline_mae,
                "absolute_improvement": mae_improvement,
                "rule": "model_mae < historical_mean_mae",
            },
            "net_long_short_sharpe": {
                "passed": economic_passed,
                "value": net_sharpe,
                "rule": "sharpe > 0 after transaction costs",
            },
        },
        "reasons": reasons,
    }


def _feature_schema(
    experiment: XGBoostExperiment,
) -> dict:
    """Capture feature order, dtypes, and missingness used at fit time."""

    development = pd.concat([
        experiment.split.train,
        experiment.split.validation,
    ], ignore_index=True)
    return {
        "schema_version": 1,
        "target": TARGET_COLUMN,
        "feature_count": len(experiment.feature_columns),
        "features": [
            {
                "position": position,
                "name": feature,
                "dtype": str(development[feature].dtype),
                "development_missing_rate": float(
                    development[feature].isna().mean()
                ),
                "test_missing_rate": float(
                    experiment.split.test[feature].isna().mean()
                ),
            }
            for position, feature in enumerate(experiment.feature_columns)
        ],
    }


def _reference_rows(
    experiment: XGBoostExperiment,
) -> tuple[pd.DataFrame, list[dict]]:
    """Select stable test examples used to verify future model loading."""

    ordered = experiment.split.test.sort_values(
        ["feature_as_of_date", "ticker", "event_key"],
        kind="stable",
    )
    count = min(REFERENCE_ROW_COUNT, len(ordered))
    positions = np.linspace(0, len(ordered) - 1, count, dtype=int)
    selected = ordered.iloc[positions]
    features = selected[["event_key", *experiment.feature_columns]].copy()
    predictions = experiment.model.predict(
        selected[list(experiment.feature_columns)]
    )
    expected = [
        {
            "event_key": event_key,
            "prediction": float(prediction),
        }
        for event_key, prediction in zip(
            selected["event_key"],
            predictions,
            strict=True,
        )
    ]
    return features, expected


def _model_card(
    version: str,
    experiment: XGBoostExperiment,
    promotion: dict,
    backtest: BacktestResult | None,
) -> str:
    """Create a concise human-readable record of intended and unsafe uses."""

    model_row = experiment.test_metrics.loc[
        experiment.test_metrics["model"].str.startswith("xgboost_selected_")
    ].iloc[0]
    lines = [
        f"# AlphaLens Model Card: {version}",
        "",
        "## Purpose",
        "",
        "Predict 30-trading-session stock excess return versus SPY after an ",
        "earnings call or SEC 10-K/10-Q event.",
        "",
        "## Status",
        "",
        f"**{promotion['status'].upper()}**",
        "",
        "## Test Metrics",
        "",
        f"- MAE: {float(model_row['mae']):.6f}",
        f"- RMSE: {float(model_row['rmse']):.6f}",
        f"- Directional accuracy: {float(model_row['directional_accuracy']):.3%}",
        f"- Spearman IC: {float(model_row['spearman_ic']):.6f}",
    ]

    if backtest is not None:
        strategy = backtest.summary.loc[
            backtest.summary["strategy"] == "xgboost_long_short_net"
        ].iloc[0]
        lines.extend([
            "",
            "## Net Backtest",
            "",
            f"- Total return: {float(strategy['total_return']):.3%}",
            f"- Sharpe: {float(strategy['sharpe']):.4f}",
            f"- Maximum drawdown: {float(strategy['max_drawdown']):.3%}",
        ])

    lines.extend([
        "",
        "## Limitations",
        "",
        "- The universe contains only 20 large US equities.",
        "- The untouched test period is approximately one year.",
        "- SHAP explanations describe model behavior, not causality.",
        "- Rejected models must not be served as production predictions.",
        "- This research artifact is not investment advice.",
        "",
    ])
    return "\n".join(lines)


def _registry_index(path: Path) -> dict:
    """Read the mutable registry index or return its initial structure."""

    if not path.exists():
        return {
            "schema_version": REGISTRY_SCHEMA_VERSION,
            "latest_version": None,
            "champion_version": None,
            "models": [],
        }

    return json.loads(path.read_text(encoding="utf-8"))


def _update_registry_index(
    registry_directory: Path,
    manifest: dict,
    manifest_hash: str,
) -> None:
    """Atomically publish the new version pointer after artifacts are durable."""

    index_path = registry_directory / "registry.json"
    index = _registry_index(index_path)
    entry = {
        "version": manifest["version"],
        "created_at": manifest["created_at"],
        "status": manifest["promotion"]["status"],
        "manifest_sha256": manifest_hash,
    }
    index["models"] = [
        item
        for item in index.get("models", [])
        if item.get("version") != manifest["version"]
    ] + [entry]
    index["latest_version"] = manifest["version"]

    if entry["status"] == "champion":
        index["champion_version"] = manifest["version"]

    index["updated_at"] = _utc_now().isoformat()
    temporary = registry_directory / f".registry-{uuid.uuid4().hex}.tmp"
    _json_write(temporary, index)
    os.replace(temporary, index_path)


def register_model(
    experiment: XGBoostExperiment,
    *,
    registry_directory: Path = DEFAULT_REGISTRY_DIRECTORY,
    backtest: BacktestResult | None = None,
    interpretation: InterpretationReport | None = None,
    version: str | None = None,
) -> dict:
    """Write one immutable model package and atomically publish its index."""

    validate_purged_split(experiment.split)
    development = pd.concat([
        experiment.split.train,
        experiment.split.validation,
    ], ignore_index=True)
    fingerprints = {
        "development": dataset_fingerprint(
            development,
            experiment.feature_columns,
        ),
        "test": dataset_fingerprint(
            experiment.split.test,
            experiment.feature_columns,
        ),
    }
    created_at = _utc_now()
    resolved_version = version or (
        f"xgboost-{created_at.strftime('%Y%m%dT%H%M%SZ')}-"
        f"{fingerprints['development'][:8]}"
    )

    if (
        not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", resolved_version)
        or resolved_version in {".", ".."}
    ):
        raise ValueError("version contains unsupported path characters.")

    registry_directory = Path(registry_directory)
    destination = registry_directory / resolved_version

    if destination.exists():
        raise FileExistsError(f"Registry version already exists: {destination}")

    registry_directory.mkdir(parents=True, exist_ok=True)
    temporary = registry_directory / f".tmp-{resolved_version}-{uuid.uuid4().hex}"
    temporary.mkdir()
    published = False

    try:
        model_path = temporary / "model.json"
        metrics_path = temporary / "metrics.json"
        schema_path = temporary / "feature_schema.json"
        references_path = temporary / "reference_features.csv"
        predictions_path = temporary / "reference_predictions.json"
        card_path = temporary / "model_card.md"
        experiment.model.save_model(model_path)
        metrics = _json_safe(metrics_to_json(experiment))
        _json_write(metrics_path, metrics)
        _json_write(schema_path, _feature_schema(experiment))
        references, expected_predictions = _reference_rows(experiment)
        references.to_csv(references_path, index=False)
        _json_write(predictions_path, expected_predictions)
        promotion = evaluate_promotion(experiment, backtest)
        card_path.write_text(
            _model_card(resolved_version, experiment, promotion, backtest),
            encoding="utf-8",
        )
        artifact_paths = {
            "model": model_path,
            "metrics": metrics_path,
            "feature_schema": schema_path,
            "reference_features": references_path,
            "reference_predictions": predictions_path,
            "model_card": card_path,
        }
        manifest = {
            "schema_version": REGISTRY_SCHEMA_VERSION,
            "version": resolved_version,
            "created_at": created_at.isoformat(),
            "git_commit": _git_commit(),
            "model_family": "xgboost",
            "library_versions": {
                "xgboost": xgb.__version__,
                "shap": shap.__version__,
                "pandas": pd.__version__,
                "numpy": np.__version__,
            },
            "target": TARGET_COLUMN,
            "selected_parameters": experiment.selected_parameters,
            "selected_boosting_rounds": experiment.selected_boosting_rounds,
            "feature_count": len(experiment.feature_columns),
            "feature_columns": list(experiment.feature_columns),
            "split_counts": validate_purged_split(experiment.split),
            "training_window": {
                "first_feature_date": _json_safe(
                    development["feature_as_of_date"].min()
                ),
                "last_feature_date": _json_safe(
                    development["feature_as_of_date"].max()
                ),
                "validation_start": _json_safe(
                    experiment.split.validation_start
                ),
                "test_start": _json_safe(experiment.split.test_start),
            },
            "dataset_fingerprints": fingerprints,
            "promotion": promotion,
            "backtest": (
                _records(backtest.summary) if backtest is not None else None
            ),
            "interpretability": (
                {
                    "additivity": interpretation.additivity,
                    "top_features": _records(
                        interpretation.global_importance.head(15)
                    ),
                    "stability": _records(interpretation.stability),
                }
                if interpretation is not None
                else None
            ),
            "artifacts": {
                name: {
                    "file": path.name,
                    "sha256": _sha256(path),
                    "bytes": path.stat().st_size,
                }
                for name, path in artifact_paths.items()
            },
        }
        manifest_path = temporary / "manifest.json"
        _json_write(manifest_path, _json_safe(manifest))
        temporary_verification = verify_registered_model(temporary)

        if not temporary_verification["passed"]:
            raise RuntimeError(
                "Temporary model package failed verification: "
                f"{temporary_verification}"
            )

        os.replace(temporary, destination)
        published = True
        _update_registry_index(
            registry_directory,
            manifest,
            _sha256(destination / "manifest.json"),
        )
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)

        if published:
            shutil.rmtree(destination, ignore_errors=True)

        raise

    verification = verify_registered_model(destination)

    if not verification["passed"]:
        raise RuntimeError(f"Registered model verification failed: {verification}")

    return manifest


def verify_registered_model(version_directory: Path) -> dict:
    """Verify checksums, feature order, and saved reference predictions."""

    directory = Path(version_directory)
    manifest_path = directory / "manifest.json"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing registry manifest: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors = []

    for name, metadata in manifest.get("artifacts", {}).items():
        path = directory / metadata["file"]

        if not path.exists():
            errors.append(f"{name}: missing {metadata['file']}")
        elif _sha256(path) != metadata["sha256"]:
            errors.append(f"{name}: checksum mismatch")

    max_difference = None
    reference_rows = 0

    try:
        model = xgb.XGBRegressor()
        model.load_model(directory / manifest["artifacts"]["model"]["file"])
        expected_features = manifest["feature_columns"]
        model_features = model.get_booster().feature_names or []

        if model_features != expected_features:
            errors.append("model feature order does not match manifest")

        references = pd.read_csv(
            directory / manifest["artifacts"]["reference_features"]["file"]
        )
        reference_rows = len(references)
        expected = json.loads((
            directory
            / manifest["artifacts"]["reference_predictions"]["file"]
        ).read_text(encoding="utf-8"))
        actual = model.predict(references[expected_features])
        expected_values = np.asarray(
            [row["prediction"] for row in expected],
            dtype=float,
        )
        max_difference = float(np.max(np.abs(actual - expected_values)))

        if max_difference > REFERENCE_TOLERANCE:
            errors.append(
                "reference prediction mismatch: "
                f"{max_difference:.8g} > {REFERENCE_TOLERANCE}"
            )
    except Exception as error:
        errors.append(f"model/reference validation failed: {error}")

    return {
        "passed": not errors,
        "version": manifest.get("version"),
        "status": manifest.get("promotion", {}).get("status"),
        "artifacts_checked": len(manifest.get("artifacts", {})),
        "reference_rows": reference_rows,
        "max_reference_difference": max_difference,
        "errors": errors,
    }


def load_registered_model(
    registry_directory: Path = DEFAULT_REGISTRY_DIRECTORY,
    *,
    version: str = "champion",
    allow_rejected: bool = False,
) -> RegisteredModel:
    """Resolve, verify, and load a registry model under promotion policy."""

    registry_directory = Path(registry_directory)
    index = _registry_index(registry_directory / "registry.json")

    if version in {"latest", "champion"}:
        resolved = index.get(f"{version}_version")

        if not resolved:
            raise LookupError(f"Registry has no {version} model.")
    else:
        resolved = version

    directory = registry_directory / resolved
    entry = next(
        (
            item
            for item in index.get("models", [])
            if item.get("version") == resolved
        ),
        None,
    )

    if entry is None:
        raise LookupError(f"Model {resolved} is not present in registry index.")

    manifest_path = directory / "manifest.json"

    if _sha256(manifest_path) != entry.get("manifest_sha256"):
        raise RuntimeError(f"Manifest checksum mismatch for model {resolved}.")

    verification = verify_registered_model(directory)

    if not verification["passed"]:
        raise RuntimeError(f"Registry verification failed: {verification}")

    manifest = json.loads(
        (directory / "manifest.json").read_text(encoding="utf-8")
    )
    status = manifest["promotion"]["status"]

    if status != "champion" and not allow_rejected:
        raise PermissionError(
            f"Model {resolved} has status {status}; deployment requires champion."
        )

    model = xgb.XGBRegressor()
    model.load_model(directory / manifest["artifacts"]["model"]["file"])
    return RegisteredModel(
        version=resolved,
        directory=directory,
        manifest=manifest,
        model=model,
    )


def main() -> None:
    """Train, evaluate, explain, backtest, and register one model version."""

    parser = argparse.ArgumentParser(
        description="Create an integrity-checked AlphaLens model package.",
    )
    parser.add_argument(
        "--registry-directory",
        type=Path,
        default=DEFAULT_REGISTRY_DIRECTORY,
    )
    parser.add_argument("--version", default=None)
    arguments = parser.parse_args()
    engine = get_database_engine()
    dataset = build_event_feature_dataset(engine, horizon=30)
    experiment = run_xgboost_experiment(dataset)
    model_name = next(
        name
        for name in experiment.test_predictions["model"].unique()
        if name.startswith("xgboost_selected_")
    )
    predictions = experiment.test_predictions.loc[
        experiment.test_predictions["model"] == model_name
    ]
    backtest = run_event_backtest(
        predictions,
        load_adjusted_prices(engine),
    )
    interpretation = analyze_experiment(experiment)
    manifest = register_model(
        experiment,
        registry_directory=arguments.registry_directory,
        backtest=backtest,
        interpretation=interpretation,
        version=arguments.version,
    )
    directory = arguments.registry_directory / manifest["version"]
    verification = verify_registered_model(directory)

    print("\nRegistered model")
    print({
        "version": manifest["version"],
        "status": manifest["promotion"]["status"],
        "directory": str(directory),
    })
    print("\nPromotion")
    print(json.dumps(manifest["promotion"], indent=2))
    print("\nVerification")
    print(verification)


if __name__ == "__main__":
    main()
