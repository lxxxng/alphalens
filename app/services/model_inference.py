"""Read model-registry status and serve point-in-time research predictions."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
from pathlib import Path

import numpy as np


DEFAULT_REGISTRY_DIRECTORY = Path("data/ml/registry")
MODEL_VERSION_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


def model_registry_directory() -> Path:
    """Resolve the model registry while keeping local development zero-config."""

    return Path(
        os.getenv(
            "ALPHALENS_MODEL_REGISTRY_DIRECTORY",
            str(DEFAULT_REGISTRY_DIRECTORY),
        )
    )


def _read_json(path: Path) -> dict:
    """Read one registry JSON object with a useful malformed-file error."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise RuntimeError(f"Malformed model registry file: {path.name}") from error

    if not isinstance(payload, dict):
        raise RuntimeError(f"Model registry file is not an object: {path.name}")

    return payload


def _sha256(path: Path) -> str:
    """Hash a manifest without importing the native model runtime."""

    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def _version_directory(registry: Path, version: str) -> Path:
    """Resolve a registry version without accepting path traversal."""

    if not MODEL_VERSION_PATTERN.fullmatch(version) or version in {".", ".."}:
        raise RuntimeError("Model registry contains an unsafe version identifier.")

    directory = (registry / version).resolve()
    root = registry.resolve()

    if directory.parent != root:
        raise RuntimeError("Model version resolves outside the registry.")

    return directory


def _net_backtest(manifest: dict) -> dict | None:
    """Extract the deployment gate's transaction-cost-aware result."""

    return next(
        (
            row
            for row in manifest.get("backtest") or []
            if row.get("strategy") == "xgboost_long_short_net"
        ),
        None,
    )


def _model_summary(manifest: dict, *, manifest_verified: bool) -> dict:
    """Return stable public metadata instead of the full internal manifest."""

    promotion = manifest.get("promotion") or {}
    interpretation = manifest.get("interpretability") or {}
    return {
        "version": manifest.get("version"),
        "family": manifest.get("model_family"),
        "status": promotion.get("status", "unknown"),
        "created_at": manifest.get("created_at"),
        "target": manifest.get("target"),
        "feature_count": manifest.get("feature_count"),
        "training_window": manifest.get("training_window"),
        "promotion_passed": bool(promotion.get("passed")),
        "promotion_checks": promotion.get("checks") or {},
        "rejection_reasons": promotion.get("reasons") or [],
        "net_backtest": _net_backtest(manifest),
        "top_features": (interpretation.get("top_features") or [])[:5],
        "manifest_verified": manifest_verified,
    }


def get_model_status(registry_directory: Path | None = None) -> dict:
    """Describe serving readiness without requiring XGBoost to be installed."""

    registry = Path(registry_directory or model_registry_directory())
    index_path = registry / "registry.json"
    runtime_available = importlib.util.find_spec("xgboost") is not None

    if not index_path.exists():
        return {
            "registry_available": False,
            "inference_runtime_available": runtime_available,
            "serving_status": "unavailable",
            "message": "No registered model was found.",
            "latest_model": None,
            "champion_version": None,
        }

    try:
        index = _read_json(index_path)
        latest_version = index.get("latest_version")

        if not latest_version:
            return {
                "registry_available": True,
                "inference_runtime_available": runtime_available,
                "serving_status": "unavailable",
                "message": "The model registry does not contain a model version.",
                "latest_model": None,
                "champion_version": index.get("champion_version"),
            }

        entry = next(
            (
                item
                for item in index.get("models", [])
                if item.get("version") == latest_version
            ),
            None,
        )

        if entry is None:
            raise RuntimeError("Latest model is missing from the registry index.")

        directory = _version_directory(registry, latest_version)
        manifest_path = directory / "manifest.json"
        manifest = _read_json(manifest_path)
        manifest_verified = _sha256(manifest_path) == entry.get(
            "manifest_sha256"
        )

        if not manifest_verified:
            raise RuntimeError("Latest model manifest failed its checksum check.")

        champion = index.get("champion_version")
        ready = bool(champion and runtime_available)
        return {
            "registry_available": True,
            "inference_runtime_available": runtime_available,
            "serving_status": "ready" if ready else "blocked",
            "message": (
                "Champion model is ready for production inference."
                if ready
                else (
                    "The latest model is research-only and cannot serve "
                    "production predictions."
                    if not champion
                    else "Install the inference runtime to serve the champion."
                )
            ),
            "latest_model": _model_summary(
                manifest,
                manifest_verified=manifest_verified,
            ),
            "champion_version": champion,
        }
    except (OSError, RuntimeError) as error:
        return {
            "registry_available": True,
            "inference_runtime_available": runtime_available,
            "serving_status": "error",
            "message": str(error),
            "latest_model": None,
            "champion_version": None,
        }


def normalize_prediction_tickers(tickers: list[str]) -> list[str]:
    """Normalize and bound the same small comparison universe used by the UI."""

    normalized = []

    for raw_ticker in tickers:
        ticker = raw_ticker.strip().upper()

        if not ticker:
            continue

        if not re.fullmatch(r"[A-Z0-9.-]{1,20}", ticker):
            raise ValueError(f"Unsupported ticker: {raw_ticker}")

        if ticker not in normalized:
            normalized.append(ticker)

    if not normalized:
        raise ValueError("At least one ticker is required.")

    if len(normalized) > 4:
        raise ValueError("At most four tickers can be compared.")

    return normalized


def _optional_float(value) -> float | None:
    """Convert pandas/numpy values to strict JSON numbers."""

    if value is None:
        return None

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None

    return numeric if np.isfinite(numeric) else None


def _optional_text(value) -> str | None:
    """Normalize nullable dataframe labels for strict API serialization."""

    if value is None:
        return None

    try:
        if bool(np.asarray(value != value).item()):
            return None
    except (TypeError, ValueError):
        return None

    text = str(value).strip()
    return text or None


def predict_latest_events(
    tickers: list[str],
    *,
    event_source: str = "latest",
    research_preview: bool = False,
    registry_directory: Path | None = None,
) -> dict:
    """Score each ticker's latest event under explicit promotion policy."""

    if event_source not in {"latest", "earnings_call", "sec_filing"}:
        raise ValueError("Unsupported event source.")

    normalized_tickers = normalize_prediction_tickers(tickers)
    registry = Path(registry_directory or model_registry_directory())

    # Native model imports stay inside the serving path so status and the rest
    # of FastAPI remain usable in lightweight environments.
    from pipelines.ml.features import build_event_feature_dataset
    from pipelines.ml.registry import load_registered_model

    loaded = load_registered_model(
        registry,
        version="latest" if research_preview else "champion",
        allow_rejected=research_preview,
    )
    feature_columns = loaded.manifest.get("feature_columns") or []

    if not feature_columns:
        raise RuntimeError("Registered model does not declare feature columns.")

    include_topics = any(
        column.startswith("topic_") for column in feature_columns
    )
    dataset = build_event_feature_dataset(include_topics=include_topics)
    missing_features = set(feature_columns) - set(dataset.columns)

    if missing_features:
        raise RuntimeError(
            "Current feature dataset is incompatible with the registered model: "
            + ", ".join(sorted(missing_features))
        )

    candidates = dataset.loc[
        dataset["ticker"].isin(normalized_tickers)
        & dataset["feature_as_of_date"].notna()
    ].copy()

    if event_source != "latest":
        candidates = candidates.loc[candidates["event_source"] == event_source]

    candidates = candidates.sort_values(
        ["ticker", "event_date", "feature_as_of_date", "event_key"],
        kind="stable",
    )
    selected = candidates.groupby("ticker", sort=False).tail(1).copy()
    selected = selected.set_index("ticker").reindex(normalized_tickers).dropna(
        subset=["event_key"]
    ).reset_index()
    selected["prediction"] = loaded.model.predict(selected[feature_columns])
    observed_latest = (
        dataset.loc[dataset["ticker"].isin(normalized_tickers)]
        .groupby("ticker")["event_date"]
        .max()
    )
    target_column = loaded.manifest.get("target", "excess_return_30d")
    predictions = []

    for row in selected.to_dict(orient="records"):
        prediction = float(row["prediction"])
        event_date = row.get("event_date")
        latest_date = observed_latest.get(row["ticker"])
        predictions.append({
            "ticker": row["ticker"],
            "event_key": row["event_key"],
            "event_source": row["event_source"],
            "event_date": (
                event_date.date().isoformat()
                if hasattr(event_date, "date")
                else str(event_date)
            ),
            "feature_as_of_date": row["feature_as_of_date"].date().isoformat(),
            "fiscal_period": _optional_text(row.get("fiscal_period")),
            "form_type": _optional_text(row.get("form_type")),
            "predicted_excess_return_30d": prediction,
            "direction": "positive" if prediction >= 0 else "negative",
            "target_available": bool(row.get("target_available", False)),
            "realized_excess_return_30d": _optional_float(
                row.get(target_column)
            ),
            "is_latest_observed_event": bool(
                event_date == latest_date
            ),
        })

    predicted_tickers = {row["ticker"] for row in predictions}
    skipped = [
        {
            "ticker": ticker,
            "reason": "No scorable event with point-in-time market features.",
        }
        for ticker in normalized_tickers
        if ticker not in predicted_tickers
    ]

    if not predictions:
        raise LookupError("No scorable events were found for the requested tickers.")

    return {
        "mode": "research_preview" if research_preview else "production",
        "model": _model_summary(
            loaded.manifest,
            manifest_verified=True,
        ),
        "target_description": (
            "30-trading-session stock excess return versus SPY after the event"
        ),
        "event_source": event_source,
        "predictions": predictions,
        "skipped": skipped,
        "disclaimer": (
            "Experimental research output, not investment advice. "
            "Research-preview models have not passed deployment gates."
        ),
    }
