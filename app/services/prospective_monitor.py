"""Read-only status view over the prospective prediction ledger."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from pipelines.ml.dataset import get_database_engine, load_adjusted_prices
from pipelines.ml.prospective import (
    MINIMUM_REVIEW_EVENTS,
    SELECTED_MODEL,
    evaluate_prospective_ledger,
    load_frozen_model,
    load_prediction_ledger,
)


def _optional_float(value) -> float | None:
    if value is None:
        return None

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None

    return numeric if np.isfinite(numeric) else None


def _iso(value) -> str | None:
    if value is None or value is pd.NaT:
        return None

    if isinstance(value, (date, datetime, pd.Timestamp)):
        return value.isoformat()

    text = str(value).strip()
    return text or None


def _metric(evaluation: dict, model: str) -> dict | None:
    return next(
        (
            row
            for row in evaluation.get("forecast_metrics", [])
            if row.get("model") == model
        ),
        None,
    )


def _net_strategy(evaluation: dict) -> dict | None:
    return next(
        (
            row
            for row in evaluation.get("strategy_performance", [])
            if row.get("strategy")
            == "prospective_elasticnet_10d_long_short_net"
        ),
        None,
    )


def _recent_predictions(ledger: pd.DataFrame, limit: int) -> list[dict]:
    """Expose audit state without returning stored feature snapshots."""

    if ledger.empty:
        return []

    ordered = ledger.sort_values(
        ["prediction_generated_at", "prediction_id"],
        ascending=False,
        kind="stable",
    ).head(limit)
    rows = []

    for row in ordered.to_dict(orient="records"):
        rows.append({
            "prediction_id": int(row["prediction_id"]),
            "ticker": row["ticker"],
            "event_source": row["event_source"],
            "event_date": _iso(row.get("event_date")),
            "feature_as_of_date": _iso(row.get("feature_as_of_date")),
            "prediction_generated_at": _iso(
                row.get("prediction_generated_at")
            ),
            "predicted_excess_return": _optional_float(
                row.get("predicted_excess_return")
            ),
            "status": row["status"],
            "target_trading_date": _iso(row.get("target_trading_date")),
            "realized_excess_return": _optional_float(
                row.get("realized_excess_return")
            ),
            "outcome_error": row.get("outcome_error"),
        })

    return rows


def _status_message(evaluation: dict) -> str:
    counts = evaluation["counts"]
    matured = int(counts["matured"])
    required = int(evaluation["minimum_review_events"])

    if matured == 0:
        return (
            "Waiting for future 10-session outcomes. Predictions enter this "
            "monitor only when recorded before their return window closes."
        )

    if matured < required:
        remaining = required - matured
        return (
            f"Collecting future evidence: {remaining} more matured "
            f"event{'s' if remaining != 1 else ''} required for review."
        )

    if evaluation["status"] == "eligible_for_review":
        return (
            "Every prospective gate currently passes. Manual promotion "
            "review is permitted; no automatic promotion has occurred."
        )

    return (
        "The sample threshold is met, but one or more prospective quality "
        "gates currently fail."
    )


def get_prospective_model_status(
    *,
    engine=None,
    artifact_directory: Path | None = None,
    recent_limit: int = 8,
) -> dict:
    """Return artifact, collection progress, metrics, and recent audit rows."""

    if not 1 <= recent_limit <= 50:
        raise ValueError("recent_limit must be between 1 and 50.")

    try:
        artifact = load_frozen_model(artifact_directory)
    except FileNotFoundError as error:
        return {
            "available": False,
            "status": "unavailable",
            "message": str(error),
            "model": None,
            "counts": {"total": 0, "pending": 0, "matured": 0, "invalid": 0},
            "progress": {
                "matured": 0,
                "required": MINIMUM_REVIEW_EVENTS,
                "ratio": 0.0,
            },
            "checks": {},
            "forecast_metrics": None,
            "baseline_metrics": None,
            "net_strategy": None,
            "recent_predictions": [],
        }

    resolved_engine = engine if engine is not None else get_database_engine()
    ledger = load_prediction_ledger(
        resolved_engine,
        model_version=artifact["version"],
    )
    has_matured = bool(
        not ledger.empty and (ledger["status"] == "MATURED").any()
    )
    prices = (
        load_adjusted_prices(resolved_engine)
        if has_matured
        else pd.DataFrame()
    )
    evaluation = evaluate_prospective_ledger(
        ledger,
        prices,
        model_version=artifact["version"],
    )
    counts = evaluation["counts"]
    matured = int(counts["matured"])
    required = int(evaluation["minimum_review_events"])
    return {
        "available": True,
        "status": evaluation["status"],
        "message": _status_message(evaluation),
        "model": {
            "version": artifact["version"],
            "family": artifact["model_family"],
            "model_name": artifact["model_name"],
            "created_at": artifact["created_at"],
            "training_cutoff": artifact["training_cutoff"],
            "training_rows": int(artifact["training_rows"]),
            "horizon_sessions": int(artifact["horizon_sessions"]),
            "target": artifact["target"],
            "benchmark_ticker": artifact["benchmark_ticker"],
            "artifact_verified": True,
            "production_eligible": False,
        },
        "counts": counts,
        "progress": {
            "matured": matured,
            "required": required,
            "ratio": min(matured / required, 1.0),
        },
        "checks": evaluation.get("checks", {}),
        "forecast_metrics": _metric(evaluation, SELECTED_MODEL),
        "baseline_metrics": _metric(evaluation, "frozen_historical_mean"),
        "net_strategy": _net_strategy(evaluation),
        "recent_predictions": _recent_predictions(ledger, recent_limit),
    }
