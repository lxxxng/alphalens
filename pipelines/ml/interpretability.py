"""SHAP explanations, stability checks, and model-error analysis."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import xgboost as xgb

from pipelines.ml.baselines import TARGET_COLUMN
from pipelines.ml.features import build_event_feature_dataset
from pipelines.ml.xgboost_model import (
    XGBoostExperiment,
    run_xgboost_experiment,
)


DEFAULT_LOCAL_FEATURES = 5
DEFAULT_STABILITY_FEATURES = 10
ADDITIVITY_TOLERANCE = 1e-5


@dataclass
class InterpretationReport:
    """Model explanations and diagnostics derived from one locked test fold."""

    explanation: shap.Explanation
    global_importance: pd.DataFrame
    group_importance: pd.DataFrame
    stability: pd.DataFrame
    local_contributions: pd.DataFrame
    errors: pd.DataFrame
    error_slices: pd.DataFrame
    additivity: dict


def _validate_analysis_inputs(
    frame: pd.DataFrame,
    feature_columns: tuple[str, ...],
) -> None:
    """Reject incomplete test frames before asking SHAP to explain them."""

    metadata = {
        "event_key",
        "ticker",
        "event_source",
        "event_date",
        "feature_as_of_date",
        TARGET_COLUMN,
    }
    missing = (metadata | set(feature_columns)) - set(frame.columns)

    if missing:
        raise ValueError(
            "analysis frame is missing columns: "
            + ", ".join(sorted(missing))
        )

    if frame.empty:
        raise ValueError("analysis frame must not be empty.")

    if not feature_columns:
        raise ValueError("feature_columns must not be empty.")

    if frame["event_key"].duplicated().any():
        raise ValueError("analysis frame contains duplicate event keys.")


def _base_values(explanation: shap.Explanation, rows: int) -> np.ndarray:
    """Normalize SHAP's scalar or row-level expected values to one vector."""

    base = np.asarray(explanation.base_values, dtype=float)

    if base.ndim == 0:
        return np.full(rows, float(base))

    return base.reshape(rows, -1)[:, 0]


def _importance_table(
    values: np.ndarray,
    feature_columns: tuple[str, ...],
) -> pd.DataFrame:
    """Summarize average magnitude, direction, and usage for every feature."""

    importance = pd.DataFrame({
        "feature": feature_columns,
        "mean_abs_shap": np.abs(values).mean(axis=0),
        "mean_shap": values.mean(axis=0),
        "shap_std": values.std(axis=0),
        "nonzero_share": (np.abs(values) > 1e-12).mean(axis=0),
    }).sort_values(
        ["mean_abs_shap", "feature"],
        ascending=[False, True],
        kind="stable",
    ).reset_index(drop=True)
    importance["rank"] = np.arange(1, len(importance) + 1)
    return importance


def _group_importance(
    values: np.ndarray,
    frame: pd.DataFrame,
    feature_columns: tuple[str, ...],
) -> pd.DataFrame:
    """Calculate comparable SHAP rankings by event source and test half."""

    dates = pd.to_datetime(frame["feature_as_of_date"], errors="coerce")
    midpoint = dates.sort_values(kind="stable").iloc[len(dates) // 2]
    group_specs = {
        "event_source": frame["event_source"].astype(str),
        "test_half": pd.Series(
            np.where(dates <= midpoint, "early", "late"),
            index=frame.index,
        ),
    }
    rows = []

    for group_type, labels in group_specs.items():
        for group_value in sorted(labels.dropna().unique()):
            positions = np.flatnonzero(labels.to_numpy() == group_value)
            table = _importance_table(values[positions], feature_columns)
            table.insert(0, "rows", len(positions))
            table.insert(0, "group_value", group_value)
            table.insert(0, "group_type", group_type)
            rows.append(table)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _importance_stability(
    group_importance: pd.DataFrame,
    *,
    top_n: int,
) -> pd.DataFrame:
    """Compare SHAP rankings between event groups and chronological halves."""

    if top_n < 1:
        raise ValueError("top_n must be at least one.")

    rows = []

    for group_type, type_frame in group_importance.groupby("group_type"):
        groups = sorted(type_frame["group_value"].unique())

        for left, right in combinations(groups, 2):
            left_frame = type_frame[type_frame["group_value"] == left]
            right_frame = type_frame[type_frame["group_value"] == right]
            paired = left_frame[["feature", "mean_abs_shap"]].merge(
                right_frame[["feature", "mean_abs_shap"]],
                on="feature",
                suffixes=("_left", "_right"),
                validate="one_to_one",
            )
            left_top = set(
                left_frame.nsmallest(top_n, "rank")["feature"]
            )
            right_top = set(
                right_frame.nsmallest(top_n, "rank")["feature"]
            )
            union = left_top | right_top
            rows.append({
                "group_type": group_type,
                "left_group": left,
                "right_group": right,
                "features": len(paired),
                "importance_spearman": paired[
                    ["mean_abs_shap_left", "mean_abs_shap_right"]
                ].corr(method="spearman").iloc[0, 1],
                "top_n": min(top_n, len(paired)),
                "top_overlap": len(left_top & right_top),
                "top_jaccard": (
                    len(left_top & right_top) / len(union) if union else 1.0
                ),
            })

    return pd.DataFrame(rows)


def _local_contributions(
    values: np.ndarray,
    base_values: np.ndarray,
    frame: pd.DataFrame,
    feature_columns: tuple[str, ...],
    predictions: np.ndarray,
    *,
    top_n: int,
) -> pd.DataFrame:
    """Keep the largest signed contributions for each individual event."""

    if top_n < 1:
        raise ValueError("top_n must be at least one.")

    rows = []
    feature_values = frame[list(feature_columns)]

    for position, event in enumerate(frame.itertuples(index=False)):
        order = np.argsort(np.abs(values[position]))[::-1][:top_n]

        for rank, feature_index in enumerate(order, start=1):
            shap_value = float(values[position, feature_index])
            feature = feature_columns[feature_index]
            value = feature_values.iloc[position, feature_index]
            rows.append({
                "event_key": event.event_key,
                "ticker": event.ticker,
                "event_source": event.event_source,
                "feature_as_of_date": event.feature_as_of_date,
                "prediction": float(predictions[position]),
                "actual": float(getattr(event, TARGET_COLUMN)),
                "base_value": float(base_values[position]),
                "local_rank": rank,
                "feature": feature,
                "feature_value": (
                    float(value) if not pd.isna(value) else float("nan")
                ),
                "shap_value": shap_value,
                "effect": (
                    "raises_prediction"
                    if shap_value > 0
                    else "lowers_prediction" if shap_value < 0 else "neutral"
                ),
            })

    return pd.DataFrame(rows)


def _error_tables(
    frame: pd.DataFrame,
    predictions: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build event-level errors and robust source/ticker slice summaries."""

    errors = frame[[
        "event_key",
        "ticker",
        "event_source",
        "event_date",
        "feature_as_of_date",
        TARGET_COLUMN,
    ]].copy()
    errors["prediction"] = predictions
    errors["residual"] = errors[TARGET_COLUMN] - errors["prediction"]
    errors["absolute_error"] = errors["residual"].abs()
    errors["direction_correct"] = (
        (errors["prediction"] > 0) == (errors[TARGET_COLUMN] > 0)
    )
    errors = errors.sort_values(
        ["absolute_error", "event_key"],
        ascending=[False, True],
        kind="stable",
    ).reset_index(drop=True)
    slices = []

    def add_slice(slice_type: str, slice_value: str, group: pd.DataFrame):
        residual = group["residual"]
        slices.append({
            "slice_type": slice_type,
            "slice_value": slice_value,
            "rows": len(group),
            "mae": float(group["absolute_error"].mean()),
            "rmse": math.sqrt(float((residual ** 2).mean())),
            "directional_accuracy": float(group["direction_correct"].mean()),
            "mean_prediction": float(group["prediction"].mean()),
            "mean_actual": float(group[TARGET_COLUMN].mean()),
            "prediction_bias": float((-residual).mean()),
        })

    add_slice("overall", "all", errors)

    for column in ("event_source", "ticker"):
        for value, group in errors.groupby(column, sort=True):
            add_slice(column, str(value), group)

    return errors, pd.DataFrame(slices).sort_values(
        ["slice_type", "mae"],
        ascending=[True, False],
        kind="stable",
    ).reset_index(drop=True)


def analyze_xgboost_model(
    model: xgb.XGBRegressor,
    frame: pd.DataFrame,
    feature_columns: tuple[str, ...],
    *,
    top_local_features: int = DEFAULT_LOCAL_FEATURES,
    top_stability_features: int = DEFAULT_STABILITY_FEATURES,
) -> InterpretationReport:
    """Explain a locked XGBoost model and diagnose its test errors."""

    _validate_analysis_inputs(frame, feature_columns)
    analysis_frame = frame.reset_index(drop=True).copy()
    features = analysis_frame[list(feature_columns)]
    predictions = np.asarray(model.predict(features), dtype=float)
    explainer = shap.TreeExplainer(model)
    explanation = explainer(features, check_additivity=True)
    values = np.asarray(explanation.values, dtype=float)

    if values.shape != (len(analysis_frame), len(feature_columns)):
        raise ValueError(
            "SHAP returned an unexpected matrix shape: "
            f"{values.shape}."
        )

    base_values = _base_values(explanation, len(analysis_frame))
    reconstructed = base_values + values.sum(axis=1)
    absolute_difference = np.abs(reconstructed - predictions)
    additivity = {
        "rows": len(analysis_frame),
        "max_absolute_error": float(absolute_difference.max()),
        "mean_absolute_error": float(absolute_difference.mean()),
        "tolerance": ADDITIVITY_TOLERANCE,
        "passed": bool(absolute_difference.max() <= ADDITIVITY_TOLERANCE),
    }

    if not additivity["passed"]:
        raise ValueError(f"SHAP additivity validation failed: {additivity}")

    global_importance = _importance_table(values, feature_columns)
    group_importance = _group_importance(
        values,
        analysis_frame,
        feature_columns,
    )
    stability = _importance_stability(
        group_importance,
        top_n=top_stability_features,
    )
    local = _local_contributions(
        values,
        base_values,
        analysis_frame,
        feature_columns,
        predictions,
        top_n=top_local_features,
    )
    errors, error_slices = _error_tables(analysis_frame, predictions)
    return InterpretationReport(
        explanation=explanation,
        global_importance=global_importance,
        group_importance=group_importance,
        stability=stability,
        local_contributions=local,
        errors=errors,
        error_slices=error_slices,
        additivity=additivity,
    )


def analyze_experiment(
    experiment: XGBoostExperiment,
    *,
    top_local_features: int = DEFAULT_LOCAL_FEATURES,
) -> InterpretationReport:
    """Explain the final model on exactly its untouched test partition."""

    return analyze_xgboost_model(
        experiment.model,
        experiment.split.test,
        experiment.feature_columns,
        top_local_features=top_local_features,
    )


def _json_records(frame: pd.DataFrame) -> list[dict]:
    """Serialize report tables using JSON-safe pandas conversion."""

    return json.loads(frame.to_json(orient="records", date_format="iso"))


def main() -> None:
    """Generate reproducible SHAP and test-error artifacts."""

    parser = argparse.ArgumentParser(
        description="Explain AlphaLens XGBoost test predictions with SHAP.",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("data/ml"),
    )
    parser.add_argument(
        "--top-local-features",
        type=int,
        default=DEFAULT_LOCAL_FEATURES,
    )
    arguments = parser.parse_args()
    dataset = build_event_feature_dataset(horizon=30, include_topics=True)
    experiment = run_xgboost_experiment(dataset)
    report = analyze_experiment(
        experiment,
        top_local_features=arguments.top_local_features,
    )
    output = arguments.output_directory
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "global": output / "shap_global_importance.csv",
        "groups": output / "shap_group_importance.csv",
        "stability": output / "shap_stability.csv",
        "local": output / "shap_local_contributions.csv",
        "errors": output / "model_errors.csv",
        "error_slices": output / "model_error_slices.csv",
        "summary": output / "interpretability_summary.json",
    }
    report.global_importance.to_csv(paths["global"], index=False)
    report.group_importance.to_csv(paths["groups"], index=False)
    report.stability.to_csv(paths["stability"], index=False)
    report.local_contributions.to_csv(paths["local"], index=False)
    report.errors.to_csv(paths["errors"], index=False)
    report.error_slices.to_csv(paths["error_slices"], index=False)
    paths["summary"].write_text(
        json.dumps({
            "shap_version": shap.__version__,
            "additivity": report.additivity,
            "top_global_features": _json_records(
                report.global_importance.head(15)
            ),
            "stability": _json_records(report.stability),
            "error_slices": _json_records(report.error_slices),
        }, indent=2),
        encoding="utf-8",
    )

    print("\nSHAP additivity")
    print(report.additivity)
    print("\nTop global features")
    print(report.global_importance.head(15).to_string(index=False))
    print("\nImportance stability")
    print(report.stability.to_string(index=False))
    print("\nError slices")
    print(report.error_slices.head(15).to_string(index=False))
    print("\nArtifacts")
    for path in paths.values():
        print(path)


if __name__ == "__main__":
    main()
