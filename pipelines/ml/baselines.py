"""Purged chronological splits and baseline models for AlphaLens."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pipelines.ml.features import (
    build_event_feature_dataset,
    model_feature_columns,
)


DEFAULT_VALIDATION_START = "2024-07-01"
DEFAULT_TEST_START = "2025-07-01"
DEFAULT_RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)
TARGET_COLUMN = "excess_return_30d"


@dataclass(frozen=True)
class PurgedSplit:
    """Chronological partitions plus observations removed at boundaries."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    purged: pd.DataFrame
    validation_start: pd.Timestamp
    test_start: pd.Timestamp


@dataclass
class BaselineExperiment:
    """Models, predictions, and metrics from one baseline experiment."""

    split: PurgedSplit
    feature_columns: tuple[str, ...]
    selected_alpha: float
    validation_metrics: pd.DataFrame
    test_metrics: pd.DataFrame
    validation_predictions: pd.DataFrame
    test_predictions: pd.DataFrame
    ridge_model: Pipeline


def purged_time_split(
    dataset: pd.DataFrame,
    *,
    validation_start: str | pd.Timestamp = DEFAULT_VALIDATION_START,
    test_start: str | pd.Timestamp = DEFAULT_TEST_START,
) -> PurgedSplit:
    """Split labels chronologically and remove windows crossing boundaries."""

    required = {
        "event_key",
        "feature_as_of_date",
        "target_trading_date",
        "target_available",
        TARGET_COLUMN,
    }
    if not required.issubset(dataset.columns):
        raise ValueError("dataset is missing chronological split columns.")

    validation_start = pd.Timestamp(validation_start)
    test_start = pd.Timestamp(test_start)

    if validation_start >= test_start:
        raise ValueError("validation_start must be before test_start.")

    labeled = dataset.loc[dataset["target_available"]].copy()
    labeled["feature_as_of_date"] = pd.to_datetime(
        labeled["feature_as_of_date"],
        errors="coerce",
    )
    labeled["target_trading_date"] = pd.to_datetime(
        labeled["target_trading_date"],
        errors="coerce",
    )
    labeled = labeled.dropna(subset=[
        "feature_as_of_date",
        "target_trading_date",
        TARGET_COLUMN,
    ]).sort_values(
        ["feature_as_of_date", "ticker", "event_source", "event_id"],
        kind="stable",
    )

    train_mask = (
        (labeled["feature_as_of_date"] < validation_start)
        & (labeled["target_trading_date"] < validation_start)
    )
    validation_mask = (
        (labeled["feature_as_of_date"] >= validation_start)
        & (labeled["feature_as_of_date"] < test_start)
        & (labeled["target_trading_date"] < test_start)
    )
    test_mask = labeled["feature_as_of_date"] >= test_start
    assigned = train_mask | validation_mask | test_mask
    split = PurgedSplit(
        train=labeled.loc[train_mask].copy(),
        validation=labeled.loc[validation_mask].copy(),
        test=labeled.loc[test_mask].copy(),
        purged=labeled.loc[~assigned].copy(),
        validation_start=validation_start,
        test_start=test_start,
    )
    validate_purged_split(split)
    return split


def validate_purged_split(split: PurgedSplit) -> dict[str, int]:
    """Enforce non-overlapping identities and target windows across folds."""

    partitions = (split.train, split.validation, split.test, split.purged)
    keys = [set(frame["event_key"]) for frame in partitions]
    overlap_count = sum(
        len(keys[left] & keys[right])
        for left in range(len(keys))
        for right in range(left + 1, len(keys))
    )
    checks = {
        "train_rows": len(split.train),
        "validation_rows": len(split.validation),
        "test_rows": len(split.test),
        "purged_rows": len(split.purged),
        "overlapping_event_keys": overlap_count,
        "train_targets_cross_validation": int((
            split.train["target_trading_date"] >= split.validation_start
        ).sum()),
        "validation_targets_cross_test": int((
            split.validation["target_trading_date"] >= split.test_start
        ).sum()),
    }
    failures = {
        key: value
        for key, value in checks.items()
        if key not in {
            "train_rows",
            "validation_rows",
            "test_rows",
            "purged_rows",
        } and value
    }

    if failures:
        raise ValueError(f"Purged split validation failed: {failures}")

    if min(len(split.train), len(split.validation), len(split.test)) == 0:
        raise ValueError("Each chronological split must contain observations.")

    return checks


def build_ridge_pipeline(alpha: float) -> Pipeline:
    """Create a missing-aware regularized linear regression baseline."""

    if alpha <= 0:
        raise ValueError("Ridge alpha must be positive.")

    return Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                add_indicator=True,
                keep_empty_features=True,
            ),
        ),
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=alpha)),
    ])


def evaluate_predictions(
    actual,
    predicted,
    *,
    model_name: str,
    split_name: str,
) -> dict:
    """Calculate regression, direction, and rank-quality metrics."""

    actual = pd.Series(actual, dtype=float).reset_index(drop=True)
    predicted = pd.Series(predicted, dtype=float).reset_index(drop=True)

    if len(actual) != len(predicted) or actual.empty:
        raise ValueError("actual and predicted must be equal non-empty lengths.")

    prediction_std = float(predicted.std(ddof=0))
    has_prediction_variation = prediction_std > 1e-12
    pearson = (
        float(actual.corr(predicted, method="pearson"))
        if has_prediction_variation
        else float("nan")
    )
    spearman = (
        float(actual.corr(predicted, method="spearman"))
        if has_prediction_variation
        else float("nan")
    )
    return {
        "split": split_name,
        "model": model_name,
        "rows": len(actual),
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": math.sqrt(float(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
        "directional_accuracy": float(
            ((predicted > 0) == (actual > 0)).mean()
        ),
        "pearson": pearson,
        "spearman_ic": spearman,
        "prediction_mean": float(predicted.mean()),
        "prediction_std": prediction_std,
        "actual_mean": float(actual.mean()),
    }


def _prediction_frame(
    frame: pd.DataFrame,
    predictions,
    model_name: str,
) -> pd.DataFrame:
    """Keep source identity beside predictions for later diagnostics."""

    result = frame[[
        "event_key",
        "ticker",
        "event_source",
        "event_date",
        "feature_as_of_date",
        TARGET_COLUMN,
    ]].copy()
    result["model"] = model_name
    result["prediction"] = np.asarray(predictions, dtype=float)
    result["residual"] = result[TARGET_COLUMN] - result["prediction"]
    result["direction_correct"] = (
        (result["prediction"] > 0) == (result[TARGET_COLUMN] > 0)
    )
    return result


def run_baseline_experiment(
    dataset: pd.DataFrame,
    *,
    validation_start: str | pd.Timestamp = DEFAULT_VALIDATION_START,
    test_start: str | pd.Timestamp = DEFAULT_TEST_START,
    ridge_alphas: tuple[float, ...] = DEFAULT_RIDGE_ALPHAS,
    include_topics: bool = True,
) -> BaselineExperiment:
    """Select Ridge on validation, then evaluate once on the test period."""

    if not ridge_alphas:
        raise ValueError("At least one Ridge alpha is required.")

    split = purged_time_split(
        dataset,
        validation_start=validation_start,
        test_start=test_start,
    )
    features = model_feature_columns(include_topics=include_topics)
    missing = set(features) - set(dataset.columns)

    if missing:
        raise ValueError(
            "dataset is missing model features: "
            + ", ".join(sorted(missing))
        )

    train_x = split.train[list(features)]
    train_y = split.train[TARGET_COLUMN]
    validation_x = split.validation[list(features)]
    validation_y = split.validation[TARGET_COLUMN]
    validation_rows = []
    validation_prediction_frames = []

    zero_predictions = np.zeros(len(validation_y))
    mean_predictions = np.full(len(validation_y), float(train_y.mean()))
    for model_name, predictions in (
        ("zero_excess", zero_predictions),
        ("historical_mean", mean_predictions),
    ):
        validation_rows.append(evaluate_predictions(
            validation_y,
            predictions,
            model_name=model_name,
            split_name="validation",
        ))
        validation_prediction_frames.append(_prediction_frame(
            split.validation,
            predictions,
            model_name,
        ))

    ridge_candidates = []
    for alpha in ridge_alphas:
        model = build_ridge_pipeline(float(alpha))
        model.fit(train_x, train_y)
        predictions = model.predict(validation_x)
        model_name = f"ridge_alpha_{float(alpha):g}"
        metrics = evaluate_predictions(
            validation_y,
            predictions,
            model_name=model_name,
            split_name="validation",
        )
        validation_rows.append(metrics)
        validation_prediction_frames.append(_prediction_frame(
            split.validation,
            predictions,
            model_name,
        ))
        ridge_candidates.append((metrics["mae"], metrics["rmse"], float(alpha)))

    selected_alpha = min(ridge_candidates)[2]
    development = pd.concat(
        [split.train, split.validation],
        ignore_index=True,
    ).sort_values("feature_as_of_date", kind="stable")
    ridge_model = build_ridge_pipeline(selected_alpha)
    ridge_model.fit(development[list(features)], development[TARGET_COLUMN])
    test_x = split.test[list(features)]
    test_y = split.test[TARGET_COLUMN]
    test_predictions_by_model = {
        "zero_excess": np.zeros(len(test_y)),
        "historical_mean": np.full(
            len(test_y),
            float(development[TARGET_COLUMN].mean()),
        ),
        f"ridge_alpha_{selected_alpha:g}": ridge_model.predict(test_x),
    }
    test_rows = []
    test_prediction_frames = []

    for model_name, predictions in test_predictions_by_model.items():
        test_rows.append(evaluate_predictions(
            test_y,
            predictions,
            model_name=model_name,
            split_name="test",
        ))
        test_prediction_frames.append(_prediction_frame(
            split.test,
            predictions,
            model_name,
        ))

    return BaselineExperiment(
        split=split,
        feature_columns=features,
        selected_alpha=selected_alpha,
        validation_metrics=pd.DataFrame(validation_rows).sort_values(
            ["mae", "rmse"],
            kind="stable",
        ).reset_index(drop=True),
        test_metrics=pd.DataFrame(test_rows).sort_values(
            ["mae", "rmse"],
            kind="stable",
        ).reset_index(drop=True),
        validation_predictions=pd.concat(
            validation_prediction_frames,
            ignore_index=True,
        ),
        test_predictions=pd.concat(test_prediction_frames, ignore_index=True),
        ridge_model=ridge_model,
    )


def metrics_to_json(experiment: BaselineExperiment) -> dict:
    """Serialize experiment settings and metrics without model internals."""

    checks = validate_purged_split(experiment.split)
    return {
        "target": TARGET_COLUMN,
        "validation_start": experiment.split.validation_start.date().isoformat(),
        "test_start": experiment.split.test_start.date().isoformat(),
        "selected_ridge_alpha": experiment.selected_alpha,
        "feature_count": len(experiment.feature_columns),
        "split_counts": checks,
        "validation_metrics": json.loads(
            experiment.validation_metrics.to_json(orient="records")
        ),
        "test_metrics": json.loads(
            experiment.test_metrics.to_json(orient="records")
        ),
    }


def main() -> None:
    """Run the reproducible baseline experiment from PowerShell."""

    parser = argparse.ArgumentParser(
        description="Evaluate purged chronological AlphaLens baselines.",
    )
    parser.add_argument(
        "--validation-start",
        default=DEFAULT_VALIDATION_START,
    )
    parser.add_argument("--test-start", default=DEFAULT_TEST_START)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--predictions", type=Path, default=None)
    arguments = parser.parse_args()
    dataset = build_event_feature_dataset(horizon=30, include_topics=True)
    experiment = run_baseline_experiment(
        dataset,
        validation_start=arguments.validation_start,
        test_start=arguments.test_start,
    )

    print("\nSplit counts")
    print(validate_purged_split(experiment.split))
    print(f"Selected Ridge alpha: {experiment.selected_alpha:g}")
    print("\nValidation metrics")
    print(experiment.validation_metrics.to_string(index=False))
    print("\nTest metrics")
    print(experiment.test_metrics.to_string(index=False))

    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(
            json.dumps(metrics_to_json(experiment), indent=2),
            encoding="utf-8",
        )
        print(f"\nMetrics: {arguments.output}")

    if arguments.predictions:
        arguments.predictions.parent.mkdir(parents=True, exist_ok=True)
        experiment.test_predictions.to_csv(arguments.predictions, index=False)
        print(f"Predictions: {arguments.predictions}")


if __name__ == "__main__":
    main()
