"""Leakage-safe XGBoost selection and evaluation for AlphaLens."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import xgboost as xgb

from pipelines.ml.baselines import (
    DEFAULT_TEST_START,
    DEFAULT_VALIDATION_START,
    TARGET_COLUMN,
    BaselineExperiment,
    PurgedSplit,
    _prediction_frame,
    evaluate_predictions,
    run_baseline_experiment,
    validate_purged_split,
)
from pipelines.ml.features import (
    build_event_feature_dataset,
    model_feature_columns,
)


DEFAULT_MAX_ESTIMATORS = 600
DEFAULT_EARLY_STOPPING_ROUNDS = 40
DEFAULT_PARAMETER_GRID = (
    {
        "max_depth": 2,
        "learning_rate": 0.03,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 5.0,
        "reg_alpha": 0.0,
    },
    {
        "max_depth": 2,
        "learning_rate": 0.05,
        "min_child_weight": 10,
        "subsample": 0.8,
        "colsample_bytree": 1.0,
        "reg_lambda": 10.0,
        "reg_alpha": 0.0,
    },
    {
        "max_depth": 3,
        "learning_rate": 0.03,
        "min_child_weight": 10,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 10.0,
        "reg_alpha": 0.01,
    },
    {
        "max_depth": 3,
        "learning_rate": 0.05,
        "min_child_weight": 15,
        "subsample": 1.0,
        "colsample_bytree": 0.8,
        "reg_lambda": 15.0,
        "reg_alpha": 0.01,
    },
    {
        "max_depth": 4,
        "learning_rate": 0.03,
        "min_child_weight": 15,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 20.0,
        "reg_alpha": 0.05,
    },
    {
        "max_depth": 4,
        "learning_rate": 0.05,
        "min_child_weight": 20,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "reg_lambda": 20.0,
        "reg_alpha": 0.05,
    },
)


@dataclass
class XGBoostExperiment:
    """Selected model, chronological partitions, and evaluation artifacts."""

    split: PurgedSplit
    feature_columns: tuple[str, ...]
    selected_parameters: dict
    selected_boosting_rounds: int
    validation_metrics: pd.DataFrame
    test_metrics: pd.DataFrame
    validation_predictions: pd.DataFrame
    test_predictions: pd.DataFrame
    model: xgb.XGBRegressor


def build_xgboost_regressor(
    parameters: dict,
    *,
    n_estimators: int,
    early_stopping_rounds: int | None = None,
) -> xgb.XGBRegressor:
    """Construct a deterministic CPU model with conservative regularization."""

    if n_estimators < 1:
        raise ValueError("n_estimators must be at least one.")

    if early_stopping_rounds is not None and early_stopping_rounds < 1:
        raise ValueError("early_stopping_rounds must be positive when set.")

    return xgb.XGBRegressor(
        objective="reg:squarederror",
        eval_metric="mae",
        tree_method="hist",
        n_estimators=n_estimators,
        early_stopping_rounds=early_stopping_rounds,
        random_state=42,
        n_jobs=1,
        verbosity=0,
        **parameters,
    )


def _selected_rounds(model: xgb.XGBRegressor, maximum: int) -> int:
    """Return the validation-selected tree count or the configured maximum."""

    best_iteration = getattr(model, "best_iteration", None)
    return int(best_iteration + 1) if best_iteration is not None else maximum


def _baseline_metric_rows(
    baseline: BaselineExperiment,
    split_name: str,
) -> pd.DataFrame:
    """Keep the locked naive and Ridge comparisons beside tree metrics."""

    source = (
        baseline.validation_metrics
        if split_name == "validation"
        else baseline.test_metrics
    )
    return source.assign(
        candidate_id=pd.NA,
        boosting_rounds=pd.NA,
        parameters=None,
    )


def run_xgboost_experiment(
    dataset: pd.DataFrame,
    *,
    validation_start: str | pd.Timestamp = DEFAULT_VALIDATION_START,
    test_start: str | pd.Timestamp = DEFAULT_TEST_START,
    parameter_grid: tuple[dict, ...] = DEFAULT_PARAMETER_GRID,
    max_estimators: int = DEFAULT_MAX_ESTIMATORS,
    early_stopping_rounds: int = DEFAULT_EARLY_STOPPING_ROUNDS,
    include_topics: bool = True,
) -> XGBoostExperiment:
    """Tune on validation, refit once, and evaluate the untouched test fold."""

    if not parameter_grid:
        raise ValueError("parameter_grid must contain at least one candidate.")

    baseline = run_baseline_experiment(
        dataset,
        validation_start=validation_start,
        test_start=test_start,
        include_topics=include_topics,
    )
    split = baseline.split
    features = model_feature_columns(include_topics=include_topics)
    train_x = split.train[list(features)]
    train_y = split.train[TARGET_COLUMN]
    validation_x = split.validation[list(features)]
    validation_y = split.validation[TARGET_COLUMN]
    candidate_rows = []
    candidate_predictions = []
    candidates = []

    for index, parameters in enumerate(parameter_grid, start=1):
        model = build_xgboost_regressor(
            dict(parameters),
            n_estimators=max_estimators,
            early_stopping_rounds=early_stopping_rounds,
        )
        model.fit(
            train_x,
            train_y,
            eval_set=[(validation_x, validation_y)],
            verbose=False,
        )
        predictions = model.predict(validation_x)
        candidate_id = f"xgb_{index:02d}"
        rounds = _selected_rounds(model, max_estimators)
        metrics = evaluate_predictions(
            validation_y,
            predictions,
            model_name=candidate_id,
            split_name="validation",
        )
        metrics.update({
            "candidate_id": candidate_id,
            "boosting_rounds": rounds,
            "parameters": dict(parameters),
        })
        candidate_rows.append(metrics)
        candidate_predictions.append(_prediction_frame(
            split.validation,
            predictions,
            candidate_id,
        ))
        candidates.append((
            metrics["mae"],
            metrics["rmse"],
            index,
            dict(parameters),
            rounds,
        ))

    _, _, selected_index, selected_parameters, selected_rounds = min(
        candidates,
        key=lambda item: (item[0], item[1], item[2]),
    )
    development = pd.concat(
        [split.train, split.validation],
        ignore_index=True,
    ).sort_values("feature_as_of_date", kind="stable")
    final_model = build_xgboost_regressor(
        selected_parameters,
        n_estimators=selected_rounds,
        early_stopping_rounds=None,
    )
    final_model.fit(
        development[list(features)],
        development[TARGET_COLUMN],
        verbose=False,
    )
    test_predictions = final_model.predict(split.test[list(features)])
    model_name = f"xgboost_selected_{selected_index:02d}"
    test_row = evaluate_predictions(
        split.test[TARGET_COLUMN],
        test_predictions,
        model_name=model_name,
        split_name="test",
    )
    test_row.update({
        "candidate_id": f"xgb_{selected_index:02d}",
        "boosting_rounds": selected_rounds,
        "parameters": selected_parameters,
    })
    validation_metrics = pd.concat([
        _baseline_metric_rows(baseline, "validation"),
        pd.DataFrame(candidate_rows),
    ], ignore_index=True).sort_values(
        ["mae", "rmse"],
        kind="stable",
    ).reset_index(drop=True)
    test_metrics = pd.concat([
        _baseline_metric_rows(baseline, "test"),
        pd.DataFrame([test_row]),
    ], ignore_index=True).sort_values(
        ["mae", "rmse"],
        kind="stable",
    ).reset_index(drop=True)

    return XGBoostExperiment(
        split=split,
        feature_columns=features,
        selected_parameters=selected_parameters,
        selected_boosting_rounds=selected_rounds,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        validation_predictions=pd.concat([
            baseline.validation_predictions,
            *candidate_predictions,
        ], ignore_index=True),
        test_predictions=pd.concat([
            baseline.test_predictions,
            _prediction_frame(split.test, test_predictions, model_name),
        ], ignore_index=True),
        model=final_model,
    )


def metrics_to_json(experiment: XGBoostExperiment) -> dict:
    """Serialize model selection and results without binary model state."""

    return {
        "target": TARGET_COLUMN,
        "xgboost_version": xgb.__version__,
        "feature_count": len(experiment.feature_columns),
        "feature_columns": list(experiment.feature_columns),
        "validation_start": (
            experiment.split.validation_start.date().isoformat()
        ),
        "test_start": experiment.split.test_start.date().isoformat(),
        "split_counts": validate_purged_split(experiment.split),
        "selected_parameters": experiment.selected_parameters,
        "selected_boosting_rounds": experiment.selected_boosting_rounds,
        "validation_metrics": json.loads(
            experiment.validation_metrics.to_json(orient="records")
        ),
        "test_metrics": json.loads(
            experiment.test_metrics.to_json(orient="records")
        ),
    }


def main() -> None:
    """Train and evaluate the reproducible XGBoost experiment."""

    parser = argparse.ArgumentParser(
        description="Tune and evaluate AlphaLens XGBoost chronologically.",
    )
    parser.add_argument("--validation-start", default=DEFAULT_VALIDATION_START)
    parser.add_argument("--test-start", default=DEFAULT_TEST_START)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/ml/xgboost_metrics.json"),
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("data/ml/xgboost_test_predictions.csv"),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("data/ml/xgboost_model.json"),
    )
    arguments = parser.parse_args()
    dataset = build_event_feature_dataset(horizon=30, include_topics=True)
    experiment = run_xgboost_experiment(
        dataset,
        validation_start=arguments.validation_start,
        test_start=arguments.test_start,
    )

    print("\nSplit counts")
    print(validate_purged_split(experiment.split))
    print("\nSelected parameters")
    print(experiment.selected_parameters)
    print(f"Selected boosting rounds: {experiment.selected_boosting_rounds}")
    print("\nValidation metrics")
    print(experiment.validation_metrics.drop(
        columns=["parameters"],
    ).to_string(index=False))
    print("\nTest metrics")
    print(experiment.test_metrics.drop(
        columns=["parameters"],
    ).to_string(index=False))

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(metrics_to_json(experiment), indent=2),
        encoding="utf-8",
    )
    arguments.predictions.parent.mkdir(parents=True, exist_ok=True)
    experiment.test_predictions.to_csv(arguments.predictions, index=False)
    arguments.model.parent.mkdir(parents=True, exist_ok=True)
    experiment.model.save_model(arguments.model)
    print(f"\nMetrics: {arguments.output}")
    print(f"Predictions: {arguments.predictions}")
    print(f"Model: {arguments.model}")


if __name__ == "__main__":
    main()
