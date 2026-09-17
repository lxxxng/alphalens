"""Expanding-window model comparison without reusing the inspected test set."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pipelines.ml.baselines import (
    TARGET_COLUMN,
    _prediction_frame,
    evaluate_predictions,
)
from pipelines.ml.features import (
    build_event_feature_dataset,
    model_feature_columns,
)
from pipelines.ml.xgboost_model import build_xgboost_regressor


# Every validation window ends before the previously inspected July 2025 test
# boundary. Future observations are deliberately excluded from model selection.
DEFAULT_FOLD_WINDOWS = (
    ("2023-07-01", "2024-01-01"),
    ("2024-01-01", "2024-07-01"),
    ("2024-07-01", "2025-01-01"),
    ("2025-01-01", "2025-07-01"),
)


@dataclass(frozen=True)
class CandidateSpec:
    """One predeclared model family and immutable parameter configuration."""

    name: str
    family: str
    parameters: dict


@dataclass(frozen=True)
class WalkForwardFold:
    """An expanding training window and its purged validation window."""

    name: str
    train: pd.DataFrame
    validation: pd.DataFrame
    purged: pd.DataFrame
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp


@dataclass
class BenchmarkResult:
    """Fold-level evidence and an explicitly non-production recommendation."""

    folds: tuple[WalkForwardFold, ...]
    candidates: tuple[CandidateSpec, ...]
    feature_columns: tuple[str, ...]
    fold_metrics: pd.DataFrame
    summary: pd.DataFrame
    oof_predictions: pd.DataFrame
    selection: dict


DEFAULT_CANDIDATES = (
    CandidateSpec("zero_excess", "baseline_zero", {}),
    CandidateSpec("historical_mean", "baseline_mean", {}),
    CandidateSpec("ridge_alpha_1", "ridge", {"alpha": 1.0}),
    CandidateSpec("ridge_alpha_10", "ridge", {"alpha": 10.0}),
    CandidateSpec("ridge_alpha_100", "ridge", {"alpha": 100.0}),
    CandidateSpec(
        "elasticnet_a001_l10",
        "elasticnet",
        {"alpha": 0.001, "l1_ratio": 0.1},
    ),
    CandidateSpec(
        "elasticnet_a01_l50",
        "elasticnet",
        {"alpha": 0.01, "l1_ratio": 0.5},
    ),
    CandidateSpec(
        "extra_trees_depth4",
        "extra_trees",
        {
            "n_estimators": 300,
            "max_depth": 4,
            "min_samples_leaf": 8,
            "max_features": 0.7,
        },
    ),
    CandidateSpec(
        "extra_trees_depth6",
        "extra_trees",
        {
            "n_estimators": 300,
            "max_depth": 6,
            "min_samples_leaf": 5,
            "max_features": 0.7,
        },
    ),
    CandidateSpec(
        "xgboost_registered_shape",
        "xgboost",
        {
            "n_estimators": 69,
            "max_depth": 2,
            "learning_rate": 0.03,
            "min_child_weight": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 5.0,
            "reg_alpha": 0.0,
        },
    ),
    CandidateSpec(
        "xgboost_conservative",
        "xgboost",
        {
            "n_estimators": 150,
            "max_depth": 2,
            "learning_rate": 0.02,
            "min_child_weight": 10,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 10.0,
            "reg_alpha": 0.01,
        },
    ),
    CandidateSpec(
        "catboost_conservative",
        "catboost",
        {
            "iterations": 250,
            "depth": 4,
            "learning_rate": 0.03,
            "l2_leaf_reg": 10.0,
        },
    ),
)


def _labeled_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    """Normalize the chronology used by every walk-forward fold."""

    required = {
        "event_key",
        "ticker",
        "event_source",
        "event_id",
        "feature_as_of_date",
        "target_trading_date",
        "target_available",
        TARGET_COLUMN,
    }
    missing = required - set(dataset.columns)

    if missing:
        raise ValueError(
            "dataset is missing benchmark columns: "
            + ", ".join(sorted(missing))
        )

    labeled = dataset.loc[dataset["target_available"]].copy()
    labeled["feature_as_of_date"] = pd.to_datetime(
        labeled["feature_as_of_date"],
        errors="coerce",
    )
    labeled["target_trading_date"] = pd.to_datetime(
        labeled["target_trading_date"],
        errors="coerce",
    )
    return labeled.dropna(subset=[
        "feature_as_of_date",
        "target_trading_date",
        TARGET_COLUMN,
    ]).sort_values(
        ["feature_as_of_date", "ticker", "event_source", "event_id"],
        kind="stable",
    )


def make_walk_forward_folds(
    dataset: pd.DataFrame,
    *,
    fold_windows: tuple[tuple[str, str], ...] = DEFAULT_FOLD_WINDOWS,
) -> tuple[WalkForwardFold, ...]:
    """Create expanding folds while purging labels that cross either boundary."""

    if not fold_windows:
        raise ValueError("At least one walk-forward fold is required.")

    labeled = _labeled_dataset(dataset)
    folds = []
    previous_start = None

    for index, (raw_start, raw_end) in enumerate(fold_windows, start=1):
        start = pd.Timestamp(raw_start)
        end = pd.Timestamp(raw_end)

        if start >= end:
            raise ValueError("Each fold start must be before its end.")

        if previous_start is not None and start <= previous_start:
            raise ValueError("Fold starts must be strictly increasing.")

        train_mask = (
            (labeled["feature_as_of_date"] < start)
            & (labeled["target_trading_date"] < start)
        )
        validation_mask = (
            (labeled["feature_as_of_date"] >= start)
            & (labeled["feature_as_of_date"] < end)
            & (labeled["target_trading_date"] < end)
        )
        boundary_scope = labeled["feature_as_of_date"] < end
        assigned = train_mask | validation_mask
        fold = WalkForwardFold(
            name=f"fold_{index:02d}",
            train=labeled.loc[train_mask].copy(),
            validation=labeled.loc[validation_mask].copy(),
            purged=labeled.loc[boundary_scope & ~assigned].copy(),
            validation_start=start,
            validation_end=end,
        )
        validate_walk_forward_fold(fold)
        folds.append(fold)
        previous_start = start

    return tuple(folds)


def validate_walk_forward_fold(fold: WalkForwardFold) -> dict[str, int]:
    """Reject identity overlap and target windows leaking across boundaries."""

    train_keys = set(fold.train["event_key"])
    validation_keys = set(fold.validation["event_key"])
    purged_keys = set(fold.purged["event_key"])
    checks = {
        "train_rows": len(fold.train),
        "validation_rows": len(fold.validation),
        "purged_rows": len(fold.purged),
        "overlapping_event_keys": (
            len(train_keys & validation_keys)
            + len(train_keys & purged_keys)
            + len(validation_keys & purged_keys)
        ),
        "train_targets_cross_validation": int((
            fold.train["target_trading_date"] >= fold.validation_start
        ).sum()),
        "validation_targets_cross_end": int((
            fold.validation["target_trading_date"] >= fold.validation_end
        ).sum()),
        "validation_features_before_start": int((
            fold.validation["feature_as_of_date"] < fold.validation_start
        ).sum()),
        "validation_features_at_or_after_end": int((
            fold.validation["feature_as_of_date"] >= fold.validation_end
        ).sum()),
    }
    failures = {
        key: value
        for key, value in checks.items()
        if key not in {"train_rows", "validation_rows", "purged_rows"}
        and value
    }

    if failures:
        raise ValueError(f"Walk-forward fold validation failed: {failures}")

    if not len(fold.train) or not len(fold.validation):
        raise ValueError("Every fold requires non-empty train and validation rows.")

    return checks


def _linear_pipeline(model) -> Pipeline:
    """Build a stable missing-aware pipeline for regularized linear models."""

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
        ("model", model),
    ])


def build_candidate_estimator(candidate: CandidateSpec):
    """Construct one deterministic estimator from its declared family."""

    parameters = dict(candidate.parameters)

    if candidate.family == "ridge":
        return _linear_pipeline(Ridge(**parameters))

    if candidate.family == "elasticnet":
        return _linear_pipeline(ElasticNet(
            max_iter=20_000,
            selection="cyclic",
            random_state=42,
            **parameters,
        ))

    if candidate.family == "extra_trees":
        return Pipeline([
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                    keep_empty_features=True,
                ),
            ),
            ("model", ExtraTreesRegressor(
                random_state=42,
                n_jobs=1,
                **parameters,
            )),
        ])

    if candidate.family == "xgboost":
        n_estimators = int(parameters.pop("n_estimators"))
        return build_xgboost_regressor(
            parameters,
            n_estimators=n_estimators,
            early_stopping_rounds=None,
        )

    if candidate.family == "catboost":
        try:
            from catboost import CatBoostRegressor
        except ImportError as error:
            raise ImportError(
                "CatBoost is required for the model benchmark. "
                "Install requirements-research.txt."
            ) from error

        return CatBoostRegressor(
            loss_function="RMSE",
            random_seed=42,
            thread_count=1,
            verbose=False,
            allow_writing_files=False,
            **parameters,
        )

    raise ValueError(f"Unsupported model family: {candidate.family}")


def _fold_prediction(
    fold: WalkForwardFold,
    candidate: CandidateSpec,
    features: tuple[str, ...],
) -> tuple[dict, pd.DataFrame]:
    """Fit one fold using only prior labels and return auditable predictions."""

    train_y = fold.train[TARGET_COLUMN]

    if candidate.family == "baseline_zero":
        predictions = np.zeros(len(fold.validation))
    elif candidate.family == "baseline_mean":
        predictions = np.full(len(fold.validation), float(train_y.mean()))
    else:
        estimator = build_candidate_estimator(candidate)
        estimator.fit(fold.train[list(features)], train_y)
        predictions = estimator.predict(fold.validation[list(features)])

    metrics = evaluate_predictions(
        fold.validation[TARGET_COLUMN],
        predictions,
        model_name=candidate.name,
        split_name=fold.name,
    )
    metrics.update({
        "fold": fold.name,
        "family": candidate.family,
        "validation_start": fold.validation_start,
        "validation_end": fold.validation_end,
        "train_rows": len(fold.train),
        "purged_rows": len(fold.purged),
        "parameters": dict(candidate.parameters),
    })
    prediction_frame = _prediction_frame(
        fold.validation,
        predictions,
        candidate.name,
    )
    prediction_frame.insert(0, "fold", fold.name)
    prediction_frame.insert(2, "family", candidate.family)
    return metrics, prediction_frame


def _summarize_candidates(
    fold_metrics: pd.DataFrame,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Combine fold stability with pooled out-of-fold prediction quality."""

    rows = []

    for (model, family), group in fold_metrics.groupby(
        ["model", "family"],
        sort=False,
    ):
        model_predictions = predictions.loc[predictions["model"] == model]
        pooled = evaluate_predictions(
            model_predictions[TARGET_COLUMN],
            model_predictions["prediction"],
            model_name=model,
            split_name="walk_forward_oof",
        )
        rows.append({
            "model": model,
            "family": family,
            "folds": len(group),
            "validation_rows": int(group["rows"].sum()),
            "mean_mae": float(group["mae"].mean()),
            "std_mae": float(group["mae"].std(ddof=0)),
            "worst_fold_mae": float(group["mae"].max()),
            "mean_rmse": float(group["rmse"].mean()),
            "mean_directional_accuracy": float(
                group["directional_accuracy"].mean()
            ),
            "mean_spearman_ic": float(group["spearman_ic"].mean()),
            "pooled_mae": pooled["mae"],
            "pooled_rmse": pooled["rmse"],
            "pooled_directional_accuracy": pooled["directional_accuracy"],
            "pooled_spearman_ic": pooled["spearman_ic"],
        })

    return pd.DataFrame(rows).sort_values(
        ["mean_mae", "std_mae", "pooled_mae", "model"],
        kind="stable",
    ).reset_index(drop=True)


def _selection_summary(
    summary: pd.DataFrame,
    fold_metrics: pd.DataFrame,
) -> dict:
    """Nominate a future-holdout candidate without declaring a champion."""

    learned = summary.loc[~summary["family"].str.startswith("baseline_")]

    if learned.empty:
        raise ValueError("Benchmark requires at least one learned model.")

    selected = learned.iloc[0]
    baseline = summary.loc[summary["model"] == "historical_mean"]

    if len(baseline) != 1:
        raise ValueError("Benchmark requires exactly one historical_mean model.")

    baseline = baseline.iloc[0]
    selected_folds = fold_metrics.loc[
        fold_metrics["model"] == selected["model"],
        ["fold", "mae"],
    ].rename(columns={"mae": "selected_mae"})
    baseline_folds = fold_metrics.loc[
        fold_metrics["model"] == "historical_mean",
        ["fold", "mae"],
    ].rename(columns={"mae": "baseline_mae"})
    comparison = selected_folds.merge(
        baseline_folds,
        on="fold",
        validate="one_to_one",
    )
    fold_wins = int((
        comparison["selected_mae"] < comparison["baseline_mae"]
    ).sum())
    improvement = float(baseline["mean_mae"] - selected["mean_mae"])
    minimum_wins = len(comparison) // 2 + 1
    eligible_for_fresh_holdout = bool(
        improvement > 0
        and fold_wins >= minimum_wins
        and selected["pooled_spearman_ic"] > 0
    )
    return {
        "selected_model": selected["model"],
        "selected_family": selected["family"],
        "selection_metric": "lowest mean fold MAE among learned models",
        "mean_mae": float(selected["mean_mae"]),
        "historical_mean_mae": float(baseline["mean_mae"]),
        "absolute_mae_improvement": improvement,
        "fold_wins_vs_historical_mean": fold_wins,
        "fold_count": len(comparison),
        "pooled_spearman_ic": float(selected["pooled_spearman_ic"]),
        "eligible_for_fresh_holdout": eligible_for_fresh_holdout,
        "status": (
            "fresh_holdout_candidate"
            if eligible_for_fresh_holdout
            else "no_robust_candidate"
        ),
        "production_eligible": False,
        "reason": (
            "Walk-forward results may nominate a future-holdout candidate, "
            "but cannot promote a model on already-inspected history."
        ),
    }


def run_walk_forward_benchmark(
    dataset: pd.DataFrame,
    *,
    fold_windows: tuple[tuple[str, str], ...] = DEFAULT_FOLD_WINDOWS,
    candidates: tuple[CandidateSpec, ...] = DEFAULT_CANDIDATES,
    include_topics: bool = True,
) -> BenchmarkResult:
    """Evaluate fixed model candidates across purged expanding-time folds."""

    if not candidates:
        raise ValueError("At least one model candidate is required.")

    names = [candidate.name for candidate in candidates]

    if len(names) != len(set(names)):
        raise ValueError("Model candidate names must be unique.")

    features = model_feature_columns(include_topics=include_topics)
    missing_features = set(features) - set(dataset.columns)

    if missing_features:
        raise ValueError(
            "dataset is missing model features: "
            + ", ".join(sorted(missing_features))
        )

    folds = make_walk_forward_folds(dataset, fold_windows=fold_windows)
    metric_rows = []
    prediction_frames = []

    for fold in folds:
        for candidate in candidates:
            metrics, predictions = _fold_prediction(
                fold,
                candidate,
                features,
            )
            metric_rows.append(metrics)
            prediction_frames.append(predictions)

    fold_metrics = pd.DataFrame(metric_rows)
    oof_predictions = pd.concat(prediction_frames, ignore_index=True)
    summary = _summarize_candidates(fold_metrics, oof_predictions)
    selection = _selection_summary(summary, fold_metrics)
    return BenchmarkResult(
        folds=folds,
        candidates=candidates,
        feature_columns=features,
        fold_metrics=fold_metrics,
        summary=summary,
        oof_predictions=oof_predictions,
        selection=selection,
    )


def benchmark_to_json(result: BenchmarkResult) -> dict:
    """Serialize the full benchmark contract for reports and later review."""

    return {
        "target": TARGET_COLUMN,
        "feature_count": len(result.feature_columns),
        "feature_columns": list(result.feature_columns),
        "selection": result.selection,
        "folds": [
            {
                "name": fold.name,
                "validation_start": fold.validation_start.date().isoformat(),
                "validation_end": fold.validation_end.date().isoformat(),
                **validate_walk_forward_fold(fold),
            }
            for fold in result.folds
        ],
        "candidates": [
            {
                "name": candidate.name,
                "family": candidate.family,
                "parameters": candidate.parameters,
            }
            for candidate in result.candidates
        ],
        "summary": json.loads(result.summary.to_json(orient="records")),
        "fold_metrics": json.loads(
            result.fold_metrics.to_json(orient="records", date_format="iso")
        ),
    }


def main() -> None:
    """Run the fixed walk-forward benchmark from PowerShell or a scheduler."""

    parser = argparse.ArgumentParser(
        description="Compare AlphaLens models using purged walk-forward folds.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/ml/model_benchmark.json"),
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("data/ml/model_benchmark_predictions.csv"),
    )
    arguments = parser.parse_args()
    dataset = build_event_feature_dataset(horizon=30, include_topics=True)
    result = run_walk_forward_benchmark(dataset)

    print("\nWalk-forward folds")
    print(pd.DataFrame(benchmark_to_json(result)["folds"]).to_string(index=False))
    print("\nModel comparison")
    print(result.summary.to_string(index=False))
    print("\nSelection")
    print(json.dumps(result.selection, indent=2))
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(benchmark_to_json(result), indent=2),
        encoding="utf-8",
    )
    arguments.predictions.parent.mkdir(parents=True, exist_ok=True)
    result.oof_predictions.to_csv(arguments.predictions, index=False)
    print(f"\nReport: {arguments.output}")
    print(f"Predictions: {arguments.predictions}")


if __name__ == "__main__":
    main()
