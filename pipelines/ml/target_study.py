"""Predeclared multi-horizon return-target robustness study for AlphaLens."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from pipelines.ml.baselines import evaluate_predictions
from pipelines.ml.dataset import (
    build_forward_return_targets,
    get_database_engine,
    load_adjusted_prices,
    load_events,
    validate_target_dataset,
)
from pipelines.ml.features import build_event_feature_dataset
from pipelines.ml.model_benchmark import (
    DEFAULT_CANDIDATES,
    DEFAULT_FOLD_WINDOWS,
    BenchmarkResult,
    CandidateSpec,
    benchmark_to_json,
    run_walk_forward_benchmark,
)


DEFAULT_HORIZONS = (5, 10, 20, 30, 60)
TARGET_STUDY_MODEL_NAMES = {
    "zero_excess",
    "historical_mean",
    "ridge_alpha_100",
    "elasticnet_a01_l50",
    "extra_trees_depth4",
    "xgboost_registered_shape",
    "catboost_conservative",
}
TARGET_STUDY_CANDIDATES = tuple(
    candidate
    for candidate in DEFAULT_CANDIDATES
    if candidate.name in TARGET_STUDY_MODEL_NAMES
)


@dataclass
class TargetStudyResult:
    """Multi-horizon benchmarks and a non-production research decision."""

    benchmarks: dict[int, BenchmarkResult]
    horizon_summary: pd.DataFrame
    source_summary: pd.DataFrame
    selection: dict


def target_column(horizon: int) -> str:
    """Return the canonical excess-return label for one trading horizon."""

    if horizon < 1:
        raise ValueError("horizon must be at least one trading session.")

    return f"excess_return_{horizon}d"


def _target_columns(horizon: int) -> tuple[str, ...]:
    """Identify horizon-specific outcomes attached to the shared features."""

    return (
        "target_trading_date",
        "target_adjusted_close",
        f"stock_forward_return_{horizon}d",
        "spy_target_adjusted_close",
        f"spy_forward_return_{horizon}d",
        target_column(horizon),
        "target_available",
        "target_status",
        "target_error",
    )


def assemble_horizon_dataset(
    base_features: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    horizon: int,
) -> pd.DataFrame:
    """Attach one horizon's labels without recalculating observable features."""

    label_columns = _target_columns(horizon)
    required_base = {"event_key", "feature_as_of_date"}
    required_targets = {
        "event_key",
        "event_source",
        "event_id",
        "event_date",
        "anchor_trading_date",
        *label_columns,
    }

    if not required_base.issubset(base_features.columns):
        raise ValueError("base_features is missing event identity columns.")

    if not required_targets.issubset(targets.columns):
        raise ValueError("targets is missing horizon label columns.")

    if base_features["event_key"].duplicated().any():
        raise ValueError("base_features contains duplicate event keys.")

    if targets["event_key"].duplicated().any():
        raise ValueError("targets contains duplicate event keys.")

    base_keys = set(base_features["event_key"])
    target_keys = set(targets["event_key"])

    if base_keys != target_keys:
        raise ValueError("Feature and target event identities do not match.")

    anchor_check = base_features[[
        "event_key",
        "feature_as_of_date",
    ]].merge(
        targets[["event_key", "anchor_trading_date"]],
        on="event_key",
        validate="one_to_one",
    )
    feature_dates = pd.to_datetime(
        anchor_check["feature_as_of_date"],
        errors="coerce",
    )
    anchor_dates = pd.to_datetime(
        anchor_check["anchor_trading_date"],
        errors="coerce",
    )
    comparable = feature_dates.notna() & anchor_dates.notna()

    if (feature_dates[comparable] != anchor_dates[comparable]).any():
        raise ValueError("Horizon targets do not share the feature anchor date.")

    stale_target_columns = [
        column
        for column in base_features.columns
        if column in {
            "target_trading_date",
            "target_adjusted_close",
            "spy_target_adjusted_close",
            "target_available",
            "target_status",
            "target_error",
        }
        or column.startswith("stock_forward_return_")
        or column.startswith("spy_forward_return_")
        or column.startswith("excess_return_")
    ]
    features = base_features.drop(columns=stale_target_columns)
    labels = targets[["event_key", *label_columns]].copy()
    result = features.merge(
        labels,
        on="event_key",
        how="left",
        validate="one_to_one",
    )
    validate_target_dataset(
        targets,
        horizon=horizon,
    )
    return result.sort_values(
        ["event_date", "ticker", "event_source", "event_id"],
        kind="stable",
    ).reset_index(drop=True)


def build_horizon_feature_datasets(
    engine=None,
    *,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    include_topics: bool = True,
) -> dict[int, pd.DataFrame]:
    """Build shared point-in-time features and attach each requested target."""

    normalized_horizons = tuple(dict.fromkeys(int(value) for value in horizons))

    if not normalized_horizons:
        raise ValueError("At least one target horizon is required.")

    for horizon in normalized_horizons:
        target_column(horizon)

    resolved_engine = engine or get_database_engine()
    # Features are horizon-independent; the 30-session build provides their
    # anchor rows once, while the lightweight target calculation varies below.
    base_features = build_event_feature_dataset(
        resolved_engine,
        horizon=30,
        include_topics=include_topics,
    )
    events = load_events(resolved_engine)
    prices = load_adjusted_prices(resolved_engine)
    datasets = {}

    for horizon in normalized_horizons:
        targets = build_forward_return_targets(
            events,
            prices,
            benchmark="SPY",
            horizon=horizon,
        )
        datasets[horizon] = assemble_horizon_dataset(
            base_features,
            targets,
            horizon=horizon,
        )

    return datasets


def _source_performance(
    horizon: int,
    benchmark: BenchmarkResult,
) -> list[dict]:
    """Measure each model on call and filing slices of the same OOF rows."""

    rows = []
    target = benchmark.target_column

    for (model, family, source), group in benchmark.oof_predictions.groupby(
        ["model", "family", "event_source"],
        sort=False,
    ):
        metrics = evaluate_predictions(
            group[target],
            group["prediction"],
            model_name=model,
            split_name=source,
        )
        rows.append({
            "horizon": horizon,
            "target": target,
            "model": model,
            "family": family,
            "event_source": source,
            "rows": metrics["rows"],
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "directional_accuracy": metrics["directional_accuracy"],
            "spearman_ic": metrics["spearman_ic"],
        })

    return rows


def _horizon_row(horizon: int, benchmark: BenchmarkResult) -> dict:
    """Summarize the predeclared model decision for one horizon."""

    selection = benchmark.selection
    baseline_mae = float(selection["historical_mean_mae"])
    absolute_improvement = float(selection["absolute_mae_improvement"])
    return {
        "horizon": horizon,
        "target": benchmark.target_column,
        "labeled_oof_rows": int(
            benchmark.oof_predictions.loc[
                benchmark.oof_predictions["model"]
                == selection["selected_model"]
            ].shape[0]
        ),
        "selected_model": selection["selected_model"],
        "selected_family": selection["selected_family"],
        "model_mean_mae": float(selection["mean_mae"]),
        "historical_mean_mae": baseline_mae,
        "absolute_mae_improvement": absolute_improvement,
        "relative_mae_improvement": (
            absolute_improvement / baseline_mae if baseline_mae else None
        ),
        "fold_wins": int(selection["fold_wins_vs_historical_mean"]),
        "fold_count": int(selection["fold_count"]),
        "pooled_spearman_ic": float(selection["pooled_spearman_ic"]),
        "eligible_for_fresh_holdout": bool(
            selection["eligible_for_fresh_holdout"]
        ),
    }


def _study_selection(horizon_summary: pd.DataFrame) -> dict:
    """Apply the locked cross-horizon rule without promoting on old history."""

    eligible = horizon_summary.loc[
        horizon_summary["eligible_for_fresh_holdout"]
    ].sort_values(
        ["relative_mae_improvement", "fold_wins", "pooled_spearman_ic"],
        ascending=[False, False, False],
        kind="stable",
    )

    if eligible.empty:
        return {
            "status": "no_robust_horizon",
            "selected_horizon": None,
            "selected_model": None,
            "selection_rule": (
                "Beat historical-mean MAE, win a strict majority of folds, "
                "and retain positive pooled Spearman IC."
            ),
            "production_eligible": False,
            "reason": (
                "No horizon passed every predeclared walk-forward condition."
            ),
        }

    selected = eligible.iloc[0]
    return {
        "status": "fresh_holdout_candidate",
        "selected_horizon": int(selected["horizon"]),
        "selected_model": selected["selected_model"],
        "selection_rule": (
            "Among eligible horizons, maximize relative MAE improvement; "
            "then fold wins and pooled Spearman IC."
        ),
        "relative_mae_improvement": float(
            selected["relative_mae_improvement"]
        ),
        "production_eligible": False,
        "reason": (
            "Multiple horizons were explored on historical folds; the winner "
            "still requires genuinely future holdout and net backtest evidence."
        ),
    }


def run_target_study(
    datasets: dict[int, pd.DataFrame],
    *,
    fold_windows: tuple[tuple[str, str], ...] = DEFAULT_FOLD_WINDOWS,
    candidates: tuple[CandidateSpec, ...] = TARGET_STUDY_CANDIDATES,
    include_topics: bool = True,
) -> TargetStudyResult:
    """Compare predeclared horizons under identical walk-forward governance."""

    if not datasets:
        raise ValueError("At least one horizon dataset is required.")

    benchmarks = {}
    horizon_rows = []
    source_rows = []

    for horizon in sorted(datasets):
        target = target_column(int(horizon))
        benchmark = run_walk_forward_benchmark(
            datasets[horizon],
            fold_windows=fold_windows,
            candidates=candidates,
            include_topics=include_topics,
            target_column=target,
        )
        benchmarks[int(horizon)] = benchmark
        horizon_rows.append(_horizon_row(int(horizon), benchmark))
        source_rows.extend(_source_performance(int(horizon), benchmark))

    horizon_summary = pd.DataFrame(horizon_rows).sort_values(
        "horizon",
        kind="stable",
    ).reset_index(drop=True)
    source_summary = pd.DataFrame(source_rows).sort_values(
        ["horizon", "model", "event_source"],
        kind="stable",
    ).reset_index(drop=True)
    return TargetStudyResult(
        benchmarks=benchmarks,
        horizon_summary=horizon_summary,
        source_summary=source_summary,
        selection=_study_selection(horizon_summary),
    )


def target_study_to_json(result: TargetStudyResult) -> dict:
    """Serialize target comparisons and every underlying benchmark audit."""

    return {
        "selection": result.selection,
        "horizon_summary": json.loads(
            result.horizon_summary.to_json(orient="records")
        ),
        "source_summary": json.loads(
            result.source_summary.to_json(orient="records")
        ),
        "benchmarks": {
            str(horizon): benchmark_to_json(benchmark)
            for horizon, benchmark in result.benchmarks.items()
        },
    }


def main() -> None:
    """Run the fixed multi-horizon study and persist ignored local reports."""

    parser = argparse.ArgumentParser(
        description="Compare excess-return targets across trading horizons.",
    )
    parser.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        default=list(DEFAULT_HORIZONS),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/ml/target_study.json"),
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("data/ml/target_study_predictions.csv"),
    )
    arguments = parser.parse_args()
    horizons = tuple(arguments.horizons)
    datasets = build_horizon_feature_datasets(horizons=horizons)
    result = run_target_study(datasets)

    print("\nHorizon comparison")
    print(result.horizon_summary.to_string(index=False))
    selected_sources = result.source_summary.loc[
        result.source_summary.apply(
            lambda row: row["model"]
            == result.benchmarks[int(row["horizon"])].selection[
                "selected_model"
            ],
            axis=1,
        )
    ]
    print("\nSelected-model event-source diagnostics")
    print(selected_sources.to_string(index=False))
    print("\nStudy decision")
    print(json.dumps(result.selection, indent=2))
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(target_study_to_json(result), indent=2),
        encoding="utf-8",
    )
    prediction_frames = []

    for horizon, benchmark in result.benchmarks.items():
        frame = benchmark.oof_predictions.copy()
        frame.insert(0, "horizon", horizon)
        prediction_frames.append(frame)

    arguments.predictions.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(prediction_frames, ignore_index=True).to_csv(
        arguments.predictions,
        index=False,
    )
    print(f"\nReport: {arguments.output}")
    print(f"Predictions: {arguments.predictions}")


if __name__ == "__main__":
    main()
