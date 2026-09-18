"""Frozen-model scoring and delayed evaluation for prospective events."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import MetaData, Table, and_, select, update

from pipelines.ml.dataset import get_database_engine, load_adjusted_prices
from pipelines.ml.features import build_event_feature_dataset, model_feature_columns


ARTIFACT_SCHEMA_VERSION = 1
BENCHMARK_TICKER = "SPY"
SELECTED_HORIZON = 10
SELECTED_MODEL = "elasticnet_a01_l50"
TARGET_COLUMN = "excess_return_10d"
DEFAULT_ARTIFACT_DIRECTORY = Path("data/ml/prospective")
DEFAULT_REPORT_PATH = Path("data/ml/prospective_evaluation.json")
MINIMUM_REVIEW_EVENTS = 40
MODEL_PARAMETERS = {"alpha": 0.01, "l1_ratio": 0.5}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _json_safe(value):
    """Convert pandas and NumPy values into strict JSON-compatible data."""

    if value is None or value is pd.NA:
        return None

    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None

    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]

    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]

    return value


def _write_json(path: Path, payload: dict) -> None:
    """Atomically write deterministic strict JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def _git_commit() -> str | None:
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


def _training_rows(
    dataset: pd.DataFrame,
    cutoff: pd.Timestamp,
) -> pd.DataFrame:
    """Keep only labels that were fully observable by the frozen cutoff."""

    target = TARGET_COLUMN
    required = {
        "event_key",
        "feature_as_of_date",
        "target_trading_date",
        "target_available",
        target,
        *model_feature_columns(),
    }
    missing = required - set(dataset.columns)

    if missing:
        raise ValueError(
            "dataset is missing prospective training columns: "
            + ", ".join(sorted(missing))
        )

    frame = dataset.copy()
    frame["feature_as_of_date"] = pd.to_datetime(
        frame["feature_as_of_date"], errors="coerce"
    )
    frame["target_trading_date"] = pd.to_datetime(
        frame["target_trading_date"], errors="coerce"
    )
    training = frame.loc[
        frame["target_available"].fillna(False)
        & (frame["feature_as_of_date"] <= cutoff)
        & (frame["target_trading_date"] <= cutoff)
    ].dropna(subset=[target])

    if len(training) < 20:
        raise ValueError("At least 20 observable events are required to freeze a model.")

    return training.sort_values(
        ["feature_as_of_date", "event_key"], kind="stable"
    ).reset_index(drop=True)


def _dataset_fingerprint(frame: pd.DataFrame, features: tuple[str, ...]) -> str:
    columns = [
        "event_key",
        "feature_as_of_date",
        "target_trading_date",
        TARGET_COLUMN,
        *features,
    ]
    ordered = frame[columns].sort_values("event_key", kind="stable").copy()

    for column in ("feature_as_of_date", "target_trading_date"):
        ordered[column] = ordered[column].astype("string")

    hashes = pd.util.hash_pandas_object(ordered, index=False).to_numpy()
    return hashlib.sha256(hashes.tobytes()).hexdigest()


def _artifact_directory(path: Path | None = None) -> Path:
    configured = os.getenv("ALPHALENS_PROSPECTIVE_MODEL_DIRECTORY")
    return Path(path or configured or DEFAULT_ARTIFACT_DIRECTORY)


def freeze_prospective_model(
    dataset: pd.DataFrame,
    *,
    artifact_directory: Path | None = None,
    training_cutoff: str | pd.Timestamp | None = None,
    frozen_at: datetime | None = None,
    version: str | None = None,
) -> dict:
    """Fit once, serialize transparent parameters, and publish a frozen pointer."""

    from pipelines.ml.model_benchmark import CandidateSpec, build_candidate_estimator

    features = model_feature_columns()
    observed_feature_dates = pd.to_datetime(
        dataset["feature_as_of_date"], errors="coerce"
    ).dropna()

    if observed_feature_dates.empty:
        raise ValueError("No feature dates are available for model freezing.")

    cutoff = pd.Timestamp(training_cutoff or observed_feature_dates.max()).normalize()
    training = _training_rows(dataset, cutoff)
    target = TARGET_COLUMN
    candidate = CandidateSpec(SELECTED_MODEL, "elasticnet", MODEL_PARAMETERS)
    estimator = build_candidate_estimator(candidate)
    estimator.fit(training[list(features)], training[target])
    imputer = estimator.named_steps["imputer"]
    scaler = estimator.named_steps["scaler"]
    model = estimator.named_steps["model"]
    fingerprint = _dataset_fingerprint(training, features)
    created_at = frozen_at or _utc_now()
    resolved_version = version or (
        f"elasticnet-10d-{created_at.strftime('%Y%m%dT%H%M%SZ')}-"
        f"{fingerprint[:8]}"
    )

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,159}", resolved_version):
        raise ValueError("version contains unsupported path characters.")

    root = _artifact_directory(artifact_directory)
    destination = root / resolved_version

    if destination.exists():
        raise FileExistsError(f"Prospective model version already exists: {destination}")

    reference = training.iloc[np.linspace(0, len(training) - 1, 5, dtype=int)]
    artifact = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "version": resolved_version,
        "status": "shadow",
        "created_at": created_at,
        "source_commit": _git_commit(),
        "training_cutoff": cutoff.date().isoformat(),
        "training_rows": len(training),
        "training_first_feature_date": training["feature_as_of_date"].min(),
        "training_last_target_date": training["target_trading_date"].max(),
        "dataset_fingerprint": fingerprint,
        "model_name": SELECTED_MODEL,
        "model_family": "elasticnet",
        "model_parameters": MODEL_PARAMETERS,
        "horizon_sessions": SELECTED_HORIZON,
        "target": target,
        "benchmark_ticker": BENCHMARK_TICKER,
        "feature_columns": list(features),
        "historical_mean_prediction": float(training[target].mean()),
        "transform": {
            "imputer_statistics": imputer.statistics_,
            "missing_indicator_features": imputer.indicator_.features_,
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
        },
        "linear_model": {
            "coefficients": model.coef_,
            "intercept": float(model.intercept_),
        },
        "reference": {
            "event_keys": reference["event_key"].tolist(),
            "features": [
                {
                    feature: _json_safe(value)
                    for feature, value in row.items()
                }
                for row in reference[list(features)].to_dict(orient="records")
            ],
            "predictions": estimator.predict(reference[list(features)]),
        },
        "governance": {
            "production_eligible": False,
            "usage": "prospective_shadow_evaluation_only",
            "minimum_review_events": MINIMUM_REVIEW_EVENTS,
        },
    }
    destination.mkdir(parents=True)
    artifact_path = destination / "model.json"
    _write_json(artifact_path, artifact)
    pointer = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "version": resolved_version,
        "artifact": f"{resolved_version}/model.json",
        "sha256": _sha256(artifact_path),
        "published_at": created_at,
    }
    _write_json(root / "current.json", pointer)
    return load_frozen_model(root)


def _predict_array(artifact: dict, features: pd.DataFrame) -> np.ndarray:
    """Replay the fitted sklearn pipeline from its transparent JSON state."""

    columns = tuple(artifact["feature_columns"])
    missing = set(columns) - set(features.columns)

    if missing:
        raise ValueError(
            "prediction frame is missing model features: "
            + ", ".join(sorted(missing))
        )

    values = features[list(columns)].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=float)

    if np.isinf(values).any():
        raise ValueError("prediction features contain infinite values.")

    transform = artifact["transform"]
    statistics = np.asarray(transform["imputer_statistics"], dtype=float)
    missing_mask = np.isnan(values)
    imputed = np.where(missing_mask, statistics, values)
    indicator_indices = np.asarray(
        transform["missing_indicator_features"], dtype=int
    )

    if indicator_indices.size:
        imputed = np.concatenate(
            [imputed, missing_mask[:, indicator_indices].astype(float)], axis=1
        )

    mean = np.asarray(transform["scaler_mean"], dtype=float)
    scale = np.asarray(transform["scaler_scale"], dtype=float)
    coefficients = np.asarray(
        artifact["linear_model"]["coefficients"], dtype=float
    )

    if not (imputed.shape[1] == len(mean) == len(scale) == len(coefficients)):
        raise ValueError("Frozen model parameter dimensions do not match.")

    scaled = (imputed - mean) / scale
    return scaled @ coefficients + float(artifact["linear_model"]["intercept"])


def load_frozen_model(artifact_directory: Path | None = None) -> dict:
    """Load, checksum, and replay references from the current shadow model."""

    root = _artifact_directory(artifact_directory)
    pointer_path = root / "current.json"

    if not pointer_path.exists():
        raise FileNotFoundError(
            "No prospective model is frozen. Run: "
            "python -m pipelines.ml.prospective freeze"
        )

    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    artifact_path = root / pointer["artifact"]

    if not artifact_path.exists() or _sha256(artifact_path) != pointer["sha256"]:
        raise ValueError("Prospective model artifact checksum verification failed.")

    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))

    if artifact.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ValueError("Unsupported prospective model artifact schema.")

    reference = pd.DataFrame(artifact["reference"]["features"])
    replay = _predict_array(artifact, reference)
    expected = np.asarray(artifact["reference"]["predictions"], dtype=float)

    if not np.allclose(replay, expected, rtol=1e-9, atol=1e-10):
        raise ValueError("Prospective model reference replay failed.")

    return artifact


def build_prediction_records(
    artifact: dict,
    dataset: pd.DataFrame,
    *,
    existing_event_keys: set[str] | None = None,
    tickers: list[str] | None = None,
    generated_at: datetime | None = None,
) -> list[dict]:
    """Create records only while each event's forward outcome is unavailable."""

    required = {
        "event_key",
        "ticker",
        "event_source",
        "event_id",
        "event_date",
        "feature_as_of_date",
        "target_available",
        "target_error",
        *artifact["feature_columns"],
    }
    missing = required - set(dataset.columns)

    if missing:
        raise ValueError(
            "dataset is missing prospective scoring columns: "
            + ", ".join(sorted(missing))
        )

    frame = dataset.copy()
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    frame["event_date"] = pd.to_datetime(frame["event_date"], errors="coerce")
    frame["feature_as_of_date"] = pd.to_datetime(
        frame["feature_as_of_date"], errors="coerce"
    )
    timestamp = generated_at or _utc_now()

    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)

    allowed_tickers = {
        str(ticker).strip().upper() for ticker in (tickers or []) if ticker
    }
    known = existing_event_keys or set()
    # A stale local price table must never turn an already elapsed outcome
    # into seemingly prospective evidence. BDay is deliberately conservative
    # around market holidays: we would rather skip a valid row than leak one.
    expected_deadline = (
        frame["feature_as_of_date"]
        + pd.offsets.BDay(int(artifact["horizon_sessions"]))
    )
    mask = (
        ~frame["event_key"].isin(known)
        & ~frame["target_available"].fillna(False)
        & frame["target_error"].eq("incomplete_forward_window")
        & frame["event_date"].notna()
        & frame["feature_as_of_date"].notna()
        & (expected_deadline.dt.date > timestamp.date())
    )

    if allowed_tickers:
        mask &= frame["ticker"].isin(allowed_tickers)

    candidates = frame.loc[mask].sort_values(
        ["feature_as_of_date", "ticker", "event_key"], kind="stable"
    )

    if candidates.empty:
        return []

    if (candidates["feature_as_of_date"].dt.date > timestamp.date()).any():
        raise ValueError("A prediction cannot predate its feature snapshot.")

    predictions = _predict_array(artifact, candidates)
    features = tuple(artifact["feature_columns"])
    records = []

    for (_, row), prediction in zip(
        candidates.iterrows(), predictions, strict=True
    ):
        snapshot = {
            feature: _json_safe(row[feature])
            for feature in features
        }
        records.append({
            "model_version": artifact["version"],
            "event_key": row["event_key"],
            "ticker": row["ticker"],
            "event_source": row["event_source"],
            "event_id": str(row["event_id"]),
            "event_date": row["event_date"].date(),
            "feature_as_of_date": row["feature_as_of_date"].date(),
            "prediction_generated_at": timestamp,
            "training_cutoff": pd.Timestamp(
                artifact["training_cutoff"]
            ).date(),
            "horizon_sessions": artifact["horizon_sessions"],
            "benchmark_ticker": artifact["benchmark_ticker"],
            "predicted_excess_return": float(prediction),
            "baseline_predicted_excess_return": float(
                artifact["historical_mean_prediction"]
            ),
            "feature_snapshot": snapshot,
            "status": "PENDING",
        })

    return records


def _prediction_table(engine) -> Table:
    try:
        return Table(
            "prospective_predictions",
            MetaData(),
            autoload_with=engine,
        )
    except Exception as error:
        raise RuntimeError(
            "prospective_predictions is unavailable. Apply "
            "db/sql/018_prospective_predictions.sql."
        ) from error


def score_pending_events(
    engine,
    artifact: dict,
    dataset: pd.DataFrame,
    *,
    tickers: list[str] | None = None,
    generated_at: datetime | None = None,
) -> dict:
    """Insert each model/event prediction once without touching prior rows."""

    table = _prediction_table(engine)

    with engine.connect() as connection:
        existing = set(connection.execute(
            select(table.c.event_key).where(
                table.c.model_version == artifact["version"]
            )
        ).scalars())

    records = build_prediction_records(
        artifact,
        dataset,
        existing_event_keys=existing,
        tickers=tickers,
        generated_at=generated_at,
    )

    if records:
        with engine.begin() as connection:
            connection.execute(table.insert(), records)

    return {
        "model_version": artifact["version"],
        "candidates": len(records),
        "inserted": len(records),
        "already_recorded": len(existing),
    }


def build_maturation_updates(
    pending: pd.DataFrame,
    dataset: pd.DataFrame,
    *,
    matured_at: datetime | None = None,
) -> list[dict]:
    """Attach outcomes only after their complete market window exists."""

    if pending.empty:
        return []

    target = TARGET_COLUMN
    stock_return = f"stock_forward_return_{SELECTED_HORIZON}d"
    benchmark_return = f"spy_forward_return_{SELECTED_HORIZON}d"
    required_pending = {
        "prediction_id",
        "event_key",
        "feature_as_of_date",
        "prediction_generated_at",
    }
    required_dataset = {
        "event_key",
        "target_available",
        "target_error",
        "target_trading_date",
        stock_return,
        benchmark_return,
        target,
    }

    if not required_pending.issubset(pending.columns):
        raise ValueError("pending ledger rows are missing identity columns.")

    if not required_dataset.issubset(dataset.columns):
        raise ValueError("dataset is missing prospective outcome columns.")

    outcomes = dataset.set_index("event_key", drop=False)
    timestamp = matured_at or _utc_now()
    updates = []

    for row in pending.itertuples(index=False):
        if row.event_key not in outcomes.index:
            continue

        outcome = outcomes.loc[row.event_key]

        if isinstance(outcome, pd.DataFrame):
            raise ValueError(f"Duplicate event key during maturation: {row.event_key}")

        if not bool(outcome["target_available"]):
            error = outcome["target_error"]
            expected_deadline = (
                pd.Timestamp(row.feature_as_of_date)
                + pd.offsets.BDay(SELECTED_HORIZON)
            )

            if pd.Timestamp(timestamp).date() >= expected_deadline.date():
                updates.append({
                    "prediction_id": row.prediction_id,
                    "status": "INVALID",
                    "outcome_error": "local_prices_stale_past_expected_horizon",
                    "updated_at": timestamp,
                })
            elif error not in {
                "incomplete_forward_window",
                "missing_post_event_anchor",
            }:
                updates.append({
                    "prediction_id": row.prediction_id,
                    "status": "INVALID",
                    "outcome_error": str(error or "target_unavailable"),
                    "updated_at": timestamp,
                })

            continue

        target_date = pd.Timestamp(outcome["target_trading_date"])
        generated = pd.Timestamp(row.prediction_generated_at)

        if generated.date() >= target_date.date():
            updates.append({
                "prediction_id": row.prediction_id,
                "status": "INVALID",
                "outcome_error": "prediction_was_not_recorded_before_target_date",
                "updated_at": timestamp,
            })
            continue

        updates.append({
            "prediction_id": row.prediction_id,
            "status": "MATURED",
            "target_trading_date": target_date.date(),
            "stock_forward_return": float(outcome[stock_return]),
            "benchmark_forward_return": float(outcome[benchmark_return]),
            "realized_excess_return": float(outcome[target]),
            "matured_at": timestamp,
            "outcome_error": None,
            "updated_at": timestamp,
        })

    return updates


def mature_predictions(
    engine,
    dataset: pd.DataFrame,
    *,
    model_version: str | None = None,
    matured_at: datetime | None = None,
) -> dict:
    """Persist newly available outcomes without rewriting forecast fields."""

    table = _prediction_table(engine)
    statement = select(table).where(table.c.status == "PENDING")

    if model_version:
        statement = statement.where(table.c.model_version == model_version)

    with engine.connect() as connection:
        pending = pd.DataFrame(connection.execute(statement).mappings().all())

    updates = build_maturation_updates(
        pending,
        dataset,
        matured_at=matured_at,
    )

    if updates:
        with engine.begin() as connection:
            for values in updates:
                prediction_id = values.pop("prediction_id")
                connection.execute(
                    update(table)
                    .where(table.c.prediction_id == prediction_id)
                    .values(**values)
                )

    return {
        "pending_checked": len(pending),
        "matured": sum(row["status"] == "MATURED" for row in updates),
        "invalid": sum(row["status"] == "INVALID" for row in updates),
        "still_pending": len(pending) - len(updates),
    }


def load_prediction_ledger(engine, *, model_version: str) -> pd.DataFrame:
    table = _prediction_table(engine)

    with engine.connect() as connection:
        rows = connection.execute(
            select(table)
            .where(table.c.model_version == model_version)
            .order_by(table.c.feature_as_of_date, table.c.ticker)
        ).mappings().all()

    return pd.DataFrame(rows)


def _prediction_metrics(actual, predicted, *, model: str) -> dict:
    actual = pd.Series(actual, dtype=float).reset_index(drop=True)
    predicted = pd.Series(predicted, dtype=float).reset_index(drop=True)

    if actual.empty or len(actual) != len(predicted):
        raise ValueError("actual and predicted must be equal non-empty lengths.")

    residual = actual - predicted
    denominator = float(((actual - actual.mean()) ** 2).sum())
    variation = float(predicted.std(ddof=0)) > 1e-12
    return {
        "model": model,
        "rows": len(actual),
        "mae": float(residual.abs().mean()),
        "rmse": math.sqrt(float((residual ** 2).mean())),
        "r2": (
            1.0 - float((residual ** 2).sum()) / denominator
            if denominator > 1e-12
            else None
        ),
        "directional_accuracy": float(
            ((predicted > 0) == (actual > 0)).mean()
        ),
        "pearson": (
            float(actual.corr(predicted, method="pearson"))
            if variation and len(actual) > 1
            else None
        ),
        "spearman_ic": (
            float(actual.corr(predicted, method="spearman"))
            if variation and len(actual) > 1
            else None
        ),
    }


def evaluate_prospective_ledger(
    ledger: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    model_version: str,
    minimum_review_events: int = MINIMUM_REVIEW_EVENTS,
) -> dict:
    """Evaluate matured forecasts and keep promotion a manual review decision."""

    if minimum_review_events < 1:
        raise ValueError("minimum_review_events must be positive.")

    if ledger.empty:
        return {
            "model_version": model_version,
            "status": "collecting",
            "production_eligible": False,
            "counts": {"total": 0, "pending": 0, "matured": 0, "invalid": 0},
            "minimum_review_events": minimum_review_events,
            "forecast_metrics": [],
            "strategy_performance": [],
        }

    counts = {
        "total": len(ledger),
        "pending": int((ledger["status"] == "PENDING").sum()),
        "matured": int((ledger["status"] == "MATURED").sum()),
        "invalid": int((ledger["status"] == "INVALID").sum()),
    }
    matured = ledger.loc[ledger["status"] == "MATURED"].copy()

    if matured.empty:
        return {
            "model_version": model_version,
            "status": "collecting",
            "production_eligible": False,
            "counts": counts,
            "minimum_review_events": minimum_review_events,
            "forecast_metrics": [],
            "strategy_performance": [],
        }

    model_metrics = _prediction_metrics(
        matured["realized_excess_return"],
        matured["predicted_excess_return"],
        model=SELECTED_MODEL,
    )
    baseline_metrics = _prediction_metrics(
        matured["realized_excess_return"],
        matured["baseline_predicted_excess_return"],
        model="frozen_historical_mean",
    )
    predictions = matured.rename(columns={
        "predicted_excess_return": "prediction",
    })
    # Portfolio machinery is needed only after outcomes mature. Keeping this
    # import lazy leaves ordinary shadow scoring independent of XGBoost.
    from pipelines.ml.backtest import run_event_backtest

    backtest = run_event_backtest(
        predictions,
        prices,
        strategy_name="prospective_elasticnet_10d",
    )
    performance = json.loads(backtest.summary.to_json(orient="records"))
    net_row = backtest.summary.loc[
        backtest.summary["strategy"]
        == "prospective_elasticnet_10d_long_short_net"
    ].iloc[0]
    net_sharpe = float(net_row["sharpe"])
    checks = {
        "minimum_events": counts["matured"] >= minimum_review_events,
        "mae_beats_frozen_mean": model_metrics["mae"] < baseline_metrics["mae"],
        "positive_spearman_ic": (
            model_metrics["spearman_ic"] is not None
            and model_metrics["spearman_ic"] > 0
        ),
        "positive_net_long_short_sharpe": (
            np.isfinite(net_sharpe) and net_sharpe > 0
        ),
    }
    eligible = all(checks.values())
    return {
        "model_version": model_version,
        "status": "eligible_for_review" if eligible else "collecting",
        "production_eligible": False,
        "counts": counts,
        "minimum_review_events": minimum_review_events,
        "checks": checks,
        "forecast_metrics": [model_metrics, baseline_metrics],
        "strategy_performance": performance,
        "decision_note": (
            "Passing every check permits manual promotion review; this job "
            "never promotes a model automatically."
        ),
    }


def run_prospective_monitor(
    engine=None,
    *,
    artifact_directory: Path | None = None,
    report_path: Path = DEFAULT_REPORT_PATH,
    tickers: list[str] | None = None,
    now: datetime | None = None,
) -> dict:
    """Mature old forecasts, record new ones, and refresh the shadow report."""

    resolved_engine = engine or get_database_engine()
    artifact = load_frozen_model(artifact_directory)
    dataset = build_event_feature_dataset(
        resolved_engine,
        horizon=SELECTED_HORIZON,
    )
    maturation = mature_predictions(
        resolved_engine,
        dataset,
        # Older immutable versions may still have open outcomes after a new
        # shadow model is frozen, so maturation always services the full ledger.
        model_version=None,
        matured_at=now,
    )
    scoring = score_pending_events(
        resolved_engine,
        artifact,
        dataset,
        tickers=tickers,
        generated_at=now,
    )
    ledger = load_prediction_ledger(
        resolved_engine,
        model_version=artifact["version"],
    )
    evaluation = evaluate_prospective_ledger(
        ledger,
        load_adjusted_prices(resolved_engine),
        model_version=artifact["version"],
    )
    report = {
        "generated_at": now or _utc_now(),
        "artifact": {
            "version": artifact["version"],
            "training_cutoff": artifact["training_cutoff"],
            "dataset_fingerprint": artifact["dataset_fingerprint"],
        },
        "maturation": maturation,
        "scoring": scoring,
        "evaluation": evaluation,
    }
    _write_json(report_path, report)
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze and monitor the 10-session prospective model.",
    )
    parser.add_argument(
        "command",
        choices=("freeze", "run"),
        help="Freeze a new shadow model or update its prediction ledger.",
    )
    parser.add_argument("--training-cutoff", default=None)
    parser.add_argument("--tickers", nargs="+", default=None)
    parser.add_argument(
        "--artifact-directory",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT_PATH,
    )
    return parser


def main() -> None:
    arguments = _build_parser().parse_args()
    engine = get_database_engine()

    if arguments.command == "freeze":
        dataset = build_event_feature_dataset(
            engine,
            horizon=SELECTED_HORIZON,
        )
        artifact = freeze_prospective_model(
            dataset,
            artifact_directory=arguments.artifact_directory,
            training_cutoff=arguments.training_cutoff,
        )
        print("\nFrozen prospective model")
        print(json.dumps({
            "version": artifact["version"],
            "training_cutoff": artifact["training_cutoff"],
            "training_rows": artifact["training_rows"],
            "status": artifact["status"],
        }, indent=2))
        return

    report = run_prospective_monitor(
        engine,
        artifact_directory=arguments.artifact_directory,
        report_path=arguments.report,
        tickers=arguments.tickers,
    )
    print("\nProspective monitor")
    print(json.dumps(_json_safe(report), indent=2, allow_nan=False))
    print(f"\nReport: {arguments.report}")


if __name__ == "__main__":
    main()
