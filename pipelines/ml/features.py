"""Point-in-time market, event, and sentiment features for AlphaLens."""

from __future__ import annotations

import argparse
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import text

from app.ml.financial_topics import (
    TOPIC_TAXONOMY,
    classify_financial_topics,
)
from app.ml.transcript_sentiment import (
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_REVISION,
    FILING_SENTIMENT_SECTION_KEYS,
)
from pipelines.ml.dataset import (
    build_modeling_targets,
    get_database_engine,
)


load_dotenv()


MARKET_FEATURE_COLUMNS = (
    "return_1d",
    "momentum_5d",
    "momentum_21d",
    "momentum_63d",
    "momentum_126d",
    "volatility_21d",
    "volatility_63d",
    "sma_ratio_20d",
    "sma_ratio_50d",
    "sma_ratio_126d",
    "volume_ratio_20d",
    "volume_zscore_20d",
    "drawdown_126d",
    "beta_63d",
    "spy_momentum_21d",
    "spy_momentum_63d",
    "spy_volatility_21d",
    "relative_momentum_21d",
    "relative_momentum_63d",
)

EVENT_FEATURE_COLUMNS = (
    "is_earnings_call",
    "is_sec_filing",
    "is_10k",
    "is_10q",
    "fiscal_quarter",
    "event_month_sin",
    "event_month_cos",
    "days_since_prior_company_event",
    "days_since_prior_source_event",
)

SENTIMENT_FEATURE_COLUMNS = (
    "sentiment_score",
    "sentiment_confidence",
    "sentiment_dispersion",
    "sentiment_coverage",
    "sentiment_scored_items",
    "sentiment_token_count",
    "management_sentiment_score",
    "analyst_sentiment_score",
    "management_analyst_gap",
    "sentiment_change_from_prior",
)

EARNINGS_RESULT_FEATURE_COLUMNS = (
    "eps_estimate",
    "reported_eps",
    "eps_surprise",
    "eps_surprise_pct",
    "eps_surprise_direction",
    "prior_eps_surprise_pct",
    "eps_surprise_change",
)

TOPIC_FEATURE_COLUMNS = tuple(
    column
    for topic in TOPIC_TAXONOMY
    for column in (
        f"topic_{topic.key}_sentiment",
        f"topic_{topic.key}_share",
    )
)


def model_feature_columns(include_topics: bool = True) -> tuple[str, ...]:
    """Return the audited feature allowlist used by later model training."""

    columns = (
        MARKET_FEATURE_COLUMNS
        + EVENT_FEATURE_COLUMNS
        + SENTIMENT_FEATURE_COLUMNS
        + EARNINGS_RESULT_FEATURE_COLUMNS
    )
    return columns + TOPIC_FEATURE_COLUMNS if include_topics else columns


def load_feature_prices(engine) -> pd.DataFrame:
    """Load historical fields used by trailing market indicators."""

    return pd.read_sql_query(
        text("""
            SELECT ticker, trading_date, adjusted_close, volume
            FROM market_prices
            ORDER BY ticker, trading_date
        """),
        engine,
    )


def load_earnings_results(engine) -> pd.DataFrame:
    """Load announced EPS results used by earnings-call event features."""

    return pd.read_sql_query(
        text("""
            SELECT
                ticker,
                earnings_date,
                earnings_timestamp,
                eps_estimate,
                reported_eps,
                eps_surprise,
                eps_surprise_pct
            FROM earnings_results
            ORDER BY ticker, earnings_date
        """),
        engine,
    )


def attach_earnings_result_features(
    events: pd.DataFrame,
    results: pd.DataFrame,
    *,
    max_match_days: int = 3,
) -> pd.DataFrame:
    """Attach the nearest observable result to each earnings-call event.

    Calls and provider result timestamps can differ by a calendar day because
    of time zones and after-hours scheduling. A bounded nearest-date match
    handles that discrepancy while the anchor-date check prevents a result
    that was not yet observable from becoming a feature.
    """

    required_events = {
        "event_key",
        "event_source",
        "ticker",
        "event_date",
        "anchor_trading_date",
    }
    required_results = {
        "ticker",
        "earnings_date",
        "eps_estimate",
        "reported_eps",
        "eps_surprise",
        "eps_surprise_pct",
    }

    if not required_events.issubset(events.columns):
        raise ValueError("events is missing earnings-result join columns.")

    if not required_results.issubset(results.columns):
        raise ValueError("results is missing earnings-result feature columns.")

    if max_match_days < 0:
        raise ValueError("max_match_days cannot be negative.")

    output = events.copy()
    output["event_date"] = pd.to_datetime(
        output["event_date"],
        errors="coerce",
    )
    output["anchor_trading_date"] = pd.to_datetime(
        output["anchor_trading_date"],
        errors="coerce",
    )
    normalized = results.copy()
    normalized["ticker"] = normalized["ticker"].astype(str).str.upper()
    normalized["earnings_date"] = pd.to_datetime(
        normalized["earnings_date"],
        errors="coerce",
    )
    normalized = normalized.dropna(subset=[
        "ticker",
        "earnings_date",
        "reported_eps",
    ]).sort_values(
        ["ticker", "earnings_date"],
        kind="stable",
    )
    normalized["prior_eps_surprise_pct"] = (
        normalized.groupby("ticker")["eps_surprise_pct"].shift()
    )
    normalized["eps_surprise_change"] = (
        normalized["eps_surprise_pct"]
        - normalized["prior_eps_surprise_pct"]
    )
    normalized["eps_surprise_direction"] = np.sign(
        pd.to_numeric(normalized["eps_surprise_pct"], errors="coerce")
    )

    feature_rows = []
    by_ticker = {
        ticker: frame
        for ticker, frame in normalized.groupby("ticker", sort=False)
    }

    for event in output.itertuples(index=False):
        empty = {
            "earnings_result_date": pd.NaT,
            "earnings_result_match_days": float("nan"),
            **{
                column: float("nan")
                for column in EARNINGS_RESULT_FEATURE_COLUMNS
            },
        }

        if (
            event.event_source != "earnings_call"
            or pd.isna(event.event_date)
            or pd.isna(event.anchor_trading_date)
        ):
            feature_rows.append(empty)
            continue

        candidates = by_ticker.get(str(event.ticker).upper())

        if candidates is None:
            feature_rows.append(empty)
            continue

        eligible = candidates.loc[
            candidates["earnings_date"] <= event.anchor_trading_date
        ].copy()
        eligible["match_days"] = (
            eligible["earnings_date"] - event.event_date
        ).dt.days.abs()
        eligible = eligible.loc[eligible["match_days"] <= max_match_days]

        if eligible.empty:
            feature_rows.append(empty)
            continue

        match = eligible.sort_values(
            ["match_days", "earnings_date"],
            ascending=[True, False],
            kind="stable",
        ).iloc[0]
        feature_rows.append({
            "earnings_result_date": match["earnings_date"],
            "earnings_result_match_days": int(match["match_days"]),
            **{
                column: match[column]
                for column in EARNINGS_RESULT_FEATURE_COLUMNS
            },
        })

    return pd.concat(
        [output.reset_index(drop=True), pd.DataFrame(feature_rows)],
        axis=1,
    )


def _validate_price_history(prices: pd.DataFrame) -> pd.DataFrame:
    """Normalize market history while preserving one row per session."""

    required = {"ticker", "trading_date", "adjusted_close", "volume"}
    missing = required - set(prices.columns)

    if missing:
        raise ValueError(
            "prices is missing columns: " + ", ".join(sorted(missing))
        )

    normalized = prices.copy()
    normalized["ticker"] = normalized["ticker"].astype(str).str.upper()
    normalized["trading_date"] = pd.to_datetime(
        normalized["trading_date"],
        errors="coerce",
    )
    normalized["adjusted_close"] = pd.to_numeric(
        normalized["adjusted_close"],
        errors="coerce",
    )
    normalized["volume"] = pd.to_numeric(
        normalized["volume"],
        errors="coerce",
    )

    if normalized.duplicated(["ticker", "trading_date"]).any():
        raise ValueError("prices contains duplicate ticker/date rows.")

    return normalized.sort_values(
        ["ticker", "trading_date"],
        kind="stable",
    ).reset_index(drop=True)


def calculate_market_feature_panel(
    prices: pd.DataFrame,
    benchmark: str = "SPY",
) -> pd.DataFrame:
    """Calculate trailing indicators for every ticker and trading session."""

    normalized = _validate_price_history(prices)
    frames = []

    for ticker, group in normalized.groupby("ticker", sort=False):
        frame = group.copy().set_index("trading_date")
        adjusted_close = frame["adjusted_close"]
        daily_return = adjusted_close.pct_change(fill_method=None)
        frame["return_1d"] = daily_return

        for window in (5, 21, 63, 126):
            frame[f"momentum_{window}d"] = adjusted_close.pct_change(
                periods=window,
                fill_method=None,
            )

        for window in (21, 63):
            frame[f"volatility_{window}d"] = (
                daily_return.rolling(window, min_periods=window).std()
                * math.sqrt(252)
            )

        for window in (20, 50, 126):
            moving_average = adjusted_close.rolling(
                window,
                min_periods=window,
            ).mean()
            frame[f"sma_ratio_{window}d"] = (
                adjusted_close / moving_average - 1.0
            )

        volume_mean = frame["volume"].rolling(20, min_periods=20).mean()
        volume_std = frame["volume"].rolling(20, min_periods=20).std()
        frame["volume_ratio_20d"] = frame["volume"] / volume_mean
        frame["volume_zscore_20d"] = (
            (frame["volume"] - volume_mean) / volume_std.replace(0, pd.NA)
        )
        rolling_high = adjusted_close.rolling(126, min_periods=126).max()
        frame["drawdown_126d"] = adjusted_close / rolling_high - 1.0
        frames.append(frame.reset_index())

    panel = pd.concat(frames, ignore_index=True)
    benchmark = benchmark.strip().upper()
    benchmark_frame = panel.loc[
        panel["ticker"] == benchmark,
        [
            "trading_date",
            "return_1d",
            "momentum_21d",
            "momentum_63d",
            "volatility_21d",
        ],
    ].rename(columns={
        "return_1d": "spy_return_1d",
        "momentum_21d": "spy_momentum_21d",
        "momentum_63d": "spy_momentum_63d",
        "volatility_21d": "spy_volatility_21d",
    })

    if benchmark_frame.empty:
        raise ValueError(f"Benchmark {benchmark} is missing from prices.")

    panel = panel.merge(
        benchmark_frame,
        on="trading_date",
        how="left",
        validate="many_to_one",
    )
    panel["relative_momentum_21d"] = (
        panel["momentum_21d"] - panel["spy_momentum_21d"]
    )
    panel["relative_momentum_63d"] = (
        panel["momentum_63d"] - panel["spy_momentum_63d"]
    )

    beta_frames = []
    for _, group in panel.groupby("ticker", sort=False):
        frame = group.sort_values("trading_date").copy()
        covariance = frame["return_1d"].rolling(63, min_periods=63).cov(
            frame["spy_return_1d"]
        )
        benchmark_variance = frame["spy_return_1d"].rolling(
            63,
            min_periods=63,
        ).var()
        frame["beta_63d"] = covariance / benchmark_variance.replace(0, pd.NA)
        beta_frames.append(frame)

    return pd.concat(beta_frames, ignore_index=True).sort_values(
        ["ticker", "trading_date"],
        kind="stable",
    ).reset_index(drop=True)


def attach_market_features(
    events: pd.DataFrame,
    panel: pd.DataFrame,
) -> pd.DataFrame:
    """Join indicators from the exact feature-as-of session of each event."""

    required_events = {"event_key", "ticker", "anchor_trading_date"}
    required_panel = {"ticker", "trading_date", *MARKET_FEATURE_COLUMNS}

    if not required_events.issubset(events.columns):
        raise ValueError("events is missing market-join columns.")

    if not required_panel.issubset(panel.columns):
        raise ValueError("panel is missing calculated market features.")

    feature_panel = panel[[
        "ticker",
        "trading_date",
        *MARKET_FEATURE_COLUMNS,
    ]].rename(columns={"trading_date": "feature_as_of_date"})
    result = events.copy()
    result["anchor_trading_date"] = pd.to_datetime(
        result["anchor_trading_date"],
        errors="coerce",
    )
    result = result.merge(
        feature_panel,
        left_on=["ticker", "anchor_trading_date"],
        right_on=["ticker", "feature_as_of_date"],
        how="left",
        validate="many_to_one",
    )
    return result


def load_sentiment_items(engine) -> pd.DataFrame:
    """Load eligible event text with results from the configured FinBERT."""

    model_name = os.getenv("FINBERT_MODEL_NAME", DEFAULT_MODEL_NAME)
    model_revision = os.getenv(
        "FINBERT_MODEL_REVISION",
        DEFAULT_MODEL_REVISION,
    )
    filing_sections = ", ".join(
        f"'{section}'" for section in FILING_SENTIMENT_SECTION_KEYS
    )
    query = text(f"""
        SELECT
            CONCAT('earnings_call:', turns.transcript_id) AS event_key,
            'earnings_call' AS event_source,
            CASE
                WHEN LOWER(CONCAT_WS(
                    ' ', turns.speaker_name, turns.speaker_role,
                    turns.speaker_title
                )) LIKE '%analyst%'
                THEN 'analyst'
                ELSE 'management'
            END AS speaker_group,
            turns.content,
            scores.status,
            scores.sentiment_score,
            scores.confidence,
            scores.token_count
        FROM earnings_transcript_turns turns
        LEFT JOIN earnings_transcript_turn_sentiment scores
          ON scores.turn_id = turns.turn_id
         AND scores.model_name = :model_name
         AND scores.model_revision = :model_revision
        WHERE NULLIF(BTRIM(turns.content), '') IS NOT NULL
          AND LOWER(CONCAT_WS(
              ' ', turns.speaker_name, turns.speaker_role,
              turns.speaker_title
          )) NOT LIKE '%operator%'

        UNION ALL

        SELECT
            CONCAT('sec_filing:', chunks.accession_number) AS event_key,
            'sec_filing' AS event_source,
            'filing' AS speaker_group,
            chunks.content,
            scores.status,
            scores.sentiment_score,
            scores.confidence,
            scores.token_count
        FROM filing_chunks chunks
        LEFT JOIN filing_chunk_sentiment scores
          ON scores.chunk_id = chunks.chunk_id
         AND scores.model_name = :model_name
         AND scores.model_revision = :model_revision
        WHERE chunks.section_key IN ({filing_sections})
    """)
    return pd.read_sql_query(
        query,
        engine,
        params={"model_name": model_name, "model_revision": model_revision},
    )


def _weighted_sentiment(rows: pd.DataFrame) -> dict:
    """Aggregate scored items without allowing tiny snippets to dominate."""

    scored = rows.loc[
        (rows["status"] == "SCORED")
        & rows["sentiment_score"].notna()
    ].copy()

    if scored.empty:
        return {
            "score": float("nan"),
            "confidence": float("nan"),
            "dispersion": float("nan"),
            "scored_items": 0,
            "token_count": 0,
        }

    weights = pd.to_numeric(
        scored["token_count"],
        errors="coerce",
    ).fillna(1).clip(lower=1)
    scores = pd.to_numeric(scored["sentiment_score"], errors="coerce")
    confidence = pd.to_numeric(scored["confidence"], errors="coerce")
    total_weight = float(weights.sum())
    score = float((scores * weights).sum() / total_weight)
    variance = float((((scores - score) ** 2) * weights).sum() / total_weight)
    return {
        "score": score,
        "confidence": float((confidence * weights).sum() / total_weight),
        "dispersion": math.sqrt(max(0.0, variance)),
        "scored_items": int(len(scored)),
        "token_count": int(total_weight),
    }


def aggregate_sentiment_features(
    items: pd.DataFrame,
    include_topics: bool = True,
) -> pd.DataFrame:
    """Create one auditable FinBERT feature row for each source event."""

    required = {
        "event_key",
        "event_source",
        "speaker_group",
        "content",
        "status",
        "sentiment_score",
        "confidence",
        "token_count",
    }
    if not required.issubset(items.columns):
        raise ValueError("items is missing sentiment aggregation columns.")

    feature_rows = []

    for event_key, group in items.groupby("event_key", sort=False):
        overall = _weighted_sentiment(group)
        management = _weighted_sentiment(
            group[group["speaker_group"] == "management"]
        )
        analyst = _weighted_sentiment(
            group[group["speaker_group"] == "analyst"]
        )
        row = {
            "event_key": event_key,
            "sentiment_score": overall["score"],
            "sentiment_confidence": overall["confidence"],
            "sentiment_dispersion": overall["dispersion"],
            "sentiment_coverage": (
                overall["scored_items"] / len(group) if len(group) else 0.0
            ),
            "sentiment_scored_items": overall["scored_items"],
            "sentiment_token_count": overall["token_count"],
            "management_sentiment_score": management["score"],
            "analyst_sentiment_score": analyst["score"],
            "management_analyst_gap": (
                management["score"] - analyst["score"]
                if not pd.isna(management["score"])
                and not pd.isna(analyst["score"])
                else float("nan")
            ),
        }

        if include_topics:
            topic_rows = defaultdict(list)
            topic_population = group.loc[
                (group["status"] == "SCORED")
                & (
                    (group["event_source"] == "sec_filing")
                    | (group["speaker_group"] == "management")
                )
            ]

            for item in topic_population.itertuples(index=False):
                for match in classify_financial_topics(item.content):
                    topic_rows[match["topic_key"]].append({
                        "score": float(item.sentiment_score),
                        "weight": max(1, int(item.token_count or 1)),
                    })

            population_count = len(topic_population)
            for topic in TOPIC_TAXONOMY:
                matches = topic_rows.get(topic.key, [])
                score_column = f"topic_{topic.key}_sentiment"
                share_column = f"topic_{topic.key}_share"

                if matches:
                    total_weight = sum(item["weight"] for item in matches)
                    row[score_column] = sum(
                        item["score"] * item["weight"]
                        for item in matches
                    ) / total_weight
                    row[share_column] = len(matches) / population_count
                else:
                    row[score_column] = float("nan")
                    row[share_column] = 0.0

        feature_rows.append(row)

    return pd.DataFrame(feature_rows)


def add_event_features(events: pd.DataFrame) -> pd.DataFrame:
    """Add calendar and spacing features available by each event date."""

    result = events.copy()
    result["event_date"] = pd.to_datetime(result["event_date"], errors="coerce")
    result["is_earnings_call"] = (
        result["event_source"] == "earnings_call"
    ).astype(int)
    result["is_sec_filing"] = (
        result["event_source"] == "sec_filing"
    ).astype(int)
    normalized_form = result["form_type"].fillna("").str.upper()
    result["is_10k"] = (normalized_form == "10-K").astype(int)
    result["is_10q"] = (normalized_form == "10-Q").astype(int)
    result["fiscal_quarter"] = pd.to_numeric(
        result["fiscal_period"].str.extract(r"Q([1-4])", expand=False),
        errors="coerce",
    )
    month = result["event_date"].dt.month
    result["event_month_sin"] = (2 * math.pi * month / 12).map(math.sin)
    result["event_month_cos"] = (2 * math.pi * month / 12).map(math.cos)

    ordered = result.sort_values(
        ["ticker", "event_date", "event_source", "event_id"],
        kind="stable",
    )
    ordered["days_since_prior_company_event"] = (
        ordered.groupby("ticker")["event_date"].diff().dt.days
    )
    ordered["days_since_prior_source_event"] = (
        ordered.groupby(["ticker", "event_source"])["event_date"]
        .diff()
        .dt.days
    )
    return ordered.sort_index()


def build_event_feature_dataset(
    engine=None,
    *,
    horizon: int = 30,
    include_topics: bool = True,
) -> pd.DataFrame:
    """Build targets and point-in-time features from the current corpus."""

    resolved_engine = engine or get_database_engine()
    targets = build_modeling_targets(resolved_engine, horizon=horizon)
    panel = calculate_market_feature_panel(
        load_feature_prices(resolved_engine)
    )
    featured = attach_market_features(targets, panel)
    featured = attach_earnings_result_features(
        featured,
        load_earnings_results(resolved_engine),
    )
    sentiment = aggregate_sentiment_features(
        load_sentiment_items(resolved_engine),
        include_topics=include_topics,
    )
    featured = featured.merge(
        sentiment,
        on="event_key",
        how="left",
        validate="one_to_one",
    )
    featured = add_event_features(featured)
    featured = featured.sort_values(
        ["event_date", "ticker", "event_source", "event_id"],
        kind="stable",
    ).reset_index(drop=True)
    featured["sentiment_change_from_prior"] = (
        featured.groupby(["ticker", "event_source"])["sentiment_score"]
        .diff()
    )
    validate_feature_dataset(featured, include_topics=include_topics)
    return featured


def validate_feature_dataset(
    dataset: pd.DataFrame,
    *,
    include_topics: bool = True,
) -> dict[str, int]:
    """Validate feature chronology and ensure the allowlist is present."""

    missing = set(model_feature_columns(include_topics)) - set(dataset.columns)
    if missing:
        raise ValueError(
            "Feature dataset is missing columns: "
            + ", ".join(sorted(missing))
        )

    anchored = dataset["feature_as_of_date"].notna()
    matched_results = dataset["earnings_result_date"].notna()
    checks = {
        "rows": len(dataset),
        "labeled_rows": int(dataset["target_available"].sum()),
        "duplicate_event_keys": int(dataset["event_key"].duplicated().sum()),
        "feature_date_mismatch": int((
            dataset.loc[anchored, "feature_as_of_date"]
            != dataset.loc[anchored, "anchor_trading_date"]
        ).sum()),
        "feature_date_not_after_event": int((
            dataset.loc[anchored, "feature_as_of_date"]
            <= dataset.loc[anchored, "event_date"]
        ).sum()),
        "earnings_result_after_anchor": int((
            dataset.loc[matched_results, "earnings_result_date"]
            > dataset.loc[matched_results, "anchor_trading_date"]
        ).sum()),
        "earnings_result_on_non_call": int((
            dataset.loc[matched_results, "event_source"]
            != "earnings_call"
        ).sum()),
        "infinite_feature_values": int(np.isinf(
            dataset[list(model_feature_columns(include_topics))]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(dtype=float)
        ).sum()),
    }
    failures = {
        key: value
        for key, value in checks.items()
        if key not in {"rows", "labeled_rows"} and value
    }

    if failures:
        raise ValueError(f"Feature dataset validation failed: {failures}")

    return checks


def summarize_feature_coverage(
    dataset: pd.DataFrame,
    *,
    include_topics: bool = True,
) -> pd.DataFrame:
    """Report non-null coverage for every approved model feature."""

    labeled = dataset[dataset["target_available"]]
    rows = []

    for column in model_feature_columns(include_topics):
        non_null = int(labeled[column].notna().sum())
        rows.append({
            "feature": column,
            "non_null_rows": non_null,
            "missing_rows": len(labeled) - non_null,
            "coverage": round(non_null / len(labeled), 4) if len(labeled) else 0,
        })

    return pd.DataFrame(rows).sort_values(
        ["coverage", "feature"],
        ascending=[True, True],
    ).reset_index(drop=True)


def main() -> None:
    """Build the feature dataset and optionally save an ignored local CSV."""

    parser = argparse.ArgumentParser(
        description="Build point-in-time AlphaLens model features.",
    )
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--without-topics", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    arguments = parser.parse_args()
    include_topics = not arguments.without_topics
    dataset = build_event_feature_dataset(
        horizon=arguments.horizon,
        include_topics=include_topics,
    )
    coverage = summarize_feature_coverage(
        dataset,
        include_topics=include_topics,
    )

    print(
        f"Built {len(dataset):,} event rows with "
        f"{int(dataset['target_available'].sum()):,} labels and "
        f"{len(model_feature_columns(include_topics))} approved features."
    )
    print("\nLowest feature coverage")
    print(coverage.head(15).to_string(index=False))

    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_csv(arguments.output, index=False)
        print(f"\nDataset: {arguments.output}")


if __name__ == "__main__":
    main()
