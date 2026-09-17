"""Database-backed coverage audit for the AlphaLens modeling corpus."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect, text

from app.ml.transcript_sentiment import (
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_REVISION,
)


load_dotenv()


DATASETS = (
    ("companies", "reference", True),
    ("market_prices", "market", True),
    ("filings", "SEC events", True),
    ("filing_chunks", "SEC text", True),
    ("filing_chunk_sentiment", "SEC sentiment", True),
    ("earnings_transcripts", "earnings events", True),
    ("earnings_transcript_turns", "earnings text", True),
    (
        "earnings_transcript_turn_sentiment",
        "earnings sentiment",
        True,
    ),
    # The planned earnings-surprise feature needs a dedicated point-in-time
    # source. Keeping it in the inventory makes that gap visible in reports.
    ("earnings_results", "earnings fundamentals", False),
)

FILING_SENTIMENT_SECTIONS = (
    "item_7_mda",
    "part1_item2_mda",
    "item_1a_risk_factors",
    "part2_item1a_risk_factors",
    "item_7a_market_risk",
    "part1_item3_market_risk",
)


@dataclass(frozen=True)
class DataAuditReport:
    """Tables and conclusions produced by one reproducible audit run."""

    generated_at: datetime
    inventory: pd.DataFrame
    market_coverage: pd.DataFrame
    market_quality: pd.DataFrame
    event_coverage: pd.DataFrame
    event_horizon_coverage: pd.DataFrame
    sentiment_coverage: pd.DataFrame
    findings: tuple[str, ...]

    def to_dict(self) -> dict:
        """Return a JSON-safe report for CI artifacts and later comparison."""

        return {
            "generated_at": self.generated_at.isoformat(),
            "inventory": _records(self.inventory),
            "market_coverage": _records(self.market_coverage),
            "market_quality": _records(self.market_quality),
            "event_coverage": _records(self.event_coverage),
            "event_horizon_coverage": _records(
                self.event_horizon_coverage
            ),
            "sentiment_coverage": _records(self.sentiment_coverage),
            "findings": list(self.findings),
        }


def _records(frame: pd.DataFrame) -> list[dict]:
    """Serialize pandas timestamps and nullable values consistently."""

    return json.loads(frame.to_json(orient="records", date_format="iso"))


def get_database_engine(database_url: str | None = None):
    """Create the PostgreSQL engine used by notebooks and CLI audits."""

    resolved_url = database_url or os.getenv("DATABASE_URL")

    if not resolved_url:
        raise ValueError("DATABASE_URL was not found in .env.")

    return create_engine(resolved_url, pool_pre_ping=True)


def load_inventory(engine) -> pd.DataFrame:
    """Report which modeling datasets exist and their current row counts."""

    inspector = inspect(engine)
    rows = []

    with engine.connect() as connection:
        for table_name, role, required in DATASETS:
            available = inspector.has_table(table_name)
            row_count = None

            if available:
                # Names come only from the fixed DATASETS constant above.
                row_count = int(connection.execute(
                    text(f'SELECT COUNT(*) FROM "{table_name}"')
                ).scalar_one())

            rows.append({
                "dataset": table_name,
                "role": role,
                "required_for_audit": required,
                "available": available,
                "rows": row_count,
            })

    return pd.DataFrame(rows)


def load_market_coverage(engine) -> pd.DataFrame:
    """Summarize date and field coverage for each market ticker."""

    query = text("""
        SELECT
            ticker,
            COUNT(*)::BIGINT AS rows,
            MIN(trading_date) AS first_date,
            MAX(trading_date) AS last_date,
            COUNT(adjusted_close)::BIGINT AS adjusted_close_rows,
            COUNT(volume)::BIGINT AS volume_rows
        FROM market_prices
        GROUP BY ticker
        ORDER BY ticker
    """)
    return pd.read_sql_query(query, engine)


def load_market_quality(engine) -> pd.DataFrame:
    """Return high-signal OHLCV integrity checks as metric/value rows."""

    query = text("""
        SELECT
            COUNT(*)::BIGINT AS total_rows,
            COUNT(*) FILTER (
                WHERE open IS NULL OR high IS NULL OR low IS NULL
                   OR close IS NULL OR adjusted_close IS NULL
            )::BIGINT AS missing_price_rows,
            COUNT(*) FILTER (
                WHERE open <= 0 OR high <= 0 OR low <= 0
                   OR close <= 0 OR adjusted_close <= 0
            )::BIGINT AS nonpositive_price_rows,
            COUNT(*) FILTER (
                WHERE high < low OR high < open OR high < close
                   OR low > open OR low > close
            )::BIGINT AS invalid_ohlc_rows,
            COUNT(*) FILTER (
                WHERE volume IS NULL OR volume < 0
            )::BIGINT AS invalid_volume_rows,
            COUNT(*) FILTER (
                WHERE ticker <> 'SPY' AND NOT EXISTS (
                    SELECT 1
                    FROM market_prices benchmark
                    WHERE benchmark.ticker = 'SPY'
                      AND benchmark.trading_date = market_prices.trading_date
                )
            )::BIGINT AS rows_without_spy
        FROM market_prices
    """)
    result = pd.read_sql_query(query, engine).iloc[0]
    return pd.DataFrame({
        "metric": result.index,
        "value": [int(value) for value in result.values],
    })


def load_event_coverage(engine) -> pd.DataFrame:
    """Combine filing and earnings-call coverage into one ticker table."""

    query = text("""
        WITH transcript_summary AS (
            SELECT
                ticker,
                COUNT(*)::BIGINT AS transcripts,
                COUNT(call_date)::BIGINT AS dated_transcripts,
                MIN(call_date) AS first_call_date,
                MAX(call_date) AS last_call_date
            FROM earnings_transcripts
            GROUP BY ticker
        ),
        filing_summary AS (
            SELECT
                ticker,
                COUNT(*) FILTER (WHERE form_type = '10-K')::BIGINT
                    AS filings_10k,
                COUNT(*) FILTER (WHERE form_type = '10-Q')::BIGINT
                    AS filings_10q,
                MIN(filing_date) AS first_filing_date,
                MAX(filing_date) AS last_filing_date
            FROM filings
            WHERE form_type IN ('10-K', '10-Q')
            GROUP BY ticker
        )
        SELECT
            companies.ticker,
            COALESCE(transcript_summary.transcripts, 0)::BIGINT
                AS transcripts,
            COALESCE(transcript_summary.dated_transcripts, 0)::BIGINT
                AS dated_transcripts,
            transcript_summary.first_call_date,
            transcript_summary.last_call_date,
            COALESCE(filing_summary.filings_10k, 0)::BIGINT
                AS filings_10k,
            COALESCE(filing_summary.filings_10q, 0)::BIGINT
                AS filings_10q,
            filing_summary.first_filing_date,
            filing_summary.last_filing_date
        FROM companies
        LEFT JOIN transcript_summary USING (ticker)
        LEFT JOIN filing_summary USING (ticker)
        ORDER BY companies.ticker
    """)
    return pd.read_sql_query(query, engine)


def load_event_dates(engine) -> pd.DataFrame:
    """Load observable event dates without introducing fiscal-period leakage."""

    query = text("""
        SELECT
            'earnings_call' AS source,
            transcript_id::TEXT AS event_id,
            ticker,
            call_date AS event_date
        FROM earnings_transcripts
        UNION ALL
        SELECT
            'sec_filing' AS source,
            accession_number AS event_id,
            ticker,
            filing_date AS event_date
        FROM filings
        WHERE form_type IN ('10-K', '10-Q')
        ORDER BY ticker, event_date, source
    """)
    return pd.read_sql_query(query, engine)


def load_price_calendar(engine) -> pd.DataFrame:
    """Load only the fields required to assess forward target availability."""

    return pd.read_sql_query(
        text("""
            SELECT ticker, trading_date
            FROM market_prices
            WHERE ticker <> 'SPY'
            ORDER BY ticker, trading_date
        """),
        engine,
    )


def assess_event_horizon(
    events: pd.DataFrame,
    price_calendar: pd.DataFrame,
    horizon: int = 30,
    max_anchor_gap_days: int = 7,
) -> pd.DataFrame:
    """Mark events that have an anchor and a complete forward-day horizon."""

    if horizon < 1:
        raise ValueError("horizon must be at least one trading day.")

    if max_anchor_gap_days < 1:
        raise ValueError("max_anchor_gap_days must be at least one day.")

    required_event_columns = {"source", "event_id", "ticker", "event_date"}
    required_price_columns = {"ticker", "trading_date"}

    if not required_event_columns.issubset(events.columns):
        raise ValueError("events is missing required event columns.")

    if not required_price_columns.issubset(price_calendar.columns):
        raise ValueError("price_calendar is missing required price columns.")

    result = events.copy()
    result["event_date"] = pd.to_datetime(result["event_date"])
    calendar = price_calendar.copy()
    calendar["trading_date"] = pd.to_datetime(calendar["trading_date"])
    dates_by_ticker = {
        ticker: pd.DatetimeIndex(group["trading_date"].dropna().unique()).sort_values()
        for ticker, group in calendar.groupby("ticker", sort=False)
    }

    anchors = []
    horizon_dates = []
    available = []

    for event in result.itertuples(index=False):
        dates = dates_by_ticker.get(event.ticker, pd.DatetimeIndex([]))

        if pd.isna(event.event_date) or dates.empty:
            anchors.append(pd.NaT)
            horizon_dates.append(pd.NaT)
            available.append(False)
            continue

        # Event dates do not reliably preserve pre-market versus after-hours
        # timing. Start on the next session so same-day prices cannot leak.
        anchor_index = int(dates.searchsorted(event.event_date, side="right"))
        target_index = anchor_index + horizon

        if anchor_index >= len(dates):
            anchors.append(pd.NaT)
            horizon_dates.append(pd.NaT)
            available.append(False)
            continue

        anchors.append(dates[anchor_index])

        anchor_gap_days = int(
            (dates[anchor_index].normalize() - event.event_date.normalize()).days
        )
        if anchor_gap_days > max_anchor_gap_days:
            horizon_dates.append(pd.NaT)
            available.append(False)
            continue

        if target_index < len(dates):
            horizon_dates.append(dates[target_index])
            available.append(True)
        else:
            horizon_dates.append(pd.NaT)
            available.append(False)

    result["anchor_trading_date"] = anchors
    result[f"horizon_{horizon}d_date"] = horizon_dates
    result[f"forward_{horizon}d_available"] = available
    return result


def summarize_event_horizon(
    assessed_events: pd.DataFrame,
    horizon: int = 30,
) -> pd.DataFrame:
    """Aggregate event-level horizon checks for a concise audit view."""

    availability_column = f"forward_{horizon}d_available"

    if availability_column not in assessed_events.columns:
        raise ValueError(
            f"assessed_events is missing {availability_column}."
        )

    summary = (
        assessed_events.assign(
            dated=assessed_events["event_date"].notna(),
        )
        .groupby("source", as_index=False)
        .agg(
            events=("event_id", "size"),
            dated_events=("dated", "sum"),
            forward_horizon_available=(availability_column, "sum"),
        )
    )
    summary["forward_horizon_coverage"] = (
        summary["forward_horizon_available"] / summary["events"]
    ).round(4)
    summary["horizon_trading_days"] = horizon
    return summary


def load_sentiment_coverage(engine) -> pd.DataFrame:
    """Measure current pinned-model coverage for both narrative corpora."""

    model_name = os.getenv("FINBERT_MODEL_NAME", DEFAULT_MODEL_NAME)
    model_revision = os.getenv(
        "FINBERT_MODEL_REVISION",
        DEFAULT_MODEL_REVISION,
    )
    section_list = ", ".join(
        f"'{section}'" for section in FILING_SENTIMENT_SECTIONS
    )
    query = text(f"""
        WITH transcript_scope AS (
            SELECT
                turns.turn_id,
                turns.content,
                LOWER(CONCAT_WS(
                    ' ',
                    turns.speaker_name,
                    turns.speaker_role,
                    turns.speaker_title
                )) LIKE '%operator%' AS is_operator
            FROM earnings_transcript_turns turns
        ),
        transcript_summary AS (
            SELECT
                'transcript_turns' AS source,
                COUNT(*)::BIGINT AS total_items,
                COUNT(*) FILTER (
                    WHERE NULLIF(BTRIM(scope.content), '') IS NOT NULL
                      AND NOT scope.is_operator
                )::BIGINT AS eligible_items,
                COUNT(*) FILTER (WHERE scores.status = 'SCORED')::BIGINT
                    AS scored_items,
                COUNT(*) FILTER (WHERE scores.status = 'SKIPPED')::BIGINT
                    AS skipped_items,
                COUNT(*) FILTER (WHERE scores.status = 'FAILED')::BIGINT
                    AS failed_items
            FROM transcript_scope scope
            LEFT JOIN earnings_transcript_turn_sentiment scores
              ON scores.turn_id = scope.turn_id
             AND scores.model_name = :model_name
             AND scores.model_revision = :model_revision
        ),
        filing_scope AS (
            SELECT chunk_id
            FROM filing_chunks
            WHERE section_key IN ({section_list})
        ),
        filing_summary AS (
            SELECT
                'filing_chunks' AS source,
                COUNT(*)::BIGINT AS total_items,
                COUNT(*)::BIGINT AS eligible_items,
                COUNT(*) FILTER (WHERE scores.status = 'SCORED')::BIGINT
                    AS scored_items,
                0::BIGINT AS skipped_items,
                COUNT(*) FILTER (WHERE scores.status = 'FAILED')::BIGINT
                    AS failed_items
            FROM filing_scope scope
            LEFT JOIN filing_chunk_sentiment scores
              ON scores.chunk_id = scope.chunk_id
             AND scores.model_name = :model_name
             AND scores.model_revision = :model_revision
        )
        SELECT * FROM transcript_summary
        UNION ALL
        SELECT * FROM filing_summary
        ORDER BY source
    """)
    coverage = pd.read_sql_query(
        query,
        engine,
        params={"model_name": model_name, "model_revision": model_revision},
    )
    coverage["coverage"] = (
        coverage["scored_items"] / coverage["eligible_items"].replace(0, pd.NA)
    ).fillna(0.0).round(4)
    coverage["model_name"] = model_name
    coverage["model_revision"] = model_revision
    return coverage


def build_findings(
    inventory: pd.DataFrame,
    market_coverage: pd.DataFrame,
    market_quality: pd.DataFrame,
    event_horizon_coverage: pd.DataFrame,
    sentiment_coverage: pd.DataFrame,
) -> tuple[str, ...]:
    """Turn audit tables into explicit modeling decisions and blockers."""

    findings = []
    unavailable = inventory.loc[~inventory["available"], "dataset"].tolist()

    if "earnings_results" in unavailable:
        findings.append(
            "Earnings-surprise features are not available yet; add a "
            "point-in-time earnings-results source before using them."
        )

    equities = market_coverage[market_coverage["ticker"] != "SPY"]
    if not equities.empty:
        findings.append(
            f"Market history covers {len(equities)} equities plus SPY from "
            f"{market_coverage['first_date'].min()} through "
            f"{market_coverage['last_date'].max()}."
        )

    quality = dict(zip(
        market_quality["metric"],
        market_quality["value"],
        strict=True,
    ))
    issue_metrics = (
        "missing_price_rows",
        "nonpositive_price_rows",
        "invalid_ohlc_rows",
        "invalid_volume_rows",
        "rows_without_spy",
    )
    issue_count = sum(int(quality.get(metric, 0)) for metric in issue_metrics)
    findings.append(
        "Core OHLCV integrity checks passed with no flagged rows."
        if issue_count == 0
        else f"Core OHLCV checks flagged {issue_count:,} row-level issues."
    )

    for row in event_horizon_coverage.itertuples(index=False):
        findings.append(
            f"{row.source} has {int(row.forward_horizon_available):,} of "
            f"{int(row.events):,} events with a complete "
            f"{int(row.horizon_trading_days)}-trading-day forward window."
        )

    for row in sentiment_coverage.itertuples(index=False):
        findings.append(
            f"{row.source} FinBERT coverage is {float(row.coverage):.1%} "
            f"({int(row.scored_items):,}/{int(row.eligible_items):,} "
            "eligible items)."
        )

    findings.append(
        "Use filing_date and call_date as event availability dates; fiscal "
        "period and report_date must not be used as feature timestamps."
    )
    return tuple(findings)


def run_data_audit(engine=None, horizon: int = 30) -> DataAuditReport:
    """Execute the complete modeling-data audit against PostgreSQL."""

    resolved_engine = engine or get_database_engine()
    inventory = load_inventory(resolved_engine)
    missing_required = inventory.loc[
        inventory["required_for_audit"] & ~inventory["available"],
        "dataset",
    ].tolist()

    if missing_required:
        raise RuntimeError(
            "Required modeling tables are missing: "
            + ", ".join(missing_required)
        )

    market_coverage = load_market_coverage(resolved_engine)
    market_quality = load_market_quality(resolved_engine)
    event_coverage = load_event_coverage(resolved_engine)
    assessed_events = assess_event_horizon(
        load_event_dates(resolved_engine),
        load_price_calendar(resolved_engine),
        horizon=horizon,
    )
    event_horizon_coverage = summarize_event_horizon(
        assessed_events,
        horizon=horizon,
    )
    sentiment_coverage = load_sentiment_coverage(resolved_engine)
    findings = build_findings(
        inventory,
        market_coverage,
        market_quality,
        event_horizon_coverage,
        sentiment_coverage,
    )
    return DataAuditReport(
        generated_at=datetime.now(timezone.utc),
        inventory=inventory,
        market_coverage=market_coverage,
        market_quality=market_quality,
        event_coverage=event_coverage,
        event_horizon_coverage=event_horizon_coverage,
        sentiment_coverage=sentiment_coverage,
        findings=findings,
    )


def main() -> None:
    """Run the audit from PowerShell and optionally persist a JSON snapshot."""

    parser = argparse.ArgumentParser(
        description="Audit AlphaLens data before building ML features.",
    )
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--output", type=Path, default=None)
    arguments = parser.parse_args()
    report = run_data_audit(horizon=arguments.horizon)

    print("\nDataset inventory")
    print(report.inventory.to_string(index=False))
    print("\n30-trading-day event coverage")
    print(report.event_horizon_coverage.to_string(index=False))
    print("\nFinBERT coverage")
    print(report.sentiment_coverage.to_string(index=False))
    print("\nFindings")
    for finding in report.findings:
        print(f"- {finding}")

    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(
            json.dumps(report.to_dict(), indent=2),
            encoding="utf-8",
        )
        print(f"\nReport: {arguments.output}")


if __name__ == "__main__":
    main()
