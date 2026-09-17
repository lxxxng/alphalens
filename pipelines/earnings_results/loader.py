"""Idempotent PostgreSQL loader for announced earnings results."""

from __future__ import annotations

import pandas as pd
from sqlalchemy import MetaData, Table, func
from sqlalchemy.dialects.postgresql import insert

from pipelines.market_data.loader import get_database_engine


DATABASE_COLUMNS = (
    "ticker",
    "earnings_date",
    "earnings_timestamp",
    "eps_estimate",
    "reported_eps",
    "eps_surprise",
    "eps_surprise_pct",
    "source_provider",
    "source_url",
    "raw_payload",
)


def load_earnings_results(
    results: pd.DataFrame,
    *,
    engine=None,
    batch_size: int = 500,
) -> int:
    """Upsert normalized results by ticker, announcement date, and source."""

    if results.empty:
        return 0

    missing = set(DATABASE_COLUMNS) - set(results.columns)

    if missing:
        raise ValueError(
            "Earnings results are missing columns: "
            + ", ".join(sorted(missing))
        )

    if batch_size < 1:
        raise ValueError("batch_size must be at least one row.")

    resolved_engine = engine or get_database_engine()

    try:
        table = Table(
            "earnings_results",
            MetaData(),
            autoload_with=resolved_engine,
        )
    except Exception as error:
        raise RuntimeError(
            "earnings_results is unavailable. Apply "
            "db/sql/017_earnings_results.sql."
        ) from error

    database_rows = results[list(DATABASE_COLUMNS)].copy()
    database_rows = database_rows.astype(object).where(
        pd.notna(database_rows),
        None,
    )
    records = database_rows.to_dict(orient="records")

    with resolved_engine.begin() as connection:
        for start in range(0, len(records), batch_size):
            statement = insert(table)
            upsert = statement.on_conflict_do_update(
                index_elements=[
                    "ticker",
                    "earnings_date",
                    "source_provider",
                ],
                set_={
                    "earnings_timestamp": statement.excluded.earnings_timestamp,
                    "eps_estimate": statement.excluded.eps_estimate,
                    "reported_eps": statement.excluded.reported_eps,
                    "eps_surprise": statement.excluded.eps_surprise,
                    "eps_surprise_pct": statement.excluded.eps_surprise_pct,
                    "source_url": statement.excluded.source_url,
                    "raw_payload": statement.excluded.raw_payload,
                    "updated_at": func.now(),
                },
            )
            connection.execute(upsert, records[start:start + batch_size])

    return len(records)

