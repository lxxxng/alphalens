"""Persistence helpers for saved AlphaLens research runs."""

import os

from dotenv import load_dotenv
from sqlalchemy import (
    MetaData,
    Table,
    create_engine,
    delete,
    desc,
    insert,
    select,
)


load_dotenv()


def get_database_engine():
    """Create a PostgreSQL engine from the local AlphaLens configuration."""

    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise ValueError(
            "DATABASE_URL was not found in .env."
        )

    return create_engine(
        database_url,
        pool_pre_ping=True,
    )


def get_research_runs_table(engine) -> Table:
    """Reflect the migration-managed research_runs table."""

    return Table(
        "research_runs",
        MetaData(),
        autoload_with=engine,
    )


def collect_result_tickers(
    request_data: dict,
    result: dict,
) -> list[str]:
    """Collect unique tickers represented by a saved result."""

    candidates = [
        request_data.get("ticker"),
        *[
            item.get("ticker")
            for item in result.get("market_context", [])
        ],
        *[
            item.get("ticker")
            for item in result.get("sources", [])
        ],
    ]
    tickers = []

    for candidate in candidates:
        if not candidate:
            continue

        normalized = str(candidate).upper()

        if normalized not in tickers:
            tickers.append(normalized)

    return tickers


def save_research_run(
    request_data: dict,
    result: dict,
) -> int:
    """Insert one immutable research snapshot and return its run ID."""

    engine = get_database_engine()
    research_runs = get_research_runs_table(engine)
    market_context = result.get("market_context", [])
    sources = result.get("sources", [])
    ticker_filter = request_data.get("ticker")

    statement = (
        insert(research_runs)
        .values(
            question=result["question"],
            answer=result["answer"],
            tickers=collect_result_tickers(
                request_data=request_data,
                result=result,
            ),
            ticker_filter=(
                str(ticker_filter).upper()
                if ticker_filter
                else None
            ),
            source_type=request_data.get(
                "source_type",
                "auto",
            ),
            top_k=request_data.get("top_k", 5),
            form_type=request_data.get("form_type"),
            section_key=request_data.get("section_key"),
            fiscal_period=request_data.get("fiscal_period"),
            source_count=len(sources),
            market_snapshot_count=len(market_context),
            market_context=market_context,
            sources=sources,
        )
        .returning(research_runs.c.run_id)
    )

    with engine.begin() as connection:
        return int(
            connection.execute(statement).scalar_one()
        )


def list_research_runs(
    limit: int = 20,
) -> list[dict]:
    """Return recent lightweight run summaries, newest first."""

    engine = get_database_engine()
    research_runs = get_research_runs_table(engine)
    statement = (
        select(
            research_runs.c.run_id,
            research_runs.c.question,
            research_runs.c.tickers,
            research_runs.c.source_type,
            research_runs.c.source_count,
            research_runs.c.market_snapshot_count,
            research_runs.c.created_at,
        )
        .order_by(
            desc(research_runs.c.created_at),
            desc(research_runs.c.run_id),
        )
        .limit(limit)
    )

    with engine.connect() as connection:
        rows = connection.execute(statement).mappings().all()

    return [
        {
            **dict(row),
            "created_at": row["created_at"].isoformat(),
        }
        for row in rows
    ]


def get_research_run(
    run_id: int,
) -> dict | None:
    """Return one complete saved run without rerunning AlphaLens."""

    engine = get_database_engine()
    research_runs = get_research_runs_table(engine)
    statement = select(research_runs).where(
        research_runs.c.run_id == run_id
    )

    with engine.connect() as connection:
        row = connection.execute(statement).mappings().first()

    if row is None:
        return None

    result = dict(row)
    result["created_at"] = row["created_at"].isoformat()
    return result


def delete_research_run(
    run_id: int,
) -> bool:
    """Delete one saved run and report whether it existed."""

    engine = get_database_engine()
    research_runs = get_research_runs_table(engine)
    statement = delete(research_runs).where(
        research_runs.c.run_id == run_id
    )

    with engine.begin() as connection:
        result = connection.execute(statement)

    return result.rowcount > 0
