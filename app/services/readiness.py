"""Runtime dependency checks for container and deployment readiness."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from sqlalchemy import create_engine, text


REQUIRED_FAISS_FILES = (
    "sec_chunks.faiss",
    "sec_chunks.meta.json",
    "transcript_chunks.faiss",
    "transcript_chunks.meta.json",
)
REQUIRED_DATABASE_TABLES = (
    "companies",
    "market_prices",
    "filing_chunks",
    "earnings_transcript_chunks",
    "event_briefs",
    "event_alerts",
)


@lru_cache(maxsize=1)
def _database_engine(database_url: str):
    """Keep one tiny pool for frequent orchestrator readiness probes."""

    return create_engine(
        database_url,
        pool_pre_ping=True,
        pool_size=1,
        max_overflow=0,
        connect_args={"connect_timeout": 3},
    )


def check_readiness() -> dict:
    """Check local retrieval assets and a live PostgreSQL connection."""

    checks = {}
    database_url = os.getenv("DATABASE_URL", "").strip()

    if not database_url:
        checks["database"] = {
            "ready": False,
            "message": "DATABASE_URL is not configured.",
        }
    else:
        try:
            engine = _database_engine(database_url)

            with engine.connect() as connection:
                connection.execute(text("SELECT 1"))
                missing_tables = [
                    table_name
                    for table_name in REQUIRED_DATABASE_TABLES
                    if connection.execute(
                        text("SELECT to_regclass(:table_name)"),
                        {"table_name": f"public.{table_name}"},
                    ).scalar_one_or_none() is None
                ]

            checks["database"] = {
                "ready": not missing_tables,
                "missing_tables": missing_tables,
            }
        except Exception as error:
            checks["database"] = {
                "ready": False,
                "message": f"Database unavailable ({type(error).__name__}).",
            }

    faiss_directory = Path(
        os.getenv("ALPHALENS_FAISS_DIRECTORY", "data/faiss")
    )
    missing_files = [
        filename
        for filename in REQUIRED_FAISS_FILES
        if not (faiss_directory / filename).is_file()
    ]
    checks["faiss"] = {
        "ready": not missing_files,
        "directory": str(faiss_directory),
        "missing_files": missing_files,
    }
    ready = all(check["ready"] for check in checks.values())
    return {
        "status": "ready" if ready else "not_ready",
        "service": "alphalens",
        "checks": checks,
    }
