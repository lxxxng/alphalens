"""Validate the local services and corpus required by live evaluations."""

import os
import sys
from pathlib import Path

from sqlalchemy import text

from app.rag.retrievers.base import (
    FAISS_DIRECTORY,
    SEC_INDEX_PATH,
    SEC_METADATA_PATH,
    TRANSCRIPT_INDEX_PATH,
    TRANSCRIPT_METADATA_PATH,
    get_database_engine,
    load_faiss_index,
    load_index_metadata,
)


REQUIRED_CORPUS_TABLES = [
    "filing_chunks",
    "earnings_transcript_chunks",
    "market_prices",
]


def check_environment() -> list[str]:
    """Return configuration errors without making an OpenAI request."""

    errors = []

    if not os.getenv("OPENAI_API_KEY"):
        errors.append("OPENAI_API_KEY is missing.")

    if not os.getenv("DATABASE_URL"):
        errors.append("DATABASE_URL is missing.")

    report_directory = Path(
        os.getenv(
            "ALPHALENS_EVAL_REPORT_DIRECTORY",
            "data/evals",
        )
    )
    try:
        report_directory.mkdir(parents=True, exist_ok=True)
        probe = report_directory / ".alphalens-write-test"
        probe.write_text("ok", encoding="ascii")
        probe.unlink()
    except Exception as exception:
        errors.append(
            f"Evaluation report directory is not writable: {exception}"
        )

    required_files = [
        SEC_INDEX_PATH,
        SEC_METADATA_PATH,
        TRANSCRIPT_INDEX_PATH,
        TRANSCRIPT_METADATA_PATH,
    ]
    for path in required_files:
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"Required FAISS artifact is missing or empty: {path}")

    # Metadata and index loading catch mismatched or corrupt artifacts before
    # the quality suite spends requests creating query embeddings.
    if not any("FAISS artifact" in error for error in errors):
        for index_path, metadata_path in [
            (SEC_INDEX_PATH, SEC_METADATA_PATH),
            (TRANSCRIPT_INDEX_PATH, TRANSCRIPT_METADATA_PATH),
        ]:
            try:
                metadata = load_index_metadata(metadata_path)
                index = load_faiss_index(index_path)
                if index.d != metadata.get("dimension"):
                    errors.append(
                        f"FAISS dimension mismatch for {index_path}: "
                        f"index={index.d}, metadata={metadata.get('dimension')}"
                    )
                if index.ntotal != metadata.get("vectors"):
                    errors.append(
                        f"FAISS vector-count mismatch for {index_path}: "
                        f"index={index.ntotal}, metadata={metadata.get('vectors')}"
                    )
            except Exception as exception:
                errors.append(f"Could not load {index_path}: {exception}")

    if os.getenv("DATABASE_URL"):
        try:
            engine = get_database_engine()
            with engine.connect() as connection:
                for table_name in REQUIRED_CORPUS_TABLES:
                    # Table names come only from the constant allowlist above.
                    count = connection.execute(
                        text(f"SELECT COUNT(*) FROM {table_name}")
                    ).scalar_one()
                    if count < 1:
                        errors.append(f"Corpus table is empty: {table_name}")
                    else:
                        print(f"[OK] {table_name}: {count:,} rows")
        except Exception as exception:
            errors.append(f"Database corpus check failed: {exception}")

    return errors


def main() -> int:
    """Print a concise preflight report and return a CI-friendly exit code."""

    print(f"FAISS directory: {Path(FAISS_DIRECTORY).resolve()}")
    errors = check_environment()

    if errors:
        print("\nQuality-gate preflight failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 2

    print("[OK] Corpus quality-gate environment is ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
