"""
AlphaLens - SEC Filing Document Downloader

Purpose:
    Downloads the actual HTML documents for SEC 10-K and
    10-Q filings whose metadata is already stored in PostgreSQL.


Current SEC pipeline:

    SEC submissions API
        ↓
    extractor.py
        ↓
    filing metadata
        ↓
    PostgreSQL
        ↓
    downloader.py
        ↓
    SEC Archive
        ↓
    raw HTML files
        ↓
    data/sec/raw/


Current strategy:
    Download every 10-K and 10-Q filing already stored in
    PostgreSQL.

The metadata extractor controls the historical window. It currently
loads approximately five years of filings, so this downloader performs
the matching five-year document backfill.
"""

import os
import time

from pathlib import Path

import requests

from dotenv import load_dotenv

from sqlalchemy import (
    create_engine,
    text,
)

# ------------------------------------------------------------
# This function lives in extractor.py.
#
# Because downloader.py is a DIFFERENT FILE,
# we need to import it.
#
# It creates our requests.Session containing the SEC
# User-Agent from .env.
# ------------------------------------------------------------

from pipelines.sec.extractor import create_sec_session


# ============================================================
# Environment
# ============================================================

load_dotenv()


# ============================================================
# Raw SEC storage directory
# ============================================================
#
# Filing HTML will initially be stored locally:
#
# data/
# └── sec/
#     └── raw/
#         ├── AAPL/
#         ├── MSFT/
#         └── ...
#
# Path automatically handles Windows/Linux path differences.
# ============================================================

RAW_FILINGS_DIR = Path(
    "data/sec/raw"
)


# ============================================================
# get_database_engine()
# ============================================================

def get_database_engine():
    """
    Create a SQLAlchemy connection to PostgreSQL.

    DATABASE_URL is read from the .env file.

    Returns
    -------
    sqlalchemy.Engine
        Database engine used by downloader.py.
    """

    database_url = os.getenv(
        "DATABASE_URL"
    )

    if not database_url:
        raise ValueError(
            "DATABASE_URL was not found in .env."
        )

    return create_engine(
        database_url,
        pool_pre_ping=True,
    )


# ============================================================
# build_filing_url()
# ============================================================

def build_filing_url(
    cik: str,
    accession_number: str,
    primary_document: str,
) -> str:
    """
    Construct the SEC URL for a filing's primary document.

    Parameters
    ----------
    cik : str
        SEC company CIK.

        Example:
            0000320193

    accession_number : str
        Unique SEC filing identifier.

        Example:
            0000320193-25-000079

    primary_document : str
        Main filing HTML filename.

        Example:
            aapl-20250927.htm


    Returns
    -------
    str
        Complete SEC Archive URL.


    SEC Archive URL format
    ----------------------

    https://www.sec.gov/Archives/edgar/data/
        {CIK without leading zeros}/
        {accession number without dashes}/
        {primary document}


    Example conceptually:

        CIK:
            0000320193

                ↓ remove leading zeros

            320193


        Accession:
            0000320193-25-000079

                ↓ remove dashes

            000032019325000079


        Primary document:
            aapl-20250927.htm


        Final path:

        /Archives/edgar/data/
        320193/
        000032019325000079/
        aapl-20250927.htm
    """

    # SEC Archive URLs use the numeric CIK without
    # the leading zero padding.
    cik_for_url = str(
        int(cik)
    )

    # Remove "-" from accession number.
    accession_for_url = (
        accession_number.replace(
            "-",
            "",
        )
    )

    return (
        "https://www.sec.gov/Archives/edgar/data/"
        f"{cik_for_url}/"
        f"{accession_for_url}/"
        f"{primary_document}"
    )


# ============================================================
# get_filing_candidates()
# ============================================================

def get_filing_candidates(
    engine,
):
    """
    Retrieve every stored 10-K and 10-Q filing candidate.

    Download scope
    --------------

    The filings table is the source of truth:

        database
            ↓
        URL construction
            ↓
        SEC download
            ↓
        local file storage

    the metadata extractor already limits rows to the configured
    five-year SEC window.


    Ordering
    --------

    We order filings by:

        ticker
        +
        form_type

    Example:

        AAPL 10-Q 2026Q2
        AAPL 10-Q 2026Q1
        AAPL 10-K 2025
        AAPL 10-Q 2025Q3

    The downloader keeps all returned rows:

        every stored 10-K / 10-Q
    """

    query = text(
        """
        SELECT
            ticker,
            cik,
            form_type,
            filing_date,
            accession_number,
            primary_document

        FROM filings

        WHERE form_type IN (
            '10-K',
            '10-Q'
        )
          AND primary_document IS NOT NULL

        ORDER BY
            ticker,
            filing_date DESC,
            form_type;
        """
    )

    # engine.connect() opens a database connection.
    with engine.connect() as connection:

        result = connection.execute(
            query
        )

        # .mappings() makes each row behave like a dictionary:
        #
        # row["ticker"]
        # row["cik"]
        # row["accession_number"]
        #
        rows = result.mappings().all()

    return rows


# Backward-compatible name for older notebooks or scripts.
get_latest_filing_candidates = get_filing_candidates


# ============================================================
# build_local_file_path()
# ============================================================

def build_local_file_path(
    ticker: str,
    accession_number: str,
    primary_document: str,
) -> Path:
    """
    Decide where one SEC document should be stored locally.

    Example:

    data/
    └── sec/
        └── raw/
            └── AAPL/
                └── 000032019325000079/
                    └── aapl-20250927.htm


    Why use one accession directory per filing?
    -------------------------------------------

    One SEC submission can contain multiple files later.

    By organizing files under the accession number, we leave
    room for additional filing documents if AlphaLens needs them.
    """

    # Remove dashes so the folder resembles SEC's archive path.
    accession_folder = (
        accession_number.replace(
            "-",
            "",
        )
    )

    return (
        RAW_FILINGS_DIR
        / ticker
        / accession_folder
        / primary_document
    )


# ============================================================
# update_download_success()
# ============================================================

def update_download_success(
    engine,
    accession_number: str,
    source_url: str,
    local_path: Path,
):
    """
    Mark one filing as successfully downloaded.
    """

    query = text(
        """
        UPDATE filings

        SET
            source_url = :source_url,
            raw_file_path = :raw_file_path,
            download_status = 'DOWNLOADED',
            downloaded_at = NOW(),
            download_error = NULL,
            updated_at = NOW()

        WHERE accession_number = :accession_number;
        """
    )

    with engine.begin() as connection:

        connection.execute(
            query,
            {
                "source_url": source_url,

                # Store a readable relative path rather
                # than an absolute Windows path.
                "raw_file_path":
                    local_path.as_posix(),

                "accession_number":
                    accession_number,
            },
        )


# ============================================================
# update_download_failure()
# ============================================================

def update_download_failure(
    engine,
    accession_number: str,
    source_url: str,
    error_message: str,
):
    """
    Record a failed SEC download.

    Keeping errors in PostgreSQL makes it easier later to ask:

        Which documents failed?
        Why did they fail?
        Which ones need retrying?
    """

    query = text(
        """
        UPDATE filings

        SET
            source_url = :source_url,
            download_status = 'FAILED',
            download_error = :download_error,
            updated_at = NOW()

        WHERE accession_number = :accession_number;
        """
    )

    with engine.begin() as connection:

        connection.execute(
            query,
            {
                "source_url":
                    source_url,

                # Avoid storing enormous exception strings.
                "download_error":
                    error_message[:1000],

                "accession_number":
                    accession_number,
            },
        )


# ============================================================
# download_filing()
# ============================================================

def download_filing(
    session: requests.Session,
    engine,
    filing,
):
    """
    Download one SEC filing document.

    Parameters
    ----------
    session : requests.Session
        SEC HTTP session.

    engine : sqlalchemy.Engine
        PostgreSQL connection engine.

    filing
        Row returned from get_filing_candidates().


    Returns
    -------
    bool
        True if successful.
        False if failed.
    """

    ticker = filing["ticker"]

    cik = filing["cik"]

    form_type = filing["form_type"]

    accession_number = (
        filing["accession_number"]
    )

    primary_document = (
        filing["primary_document"]
    )

    # ========================================================
    # Build SEC URL
    # ========================================================

    source_url = build_filing_url(
        cik=cik,
        accession_number=accession_number,
        primary_document=primary_document,
    )

    # ========================================================
    # Build local path
    # ========================================================

    local_path = build_local_file_path(
        ticker=ticker,
        accession_number=accession_number,
        primary_document=primary_document,
    )

    # Create parent folders if they don't already exist.
    #
    # Example:
    #
    # data/sec/raw/AAPL/000032019325000079/
    #
    local_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Idempotency check
    # ========================================================
    #
    # If the file already exists locally, we don't need to
    # download it again.
    # ========================================================

    if local_path.exists():

        print(
            f"[SKIPPED] {ticker} {form_type}: "
            "file already exists"
        )

        update_download_success(
            engine=engine,
            accession_number=accession_number,
            source_url=source_url,
            local_path=local_path,
        )

        return True

    # ========================================================
    # Download from SEC
    # ========================================================

    try:

        response = session.get(
            source_url,
            timeout=60,
        )

        response.raise_for_status()

        # ====================================================
        # Save raw response bytes
        # ====================================================
        #
        # We use response.content instead of response.text.
        #
        # response.content gives us the original bytes returned
        # by SEC and avoids unnecessary text conversion during
        # the raw-ingestion stage.
        # ====================================================

        local_path.write_bytes(
            response.content
        )

        # Update PostgreSQL.
        update_download_success(
            engine=engine,
            accession_number=accession_number,
            source_url=source_url,
            local_path=local_path,
        )

        print(
            f"[OK] {ticker} {form_type}: "
            f"{local_path}"
        )

        return True

    except (
        requests.RequestException,
        OSError,
    ) as error:

        update_download_failure(
            engine=engine,
            accession_number=accession_number,
            source_url=source_url,
            error_message=str(error),
        )

        print(
            f"[FAILED] {ticker} {form_type}: "
            f"{error}"
        )

        return False


# ============================================================
# download_filings()
# ============================================================

def download_filings():
    """
    Download all stored 10-K and 10-Q filings.

    Current target:
        approximately five years of SEC documents.

    This function coordinates:

        PostgreSQL
            ↓
        filing candidates
            ↓
        SEC Archive
            ↓
        raw HTML files
    """

    engine = get_database_engine()

    session = create_sec_session()

    filings = get_filing_candidates(
        engine
    )

    print(
        f"\nDocuments selected for download: "
        f"{len(filings)}"
    )

    successful = 0

    failed = 0

    # ========================================================
    # Download one filing at a time
    # ========================================================

    for filing in filings:

        result = download_filing(
            session=session,
            engine=engine,
            filing=filing,
        )

        if result:
            successful += 1
        else:
            failed += 1

        # Stay comfortably below SEC request-rate guidance.
        time.sleep(0.2)

    print("\n========================================")
    print("SEC DOCUMENT DOWNLOAD COMPLETE")
    print("========================================")

    print(
        f"Successful: {successful}"
    )

    print(
        f"Failed: {failed}"
    )


# Backward-compatible name for older notebooks or scripts.
download_latest_filings = download_filings


# ============================================================
# Script Entry Point
# ============================================================

if __name__ == "__main__":

    download_filings()
