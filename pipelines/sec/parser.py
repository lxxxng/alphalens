"""
AlphaLens - SEC Filing HTML Parser

Purpose:
    Converts downloaded SEC 10-K / 10-Q HTML documents
    into cleaner plain-text files.

Pipeline:

    PostgreSQL filings table
            ↓
    raw_file_path
            ↓
    SEC .htm file
            ↓
    BeautifulSoup
            ↓
    remove HTML noise
            ↓
    clean visible text
            ↓
    .txt file
            ↓
    clean_text_path stored in PostgreSQL


Important:
    We NEVER modify the original SEC HTML file.

    Raw HTML:
        data/sec/raw/

    Clean text:
        data/sec/clean/

Keeping the raw source means we can improve our parser later
without downloading the filing again.
"""

import os
import re

from pathlib import Path

from bs4 import BeautifulSoup

from dotenv import load_dotenv

from sqlalchemy import (
    create_engine,
    text,
)


# ============================================================
# Environment
# ============================================================

load_dotenv()


# ============================================================
# Clean Filing Directory
# ============================================================

CLEAN_FILINGS_DIR = Path(
    "data/sec/clean"
)


# ============================================================
# get_database_engine()
# ============================================================

def get_database_engine():
    """
    Create a SQLAlchemy connection engine for PostgreSQL.

    The database connection string comes from:

        .env

    Variable:

        DATABASE_URL
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
# get_downloaded_filings()
# ============================================================

def get_downloaded_filings(
    engine,
):
    """
    Retrieve filings that have already been downloaded.

    Returns
    -------
    list
        PostgreSQL rows containing information needed
        to locate and parse each raw SEC filing.

    We only want rows where:

        download_status = DOWNLOADED

    because there is no HTML file to parse if the download
    never succeeded.
    """

    query = text(
        """
        SELECT
            accession_number,
            ticker,
            form_type,
            filing_date,
            raw_file_path,
            clean_text_path,
            parse_status

        FROM filings

        WHERE download_status = 'DOWNLOADED'
          AND raw_file_path IS NOT NULL

        ORDER BY
            ticker,
            form_type,
            filing_date DESC;
        """
    )

    with engine.connect() as connection:

        result = connection.execute(
            query
        )

        # mappings() makes rows dictionary-like.
        #
        # Example:
        #
        # filing["ticker"]
        # filing["raw_file_path"]
        #
        filings = result.mappings().all()

    return filings


# ============================================================
# build_clean_file_path()
# ============================================================

def build_clean_file_path(
    ticker: str,
    accession_number: str,
    raw_file_path: str,
) -> Path:
    """
    Build the location for the cleaned text document.

    Example raw file:

        data/sec/raw/
        AAPL/
        000032019325000079/
        aapl-20250927.htm

    Clean version:

        data/sec/clean/
        AAPL/
        000032019325000079/
        aapl-20250927.txt


    Parameters
    ----------
    ticker : str
        Example:
            AAPL

    accession_number : str
        Example:
            0000320193-25-000079

    raw_file_path : str
        Path to downloaded SEC HTML.


    Returns
    -------
    pathlib.Path
        Location where parsed text should be saved.
    """

    # SEC folder names use accession numbers without dashes.
    accession_folder = (
        accession_number.replace(
            "-",
            "",
        )
    )

    # Get the original HTML filename.
    #
    # Example:
    #
    # aapl-20250927.htm
    #
    raw_path = Path(
        raw_file_path
    )

    # .stem removes the extension.
    #
    # aapl-20250927.htm
    #
    # becomes:
    #
    # aapl-20250927
    #
    clean_filename = (
        raw_path.stem + ".txt"
    )

    return (
        CLEAN_FILINGS_DIR
        / ticker
        / accession_folder
        / clean_filename
    )


# ============================================================
# clean_html_to_text()
# ============================================================

def clean_html_to_text(
    html_content: bytes,
) -> str:
    """
    Convert SEC HTML into cleaner plain text.

    Parameters
    ----------
    html_content : bytes
        Raw bytes read from the downloaded .htm file.

    Returns
    -------
    str
        Cleaned human-readable filing text.


    What are we removing?
    ---------------------

    HTML documents contain things that are useful to browsers
    but useless to our RAG system.

    Examples:

        <script>
        <style>
        CSS
        JavaScript

    We remove those.


    What are we keeping?
    --------------------

    We keep visible filing content such as:

        Item 1. Business

        Item 1A. Risk Factors

        Management's Discussion and Analysis

        financial statement text

        tables

        footnotes


    Inline XBRL
    -----------

    SEC filings may contain tags such as:

        <ix:nonfraction>

    These tags contain actual financial values.

    Therefore we DO NOT simply delete Inline XBRL tags.

    BeautifulSoup's get_text() extracts the text inside them.
    """

    # ========================================================
    # Parse HTML
    # ========================================================
    #
    # lxml converts the HTML bytes into a tree that
    # BeautifulSoup can navigate.
    # ========================================================

    soup = BeautifulSoup(
        html_content,
        "lxml",
    )


    # ========================================================
    # Remove non-content elements
    # ========================================================
    #
    # script:
    #     JavaScript code
    #
    # style:
    #     CSS formatting
    #
    # noscript:
    #     Browser fallback content
    #
    # These do not help our RAG system.
    # ========================================================

    for element in soup(
        [
            "script",
            "style",
            "noscript",
        ]
    ):

        element.decompose()


    # ========================================================
    # Convert <br> tags into actual line breaks
    # ========================================================
    #
    # HTML:
    #
    # Revenue increased<br>Operating margin increased
    #
    # becomes:
    #
    # Revenue increased
    # Operating margin increased
    # ========================================================

    for br in soup.find_all("br"):

        br.replace_with("\n")


    # ========================================================
    # Extract Visible Text
    # ========================================================
    #
    # separator="\n"
    #
    # remove htmls tags and place line breaks between
    # separate HTML tags

    # BEFORE:
    # <h1>Apple Inc.</h1>
    # <p>Annual Report</p>
    # <table><tr><td>Revenue</td></tr></table>
    #
    # AFTER:
    # Apple Inc.
    # Annual Report
    # Revenue
    # ========================================================

    raw_text = soup.get_text(
        separator="\n"      
    )


    # ========================================================
    # Normalize Individual Lines
    # ========================================================
    #
    # SEC HTML can contain large amounts of whitespace:
    #
    # "Risk     Factors"
    #
    # or:
    #
    # "    Revenue increased     "
    #
    # This converts repeated whitespace into one space.
    # ========================================================

    cleaned_lines = []

    for line in raw_text.splitlines():

        # Replace repeated spaces/tabs with a single space.
        line = re.sub(
            r"[ \t]+",
            " ",
            line,
        )

        # Remove whitespace from beginning/end.
        line = line.strip()

        # Ignore completely empty lines.
        if not line:
            continue

        cleaned_lines.append(
            line
        )


    # ========================================================
    # Recombine Lines
    # ========================================================

    clean_text = "\n".join(
        cleaned_lines
    )

    return clean_text


# ============================================================
# update_parse_success()
# ============================================================

def update_parse_success(
    engine,
    accession_number: str,
    clean_file_path: Path,
):
    """
    Mark one filing as successfully parsed.
    """

    query = text(
        """
        UPDATE filings

        SET
            clean_text_path = :clean_text_path,
            parse_status = 'PARSED',
            parsed_at = NOW(),
            parse_error = NULL,
            updated_at = NOW()

        WHERE accession_number = :accession_number;
        """
    )

    with engine.begin() as connection:

        connection.execute(
            query,
            {
                "clean_text_path":
                    clean_file_path.as_posix(),

                "accession_number":
                    accession_number,
            },
        )


# ============================================================
# update_parse_failure()
# ============================================================

def update_parse_failure(
    engine,
    accession_number: str,
    error_message: str,
):
    """
    Record a parsing failure in PostgreSQL.
    """

    query = text(
        """
        UPDATE filings

        SET
            parse_status = 'FAILED',
            parse_error = :parse_error,
            updated_at = NOW()

        WHERE accession_number = :accession_number;
        """
    )

    with engine.begin() as connection:

        connection.execute(
            query,
            {
                "parse_error":
                    error_message[:1000],

                "accession_number":
                    accession_number,
            },
        )


# ============================================================
# parse_filing()
# ============================================================

def parse_filing(
    engine,
    filing,
) -> bool:
    """
    Parse one downloaded SEC filing.

    Flow:

        raw HTML file
             ↓
        read bytes
             ↓
        clean_html_to_text()
             ↓
        clean text
             ↓
        save .txt
             ↓
        update PostgreSQL


    Returns
    -------
    bool

        True:
            parsing succeeded

        False:
            parsing failed
    """

    ticker = filing["ticker"]

    form_type = filing["form_type"]

    accession_number = (
        filing["accession_number"]
    )

    raw_file_path = Path(
        filing["raw_file_path"]
    )

    # Determine where clean text should be stored.
    clean_file_path = build_clean_file_path(
        ticker=ticker,
        accession_number=accession_number,
        raw_file_path=str(raw_file_path),
    )


    # ========================================================
    # Idempotency
    # ========================================================
    #
    # If the clean text already exists, don't perform
    # the same parsing work again.
    # ========================================================

    if clean_file_path.exists():

        update_parse_success(
            engine=engine,
            accession_number=accession_number,
            clean_file_path=clean_file_path,
        )

        print(
            f"[SKIPPED] {ticker} {form_type}: "
            "clean text already exists"
        )

        return True


    try:

        # ====================================================
        # Read raw SEC HTML
        # ====================================================

        html_content = (
            raw_file_path.read_bytes()
        )


        # ====================================================
        # Convert HTML → clean text
        # ====================================================

        clean_text = clean_html_to_text(
            html_content
        )


        # ====================================================
        # Basic validation
        # ====================================================
        #
        # A real 10-K / 10-Q should contain a significant
        # amount of text.
        #
        # If our result is extremely small, something probably
        # went wrong during download or parsing.
        # ====================================================

        if len(clean_text) < 1000:

            raise ValueError(
                "Parsed filing contains less than "
                "1,000 characters."
            )


        # ====================================================
        # Create output directories
        # ====================================================

        clean_file_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )


        # ====================================================
        # Save clean text
        # ========================================================

        clean_file_path.write_text(
            clean_text,
            encoding="utf-8",
        )


        # ====================================================
        # Update PostgreSQL
        # ====================================================

        update_parse_success(
            engine=engine,
            accession_number=accession_number,
            clean_file_path=clean_file_path,
        )


        print(
            f"[OK] {ticker} {form_type}: "
            f"{len(clean_text):,} characters"
        )

        return True


    except Exception as error:

        update_parse_failure(
            engine=engine,
            accession_number=accession_number,
            error_message=str(error),
        )

        print(
            f"[FAILED] {ticker} {form_type}: "
            f"{error}"
        )

        return False


# ============================================================
# parse_downloaded_filings()
# ============================================================

def parse_downloaded_filings():
    """
    Parse every SEC filing that AlphaLens has downloaded.

    Current dataset:
        every SEC document that downloader.py has successfully
        downloaded, normally approximately five years of 10-K
        and 10-Q filings across 20 companies.
    """

    engine = get_database_engine()

    filings = get_downloaded_filings(
        engine
    )

    print(
        f"\nDownloaded filings found: "
        f"{len(filings)}"
    )

    successful = 0
    failed = 0

    for filing in filings:

        result = parse_filing(
            engine=engine,
            filing=filing,
        )

        if result:
            successful += 1
        else:
            failed += 1


    print("\n========================================")
    print("SEC PARSING COMPLETE")
    print("========================================")

    print(
        f"Successful: {successful}"
    )

    print(
        f"Failed: {failed}"
    )


# ============================================================
# Script Entry Point
# ============================================================

if __name__ == "__main__":

    parse_downloaded_filings()
