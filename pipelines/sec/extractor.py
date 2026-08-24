"""
AlphaLens - SEC EDGAR Filing Metadata Extractor

Purpose:
    Retrieves SEC filing metadata for the companies used
    by AlphaLens.

At this stage we are NOT downloading the full filing text.

Current SEC pipeline:

    Stock ticker
        ↓
    Find SEC CIK
        ↓
    SEC submissions API
        ↓
    Filing metadata
        ↓
    Keep 10-K / 10-Q
        ↓
    Pandas DataFrame

Later:

    Filing metadata
        ↓
    Download actual HTML filing
        ↓
    Extract text
        ↓
    Split into chunks
        ↓
    Embeddings
        ↓
    FAISS
        ↓
    RAG
"""

import os
import time

import pandas as pd
import requests

from dotenv import load_dotenv

# Reuse the same 20 company tickers from our OHLCV extractor.
#
# This prevents us from maintaining two separate ticker lists.
#
# SPY is stored separately as BENCHMARK in extractor.py,
# so it will NOT be included in the SEC company pipeline.
from pipelines.market_data.extractor import TICKERS


# ============================================================
# Load environment variables
# ============================================================
#
# This reads:
#
#     .env
#
# and makes variables such as:
#
#     SEC_USER_AGENT
#
# available through os.getenv().
# ============================================================

load_dotenv()


# ============================================================
# SEC endpoints
# ============================================================

# SEC publishes a JSON file mapping:
#
#     ticker
#        ↓
#     company name
#        ↓
#     CIK
#
# Example:
#
#     AAPL → Apple Inc. → 320193
#
SEC_TICKER_URL = (
    "https://www.sec.gov/files/company_tickers.json"
)


# SEC submissions API.
#
# Later we append:
#
#     /CIK0000320193.json
#
# for Apple.
#
SEC_SUBMISSIONS_BASE_URL = (
    "https://data.sec.gov/submissions"
)


# ============================================================
# create_sec_session()
# ============================================================

def create_sec_session() -> requests.Session:
    """
    Create a reusable HTTP session for SEC requests.

    Returns
    -------
    requests.Session
        A requests Session containing the HTTP headers
        AlphaLens should send to the SEC.

    Why use requests.Session?
    -------------------------
    Without a session:

        Request 1 → new connection
        Request 2 → new connection
        Request 3 → new connection

    With a session:

        one reusable session
              ↓
        multiple SEC requests

    This is cleaner and more efficient.

    Why User-Agent?
    ---------------
    SEC asks automated applications to identify themselves.

    We store it in .env:

        SEC_USER_AGENT=AlphaLens your-email@example.com

    rather than hard-coding personal information in GitHub.
    """

    # Read the SEC user agent from .env.
    user_agent = os.getenv("SEC_USER_AGENT")

    # Stop immediately if it was not configured.
    if not user_agent:
        raise ValueError(
            "SEC_USER_AGENT was not found in .env.\n"
            "Example:\n"
            "SEC_USER_AGENT=AlphaLens your-email@example.com"
        )

    # Create a reusable HTTP session.
    session = requests.Session()

    # These headers will automatically be included
    # in requests made using this session.
    session.headers.update(
        {
            "User-Agent": user_agent,

            # Tell SEC that JSON is preferred where applicable.
            "Accept": "application/json",

            # Allow compressed responses.
            "Accept-Encoding": "gzip, deflate",
        }
    )

    return session


# ============================================================
# fetch_ticker_cik_mapping()
# ============================================================

def fetch_ticker_cik_mapping(
    session: requests.Session,
) -> dict:
    """
    Download SEC's ticker → CIK mapping.

    Parameters
    ----------
    session : requests.Session
        SEC HTTP session.

    Returns
    -------
    dict
        Mapping keyed by ticker.

    Example result:

        {
            "AAPL": {
                "cik": "0000320193",
                "company_name": "Apple Inc."
            },

            "MSFT": {
                ...
            }
        }

    Why do we need this?
    --------------------
    Our AlphaLens data uses stock tickers:

        AAPL

    SEC APIs primarily use CIKs:

        0000320193

    Therefore:

        AAPL
          ↓
        ticker mapping
          ↓
        0000320193
    """

    print("Downloading SEC ticker/CIK mapping...")

    # Send HTTP GET request.
    response = session.get(
        SEC_TICKER_URL,
        timeout=30,
    )

    # If SEC returned an HTTP error such as:
    #
    #     403 Forbidden
    #     404 Not Found
    #     429 Too Many Requests
    #     500 Server Error
    #
    # raise an exception instead of continuing with bad data.
    response.raise_for_status()

    # Convert JSON response into Python dictionaries.
    data = response.json()

    mapping = {}

    # SEC's JSON looks conceptually like:
    #
    # {
    #     "0": {
    #         "cik_str": 320193,
    #         "ticker": "AAPL",
    #         "title": "Apple Inc."
    #     },
    #
    #     "1": {
    #         ...
    #     }
    # }
    #
    # We transform it into something easier to use:
    #
    # {
    #     "AAPL": {
    #         "cik": "0000320193",
    #         "company_name": "Apple Inc."
    #     }
    # }
    #
    for company in data.values():

        ticker = company["ticker"].upper()

        # Convert the numeric CIK into a string.
        cik = str(
            company["cik_str"]
        )

        # SEC submission endpoints expect 10 digits.
        #
        # Example:
        #
        #     320193
        #
        # becomes:
        #
        #     0000320193
        #
        cik = cik.zfill(10)

        mapping[ticker] = {
            "cik": cik,
            "company_name": company["title"],
        }

    print(
        f"Loaded {len(mapping)} ticker/CIK mappings."
    )

    return mapping


# ============================================================
# fetch_company_submissions()
# ============================================================

def fetch_company_submissions(
    session: requests.Session,
    cik: str,
) -> dict:
    """
    Download recent SEC filing history for one company.

    Parameters
    ----------
    session : requests.Session
        SEC HTTP session.

    cik : str
        10-digit CIK.

        Example:

            0000320193

    Returns
    -------
    dict
        SEC submissions JSON.

    Example
    -------

    For Apple:

        https://data.sec.gov/submissions/
        CIK0000320193.json

    The JSON contains metadata about filings such as:

        10-K
        10-Q
        8-K
        DEF 14A
        Form 4
        etc.

    We will later filter this down to only:

        10-K
        10-Q
    """

    url = (
        f"{SEC_SUBMISSIONS_BASE_URL}/"
        f"CIK{cik}.json"
    )

    response = session.get(
        url,
        timeout=30,
    )

    response.raise_for_status()

    return response.json()


# ============================================================
# extract_10k_10q()
# ============================================================

def extract_10k_10q(
    ticker: str,
    company_name: str,
    cik: str,
    submissions: dict,
) -> pd.DataFrame:
    """
    Convert SEC submissions JSON into clean 10-K/10-Q rows.

    Parameters
    ----------
    ticker : str
        Stock ticker.

        Example:
            AAPL

    company_name : str
        SEC company name.

        Example:
            Apple Inc.

    cik : str
        SEC Central Index Key.

    submissions : dict
        Raw submissions JSON returned by SEC.

    Returns
    -------
    pandas.DataFrame
        Filing metadata containing one row per 10-K/10-Q.


    What is a 10-K?
    ---------------
    Annual company filing.

    Usually includes:

        business description
        risk factors
        audited financial statements
        management discussion
        financial notes


    What is a 10-Q?
    ---------------
    Quarterly filing.

    Usually includes:

        quarterly financial statements
        management discussion
        updated disclosures
        updated risks
    """

    # ========================================================
    # Get recent filing information
    # ========================================================
    #
    # SEC stores it here:
    #
    # submissions
    #   └── filings
    #         └── recent
    #
    recent = submissions["filings"]["recent"]

    # ========================================================
    # Convert SEC's column-oriented JSON into a DataFrame
    # ========================================================
    #
    # SEC data is conceptually:
    #
    # form:
    #   ["10-Q", "8-K", "10-K", ...]
    #
    # filingDate:
    #   ["2026-...", "2026-...", ...]
    #
    # accessionNumber:
    #   [...]
    #
    # Pandas turns those parallel arrays into rows.
    # ========================================================

    filings = pd.DataFrame(recent)

    # ========================================================
    # Keep only 10-K and 10-Q
    # ========================================================

    filings = filings[
        filings["form"].isin(
            [
                "10-K",
                "10-Q",
            ]
        )
    ].copy()

    # ========================================================
    # Add our own company metadata
    # ========================================================

    filings["ticker"] = ticker
    filings["company_name"] = company_name
    filings["cik"] = cik

    # ========================================================
    # Keep only the columns AlphaLens currently needs
    # ========================================================

    filings = filings[
        [
            "ticker",
            "company_name",
            "cik",
            "form",
            "filingDate",
            "reportDate",
            "accessionNumber",
            "primaryDocument",
        ]
    ]

    # ========================================================
    # Rename SEC column names to our naming style
    # ========================================================
    #
    # SEC:
    #
    #     filingDate
    #
    # AlphaLens:
    #
    #     filing_date
    #
    # Using snake_case consistently makes our Python
    # and PostgreSQL schemas easier to work with.
    # ========================================================

    filings = filings.rename(
        columns={
            "form": "form_type",
            "filingDate": "filing_date",
            "reportDate": "report_date",
            "accessionNumber": "accession_number",
            "primaryDocument": "primary_document",
        }
    )

    # Convert SEC date strings into Pandas dates.
    filings["filing_date"] = pd.to_datetime(
        filings["filing_date"]
    )

    filings["report_date"] = pd.to_datetime(
        filings["report_date"],
        errors="coerce",
    )

    return filings


# ============================================================
# extract_sec_filing_metadata()
# ============================================================

def extract_sec_filing_metadata() -> pd.DataFrame:
    """
    Extract SEC 10-K/10-Q metadata for all AlphaLens companies.

    Returns
    -------
    pandas.DataFrame
        Combined filing metadata for the 20 AlphaLens companies.

    Full process
    ------------

        20 tickers
            ↓
        SEC ticker → CIK mapping
            ↓
        loop through companies
            ↓
        download submissions JSON
            ↓
        filter 10-K / 10-Q
            ↓
        combine all DataFrames
    """

    # Create our HTTP session.
    session = create_sec_session()

    # Download ticker → CIK information once.
    ticker_mapping = fetch_ticker_cik_mapping(
        session
    )

    # Store each company's DataFrame here temporarily.
    company_frames = []

    # ========================================================
    # Process each AlphaLens company
    # ========================================================

    for ticker in TICKERS:

        print(
            f"\nFetching SEC filings for {ticker}..."
        )

        # Find company information using ticker.
        company = ticker_mapping.get(ticker)

        # If the ticker cannot be found, skip it rather
        # than crashing the entire pipeline.
        if company is None:

            print(
                f"[SKIPPED] No SEC CIK found for {ticker}"
            )

            continue

        cik = company["cik"]

        company_name = company["company_name"]

        # ====================================================
        # Request company submissions
        # ====================================================

        try:

            submissions = fetch_company_submissions(
                session=session,
                cik=cik,
            )

        except requests.RequestException as error:

            # One company failing should not necessarily
            # stop the remaining 19 companies.
            print(
                f"[FAILED] {ticker}: {error}"
            )

            continue

        # ====================================================
        # Extract only 10-K / 10-Q rows
        # ====================================================

        filing_data = extract_10k_10q(
            ticker=ticker,
            company_name=company_name,
            cik=cik,
            submissions=submissions,
        )

        company_frames.append(
            filing_data
        )

        print(
            f"[OK] {ticker}: "
            f"{len(filing_data)} filings found"
        )

        # ====================================================
        # Rate limiting
        # ====================================================
        #
        # 0.2 seconds between requests means roughly:
        #
        #     maximum ~5 requests/second
        #
        # which stays below SEC's current 10 request/sec
        # fair-access guideline.
        # ====================================================

        time.sleep(0.2)

    # ========================================================
    # Validation
    # ========================================================

    if not company_frames:

        raise ValueError(
            "No SEC filing metadata was downloaded."
        )

    # ========================================================
    # Combine all companies
    # ========================================================

    all_filings = pd.concat(
        company_frames,
        ignore_index=True,
    )

    # Sort:
    #
    # ticker alphabetically
    #
    # then newest filing first.
    all_filings = all_filings.sort_values(
        by=[
            "ticker",
            "filing_date",
        ],
        ascending=[
            True,
            False,
        ],
    ).reset_index(
        drop=True
    )

    return all_filings


# ============================================================
# Script Entry Point
# ============================================================

if __name__ == "__main__":

    # Run SEC extraction.
    filings = extract_sec_filing_metadata()

    print("\n========================================")
    print("SEC EXTRACTION COMPLETE")
    print("========================================")

    print(
        f"\nTotal filings found: {len(filings)}"
    )

    # Show the first 20 filings.
    print("\nFirst 20 rows:")

    print(
        filings.head(20).to_string(
            index=False
        )
    )

    # ========================================================
    # Count filings by ticker + filing type
    # ========================================================
    #
    # Example:
    #
    # AAPL
    #     10-K     5
    #     10-Q    15
    #
    # MSFT
    #     10-K     5
    #     10-Q    15
    #
    # ========================================================

    print("\nFilings by ticker and type:")

    print(
        filings.groupby(
            [
                "ticker",
                "form_type",
            ]
        ).size()
    )