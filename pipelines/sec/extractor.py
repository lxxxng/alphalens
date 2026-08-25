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
# SEC History Configuration
# ============================================================

# AlphaLens will keep SEC 10-K and 10-Q filings from the
# most recent five years.
#
# Example:
#
# If today is:
#     2026-08-24
#
# cutoff becomes approximately:
#     2021-08-24
#
SEC_LOOKBACK_YEARS = 5

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
# fetch_historical_submission_file()
# ============================================================

def fetch_historical_submission_file(
    session: requests.Session,
    file_name: str,
) -> dict:
    """
    Download one older SEC submissions-history JSON file.

    Why is this needed?
    -------------------

    The main company submissions endpoint:

        CIK##########.json

    contains:

        submissions["filings"]["recent"]

    But for companies with more filing history, SEC may move
    older records into additional JSON files.

    The main JSON tells us about those files here:

        submissions["filings"]["files"]

    Example conceptually:

        [
            {
                "name":
                    "CIK0000320193-submissions-001.json",

                "filingFrom":
                    "1994-01-01",

                "filingTo":
                    "2024-01-01"
            }
        ]

    We then download:

        https://data.sec.gov/submissions/
        CIK0000320193-submissions-001.json


    Parameters
    ----------
    session:
        Our reusable SEC HTTP session.

    file_name:
        Historical JSON filename supplied by SEC.


    Returns
    -------
    dict:
        Historical filing metadata.
    """

    url = (
        f"{SEC_SUBMISSIONS_BASE_URL}/"
        f"{file_name}"
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

# ============================================================
# extract_10k_10q()
# ============================================================

def extract_10k_10q(
    session: requests.Session,
    ticker: str,
    company_name: str,
    cik: str,
    submissions: dict,
) -> pd.DataFrame:
    """
    Extract approximately five years of SEC 10-K / 10-Q
    filing metadata for one company.

    Data sources
    ------------

    SEC divides filing history into two possible locations:

        1. filings["recent"]

            Recent filing metadata contained directly inside
            the company's main submissions JSON.

        2. filings["files"]

            References to additional historical JSON files.

    Therefore our pipeline becomes:

        recent filings
              +
        historical filing files
              ↓
        combine
              ↓
        keep 10-K / 10-Q
              ↓
        keep last 5 years
              ↓
        remove duplicates


    Parameters
    ----------
    session:
        Reusable SEC requests session.

    ticker:
        Stock ticker.

        Example:
            AAPL

    company_name:
        Company name from SEC.

    cik:
        10-digit SEC CIK.

    submissions:
        Main company submissions JSON.


    Returns
    -------
    pandas.DataFrame:
        Approximately five years of 10-K / 10-Q metadata.
    """

    # ========================================================
    # Calculate our five-year cutoff
    # ========================================================
    #
    # Example:
    #
    # today:
    #     2026-08-24
    #
    # minus 5 years:
    #     2021-08-24
    #
    # DateOffset(years=5) is preferable to simply subtracting
    # 365 * 5 days because leap years exist.
    # ========================================================

    cutoff_date = (
        pd.Timestamp.now()
        .normalize()
        - pd.DateOffset(
            years=SEC_LOOKBACK_YEARS
        )
    )


    # ========================================================
    # STEP 1 - Recent filings
    # ========================================================

    recent = (
        submissions
        .get("filings", {})
        .get("recent", {})
    )


    # SEC recent data is stored as parallel arrays.
    #
    # Example:
    #
    # form:
    #     ["10-Q", "8-K", "10-K"]
    #
    # filingDate:
    #     ["2026-08-03", ...]
    #
    # DataFrame converts those parallel arrays into rows.
    recent_filings = pd.DataFrame(
        recent
    )


    # Store all filing DataFrames here.
    filing_frames = []


    if not recent_filings.empty:

        filing_frames.append(
            recent_filings
        )


    # ========================================================
    # STEP 2 - Historical SEC files
    # ========================================================
    #
    # SEC may provide something like:
    #
    # submissions
    #   └── filings
    #         ├── recent
    #         └── files
    #               ├── historical file 1
    #               ├── historical file 2
    #               └── ...
    #
    # Not every company needs historical files.
    #
    # This explains why some companies may have:
    #
    # AAPL = many filings
    #
    # while:
    #
    # another ticker = only 1 filing
    #
    # when we looked only at "recent".
    # ========================================================

    historical_files = (
        submissions
        .get("filings", {})
        .get("files", [])
    )


    for file_info in historical_files:

        file_name = (
            file_info.get("name")
        )

        filing_to = pd.to_datetime(
            file_info.get("filingTo"),
            errors="coerce",
        )


        # ----------------------------------------------------
        # Skip historical files that are completely outside
        # our five-year window.
        # ----------------------------------------------------
        #
        # Example:
        #
        # cutoff:
        #     2021-08-24
        #
        # historical file contains:
        #     1994 → 2015
        #
        # There is no reason to download that file.
        # ----------------------------------------------------

        if (
            pd.notna(filing_to)
            and filing_to < cutoff_date
        ):

            continue


        # If SEC didn't give us a filename, skip safely.
        if not file_name:

            continue


        print(
            f"    Fetching historical SEC file: "
            f"{file_name}"
        )


        try:

            historical_json = (
                fetch_historical_submission_file(
                    session=session,
                    file_name=file_name,
                )
            )


        except requests.RequestException as error:

            # One old history file failing should not destroy
            # the whole company's extraction.
            print(
                f"    [WARNING] Historical file failed: "
                f"{error}"
            )

            continue


        # Historical SEC submission files use the same
        # column-oriented structure.
        historical_df = pd.DataFrame(
            historical_json
        )


        if not historical_df.empty:

            filing_frames.append(
                historical_df
            )


        # Be polite to SEC between historical requests.
        time.sleep(0.2)


    # ========================================================
    # STEP 3 - Validate
    # ========================================================

    if not filing_frames:

        return pd.DataFrame(
            columns=[
                "ticker",
                "company_name",
                "cik",
                "form_type",
                "filing_date",
                "report_date",
                "accession_number",
                "primary_document",
            ]
        )


    # ========================================================
    # STEP 4 - Combine recent + historical filings
    # ========================================================

    filings = pd.concat(
        filing_frames,
        ignore_index=True,
    )


    # ========================================================
    # STEP 5 - Keep only 10-K and 10-Q
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
    # STEP 6 - Convert dates
    # ========================================================

    filings["filingDate"] = pd.to_datetime(
        filings["filingDate"],
        errors="coerce",
    )


    filings["reportDate"] = pd.to_datetime(
        filings["reportDate"],
        errors="coerce",
    )


    # Remove malformed records without a filing date.
    filings = filings[
        filings["filingDate"].notna()
    ].copy()


    # ========================================================
    # STEP 7 - Keep only last five years
    # ========================================================

    filings = filings[
        filings["filingDate"]
        >= cutoff_date
    ].copy()


    # ========================================================
    # STEP 8 - Remove duplicate filings
    # ========================================================
    #
    # accessionNumber uniquely identifies one SEC filing.
    #
    # A record could theoretically appear in both:
    #
    #     recent
    #
    # and:
    #
    #     historical file
    #
    # around the boundary.
    #
    # We only want it once.
    # ========================================================

    filings = filings.drop_duplicates(
        subset=[
            "accessionNumber"
        ],
        keep="first",
    )


    # ========================================================
    # STEP 9 - Add AlphaLens company metadata
    # ========================================================

    filings["ticker"] = ticker

    filings["company_name"] = (
        company_name
    )

    filings["cik"] = cik


    # ========================================================
    # STEP 10 - Keep only required columns
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
    # STEP 11 - Rename to AlphaLens snake_case format
    # ========================================================

    filings = filings.rename(
        columns={
            "form":
                "form_type",

            "filingDate":
                "filing_date",

            "reportDate":
                "report_date",

            "accessionNumber":
                "accession_number",

            "primaryDocument":
                "primary_document",
        }
    )


    # ========================================================
    # STEP 12 - Sort newest first
    # ========================================================

    filings = filings.sort_values(
        by="filing_date",
        ascending=False,
    ).reset_index(
        drop=True
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
            session=session,
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
