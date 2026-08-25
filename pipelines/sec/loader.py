"""
AlphaLens - SEC Filing Metadata Loader

Purpose:
    Loads SEC 10-K / 10-Q metadata extracted by extractor.py
    into PostgreSQL.

Pipeline:

    SEC EDGAR
        ↓
    extractor.py
        ↓
    Pandas DataFrame
        ↓
    loader.py
        ↓
    PostgreSQL
        ├── companies
        └── filings


Why two tables?
---------------

companies:
    One row per company.

    Example:

        AAPL
        Apple Inc.
        CIK 0000320193


filings:
    Many filings can belong to one company.

    Example:

        AAPL
          ├── 2025 10-K
          ├── 2026 Q1 10-Q
          ├── 2026 Q2 10-Q
          └── ...


Relationship:

    companies
       1
       │
       │
       ▼
       many
    filings
"""

import os

import pandas as pd

from dotenv import load_dotenv

from sqlalchemy import (
    bindparam,
    MetaData,
    Table,
    create_engine,
    text,
)

from sqlalchemy.dialects.postgresql import insert

from pipelines.market_data.extractor import TICKERS
from pipelines.sec.extractor import SEC_LOOKBACK_YEARS


# ============================================================
# Load Environment Variables
# ============================================================

load_dotenv()


# ============================================================
# get_database_engine()
# ============================================================

def get_database_engine():
    """
    Create a SQLAlchemy connection engine for PostgreSQL.

    DATABASE_URL is read from .env.

    Example:

        postgresql+psycopg2://
        alphalens:password@localhost:5432/alphalens
    """

    database_url = os.getenv(
        "DATABASE_URL"
    )

    if not database_url:
        raise ValueError(
            "DATABASE_URL was not found in .env."
        )

    engine = create_engine(
        database_url,
        pool_pre_ping=True,
    )

    return engine


# ============================================================
# load_companies()
# ============================================================

def load_companies(
    filings: pd.DataFrame,
):
    """
    Insert/update SEC company information in the companies table.

    Parameters
    ----------
    filings : pandas.DataFrame
        DataFrame returned by the SEC extractor.

    Why do this?
    ------------
    The SEC extractor already gives us:

        ticker
        company_name
        cik

    Instead of repeatedly storing company_name in every filing,
    we keep one company record inside:

        companies

    Example:

        ticker = AAPL
        company_name = Apple Inc.
        cik = 0000320193
    """

    engine = get_database_engine()

    # ========================================================
    # Extract unique companies
    # ========================================================
    #
    # The filings DataFrame might contain:
    #
    # AAPL 10-K
    # AAPL 10-Q
    # AAPL 10-Q
    # MSFT 10-K
    #
    # We only want:
    #
    # AAPL
    # MSFT
    #
    # once each.
    # ========================================================

    companies_data = (
        filings[
            [
                "ticker",
                "company_name",
                "cik",
            ]
        ]
        .drop_duplicates(
            subset=["ticker"]
        )
        .copy()
    )

    # Convert DataFrame rows into dictionaries.
    records = companies_data.to_dict(
        orient="records"
    )

    # 1. MetaData() creates a container.
    # 2. autoload_with=engine reads the existing companies table.
    # 3. SQLAlchemy stores that structure inside companies_table.
    # 4. insert(companies_table) uses it to generate correct SQL.
    metadata = MetaData()

    companies_table = Table(
        "companies",
        metadata,
        autoload_with=engine,
    )

    # ========================================================
    # UPSERT Companies
    # ========================================================
    #
    # If AAPL does not exist:
    #
    #     INSERT
    #
    # If AAPL already exists:
    #
    #     UPDATE company_name / cik
    #
    # ========================================================

    insert_statement = insert(
        companies_table
    )

    upsert_statement = (
        insert_statement
        .on_conflict_do_update(

            # ticker is the companies primary key.
            index_elements=[
                "ticker"
            ],

            set_={
                "company_name":
                    insert_statement.excluded.company_name,

                "cik":
                    insert_statement.excluded.cik,
            },
        )
    )

    with engine.begin() as connection:

        connection.execute(
            upsert_statement,
            records,
        )

    print(
        f"[OK] Loaded {len(records)} companies"
    )


# ============================================================
# load_filing_metadata()
# ============================================================

def load_filing_metadata(
    filings: pd.DataFrame,
    batch_size: int = 500,
):
    """
    Insert/update SEC filing metadata in PostgreSQL.

    Parameters
    ----------
    filings : pandas.DataFrame
        Output from extract_sec_filing_metadata().

    batch_size : int
        Maximum number of filings processed per SQL batch.

    Returns
    -------
    int
        Number of filing rows processed.


    Primary key
    -----------

        accession_number

    Why?

    Every SEC filing has a unique accession number.

    Example:

        0000320193-25-000079

    Therefore if the pipeline downloads the same filing again,
    PostgreSQL recognizes it as the same record.
    """

    if filings.empty:

        raise ValueError(
            "Cannot load SEC filings because "
            "the DataFrame is empty."
        )

    required_columns = [
        "ticker",
        "cik",
        "form_type",
        "filing_date",
        "report_date",
        "accession_number",
        "primary_document",
    ]

    # ========================================================
    # Check required columns
    # ========================================================

    missing_columns = [
        column
        for column in required_columns
        if column not in filings.columns
    ]

    if missing_columns:

        raise ValueError(
            f"Missing filing columns: "
            f"{missing_columns}"
        )

    # ========================================================
    # Prepare data
    # ========================================================

    database_data = filings[
        required_columns
    ].copy()

    # PostgreSQL DATE works well with normal Python date objects.
    database_data["filing_date"] = (
        pd.to_datetime(
            database_data["filing_date"]
        )
        .dt.date
    )

    database_data["report_date"] = (
        pd.to_datetime(
            database_data["report_date"],
            errors="coerce",
        )
        .dt.date
    )

    # ========================================================
    # Convert Pandas missing values
    # ========================================================
    #
    # Pandas:
    #
    #     NaN
    #     NaT
    #     pd.NA
    #
    # PostgreSQL:
    #
    #     NULL
    #
    # SQLAlchemy converts Python None → PostgreSQL NULL.
    # ========================================================

    database_data = (
        database_data
        .astype(object)
        .where(
            pd.notna(database_data),
            None,
        )
    )

    records = database_data.to_dict(
        orient="records"
    )

    total_rows = len(records)

    print(
        f"\nPreparing to load "
        f"{total_rows} SEC filings..."
    )

    engine = get_database_engine()

    # 1. MetaData() creates a container.
    # 2. autoload_with=engine reads the existing filings table.
    # 3. SQLAlchemy stores that structure inside filings_table.
    # 4. insert(filings_table) uses it to generate correct SQL.
    metadata = MetaData()

    filings_table = Table(
        "filings",
        metadata,
        autoload_with=engine,
    )

    # ========================================================
    # Database Transaction
    # ========================================================

    with engine.begin() as connection:

        for start in range(
            0,
            total_rows,
            batch_size,
        ):

            end = start + batch_size

            batch = records[
                start:end
            ]

            # PostgreSQL INSERT command.
            insert_statement = insert(
                filings_table
            )

            # =================================================
            # UPSERT
            # =================================================
            #
            # If accession_number already exists:
            #
            # update the existing metadata.
            #
            # Otherwise:
            #
            # insert a new filing.
            # =================================================

            upsert_statement = (
                insert_statement
                .on_conflict_do_update(

                    index_elements=[
                        "accession_number"
                    ],

                    set_={
                        "ticker":
                            insert_statement.excluded.ticker,

                        "cik":
                            insert_statement.excluded.cik,

                        "form_type":
                            insert_statement.excluded.form_type,

                        "filing_date":
                            insert_statement.excluded.filing_date,

                        "report_date":
                            insert_statement.excluded.report_date,

                        "primary_document":
                            insert_statement.excluded.primary_document,

                        # Whenever an existing row is updated,
                        # refresh updated_at.
                        "updated_at":
                            pd.Timestamp.now(),
                    },
                )
            )

            connection.execute(
                upsert_statement,
                batch,
            )

            processed = min(
                end,
                total_rows,
            )

            print(
                f"Loaded "
                f"{processed}/{total_rows} filings"
            )

    print(
        "\nSEC filing metadata successfully "
        "loaded into PostgreSQL."
    )

    return total_rows


# ============================================================
# prune_stale_sec_filings()
# ============================================================

def prune_stale_sec_filings() -> int:
    """
    Remove SEC filing metadata outside the current pipeline scope.

    The loader uses UPSERTs, which is perfect for refreshing current
    filings but does not remove rows from older experiments or from a
    previous ticker universe. This keeps the filings table aligned with:

        current AlphaLens tickers
        +
        configured SEC lookback window

    Deleting from filings also removes related filing_sections because
    that table has ON DELETE CASCADE on accession_number.
    """

    cutoff_date = (
        pd.Timestamp.now()
        .normalize()
        - pd.DateOffset(
            years=SEC_LOOKBACK_YEARS
        )
    ).date()

    query = (
        text(
            """
            DELETE FROM filings

            WHERE ticker NOT IN :current_tickers
               OR filing_date < :cutoff_date

            RETURNING 1;
            """
        )
        .bindparams(
            bindparam(
                "current_tickers",
                expanding=True,
            )
        )
    )

    engine = get_database_engine()

    with engine.begin() as connection:

        result = connection.execute(
            query,
            {
                "current_tickers": TICKERS,
                "cutoff_date": cutoff_date,
            },
        )

        deleted_rows = len(
            result.fetchall()
        )

    print(
        f"[OK] Pruned {deleted_rows} stale SEC filings"
    )

    return deleted_rows


# ============================================================
# load_sec_data()
# ============================================================

def load_sec_data(
    filings: pd.DataFrame,
):
    """
    Load all SEC-related relational data.

    Order matters:

        1. companies
        2. filings

    Why?

    filings.ticker has a FOREIGN KEY pointing at:

        companies.ticker

    Therefore the company must exist before its filing
    can reference it.
    """

    # First ensure company records exist.
    load_companies(
        filings
    )

    # Then insert their filings.
    loaded_rows = load_filing_metadata(
        filings
    )

    # Finally remove older/out-of-universe metadata so downstream
    # document download, parsing, and section extraction match the
    # current five-year SEC universe.
    prune_stale_sec_filings()

    return loaded_rows
