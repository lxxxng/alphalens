"""
AlphaLens - Market Data Loader

Purpose:
    Loads the cleaned OHLCV DataFrame produced by transformer.py
    into PostgreSQL.

Pipeline position:

    Yahoo Finance
        ↓
    extractor.py
        ↓
    transformer.py
        ↓
    loader.py
        ↓
    PostgreSQL
        ↓
    market_prices


Important concept - UPSERT
--------------------------

The table uses this primary key:

    (ticker, trading_date)

Example:

    AAPL + 2026-08-20

must only exist once.

If the pipeline runs again tomorrow, it may download
2026-08-20 again.

Instead of creating duplicates, PostgreSQL will:

    INSERT new rows

or

    UPDATE existing rows

This behaviour is called an UPSERT.


Before loading data
-------------------

The PostgreSQL table market_prices must exist before this loader runs.
If PostgreSQL is running in Docker, apply the initial schema first:

    Get-Content db\\sql\\001_initial_schema.sql | docker exec -i alphalens-postgres psql -U alphalens -d alphalens

The equivalent input-redirection command is:

    docker exec -i alphalens-postgres psql -U alphalens -d alphalens < db\\sql\\001_initial_schema.sql

Windows PowerShell may not handle input redirection with < reliably
for native commands, so the Get-Content pipe command is recommended.

To inspect the database afterward, open psql:

    docker exec -it alphalens-postgres psql -U alphalens -d alphalens

Then run these commands inside PostgreSQL:

    \\dt
    \\d market_prices

The table list should include companies and market_prices.

Command options:

    docker exec -i
        Runs a command inside the container and keeps standard input
        open so the SQL file can be passed through the PowerShell pipe.

    psql -U alphalens
        Connects to PostgreSQL as the alphalens user. -U means user.

    psql -d alphalens
        Connects to the alphalens database. -d means database.

    docker exec -it
        Opens an interactive terminal. -i keeps input open and -t
        allocates a terminal so you can type PostgreSQL commands.
"""

import os

import pandas as pd

from dotenv import load_dotenv

from sqlalchemy import (
    MetaData,
    Table,
    create_engine,
)

from sqlalchemy.dialects.postgresql import insert


# ============================================================
# Load Environment Variables
# ============================================================
#
# Reads variables from:
#
#     .env
#
# Example:
#
# DATABASE_URL=
# postgresql+psycopg2://alphalens:password@localhost:5432/alphalens
#
# Keeping credentials in .env prevents passwords from being
# hard-coded into Python source code.
# ============================================================

load_dotenv()


# ============================================================
# get_database_engine()
# ============================================================

def get_database_engine():
    """
    Create and return a SQLAlchemy PostgreSQL engine.

    Returns
    -------
    sqlalchemy.Engine
        SQLAlchemy database engine used to communicate
        with PostgreSQL.

    Raises
    ------
    ValueError
        If DATABASE_URL does not exist in the .env file.

    What is an Engine?
    ------------------
    Think of the SQLAlchemy Engine as Python's main gateway
    to the PostgreSQL database.

        Python
           ↓
        SQLAlchemy Engine
           ↓
        psycopg2
           ↓
        PostgreSQL
    """

    # Read DATABASE_URL from the environment.
    database_url = os.getenv("DATABASE_URL")

    # Stop immediately if the database configuration
    # has not been provided.
    if not database_url:
        raise ValueError(
            "DATABASE_URL was not found. "
            "Check your .env file."
        )

    # Create the SQLAlchemy Engine.
    #
    # pool_pre_ping=True:
    #     Checks that a database connection is still alive
    #     before SQLAlchemy reuses it.
    #
    #     This is useful for longer-running applications where
    #     database connections may occasionally become stale.
    engine = create_engine(
        database_url,
        pool_pre_ping=True,
    )

    return engine


# ============================================================
# load_market_data()
# ============================================================

def load_market_data(
    clean_data: pd.DataFrame,
    batch_size: int = 1000,
):
    """
    Insert or update transformed OHLCV data in PostgreSQL.

    Parameters
    ----------
    clean_data : pandas.DataFrame
        DataFrame produced by transform_market_data().

        Expected columns:

            ticker
            trading_date
            open
            high
            low
            close
            adjusted_close
            volume

    batch_size : int
        Number of rows inserted in each database batch.

        Example:

            13,900 rows
               ↓
            batch_size = 1000
               ↓
            about 14 database batches

        Batching prevents us from constructing one extremely
        large database operation.

    Returns
    -------
    int
        Number of rows that were submitted to PostgreSQL.

    Important:
        The function performs an UPSERT.

        New ticker/date:
            INSERT

        Existing ticker/date:
            UPDATE
    """

    # ========================================================
    # Validation - DataFrame cannot be empty
    # ========================================================

    if clean_data.empty:
        raise ValueError(
            "Cannot load market data because "
            "the transformed DataFrame is empty."
        )

    # ========================================================
    # Required Columns
    # ========================================================

    required_columns = [
        "ticker",
        "trading_date",
        "open",
        "high",
        "low",
        "close",
        "adjusted_close",
        "volume",
    ]

    # Find any expected columns that are missing.
    missing_columns = [
        column
        for column in required_columns
        if column not in clean_data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # ========================================================
    # Connect to PostgreSQL
    # ========================================================

    engine = get_database_engine()

    # ========================================================
    # Read Existing PostgreSQL Table Definition
    # ========================================================
    #
    # MetaData is SQLAlchemy's representation of database
    # schema information.
    #
    # autoload_with=engine means:
    #
    #     "Ask PostgreSQL what columns market_prices has."
    #
    # Therefore we don't need to manually redefine the table
    # structure inside Python.
    # ========================================================

    metadata = MetaData()

    market_prices = Table(
        "market_prices",
        metadata,
        autoload_with=engine,
    )

    # ========================================================
    # Prepare DataFrame for PostgreSQL
    # ========================================================

    database_data = clean_data[
        required_columns
    ].copy()

    # --------------------------------------------------------
    # Convert Pandas missing values to Python None
    # --------------------------------------------------------
    #
    # Pandas may represent missing data using:
    #
    #     NaN
    #     pd.NA
    #
    # PostgreSQL expects:
    #
    #     NULL
    #
    # SQLAlchemy converts Python None into SQL NULL.
    # --------------------------------------------------------

    database_data = (
        database_data
        .astype(object)
        .where(pd.notna(database_data), None)
    )

    # --------------------------------------------------------
    # Convert the DataFrame into Python dictionaries
    # --------------------------------------------------------
    #
    # DataFrame:
    #
    # ticker | trading_date | close
    # AAPL   | 2026-08-20   | 227
    #
    # becomes:
    #
    # {
    #     "ticker": "AAPL",
    #     "trading_date": ...,
    #     "close": 227
    # }
    #
    # SQLAlchemy can insert dictionaries directly.
    # --------------------------------------------------------

    # orient="records" converts each DataFrame row into one
    # dictionary, using column names as dictionary keys. This
    # is the format SQLAlchemy expects for batch inserts.
    records = database_data.to_dict(
        orient="records"
    )

    total_rows = len(records)

    print(
        f"\nPreparing to load {total_rows} "
        "rows into PostgreSQL..."
    )

    # ========================================================
    # Database Transaction
    # ========================================================
    #
    # engine.begin() starts a transaction.
    #
    # If everything succeeds:
    #
    #     COMMIT
    #
    # If an exception occurs:
    #
    #     ROLLBACK
    #
    # This prevents partially completed transactions from
    # leaving the database in an inconsistent state.
    # ========================================================

    with engine.begin() as connection:

        # Process rows in batches.
        for start in range(
            0,
            total_rows,
            batch_size,
        ):

            end = start + batch_size

            batch = records[start:end]

            # =================================================
            # PostgreSQL INSERT statement
            # =================================================

            # Build an INSERT statement for the market_prices table.
            # The row values are supplied later when this statement
            # is executed with the current batch of dictionaries.
            insert_statement = insert(
                market_prices
            )

            # =================================================
            # ON CONFLICT = UPSERT
            # =================================================
            #
            # Conflict definition:
            #
            #     ticker + trading_date
            #
            # Example:
            #
            # Database already contains:
            #
            # AAPL | 2026-08-20
            #
            # Pipeline tries to insert:
            #
            # AAPL | 2026-08-20
            #
            # PostgreSQL detects the primary-key conflict.
            #
            # Instead of failing, update the OHLCV values.
            # =================================================

            upsert_statement = (
                insert_statement
                .on_conflict_do_update(

                    index_elements=[
                        "ticker",
                        "trading_date",
                    ],

                    set_={
                        # insert_statement.excluded refers to the values
                        # from the incoming row that caused the conflict.
                        # Use them to update the existing database row.
                        "open":
                            insert_statement.excluded.open,

                        "high":
                            insert_statement.excluded.high,

                        "low":
                            insert_statement.excluded.low,

                        "close":
                            insert_statement.excluded.close,

                        "adjusted_close":
                            insert_statement.excluded.adjusted_close,

                        "volume":
                            insert_statement.excluded.volume,
                    },
                )
            )

            # Execute this batch.
            connection.execute(
                upsert_statement,
                batch,
            )

            # Display progress.
            processed = min(
                end,
                total_rows,
            )

            print(
                f"Loaded {processed}/{total_rows} rows"
            )

    print(
        "\nMarket data successfully loaded "
        "into PostgreSQL."
    )

    return total_rows