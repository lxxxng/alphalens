"""
AlphaLens - Market Data Pipeline

Purpose:
    Runs the complete OHLCV ETL pipeline.

ETL means:

    Extract
        ↓
    Transform
        ↓
    Load


Full flow:

    Yahoo Finance
          ↓
    extract_market_data()
          ↓
    Raw Pandas DataFrame
          ↓
    transform_market_data()
          ↓
    Clean Pandas DataFrame
          ↓
    load_market_data()
          ↓
    PostgreSQL


Before running this pipeline
----------------------------

The PostgreSQL table market_prices must exist before the LOAD step runs.
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


from pipelines.market_data.extractor import (
    extract_market_data,
)

from pipelines.market_data.transformer import (
    transform_market_data,
)

from pipelines.market_data.loader import (
    load_market_data,
)


def run_market_pipeline():
    """
    Execute the complete AlphaLens market-data ETL pipeline.

    Steps
    -----
    1. Extract OHLCV from Yahoo Finance.
    2. Transform Yahoo's MultiIndex DataFrame.
    3. Load normalized rows into PostgreSQL.
    """

    # ========================================================
    # STEP 1 - EXTRACT
    # ========================================================

    print("\n========================================")
    print("STEP 1 - EXTRACT")
    print("========================================")

    raw_data = extract_market_data()

    print(
        f"Raw rows: {len(raw_data)}"
    )

    # ========================================================
    # STEP 2 - TRANSFORM
    # ========================================================

    print("\n========================================")
    print("STEP 2 - TRANSFORM")
    print("========================================")

    clean_data = transform_market_data(
        raw_data
    )

    print(
        f"Transformed rows: {len(clean_data)}"
    )

    # ========================================================
    # STEP 3 - LOAD
    # ========================================================

    print("\n========================================")
    print("STEP 3 - LOAD")
    print("========================================")

    loaded_rows = load_market_data(
        clean_data
    )

    print("\n========================================")
    print("PIPELINE COMPLETE")
    print("========================================")

    print(
        f"Rows processed: {loaded_rows}"
    )


# ============================================================
# Script Entry Point
# ============================================================

if __name__ == "__main__":
    run_market_pipeline()