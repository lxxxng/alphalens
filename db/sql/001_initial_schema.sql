/*
============================================================
AlphaLens - Initial PostgreSQL Schema
============================================================

Purpose:
    Creates the first database tables used by AlphaLens.

Tables:
    1. companies
       Stores basic information about each company.

    2. market_prices
       Stores one daily OHLCV row for each ticker.

Primary Key:
    (ticker, trading_date)

    This means:

        AAPL + 2026-08-20

    can only exist once.

    Therefore rerunning the ETL pipeline cannot create
    duplicate rows for the same ticker/date combination.

Applying this schema from PowerShell:

    Get-Content db\sql\001_initial_schema.sql | docker exec -i alphalens-postgres psql -U alphalens -d alphalens

The equivalent input-redirection command is:

    docker exec -i alphalens-postgres psql -U alphalens -d alphalens < db\sql\001_initial_schema.sql

The pipe command is recommended in Windows PowerShell because
input redirection with < may not work reliably for native commands.

To verify the tables, open PostgreSQL with:

    docker exec -it alphalens-postgres psql -U alphalens -d alphalens

Then run inside PostgreSQL:

    \dt
    \d market_prices

The table list should include companies and market_prices.
============================================================
*/


-- ==========================================================
-- Companies
-- ==========================================================

CREATE TABLE IF NOT EXISTS companies (

    -- Stock ticker, e.g. AAPL, MSFT, NVDA
    ticker VARCHAR(20) PRIMARY KEY,

    -- Full company name
    company_name VARCHAR(255),

    -- SEC Central Index Key.
    -- We will use this later when downloading SEC filings.
    cik VARCHAR(20),

    -- Example: Technology, Financial Services, Energy
    sector VARCHAR(100),

    -- More specific business classification
    industry VARCHAR(150),

    -- Records when this database row was created
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);


-- ==========================================================
-- Daily OHLCV Market Prices
-- ==========================================================

CREATE TABLE IF NOT EXISTS market_prices (

    -- Stock symbol
    ticker VARCHAR(20) NOT NULL,

    -- Trading day
    trading_date DATE NOT NULL,

    -- Opening price
    open DOUBLE PRECISION,

    -- Highest price during the trading day
    high DOUBLE PRECISION,

    -- Lowest price during the trading day
    low DOUBLE PRECISION,

    -- Closing price
    close DOUBLE PRECISION,

    -- Closing price adjusted for events such as
    -- stock splits and dividends
    adjusted_close DOUBLE PRECISION,

    -- Number of shares traded
    volume BIGINT,

    -- Time AlphaLens first inserted the row
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- One ticker can only have one row for one trading date.
    PRIMARY KEY (ticker, trading_date)
);


-- ==========================================================
-- Trading Date Index
-- ==========================================================
--
-- Helps queries that search/filter by date.
--
-- Example:
--
-- SELECT *
-- FROM market_prices
-- WHERE trading_date >= '2026-01-01';
-- ==========================================================

CREATE INDEX IF NOT EXISTS idx_market_prices_date
ON market_prices (trading_date);