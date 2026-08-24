/*
============================================================
AlphaLens - SEC Filing Metadata Schema
============================================================

Purpose:
    Stores metadata for SEC 10-K and 10-Q filings.

Important:
    We are NOT storing the entire filing document here yet.

At this stage PostgreSQL stores information such as:

    ticker
    CIK
    filing type
    filing date
    report date
    accession number
    primary HTML document filename

Later:
    accession_number + CIK + primary_document
                    ↓
             construct SEC URL
                    ↓
             download filing HTML
*/


-- ==========================================================
-- SEC Filings
-- ==========================================================

CREATE TABLE IF NOT EXISTS filings (

    /*
    SEC accession number.

    Example:
        0000320193-25-000079

    This uniquely identifies one SEC submission.

    Because it is unique for every filing, we use it
    as the PRIMARY KEY.
    */
    accession_number VARCHAR(30) PRIMARY KEY,


    /*
    Stock ticker.

    Example:
        AAPL
        MSFT
        NVDA

    REFERENCES companies(ticker) creates a relationship
    between this filing and the companies table.
    */
    ticker VARCHAR(20) NOT NULL
        REFERENCES companies(ticker),


    /*
    SEC Central Index Key.

    Example:
        0000320193

    Stored as text rather than a number because the
    leading zeros are important when constructing SEC URLs.
    */
    cik VARCHAR(10) NOT NULL,


    /*
    Filing type.

    For this project initially:
        10-K = annual report
        10-Q = quarterly report
    */
    form_type VARCHAR(10) NOT NULL,


    /*
    Date the company submitted the filing to SEC.
    */
    filing_date DATE NOT NULL,


    /*
    Financial period covered by the filing.

    This can be earlier than filing_date.

    Example:
        report_date = 2026-06-30
        filing_date = 2026-07-30
    */
    report_date DATE,


    /*
    Main HTML document supplied in the SEC filing.

    Example:
        aapl-20250927.htm
    */
    primary_document TEXT NOT NULL,


    /*
    We will populate this later after constructing
    the full SEC Archive URL.
    */
    source_url TEXT,


    /*
    When AlphaLens first inserted this metadata.
    */
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),


    /*
    Useful later if metadata gets refreshed.
    */
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);


-- ==========================================================
-- Index: Ticker
-- ==========================================================
--
-- Helps queries such as:
--
-- SELECT *
-- FROM filings
-- WHERE ticker = 'AAPL';
-- ==========================================================

CREATE INDEX IF NOT EXISTS idx_filings_ticker
ON filings (ticker);


-- ==========================================================
-- Index: Filing Date
-- ==========================================================
--
-- Helps date-based searches.
-- ==========================================================

CREATE INDEX IF NOT EXISTS idx_filings_filing_date
ON filings (filing_date);


-- ==========================================================
-- Combined Index
-- ==========================================================
--
-- Useful when asking:
--
-- "Give me Apple's latest filings"
-- ==========================================================

CREATE INDEX IF NOT EXISTS idx_filings_ticker_date
ON filings (ticker, filing_date DESC);