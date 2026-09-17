/*
============================================================
AlphaLens - Point-in-Time Earnings Results
============================================================

Purpose:
    Store reported-versus-estimated EPS from Yahoo Finance for event-level
    modeling. Only announced results are loaded; future estimate-only rows
    are rejected by the extractor.

Point-in-time rule:
    earnings_date and earnings_timestamp describe when the result became
    observable. Model features must join on this event date, never on a
    fiscal period end date.
============================================================
*/


CREATE TABLE IF NOT EXISTS earnings_results (

    earnings_result_id BIGSERIAL PRIMARY KEY,

    ticker VARCHAR(20) NOT NULL
        REFERENCES companies(ticker)
        ON DELETE CASCADE,

    -- Yahoo's local calendar date for the earnings announcement.
    earnings_date DATE NOT NULL,

    -- Time-zone-aware source timestamp, normalized to UTC by the extractor.
    earnings_timestamp TIMESTAMPTZ NOT NULL,

    eps_estimate DOUBLE PRECISION,

    reported_eps DOUBLE PRECISION NOT NULL,

    -- Absolute reported EPS minus consensus estimate.
    eps_surprise DOUBLE PRECISION,

    -- Decimal rate: 0.0927 represents a 9.27% positive surprise.
    eps_surprise_pct DOUBLE PRECISION,

    source_provider VARCHAR(40) NOT NULL DEFAULT 'yahoo_finance',

    source_url TEXT,

    raw_payload JSONB NOT NULL DEFAULT '{}'::JSONB,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE (ticker, earnings_date, source_provider)
);


CREATE INDEX IF NOT EXISTS idx_earnings_results_ticker_date
ON earnings_results (ticker, earnings_date DESC);

