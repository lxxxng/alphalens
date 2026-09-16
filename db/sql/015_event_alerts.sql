/*
============================================================
AlphaLens - Watched Company Event Alerts
============================================================

Purpose:
    Persist one deduplicated in-app alert when scheduled ingestion first
    discovers an SEC filing or earnings call for a monitored company.

Behavior:
    Provider retries and pipeline reruns cannot duplicate an event because
    event_type + source_record_id is unique. Alerts remain available after a
    company leaves a watchlist so the user's read history is not rewritten.
============================================================
*/


CREATE TABLE IF NOT EXISTS event_alerts (

    alert_id BIGSERIAL PRIMARY KEY,

    ticker VARCHAR(20) NOT NULL
        REFERENCES companies(ticker)
        ON DELETE CASCADE,

    event_type VARCHAR(20) NOT NULL
        CHECK (event_type IN ('filing', 'earnings')),

    source_record_id TEXT NOT NULL,

    event_date DATE,

    title TEXT NOT NULL,

    message TEXT,

    source_url TEXT,

    ingestion_run_id BIGINT
        REFERENCES ingestion_runs(run_id)
        ON DELETE SET NULL,

    is_read BOOLEAN NOT NULL DEFAULT FALSE,

    read_at TIMESTAMPTZ,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE (event_type, source_record_id)
);


CREATE INDEX IF NOT EXISTS idx_event_alerts_unread_created
ON event_alerts (is_read, created_at DESC);


CREATE INDEX IF NOT EXISTS idx_event_alerts_ticker_created
ON event_alerts (ticker, created_at DESC);
