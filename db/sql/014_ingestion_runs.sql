/*
============================================================
AlphaLens - Scheduled Ingestion Run History
============================================================

Purpose:
    Make automated refreshes observable. Each run records its ticker scope,
    current stage, terminal status, and compact per-stage results so failures
    can be diagnosed without relying only on a scheduler console window.
============================================================
*/


CREATE TABLE IF NOT EXISTS ingestion_runs (

    run_id BIGSERIAL PRIMARY KEY,

    trigger_type VARCHAR(30) NOT NULL DEFAULT 'manual',

    scope VARCHAR(30) NOT NULL,

    tickers TEXT[] NOT NULL DEFAULT '{}',

    status VARCHAR(20) NOT NULL DEFAULT 'RUNNING'
        CHECK (status IN ('RUNNING', 'SUCCEEDED', 'FAILED', 'SKIPPED')),

    current_stage VARCHAR(80),

    stage_results JSONB NOT NULL DEFAULT '[]'::JSONB,

    error TEXT,

    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    completed_at TIMESTAMPTZ,

    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);


CREATE INDEX IF NOT EXISTS idx_ingestion_runs_started_at
ON ingestion_runs (started_at DESC);


CREATE INDEX IF NOT EXISTS idx_ingestion_runs_status
ON ingestion_runs (status, started_at DESC);
