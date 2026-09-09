/*
============================================================
AlphaLens - Saved Research Runs
============================================================

Purpose:
    Preserve successful generated answers together with the exact market
    context and citations shown to the user.

Behavior:
    Research runs are immutable snapshots. Opening an old run should show
    the original evidence rather than silently running retrieval again.
============================================================
*/


CREATE TABLE IF NOT EXISTS research_runs (

    -- Stable identifier returned to the API and frontend.
    run_id BIGSERIAL PRIMARY KEY,

    question TEXT NOT NULL,

    answer TEXT NOT NULL,

    -- Tickers are stored as JSON because one question may compare several
    -- companies. The original explicit ticker filter remains separate.
    tickers JSONB NOT NULL DEFAULT '[]'::JSONB,

    ticker_filter VARCHAR(20),

    source_type VARCHAR(20) NOT NULL,

    top_k INTEGER NOT NULL
        CHECK (top_k BETWEEN 1 AND 20),

    form_type VARCHAR(20),

    section_key VARCHAR(100),

    fiscal_period VARCHAR(20),

    -- Counts let the history list stay lightweight without loading large
    -- evidence arrays for every saved run.
    source_count INTEGER NOT NULL DEFAULT 0,

    market_snapshot_count INTEGER NOT NULL DEFAULT 0,

    market_context JSONB NOT NULL DEFAULT '[]'::JSONB,

    sources JSONB NOT NULL DEFAULT '[]'::JSONB,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);


CREATE INDEX IF NOT EXISTS idx_research_runs_created_at
ON research_runs (created_at DESC, run_id DESC);


CREATE INDEX IF NOT EXISTS idx_research_runs_ticker_filter
ON research_runs (ticker_filter, created_at DESC);
