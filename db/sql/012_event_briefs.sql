/*
============================================================
AlphaLens - Saved Event Briefs
============================================================

Purpose:
    Preserve generated event briefs with the exact evidence, market facts,
    sentiment signals, and model version used at generation time.

Behavior:
    Rows are immutable snapshots. Multiple rows may share a cache key when a
    user explicitly refreshes a brief; normal generation reuses the newest
    matching snapshot while the underlying event fingerprint is unchanged.
============================================================
*/


CREATE TABLE IF NOT EXISTS event_briefs (

    brief_id BIGSERIAL PRIMARY KEY,

    cache_key CHAR(64) NOT NULL,

    tickers JSONB NOT NULL DEFAULT '[]'::JSONB,

    event_type VARCHAR(20) NOT NULL,

    fiscal_period VARCHAR(20),

    form_type VARCHAR(20),

    top_k INTEGER NOT NULL
        CHECK (top_k BETWEEN 1 AND 12),

    focus TEXT,

    question TEXT NOT NULL,

    headline TEXT NOT NULL,

    brief JSONB NOT NULL,

    market_context JSONB NOT NULL DEFAULT '[]'::JSONB,

    sentiment_context JSONB NOT NULL DEFAULT '{}'::JSONB,

    sources JSONB NOT NULL DEFAULT '[]'::JSONB,

    source_count INTEGER NOT NULL DEFAULT 0,

    event_fingerprint JSONB NOT NULL DEFAULT '{}'::JSONB,

    chain_version VARCHAR(100) NOT NULL,

    model_name VARCHAR(100) NOT NULL,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);


CREATE INDEX IF NOT EXISTS idx_event_briefs_created_at
ON event_briefs (created_at DESC, brief_id DESC);


CREATE INDEX IF NOT EXISTS idx_event_briefs_cache_key
ON event_briefs (cache_key, created_at DESC, brief_id DESC);

