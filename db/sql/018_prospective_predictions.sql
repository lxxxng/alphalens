/*
============================================================
AlphaLens - Prospective Return Prediction Ledger
============================================================

Purpose:
    Preserve predictions made before their 10-session outcomes are known.
    Original forecasts and feature snapshots are immutable evidence; a later
    maturation pass may only attach realized returns and evaluation status.
============================================================
*/


CREATE TABLE IF NOT EXISTS prospective_predictions (

    prediction_id BIGSERIAL PRIMARY KEY,

    model_version VARCHAR(160) NOT NULL,

    event_key TEXT NOT NULL,

    ticker VARCHAR(20) NOT NULL
        REFERENCES companies(ticker)
        ON DELETE CASCADE,

    event_source VARCHAR(30) NOT NULL
        CHECK (event_source IN ('earnings_call', 'sec_filing')),

    event_id TEXT NOT NULL,

    event_date DATE NOT NULL,

    feature_as_of_date DATE NOT NULL,

    prediction_generated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    training_cutoff DATE NOT NULL,

    horizon_sessions INTEGER NOT NULL
        CHECK (horizon_sessions > 0),

    benchmark_ticker VARCHAR(20) NOT NULL DEFAULT 'SPY',

    predicted_excess_return DOUBLE PRECISION NOT NULL,

    baseline_predicted_excess_return DOUBLE PRECISION NOT NULL,

    feature_snapshot JSONB NOT NULL,

    status VARCHAR(20) NOT NULL DEFAULT 'PENDING'
        CHECK (status IN ('PENDING', 'MATURED', 'INVALID')),

    target_trading_date DATE,

    stock_forward_return DOUBLE PRECISION,

    benchmark_forward_return DOUBLE PRECISION,

    realized_excess_return DOUBLE PRECISION,

    matured_at TIMESTAMPTZ,

    outcome_error TEXT,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE (model_version, event_key),

    CHECK (
        status <> 'MATURED'
        OR (
            target_trading_date IS NOT NULL
            AND stock_forward_return IS NOT NULL
            AND benchmark_forward_return IS NOT NULL
            AND realized_excess_return IS NOT NULL
            AND matured_at IS NOT NULL
        )
    )
);


CREATE INDEX IF NOT EXISTS idx_prospective_predictions_status
ON prospective_predictions (model_version, status, feature_as_of_date);


CREATE INDEX IF NOT EXISTS idx_prospective_predictions_ticker
ON prospective_predictions (ticker, prediction_generated_at DESC);
