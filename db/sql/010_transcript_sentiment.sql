/*
============================================================
AlphaLens - Versioned Transcript Sentiment
============================================================

Purpose:
    Preserve reproducible FinBERT results for each transcript turn while the
    turn table exposes the latest label and polarity score for convenient UI
    and API reads.
============================================================
*/


CREATE TABLE IF NOT EXISTS earnings_transcript_turn_sentiment (

    turn_sentiment_id BIGSERIAL PRIMARY KEY,

    turn_id BIGINT NOT NULL
        REFERENCES earnings_transcript_turns(turn_id)
        ON DELETE CASCADE,

    transcript_id BIGINT NOT NULL
        REFERENCES earnings_transcripts(transcript_id)
        ON DELETE CASCADE,

    model_name VARCHAR(200) NOT NULL,

    -- The configured Hugging Face revision, such as main or a commit hash.
    model_revision VARCHAR(100) NOT NULL,

    -- Resolved model commit when Transformers exposes it at runtime.
    model_commit VARCHAR(100),

    sentiment_label VARCHAR(20),

    -- Positive probability minus negative probability, bounded to [-1, 1].
    sentiment_score DOUBLE PRECISION
        CHECK (sentiment_score BETWEEN -1 AND 1),

    confidence DOUBLE PRECISION
        CHECK (confidence BETWEEN 0 AND 1),

    probabilities JSONB,

    token_count INTEGER,

    segment_count INTEGER,

    status VARCHAR(20) NOT NULL
        CHECK (status IN ('SCORED', 'SKIPPED', 'FAILED')),

    error TEXT,

    scored_at TIMESTAMPTZ,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE (turn_id, model_name, model_revision)
);


CREATE INDEX IF NOT EXISTS idx_turn_sentiment_transcript
ON earnings_transcript_turn_sentiment (
    transcript_id,
    model_name,
    model_revision,
    status
);


CREATE INDEX IF NOT EXISTS idx_turn_sentiment_status
ON earnings_transcript_turn_sentiment (
    model_name,
    model_revision,
    status
);
