/*
============================================================
AlphaLens - Versioned SEC Filing Sentiment
============================================================

Purpose:
    Preserve financial-language sentiment for selected narrative filing
    chunks. Financial statements and other table-heavy sections are excluded
    by the pipeline because polarity scores are not meaningful there.
============================================================
*/


CREATE TABLE IF NOT EXISTS filing_chunk_sentiment (

    chunk_sentiment_id BIGSERIAL PRIMARY KEY,

    chunk_id BIGINT NOT NULL
        REFERENCES filing_chunks(chunk_id)
        ON DELETE CASCADE,

    accession_number VARCHAR(30) NOT NULL
        REFERENCES filings(accession_number)
        ON DELETE CASCADE,

    model_name VARCHAR(200) NOT NULL,

    model_revision VARCHAR(100) NOT NULL,

    model_commit VARCHAR(100),

    sentiment_label VARCHAR(20),

    sentiment_score DOUBLE PRECISION
        CHECK (sentiment_score BETWEEN -1 AND 1),

    confidence DOUBLE PRECISION
        CHECK (confidence BETWEEN 0 AND 1),

    probabilities JSONB,

    token_count INTEGER,

    segment_count INTEGER,

    status VARCHAR(20) NOT NULL
        CHECK (status IN ('SCORED', 'FAILED')),

    error TEXT,

    scored_at TIMESTAMPTZ,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE (chunk_id, model_name, model_revision)
);


CREATE INDEX IF NOT EXISTS idx_filing_chunk_sentiment_filing
ON filing_chunk_sentiment (
    accession_number,
    model_name,
    model_revision,
    status
);


CREATE INDEX IF NOT EXISTS idx_filing_chunk_sentiment_status
ON filing_chunk_sentiment (
    model_name,
    model_revision,
    status
);
