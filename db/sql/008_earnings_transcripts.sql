/*
============================================================
AlphaLens - Earnings Call Transcripts
============================================================

Purpose:
    Store quarterly earnings call transcripts and speaker turns.

Provider:
    The first implementation uses Alpha Vantage's
    EARNINGS_CALL_TRANSCRIPT endpoint.

Relationship:

    companies
        1
        |
        v
    earnings_transcripts
        1
        |
        v
    earnings_transcript_turns
        1
        |
        v
    earnings_transcript_chunks
============================================================
*/


CREATE TABLE IF NOT EXISTS earnings_transcripts (

    -- Internal AlphaLens identifier for one transcript.
    transcript_id BIGSERIAL PRIMARY KEY,

    -- Stock ticker, e.g. AAPL, MSFT, NVDA.
    --
    -- This links transcript data to the same companies table used by
    -- market prices and SEC filings.
    ticker VARCHAR(20) NOT NULL
        REFERENCES companies(ticker),

    -- Fiscal year reported by the earnings call.
    --
    -- Stored separately from fiscal_period so SQL can filter/sort without
    -- string parsing.
    fiscal_year INTEGER NOT NULL,

    -- Fiscal quarter reported by the earnings call.
    fiscal_quarter INTEGER NOT NULL
        CHECK (fiscal_quarter BETWEEN 1 AND 4),

    -- Compact label such as 2026Q3.
    fiscal_period VARCHAR(6) NOT NULL,

    -- Date of the earnings call if the provider supplies it.
    call_date DATE,

    -- Provider title/event label if available.
    title TEXT,

    -- Lets the schema support another provider later, such as FMP,
    -- without mixing records from different sources.
    source_provider VARCHAR(50) NOT NULL DEFAULT 'alpha_vantage',

    -- Redacted provider URL used for traceability.
    --
    -- The actual API key should never be stored here.
    source_url TEXT,

    -- Full transcript text.
    --
    -- Speaker turns are also stored in a child table, but keeping the full
    -- document here is useful for fallback chunking and diagnostics.
    content TEXT NOT NULL,

    -- Character count for validation and diagnostics.
    char_count INTEGER NOT NULL,

    -- Number of normalized speaker turns extracted from the provider
    -- payload.
    turn_count INTEGER NOT NULL DEFAULT 0,

    -- Original provider JSON.
    --
    -- This makes it possible to debug provider changes without re-calling
    -- the API.
    raw_payload JSONB,

    -- Current ingestion state.
    --
    -- Initial states:
    --     INGESTED
    --     FAILED
    ingest_status VARCHAR(20) NOT NULL DEFAULT 'INGESTED',

    -- Error message if ingestion fails in a later enhancement.
    ingest_error TEXT,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Idempotency key:
    --
    -- One provider should only have one transcript per ticker/fiscal
    -- quarter. Rerunning the loader updates this row instead of inserting
    -- duplicates.
    UNIQUE (
        ticker,
        fiscal_year,
        fiscal_quarter,
        source_provider
    )
);


CREATE INDEX IF NOT EXISTS idx_earnings_transcripts_ticker_period
ON earnings_transcripts (
    ticker,
    fiscal_year DESC,
    fiscal_quarter DESC
);


CREATE INDEX IF NOT EXISTS idx_earnings_transcripts_call_date
ON earnings_transcripts (call_date);


CREATE TABLE IF NOT EXISTS earnings_transcript_turns (

    -- Internal AlphaLens identifier for one speaker turn.
    turn_id BIGSERIAL PRIMARY KEY,

    -- Parent transcript.
    transcript_id BIGINT NOT NULL
        REFERENCES earnings_transcripts(transcript_id)
        ON DELETE CASCADE,

    -- Position of this turn inside the call. Starts at 0.
    turn_index INTEGER NOT NULL,

    -- Speaker name, e.g. Tim Cook, Luca Maestri, Operator.
    speaker_name TEXT,

    -- Speaker title when supplied by the provider.
    speaker_title TEXT,

    -- Provider role/session metadata if available.
    --
    -- Examples might include management, analyst, operator, Q&A, etc.
    speaker_role TEXT,

    -- The actual spoken content for this turn.
    content TEXT NOT NULL,

    -- Provider sentiment label if available.
    sentiment_label TEXT,

    -- Provider numeric sentiment score if available.
    sentiment_score DOUBLE PRECISION,

    -- Character count for diagnostics.
    char_count INTEGER NOT NULL,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Reruns should replace/update the same turn position rather than
    -- creating duplicates.
    UNIQUE (
        transcript_id,
        turn_index
    )
);


CREATE INDEX IF NOT EXISTS idx_earnings_transcript_turns_transcript
ON earnings_transcript_turns (transcript_id);


CREATE TABLE IF NOT EXISTS earnings_transcript_chunks (

    -- Unique AlphaLens identifier for a transcript chunk.
    --
    -- This will later be useful as the vector ID if transcript embeddings
    -- are added.
    chunk_id BIGSERIAL PRIMARY KEY,

    -- Parent transcript.
    transcript_id BIGINT NOT NULL
        REFERENCES earnings_transcripts(transcript_id)
        ON DELETE CASCADE,

    -- Denormalized metadata for faster later retrieval/filtering.
    ticker VARCHAR(20) NOT NULL,

    fiscal_year INTEGER NOT NULL,

    fiscal_quarter INTEGER NOT NULL,

    fiscal_period VARCHAR(6) NOT NULL,

    source_provider VARCHAR(50) NOT NULL,

    -- Position of the chunk inside the transcript. Starts at 0.
    chunk_index INTEGER NOT NULL,

    -- Speakers represented in this chunk.
    speaker_names TEXT[],

    -- Text sent to a future embedding model.
    content TEXT NOT NULL,

    -- Token count estimated with tiktoken.
    token_count INTEGER NOT NULL,

    -- Character count for diagnostics.
    char_count INTEGER NOT NULL,

    -- Embedding state.
    --
    -- This is tracking only. The current transcript pipeline does not
    -- create embeddings yet.
    embedding_status VARCHAR(20) NOT NULL DEFAULT 'PENDING',

    -- Embedding model name once vectors are generated in a future step.
    embedding_model VARCHAR(100),

    -- Timestamp of future successful embedding.
    embedded_at TIMESTAMPTZ,

    -- Error from a future failed embedding attempt.
    embedding_error TEXT,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Each transcript should only have one chunk 0, one chunk 1, etc.
    UNIQUE (
        transcript_id,
        chunk_index
    )
);


CREATE INDEX IF NOT EXISTS idx_earnings_transcript_chunks_transcript
ON earnings_transcript_chunks (transcript_id);


CREATE INDEX IF NOT EXISTS idx_earnings_transcript_chunks_ticker_period
ON earnings_transcript_chunks (
    ticker,
    fiscal_year DESC,
    fiscal_quarter DESC
);


CREATE INDEX IF NOT EXISTS idx_earnings_transcript_chunks_embedding_status
ON earnings_transcript_chunks (embedding_status);
