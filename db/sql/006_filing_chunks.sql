/*
============================================================
AlphaLens - SEC Filing Chunks
============================================================

Purpose:
    Stores smaller RAG-ready pieces of SEC filing sections.

Current hierarchy:

    filings
        ↓
    filing_sections
        ↓
    filing_chunks


Example:

    AAPL 10-K
        ↓
    Item 1A - Risk Factors
        ↓
    chunk 0
    chunk 1
    chunk 2
    ...


Why chunks?
-----------

A single SEC section can contain tens of thousands of tokens.

For vector search, smaller chunks give much more precise
retrieval.

For example:

    User:
        "What cybersecurity risks does Apple mention?"

Instead of retrieving the entire Risk Factors section,
FAISS will eventually retrieve only the most relevant chunks.


Important:
    chunk_id will later be useful as the identifier connecting:

        FAISS vector
            ↕
        PostgreSQL filing_chunks row
============================================================
*/


CREATE TABLE IF NOT EXISTS filing_chunks (

    /*
    Unique AlphaLens identifier for this chunk.

    Later we can use this same ID when inserting the
    embedding vector into FAISS.

    Example:

        FAISS returns vector ID 18291

                    ↓

        SELECT *
        FROM filing_chunks
        WHERE chunk_id = 18291;
    */
    chunk_id BIGSERIAL PRIMARY KEY,


    /*
    Section this chunk belongs to.

    Example:

        filing_sections.section_id = 123

    One section can contain many chunks.
    */
    section_id BIGINT NOT NULL
        REFERENCES filing_sections(section_id)
        ON DELETE CASCADE,


    /*
    SEC filing identifier.

    Technically we could obtain this through section_id,
    but storing it here makes later RAG metadata lookup
    easier and faster.
    */
    accession_number VARCHAR(30) NOT NULL,


    /*
    Company ticker.

    Examples:

        AAPL
        MSFT
        NVDA

    This is deliberately duplicated from the filings table.

    Why?

    Later we may want to filter RAG searches by:

        ticker = 'AAPL'

    without performing several joins first.
    */
    ticker VARCHAR(20) NOT NULL,


    /*
    SEC filing type.

    Currently:

        10-K
        10-Q
    */
    form_type VARCHAR(10) NOT NULL,


    /*
    Date this filing was submitted to SEC.
    */
    filing_date DATE NOT NULL,


    /*
    Machine-friendly section identifier.

    Example:

        item_1a_risk_factors
        item_7_mda
    */
    section_key VARCHAR(100) NOT NULL,


    /*
    Human-readable section name.

    Example:

        Risk Factors
    */
    section_title VARCHAR(255) NOT NULL,


    /*
    Position of this chunk inside its section.

    Starts at 0.

    Example:

        Risk Factors

            chunk_index 0
            chunk_index 1
            chunk_index 2
            ...
    */
    chunk_index INTEGER NOT NULL,


    /*
    Actual text that will eventually be embedded.
    */
    content TEXT NOT NULL,


    /*
    Number of embedding-model tokens in this chunk.

    Useful for:

        debugging
        cost estimation
        context-window management
    */
    token_count INTEGER NOT NULL,


    /*
    Number of normal text characters.

    Mostly useful for diagnostics.
    */
    char_count INTEGER NOT NULL,


    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),


    /*
    One section cannot have two chunk 0s,
    two chunk 1s, etc.
    */
    UNIQUE (
        section_id,
        chunk_index
    )
);


-- ==========================================================
-- Index: Section
-- ==========================================================

CREATE INDEX IF NOT EXISTS idx_filing_chunks_section
ON filing_chunks (section_id);


-- ==========================================================
-- Index: Ticker
-- ==========================================================
--
-- Later:
--
--     retrieve only NVDA chunks
--
CREATE INDEX IF NOT EXISTS idx_filing_chunks_ticker
ON filing_chunks (ticker);


-- ==========================================================
-- Index: Section Type
-- ==========================================================
--
-- Later:
--
--     retrieve only Risk Factors chunks
--
CREATE INDEX IF NOT EXISTS idx_filing_chunks_section_key
ON filing_chunks (section_key);


-- ==========================================================
-- Combined RAG Metadata Index
-- ==========================================================
--
-- Useful for filtering:
--
-- ticker = AAPL
-- form_type = 10-K
-- filing_date ...
--
CREATE INDEX IF NOT EXISTS idx_filing_chunks_metadata
ON filing_chunks (
    ticker,
    form_type,
    filing_date
);