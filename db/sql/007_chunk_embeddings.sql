/*
============================================================
AlphaLens - Chunk Embedding Tracking
============================================================

Purpose:
    Track whether each SEC chunk has been converted into
    an embedding and stored in the FAISS index.

Important:
    The embedding vector itself is NOT stored in PostgreSQL.

PostgreSQL:
    stores text + metadata + chunk_id

FAISS:
    stores vectors + chunk_id


Relationship:

    PostgreSQL

        chunk_id = 18423
        ticker = AAPL
        content = "..."

                ↕

    FAISS

        vector ID = 18423
        vector = [0.023, -0.018, ...]


Later, when FAISS says:

    "vector 18423 is similar to the question"

we can retrieve:

    SELECT *
    FROM filing_chunks
    WHERE chunk_id = 18423;
============================================================
*/


-- Current embedding state.
--
-- PENDING
-- EMBEDDED
-- FAILED
--
ALTER TABLE filing_chunks
ADD COLUMN IF NOT EXISTS embedding_status VARCHAR(20)
NOT NULL DEFAULT 'PENDING';


-- Which embedding model generated this vector.
ALTER TABLE filing_chunks
ADD COLUMN IF NOT EXISTS embedding_model VARCHAR(100);


-- When embedding successfully completed.
ALTER TABLE filing_chunks
ADD COLUMN IF NOT EXISTS embedded_at TIMESTAMPTZ;


-- Error from the most recent failed embedding attempt.
ALTER TABLE filing_chunks
ADD COLUMN IF NOT EXISTS embedding_error TEXT;


CREATE INDEX IF NOT EXISTS idx_filing_chunks_embedding_status
ON filing_chunks (embedding_status);