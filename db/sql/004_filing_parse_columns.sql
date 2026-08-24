/*
============================================================
AlphaLens - SEC Filing Text Parsing Tracking
============================================================

Purpose:
    Track conversion of downloaded SEC HTML filings into
    clean plain-text documents.

Pipeline:

    raw_file_path
          ↓
    HTML parser
          ↓
    clean_text_path

Parsing status:

    PENDING
        Downloaded but not parsed yet.

    PARSED
        HTML was successfully converted to text.

    FAILED
        Parsing failed.
============================================================
*/


-- Path to the cleaned .txt version of the filing.
ALTER TABLE filings
ADD COLUMN IF NOT EXISTS clean_text_path TEXT;


-- Parsing status.
ALTER TABLE filings
ADD COLUMN IF NOT EXISTS parse_status VARCHAR(20)
NOT NULL DEFAULT 'PENDING';


-- When parsing successfully completed.
ALTER TABLE filings
ADD COLUMN IF NOT EXISTS parsed_at TIMESTAMPTZ;


-- Store parsing error if something goes wrong.
ALTER TABLE filings
ADD COLUMN IF NOT EXISTS parse_error TEXT;


-- Useful later for finding documents still needing parsing.
CREATE INDEX IF NOT EXISTS idx_filings_parse_status
ON filings (parse_status);