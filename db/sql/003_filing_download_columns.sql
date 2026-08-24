/*
============================================================
AlphaLens - SEC Filing Download Tracking
============================================================

Purpose:
    Adds fields used to track the downloading of actual
    10-K / 10-Q HTML documents.

The filings table currently contains metadata.

After this migration it can also record:

    - where the filing came from
    - where we saved it locally
    - whether the download succeeded
    - when it was downloaded
    - any error that occurred
============================================================
*/


-- Full SEC URL of the primary filing document.
ALTER TABLE filings
ADD COLUMN IF NOT EXISTS source_url TEXT;


-- Location of the downloaded raw HTML file.
--
-- Example:
--
-- data/sec/raw/AAPL/000032019325000079/aapl-20250927.htm
--
ALTER TABLE filings
ADD COLUMN IF NOT EXISTS raw_file_path TEXT;


-- Current download state.
--
-- PENDING
-- DOWNLOADED
-- FAILED
--
ALTER TABLE filings
ADD COLUMN IF NOT EXISTS download_status VARCHAR(20)
NOT NULL DEFAULT 'PENDING';


-- When the document was successfully downloaded.
ALTER TABLE filings
ADD COLUMN IF NOT EXISTS downloaded_at TIMESTAMPTZ;


-- Stores an error message if download fails.
ALTER TABLE filings
ADD COLUMN IF NOT EXISTS download_error TEXT;


-- Useful later when finding documents that still
-- need to be downloaded.
CREATE INDEX IF NOT EXISTS idx_filings_download_status
ON filings (download_status);