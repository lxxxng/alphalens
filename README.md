# AlphaLens

AlphaLens is a financial data pipeline that collects:

- Daily OHLCV market data
- SEC `10-K` and `10-Q` filing metadata
- Five-year SEC filing documents for later analysis

For complete new-device setup, pipeline commands, verification, and recovery
notes, see [RUNBOOK.md](RUNBOOK.md).

## Quick Start

Run these steps in order:

1. Create the Python environment.
2. Start PostgreSQL with Docker.
3. Apply the database migrations.
4. Run the market-data pipeline.
5. Run the SEC metadata pipeline.
6. Download the five-year SEC filing document backfill.
7. Parse downloaded SEC documents into clean text.
8. Extract filing sections into PostgreSQL.
9. Chunk SEC sections for later embeddings.

## 1. Create the Environment

```powershell
python -m venv .venv
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Review `.env` and set the required database and SEC values:

```text
DATABASE_URL=postgresql+psycopg2://alphalens:password@localhost:5432/alphalens
SEC_USER_AGENT=AlphaLens your-email@example.com
```

## 2. Start PostgreSQL

```powershell
docker compose up -d
docker compose ps
```

## 3. Apply Database Migrations

Run each migration in order:

```powershell
Get-Content db\sql\001_initial_schema.sql | docker exec -i alphalens-postgres psql -U alphalens -d alphalens
Get-Content db\sql\002_sec_filings.sql | docker exec -i alphalens-postgres psql -U alphalens -d alphalens
Get-Content db\sql\003_filing_download_columns.sql | docker exec -i alphalens-postgres psql -U alphalens -d alphalens
Get-Content db\sql\004_filing_parse_columns.sql | docker exec -i alphalens-postgres psql -U alphalens -d alphalens
Get-Content db\sql\005_filing_sections.sql | docker exec -i alphalens-postgres psql -U alphalens -d alphalens
Get-Content db\sql\006_filing_chunks.sql | docker exec -i alphalens-postgres psql -U alphalens -d alphalens
Get-Content db\sql\007_chunk_embeddings.sql | docker exec -i alphalens-postgres psql -U alphalens -d alphalens
```

Verify the tables:

```powershell
docker exec -it alphalens-postgres psql -U alphalens -d alphalens
```

```sql
\dt
\q
```

## 4. Run Market Data

This runs extraction, transformation, and PostgreSQL loading for all configured tickers:

```powershell
python -m pipelines.market_data.run_pipeline
```

### Verify Market Data

Check the total row count. You should see around 14,000 rows, depending on the trading date range:

```powershell
docker exec -it alphalens-postgres psql -U alphalens -d alphalens
```

```sql
SELECT COUNT(*)
FROM market_prices;
```

Check Apple:

```sql
SELECT *
FROM market_prices
WHERE ticker = 'AAPL'
ORDER BY trading_date DESC
LIMIT 10;
```

Check every ticker:

```sql
SELECT
    ticker,
    COUNT(*) AS row_count,
    MIN(trading_date) AS first_date,
    MAX(trading_date) AS latest_date
FROM market_prices
GROUP BY ticker
ORDER BY ticker;
```

You should see all 21 configured symbols.

### Verify Market-Data Idempotency

Exit `psql`, run the pipeline again, and check the count again:

```sql
\q
```

```powershell
python -m pipelines.market_data.run_pipeline
docker exec -it alphalens-postgres psql -U alphalens -d alphalens -c "SELECT COUNT(*) FROM market_prices;"
```

The row count should not double. The same ticker and trading date are updated by `ON CONFLICT` instead of inserted as duplicates.

## 5. Run the SEC Pipeline

Run the SEC extractor by itself to download and display filing metadata:

```powershell
python -m pipelines.sec.extractor
```

Run the complete SEC metadata pipeline. It extracts `10-K` and `10-Q` metadata, then loads the `companies` and `filings` tables:

```powershell
python -m pipelines.sec.run_pipeline
```

Download all stored `10-K` and `10-Q` HTML documents. Because the SEC metadata extractor keeps approximately five years of filings, this performs the matching five-year document backfill:

```powershell
python -m pipelines.sec.downloader
```

## 6. Verify SEC Downloads

Enter PostgreSQL:

```powershell
docker exec -it alphalens-postgres psql -U alphalens -d alphalens
```

List downloaded filings:

```sql
SELECT
    ticker,
    form_type,
    filing_date,
    download_status,
    raw_file_path
FROM filings
WHERE download_status = 'DOWNLOADED'
ORDER BY ticker, form_type;
```

Check download counts by ticker:

```sql
SELECT
    ticker,
    COUNT(*) AS filings,
    MIN(filing_date) AS oldest,
    MAX(filing_date) AS newest
FROM filings
WHERE download_status = 'DOWNLOADED'
GROUP BY ticker
ORDER BY ticker;
```

Most tickers should show 20 filings. A ticker can show slightly fewer, such as NVDA with 19, depending on its filing dates inside the rolling five-year window.

You should see rows similar to:

```text
ticker | form_type | filing_date | download_status | raw_file_path
-------+-----------+-------------+-----------------+---------------------
AAPL   | 10-K      | ...         | DOWNLOADED      | data/sec/raw/...
AAPL   | 10-Q      | ...         | DOWNLOADED      | data/sec/raw/...
AMZN   | 10-K      | ...         | DOWNLOADED      | data/sec/raw/...
...
```

Check download counts:

```sql
SELECT
    download_status,
    COUNT(*)
FROM filings
GROUP BY download_status;
```

You might see:

```text
download_status | count
----------------+-------
DOWNLOADED      | 399
```

The exact count can vary slightly by filing calendar and ticker, but most companies should have about 20 documents for a five-year window.

## 7. Parse SEC Filings

Convert downloaded SEC HTML files into clean `.txt` files:

```powershell
python -m pipelines.sec.parser
```

The parser reads files from `data/sec/raw/`, writes cleaned text to `data/sec/clean/`, and updates each filing’s `parse_status` in PostgreSQL.

### Verify Parsed Filings

```powershell
docker exec -it alphalens-postgres psql -U alphalens -d alphalens
```

Check parsing results:

```sql
SELECT
    ticker,
    form_type,
    parse_status,
    clean_text_path,
    parsed_at,
    parse_error
FROM filings
WHERE download_status = 'DOWNLOADED'
ORDER BY ticker, form_type;
```

Successful rows should have `parse_status = 'PARSED'` and a path under `data/sec/clean/`. Rows with `parse_status = 'FAILED'` include details in `parse_error`.

## 8. Extract SEC Sections

Extract useful sections such as Risk Factors, MD&A, and Financial Statements from the parsed filing text:

```powershell
python -m pipelines.sec.section_extractor
```

The section extractor reads parsed files from `data/sec/clean/` and stores extracted text in PostgreSQL table `filing_sections`.

### Verify SEC Sections

```sql
SELECT COUNT(*) AS total_sections
FROM filing_sections;
```

Check section counts by ticker:

```sql
SELECT
    f.ticker,
    COUNT(*) AS sections
FROM filing_sections fs
JOIN filings f
    ON f.accession_number = fs.accession_number
GROUP BY f.ticker
ORDER BY f.ticker;
```

## 9. Chunk SEC Sections

Split extracted SEC sections into smaller RAG-ready chunks:

```powershell
python -m pipelines.sec.chunker
```

The chunker reads from `filing_sections`, writes to `filing_chunks`, and can be run again safely. Existing chunks for each section are replaced with the current chunking output.

### Verify SEC Chunks

Check that every extracted filing section has at least one chunk:

```sql
SELECT
    COUNT(*) AS sections_without_chunks
FROM filing_sections fs
LEFT JOIN filing_chunks fc
    ON fs.section_id = fc.section_id
WHERE fc.chunk_id IS NULL;
```

Expected result:

```text
sections_without_chunks
-----------------------
0
```

Check the chunk size summary:

```sql
SELECT
    MIN(token_count) AS smallest,
    ROUND(AVG(token_count), 2) AS average,
    MAX(token_count) AS largest
FROM filing_chunks;
```

Example result:

```text
smallest | average | largest
---------+---------+--------
11       | 675.26  | 700
```

Check the chunk size distribution:

```sql
SELECT
    CASE
        WHEN token_count < 100
            THEN '<100'
        WHEN token_count < 300
            THEN '100-299'
        WHEN token_count < 500
            THEN '300-499'
        WHEN token_count < 650
            THEN '500-649'
        ELSE
            '650-700'
    END AS token_range,
    COUNT(*) AS chunks
FROM filing_chunks
GROUP BY token_range
ORDER BY MIN(token_count);
```

Example result:

```text
token_range | chunks
------------+-------
<100        | 453
100-299     | 621
300-499     | 485
500-649     | 249
650-700     | 30266
```

## 10. Stop PostgreSQL

Stop PostgreSQL without deleting its data volume:

```powershell
docker compose down
```
