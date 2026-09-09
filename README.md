# AlphaLens

AlphaLens is a financial data pipeline that collects:

- Daily OHLCV market data
- SEC `10-K` and `10-Q` filing metadata
- Five-year SEC filing documents for later analysis
- Quarterly earnings call transcripts and speaker turns

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
10. Run the earnings transcript pipeline.

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
EARNINGSCALLS_API_KEY=your-earningscalls-dev-api-key
TRANSCRIPT_LOOKBACK_YEARS=5
EARNINGSCALLS_REQUEST_SECONDS=3.1
OPENAI_API_KEY=your-openai-api-key
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
Get-Content db\sql\008_earnings_transcripts.sql | docker exec -i alphalens-postgres psql -U alphalens -d alphalens
Get-Content db\sql\009_research_runs.sql | docker exec -i alphalens-postgres psql -U alphalens -d alphalens
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

## 10. Run Earnings Transcripts

AlphaLens uses the official EarningsCalls.dev API for quarterly earnings
calls. Set `EARNINGSCALLS_API_KEY` in `.env`, then run a small resumable
test:

```powershell
python -m pipelines.transcripts.run_pipeline --tickers AAPL --max-transcripts-per-ticker 1
python -m pipelines.transcripts.chunker
python -m pipelines.transcripts.embedder
```

Then run the full configured ticker backfill:

```powershell
python -m pipelines.transcripts.run_pipeline
python -m pipelines.transcripts.chunker
python -m pipelines.transcripts.embedder
```

The pipeline stores one row per ticker/fiscal quarter in
`earnings_transcripts`, normalized speaker turns in
`earnings_transcript_turns`, RAG-ready chunks in
`earnings_transcript_chunks`, and transcript vectors in
`data/faiss/transcript_chunks.faiss`.

### Verify Earnings Transcripts

```sql
SELECT
    ticker,
    COUNT(*) AS transcripts,
    MIN(fiscal_period) AS oldest,
    MAX(fiscal_period) AS newest
FROM earnings_transcripts
GROUP BY ticker
ORDER BY ticker;
```

Check speaker turns and chunks:

```sql
SELECT COUNT(*) AS turns
FROM earnings_transcript_turns;

SELECT
    MIN(token_count) AS smallest,
    ROUND(AVG(token_count), 2) AS average,
    MAX(token_count) AS largest
FROM earnings_transcript_chunks;
```

Check transcript embedding status:

```sql
SELECT
    embedding_status,
    COUNT(*)
FROM earnings_transcript_chunks
GROUP BY embedding_status
ORDER BY embedding_status;
```

### Ask RAG Questions Over Transcripts

The research API can search SEC filings, earnings transcripts, or both.
Use `source_type` to control retrieval:

```json
{
  "question": "What did Walmart management say about margins on the earnings call?",
  "ticker": "WMT",
  "source_type": "transcripts",
  "top_k": 5
}
```

Supported `source_type` values are `auto`, `filings`, `transcripts`, and
`both`. In `auto` mode, transcript-style questions search earnings calls,
SEC-style questions search filings, and broad questions search both.

When a transcript question implies a single call, such as "the earnings
call" or "latest call", and no `fiscal_period` is supplied, AlphaLens
searches the latest stored transcript for the requested ticker.

Run the local research UI:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Then open:

```text
http://127.0.0.1:8000
```

The UI loads its dropdown choices from metadata endpoints:

```text
GET /api/metadata/tickers
GET /api/metadata/transcript-periods?ticker=WMT
GET /api/metadata/filing-types?ticker=NVDA
GET /api/metadata/filing-sections?ticker=NVDA&form_type=10-K
```

These routes read the local database and do not call OpenAI.

Check whether the configured API key can authenticate with OpenAI:

```text
GET /api/health/openai
```

The check lists available models, so it does not generate an answer or use
generation tokens. The response distinguishes a missing key, rejected key,
rate limit, timeout, and network connection failure. The research console
also shows this status beside the API Docs link; click it to rerun the check.

Preview retrieval without generating an answer:

```json
{
  "question": "What did Walmart management say about margins on the earnings call?",
  "ticker": "WMT",
  "source_type": "transcripts",
  "top_k": 3
}
```

Send that payload to:

```text
POST /api/retrieval/preview
```

This returns `sources` and `market_context` but no generated answer. It is
useful for checking whether retrieval found the right evidence before using
the full `POST /api/research` answer-generation endpoint.

### Market Data In RAG

Market prices are used as structured SQL context, not vector embeddings.
When a question asks about stock performance, returns, volatility, volume,
or SPY-relative performance, AlphaLens calculates a `market_context`
snapshot from `market_prices` and includes it in the answer prompt.
The snapshot includes both the company's returns and SPY's absolute returns,
as well as the percentage-point spread, so comparisons do not require the
model to derive a missing benchmark value.

Example:

```json
{
  "question": "How has NVIDIA stock performed over the last year versus SPY?",
  "ticker": "NVDA",
  "source_type": "auto",
  "top_k": 3
}
```

For pure market-performance questions in `auto` mode, AlphaLens answers from
the structured market snapshot without pulling unrelated filing/transcript
chunks into the prompt.

The research console also loads adjusted-close history for the selected
ticker and SPY. Both series are indexed to 100 at the start of the selected
1M, 3M, 1Y, or 5Y period; the chart presents that index as cumulative return
from 0% so the left axis cannot be mistaken for a share price. The same API
response includes dated earnings-call and SEC filing events. Event reactions
compare adjusted close on the event date (or the preceding trading session)
with the next and fifth subsequent trading sessions.

```text
GET /api/market/prices?ticker=NVDA&period=1Y
```

### Saved Research History

Successful `POST /api/research` answers are saved automatically with their
filters, market snapshots, and exact source records. Retrieval previews are
not saved. Opening history returns the stored snapshot and does not rerun
retrieval or call OpenAI.

```text
GET /api/research/history?limit=20
GET /api/research/history/1
DELETE /api/research/history/1
```

The research console lists recent runs below the query controls. A saved run
can be reopened with its original question and filters, or deleted locally.

### Retrieval Evaluations

Run the versioned retrieval suite without generating answers:

```powershell
python -m evals.run_retrieval
```

The suite checks ticker detection, source routing, source counts and types,
fiscal-period and filing filters, market context, and required evidence
terms. Text retrieval cases create one query embedding per searched corpus,
but the answer-generation model is never called. Results are written to
`data/evals/retrieval_report.json`; the command exits with code 1 if any case
fails, making it suitable for CI.

Run a smaller selection while developing:

```powershell
python -m evals.run_retrieval --case-id wmt_latest_margin_call
python -m evals.run_retrieval --limit 3 --fail-under 0.8
```

Add or revise human-approved cases in `evals/retrieval_cases.json` as the
corpus and expected behavior evolve.

### Response-Quality Evaluations

Run the smaller end-to-end suite after retrieval passes:

```powershell
python -m evals.run_responses
```

Each case retrieves evidence once, generates an answer, checks citation labels
and expected behavior deterministically, then uses a strict structured model
judge to score groundedness, relevance, completeness, and citation quality from
1 to 5. Evaluation answers call the generator directly and are not added to
saved research history. The report is written to
`data/evals/response_report.json`, and the command exits with code 1 when the
configured pass-rate gate is missed.

Run one case while developing, or tune the release gate explicitly:

```powershell
python -m evals.run_responses --case-id wmt_latest_margin_answer
python -m evals.run_responses --fail-under 0.8 --min-judge-score 4
```

The five-case suite runs sequentially and normally uses about 14 OpenAI API
requests: retrieval embeddings, answer generations, and one judge call per
case. `EVAL_JUDGE_MODEL` can select a judge independently from `RAG_MODEL`.
Keep the deterministic checks as the hard guardrails and periodically review
model-judge failures with a human before changing rubrics or thresholds.

Open the internal evaluation monitor while the API is running:

```text
http://127.0.0.1:8000/evals
```

The dashboard reads local JSON reports only; refreshing it does not call
OpenAI. It shows latest suite pass rates, response-judge dimensions, individual
case failures, judge explanations, and archived pass-rate trends. Full quality
gate runs preserve timestamped copies under `data/evals/history/`.

### Continuous-Integration Quality Gate

`.github/workflows/ci.yml` runs unit tests on a GitHub-hosted runner for every
push and pull request. The corpus-backed retrieval and response suites run on
a Windows self-hosted runner because the PostgreSQL corpus and FAISS indexes
are intentionally not committed to Git.

Configure the repository before running the corpus-backed job:

1. In GitHub, open **Settings > Actions > Runners**, add a Windows self-hosted
   runner, ensure its runner version is at least `2.327.1`, and assign it the
   custom label `alphalens-evals`.
2. Add Actions secrets `DATABASE_URL` and `OPENAI_API_KEY`.
3. Add the Actions variable `ALPHALENS_FAISS_DIRECTORY` containing the absolute
   path to this machine's populated `data\faiss` directory.
4. Add `ALPHALENS_EVAL_REPORT_DIRECTORY` with the absolute path to the main
   workspace's persistent `data\evals` directory.
5. Add `ALPHALENS_PYTHON` with the absolute path to the tested local interpreter,
   such as `C:\Users\Lixing\Desktop\alphalens\.venv\Scripts\python.exe`.
6. Optionally add `RAG_MODEL` and `EVAL_JUDGE_MODEL` repository variables.
7. Create an Actions environment named `corpus-evals`. For a private side
   project, required reviewers and deployment protection rules are optional.
8. Optionally protect `main` and require `Unit tests` and
   `Corpus quality gate` before merging.

The self-hosted runner account must be able to reach PostgreSQL through the
configured `DATABASE_URL` and read the FAISS directory. Keep the runner private
to this repository and approve the protected environment only for trusted
changes because pull-request code executes on that machine.

Run the same gate locally:

```powershell
.\scripts\run_quality_gate.ps1
```

Preflight validates secrets, database row counts, index files, metadata, and
embedding dimensions before any OpenAI request is made. GitHub uploads both
JSON evaluation reports for 30 days even when the quality threshold fails.

On the Windows self-hosted runner, the workflow uses `ALPHALENS_PYTHON`
directly instead of asking `actions/setup-python` to install another Python.
This avoids requiring an administrator runner process, 7-Zip, and a separate
GitHub tool cache for the corpus job.

## 11. Stop PostgreSQL

Stop PostgreSQL without deleting its data volume:

```powershell
docker compose down
```
