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
Get-Content db\sql\010_transcript_sentiment.sql | docker exec -i alphalens-postgres psql -U alphalens -d alphalens
Get-Content db\sql\011_filing_sentiment.sql | docker exec -i alphalens-postgres psql -U alphalens -d alphalens
Get-Content db\sql\012_event_briefs.sql | docker exec -i alphalens-postgres psql -U alphalens -d alphalens
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

### Score Transcript Sentiment

The sentiment worker uses the public `ProsusAI/finbert` financial-text model.
Its heavier runtime is isolated from the ordinary API dependencies:

```powershell
pip install -r requirements-ml.txt

# Download the pinned model and score a small, resumable sample.
python -m pipelines.transcripts.sentiment --tickers WMT --limit 10

# Continue through every pending turn in the corpus.
python -m pipelines.transcripts.sentiment
```

Operator-only turns are recorded as `SKIPPED` by default. Every scored turn
stores its label, polarity score, confidence, three-class probabilities,
token/segment counts, and model version in
`earnings_transcript_turn_sentiment`. The latest label and polarity score are
also copied onto `earnings_transcript_turns` for lightweight API reads.
Rerunning the command processes only missing turns. Use `--retry-failed` to
retry model failures or `--include-operators` to score operator speech.

```sql
SELECT
    status,
    sentiment_label,
    COUNT(*)
FROM earnings_transcript_turn_sentiment
GROUP BY status, sentiment_label
ORDER BY status, sentiment_label;
```

SEC sentiment uses the same pinned model but targets only MD&A, Risk Factors,
and Market Risk chunks. Financial-statement tables are excluded because their
language polarity is not a reliable analytical signal.

```powershell
# Small SEC integration run, followed by the resumable full backfill.
python -m pipelines.sec.sentiment --tickers WMT --limit 10
python -m pipelines.sec.sentiment
```

Coverage-aware read APIs expose transcript speaker-group trends and filing
section summaries even while a backfill is still running:

```text
GET /api/sentiment/transcripts?ticker=WMT
GET /api/sentiment/transcripts/287
GET /api/sentiment/filings?ticker=WMT&form_type=10-K
GET /api/sentiment/filings/0000104169-21-000058
GET /api/sentiment/topics
```

Sentiment responses also include deterministic multi-label topic summaries
for margins, guidance, growth, demand, costs, pricing, supply chain, capital
allocation, risk, and technology. Transcript topic scores use management turns
only and timeline responses include the change from the previous call. SEC
topic scores use the same taxonomy across the selected narrative sections.
The API returns `topic_classifier_version` and exact matched terms so this
baseline remains reproducible and auditable before introducing a learned topic
classifier.

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

To compare companies, send `tickers` with up to four symbols. Retrieval keeps
evidence balanced across the selected companies, and the first ticker controls
the fiscal-period and filing metadata filters:

```json
{
  "question": "Compare Walmart and Costco margin commentary.",
  "tickers": ["WMT", "COST"],
  "source_type": "both",
  "top_k": 3
}
```

The singular `ticker` field remains supported for existing API clients.

Supported `source_type` values are `auto`, `filings`, `transcripts`, and
`both`. In `auto` mode, transcript-style questions search earnings calls,
SEC-style questions search filings, and broad questions search both.

When a transcript question implies a single call, such as "the earnings
call" or "latest call", and no `fiscal_period` is supplied, AlphaLens
searches the latest stored transcript for the requested ticker.

Both normal RAG answers and structured event briefs use LangChain Expression
Language (LCEL) with `ChatOpenAI`. The event workflow composes retrieval,
structured market context, versioned FinBERT topic signals, prompt assembly,
schema-constrained generation, and deterministic citation validation.

Generate an earnings, filing, or combined event brief:

```json
{
  "ticker": "WMT",
  "event_type": "earnings",
  "fiscal_period": "2026Q4",
  "top_k": 6
}
```

Comparative briefs accept up to four tickers and use the supplied focus to
steer retrieval and structured generation without blending company signals:

```json
{
  "tickers": ["WMT", "COST"],
  "event_type": "combined",
  "focus": "Compare margin quality and management outlook.",
  "top_k": 3
}
```

Without `fiscal_period`, earnings scope means the latest stored call for each
company. Filing scope means the latest filing matching `form_type`, or the
latest SEC filing when no form is selected.

Send the payload to:

```text
POST /api/briefs/generate
```

The response separates the executive summary, developments, topic signals,
market reaction, risks, watch items, and limitations while returning the exact
market, sentiment, and citation records used. Successful results are persisted
as immutable snapshots. An unchanged event scope reuses its newest saved brief;
send `"refresh": true` to generate and save a new snapshot intentionally.

Saved brief endpoints:

```text
GET    /api/briefs/history
GET    /api/briefs/history/{brief_id}
DELETE /api/briefs/history/{brief_id}
```

Run the local research UI:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Then open:

```text
http://127.0.0.1:8000
```

The research workspace keeps the price and event chart as its primary view.
The supporting signal monitor can switch between management-call and SEC
narrative sentiment, compare each score with the prior event, and show current
topic-level polarity and coverage. With multiple selected companies, ticker
tabs retain company-specific detail while Compare overlays sentiment histories
and displays topic scores side by side. A fixed Latest Event Brief band beneath
the chart generates the LCEL workflow from each selected company's latest
earnings call and SEC filing. Its executive summary stays visible while the
detailed sections and evidence remain expandable, keeping sentiment close by
even for multi-company comparisons. Recent Briefs restores the original
ticker scope, generated content, and evidence without calling OpenAI again.
On desktop, Pin Chart converts the market
chart into a compact sticky monitor while the brief and sentiment sections
scroll beneath it. The attached right-hand query rail is reserved for ad hoc
questions; its generated answers switch directly to Research without losing
the Market view. Saved runs can also be opened directly with `?run_id=<id>`.
A persisted header toggle switches every surface in the Research and
Evaluation workspaces between light and dark themes.

The UI loads its dropdown choices and automatic company matches from metadata
endpoints:

```text
GET /api/metadata/tickers
GET /api/metadata/resolve-tickers?question=Compare%20Walmart%20and%20Costco
GET /api/metadata/transcript-periods?ticker=WMT
GET /api/metadata/filing-types?ticker=NVDA
GET /api/metadata/filing-sections?ticker=NVDA&form_type=10-K
GET /api/transcripts/269
```

These routes read the local database and do not call OpenAI.

Earnings events and transcript citations open AlphaLens's local transcript
reader at `/transcripts/{transcript_id}`. The reader uses the transcript and
ordered speaker turns already stored in PostgreSQL, so the browser never
opens the provider's authenticated API URL and does not need an
`EARNINGSCALLS_API_KEY`. That key is required only when ingesting new calls.
Stored fiscal-period keys remain compact values such as `2026Q2` for stable
filtering, while the UI displays them as `FY2026 Q2` beside the calendar date.

Ticker Auto mode is enabled by default. As the question changes, matching
company names and ticker symbols become selected chips and update the market
comparison chart. Changing a chip manually disables Auto until it is enabled
again. A `tickers` URL parameter is also treated as an explicit manual choice.

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

The research console also loads adjusted-close history for up to four selected
companies and SPY. Every series is indexed to 100 at the start of the selected
1M, 3M, 1Y, or 5Y period; the chart presents that index as cumulative return
from 0% so the left axis cannot be mistaken for a share price. The same API
response includes dated earnings-call and SEC filing events for each company.
Click an E marker to open the local transcript reader, or an F marker to open
the public SEC filing. Event reactions compare adjusted close on the event
date (or the preceding trading session) with the next and fifth subsequent
trading sessions.

```text
GET /api/market/prices?ticker=NVDA&period=1Y
GET /api/market/prices?ticker=WMT&tickers=NVDA,COST&period=1Y
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
per-company evidence coverage, fiscal-period and filing filters, market
context, required evidence terms, and the four-company limit. Text retrieval
cases create one query embedding per company/corpus pair, but the
answer-generation model is never called. Results are written to
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
1 to 5. Multi-company cases also require evidence and cited sources from every
requested ticker, with the judge checking company attribution. Evaluation
answers call the generator directly and are not added to saved research history.
The report is written to
`data/evals/response_report.json`, and the command exits with code 1 when the
configured pass-rate gate is missed.

Run one case while developing, or tune the release gate explicitly:

```powershell
python -m evals.run_responses --case-id wmt_latest_margin_answer
python -m evals.run_responses --case-id wmt_cost_latest_margin_comparison_answer
python -m evals.run_responses --fail-under 0.8 --min-judge-score 4
```

The nine-case suite runs sequentially and normally uses about 29 OpenAI API
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
