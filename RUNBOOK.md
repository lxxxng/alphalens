# AlphaLens Runbook

Run these PowerShell commands from the project folder.

## 1. New device: clone the project

```powershell
git clone https://github.com/lxxxng/alphalens.git
Set-Location alphalens
```

## 2. Create the Python environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run this once:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

## 3. Create `.env`

```powershell
Copy-Item .env.example .env
```

Open `.env` and replace the placeholders:

```dotenv
POSTGRES_USER=alphalens
POSTGRES_PASSWORD=your-password
POSTGRES_DB=alphalens
DATABASE_URL=postgresql+psycopg2://alphalens:your-password@localhost:5432/alphalens

SEC_USER_AGENT=AlphaLens your-email@example.com
OPENAI_API_KEY=your-openai-api-key
EARNINGSCALLS_API_KEY=your-earningscalls-dev-api-key
TRANSCRIPT_LOOKBACK_YEARS=5
EARNINGSCALLS_REQUEST_SECONDS=3.1
```

Do not commit `.env`.

## 4. Start PostgreSQL

Start Docker Desktop, then run:

```powershell
docker compose up -d
docker compose ps
```

## 5. Apply all SQL files in order

```powershell
Get-Content -Raw db\sql\001_initial_schema.sql | docker exec -i alphalens-postgres psql -v ON_ERROR_STOP=1 -U alphalens -d alphalens
Get-Content -Raw db\sql\002_sec_filings.sql | docker exec -i alphalens-postgres psql -v ON_ERROR_STOP=1 -U alphalens -d alphalens
Get-Content -Raw db\sql\003_filing_download_columns.sql | docker exec -i alphalens-postgres psql -v ON_ERROR_STOP=1 -U alphalens -d alphalens
Get-Content -Raw db\sql\004_filing_parse_columns.sql | docker exec -i alphalens-postgres psql -v ON_ERROR_STOP=1 -U alphalens -d alphalens
Get-Content -Raw db\sql\005_filing_sections.sql | docker exec -i alphalens-postgres psql -v ON_ERROR_STOP=1 -U alphalens -d alphalens
Get-Content -Raw db\sql\006_filing_chunks.sql | docker exec -i alphalens-postgres psql -v ON_ERROR_STOP=1 -U alphalens -d alphalens
Get-Content -Raw db\sql\007_chunk_embeddings.sql | docker exec -i alphalens-postgres psql -v ON_ERROR_STOP=1 -U alphalens -d alphalens
Get-Content -Raw db\sql\008_earnings_transcripts.sql | docker exec -i alphalens-postgres psql -v ON_ERROR_STOP=1 -U alphalens -d alphalens
Get-Content -Raw db\sql\009_research_runs.sql | docker exec -i alphalens-postgres psql -v ON_ERROR_STOP=1 -U alphalens -d alphalens
```

## 6. Run all pipelines in order

```powershell
# Market prices
python -m pipelines.market_data.run_pipeline

# SEC filing metadata
python -m pipelines.sec.run_pipeline

# Download filing HTML
python -m pipelines.sec.downloader

# Convert HTML to clean text
python -m pipelines.sec.parser

# Extract filing sections
python -m pipelines.sec.section_extractor

# Split sections into chunks
python -m pipelines.sec.chunker

# Create OpenAI embeddings and the FAISS index
# Model: text-embedding-3-small
python -m pipelines.sec.embedder

# Earnings call transcripts
python -m pipelines.transcripts.run_pipeline
python -m pipelines.transcripts.chunker
python -m pipelines.transcripts.embedder

# Small/resumable transcript test
python -m pipelines.transcripts.run_pipeline --tickers AAPL --max-transcripts-per-ticker 1
```

The embedder can be rerun. It resumes from the existing FAISS index.

## 7. Quick verification

```powershell
# Main database counts
docker exec alphalens-postgres psql -U alphalens -d alphalens -P pager=off -c "SELECT 'market_prices' AS name, COUNT(*) FROM market_prices UNION ALL SELECT 'filings', COUNT(*) FROM filings UNION ALL SELECT 'filing_sections', COUNT(*) FROM filing_sections UNION ALL SELECT 'filing_chunks', COUNT(*) FROM filing_chunks;"

# Transcript counts
docker exec alphalens-postgres psql -U alphalens -d alphalens -P pager=off -c "SELECT 'earnings_transcripts' AS name, COUNT(*) FROM earnings_transcripts UNION ALL SELECT 'earnings_transcript_turns', COUNT(*) FROM earnings_transcript_turns UNION ALL SELECT 'earnings_transcript_chunks', COUNT(*) FROM earnings_transcript_chunks;"

# Transcript embedding status
docker exec alphalens-postgres psql -U alphalens -d alphalens -P pager=off -c "SELECT embedding_status, COUNT(*) FROM earnings_transcript_chunks GROUP BY embedding_status ORDER BY embedding_status;"

# Transcript RAG retrieval smoke test
.\.venv\Scripts\python.exe -c "from app.rag.retriever import semantic_search; results=semantic_search('What did Walmart management say about margins on the earnings call?', top_k=3, ticker='WMT', corpus='transcripts'); print([(r['source_type'], r['ticker'], r['fiscal_period'], r['chunk_id'], round(r['score'], 4)) for r in results])"

# Local research UI
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000

# UI metadata smoke tests
Invoke-RestMethod "http://127.0.0.1:8000/api/metadata/tickers"
Invoke-RestMethod "http://127.0.0.1:8000/api/metadata/transcript-periods?ticker=WMT"
Invoke-RestMethod "http://127.0.0.1:8000/api/metadata/filing-types?ticker=NVDA"
Invoke-RestMethod "http://127.0.0.1:8000/api/metadata/filing-sections?ticker=NVDA&form_type=10-K"

# OpenAI key and connection check (does not generate an answer)
Invoke-RestMethod "http://127.0.0.1:8000/api/health/openai"

# Evidence preview smoke test, retrieval only and no generated answer
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/api/retrieval/preview" -ContentType "application/json" -Body '{"question":"How has NVIDIA stock performed over the last year versus SPY?","ticker":"NVDA","source_type":"auto","top_k":3}'

# Market context smoke test
.\.venv\Scripts\python.exe -c "from app.services.market_context import get_market_context, build_market_context_text; data=get_market_context(['NVDA']); print(data[0]['ticker'], data[0]['latest_trading_date'], round(data[0]['returns']['1Y'], 4)); print(build_market_context_text(data).splitlines()[:5])"

# Chart-ready indexed price history for the frontend
Invoke-RestMethod "http://127.0.0.1:8000/api/market/prices?ticker=NVDA&period=1Y"

# Saved research history (full answers are saved after POST /api/research)
Invoke-RestMethod "http://127.0.0.1:8000/api/research/history?limit=20"

# Retrieval regression suite (embeddings only; no generated answers)
python -m evals.run_retrieval

# End-to-end answer-quality suite (generation plus structured model judge)
# Run after retrieval passes; results go to data/evals/response_report.json
python -m evals.run_responses

# Fast response-eval iteration on one golden question
python -m evals.run_responses --case-id wmt_latest_margin_answer

# Download, parsing, and embedding status
docker exec alphalens-postgres psql -U alphalens -d alphalens -P pager=off -c "SELECT download_status, parse_status, COUNT(*) FROM filings GROUP BY download_status, parse_status; SELECT embedding_status, COUNT(*) FROM filing_chunks GROUP BY embedding_status;"

# FAISS metadata
Get-Content data\faiss\sec_chunks.meta.json
Get-Content data\faiss\transcript_chunks.meta.json
```

## 8. Existing device: update and rerun

```powershell
git pull
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
docker compose up -d
```

Then repeat steps 5, 6, and 7.

## 9. Stop PostgreSQL

```powershell
docker compose down
```

Do not add `-v` unless you want to delete the PostgreSQL data.

## Files not uploaded to Git

These must be recreated or copied to a new device:

```text
.env
.venv/
PostgreSQL Docker data
data/sec/raw/
data/sec/clean/
data/faiss/
```
