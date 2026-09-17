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
Get-Content -Raw db\sql\010_transcript_sentiment.sql | docker exec -i alphalens-postgres psql -v ON_ERROR_STOP=1 -U alphalens -d alphalens
Get-Content -Raw db\sql\011_filing_sentiment.sql | docker exec -i alphalens-postgres psql -v ON_ERROR_STOP=1 -U alphalens -d alphalens
Get-Content -Raw db\sql\012_event_briefs.sql | docker exec -i alphalens-postgres psql -v ON_ERROR_STOP=1 -U alphalens -d alphalens
Get-Content -Raw db\sql\013_watchlists.sql | docker exec -i alphalens-postgres psql -v ON_ERROR_STOP=1 -U alphalens -d alphalens
Get-Content -Raw db\sql\014_ingestion_runs.sql | docker exec -i alphalens-postgres psql -v ON_ERROR_STOP=1 -U alphalens -d alphalens
Get-Content -Raw db\sql\015_event_alerts.sql | docker exec -i alphalens-postgres psql -v ON_ERROR_STOP=1 -U alphalens -d alphalens
Get-Content -Raw db\sql\016_automated_event_briefs.sql | docker exec -i alphalens-postgres psql -v ON_ERROR_STOP=1 -U alphalens -d alphalens
Get-Content -Raw db\sql\017_earnings_results.sql | docker exec -i alphalens-postgres psql -v ON_ERROR_STOP=1 -U alphalens -d alphalens
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

# Versioned FinBERT sentiment (install requirements-ml.txt once)
.\.venv\Scripts\python.exe -m pipelines.transcripts.sentiment --tickers WMT --limit 10

# Modeling-data audit and interactive notebook (install requirements-research.txt once)
.\.venv\Scripts\python.exe -m pipelines.ml.data_audit --horizon 30
.\.venv\Scripts\python.exe -m jupyter lab notebooks\01_data_audit.ipynb

# Point-in-time 30-session stock, SPY, and excess-return targets
.\.venv\Scripts\python.exe -m pipelines.ml.dataset --horizon 30 --output data\ml\event_targets_30d.csv
.\.venv\Scripts\python.exe -m jupyter lab notebooks\02_target_construction.ipynb

# Announced EPS results and surprises (apply migration 017 first)
.\.venv\Scripts\python.exe -m pipelines.earnings_results.run_pipeline --limit 24
.\.venv\Scripts\python.exe -m jupyter lab notebooks\05_earnings_surprises.ipynb

# Point-in-time market, event, FinBERT, and topic features
.\.venv\Scripts\python.exe -m pipelines.ml.features --horizon 30 --output data\ml\event_features_30d.csv
.\.venv\Scripts\python.exe -m jupyter lab notebooks\03_feature_analysis.ipynb

# Purged chronological naive and Ridge baselines
.\.venv\Scripts\python.exe -m pipelines.ml.baselines --output data\ml\baseline_metrics.json --predictions data\ml\baseline_test_predictions.csv
.\.venv\Scripts\python.exe -m jupyter lab notebooks\04_baseline_models.ipynb

# Validation-selected XGBoost versus locked baselines
.\.venv\Scripts\python.exe -m pipelines.ml.xgboost_model
.\.venv\Scripts\python.exe -m jupyter lab notebooks\06_xgboost_model.ipynb

# Event-driven portfolio backtest with turnover and transaction costs
.\.venv\Scripts\python.exe -m pipelines.ml.backtest
.\.venv\Scripts\python.exe -m jupyter lab notebooks\07_strategy_backtest.ipynb

# SHAP explanations, stability checks, and test-error slices
.\.venv\Scripts\python.exe -m pipelines.ml.interpretability
.\.venv\Scripts\python.exe -m jupyter lab notebooks\08_shap_error_analysis.ipynb

# Immutable model bundle, promotion gates, and integrity verification
.\.venv\Scripts\python.exe -m pipelines.ml.registry
.\.venv\Scripts\python.exe -m jupyter lab notebooks\09_model_registry.ipynb

# Registry status and guarded research-only prediction smoke tests
Invoke-RestMethod "http://127.0.0.1:8000/api/models/status"
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/api/models/predict" -ContentType "application/json" -Body '{"tickers":["WMT","NVDA"],"event_source":"latest","research_preview":true}'

# Purged expanding-window Ridge, Elastic Net, Extra Trees, XGBoost, and
# CatBoost comparison. This writes ignored JSON and prediction artifacts under
# data/ml and does not promote a registry model.
.\.venv\Scripts\python.exe -m pipelines.ml.model_benchmark
.\.venv\Scripts\python.exe -m jupyter lab notebooks\10_walk_forward_model_benchmark.ipynb

# SEC narrative sentiment targets MD&A, Risk Factors, and Market Risk
.\.venv\Scripts\python.exe -m pipelines.sec.sentiment --tickers WMT --limit 10

# Coverage-aware sentiment API smoke tests
Invoke-RestMethod "http://127.0.0.1:8000/api/sentiment/transcripts?ticker=WMT"
Invoke-RestMethod "http://127.0.0.1:8000/api/sentiment/transcripts/287"
Invoke-RestMethod "http://127.0.0.1:8000/api/sentiment/filings?ticker=WMT&form_type=10-K"
Invoke-RestMethod "http://127.0.0.1:8000/api/sentiment/filings/0000104169-21-000058"
Invoke-RestMethod "http://127.0.0.1:8000/api/sentiment/topics"

# Local research UI
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000

# Containerized API + PostgreSQL (stop the command above before using port 8000)
docker compose config
docker compose up -d --build
docker compose ps
Invoke-RestMethod "http://127.0.0.1:8000/health"
Invoke-RestMethod "http://127.0.0.1:8000/ready"
docker compose logs -f app

# UI metadata smoke tests
Invoke-RestMethod "http://127.0.0.1:8000/api/metadata/tickers"
Invoke-RestMethod "http://127.0.0.1:8000/api/metadata/resolve-tickers?question=Compare%20Costco%20and%20Walmart"
Invoke-RestMethod "http://127.0.0.1:8000/api/metadata/transcript-periods?ticker=WMT"
Invoke-RestMethod "http://127.0.0.1:8000/api/metadata/filing-types?ticker=NVDA"
Invoke-RestMethod "http://127.0.0.1:8000/api/metadata/filing-sections?ticker=NVDA&form_type=10-K"

# Replace 269 with a transcript_id from your database. This reads PostgreSQL
# only; the EarningsCalls.dev API key is not sent to the browser.
Invoke-RestMethod "http://127.0.0.1:8000/api/transcripts/269"

# OpenAI key and connection check (does not generate an answer)
Invoke-RestMethod "http://127.0.0.1:8000/api/health/openai"

# Evidence preview smoke test, retrieval only and no generated answer
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/api/retrieval/preview" -ContentType "application/json" -Body '{"question":"How has NVIDIA stock performed over the last year versus SPY?","ticker":"NVDA","source_type":"auto","top_k":3}'

# Multi-company evidence preview (up to four tickers)
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/api/retrieval/preview" -ContentType "application/json" -Body '{"question":"Compare Walmart and Costco margin commentary.","tickers":["WMT","COST"],"source_type":"both","top_k":3}'

# LCEL event brief: retrieval + market context + sentiment + structured output
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/api/briefs/generate" -ContentType "application/json" -Body '{"ticker":"WMT","event_type":"earnings","fiscal_period":"2026Q4","top_k":6}'

# Market context smoke test
.\.venv\Scripts\python.exe -c "from app.services.market_context import get_market_context, build_market_context_text; data=get_market_context(['NVDA']); print(data[0]['ticker'], data[0]['latest_trading_date'], round(data[0]['returns']['1Y'], 4)); print(build_market_context_text(data).splitlines()[:5])"

# Chart-ready price history plus earnings/filing events and forward reactions
Invoke-RestMethod "http://127.0.0.1:8000/api/market/prices?ticker=NVDA&period=1Y"

# Multi-company chart history; the first ticker is the primary comparison
Invoke-RestMethod "http://127.0.0.1:8000/api/market/prices?ticker=WMT&tickers=NVDA,COST&period=1Y"

# Saved research history (full answers are saved after POST /api/research)
Invoke-RestMethod "http://127.0.0.1:8000/api/research/history?limit=20"

# Watchlists read local prices, sentiment, and event metadata only
$watchlists = Invoke-RestMethod "http://127.0.0.1:8000/api/watchlists"
$watchlistId = $watchlists.watchlists[0].watchlist_id
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/api/watchlists/$watchlistId/items" -ContentType "application/json" -Body '{"tickers":["WMT","NVDA"]}'
Invoke-RestMethod "http://127.0.0.1:8000/api/watchlists/$watchlistId"

# Preview the watchlist ticker scope without external API calls or DB writes
python -m pipelines.scheduled_ingestion --scope watchlists --dry-run

# Run an incremental refresh now and save a timestamped local log
.\scripts\run_scheduled_ingestion.ps1 -Scope watchlists

# Bound OpenAI usage or disable automatic briefs with 0
.\scripts\run_scheduled_ingestion.ps1 -Scope watchlists -MaxAutoBriefs 5

# One-time Windows Task Scheduler registration (daily at 06:30 local time)
.\scripts\register_ingestion_task.ps1 -DailyAt "06:30" -Scope watchlists

# Inspect recent scheduler outcomes and per-stage JSON results
docker exec alphalens-postgres psql -U alphalens -d alphalens -P pager=off -c "SELECT run_id, trigger_type, scope, tickers, status, current_stage, started_at, completed_at, error FROM ingestion_runs ORDER BY run_id DESC LIMIT 10;"

# In-app watched-company alerts and unread count
Invoke-RestMethod "http://127.0.0.1:8000/api/alerts?limit=30"

# Inspect automatic brief queue, retries, and deterministic quality results
docker exec alphalens-postgres psql -U alphalens -d alphalens -P pager=off -c "SELECT alert_id, ticker, event_type, brief_status, brief_attempt_count, brief_id, brief_evaluation->>'score' AS quality_score, brief_error FROM event_alerts ORDER BY alert_id DESC LIMIT 20;"

# Retrieval regression suite (embeddings only; no generated answers)
python -m evals.run_retrieval

# End-to-end answer-quality suite (generation plus structured model judge)
# Run after retrieval passes; results go to data/evals/response_report.json
python -m evals.run_responses

# Fast response-eval iteration on one golden question
python -m evals.run_responses --case-id wmt_latest_margin_answer

# Complete local release gate: unit tests, preflight, retrieval, responses
.\scripts\run_quality_gate.ps1

# Validate the runner's corpus configuration without making OpenAI requests
.\scripts\run_quality_gate.ps1 -SkipUnitTests -PreflightOnly

# Internal evaluation dashboard (served by the normal FastAPI command)
# http://127.0.0.1:8000/evals

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
