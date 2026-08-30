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
```

The embedder can be rerun. It resumes from the existing FAISS index.

## 7. Quick verification

```powershell
# Main database counts
docker exec alphalens-postgres psql -U alphalens -d alphalens -P pager=off -c "SELECT 'market_prices' AS name, COUNT(*) FROM market_prices UNION ALL SELECT 'filings', COUNT(*) FROM filings UNION ALL SELECT 'filing_sections', COUNT(*) FROM filing_sections UNION ALL SELECT 'filing_chunks', COUNT(*) FROM filing_chunks;"

# Download, parsing, and embedding status
docker exec alphalens-postgres psql -U alphalens -d alphalens -P pager=off -c "SELECT download_status, parse_status, COUNT(*) FROM filings GROUP BY download_status, parse_status; SELECT embedding_status, COUNT(*) FROM filing_chunks GROUP BY embedding_status;"

# FAISS metadata
Get-Content data\faiss\sec_chunks.meta.json
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
