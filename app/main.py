"""
AlphaLens - FastAPI Application

Purpose
-------
Main entry point for the AlphaLens backend API.

Run locally:

    uvicorn app.main:app --reload


Architecture:

Client
   ↓
FastAPI
   ↓
/api/research
   ↓
AlphaLens RAG
"""


from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.research import (
    router as research_router,
)


STATIC_DIRECTORY = (
    Path(__file__).resolve().parent
    / "static"
)


# ============================================================
# FastAPI Application
# ============================================================

app = FastAPI(

    title="AlphaLens API",

    description=(
        "AI-powered equity research using "
        "SEC filings, earnings transcripts, semantic retrieval and RAG."
    ),

    version="0.1.0",
)


# ============================================================
# Routers
# ============================================================
#
# research_router contains:
#
#     POST /api/research
#
# ============================================================

app.include_router(
    research_router
)


app.mount(
    "/static",
    StaticFiles(directory=STATIC_DIRECTORY),
    name="static",
)


# ============================================================
# Health Endpoint
# ============================================================

@app.get(
    "/health",
    tags=["System"],
)
def health():
    """
    Simple health check.

    Later AWS / Docker can use this endpoint to determine
    whether the AlphaLens API process is running.

    Important:
        This currently checks the API process only.

        Later we can make a deeper health check for:

            PostgreSQL
            FAISS
            OpenAI configuration
    """

    return {
        "status": "ok",
        "service": "alphalens",
    }


# ============================================================
# Root Endpoint
# ============================================================

@app.get(
    "/",
    tags=["System"],
)
def root():
    """
    Serve the AlphaLens research UI.
    """

    return FileResponse(
        STATIC_DIRECTORY / "index.html"
    )
