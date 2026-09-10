"""
AlphaLens - FastAPI Application

Purpose
-------
Main entry point for the AlphaLens backend API.

Run locally:

    uvicorn app.main:app --reload


Architecture:

Client
   ->
FastAPI
   ->
/api/research
   ->
AlphaLens RAG
"""


from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.research import (
    router as research_router,
)
from app.api.evaluations import (
    router as evaluations_router,
)
from app.api.sentiment import (
    router as sentiment_router,
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

app.include_router(
    evaluations_router
)

app.include_router(
    sentiment_router
)


# Serve the lightweight browser UI from the same FastAPI process as the API.
# That keeps local development simple and avoids a separate frontend server.
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


@app.get(
    "/evals",
    tags=["System"],
)
def evaluations_page():
    """Serve the internal evaluation dashboard."""

    return FileResponse(
        STATIC_DIRECTORY / "evals.html"
    )


@app.get(
    "/transcripts/{transcript_id}",
    tags=["System"],
)
def transcript_page(transcript_id: int):
    """Serve the local earnings-transcript reader."""

    return FileResponse(
        STATIC_DIRECTORY / "transcript.html"
    )
