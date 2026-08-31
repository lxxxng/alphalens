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


from fastapi import FastAPI

from app.api.research import (
    router as research_router,
)


# ============================================================
# FastAPI Application
# ============================================================

app = FastAPI(

    title="AlphaLens API",

    description=(
        "AI-powered equity research using "
        "SEC filings, semantic retrieval and RAG."
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
    Basic API landing endpoint.
    """

    return {
        "name": "AlphaLens API",
        "version": "0.1.0",
        "docs": "/docs",
    }