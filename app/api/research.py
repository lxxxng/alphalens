"""
AlphaLens - Research API

Purpose
-------
Expose the AlphaLens RAG system through an HTTP API.

Instead of running Python directly:

    answer_question(
        "What cybersecurity risks does NVIDIA face?"
    )

an external application can send:

    POST /api/research

    {
        "question":
            "What cybersecurity risks does NVIDIA face?"
    }


The API then returns:

    {
        "question": "...",

        "answer":
            "... [S1] [S2]",

        "sources": [...]
    }


Why FastAPI?
------------

FastAPI turns our Python RAG functions into a backend service.

Later this allows:

    Web frontend
         ↓
    FastAPI
         ↓
    AlphaLens RAG


or:

    Power BI
         ↓
    FastAPI


or:

    another application
         ↓
    FastAPI


Current architecture
--------------------

Client
   ↓
POST /api/research
   ↓
FastAPI
   ↓
company_resolver.py
   ↓
retriever.py
   ↓
FAISS + PostgreSQL
   ↓
generator.py
   ↓
OpenAI
   ↓
grounded answer
"""

from typing import Optional

from fastapi import (
    APIRouter,
    HTTPException,
)

from pydantic import (
    BaseModel,
    Field,
)

from app.rag.generator import (
    answer_question,
)


# ============================================================
# Router
# ============================================================
#
# APIRouter lets us keep research-related endpoints in their
# own file rather than putting every endpoint inside main.py.
#
# main.py will later attach this router to the application.
# ============================================================

router = APIRouter(
    prefix="/api",
    tags=["Research"],
)


# ============================================================
# Request Model
# ============================================================

class ResearchRequest(BaseModel):
    """
    JSON structure accepted by:

        POST /api/research


    Minimum request:

        {
            "question":
                "What cybersecurity risks does NVIDIA face?"
        }


    Optional advanced request:

        {
            "question":
                "What risks does NVIDIA discuss?",

            "top_k": 7,

            "ticker": "NVDA",

            "form_type": "10-K",

            "section_key":
                "item_1a_risk_factors"
        }


    Normally the user does NOT need to supply ticker.

    company_resolver.py automatically detects it from:

        "NVIDIA"

            ↓

        NVDA
    """

    # --------------------------------------------------------
    # User's research question
    # --------------------------------------------------------

    question: str = Field(

        ...,

        # Reject completely tiny / accidental input.
        min_length=2,

        # Prevent someone from submitting enormous text where
        # a research question is expected.
        max_length=2000,

        examples=[
            "What cybersecurity risks does NVIDIA face?"
        ],
    )


    # --------------------------------------------------------
    # Number of SEC chunks supplied to the generation model
    # --------------------------------------------------------

    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
    )


    # --------------------------------------------------------
    # Optional explicit metadata filters
    # --------------------------------------------------------
    #
    # Usually these remain None.
    #
    # They are useful for:
    #
    #     debugging
    #     API consumers
    #     future advanced search UI
    #
    ticker: Optional[str] = None

    form_type: Optional[str] = None

    section_key: Optional[str] = None


# ============================================================
# Source Response Model
# ============================================================

class ResearchSource(BaseModel):
    """
    One SEC source used by the RAG system.

    Example:

        S1
            ↓
        NVDA
        10-K
        Risk Factors
        filing date 2026-...
        chunk 18372
    """

    source: str

    chunk_id: int

    ticker: str

    form_type: str

    filing_date: str

    accession_number: str

    section_key: str

    section_title: str

    chunk_index: int

    similarity_score: float


# ============================================================
# Research Response Model
# ============================================================

class ResearchResponse(BaseModel):
    """
    JSON returned by POST /api/research.
    """

    question: str

    answer: str

    sources: list[ResearchSource]


# ============================================================
# POST /api/research
# ============================================================

@router.post(
    "/research",

    response_model=ResearchResponse,

    summary="Research SEC filings",
)
def research(
    request: ResearchRequest,
):
    """
    Ask AlphaLens a question about its SEC filing corpus.

    Flow
    ----

        HTTP request
             ↓
        validate JSON
             ↓
        answer_question()
             ↓
        company resolver
             ↓
        semantic retrieval
             ↓
        FAISS
             ↓
        PostgreSQL
             ↓
        grounded generation
             ↓
        JSON response


    Why is this endpoint a normal `def` rather than
    `async def`?
    -------------------------------------------------

    answer_question() currently performs synchronous work:

        SQLAlchemy database access
        FAISS search
        OpenAI SDK calls

    FastAPI can execute normal synchronous route functions
    safely using its worker thread handling.

    We therefore don't need to convert the entire AlphaLens
    pipeline to async code yet.
    """

    try:

        # ====================================================
        # Run AlphaLens RAG
        # ====================================================

        result = answer_question(

            question=request.question,

            top_k=request.top_k,

            ticker=request.ticker,

            form_type=request.form_type,

            section_key=request.section_key,
        )


        # FastAPI automatically converts this dictionary into
        # the ResearchResponse JSON schema.
        return result


    # ========================================================
    # User / query errors
    # ========================================================
    #
    # Example:
    #
    # company_resolver currently detects:
    #
    #     MSFT
    #     NVDA
    #
    # in:
    #
    #     "Compare Microsoft and NVIDIA"
    #
    # but multi-company retrieval isn't implemented yet.
    #
    # generator.py raises ValueError.
    #
    # Convert that into:
    #
    #     HTTP 400 Bad Request
    #
    # rather than returning an internal server error.
    # ========================================================

    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error


    # ========================================================
    # Unexpected errors
    # ========================================================
    #
    # We don't return the raw internal exception because it
    # could expose implementation/database details.
    # ========================================================

    except Exception as error:

        print(
            f"[API ERROR] "
            f"/api/research: "
            f"{error}"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "AlphaLens could not complete the "
                "research request."
            ),
        ) from error