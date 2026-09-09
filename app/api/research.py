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
         ->
    FastAPI
         ->
    AlphaLens RAG


or:

    Power BI
         ->
    FastAPI


or:

    another application
         ->
    FastAPI


Current architecture
--------------------

Client
   ->
POST /api/research
   ->
FastAPI
   ->
company_resolver.py
   ->
retriever.py
   ->
FAISS + PostgreSQL
   ->
generator.py
   ->
OpenAI
   ->
grounded answer
"""

from typing import Optional

from fastapi import (
    APIRouter,
    HTTPException,
    Query,
)

from pydantic import (
    BaseModel,
    Field,
)

from app.rag.generator import (
    answer_question,
    preview_evidence,
)

from app.services.metadata import (
    get_available_tickers,
    get_filing_sections,
    get_filing_types,
    get_transcript_periods,
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
                "item_1a_risk_factors",

            "source_type":
                "both"
        }


    Normally the user does NOT need to supply ticker.

    company_resolver.py automatically detects it from:

        "NVIDIA"

            ->

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
    # Number of chunks supplied to the generation model
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

    fiscal_period: Optional[str] = None

    source_type: str = Field(
        default="auto",
        pattern=(
            "^(auto|filing|filings|sec|transcript|transcripts|"
            "earnings|both|all)$"
        ),
    )


# ============================================================
# Source Response Model
# ============================================================

class ResearchSource(BaseModel):
    """
    One source used by the RAG system.

    Example:

        S1
            ->
        NVDA
        10-K
        Risk Factors
        filing date 2026-...
        chunk 18372
    """

    source: str

    chunk_id: int

    ticker: str

    source_type: str = "filing"

    form_type: Optional[str] = None

    filing_date: Optional[str] = None

    accession_number: Optional[str] = None

    section_key: Optional[str] = None

    section_title: Optional[str] = None

    transcript_id: Optional[int] = None

    fiscal_period: Optional[str] = None

    call_date: Optional[str] = None

    title: Optional[str] = None

    source_url: Optional[str] = None

    speaker_names: Optional[list[str]] = None

    chunk_index: int

    # Returned so the UI can show the exact evidence behind a citation.
    # This makes RAG answers auditable instead of being a black box.
    token_count: Optional[int] = None

    content: Optional[str] = None

    similarity_score: float


class MarketSnapshot(BaseModel):
    """
    Structured market metrics calculated from market_prices.
    """

    ticker: str

    latest_trading_date: str

    latest_close: Optional[float] = None

    latest_adjusted_close: Optional[float] = None

    returns: dict[str, Optional[float]]

    benchmark_ticker: str

    benchmark_relative_returns: dict[str, Optional[float]]

    annualized_volatility: Optional[float] = None

    average_volume_30d: Optional[float] = None

    row_count: int


class TickerMetadata(BaseModel):
    """
    One ticker available in the local AlphaLens database.
    """

    ticker: str

    company_name: Optional[str] = None

    filing_count: int

    transcript_count: int

    market_price_count: int


class TickersResponse(BaseModel):
    """
    Ticker choices for the research form.
    """

    tickers: list[TickerMetadata]


class TranscriptPeriodMetadata(BaseModel):
    """
    One earnings-call period available for a ticker.
    """

    fiscal_period: str

    fiscal_year: int

    fiscal_quarter: int

    call_date: Optional[str] = None

    title: Optional[str] = None

    turn_count: int

    char_count: int


class TranscriptPeriodsResponse(BaseModel):
    """
    Transcript period choices for one ticker.
    """

    ticker: str

    periods: list[TranscriptPeriodMetadata]


class FilingTypeMetadata(BaseModel):
    """
    One SEC filing type available in the corpus.
    """

    form_type: str

    filing_count: int


class FilingTypesResponse(BaseModel):
    """
    SEC filing type choices for the research form.
    """

    ticker: Optional[str] = None

    form_types: list[FilingTypeMetadata]


class FilingSectionMetadata(BaseModel):
    """
    One SEC section filter available in filing chunks.
    """

    section_key: str

    section_title: Optional[str] = None

    chunk_count: int


class FilingSectionsResponse(BaseModel):
    """
    SEC section choices for the research form.
    """

    ticker: Optional[str] = None

    form_type: Optional[str] = None

    sections: list[FilingSectionMetadata]


# ============================================================
# Research Response Model
# ============================================================

class ResearchResponse(BaseModel):
    """
    JSON returned by POST /api/research.
    """

    question: str

    answer: str

    market_context: list[MarketSnapshot] = Field(
        default_factory=list
    )

    sources: list[ResearchSource]


class EvidencePreviewResponse(BaseModel):
    """
    JSON returned by POST /api/retrieval/preview.
    """

    question: str

    market_context: list[MarketSnapshot] = Field(
        default_factory=list
    )

    sources: list[ResearchSource]


# ============================================================
# POST /api/research
# ============================================================

@router.post(
    "/research",

    response_model=ResearchResponse,

    summary="Research AlphaLens sources",
)
def research(
    request: ResearchRequest,
):
    """
    Ask AlphaLens a question about its SEC filing corpus.

    Flow
    ----

        HTTP request
             ->
        validate JSON
             ->
        answer_question()
             ->
        company resolver
             ->
        semantic retrieval
             ->
        FAISS
             ->
        PostgreSQL
             ->
        grounded generation
             ->
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

            fiscal_period=request.fiscal_period,

            source_type=request.source_type,
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


# ============================================================
# POST /api/retrieval/preview
# ============================================================

@router.post(
    "/retrieval/preview",

    response_model=EvidencePreviewResponse,

    summary="Preview retrieved AlphaLens evidence",
)
def retrieval_preview(
    request: ResearchRequest,
):
    """
    Retrieve evidence without calling the answer-generation model.

    This endpoint is useful when debugging:

        ticker detection
        source_type routing
        transcript fiscal-period filters
        SEC filing filters
        market_context calculations

    It may still create a query embedding for semantic search, but it
    does not send retrieved evidence to the answer generator.
    """

    try:

        result = preview_evidence(

            question=request.question,

            top_k=request.top_k,

            ticker=request.ticker,

            form_type=request.form_type,

            section_key=request.section_key,

            fiscal_period=request.fiscal_period,

            source_type=request.source_type,
        )


        return result


    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error


    except Exception as error:

        print(
            f"[API ERROR] "
            f"/api/retrieval/preview: "
            f"{error}"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "AlphaLens could not preview retrieved "
                "evidence."
            ),
        ) from error


# ============================================================
# Metadata Endpoints
# ============================================================

@router.get(
    "/metadata/tickers",
    response_model=TickersResponse,
    summary="List available tickers",
)
def metadata_tickers():
    """
    Return tickers that have local AlphaLens data.
    """

    try:
        return {
            "tickers": get_available_tickers()
        }
    except Exception as error:
        print(
            f"[API ERROR] "
            f"/api/metadata/tickers: "
            f"{error}"
        )

        raise HTTPException(
            status_code=500,
            detail="AlphaLens could not load ticker metadata.",
        ) from error


@router.get(
    "/metadata/transcript-periods",
    response_model=TranscriptPeriodsResponse,
    summary="List transcript periods for a ticker",
)
def metadata_transcript_periods(
    ticker: str = Query(
        ...,
        min_length=1,
        max_length=20,
    ),
):
    """
    Return stored earnings-call periods for one ticker.
    """

    normalized_ticker = ticker.upper()

    try:
        return {
            "ticker": normalized_ticker,
            "periods": get_transcript_periods(
                normalized_ticker
            ),
        }
    except Exception as error:
        print(
            f"[API ERROR] "
            f"/api/metadata/transcript-periods: "
            f"{error}"
        )

        raise HTTPException(
            status_code=500,
            detail="AlphaLens could not load transcript periods.",
        ) from error


@router.get(
    "/metadata/filing-types",
    response_model=FilingTypesResponse,
    summary="List SEC filing types",
)
def metadata_filing_types(
    ticker: Optional[str] = Query(
        default=None,
        min_length=1,
        max_length=20,
    ),
):
    """
    Return available SEC filing types, optionally for one ticker.
    """

    normalized_ticker = (
        ticker.upper()
        if ticker
        else None
    )

    try:
        return {
            "ticker": normalized_ticker,
            "form_types": get_filing_types(
                normalized_ticker
            ),
        }
    except Exception as error:
        print(
            f"[API ERROR] "
            f"/api/metadata/filing-types: "
            f"{error}"
        )

        raise HTTPException(
            status_code=500,
            detail="AlphaLens could not load filing types.",
        ) from error


@router.get(
    "/metadata/filing-sections",
    response_model=FilingSectionsResponse,
    summary="List SEC filing sections",
)
def metadata_filing_sections(
    ticker: Optional[str] = Query(
        default=None,
        min_length=1,
        max_length=20,
    ),
    form_type: Optional[str] = Query(
        default=None,
        min_length=1,
        max_length=10,
    ),
):
    """
    Return available filing section filters.
    """

    normalized_ticker = (
        ticker.upper()
        if ticker
        else None
    )

    normalized_form_type = (
        form_type.upper()
        if form_type
        else None
    )

    try:
        return {
            "ticker": normalized_ticker,
            "form_type": normalized_form_type,
            "sections": get_filing_sections(
                ticker=normalized_ticker,
                form_type=normalized_form_type,
            ),
        }
    except Exception as error:
        print(
            f"[API ERROR] "
            f"/api/metadata/filing-sections: "
            f"{error}"
        )

        raise HTTPException(
            status_code=500,
            detail="AlphaLens could not load filing sections.",
        ) from error
