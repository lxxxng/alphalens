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

from app.rag.company_resolver import (
    resolve_tickers,
)

from app.services.metadata import (
    get_available_tickers,
    get_filing_sections,
    get_filing_types,
    get_transcript_periods,
)

from app.services.market_context import (
    get_market_history,
)

from app.services.openai_health import (
    check_openai_connection,
)

from app.services.research_history import (
    delete_research_run,
    get_research_run,
    list_research_runs,
    save_research_run,
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

    # Explicit comparisons use this list. The singular ticker field remains
    # accepted for existing API clients and saved evaluation cases.
    tickers: list[str] = Field(
        default_factory=list,
        max_length=4,
    )

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

    # Older saved runs predate this field, so history responses default to an
    # empty mapping while all new market snapshots include benchmark values.
    benchmark_returns: dict[str, Optional[float]] = Field(
        default_factory=dict
    )

    benchmark_relative_returns: dict[str, Optional[float]]

    annualized_volatility: Optional[float] = None

    average_volume_30d: Optional[float] = None

    row_count: int


class MarketPricePoint(BaseModel):
    """One observation in a market chart series."""

    date: str

    close: float

    indexed_value: float


class MarketPriceSeries(BaseModel):
    """Chart points for one ticker."""

    ticker: str

    points: list[MarketPricePoint]


class MarketEvent(BaseModel):
    """One transcript or SEC filing event plotted against market prices."""

    event_id: str

    event_type: str

    ticker: str

    date: str

    plot_date: Optional[str] = None

    label: str

    detail: Optional[str] = None

    source_url: Optional[str] = None

    reaction_1d: Optional[float] = None

    reaction_5d: Optional[float] = None


class MarketHistoryResponse(BaseModel):
    """Company and benchmark history returned to the frontend chart."""

    ticker: str

    tickers: list[str] = Field(default_factory=list)

    benchmark_ticker: str

    period: str

    start_date: str

    end_date: str

    series: list[MarketPriceSeries]

    events: list[MarketEvent] = Field(default_factory=list)


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


class TickerResolutionResponse(BaseModel):
    """Companies detected in a natural-language research question."""

    question: str

    tickers: list[str] = Field(default_factory=list)


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

    tickers: list[str] = Field(default_factory=list)

    run_id: Optional[int] = None

    answer: str

    market_context: list[MarketSnapshot] = Field(
        default_factory=list
    )

    sources: list[ResearchSource]


class ResearchRunSummary(BaseModel):
    """One lightweight entry in the recent research list."""

    run_id: int

    question: str

    tickers: list[str] = Field(default_factory=list)

    source_type: str

    source_count: int

    market_snapshot_count: int

    created_at: str


class ResearchHistoryResponse(BaseModel):
    """Recent saved AlphaLens research runs."""

    runs: list[ResearchRunSummary]


class ResearchRunDetail(BaseModel):
    """Complete immutable snapshot of one saved research run."""

    run_id: int

    question: str

    answer: str

    tickers: list[str] = Field(default_factory=list)

    ticker_filter: Optional[str] = None

    source_type: str

    top_k: int

    form_type: Optional[str] = None

    section_key: Optional[str] = None

    fiscal_period: Optional[str] = None

    source_count: int

    market_snapshot_count: int

    market_context: list[MarketSnapshot] = Field(
        default_factory=list
    )

    sources: list[ResearchSource]

    created_at: str


class DeleteResearchRunResponse(BaseModel):
    """Confirmation returned after deleting a saved run."""

    run_id: int

    deleted: bool


class EvidencePreviewResponse(BaseModel):
    """
    JSON returned by POST /api/retrieval/preview.
    """

    question: str

    tickers: list[str] = Field(default_factory=list)

    market_context: list[MarketSnapshot] = Field(
        default_factory=list
    )

    sources: list[ResearchSource]


class OpenAIHealthResponse(BaseModel):
    """
    Connection status returned by GET /api/health/openai.
    """

    status: str

    configured: bool

    reachable: bool

    latency_ms: float

    message: str

    error_type: Optional[str] = None


# ============================================================
# GET /api/health/openai
# ============================================================

@router.get(
    "/health/openai",
    response_model=OpenAIHealthResponse,
    summary="Check the OpenAI API connection",
)
def openai_health():
    """
    Verify that the configured API key can reach OpenAI.

    This calls the Models endpoint and does not generate an answer or expose
    the key. The response remains HTTP 200 so the frontend can display the
    exact diagnostic state without treating expected configuration problems
    as a broken AlphaLens API request.
    """

    return check_openai_connection()


# ============================================================
# GET /api/market/prices
# ============================================================

@router.get(
    "/market/prices",
    response_model=MarketHistoryResponse,
    summary="Load market price history for a chart",
)
def market_prices(
    ticker: str = Query(
        ...,
        min_length=1,
        max_length=20,
    ),
    period: str = Query(
        default="1Y",
        pattern="^(1M|3M|1Y|5Y)$",
    ),
    tickers: Optional[str] = Query(
        default=None,
        description="Comma-separated additional company tickers.",
    ),
):
    """
    Return adjusted-close performance and dated company events.

    Values are indexed to 100 at the start of the requested period so the
    company and benchmark can share one meaningful chart scale. Transcript
    calls and SEC filings include forward trading-session reactions.
    """

    normalized_ticker = ticker.upper()
    comparison_tickers = [
        value.strip().upper()
        for value in (tickers or "").split(",")
        if value.strip()
    ]

    try:
        result = get_market_history(
            ticker=normalized_ticker,
            period=period,
            comparison_tickers=comparison_tickers,
        )
    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error
    except Exception as error:
        print(
            f"[API ERROR] "
            f"/api/market/prices: "
            f"{error}"
        )

        raise HTTPException(
            status_code=500,
            detail="AlphaLens could not load market price history.",
        ) from error

    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"No market prices found for {normalized_ticker}.",
        )

    return result


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

            tickers=request.tickers,

            form_type=request.form_type,

            section_key=request.section_key,

            fiscal_period=request.fiscal_period,

            source_type=request.source_type,
        )


        # History is deliberately best-effort. A temporary database problem
        # should not discard an answer that has already been generated.
        try:
            result["run_id"] = save_research_run(
                request_data=request.model_dump(),
                result=result,
            )
        except Exception as history_error:
            result["run_id"] = None
            print(
                f"[HISTORY WARNING] "
                f"Could not save research run: "
                f"{history_error}"
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
# GET /api/research/history
# ============================================================

@router.get(
    "/research/history",
    response_model=ResearchHistoryResponse,
    summary="List recent saved research runs",
)
def research_history(
    limit: int = Query(
        default=20,
        ge=1,
        le=100,
    ),
):
    """Return recent run summaries without their large evidence payloads."""

    try:
        return {
            "runs": list_research_runs(
                limit=limit
            )
        }
    except Exception as error:
        print(
            f"[API ERROR] "
            f"/api/research/history: "
            f"{error}"
        )

        raise HTTPException(
            status_code=500,
            detail="AlphaLens could not load research history.",
        ) from error


@router.get(
    "/research/history/{run_id}",
    response_model=ResearchRunDetail,
    summary="Open a saved research run",
)
def research_history_detail(
    run_id: int,
):
    """Return one saved answer and its original evidence snapshot."""

    try:
        result = get_research_run(
            run_id=run_id
        )
    except Exception as error:
        print(
            f"[API ERROR] "
            f"/api/research/history/{run_id}: "
            f"{error}"
        )

        raise HTTPException(
            status_code=500,
            detail="AlphaLens could not load the saved research run.",
        ) from error

    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"Research run {run_id} was not found.",
        )

    return result


@router.delete(
    "/research/history/{run_id}",
    response_model=DeleteResearchRunResponse,
    summary="Delete a saved research run",
)
def research_history_delete(
    run_id: int,
):
    """Delete one local history entry."""

    try:
        deleted = delete_research_run(
            run_id=run_id
        )
    except Exception as error:
        print(
            f"[API ERROR] "
            f"DELETE /api/research/history/{run_id}: "
            f"{error}"
        )

        raise HTTPException(
            status_code=500,
            detail="AlphaLens could not delete the saved research run.",
        ) from error

    if not deleted:
        raise HTTPException(
            status_code=404,
            detail=f"Research run {run_id} was not found.",
        )

    return {
        "run_id": run_id,
        "deleted": True,
    }


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

            tickers=request.tickers,

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
    "/metadata/resolve-tickers",
    response_model=TickerResolutionResponse,
    summary="Resolve companies from a question",
)
def metadata_resolve_tickers(
    question: str = Query(
        ...,
        min_length=1,
        max_length=2000,
    ),
):
    """Resolve company names and ticker symbols without calling OpenAI."""

    try:
        return {
            "question": question,
            "tickers": resolve_tickers(question),
        }
    except Exception as error:
        print(
            f"[API ERROR] "
            f"/api/metadata/resolve-tickers: "
            f"{error}"
        )

        raise HTTPException(
            status_code=500,
            detail="AlphaLens could not resolve question tickers.",
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
