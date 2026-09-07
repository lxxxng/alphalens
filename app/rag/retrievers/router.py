"""
Route AlphaLens retrieval across SEC filings and earnings transcripts.
"""

from app.rag.retrievers import sec_retriever, transcript_retriever


def normalize_corpus(
    corpus: str,
) -> str:
    """
    Normalize corpus names used by compatibility callers.
    """

    normalized = corpus.lower().strip()

    aliases = {
        "sec": "filings",
        "filing": "filings",
        "filings": "filings",
        "transcript": "transcripts",
        "transcripts": "transcripts",
        "earnings": "transcripts",
        "earnings_calls": "transcripts",
    }

    if normalized not in aliases:
        raise ValueError(
            "corpus must be one of: filings, transcripts."
        )

    return aliases[normalized]


def resolve_source_types(
    question: str,
    source_type: str,
) -> list[str]:
    """
    Decide which indexed corpora should be searched.
    """

    normalized = source_type.lower().strip()

    explicit = {
        "filing": ["filings"],
        "filings": ["filings"],
        "sec": ["filings"],
        "transcript": ["transcripts"],
        "transcripts": ["transcripts"],
        "earnings": ["transcripts"],
        "earnings_calls": ["transcripts"],
        "both": ["filings", "transcripts"],
        "all": ["filings", "transcripts"],
    }

    if normalized in explicit:
        return explicit[normalized]

    if normalized != "auto":
        raise ValueError(
            "source_type must be one of: auto, filings, "
            "transcripts, both."
        )

    lower_question = question.lower()

    transcript_terms = [
        "earnings call",
        "call",
        "transcript",
        "management said",
        "management say",
        "analyst",
        "q&a",
        "qa",
        "guidance",
    ]

    filing_terms = [
        "sec",
        "filing",
        "10-k",
        "10-q",
        "annual report",
        "quarterly report",
        "risk factor",
        "item 1a",
    ]

    wants_transcripts = any(
        term in lower_question
        for term in transcript_terms
    )

    wants_filings = any(
        term in lower_question
        for term in filing_terms
    )

    if wants_transcripts and not wants_filings:
        return ["transcripts"]

    if wants_filings and not wants_transcripts:
        return ["filings"]

    return ["filings", "transcripts"]


def semantic_search(
    query: str,
    top_k: int = 5,
    ticker: str | None = None,
    form_type: str | None = None,
    section_key: str | None = None,
    fiscal_period: str | None = None,
    corpus: str = "filings",
):
    """
    Search one corpus while preserving the old retriever interface.
    """

    normalized_corpus = normalize_corpus(
        corpus
    )

    if normalized_corpus == "transcripts":
        return transcript_retriever.semantic_search(
            query=query,
            top_k=top_k,
            ticker=ticker,
            fiscal_period=fiscal_period,
        )

    return sec_retriever.semantic_search(
        query=query,
        top_k=top_k,
        ticker=ticker,
        form_type=form_type,
        section_key=section_key,
    )


def retrieve_evidence(
    question: str,
    top_k: int,
    tickers: list[str],
    form_type: str | None = None,
    section_key: str | None = None,
    fiscal_period: str | None = None,
    source_type: str = "auto",
) -> list[dict]:
    """
    Retrieve AlphaLens evidence for one or more companies.
    """

    all_results = []
    source_types = resolve_source_types(
        question=question,
        source_type=source_type,
    )

    if not tickers:
        for current_source_type in source_types:
            all_results.extend(
                semantic_search(
                    query=question,
                    top_k=top_k,
                    form_type=form_type,
                    section_key=section_key,
                    fiscal_period=fiscal_period,
                    corpus=current_source_type,
                )
            )

        return all_results

    for ticker in tickers:
        for current_source_type in source_types:
            all_results.extend(
                semantic_search(
                    query=question,
                    top_k=top_k,
                    ticker=ticker,
                    form_type=form_type,
                    section_key=section_key,
                    fiscal_period=fiscal_period,
                    corpus=current_source_type,
                )
            )

    return all_results
