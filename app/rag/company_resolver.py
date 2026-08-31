"""
AlphaLens - Company / Ticker Resolver

Purpose
-------
Detect which AlphaLens company the user is referring to.

Example:

    "What cybersecurity risks does NVIDIA face?"

                    ↓

                 NVIDIA

                    ↓

                  NVDA


Why do this?
------------

Without metadata filtering, FAISS searches every company.

A cybersecurity question about NVIDIA could otherwise return:

    Microsoft
    Meta
    Apple
    NVIDIA

because those companies discuss similar cybersecurity topics.

If we know the user specifically mentioned NVIDIA, we can tell
the retriever:

    ticker = "NVDA"

and search only NVIDIA-related chunks.


Approach
--------

We use a deterministic mapping rather than an LLM.

Why?

    - only 20 companies
    - cheaper
    - faster
    - predictable
    - easy to debug


Sources of names:

    1. companies table
    2. stock ticker
    3. common aliases

Example:

    GOOGL

can be referred to as:

    GOOGL
    Alphabet
    Google

All should resolve to:

    GOOGL
"""

import os
import re

from dotenv import load_dotenv

from sqlalchemy import (
    MetaData,
    Table,
    create_engine,
    select,
)


# ============================================================
# Environment
# ============================================================

load_dotenv()


# ============================================================
# Common Company Aliases
# ============================================================
#
# The companies table contains official SEC company names.
#
# But users do not always type those formal names.
#
# Example:
#
#     SEC:
#         Alphabet Inc.
#
#     User:
#         Google
#
# Therefore we maintain a small alias dictionary.
#
# Since AlphaLens only tracks 20 stocks, this is completely
# manageable.
# ============================================================

COMPANY_ALIASES = {

    "AAPL": [
        "apple",
        "apple inc",
    ],

    "MSFT": [
        "microsoft",
        "microsoft corporation",
    ],

    "NVDA": [
        "nvidia",
        "nvidia corporation",
    ],

    "AMZN": [
        "amazon",
        "amazon.com",
    ],

    "GOOGL": [
        "alphabet",
        "google",
        "alphabet inc",
    ],

    "META": [
        "meta",
        "meta platforms",
        "facebook",
    ],

    "TSLA": [
        "tesla",
        "tesla inc",
    ],

    "JPM": [
        "jpmorgan",
        "jp morgan",
        "jpmorgan chase",
        "jp morgan chase",
    ],

    "BAC": [
        "bank of america",
        "bofa",
    ],

    "GS": [
        "goldman sachs",
        "goldman",
    ],

    "JNJ": [
        "johnson & johnson",
        "johnson and johnson",
        "j&j",
    ],

    "UNH": [
        "unitedhealth",
        "unitedhealth group",
    ],

    "CVX": [
        "chevron",
    ],

    "KO": [
        "coca cola",
        "coca-cola",
        "the coca-cola company",
    ],

    "WMT": [
        "walmart",
        "wal-mart",
    ],

    "COST": [
        "costco",
    ],

    "PG": [
        "procter & gamble",
        "procter and gamble",
        "p&g",
    ],

    "CAT": [
        "caterpillar",
    ],

    "BA": [
        "boeing",
    ],

    "NFLX": [
        "netflix",
    ],
}


# ============================================================
# get_database_engine()
# ============================================================

def get_database_engine():
    """
    Create PostgreSQL SQLAlchemy engine.
    """

    database_url = os.getenv(
        "DATABASE_URL"
    )

    if not database_url:

        raise ValueError(
            "DATABASE_URL was not found in .env."
        )

    return create_engine(
        database_url,
        pool_pre_ping=True,
    )


# ============================================================
# normalize_text()
# ============================================================

def normalize_text(
    value: str,
) -> str:
    """
    Normalize text before company-name comparison.

    Example:

        "NVIDIA Corporation"

                ↓

        "nvidia corporation"


    We:

        - convert to lowercase
        - normalize whitespace
        - remove leading/trailing spaces
    """

    value = value.lower()

    value = re.sub(
        r"\s+",
        " ",
        value,
    )

    return value.strip()


# ============================================================
# get_companies()
# ============================================================

def get_companies(
    engine,
):
    """
    Retrieve AlphaLens companies from PostgreSQL.

    Example:

        ticker | company_name
        -------+-----------------------
        AAPL   | Apple Inc.
        NVDA   | NVIDIA Corporation
    """

    metadata = MetaData()


    companies_table = Table(
        "companies",
        metadata,
        autoload_with=engine,
    )


    query = (
        select(
            companies_table.c.ticker,
            companies_table.c.company_name,
        )
        .order_by(
            companies_table.c.ticker
        )
    )


    with engine.connect() as connection:

        companies = (
            connection
            .execute(query)
            .mappings()
            .all()
        )


    return companies


# ============================================================
# build_alias_map()
# ============================================================

def build_alias_map(
    companies,
) -> dict:
    """
    Build:

        alias
            ↓
        ticker

    Example:

        "nvidia"              -> NVDA
        "nvidia corporation"  -> NVDA
        "nvda"                -> NVDA

        "google"              -> GOOGL
        "alphabet"            -> GOOGL
        "googl"               -> GOOGL


    We combine:

        database company names
        +
        ticker symbols
        +
        manually defined common aliases
    """

    alias_map = {}


    for company in companies:

        ticker = (
            company["ticker"]
            .upper()
        )


        company_name = (
            company["company_name"]
        )


        # ----------------------------------------------------
        # Ticker itself
        # ----------------------------------------------------
        #
        # Example:
        #
        #     "What are NVDA's risks?"
        #
        alias_map[
            ticker.lower()
        ] = ticker


        # ----------------------------------------------------
        # Official company name
        # ----------------------------------------------------

        if company_name:

            normalized_name = (
                normalize_text(
                    company_name
                )
            )

            alias_map[
                normalized_name
            ] = ticker


        # ----------------------------------------------------
        # Common human aliases
        # ----------------------------------------------------

        for alias in COMPANY_ALIASES.get(
            ticker,
            [],
        ):

            alias_map[
                normalize_text(alias)
            ] = ticker


    return alias_map


# ============================================================
# alias_in_question()
# ============================================================

def alias_in_question(
    question: str,
    alias: str,
) -> bool:
    """
    Check whether an alias appears as a complete phrase.

    Why not simply:

        alias in question

    ?

    Because small ticker symbols can accidentally appear
    inside unrelated words.

    Example:

        ticker = "CAT"

    We don't want:

        "What categories of risk..."

    to accidentally match:

        CAT


    Word-boundary style matching helps avoid that.
    """

    escaped_alias = re.escape(
        alias
    )


    pattern = (
        r"(?<!\w)"
        + escaped_alias
        + r"(?!\w)"
    )


    return bool(
        re.search(
            pattern,
            question,
            flags=re.IGNORECASE,
        )
    )


# ============================================================
# resolve_tickers()
# ============================================================

def resolve_tickers(
    question: str,
) -> list[str]:
    """
    Detect AlphaLens companies mentioned in a question.

    Returns
    -------
    list[str]

    Examples
    --------

    Question:

        "What cybersecurity risks does NVIDIA face?"

    Returns:

        ["NVDA"]


    Question:

        "Compare Microsoft and NVIDIA's AI risks."

    Returns:

        [
            "MSFT",
            "NVDA"
        ]


    Question:

        "What are the biggest cybersecurity risks?"

    Returns:

        []

    because no specific company was mentioned.
    """

    question = (
        question.strip()
    )


    if not question:

        return []


    engine = (
        get_database_engine()
    )


    companies = (
        get_companies(
            engine
        )
    )


    alias_map = (
        build_alias_map(
            companies
        )
    )


    # ========================================================
    # Important:
    # Search LONG aliases first.
    # ========================================================
    #
    # Example:
    #
    #     "jp morgan chase"
    #
    # should be checked before:
    #
    #     "jpm"
    #
    # Longer aliases are normally more specific.
    # ========================================================

    aliases = sorted(
        alias_map.keys(),
        key=len,
        reverse=True,
    )


    found_tickers = []


    for alias in aliases:

        if alias_in_question(
            question=question,
            alias=alias,
        ):

            ticker = (
                alias_map[
                    alias
                ]
            )


            # Avoid duplicates.
            #
            # Example:
            #
            # question:
            #
            #     "NVIDIA (NVDA)"
            #
            # matches:
            #
            #     NVIDIA
            #     NVDA
            #
            # but should return only:
            #
            #     ["NVDA"]
            #
            if ticker not in found_tickers:

                found_tickers.append(
                    ticker
                )


    return found_tickers


# ============================================================
# resolve_single_ticker()
# ============================================================

def resolve_single_ticker(
    question: str,
):
    """
    Convenience function for the current AlphaLens RAG flow.

    Returns
    -------
    str | None

    If exactly one company is detected:

        "NVDA"

    If no company is detected:

        None

    If multiple companies are detected:

        None

    Why?
    ----

    Our current retriever supports one ticker filter.

    Multi-company comparison retrieval will be added later.
    """

    tickers = (
        resolve_tickers(
            question
        )
    )


    if len(tickers) == 1:

        return tickers[0]


    return None


# ============================================================
# Development Tests
# ============================================================

if __name__ == "__main__":

    test_questions = [

        "What cybersecurity risks does NVIDIA face?",

        "What does Apple say about competition?",

        "What risks does Google discuss?",

        "What are KO's main business risks?",

        "What risks does Coca-Cola face?",

        "Compare Microsoft and NVIDIA's AI risks.",

        "What are the biggest cybersecurity risks?",
    ]


    for question in test_questions:

        print()

        print(
            f"Question: {question}"
        )

        print(
            f"Detected: "
            f"{resolve_tickers(question)}"
        )