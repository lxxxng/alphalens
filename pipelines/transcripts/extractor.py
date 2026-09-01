"""
AlphaLens - Earnings Call Transcript Extractor

Purpose:
    Downloads earnings call transcripts for the AlphaLens stock
    universe.

Current transcript pipeline:

    AlphaLens tickers
        |
        v
    Build fiscal quarters
        |
        v
    Alpha Vantage transcript API
        |
        v
    Normalize transcript + speaker turns
        |
        v
    loader.py
        |
        v
    PostgreSQL


Why transcripts?
----------------

SEC filings tell us what a company formally reports.

Earnings calls add a different type of signal:

    - management tone
    - analyst questions
    - forward-looking commentary
    - explanations of quarterly performance
    - risks or opportunities discussed verbally

That makes transcripts useful for later:

    - sentiment analysis
    - RAG search
    - event studies around earnings dates
    - comparing management language with SEC risk disclosures


Provider
--------

The first implementation uses Alpha Vantage:

    function=EARNINGS_CALL_TRANSCRIPT
    symbol=AAPL
    quarter=2024Q1

The endpoint is quarter-specific.

Therefore AlphaLens needs to explicitly request:

    AAPL 2026Q4
    AAPL 2026Q3
    AAPL 2026Q2
    ...

for each ticker in the stock universe.
"""

import os
import time
from urllib.parse import urlencode

import pandas as pd
import requests

from dotenv import load_dotenv

from pipelines.market_data.extractor import TICKERS


# ============================================================
# Environment
# ============================================================
#
# Reads variables from:
#
#     .env
#
# Required:
#
#     ALPHA_VANTAGE_API_KEY
#
# Optional:
#
#     TRANSCRIPT_LOOKBACK_YEARS
#     TRANSCRIPT_REQUEST_SECONDS
#
# Keeping these in .env prevents API keys and run-time tuning from
# being hard-coded into source code.
# ============================================================

load_dotenv()


# ============================================================
# Provider Configuration
# ============================================================

ALPHA_VANTAGE_BASE_URL = "https://www.alphavantage.co/query"
ALPHA_VANTAGE_FUNCTION = "EARNINGS_CALL_TRANSCRIPT"

# Default historical window.
#
# Five years lines up with the existing market-data and SEC windows,
# which makes future joins cleaner:
#
#     transcript quarter
#          +
#     nearby price action
#          +
#     matching SEC filings
TRANSCRIPT_LOOKBACK_YEARS = 5

# Alpha Vantage free-tier limits can be tight.
#
# A transcript backfill can create many requests:
#
#     20 tickers * 20 quarters = 400 requests
#
# The default pause is conservative so the pipeline behaves politely
# and is less likely to hit rate limits.
DEFAULT_REQUEST_SECONDS = 12.0

# A response shorter than this is unlikely to be a real transcript.
#
# Providers may return short error payloads, empty objects, or messages
# saying data is unavailable. This keeps those from entering the DB as
# if they were valid transcript text.
MIN_TRANSCRIPT_CHARS = 500


def get_alpha_vantage_api_key() -> str:
    """
    Read the Alpha Vantage API key from the environment.

    ALPHA_VANTAGE_API_KEY is the preferred variable name.
    ALPHAVANTAGE_API_KEY is accepted for compatibility with older
    snippets found online.
    """

    # Prefer the clearer env var name used by this project.
    #
    # Also accept ALPHAVANTAGE_API_KEY because many examples online use
    # that spelling. Supporting both makes setup a little friendlier
    # without exposing secrets in code.
    api_key = (
        os.getenv("ALPHA_VANTAGE_API_KEY")
        or os.getenv("ALPHAVANTAGE_API_KEY")
    )

    if not api_key:
        raise ValueError(
            "ALPHA_VANTAGE_API_KEY was not found in .env."
        )

    return api_key


def get_request_pause_seconds() -> float:
    """
    Return the delay between provider requests.

    The default is intentionally conservative because transcript
    backfills can involve many ticker/quarter combinations.
    """

    # Let the user tune the pause without editing Python.
    #
    # Example:
    #
    #     TRANSCRIPT_REQUEST_SECONDS=15
    #
    # for a slower but safer historical backfill.
    raw_value = os.getenv(
        "TRANSCRIPT_REQUEST_SECONDS",
        str(DEFAULT_REQUEST_SECONDS),
    )

    return float(raw_value)


def get_lookback_years() -> int:
    """
    Return the transcript lookback window.
    """

    # Keep the lookback configurable because transcript providers often
    # vary by plan. A user can start with 1 year for testing, then expand
    # to 5 years when the ingestion flow is proven.
    raw_value = os.getenv(
        "TRANSCRIPT_LOOKBACK_YEARS",
        str(TRANSCRIPT_LOOKBACK_YEARS),
    )

    lookback_years = int(raw_value)

    # A zero/negative lookback would produce confusing empty work.
    if lookback_years < 1:
        raise ValueError(
            "TRANSCRIPT_LOOKBACK_YEARS must be at least 1."
        )

    return lookback_years


def build_fiscal_periods(
    lookback_years: int | None = None,
    current_date: pd.Timestamp | None = None,
) -> list[str]:
    """
    Build descending fiscal periods such as 2026Q3, 2026Q2.

    We request Q1-Q4 for each year because companies have different
    fiscal calendars. The current calendar year is included so newly
    reported transcripts can be captured as they become available.
    """

    # If the caller does not pass a lookback, use the project default or
    # the value from .env.
    if lookback_years is None:
        lookback_years = get_lookback_years()

    # current_date exists mostly for testing.
    #
    # It lets us check that:
    #
    #     2026-09-01 + lookback 1
    #
    # produces:
    #
    #     2026Q4, 2026Q3, 2026Q2, 2026Q1
    if current_date is None:
        current_date = pd.Timestamp.now().normalize()
    else:
        current_date = pd.Timestamp(current_date).normalize()

    current_year = int(current_date.year)
    start_year = current_year - lookback_years + 1

    periods = []

    # Request newest periods first.
    #
    # This is useful when doing a slow backfill because the most recent
    # transcripts are usually the most valuable to inspect first.
    for year in range(current_year, start_year - 1, -1):
        for quarter in range(4, 0, -1):
            periods.append(
                f"{year}Q{quarter}"
            )

    return periods


def build_request_params(
    ticker: str,
    fiscal_period: str,
    api_key: str,
) -> dict:
    """
    Build Alpha Vantage request parameters.
    """

    # Keep the real API key only in request parameters.
    #
    # We do NOT store these exact params as source metadata because that
    # would put the secret key into PostgreSQL.
    return {
        "function": ALPHA_VANTAGE_FUNCTION,
        "symbol": ticker,
        "quarter": fiscal_period,
        "apikey": api_key,
    }


def build_source_url(
    ticker: str,
    fiscal_period: str,
) -> str:
    """
    Build a non-secret source URL for storage and citations.
    """

    # This is the URL shape we can safely store for traceability.
    #
    # The important source information is still visible:
    #
    #     function, symbol, quarter
    #
    # but the actual API key is replaced.
    params = {
        "function": ALPHA_VANTAGE_FUNCTION,
        "symbol": ticker,
        "quarter": fiscal_period,
        "apikey": "REDACTED",
    }

    return (
        f"{ALPHA_VANTAGE_BASE_URL}?"
        f"{urlencode(params)}"
    )


def parse_fiscal_period(
    fiscal_period: str,
) -> tuple[int, int]:
    """
    Convert 2024Q1 into year=2024, quarter=1.
    """

    # Expected format:
    #
    #     YYYYQM
    #
    # Example:
    #
    #     2024Q1
    if len(fiscal_period) != 6 or "Q" not in fiscal_period:
        raise ValueError(
            f"Invalid fiscal period: {fiscal_period}"
        )

    year_text, quarter_text = fiscal_period.upper().split("Q")

    fiscal_year = int(year_text)
    fiscal_quarter = int(quarter_text)

    if fiscal_quarter not in [1, 2, 3, 4]:
        raise ValueError(
            f"Invalid fiscal quarter: {fiscal_period}"
        )

    return fiscal_year, fiscal_quarter


def parse_optional_float(value):
    """
    Convert provider sentiment values to float when possible.
    """

    # Some provider fields may be strings such as:
    #
    #     "0.42"
    #
    # while missing values may be None or empty strings.
    if value in [None, ""]:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_optional_date(value):
    """
    Convert provider date strings to normal date objects.
    """

    if not value:
        return None

    # pandas handles many practical date formats:
    #
    #     2026-08-01
    #     2026-08-01 17:00:00
    #
    # errors="coerce" prevents a malformed provider value from crashing
    # the whole pipeline.
    parsed = pd.to_datetime(
        value,
        errors="coerce",
    )

    if pd.isna(parsed):
        return None

    return parsed.date()


def extract_turns_from_payload(
    payload,
) -> list[dict]:
    """
    Normalize provider speaker-turn records.

    Alpha Vantage documents the endpoint but does not expose a rigid
    schema in the docs, so this function accepts several common field
    names. That keeps the loader resilient if the provider includes
    extra metadata.
    """

    # Some APIs return:
    #
    #     [{speaker: ..., content: ...}, ...]
    #
    # Others return:
    #
    #     {"transcript": [{...}, {...}]}
    #
    # Support both so the rest of the pipeline receives one normalized
    # speaker-turn structure.
    if isinstance(payload, list):
        raw_turns = payload
    elif isinstance(payload, dict):
        transcript_value = payload.get("transcript")

        if isinstance(transcript_value, list):
            raw_turns = transcript_value
        else:
            raw_turns = []
    else:
        raw_turns = []

    turns = []

    for turn_index, raw_turn in enumerate(raw_turns):

        # Ignore unexpected list items instead of failing the entire
        # transcript. The raw payload is still stored for later debugging.
        if not isinstance(raw_turn, dict):
            continue

        # Providers use different names for the spoken text.
        #
        # Normalize the common variants into one AlphaLens field:
        #
        #     content
        content = (
            raw_turn.get("content")
            or raw_turn.get("speech")
            or raw_turn.get("text")
            or raw_turn.get("statement")
        )

        if not content:
            continue

        content = str(content).strip()

        if not content:
            continue

        # Keep speaker metadata because it allows richer research later:
        #
        #     CEO language vs CFO language
        #     management section vs analyst Q&A
        #     named-speaker retrieval
        speaker_name = (
            raw_turn.get("speaker")
            or raw_turn.get("name")
            or raw_turn.get("speaker_name")
        )

        speaker_title = (
            raw_turn.get("title")
            or raw_turn.get("position")
            or raw_turn.get("description")
            or raw_turn.get("speaker_title")
        )

        speaker_role = (
            raw_turn.get("role")
            or raw_turn.get("speaker_type")
            or raw_turn.get("session")
        )

        # Alpha Vantage describes turn-by-turn sentiment signals. Store
        # both label-style and numeric-style fields when available.
        sentiment_label = (
            raw_turn.get("sentiment")
            or raw_turn.get("sentiment_label")
            or raw_turn.get("tone")
        )

        sentiment_score = parse_optional_float(
            raw_turn.get("sentiment_score")
            or raw_turn.get("sentimentScore")
            or raw_turn.get("sentiment")
        )

        turns.append(
            {
                "turn_index": turn_index,
                "speaker_name": speaker_name,
                "speaker_title": speaker_title,
                "speaker_role": speaker_role,
                "content": content,
                "sentiment_label": sentiment_label,
                "sentiment_score": sentiment_score,
                "char_count": len(content),
            }
        )

    return turns


def extract_content_from_payload(
    payload,
    turns: list[dict],
) -> str:
    """
    Build one full transcript string.
    """

    # Prefer a provider-supplied full transcript if it exists.
    #
    # That avoids accidentally losing formatting or content that was not
    # included in the per-turn array.
    if isinstance(payload, dict):
        content = (
            payload.get("content")
            or payload.get("text")
        )

        if isinstance(content, str) and content.strip():
            return content.strip()

        transcript_value = payload.get("transcript")

        if isinstance(transcript_value, str) and transcript_value.strip():
            return transcript_value.strip()

    # If the provider only returns speaker turns, rebuild a readable full
    # transcript. This gives downstream systems one complete content field
    # plus the normalized turn table.
    if turns:
        lines = []

        for turn in turns:
            speaker = turn.get("speaker_name") or "Unknown Speaker"
            title = turn.get("speaker_title")

            if title:
                lines.append(
                    f"{speaker} - {title}"
                )
            else:
                lines.append(
                    speaker
                )

            lines.append(
                turn["content"]
            )

        return "\n\n".join(lines).strip()

    return ""


def normalize_transcript_payload(
    ticker: str,
    fiscal_period: str,
    payload,
    source_url: str,
) -> dict | None:
    """
    Convert an Alpha Vantage response into one database-ready record.
    """

    # Convert 2026Q3 into separate integer columns.
    #
    # Separate year/quarter fields make SQL filtering easier than parsing
    # a string every time.
    fiscal_year, fiscal_quarter = parse_fiscal_period(
        fiscal_period
    )

    # Alpha Vantage may return JSON messages for quota limits, invalid
    # keys, or missing entitlements. Those are not transcripts, so fail
    # this ticker/quarter and let the outer loop continue.
    if isinstance(payload, dict):
        for error_key in [
            "Error Message",
            "Information",
            "Note",
        ]:
            if error_key in payload:
                raise ValueError(
                    str(payload[error_key])
                )

    turns = extract_turns_from_payload(
        payload
    )

    content = extract_content_from_payload(
        payload=payload,
        turns=turns,
    )

    # A tiny "transcript" is usually a provider message or unavailable
    # quarter, not useful financial text.
    if len(content) < MIN_TRANSCRIPT_CHARS:
        return None

    call_date = None
    title = None

    if isinstance(payload, dict):
        call_date = parse_optional_date(
            payload.get("date")
            or payload.get("call_date")
            or payload.get("time")
        )

        title = (
            payload.get("title")
            or payload.get("event_title")
        )

    return {
        "ticker": ticker,
        "fiscal_year": fiscal_year,
        "fiscal_quarter": fiscal_quarter,
        "fiscal_period": fiscal_period,
        "call_date": call_date,
        "title": title,
        "source_provider": "alpha_vantage",
        "source_url": source_url,
        "content": content,
        "char_count": len(content),
        "turn_count": len(turns),
        "raw_payload": payload,
        "turns": turns,
    }


def fetch_transcript(
    session: requests.Session,
    ticker: str,
    fiscal_period: str,
    api_key: str,
) -> dict | None:
    """
    Request and normalize one ticker/period transcript.
    """

    # Build real request params with the real API key.
    request_params = build_request_params(
        ticker=ticker,
        fiscal_period=fiscal_period,
        api_key=api_key,
    )

    # requests will encode params safely and avoids us manually building a
    # URL containing the API key.
    response = session.get(
        ALPHA_VANTAGE_BASE_URL,
        params=request_params,
        timeout=60,
    )

    response.raise_for_status()

    # Convert the provider's JSON into Python data structures.
    payload = response.json()

    return normalize_transcript_payload(
        ticker=ticker,
        fiscal_period=fiscal_period,
        payload=payload,
        # Store only the redacted URL.
        source_url=build_source_url(
            ticker=ticker,
            fiscal_period=fiscal_period,
        ),
    )


def extract_earnings_transcripts(
    tickers: list[str] | None = None,
    fiscal_periods: list[str] | None = None,
) -> pd.DataFrame:
    """
    Download transcript records for the configured universe.
    """

    # Fail early if the key is missing. This is clearer than making many
    # HTTP calls that all fail.
    api_key = get_alpha_vantage_api_key()

    # Default to the same company universe used by market data and SEC.
    #
    # That makes cross-dataset joins possible later:
    #
    #     market_prices.ticker
    #     filings.ticker
    #     earnings_transcripts.ticker
    if tickers is None:
        tickers = TICKERS

    # Default to all quarters in the configured lookback window.
    if fiscal_periods is None:
        fiscal_periods = build_fiscal_periods()

    request_pause_seconds = get_request_pause_seconds()

    # Reuse one HTTP session instead of creating a new connection for
    # every ticker/quarter pair.
    session = requests.Session()

    records = []

    total_requests = len(tickers) * len(fiscal_periods)
    request_number = 0

    print(
        f"\nTranscript requests planned: {total_requests}"
    )

    for ticker in tickers:
        for fiscal_period in fiscal_periods:
            request_number += 1

            print(
                f"[{request_number}/{total_requests}] "
                f"{ticker} {fiscal_period}"
            )

            try:
                transcript = fetch_transcript(
                    session=session,
                    ticker=ticker,
                    fiscal_period=fiscal_period,
                    api_key=api_key,
                )

            except (
                requests.RequestException,
                ValueError,
            ) as error:
                # One missing or failed transcript should not stop the
                # entire universe. This mirrors the SEC extractor's
                # resilience around one-company failures.
                print(
                    f"    [SKIPPED] {ticker} {fiscal_period}: "
                    f"{error}"
                )

                if request_number < total_requests:
                    time.sleep(request_pause_seconds)

                continue

            if transcript is None:
                print(
                    f"    [MISSING] No transcript content"
                )
            else:
                records.append(transcript)

                print(
                    f"    [OK] {transcript['char_count']:,} chars "
                    f"| {transcript['turn_count']} turns"
                )

            # Sleep after each request except the final one. This keeps the
            # provider interaction polite during larger backfills.
            if request_number < total_requests:
                time.sleep(request_pause_seconds)

    return pd.DataFrame(
        records
    )


if __name__ == "__main__":
    transcripts = extract_earnings_transcripts()

    print("\n========================================")
    print("TRANSCRIPT EXTRACTION COMPLETE")
    print("========================================")

    print(
        f"Transcripts found: {len(transcripts)}"
    )

    if not transcripts.empty:
        print(
            transcripts[
                [
                    "ticker",
                    "fiscal_period",
                    "char_count",
                    "turn_count",
                ]
            ].head(20).to_string(index=False)
        )
