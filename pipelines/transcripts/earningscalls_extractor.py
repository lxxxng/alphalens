"""
AlphaLens - EarningsCalls.dev Transcript Extractor

Purpose
-------
Download full earnings-call transcripts and speaker segments from the
official EarningsCalls.dev API, then normalize them for loader.py.


Provider request flow
---------------------

    company ticker
        |
        v
    company history
        |
        v
    keep earnings events in the lookback window
        |
        v
    call details (exact fiscal year and quarter)
        |
        +--> full transcript
        |
        +--> paginated speaker segments
        |
        v
    normalized Pandas DataFrame


Resume behavior
---------------

The runner loads one ticker at a time. Before extraction it reads source
URLs already stored in PostgreSQL. Because each source URL contains the
provider's stable call ID, completed calls are skipped on the next run.
"""

import os
from datetime import date
from pathlib import Path
import time

import pandas as pd
import requests

from dotenv import load_dotenv

from pipelines.market_data.extractor import TICKERS


# ============================================================
# Environment
# ============================================================

load_dotenv(
    Path(__file__).resolve().parents[2] / ".env"
)


# ============================================================
# Provider Configuration
# ============================================================

SOURCE_PROVIDER = "earningscalls_dev"

BASE_URL = (
    "https://earningscalls.dev/api/v1"
)

DEFAULT_LOOKBACK_YEARS = 5

# Pro allows 20 requests per minute. A delay slightly above three seconds
# keeps this single-process pipeline under that advertised ceiling.
DEFAULT_REQUEST_SECONDS = 3.1

REQUEST_TIMEOUT_SECONDS = 60

# Speaker pages are usually much smaller than this. A high page size keeps
# most calls to a single speaker request while retaining pagination support.
SPEAKER_PAGE_SIZE = 100

MIN_TRANSCRIPT_CHARS = 500


# ============================================================
# Parsing Helpers
# ============================================================

def parse_optional_int(
    value,
) -> int | None:
    """
    Convert provider numeric fields while tolerating missing values.
    """

    if value is None:
        return None

    try:
        return int(
            value
        )
    except (TypeError, ValueError):
        return None


# ============================================================
# Environment Helpers
# ============================================================

def get_api_key() -> str:
    """
    Read the direct EarningsCalls.dev API key from .env.
    """

    api_key = os.getenv(
        "EARNINGSCALLS_API_KEY",
        "",
    ).strip()

    if not api_key:
        raise ValueError(
            "EARNINGSCALLS_API_KEY was not found in .env."
        )

    return api_key


def get_lookback_years() -> int:
    """
    Return the configured number of fiscal years to download.
    """

    value = int(
        os.getenv(
            "TRANSCRIPT_LOOKBACK_YEARS",
            str(DEFAULT_LOOKBACK_YEARS),
        )
    )

    if value < 1:
        raise ValueError(
            "TRANSCRIPT_LOOKBACK_YEARS must be at least 1."
        )

    return value


def get_request_pause_seconds() -> float:
    """
    Return the pause between official provider requests.
    """

    value = float(
        os.getenv(
            "EARNINGSCALLS_REQUEST_SECONDS",
            str(DEFAULT_REQUEST_SECONDS),
        )
    )

    if value < 0:
        raise ValueError(
            "EARNINGSCALLS_REQUEST_SECONDS cannot be negative."
        )

    return value


# ============================================================
# HTTP Helpers
# ============================================================

def create_session(
    api_key: str,
) -> requests.Session:
    """
    Create one reusable authenticated HTTP session.
    """

    session = requests.Session()

    session.headers.update(
        {
            "X-API-Key": api_key,
        }
    )

    return session


def fetch_json(
    session: requests.Session,
    path: str,
    request_state: dict,
    params: dict | None = None,
) -> dict:
    """
    Fetch and validate one provider JSON response.

    request_state is shared across a ticker extraction so every call is
    paced, including calls made by different helper functions.
    """

    if request_state["count"] > 0:
        time.sleep(
            request_state["pause_seconds"]
        )

    request_state["count"] += 1

    response = session.get(
        f"{BASE_URL}{path}",
        params=params,
        timeout=REQUEST_TIMEOUT_SECONDS,
        allow_redirects=False,
    )

    if response.status_code != 200:
        # Do not include request headers in errors because they contain the
        # API key. The response body is also omitted because providers can
        # change what they echo in an error.
        raise RuntimeError(
            f"EarningsCalls.dev returned HTTP {response.status_code} "
            f"for {path}."
        )

    payload = response.json()

    if not isinstance(payload, dict):
        raise RuntimeError(
            f"Expected a JSON object from {path}."
        )

    return payload


def get_data_object(
    payload: dict,
    context: str,
) -> dict:
    """
    Return a provider response's data object with a clear schema error.
    """

    data = payload.get("data")

    if not isinstance(data, dict):
        raise RuntimeError(
            f"{context} response is missing its data object."
        )

    return data


# ============================================================
# Provider URLs
# ============================================================

def build_source_url(
    earnings_call_id: int,
) -> str:
    """
    Build the non-secret transcript URL stored in PostgreSQL.
    """

    return (
        f"{BASE_URL}/transcripts/{earnings_call_id}"
    )


# ============================================================
# Company History
# ============================================================

def fetch_company_calls(
    session: requests.Session,
    ticker: str,
    request_state: dict,
) -> list[dict]:
    """
    Return earnings events for one US-listed company.
    """

    payload = fetch_json(
        session=session,
        path=f"/companies/ticker/{ticker}",
        params={"country": "US"},
        request_state=request_state,
    )

    data = get_data_object(
        payload,
        context=f"{ticker} company history",
    )

    calls = data.get(
        "earnings_calls"
    )

    if not isinstance(calls, list):
        raise RuntimeError(
            f"{ticker} company history is missing earnings_calls."
        )

    # Company histories also contain conference presentations, special
    # events, acquisitions, and shareholder meetings. Only quarterly
    # earnings calls belong in the current database table.
    earnings_calls = [
        call
        for call in calls
        if (
            isinstance(call, dict)
            and call.get("event_type") == "earnings"
            and str(call.get("id", "")).isdigit()
        )
    ]

    return earnings_calls


def is_near_lookback_window(
    call: dict,
    earliest_fiscal_year: int,
) -> bool:
    """
    Use event dates to avoid detail calls for clearly old transcripts.

    A fiscal year can differ from its calendar year, so retain a one-year
    buffer. Exact filtering happens after the call-detail response supplies
    fiscal_year.
    """

    event_date_text = str(
        call.get("event_date_time", "")
    )

    try:
        event_year = int(
            event_date_text[:4]
        )
    except ValueError:
        return True

    return event_year >= earliest_fiscal_year - 1


# ============================================================
# Transcript Components
# ============================================================

def fetch_call_details(
    session: requests.Session,
    earnings_call_id: int,
    request_state: dict,
) -> dict:
    """
    Fetch exact fiscal fields and call-level metadata.
    """

    payload = fetch_json(
        session=session,
        path=f"/earnings/{earnings_call_id}",
        request_state=request_state,
    )

    return get_data_object(
        payload,
        context=f"Call {earnings_call_id} details",
    )


def fetch_full_transcript(
    session: requests.Session,
    earnings_call_id: int,
    request_state: dict,
) -> dict:
    """
    Fetch a call's full transcript payload.
    """

    payload = fetch_json(
        session=session,
        path=f"/transcripts/{earnings_call_id}",
        request_state=request_state,
    )

    return get_data_object(
        payload,
        context=f"Call {earnings_call_id} transcript",
    )


def fetch_speaker_segments(
    session: requests.Session,
    earnings_call_id: int,
    request_state: dict,
) -> tuple[list[dict], list[dict]]:
    """
    Fetch every page of speaker segments for one call.

    Returns the detailed segments and the provider's speaker summary.
    """

    page = 1
    total_pages = 1
    segments = []
    speakers_summary = []

    while page <= total_pages:
        payload = fetch_json(
            session=session,
            path=f"/speakers/{earnings_call_id}",
            params={
                "page": page,
                "limit": SPEAKER_PAGE_SIZE,
            },
            request_state=request_state,
        )

        data = get_data_object(
            payload,
            context=(
                f"Call {earnings_call_id} speakers page {page}"
            ),
        )

        page_segments = data.get(
            "segments"
        )

        if not isinstance(page_segments, list):
            raise RuntimeError(
                f"Call {earnings_call_id} speaker response "
                "is missing segments."
            )

        segments.extend(
            segment
            for segment in page_segments
            if isinstance(segment, dict)
        )

        if page == 1:
            raw_summary = data.get(
                "speakers_summary",
                [],
            )

            if isinstance(raw_summary, list):
                speakers_summary = raw_summary

        pagination = payload.get(
            "pagination",
            {},
        )

        if isinstance(pagination, dict):
            total_pages = max(
                1,
                int(pagination.get("total_pages", 1)),
            )

        page += 1

    return segments, speakers_summary


# ============================================================
# Normalization
# ============================================================

def normalize_turns(
    segments: list[dict],
) -> list[dict]:
    """
    Convert provider speaker segments into loader-ready turns.
    """

    turns = []

    ordered_segments = sorted(
        segments,
        key=lambda segment: int(
            segment.get("component_order", 0)
        ),
    )

    for segment in ordered_segments:
        content = str(
            segment.get("text_content", "")
        ).strip()

        if not content:
            continue

        turns.append(
            {
                # Use a dense local index even if provider component values
                # contain gaps.
                "turn_index": len(turns),
                "speaker_name": segment.get("speaker_name"),
                "speaker_title": None,
                "speaker_role": segment.get("speaker_type"),
                "content": content,
                "sentiment_label": None,
                "sentiment_score": None,
                "char_count": len(content),
            }
        )

    return turns


def normalize_transcript(
    ticker: str,
    earnings_call_id: int,
    details: dict,
    transcript: dict,
    segments: list[dict],
    speakers_summary: list[dict],
) -> dict | None:
    """
    Build one database-ready transcript record.
    """

    fiscal_year = parse_optional_int(
        details.get("fiscal_year")
    )

    fiscal_quarter = parse_optional_int(
        details.get("fiscal_quarter")
    )

    if fiscal_year is None or fiscal_quarter is None:
        print(
            f"    [SKIP] Call {earnings_call_id} is missing "
            "fiscal year or quarter"
        )
        return None

    if fiscal_quarter not in [1, 2, 3, 4]:
        print(
            f"    [SKIP] Call {earnings_call_id} has invalid "
            f"fiscal quarter {fiscal_quarter}"
        )
        return None

    content = str(
        transcript.get("full_transcript_text", "")
    ).strip()

    if len(content) < MIN_TRANSCRIPT_CHARS:
        return None

    event_date = pd.to_datetime(
        details.get("event_date_time"),
        errors="coerce",
        utc=True,
    )

    call_date = (
        None
        if pd.isna(event_date)
        else event_date.date()
    )

    turns = normalize_turns(
        segments
    )

    return {
        "ticker": ticker,
        "fiscal_year": fiscal_year,
        "fiscal_quarter": fiscal_quarter,
        "fiscal_period": f"{fiscal_year}Q{fiscal_quarter}",
        "call_date": call_date,
        "title": details.get("transcript_title"),
        "source_provider": SOURCE_PROVIDER,
        "source_url": build_source_url(earnings_call_id),
        "content": content,
        "char_count": len(content),
        "turn_count": len(turns),
        "raw_payload": {
            "call_details": details,
            "transcript": transcript,
            "speaker_segments": segments,
            "speakers_summary": speakers_summary,
        },
        "turns": turns,
    }


def get_call_fiscal_period(
    details: dict,
    earnings_call_id: int,
) -> tuple[int, int] | None:
    """
    Return fiscal year and quarter when provider metadata is usable.
    """

    fiscal_year = parse_optional_int(
        details.get("fiscal_year")
    )

    fiscal_quarter = parse_optional_int(
        details.get("fiscal_quarter")
    )

    if fiscal_year is None or fiscal_quarter is None:
        print(
            f"  [SKIP] Call {earnings_call_id} is missing "
            "fiscal year or quarter"
        )
        return None

    if fiscal_quarter not in [1, 2, 3, 4]:
        print(
            f"  [SKIP] Call {earnings_call_id} has invalid "
            f"fiscal quarter {fiscal_quarter}"
        )
        return None

    return fiscal_year, fiscal_quarter


# ============================================================
# Public Extraction Function
# ============================================================

def extract_earnings_transcripts(
    tickers: list[str] | None = None,
    loaded_source_urls: set[str] | None = None,
    max_transcripts_per_ticker: int | None = None,
    current_date: date | None = None,
) -> pd.DataFrame:
    """
    Download normalized transcripts for the configured universe.

    max_transcripts_per_ticker is mainly useful for a small integration
    test before running a complete backfill.
    """

    if tickers is None:
        tickers = TICKERS

    if loaded_source_urls is None:
        loaded_source_urls = set()

    if current_date is None:
        current_date = date.today()

    if (
        max_transcripts_per_ticker is not None
        and max_transcripts_per_ticker < 1
    ):
        raise ValueError(
            "max_transcripts_per_ticker must be at least 1."
        )

    lookback_years = get_lookback_years()
    earliest_fiscal_year = (
        current_date.year - lookback_years + 1
    )

    api_key = get_api_key()
    request_state = {
        "count": 0,
        "pause_seconds": get_request_pause_seconds(),
    }

    records = []

    with create_session(api_key) as session:
        for ticker in tickers:
            ticker = ticker.upper()

            print(
                f"\nFetching EarningsCalls.dev history for {ticker}"
            )

            calls = fetch_company_calls(
                session=session,
                ticker=ticker,
                request_state=request_state,
            )

            candidates = [
                call
                for call in calls
                if is_near_lookback_window(
                    call=call,
                    earliest_fiscal_year=earliest_fiscal_year,
                )
            ]

            extracted_for_ticker = 0

            for call in candidates:
                earnings_call_id = int(
                    call["id"]
                )

                source_url = build_source_url(
                    earnings_call_id
                )

                if source_url in loaded_source_urls:
                    print(
                        f"  [SKIP] Call {earnings_call_id} already loaded"
                    )
                    continue

                details = fetch_call_details(
                    session=session,
                    earnings_call_id=earnings_call_id,
                    request_state=request_state,
                )

                fiscal_period = get_call_fiscal_period(
                    details=details,
                    earnings_call_id=earnings_call_id,
                )

                if fiscal_period is None:
                    continue

                fiscal_year, fiscal_quarter = fiscal_period

                if not (
                    earliest_fiscal_year
                    <= fiscal_year
                    <= current_date.year
                ):
                    continue

                print(
                    f"  Downloading {fiscal_year}Q{fiscal_quarter} "
                    f"(call {earnings_call_id})"
                )

                transcript = fetch_full_transcript(
                    session=session,
                    earnings_call_id=earnings_call_id,
                    request_state=request_state,
                )

                segments, speakers_summary = fetch_speaker_segments(
                    session=session,
                    earnings_call_id=earnings_call_id,
                    request_state=request_state,
                )

                record = normalize_transcript(
                    ticker=ticker,
                    earnings_call_id=earnings_call_id,
                    details=details,
                    transcript=transcript,
                    segments=segments,
                    speakers_summary=speakers_summary,
                )

                if record is None:
                    print(
                        "    [MISSING] Full transcript text was too short"
                    )
                    continue

                records.append(
                    record
                )

                loaded_source_urls.add(
                    source_url
                )

                extracted_for_ticker += 1

                print(
                    f"    [OK] {record['char_count']:,} chars "
                    f"| {record['turn_count']} turns"
                )

                if (
                    max_transcripts_per_ticker is not None
                    and extracted_for_ticker
                    >= max_transcripts_per_ticker
                ):
                    break

    print(
        f"\nProvider requests made: {request_state['count']}"
    )

    return pd.DataFrame(
        records
    )
