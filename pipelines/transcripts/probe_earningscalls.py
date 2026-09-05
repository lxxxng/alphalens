"""
AlphaLens - EarningsCalls.dev Free-Tier Probe

Purpose
-------
Test the EarningsCalls.dev API through RapidAPI before paying for a
full-transcript plan.

This script checks two things for a small group of companies:

    1. Does the company-history endpoint return earnings calls?
    2. Does the transcript endpoint return a preview for the latest call?


Request flow
------------

    ticker
        |
        v
    GET /companies/ticker/{ticker}
        |
        v
    latest earnings-call ID
        |
        v
    GET /transcripts/{earnings_call_id}
        |
        v
    print a small response sample


Safety limits
-------------

The free plan has a small monthly request allowance, so this diagnostic:

    - accepts no more than three tickers
    - makes no more than two requests per ticker
    - waits between requests
    - does not retry failed requests automatically
    - never writes API responses to the database
    - never prints the API key


Setup
-----

Add the following variable to the project .env file:

    RAPIDAPI_KEY=your_key_here

Then run:

    python -m pipelines.transcripts.probe_earningscalls

Optional custom tickers:

    python -m pipelines.transcripts.probe_earningscalls --tickers AAPL MSFT
"""

import argparse
import json
import os
from pathlib import Path
import time

from dotenv import load_dotenv
import requests


# ============================================================
# Provider Configuration
# ============================================================

RAPIDAPI_HOST = (
    "earnings-call-transcripts1.p.rapidapi.com"
)

BASE_URL = (
    f"https://{RAPIDAPI_HOST}/api/v1"
)

FREE_PLAN_URL = (
    "https://rapidapi.com/earningscallsdev/api/"
    "earnings-call-transcripts1"
)


# ============================================================
# Probe Configuration
# ============================================================

# These large companies make useful coverage checks because each has
# several years of quarterly calls.
DEFAULT_TICKERS = [
    "AAPL",
    "MSFT",
    "NVDA",
]

# One company lookup plus one transcript-preview lookup per ticker.
REQUESTS_PER_TICKER = 2

# Keep accidental command-line input from consuming the free quota.
MAX_TICKERS = 3

# A conservative pause keeps the requests comfortably below typical
# provider rate limits.
REQUEST_PAUSE_SECONDS = 12

# Do not let a slow provider request hang forever.
REQUEST_TIMEOUT_SECONDS = 30

# We only need enough JSON to inspect the response shape and determine
# whether useful transcript text is present.
MAX_DISPLAY_CHARS = 2000


# ============================================================
# get_tickers()
# ============================================================

def get_tickers() -> list[str]:
    """
    Read and validate optional ticker symbols from the command line.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Test EarningsCalls.dev's free RapidAPI tier "
            "without loading PostgreSQL."
        )
    )

    parser.add_argument(
        "--tickers",
        nargs="+",
        default=DEFAULT_TICKERS,
        help=(
            "One to three ticker symbols. "
            "Default: AAPL MSFT NVDA"
        ),
    )

    arguments = parser.parse_args()

    # Normalize symbols so "aapl" and "AAPL" are treated as the same
    # company. dict.fromkeys removes duplicates while preserving order.
    tickers = list(
        dict.fromkeys(
            ticker.upper()
            for ticker in arguments.tickers
        )
    )

    if len(tickers) > MAX_TICKERS:
        parser.error(
            f"Use no more than {MAX_TICKERS} tickers."
        )

    # This probe targets simple US ticker symbols. Reject punctuation so
    # user input cannot unexpectedly alter the request path.
    if any(
        not ticker.isalnum()
        for ticker in tickers
    ):
        parser.error(
            "Use simple ticker symbols, such as AAPL MSFT NVDA."
        )

    return tickers


# ============================================================
# get_rapidapi_key()
# ============================================================

def get_rapidapi_key() -> str | None:
    """
    Load the RapidAPI key from the project .env file.

    Returning None lets main() print friendly setup instructions instead
    of raising a stack trace when the key has not been configured yet.
    """

    project_root = (
        Path(__file__).resolve().parents[2]
    )

    load_dotenv(
        project_root / ".env"
    )

    api_key = os.getenv(
        "RAPIDAPI_KEY",
        "",
    ).strip()

    # Treat example placeholders as missing credentials too.
    if (
        not api_key
        or api_key.lower().startswith(
            ("your_", "change_me")
        )
    ):
        return None

    return api_key


# ============================================================
# create_session()
# ============================================================

def create_session(
    api_key: str,
) -> requests.Session:
    """
    Create one reusable HTTP session with RapidAPI authentication.
    """

    session = requests.Session()

    session.headers.update(
        {
            "X-RapidAPI-Key": api_key,
            "X-RapidAPI-Host": RAPIDAPI_HOST,
        }
    )

    return session


# ============================================================
# fetch_json()
# ============================================================

def fetch_json(
    session: requests.Session,
    path: str,
    request_state: dict,
    params: dict | None = None,
) -> dict:
    """
    Fetch one JSON response and update the shared request count.

    A small mutable state dictionary keeps the count accurate even when a
    request fails and raises an exception before returning to the caller.
    """

    # Pause before every request except the first one.
    if request_state["count"] > 0:
        time.sleep(
            REQUEST_PAUSE_SECONDS
        )

    request_state["count"] += 1
    started_at = time.monotonic()

    response = session.get(
        f"{BASE_URL}{path}",
        params=params,
        timeout=REQUEST_TIMEOUT_SECONDS,
        allow_redirects=False,
    )

    elapsed_seconds = (
        time.monotonic() - started_at
    )

    print(
        f"  HTTP {response.status_code}; "
        f"{elapsed_seconds:.2f}s"
    )

    if response.status_code != 200:
        # Keep the error generic. Printing request headers or raw request
        # exceptions could accidentally expose credentials.
        raise RuntimeError(
            f"Stopped on HTTP {response.status_code}. "
            "Check the RapidAPI key, free-plan activation, "
            "and remaining quota."
        )

    payload = response.json()

    if not isinstance(payload, dict):
        raise RuntimeError(
            "Unexpected API response: expected a JSON object."
        )

    return payload


# ============================================================
# extract_company_history()
# ============================================================

def extract_company_history(
    payload: dict,
) -> tuple[dict, list[dict]]:
    """
    Validate a company lookup and return its company data and calls.
    """

    data = payload.get(
        "data"
    )

    if not isinstance(data, dict):
        raise RuntimeError(
            "Unexpected company response: missing data object."
        )

    calls = data.get(
        "earnings_calls"
    )

    if not isinstance(calls, list):
        raise RuntimeError(
            "Unexpected company response: "
            "missing data.earnings_calls."
        )

    # Ignore malformed rows while keeping every valid call dictionary.
    valid_calls = [
        call
        for call in calls
        if isinstance(call, dict)
    ]

    return data, valid_calls


# ============================================================
# print_company_summary()
# ============================================================

def print_company_summary(
    data: dict,
    calls: list[dict],
) -> None:
    """
    Print the company name, number of calls, and returned date range.
    """

    print(
        f"  Company: "
        f"{data.get('company_name', '(not supplied)')}"
    )

    print(
        f"  Calls returned: {len(calls)} "
        "(coverage is not guaranteed complete)"
    )

    dates = sorted(
        str(call["event_date_time"])[:10]
        for call in calls
        if call.get("event_date_time")
    )

    if dates:
        print(
            f"  Call dates returned: "
            f"{dates[0]} to {dates[-1]}"
        )


# ============================================================
# get_latest_call()
# ============================================================

def get_latest_call(
    calls: list[dict],
) -> dict | None:
    """
    Return the first call with a usable numeric ID.

    The company endpoint documents its earnings_calls array as newest
    first, so the first valid item represents the latest available call.
    """

    for call in calls:
        call_id = str(
            call.get("id", "")
        )

        if call_id.isdigit():
            return call

    return None


# ============================================================
# print_preview_sample()
# ============================================================

def print_preview_sample(
    preview: dict,
    api_key: str,
) -> None:
    """
    Print a bounded, redacted sample of a transcript response.
    """

    if not preview.get("data"):
        raise RuntimeError(
            "Preview response is empty or has an unexpected format."
        )

    sample = json.dumps(
        preview,
        ensure_ascii=True,
        indent=2,
    )

    # The API should never echo credentials, but redact defensively before
    # printing any provider response.
    sample = sample.replace(
        api_key,
        "REDACTED",
    )

    print(
        sample[:MAX_DISPLAY_CHARS]
    )

    if len(sample) > MAX_DISPLAY_CHARS:
        print(
            f"  [Display limited to "
            f"{MAX_DISPLAY_CHARS:,} characters]"
        )


# ============================================================
# probe_ticker()
# ============================================================

def probe_ticker(
    session: requests.Session,
    ticker: str,
    api_key: str,
    request_state: dict,
) -> bool:
    """
    Check company history and the latest transcript for one ticker.

    Return whether the ticker produced a call ID to test.
    """

    print(
        f"\n{ticker}: company history (US listings)"
    )

    history = fetch_json(
        session=session,
        path=f"/companies/ticker/{ticker}",
        params={"country": "US"},
        request_state=request_state,
    )

    data, calls = extract_company_history(
        history
    )

    print_company_summary(
        data=data,
        calls=calls,
    )

    latest_call = get_latest_call(
        calls
    )

    if latest_call is None:
        print(
            "  No call IDs available to test a preview."
        )

        return False

    call_id = str(
        latest_call["id"]
    )

    print(
        f"  Preview for call {call_id}: "
        f"{latest_call.get('transcript_title', '')}"
    )

    preview = fetch_json(
        session=session,
        path=f"/transcripts/{call_id}",
        request_state=request_state,
    )

    print_preview_sample(
        preview=preview,
        api_key=api_key,
    )

    return True


# ============================================================
# main()
# ============================================================

def main() -> int:
    """
    Run the bounded free-tier diagnostic.
    """

    tickers = get_tickers()
    api_key = get_rapidapi_key()

    if api_key is None:
        print(
            "Live test not run: add RAPIDAPI_KEY "
            "to the project .env file."
        )

        print(
            "Enable the free plan at:"
        )

        print(
            FREE_PLAN_URL
        )

        return 1

    maximum_requests = (
        REQUESTS_PER_TICKER
        * len(tickers)
    )

    print(
        f"Free-tier probe: at most {maximum_requests} requests; "
        "no automatic retries."
    )

    print(
        "Transcript responses are previews, "
        "not full-text validation."
    )

    # Keep the counter in shared mutable state so failed requests are still
    # included in the final total.
    request_state = {
        "count": 0,
    }

    failed = False

    # Reuse one connection for every provider call.
    with create_session(api_key) as session:
        try:
            for ticker in tickers:
                found_call = probe_ticker(
                    session=session,
                    ticker=ticker,
                    api_key=api_key,
                    request_state=request_state,
                )

                if not found_call:
                    failed = True

        except (
            requests.RequestException,
            ValueError,
            RuntimeError,
        ) as error:
            # A failure stops the probe immediately. Automatic retries could
            # silently consume the rest of a small free-tier allowance.
            if isinstance(error, RuntimeError):
                print(
                    str(error)
                )
            else:
                print(
                    f"Stopped: {type(error).__name__}; "
                    "no retry was made."
                )

            failed = True

    print(
        f"\nRequests attempted: {request_state['count']}"
    )

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
