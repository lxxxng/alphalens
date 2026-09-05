"""Check EarningsCalls.dev's free RapidAPI tier with at most six requests.

Add RAPIDAPI_KEY to the project .env after enabling the free RapidAPI plan.
Run: python -m pipelines.transcripts.probe_earningscalls
This diagnostic prints metadata and previews; it does not load the database.
"""

import argparse
import json
import os
from pathlib import Path
import time

from dotenv import load_dotenv
import requests


HOST = "earnings-call-transcripts1.p.rapidapi.com"
BASE_URL = f"https://{HOST}/api/v1"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", nargs="+", default=["AAPL", "MSFT", "NVDA"])
    args = parser.parse_args()
    tickers = list(dict.fromkeys(ticker.upper() for ticker in args.tickers))
    if len(tickers) > 3 or any(not ticker.isalnum() for ticker in tickers):
        parser.error("Use one to three simple ticker symbols, such as AAPL MSFT NVDA.")

    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    api_key = os.getenv("RAPIDAPI_KEY", "").strip()
    if not api_key or api_key.lower().startswith(("your_", "change_me")):
        print("Live test not run: add RAPIDAPI_KEY to the project .env file.")
        print("Enable the free plan at:")
        print("https://rapidapi.com/earningscallsdev/api/earnings-call-transcripts1")
        return 1

    print(f"Free-tier probe: at most {2 * len(tickers)} requests; no automatic retries.")
    print("Transcript responses are previews, not full-text validation.")
    request_count = 0
    failed = False

    with requests.Session() as session:
        session.headers.update({"X-RapidAPI-Key": api_key, "X-RapidAPI-Host": HOST})

        def fetch(path: str, params=None):
            nonlocal request_count
            if request_count:
                time.sleep(12)
            request_count += 1
            started = time.monotonic()
            response = session.get(
                f"{BASE_URL}{path}", params=params, timeout=30, allow_redirects=False
            )
            print(f"  HTTP {response.status_code}; {time.monotonic() - started:.2f}s")
            if response.status_code != 200:
                # Never print request headers or an exception containing credentials.
                raise RuntimeError(
                    f"Stopped on HTTP {response.status_code}. Check the RapidAPI "
                    "key, free-plan activation, and remaining quota."
                )
            return response.json()

        try:
            for ticker in tickers:
                print(f"\n{ticker}: company history (US listings)")
                payload = fetch(f"/companies/ticker/{ticker}", {"country": "US"})
                data = payload.get("data") if isinstance(payload, dict) else None
                calls = data.get("earnings_calls") if isinstance(data, dict) else None
                if not isinstance(calls, list):
                    raise RuntimeError("Unexpected company response: missing data.earnings_calls.")
                print(f"  Company: {data.get('company_name', '(not supplied)')}")
                print(f"  Calls returned: {len(calls)} (coverage is not guaranteed complete)")
                dates = sorted(
                    str(call["event_date_time"])[:10]
                    for call in calls
                    if isinstance(call, dict) and call.get("event_date_time")
                )
                if dates:
                    print(f"  Call dates returned: {dates[0]} to {dates[-1]}")
                candidates = [call for call in calls if isinstance(call, dict) and call.get("id")]
                if not candidates:
                    print("  No call IDs available to test a preview.")
                    failed = True
                    continue
                # The company endpoint documents newest-first ordering.
                call = candidates[0]
                call_id = str(call["id"])
                if not call_id.isdigit():
                    raise RuntimeError("Unexpected nonnumeric earnings call ID.")
                print(f"  Preview for call {call_id}: {call.get('transcript_title', '')}")
                preview = fetch(f"/transcripts/{call_id}")
                if not isinstance(preview, dict) or not preview.get("data"):
                    raise RuntimeError("Preview response is empty or has an unexpected format.")
                # A bounded sample exposes the real response schema for evaluation.
                sample = json.dumps(preview, ensure_ascii=True, indent=2).replace(api_key, "REDACTED")
                print(sample[:2000])
                if len(sample) > 2000:
                    print("  [Display limited to 2,000 characters]")
        except (requests.RequestException, ValueError, RuntimeError) as error:
            if isinstance(error, RuntimeError):
                print(str(error))
            else:
                print(f"Stopped: {type(error).__name__}; no retry was made.")
            failed = True

    print(f"\nRequests attempted: {request_count}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
