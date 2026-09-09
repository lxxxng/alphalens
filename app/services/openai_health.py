"""
OpenAI connectivity diagnostics for AlphaLens.

The health check intentionally lists models instead of generating text. That
proves the configured API key can authenticate without spending generation
tokens or depending on the RAG prompt path.
"""

import os
from time import perf_counter
from typing import Callable

from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    OpenAI,
    PermissionDeniedError,
    RateLimitError,
)


load_dotenv()


OPENAI_HEALTH_TIMEOUT_SECONDS = 5.0


def _result(
    *,
    status: str,
    configured: bool,
    reachable: bool,
    message: str,
    started_at: float,
    error_type: str | None = None,
) -> dict:
    """Build the stable response shape returned by the health endpoint."""

    return {
        "status": status,
        "configured": configured,
        "reachable": reachable,
        "latency_ms": round(
            (perf_counter() - started_at) * 1000,
            1,
        ),
        "message": message,
        "error_type": error_type,
    }


def check_openai_connection(
    *,
    client_factory: Callable[..., OpenAI] = OpenAI,
) -> dict:
    """
    Check whether AlphaLens can authenticate with the OpenAI API.

    API keys are never returned or logged. Retries are disabled because this
    endpoint is an immediate diagnostic rather than a production model call.
    """

    started_at = perf_counter()
    api_key = os.getenv("OPENAI_API_KEY", "").strip()

    if not api_key or api_key.upper() == "YOUR_OPENAI_API_KEY":
        return _result(
            status="not_configured",
            configured=False,
            reachable=False,
            message="OPENAI_API_KEY is not configured.",
            error_type="missing_api_key",
            started_at=started_at,
        )

    try:
        client = client_factory(
            api_key=api_key,
            timeout=OPENAI_HEALTH_TIMEOUT_SECONDS,
            max_retries=0,
        )
        client.models.list()

        return _result(
            status="ok",
            configured=True,
            reachable=True,
            message="Connected to OpenAI.",
            started_at=started_at,
        )

    except AuthenticationError:
        message = "OpenAI rejected the API key."
        error_type = "authentication_error"
    except PermissionDeniedError:
        message = "The API key does not have permission for this request."
        error_type = "permission_denied"
    except RateLimitError:
        message = "OpenAI is reachable, but the account is rate limited."
        error_type = "rate_limit"
    except APITimeoutError:
        message = "The OpenAI connection timed out."
        error_type = "timeout"
    except APIConnectionError:
        message = "The OpenAI API could not be reached."
        error_type = "connection_error"
    except APIStatusError as error:
        # Preserve the status category without exposing the raw API response.
        message = f"OpenAI returned HTTP {error.status_code}."
        error_type = "api_status_error"
    except Exception:
        message = "The OpenAI connection check failed unexpectedly."
        error_type = "unexpected_error"

    return _result(
        status="error",
        configured=True,
        reachable=False,
        message=message,
        error_type=error_type,
        started_at=started_at,
    )
