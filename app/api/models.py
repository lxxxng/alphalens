"""Model-registry status and guarded point-in-time prediction endpoints."""

from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

from app.services.model_inference import (
    get_model_status,
    normalize_prediction_tickers,
    predict_latest_events,
)


router = APIRouter(prefix="/api/models", tags=["Models"])


class PredictionRequest(BaseModel):
    """A bounded multi-company request for the latest scorable events."""

    tickers: list[str] = Field(min_length=1, max_length=4)
    event_source: Literal["latest", "earnings_call", "sec_filing"] = "latest"
    research_preview: bool = False

    @field_validator("tickers")
    @classmethod
    def validate_tickers(cls, value: list[str]) -> list[str]:
        """Apply the service's canonical normalization before execution."""

        return normalize_prediction_tickers(value)


@router.get("/status")
def model_status():
    """Return registry and production-serving readiness."""

    return get_model_status()


@router.post("/predict")
def predict(request: PredictionRequest):
    """Score current event features without bypassing promotion policy."""

    try:
        return predict_latest_events(
            request.tickers,
            event_source=request.event_source,
            research_preview=request.research_preview,
        )
    except PermissionError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except LookupError as error:
        status_code = 409 if "champion" in str(error).lower() else 404
        raise HTTPException(status_code=status_code, detail=str(error)) from error
    except (ImportError, RuntimeError, ValueError, OSError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
