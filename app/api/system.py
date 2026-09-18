"""Read-only deployment and data-freshness diagnostics."""

from fastapi import APIRouter, HTTPException

from app.services.data_freshness import get_data_freshness


router = APIRouter(prefix="/api/system", tags=["System"])


@router.get("/data-freshness")
def data_freshness():
    """Return market, scheduler, and pipeline activity timestamps."""

    try:
        return get_data_freshness()
    except (RuntimeError, ValueError, OSError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
