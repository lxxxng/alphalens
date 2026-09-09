"""Read-only API endpoints for AlphaLens evaluation diagnostics."""

from fastapi import APIRouter, Query

from app.services.evaluation_reports import load_evaluation_dashboard


router = APIRouter(
    prefix="/api/evaluations",
    tags=["Evaluations"],
)


@router.get("")
def evaluation_dashboard(
    history_limit: int = Query(default=50, ge=1, le=200),
):
    """Return latest full reports and compact historical trend points."""

    return load_evaluation_dashboard(history_limit=history_limit)
