"""API endpoints for persistent watched-company event alerts."""

from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.services.alerts import (
    AlertNotFoundError,
    list_event_alerts,
    mark_all_event_alerts_read,
    mark_event_alert_read,
)


router = APIRouter(prefix="/api/alerts", tags=["Alerts"])


class EventAlert(BaseModel):
    alert_id: int
    ticker: str
    event_type: str
    source_record_id: str
    event_date: Optional[str] = None
    title: str
    message: Optional[str] = None
    source_url: Optional[str] = None
    ingestion_run_id: Optional[int] = None
    is_read: bool
    read_at: Optional[str] = None
    created_at: str
    brief_id: Optional[int] = None
    brief_status: str = "PENDING"
    brief_attempt_count: int = 0
    brief_error: Optional[str] = None
    brief_evaluation: dict = Field(default_factory=dict)
    brief_last_attempt_at: Optional[str] = None
    brief_generated_at: Optional[str] = None


class EventAlertsResponse(BaseModel):
    alerts: list[EventAlert] = Field(default_factory=list)
    unread_count: int


class MarkAllReadResponse(BaseModel):
    updated_count: int


@router.get("", response_model=EventAlertsResponse)
def alert_index(
    limit: int = Query(default=30, ge=1, le=100),
    unread_only: bool = False,
    ticker: Optional[str] = None,
):
    """List recent event alerts with the global unread count."""

    return list_event_alerts(
        limit=limit,
        unread_only=unread_only,
        ticker=ticker,
    )


@router.patch("/{alert_id}/read", response_model=EventAlert)
def alert_mark_read(alert_id: int):
    """Mark one event alert read."""

    try:
        return mark_event_alert_read(alert_id)
    except AlertNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@router.post("/read-all", response_model=MarkAllReadResponse)
def alerts_mark_all_read():
    """Mark all current event alerts read."""

    return {"updated_count": mark_all_event_alerts_read()}
