"""CRUD API for named AlphaLens watchlists and local ticker signals."""

from typing import Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.services.watchlists import (
    UnknownTickerError,
    WatchlistConflictError,
    WatchlistNotFoundError,
    add_watchlist_items,
    create_watchlist,
    delete_watchlist,
    get_watchlist,
    list_watchlists,
    remove_watchlist_item,
    rename_watchlist,
)


router = APIRouter(prefix="/api/watchlists", tags=["Watchlists"])


class WatchlistCreate(BaseModel):
    name: str = Field(min_length=1, max_length=80)


class WatchlistUpdate(BaseModel):
    name: str = Field(min_length=1, max_length=80)


class WatchlistItemsUpdate(BaseModel):
    tickers: list[str] = Field(min_length=1, max_length=20)


class WatchlistEvent(BaseModel):
    event_type: str
    date: str
    label: str
    source_url: Optional[str] = None


class WatchlistItem(BaseModel):
    ticker: str
    company_name: Optional[str] = None
    added_at: str
    latest_price: Optional[float] = None
    latest_price_date: Optional[str] = None
    daily_return: Optional[float] = None
    sentiment_label: Optional[str] = None
    sentiment_score: Optional[float] = None
    sentiment_period: Optional[str] = None
    latest_event: Optional[WatchlistEvent] = None


class WatchlistSummary(BaseModel):
    watchlist_id: int
    name: str
    item_count: int
    created_at: str
    updated_at: str


class WatchlistDetail(WatchlistSummary):
    items: list[WatchlistItem] = Field(default_factory=list)


class WatchlistsResponse(BaseModel):
    watchlists: list[WatchlistSummary]


class DeleteWatchlistResponse(BaseModel):
    watchlist_id: int
    deleted: bool


def _detail_or_http_error(operation):
    """Map service-level domain errors to stable HTTP responses."""

    try:
        return operation()
    except WatchlistNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except WatchlistConflictError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except (UnknownTickerError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@router.get("", response_model=WatchlistsResponse)
def watchlist_index():
    """List named watchlists without loading all ticker signals."""

    return {"watchlists": list_watchlists()}


@router.post("", response_model=WatchlistSummary, status_code=status.HTTP_201_CREATED)
def watchlist_create(request: WatchlistCreate):
    """Create a named watchlist."""

    return _detail_or_http_error(lambda: create_watchlist(request.name))


@router.get("/{watchlist_id}", response_model=WatchlistDetail)
def watchlist_detail(watchlist_id: int):
    """Load members with latest local price, sentiment, and event signals."""

    return _detail_or_http_error(lambda: get_watchlist(watchlist_id))


@router.patch("/{watchlist_id}", response_model=WatchlistSummary)
def watchlist_rename(watchlist_id: int, request: WatchlistUpdate):
    """Rename a watchlist."""

    return _detail_or_http_error(
        lambda: rename_watchlist(watchlist_id, request.name)
    )


@router.delete("/{watchlist_id}", response_model=DeleteWatchlistResponse)
def watchlist_delete(watchlist_id: int):
    """Delete a named watchlist and its membership."""

    deleted = delete_watchlist(watchlist_id)

    if not deleted:
        raise HTTPException(
            status_code=404,
            detail=f"Watchlist {watchlist_id} was not found.",
        )

    return {"watchlist_id": watchlist_id, "deleted": True}


@router.post("/{watchlist_id}/items", response_model=WatchlistDetail)
def watchlist_items_add(watchlist_id: int, request: WatchlistItemsUpdate):
    """Add one or more local tickers idempotently."""

    return _detail_or_http_error(
        lambda: add_watchlist_items(watchlist_id, request.tickers)
    )


@router.delete("/{watchlist_id}/items/{ticker}", response_model=WatchlistDetail)
def watchlist_item_delete(watchlist_id: int, ticker: str):
    """Remove one ticker from a watchlist."""

    return _detail_or_http_error(
        lambda: remove_watchlist_item(watchlist_id, ticker)
    )
