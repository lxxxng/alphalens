"""Persistence and local signal summaries for AlphaLens watchlists."""

from __future__ import annotations

import os
from collections import defaultdict
from datetime import date, datetime

from dotenv import load_dotenv
from sqlalchemy import (
    MetaData,
    Table,
    and_,
    create_engine,
    delete,
    desc,
    func,
    insert,
    select,
    update,
)
from sqlalchemy.dialects.postgresql import insert as postgres_insert
from sqlalchemy.exc import IntegrityError

from app.services.sentiment import (
    model_config,
    speaker_group,
    summarize_sentiment_rows,
)


load_dotenv()


class WatchlistNotFoundError(LookupError):
    """Raised when a requested watchlist does not exist."""


class WatchlistConflictError(ValueError):
    """Raised when a watchlist name is already in use."""


class UnknownTickerError(ValueError):
    """Raised when membership contains a ticker outside the local corpus."""


def get_database_engine():
    """Create the configured PostgreSQL engine."""

    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise ValueError("DATABASE_URL was not found in .env.")

    return create_engine(database_url, pool_pre_ping=True)


def _tables(engine) -> dict[str, Table]:
    """Reflect migration-managed tables needed by watchlist summaries."""

    metadata = MetaData()
    names = (
        "watchlists",
        "watchlist_items",
        "companies",
        "market_prices",
        "earnings_transcripts",
        "earnings_transcript_turns",
        "earnings_transcript_turn_sentiment",
        "filings",
    )
    return {
        name: Table(name, metadata, autoload_with=engine)
        for name in names
    }


def normalize_watchlist_name(name: str) -> str:
    """Trim and collapse whitespace in a user-facing watchlist name."""

    normalized = " ".join(str(name or "").split())

    if not normalized:
        raise ValueError("Watchlist name cannot be empty.")

    if len(normalized) > 80:
        raise ValueError("Watchlist name cannot exceed 80 characters.")

    return normalized


def normalize_watchlist_tickers(tickers: list[str]) -> list[str]:
    """Normalize, deduplicate, and preserve requested ticker order."""

    normalized = []

    for ticker in tickers:
        value = str(ticker or "").strip().upper()

        if value and value not in normalized:
            normalized.append(value)

    return normalized


def _iso(value):
    """Serialize PostgreSQL date values without changing nulls."""

    if isinstance(value, (date, datetime)):
        return value.isoformat()

    return value


def _watchlist_summary(row: dict) -> dict:
    return {
        "watchlist_id": int(row["watchlist_id"]),
        "name": row["name"],
        "item_count": int(row.get("item_count") or 0),
        "created_at": _iso(row["created_at"]),
        "updated_at": _iso(row["updated_at"]),
    }


def list_watchlists() -> list[dict]:
    """Return named watchlists and lightweight membership counts."""

    engine = get_database_engine()
    tables = _tables(engine)
    watchlists = tables["watchlists"]
    items = tables["watchlist_items"]
    statement = (
        select(
            watchlists.c.watchlist_id,
            watchlists.c.name,
            watchlists.c.created_at,
            watchlists.c.updated_at,
            func.count(items.c.ticker).label("item_count"),
        )
        .select_from(watchlists.outerjoin(items))
        .group_by(
            watchlists.c.watchlist_id,
            watchlists.c.name,
            watchlists.c.created_at,
            watchlists.c.updated_at,
        )
        .order_by(watchlists.c.created_at, watchlists.c.watchlist_id)
    )

    with engine.connect() as connection:
        rows = connection.execute(statement).mappings().all()

    return [_watchlist_summary(dict(row)) for row in rows]


def create_watchlist(name: str) -> dict:
    """Create one named watchlist and return its summary."""

    engine = get_database_engine()
    tables = _tables(engine)
    watchlists = tables["watchlists"]
    normalized_name = normalize_watchlist_name(name)
    statement = (
        insert(watchlists)
        .values(name=normalized_name)
        .returning(watchlists)
    )

    try:
        with engine.begin() as connection:
            row = dict(connection.execute(statement).mappings().one())
    except IntegrityError as error:
        raise WatchlistConflictError(
            f'A watchlist named "{normalized_name}" already exists.'
        ) from error

    return _watchlist_summary({**row, "item_count": 0})


def rename_watchlist(watchlist_id: int, name: str) -> dict:
    """Rename a watchlist while preserving its membership."""

    engine = get_database_engine()
    tables = _tables(engine)
    watchlists = tables["watchlists"]
    normalized_name = normalize_watchlist_name(name)
    statement = (
        update(watchlists)
        .where(watchlists.c.watchlist_id == watchlist_id)
        .values(name=normalized_name, updated_at=func.now())
        .returning(watchlists)
    )

    try:
        with engine.begin() as connection:
            row = connection.execute(statement).mappings().first()
    except IntegrityError as error:
        raise WatchlistConflictError(
            f'A watchlist named "{normalized_name}" already exists.'
        ) from error

    if row is None:
        raise WatchlistNotFoundError(f"Watchlist {watchlist_id} was not found.")

    detail = get_watchlist(watchlist_id)
    return {
        key: detail[key]
        for key in (
            "watchlist_id",
            "name",
            "item_count",
            "created_at",
            "updated_at",
        )
    }


def delete_watchlist(watchlist_id: int) -> bool:
    """Delete a watchlist and its membership rows."""

    engine = get_database_engine()
    watchlists = _tables(engine)["watchlists"]
    statement = delete(watchlists).where(
        watchlists.c.watchlist_id == watchlist_id
    )

    with engine.begin() as connection:
        result = connection.execute(statement)

    return result.rowcount > 0


def _require_watchlist(connection, watchlists: Table, watchlist_id: int) -> dict:
    row = connection.execute(
        select(watchlists).where(watchlists.c.watchlist_id == watchlist_id)
    ).mappings().first()

    if row is None:
        raise WatchlistNotFoundError(f"Watchlist {watchlist_id} was not found.")

    return dict(row)


def add_watchlist_items(watchlist_id: int, tickers: list[str]) -> dict:
    """Add valid local tickers idempotently and return refreshed detail."""

    normalized = normalize_watchlist_tickers(tickers)

    if not normalized:
        raise ValueError("At least one ticker is required.")

    engine = get_database_engine()
    tables = _tables(engine)
    watchlists = tables["watchlists"]
    items = tables["watchlist_items"]
    companies = tables["companies"]

    with engine.begin() as connection:
        _require_watchlist(connection, watchlists, watchlist_id)
        available = set(connection.execute(
            select(companies.c.ticker).where(companies.c.ticker.in_(normalized))
        ).scalars())
        unknown = [ticker for ticker in normalized if ticker not in available]

        if unknown:
            raise UnknownTickerError(
                f"Unknown local ticker(s): {', '.join(unknown)}."
            )

        connection.execute(
            postgres_insert(items)
            .values([
                {"watchlist_id": watchlist_id, "ticker": ticker}
                for ticker in normalized
            ])
            .on_conflict_do_nothing(
                index_elements=[items.c.watchlist_id, items.c.ticker]
            )
        )
        connection.execute(
            update(watchlists)
            .where(watchlists.c.watchlist_id == watchlist_id)
            .values(updated_at=func.now())
        )

    return get_watchlist(watchlist_id)


def remove_watchlist_item(watchlist_id: int, ticker: str) -> dict:
    """Remove one ticker and return refreshed watchlist detail."""

    normalized = normalize_watchlist_tickers([ticker])

    if not normalized:
        raise ValueError("Ticker cannot be empty.")

    engine = get_database_engine()
    tables = _tables(engine)
    watchlists = tables["watchlists"]
    items = tables["watchlist_items"]

    with engine.begin() as connection:
        _require_watchlist(connection, watchlists, watchlist_id)
        result = connection.execute(
            delete(items).where(
                and_(
                    items.c.watchlist_id == watchlist_id,
                    items.c.ticker == normalized[0],
                )
            )
        )

        if result.rowcount == 0:
            raise UnknownTickerError(
                f"{normalized[0]} is not in this watchlist."
            )

        connection.execute(
            update(watchlists)
            .where(watchlists.c.watchlist_id == watchlist_id)
            .values(updated_at=func.now())
        )

    return get_watchlist(watchlist_id)


def _latest_price_snapshots(engine, market_prices: Table, tickers: list[str]) -> dict:
    price = func.coalesce(
        market_prices.c.adjusted_close,
        market_prices.c.close,
    ).label("price")
    ranked = select(
        market_prices.c.ticker,
        market_prices.c.trading_date,
        price,
        func.row_number().over(
            partition_by=market_prices.c.ticker,
            order_by=market_prices.c.trading_date.desc(),
        ).label("price_rank"),
    ).where(market_prices.c.ticker.in_(tickers)).subquery()
    statement = (
        select(ranked)
        .where(ranked.c.price_rank <= 2)
        .order_by(ranked.c.ticker, ranked.c.price_rank)
    )

    with engine.connect() as connection:
        rows = connection.execute(statement).mappings().all()

    grouped = defaultdict(list)

    for row in rows:
        grouped[row["ticker"]].append(dict(row))

    snapshots = {}

    for ticker, prices in grouped.items():
        latest = prices[0]
        previous = prices[1] if len(prices) > 1 else None
        latest_price = float(latest["price"]) if latest["price"] is not None else None
        previous_price = (
            float(previous["price"])
            if previous and previous["price"] is not None
            else None
        )
        daily_return = (
            latest_price / previous_price - 1
            if latest_price is not None and previous_price not in (None, 0)
            else None
        )
        snapshots[ticker] = {
            "latest_price": round(latest_price, 4) if latest_price is not None else None,
            "latest_price_date": _iso(latest["trading_date"]),
            "daily_return": round(daily_return, 6) if daily_return is not None else None,
        }

    return snapshots


def _latest_records(engine, table: Table, tickers: list[str], order_columns) -> dict:
    statement = (
        select(table)
        .where(table.c.ticker.in_(tickers))
        .order_by(table.c.ticker, *[desc(column) for column in order_columns])
    )

    with engine.connect() as connection:
        rows = connection.execute(statement).mappings().all()

    latest = {}

    for row in rows:
        latest.setdefault(row["ticker"], dict(row))

    return latest


def _latest_transcript_sentiment(
    engine,
    tables: dict[str, Table],
    transcripts: dict[str, dict],
) -> dict:
    transcript_ids = [row["transcript_id"] for row in transcripts.values()]

    if not transcript_ids:
        return {}

    turns = tables["earnings_transcript_turns"]
    sentiment = tables["earnings_transcript_turn_sentiment"]
    model_name, model_revision = model_config()
    statement = (
        select(
            turns.c.transcript_id,
            turns.c.speaker_name,
            turns.c.speaker_title,
            turns.c.speaker_role,
            sentiment.c.status,
            sentiment.c.sentiment_label,
            sentiment.c.sentiment_score,
            sentiment.c.confidence,
            sentiment.c.probabilities,
            sentiment.c.token_count,
        )
        .select_from(
            turns.join(
                sentiment,
                and_(
                    sentiment.c.turn_id == turns.c.turn_id,
                    sentiment.c.model_name == model_name,
                    sentiment.c.model_revision == model_revision,
                ),
            )
        )
        .where(turns.c.transcript_id.in_(transcript_ids))
        .order_by(turns.c.transcript_id, turns.c.turn_index)
    )

    with engine.connect() as connection:
        rows = connection.execute(statement).mappings().all()

    grouped = defaultdict(list)

    for row in rows:
        values = dict(row)

        if speaker_group(values) == "management":
            grouped[values["transcript_id"]].append(values)

    return {
        transcript_id: summarize_sentiment_rows(values)
        for transcript_id, values in grouped.items()
    }


def _event_snapshot(transcript: dict | None, filing: dict | None) -> dict | None:
    candidates = []

    if transcript and transcript.get("call_date"):
        candidates.append({
            "event_type": "earnings",
            "date": transcript["call_date"],
            "label": transcript.get("fiscal_period") or "Earnings call",
            "source_url": f'/transcripts/{transcript["transcript_id"]}',
        })

    if filing and filing.get("filing_date"):
        candidates.append({
            "event_type": "filing",
            "date": filing["filing_date"],
            "label": filing.get("form_type") or "SEC filing",
            "source_url": filing.get("source_url"),
        })

    if not candidates:
        return None

    latest = max(candidates, key=lambda item: item["date"])
    latest["date"] = _iso(latest["date"])
    return latest


def get_watchlist(watchlist_id: int) -> dict:
    """Return one watchlist enriched entirely from the local corpus."""

    engine = get_database_engine()
    tables = _tables(engine)
    watchlists = tables["watchlists"]
    items = tables["watchlist_items"]
    companies = tables["companies"]

    with engine.connect() as connection:
        watchlist = _require_watchlist(connection, watchlists, watchlist_id)
        rows = connection.execute(
            select(
                items.c.ticker,
                items.c.added_at,
                companies.c.company_name,
            )
            .select_from(items.join(companies))
            .where(items.c.watchlist_id == watchlist_id)
            .order_by(items.c.added_at, items.c.ticker)
        ).mappings().all()

    tickers = [row["ticker"] for row in rows]

    if not tickers:
        return {
            **_watchlist_summary({**watchlist, "item_count": 0}),
            "items": [],
        }

    prices = _latest_price_snapshots(engine, tables["market_prices"], tickers)
    transcripts = _latest_records(
        engine,
        tables["earnings_transcripts"],
        tickers,
        (
            tables["earnings_transcripts"].c.fiscal_year,
            tables["earnings_transcripts"].c.fiscal_quarter,
        ),
    )
    filings = _latest_records(
        engine,
        tables["filings"],
        tickers,
        (tables["filings"].c.filing_date,),
    )
    sentiment = _latest_transcript_sentiment(engine, tables, transcripts)
    enriched = []

    for row in rows:
        ticker = row["ticker"]
        transcript = transcripts.get(ticker)
        aggregate = sentiment.get(
            transcript["transcript_id"] if transcript else None,
            {},
        )
        enriched.append({
            "ticker": ticker,
            "company_name": row["company_name"],
            "added_at": _iso(row["added_at"]),
            **prices.get(ticker, {
                "latest_price": None,
                "latest_price_date": None,
                "daily_return": None,
            }),
            "sentiment_label": aggregate.get("label"),
            "sentiment_score": aggregate.get("score"),
            "sentiment_period": transcript.get("fiscal_period") if transcript else None,
            "latest_event": _event_snapshot(transcript, filings.get(ticker)),
        })

    return {
        **_watchlist_summary({**watchlist, "item_count": len(enriched)}),
        "items": enriched,
    }
