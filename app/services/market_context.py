"""
Structured market-data context for AlphaLens answers.

Market prices are numeric time series, so they should be queried and
calculated directly rather than embedded into FAISS like prose documents.
"""

import math
import os
from datetime import timedelta

from dotenv import load_dotenv
from sqlalchemy import MetaData, Table, create_engine, select


load_dotenv()


BENCHMARK_TICKER = "SPY"

RETURN_WINDOWS = {
    "1M": 30,
    "3M": 90,
    "1Y": 365,
    "5Y": 365 * 5,
}

MARKET_CHART_WINDOWS = {
    "1M": 30,
    "3M": 90,
    "1Y": 365,
    "5Y": 365 * 5,
}

MAX_MARKET_CHART_POINTS = 420

MARKET_QUESTION_TERMS = [
    "stock",
    "share price",
    "price",
    "market",
    "performance",
    "perform",
    "return",
    "returns",
    "volatility",
    "volume",
    "trading",
    "spy",
    "benchmark",
]


def get_database_engine():
    """
    Create the PostgreSQL SQLAlchemy engine from DATABASE_URL.
    """

    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise ValueError(
            "DATABASE_URL was not found in .env."
        )

    return create_engine(
        database_url,
        pool_pre_ping=True,
    )


def wants_market_context(
    question: str,
) -> bool:
    """
    Return True when a question asks for market-price context.
    """

    lower_question = question.lower()

    return any(
        term in lower_question
        for term in MARKET_QUESTION_TERMS
    )


def fetch_price_rows(
    engine,
    ticker: str,
) -> list[dict]:
    """
    Retrieve all stored market rows for one ticker.
    """

    metadata = MetaData()

    market_prices = Table(
        "market_prices",
        metadata,
        autoload_with=engine,
    )

    query = (
        select(
            market_prices.c.ticker,
            market_prices.c.trading_date,
            market_prices.c.close,
            market_prices.c.adjusted_close,
            market_prices.c.volume,
        )
        .where(
            market_prices.c.ticker == ticker.upper()
        )
        .order_by(
            market_prices.c.trading_date
        )
    )

    with engine.connect() as connection:
        return [
            dict(row)
            for row in connection.execute(
                query
            ).mappings().all()
        ]


def price_value(
    row: dict,
) -> float | None:
    """
    Use adjusted close when available, otherwise close.
    """

    value = row.get("adjusted_close")

    if value is None:
        value = row.get("close")

    if value is None:
        return None

    return float(value)


def filter_rows_for_period(
    rows: list[dict],
    start_date,
    end_date,
) -> list[dict]:
    """Keep valid price rows inside an inclusive chart window."""

    return [
        row
        for row in rows
        if (
            start_date <= row["trading_date"] <= end_date
            and price_value(row) is not None
        )
    ]


def downsample_rows(
    rows: list[dict],
    max_points: int = MAX_MARKET_CHART_POINTS,
) -> list[dict]:
    """
    Reduce long histories while preserving the first and last observations.

    Five years of daily data can exceed 1,200 records. A few hundred evenly
    spaced observations render the same trend with a smaller API response.
    """

    if len(rows) <= max_points:
        return rows

    indexes = {
        round(
            position
            * (len(rows) - 1)
            / (max_points - 1)
        )
        for position in range(max_points)
    }

    return [
        rows[index]
        for index in sorted(indexes)
    ]


def build_market_price_series(
    ticker: str,
    rows: list[dict],
) -> dict | None:
    """Convert price rows into raw and start-at-100 chart points."""

    if not rows:
        return None

    sampled_rows = downsample_rows(rows)
    starting_price = price_value(rows[0])

    if not starting_price:
        return None

    points = []

    for row in sampled_rows:
        close = price_value(row)

        if close is None:
            continue

        points.append(
            {
                "date": str(row["trading_date"]),
                "close": close,
                "indexed_value": close / starting_price * 100,
            }
        )

    return {
        "ticker": ticker.upper(),
        "points": points,
    }


def get_market_history(
    ticker: str,
    period: str = "1Y",
) -> dict | None:
    """
    Return chart-ready company and SPY histories for one trailing period.

    Both lines are independently indexed to 100 at their first observation,
    which makes their percentage performance comparable on one y-axis.
    """

    normalized_ticker = ticker.upper()
    normalized_period = period.upper()

    if normalized_period not in MARKET_CHART_WINDOWS:
        raise ValueError(
            f"Unsupported market chart period: {period}"
        )

    engine = get_database_engine()
    ticker_rows = fetch_price_rows(
        engine=engine,
        ticker=normalized_ticker,
    )

    if not ticker_rows:
        return None

    end_date = ticker_rows[-1]["trading_date"]
    start_date = end_date - timedelta(
        days=MARKET_CHART_WINDOWS[normalized_period]
    )
    requested_tickers = [normalized_ticker]

    if normalized_ticker != BENCHMARK_TICKER:
        requested_tickers.append(BENCHMARK_TICKER)

    series = []

    for requested_ticker in requested_tickers:
        rows = (
            ticker_rows
            if requested_ticker == normalized_ticker
            else fetch_price_rows(
                engine=engine,
                ticker=requested_ticker,
            )
        )
        window_rows = filter_rows_for_period(
            rows=rows,
            start_date=start_date,
            end_date=end_date,
        )
        price_series = build_market_price_series(
            ticker=requested_ticker,
            rows=window_rows,
        )

        if price_series is not None:
            series.append(price_series)

    if not series:
        return None

    return {
        "ticker": normalized_ticker,
        "benchmark_ticker": BENCHMARK_TICKER,
        "period": normalized_period,
        "start_date": series[0]["points"][0]["date"],
        "end_date": series[0]["points"][-1]["date"],
        "series": series,
    }


def find_row_on_or_before(
    rows: list[dict],
    target_date,
) -> dict | None:
    """
    Find the most recent stored trading row on or before a calendar date.
    """

    candidate = None

    for row in rows:
        if row["trading_date"] <= target_date:
            candidate = row
        else:
            break

    return candidate


def calculate_return(
    latest_row: dict,
    comparison_row: dict | None,
) -> float | None:
    """
    Calculate decimal return from comparison row to latest row.
    """

    if comparison_row is None:
        return None

    latest_price = price_value(
        latest_row
    )
    comparison_price = price_value(
        comparison_row
    )

    if not latest_price or not comparison_price:
        return None

    return (
        latest_price
        / comparison_price
        - 1
    )


def calculate_returns(
    rows: list[dict],
) -> dict[str, float | None]:
    """
    Calculate common trailing returns from stored market rows.
    """

    if not rows:
        return {
            label: None
            for label in RETURN_WINDOWS
        }

    latest_row = rows[-1]
    latest_date = latest_row["trading_date"]
    returns = {}

    for label, days in RETURN_WINDOWS.items():
        comparison_row = find_row_on_or_before(
            rows=rows,
            target_date=(
                latest_date
                - timedelta(days=days)
            ),
        )

        returns[label] = calculate_return(
            latest_row=latest_row,
            comparison_row=comparison_row,
        )

    return returns


def calculate_annualized_volatility(
    rows: list[dict],
    trading_days: int = 252,
) -> float | None:
    """
    Calculate annualized volatility from recent daily adjusted-close returns.
    """

    recent_rows = rows[-(trading_days + 1):]
    daily_returns = []

    for previous, current in zip(
        recent_rows,
        recent_rows[1:],
    ):
        previous_price = price_value(
            previous
        )
        current_price = price_value(
            current
        )

        if not previous_price or not current_price:
            continue

        daily_returns.append(
            current_price / previous_price - 1
        )

    if len(daily_returns) < 2:
        return None

    average = sum(daily_returns) / len(daily_returns)
    variance = sum(
        (value - average) ** 2
        for value in daily_returns
    ) / (len(daily_returns) - 1)

    return math.sqrt(variance) * math.sqrt(252)


def calculate_average_volume(
    rows: list[dict],
    trading_days: int = 30,
) -> float | None:
    """
    Calculate average volume over the latest stored trading days.
    """

    volumes = [
        row["volume"]
        for row in rows[-trading_days:]
        if row.get("volume") is not None
    ]

    if not volumes:
        return None

    return sum(volumes) / len(volumes)


def calculate_relative_returns(
    returns: dict[str, float | None],
    benchmark_returns: dict[str, float | None],
) -> dict[str, float | None]:
    """
    Calculate return spread versus the benchmark for each window.
    """

    relative = {}

    for label, value in returns.items():
        benchmark_value = benchmark_returns.get(
            label
        )

        if value is None or benchmark_value is None:
            relative[label] = None
        else:
            relative[label] = value - benchmark_value

    return relative


def build_ticker_snapshot(
    ticker: str,
    rows: list[dict],
    benchmark_returns: dict[str, float | None],
) -> dict | None:
    """
    Build one API-friendly market snapshot for a ticker.
    """

    if not rows:
        return None

    latest_row = rows[-1]
    returns = calculate_returns(
        rows
    )

    benchmark_relative_returns = (
        {
            label: None
            for label in RETURN_WINDOWS
        }
        if ticker.upper() == BENCHMARK_TICKER
        else calculate_relative_returns(
            returns=returns,
            benchmark_returns=benchmark_returns,
        )
    )

    return {
        "ticker": ticker.upper(),
        "latest_trading_date": str(latest_row["trading_date"]),
        "latest_close": latest_row.get("close"),
        "latest_adjusted_close": latest_row.get("adjusted_close"),
        "returns": returns,
        "benchmark_ticker": BENCHMARK_TICKER,
        "benchmark_relative_returns": benchmark_relative_returns,
        "annualized_volatility": calculate_annualized_volatility(
            rows
        ),
        "average_volume_30d": calculate_average_volume(
            rows
        ),
        "row_count": len(rows),
    }


def get_market_context(
    tickers: list[str],
) -> list[dict]:
    """
    Return structured market snapshots for the requested tickers.
    """

    requested_tickers = []

    for ticker in tickers:
        normalized = ticker.upper()

        if normalized not in requested_tickers:
            requested_tickers.append(
                normalized
            )

    if not requested_tickers:
        return []

    engine = get_database_engine()

    benchmark_rows = fetch_price_rows(
        engine=engine,
        ticker=BENCHMARK_TICKER,
    )
    benchmark_returns = calculate_returns(
        benchmark_rows
    )

    snapshots = []

    for ticker in requested_tickers:
        rows = fetch_price_rows(
            engine=engine,
            ticker=ticker,
        )

        snapshot = build_ticker_snapshot(
            ticker=ticker,
            rows=rows,
            benchmark_returns=benchmark_returns,
        )

        if snapshot is not None:
            snapshots.append(
                snapshot
            )

    return snapshots


def format_percent(
    value: float | None,
) -> str:
    """
    Format decimal returns as percentages for the LLM context.
    """

    if value is None:
        return "N/A"

    return f"{value * 100:.2f}%"


def format_number(
    value: float | None,
) -> str:
    """
    Format compact numeric values for the LLM context.
    """

    if value is None:
        return "N/A"

    return f"{value:,.2f}"


def build_market_context_text(
    market_context: list[dict],
) -> str:
    """
    Convert market snapshots into compact evidence text for the LLM.
    """

    if not market_context:
        return ""

    sections = []

    for snapshot in market_context:
        returns = snapshot["returns"]
        relative = snapshot["benchmark_relative_returns"]

        sections.append(
            "\n".join(
                [
                    f"Ticker: {snapshot['ticker']}",
                    (
                        "Latest trading date: "
                        f"{snapshot['latest_trading_date']}"
                    ),
                    (
                        "Latest close: "
                        f"{format_number(snapshot['latest_close'])}"
                    ),
                    (
                        "Latest adjusted close: "
                        f"{format_number(snapshot['latest_adjusted_close'])}"
                    ),
                    (
                        "Returns: "
                        f"1M {format_percent(returns['1M'])}, "
                        f"3M {format_percent(returns['3M'])}, "
                        f"1Y {format_percent(returns['1Y'])}, "
                        f"5Y {format_percent(returns['5Y'])}"
                    ),
                    (
                        f"Relative returns vs {snapshot['benchmark_ticker']}: "
                        f"1M {format_percent(relative['1M'])}, "
                        f"3M {format_percent(relative['3M'])}, "
                        f"1Y {format_percent(relative['1Y'])}, "
                        f"5Y {format_percent(relative['5Y'])}"
                    ),
                    (
                        "Annualized volatility: "
                        f"{format_percent(snapshot['annualized_volatility'])}"
                    ),
                    (
                        "Average volume 30d: "
                        f"{format_number(snapshot['average_volume_30d'])}"
                    ),
                ]
            )
        )

    return "\n\n".join(
        sections
    )
