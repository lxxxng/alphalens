"""Extract market data from external sources."""
"""
AlphaLens - Market Data Extractor

Purpose:
    Downloads historical daily OHLCV market data for the AlphaLens
    starter stock universe using Yahoo Finance.

What is OHLCV?
    O = Open   -> Stock price when the market opened
    H = High   -> Highest traded price during the day
    L = Low    -> Lowest traded price during the day
    C = Close  -> Stock price when the market closed
    V = Volume -> Number of shares traded during the day

Yahoo Finance also provides:
    Adj Close -> Closing price adjusted for events such as stock splits
                 and dividends.

Current pipeline:
    Yahoo Finance
        ↓
    yfinance
        ↓
    Pandas DataFrame
        ↓
    transformer.py     (next step)
        ↓
    loader.py
        ↓
    PostgreSQL

At this stage, this script ONLY downloads the data.
It does not save anything into PostgreSQL yet.
"""

import yfinance as yf

MARKET_LOOKBACK_YEARS = 5

# ============================================================
# AlphaLens Starter Stock Universe
# ============================================================
#
# We start with 20 large US-listed companies from different
# industries instead of using only technology companies.
#
# This gives us a more diverse dataset for later:
#   - momentum calculations
#   - volatility calculations
#   - factor analysis
#   - sentiment analysis
#   - machine-learning models
#
# Each item below uses the stock's Yahoo Finance ticker symbol.
# ============================================================

TICKERS = [

    # --------------------------------------------------------
    # Technology
    # --------------------------------------------------------

    "AAPL",   # Apple Inc.
              # Consumer electronics, iPhone, Mac, services

    "MSFT",   # Microsoft Corporation
              # Software, Azure cloud, enterprise technology, AI

    "NVDA",   # NVIDIA Corporation
              # GPUs, AI accelerators, semiconductor technology

    "GOOGL",  # Alphabet Inc. - Class A
              # Google Search, advertising, YouTube, Google Cloud

    "META",   # Meta Platforms Inc.
              # Facebook, Instagram, WhatsApp, AI and advertising


    # --------------------------------------------------------
    # Consumer / Internet
    # --------------------------------------------------------

    "AMZN",   # Amazon.com Inc.
              # E-commerce, AWS cloud computing, logistics

    "TSLA",   # Tesla Inc.
              # Electric vehicles, batteries and energy products

    "NFLX",   # Netflix Inc.
              # Video streaming and entertainment


    # --------------------------------------------------------
    # Financial Services
    # --------------------------------------------------------

    "JPM",    # JPMorgan Chase & Co.
              # Banking, investment banking and financial services

    "BAC",    # Bank of America Corporation
              # Consumer banking and financial services

    "GS",     # The Goldman Sachs Group Inc.
              # Investment banking and asset management


    # --------------------------------------------------------
    # Healthcare
    # --------------------------------------------------------

    "JNJ",    # Johnson & Johnson
              # Pharmaceuticals and healthcare products

    "UNH",    # UnitedHealth Group Incorporated
              # Health insurance and healthcare services


    # --------------------------------------------------------
    # Energy
    # --------------------------------------------------------

    "CVX",    # Chevron Corporation
              # Oil, natural gas and energy


    # --------------------------------------------------------
    # Consumer Defensive / Retail
    # --------------------------------------------------------

    "KO",     # The Coca-Cola Company
              # Beverages and consumer products

    "WMT",    # Walmart Inc.
              # Global retail and supermarkets

    "COST",   # Costco Wholesale Corporation
              # Membership-based warehouse retail

    "PG",     # Procter & Gamble Company
              # Consumer goods such as household and personal products


    # --------------------------------------------------------
    # Industrial
    # --------------------------------------------------------

    "CAT",    # Caterpillar Inc.
              # Construction and mining machinery

    "BA",     # The Boeing Company
              # Commercial aircraft, aerospace and defence
]


# ============================================================
# Market Benchmark
# ============================================================
#
# SPY is the SPDR S&P 500 ETF Trust.
#
# It tracks the S&P 500 index and gives us a representation of
# the overall US large-cap stock market.
#
# We keep SPY separate from our 20 companies because SPY is an
# ETF, not an individual company.
#
# Later we can use SPY to calculate "excess return".
#
# Example:
#
#     NVIDIA 30-day return = +10%
#     SPY 30-day return    =  +4%
#
#     Excess return = 10% - 4%
#                   = +6%
#
# This can eventually become the prediction target for XGBoost.
# ============================================================

BENCHMARK = "SPY"


# ============================================================
# extract_market_data()
# ============================================================

import pandas as pd
import yfinance as yf


MARKET_LOOKBACK_YEARS = 5


def extract_market_data():
    """
    Download five years of daily OHLCV data for the
    AlphaLens stock universe plus SPY benchmark.
    """

    all_tickers = TICKERS + [BENCHMARK]

    # Calculate a rolling five-year cutoff.
    #
    # Example:
    #
    # today:
    #     2026-08-24
    #
    # start:
    #     2021-08-24
    #
    start_date = (
        pd.Timestamp.now()
        .normalize()
        - pd.DateOffset(years=MARKET_LOOKBACK_YEARS)
    )

    data = yf.download(
        tickers=all_tickers,

        # yfinance accepts a YYYY-MM-DD string.
        start=start_date.strftime("%Y-%m-%d"),

        interval="1d",
        group_by="ticker",
        auto_adjust=False,
        progress=True,

        # Keep this False because we previously encountered
        # yfinance's internal SQLite "database is locked" error
        # when downloading concurrently.
        threads=False,
    )

    return data

# ============================================================
# Script Entry Point
# ============================================================
#
# Python automatically sets:
#
#     __name__ == "__main__"
#
# when this file is executed directly.
#
# For example:
#
#     python -m pipelines.market_data.extractor
#
# This allows us to test the extractor independently.
#
# If another Python module imports:
#
#     from pipelines.market_data.extractor import extract_market_data
#
# this test section will NOT automatically execute.
# ============================================================

if __name__ == "__main__":

    # Download the market data.
    df = extract_market_data()

    # --------------------------------------------------------
    # Check whether any ticker returned completely empty data
    # --------------------------------------------------------
    
    print("\nChecking downloaded symbols...")

    for ticker in TICKERS + [BENCHMARK]:

        # Select all OHLCV columns belonging to this ticker.
        ticker_data = df[ticker]

        # If every value in every row is missing, then the
        # ticker probably failed to download.
        if ticker_data.isna().all().all():
            print(f"[FAILED] {ticker} contains no data")
        else:
            print(f"[OK] {ticker}")

    # --------------------------------------------------------
    # Basic validation / inspection
    # --------------------------------------------------------

    print("\nDownload complete.")

    # DataFrame shape is:
    #
    #     (number_of_rows, number_of_columns)
    #
    # Rows are normally trading dates.
    print("\nData shape:")
    print(df.shape)

    # Show the first five trading dates.
    #
    # This is useful for verifying that actual market data
    # was returned instead of an empty DataFrame.
    print("\nFirst 5 rows:")
    print(df.head())

    # Show the column structure.
    #
    # Because multiple tickers were downloaded with
    # group_by="ticker", this will normally be a MultiIndex.
    #
    # The next AlphaLens step will be transformer.py,
    # which converts this structure into normal database rows.
    print("\nColumns:")
    print(df.columns)