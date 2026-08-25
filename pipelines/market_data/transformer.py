"""
AlphaLens - Market Data Transformer

Purpose:
    Converts the raw Yahoo Finance DataFrame returned by extractor.py
    into a clean row-based format suitable for PostgreSQL.

Raw Yahoo Finance format:
    The DataFrame uses MultiIndex columns when multiple tickers are
    downloaded with group_by="ticker".

    Example:

                  AAPL                     MSFT
                  Open   Close   Volume     Open   Close   Volume
    Date
    2026-08-20    ...    ...     ...        ...    ...     ...

Target format:
    We want one row per ticker per trading day.

    ticker | trading_date | open | high | low | close | adjusted_close | volume
    AAPL   | 2026-08-20   | ...  | ...  | ... | ...   | ...            | ...
    MSFT   | 2026-08-20   | ...  | ...  | ... | ...   | ...            | ...

Why?
    This row-based structure is much easier to:

        - insert into PostgreSQL
        - query by ticker
        - filter by date
        - calculate factors
        - join with other financial datasets
"""

import pandas as pd


def transform_market_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw Yahoo Finance data into normalized database rows.

    Parameters
    ----------
    raw_data : pandas.DataFrame
        Raw DataFrame returned by yfinance.

        Expected MultiIndex structure:

            Level 1 = ticker
            Level 2 = OHLCV field

        Example:

            ("AAPL", "Open")
            ("AAPL", "Close")
            ("MSFT", "Open")
            ("MSFT", "Close")

    Returns
    -------
    pandas.DataFrame
        Clean DataFrame with columns:

            ticker
            trading_date
            open
            high
            low
            close
            adjusted_close
            volume

    Each row represents:

        one stock
        +
        one trading date
    """

    # ========================================================
    # Validation 1 - Make sure some data was actually provided
    # ========================================================

    if raw_data.empty:
        raise ValueError(
            "Raw market data is empty. "
            "The Yahoo Finance download may have failed."
        )

    # ========================================================
    # Validation 2 - Check for MultiIndex columns
    # ========================================================
    #
    # Because extractor.py uses:
    #
    #     group_by="ticker"
    #
    # Yahoo Finance should return columns like:
    #
    #     ("AAPL", "Open")
    #     ("AAPL", "Close")
    #
    # Pandas calls this a MultiIndex.
    # ========================================================

    if not isinstance(raw_data.columns, pd.MultiIndex):
        raise ValueError(
            "Expected Yahoo Finance data to contain MultiIndex columns."
        )

    # This list will temporarily hold one cleaned DataFrame
    # for each ticker.
    transformed_frames = []

    # ========================================================
    # Find all ticker symbols inside the first column level
    # ========================================================
    #
    # Example:
    #
    # raw_data.columns level 0:
    #
    #     AAPL
    #     MSFT
    #     NVDA
    #     ...
    #
    # .unique() removes repeated ticker names.
    # ========================================================

    tickers = raw_data.columns.get_level_values(0).unique()

    print(f"Transforming {len(tickers)} symbols...")

    # ========================================================
    # Process one ticker at a time
    # ========================================================

    for ticker in tickers:

        # ----------------------------------------------------
        # Extract this ticker's OHLCV columns
        # ----------------------------------------------------
        #
        # Example:
        #
        # raw_data["AAPL"]
        #
        # becomes:
        #
        # Date         Open   High   Low   Close   Adj Close   Volume
        # 2026-08-20   ...    ...    ...   ...     ...         ...
        # ----------------------------------------------------

        ticker_data = raw_data[ticker].copy()

        # ----------------------------------------------------
        # Skip ticker if download completely failed
        # ----------------------------------------------------
        #
        # A failed Yahoo Finance download can still create
        # columns where every value is NaN.
        #
        # Example:
        #
        # KO
        # Open   NaN
        # High   NaN
        # ...
        # ----------------------------------------------------

        if ticker_data.isna().all().all():
            print(f"[SKIPPED] {ticker}: no valid data")
            continue

        # ----------------------------------------------------
        # Convert Date index into a normal column
        # ----------------------------------------------------
        #
        # Before:
        #
        # Date
        # 2026-08-20
        # 2026-08-21
        #
        # Date is currently the DataFrame index.
        #
        # reset_index() changes it into a normal column.
        # ----------------------------------------------------

        ticker_data = ticker_data.reset_index()

        # ====================================================
        # Standardize column names
        # ====================================================
        #
        # Yahoo Finance may return:
        #
        #     Date
        #     Open
        #     High
        #     Low
        #     Close
        #     Adj Close
        #     Volume
        #
        # We rename these to database-friendly lowercase names.
        # ====================================================

        ticker_data = ticker_data.rename(
            columns={
                "Date": "trading_date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Adj Close": "adjusted_close",
                "Volume": "volume",
            }
        )

        # ====================================================
        # Add ticker as its own normal column
        # ====================================================
        #
        # Before:
        #
        # ticker was stored in the MultiIndex.
        #
        # After:
        #
        # ticker | trading_date | open | close
        # AAPL   | 2026-08-20   | ...  | ...
        # ====================================================

        ticker_data["ticker"] = ticker

        # ====================================================
        # Keep only the columns AlphaLens requires
        # ====================================================

        expected_columns = [
            "ticker",
            "trading_date",
            "open",
            "high",
            "low",
            "close",
            "adjusted_close",
            "volume",
        ]

        # Some providers/data conditions may omit adjusted_close.
        # This check gives us a clearer error instead of silently
        # producing incorrect data.
        missing_columns = [
            column
            for column in expected_columns
            if column not in ticker_data.columns
        ]

        if missing_columns:
            raise ValueError(
                f"{ticker} is missing expected columns: {missing_columns}"
            )

        ticker_data = ticker_data[expected_columns]

        # ====================================================
        # Remove rows without a trading date
        # ====================================================

        ticker_data = ticker_data.dropna(
            subset=["trading_date"]
        )

        # ====================================================
        # Remove rows where ALL price fields are missing
        # ========================================================
        #
        # This prevents empty placeholder rows from entering
        # PostgreSQL.
        # ========================================================

        ticker_data = ticker_data.dropna(
            subset=[
                "open",
                "high",
                "low",
                "close",
            ],
            how="all",
        )

        # ====================================================
        # Normalize the trading_date type
        # ========================================================
        #
        # Keep this as pandas datetime64[ns] instead of converting it
        # to plain Python date objects. This preserves pandas datetime
        # operations such as .dt.year and .dt.month for later features.
        # ========================================================

        ticker_data["trading_date"] = pd.to_datetime(
            ticker_data["trading_date"]
        )

        # ====================================================
        # Normalize volume
        # ========================================================
        #
        # Volume should normally be an integer.
        #
        # Pandas nullable Int64 allows missing values if Yahoo
        # happens to return one.
        # ========================================================

        ticker_data["volume"] = (
            pd.to_numeric(
                ticker_data["volume"],
                errors="coerce",
            )
            .round()
            .astype("Int64")
        )

        # ====================================================
        # Remove duplicate ticker/date combinations
        # ========================================================
        #
        # PostgreSQL will eventually use:
        #
        #     PRIMARY KEY (ticker, trading_date)
        #
        # so duplicates are not allowed.
        # ========================================================

        ticker_data = ticker_data.drop_duplicates(
            subset=[
                "ticker",
                "trading_date",
            ]
        )

        # Save this cleaned ticker DataFrame.
        transformed_frames.append(ticker_data)

        print(
            f"[OK] {ticker}: "
            f"{len(ticker_data)} rows transformed"
        )

    # ========================================================
    # Combine every ticker back into one DataFrame
    # ========================================================

    if not transformed_frames:
        raise ValueError(
            "No valid ticker data remained after transformation."
        )

    clean_data = pd.concat(
        transformed_frames,
        # Give the combined DataFrame one fresh 0, 1, 2, ... index
        # instead of preserving duplicate date indexes from each ticker.
        ignore_index=True,
    )

    # Remove the leftover "Price" column-axis label inherited
    # from Yahoo Finance's original MultiIndex structure.
    clean_data.columns.name = None

    # ========================================================
    # Sort the final dataset
    # ========================================================
    #
    # Result:
    #
    # AAPL 2024-01-02
    # AAPL 2024-01-03
    # ...
    # MSFT 2024-01-02
    # MSFT 2024-01-03
    # ...
    # ========================================================

    clean_data = clean_data.sort_values(
        by=[
            "ticker",
            "trading_date",
        ]
        # Sorting changes the row order, so discard the old index and
        # create a new sequential index matching the sorted rows.
    ).reset_index(drop=True)

    return clean_data