"""
AlphaLens - Market Data Transformation Test

Purpose:
    Downloads raw Yahoo Finance data and immediately passes it
    through the AlphaLens transformer.

This file is temporary and is used only to verify that:

    extractor.py
        ↓
    transformer.py

works correctly before PostgreSQL is introduced.
"""

from pipelines.market_data.extractor import extract_market_data
from pipelines.market_data.transformer import transform_market_data


if __name__ == "__main__":

    # --------------------------------------------------------
    # Step 1 - Download raw Yahoo Finance data
    # --------------------------------------------------------

    raw_data = extract_market_data()

    print("\nRaw data shape:")
    print(raw_data.shape)

    # --------------------------------------------------------
    # Step 2 - Transform the raw data
    # --------------------------------------------------------

    clean_data = transform_market_data(raw_data)

    # --------------------------------------------------------
    # Step 3 - Inspect the result
    # --------------------------------------------------------

    print("\nTransformation complete.")

    print("\nClean data shape:")
    print(clean_data.shape)

    print("\nFirst 20 rows:")
    print(clean_data.head(20))

    print("\nColumns:")
    print(clean_data.columns.tolist())

    print("\nData types:")
    print(clean_data.dtypes)

    print("\nRows per ticker:")
    print(
        clean_data.groupby("ticker")
        .size()
        .sort_index()
    )