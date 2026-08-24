"""
AlphaLens - SEC Filing Metadata Pipeline

Purpose:
    Runs the first complete SEC ETL workflow.

Pipeline:

    SEC EDGAR
        ↓
    Extract 10-K / 10-Q metadata
        ↓
    Transform into Pandas rows
        ↓
    Update companies
        ↓
    Insert / update filings
        ↓
    PostgreSQL
"""


from pipelines.sec.extractor import (
    extract_sec_filing_metadata,
)

from pipelines.sec.loader import (
    load_sec_data,
)


def run_sec_pipeline():
    """
    Run the AlphaLens SEC metadata ETL pipeline.
    """

    # ========================================================
    # STEP 1 - EXTRACT
    # ========================================================

    print("\n========================================")
    print("STEP 1 - SEC EXTRACT")
    print("========================================")

    filings = extract_sec_filing_metadata()

    print(
        f"\nSEC filings extracted: "
        f"{len(filings)}"
    )

    # ========================================================
    # STEP 2 - LOAD
    # ========================================================

    print("\n========================================")
    print("STEP 2 - POSTGRESQL LOAD")
    print("========================================")

    loaded_rows = load_sec_data(
        filings
    )

    print("\n========================================")
    print("SEC PIPELINE COMPLETE")
    print("========================================")

    print(
        f"Filings processed: "
        f"{loaded_rows}"
    )


if __name__ == "__main__":

    run_sec_pipeline()