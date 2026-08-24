/*
============================================================
AlphaLens - SEC Filing Sections
============================================================

Purpose:
    Stores meaningful sections extracted from 10-K and
    10-Q filings.

Example:

    One filing:
        0000320193-25-000079

    can contain:

        Item 1  - Business
        Item 1A - Risk Factors
        Item 7  - MD&A
        Item 7A - Market Risk
        Item 8  - Financial Statements


Relationship:

    filings
       1
       │
       ▼
      many
    filing_sections
============================================================
*/


CREATE TABLE IF NOT EXISTS filing_sections (

    /*
    Internal AlphaLens identifier for a section.
    */
    section_id BIGSERIAL PRIMARY KEY,


    /*
    Which SEC filing this section came from.

    This references:

        filings.accession_number
    */
    accession_number VARCHAR(30) NOT NULL
        REFERENCES filings(accession_number)
        ON DELETE CASCADE,


    /*
    Stable machine-friendly section name.

    Examples:

        item_1_business
        item_1a_risk_factors
        item_7_mda
    */
    section_key VARCHAR(100) NOT NULL,


    /*
    Human-readable title.

    Example:

        Risk Factors
    */
    section_title VARCHAR(255) NOT NULL,


    /*
    Actual extracted filing text.
    */
    content TEXT NOT NULL,


    /*
    Number of characters in this section.

    Useful for:
        validation
        diagnostics
        later chunk planning
    */
    char_count INTEGER NOT NULL,


    /*
    When this section was first stored.
    */
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),


    /*
    Updated when we rerun/improve the section parser.
    */
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),


    /*
    One filing should only contain one stored version
    of each AlphaLens section key.
    */
    UNIQUE (
        accession_number,
        section_key
    )
);


-- Useful when finding every section belonging to one filing.
CREATE INDEX IF NOT EXISTS idx_filing_sections_accession
ON filing_sections (accession_number);


-- Useful later when searching specific section types.
CREATE INDEX IF NOT EXISTS idx_filing_sections_key
ON filing_sections (section_key);