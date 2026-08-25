"""
AlphaLens - SEC Filing Section Extractor

Purpose
-------
Extract meaningful sections from cleaned SEC 10-K and 10-Q filings.

Examples:

10-K:
    Item 1   - Business
    Item 1A  - Risk Factors
    Item 7   - Management's Discussion and Analysis
    Item 7A  - Market Risk
    Item 8   - Financial Statements

10-Q:
    Part I Item 1   - Financial Statements
    Part I Item 2   - Management's Discussion and Analysis
    Part I Item 3   - Market Risk
    Part I Item 4   - Controls and Procedures
    Part II Item 1A - Risk Factors


Pipeline
--------

SEC HTML
    ↓
parser.py
    ↓
clean filing text
    ↓
section_extractor.py
    ↓
meaningful filing sections
    ↓
PostgreSQL filing_sections


Why this is more complicated than just searching "ITEM 1A"
-----------------------------------------------------------

SEC filings often contain a Table of Contents.

For example:

    TABLE OF CONTENTS

    Item 1. Business ................................. 4
    Item 1A. Risk Factors ........................... 12
    Item 2. Properties .............................. 35

Later in the same document:

    ITEM 1. BUSINESS

    Apple designs, manufactures...

    ...many pages of actual content...

    ITEM 1A. RISK FACTORS

Therefore:

    re.search("ITEM 1A")

is NOT enough.

It might return the Table-of-Contents occurrence instead of the
actual section.

This extractor:

    1. Finds ALL occurrences of the heading.
    2. Builds a possible section from each occurrence.
    3. Looks for Table-of-Contents clues.
    4. Rejects obviously tiny candidates.
    5. Prefers non-TOC candidates.
    6. Chooses the most plausible actual section.
"""

import os
import re

from pathlib import Path

from dotenv import load_dotenv

from sqlalchemy import (
    MetaData,
    Table,
    create_engine,
    text,
)

from sqlalchemy.dialects.postgresql import insert


# ============================================================
# Environment
# ============================================================

load_dotenv()


# ============================================================
# Configuration
# ============================================================

# A section smaller than this is unlikely to be useful.
#
# IMPORTANT:
# This does NOT mean:
#
#     < 500 = definitely Table of Contents
#
# It only means:
#
#     < 500 = too small to keep as one of our RAG sections.
#
MIN_SECTION_CHARS = 50


# How many characters BEFORE a heading we inspect when
# trying to determine whether it is inside a Table of Contents.
TOC_CONTEXT_CHARS = 1500


# How many characters AFTER a heading we inspect.
#
# A Table of Contents often contains many ITEM headings
# very close together:
#
# Item 1
# Item 1A
# Item 1B
# Item 2
# Item 3
#
TOC_NEARBY_CHARS = 3000


# A heading appearing very early in the document receives
# one small "possible TOC" clue.
#
# 0.15 = first 15% of the filing.
EARLY_DOCUMENT_RATIO = 0.15


# Set this to True if you want to see every candidate that
# the extractor considers.
#
# Very useful while developing/debugging.
DEBUG_SECTION_MATCHES = False


# ============================================================
# Regex Flags
# ============================================================

REGEX_FLAGS = (
    re.IGNORECASE
    | re.MULTILINE
)


# IMPORTANT:
#
# We use MULTILINE but NOT DOTALL.
#
#
# MULTILINE means:
#
#     ^ = beginning of a LINE
#     $ = end of a LINE
#
# rather than only beginning/end of the entire document.
#
#
# This lets us match:
#
#     ITEM 1A. RISK FACTORS
#
# when it occurs on its own line.
#
#
# We intentionally DO NOT use:
#
#     re.DOTALL
#
# because DOTALL causes "." to also match newline characters.
#
# If we used DOTALL, a pattern ending in:
#
#     .*$
#
# could accidentally consume huge parts of the filing.


# ============================================================
# 10-K Section Definitions
# ============================================================

TEN_K_SECTIONS = [

    # --------------------------------------------------------
    # ITEM 1 - BUSINESS
    # --------------------------------------------------------

    {
        "key": "item_1_business",

        "title": "Business",

        "start":
            r"^\s*ITEM\s+1[\.\s:\-]+BUSINESS\b.*$",

        "ends": [
            r"^\s*ITEM\s+1A\b.*$",
        ],
    },


    # --------------------------------------------------------
    # ITEM 1A - RISK FACTORS
    # --------------------------------------------------------

    {
        "key": "item_1a_risk_factors",

        "title": "Risk Factors",

        "start":
            r"^\s*ITEM\s+1A[\.\s:\-]+RISK\s+FACTORS\b.*$",

        # Why multiple possible endings?
        #
        # Different companies / years may contain:
        #
        #     Item 1B
        #     Item 1C
        #     Item 2
        #
        # For example, if Item 1B is absent, Risk Factors
        # might run directly until Item 1C or Item 2.
        #
        # We search for ALL of them and choose whichever
        # appears first after Item 1A.
        "ends": [
            r"^\s*ITEM\s+1B\b.*$",
            r"^\s*ITEM\s+1C\b.*$",
            r"^\s*ITEM\s+2\b.*$",
        ],
    },


    # --------------------------------------------------------
    # ITEM 7 - MD&A
    # --------------------------------------------------------

    {
        "key": "item_7_mda",

        "title":
            "Management's Discussion and Analysis",

        "start":
            r"^\s*ITEM\s+7[\.\s:\-]+"
            r"MANAGEMENT['’]?S\s+DISCUSSION.*$",

        "ends": [
            r"^\s*ITEM\s+7A\b.*$",
            r"^\s*ITEM\s+8\b.*$",
        ],
    },


    # --------------------------------------------------------
    # ITEM 7A - MARKET RISK
    # --------------------------------------------------------

    {
        "key": "item_7a_market_risk",

        "title":
            "Quantitative and Qualitative Disclosures "
            "About Market Risk",

        "start":
            r"^\s*ITEM\s+7A[\.\s:\-]+"
            r"QUANTITATIVE\s+AND\s+QUALITATIVE.*$",

        "ends": [
            r"^\s*ITEM\s+8\b.*$",
        ],
    },


    # --------------------------------------------------------
    # ITEM 8 - FINANCIAL STATEMENTS
    # --------------------------------------------------------

    {
        "key": "item_8_financial_statements",

        "title":
            "Financial Statements and Supplementary Data",

        "start":
            r"^\s*ITEM\s+8[\.\s:\-]+"
            r"FINANCIAL\s+STATEMENTS.*$",

        "ends": [
            r"^\s*ITEM\s+9\b.*$",
            r"^\s*PART\s+III\b.*$",
        ],
    },
]


# ============================================================
# 10-Q Section Definitions
# ============================================================

TEN_Q_SECTIONS = [

    # --------------------------------------------------------
    # PART I - ITEM 1
    # Financial Statements
    # --------------------------------------------------------

    {
        "key": "part1_item1_financial_statements",

        "title": "Financial Statements",

        "start":
            r"^\s*ITEM\s+1[\.\s:\-]+"
            r"FINANCIAL\s+STATEMENTS\b.*$",

        "ends": [
            r"^\s*ITEM\s+2[\.\s:\-]+"
            r"MANAGEMENT.*$",
        ],
    },


    # --------------------------------------------------------
    # PART I - ITEM 2
    # MD&A
    # --------------------------------------------------------

    {
        "key": "part1_item2_mda",

        "title":
            "Management's Discussion and Analysis",

        "start":
            r"^\s*ITEM\s+2[\.\s:\-]+"
            r"MANAGEMENT['’]?S\s+DISCUSSION.*$",

        "ends": [
            r"^\s*ITEM\s+3\b.*$",
        ],
    },


    # --------------------------------------------------------
    # PART I - ITEM 3
    # Market Risk
    # --------------------------------------------------------

    {
        "key": "part1_item3_market_risk",

        "title":
            "Quantitative and Qualitative Disclosures "
            "About Market Risk",

        "start":
            r"^\s*ITEM\s+3[\.\s:\-]+"
            r"QUANTITATIVE\s+AND\s+QUALITATIVE.*$",

        "ends": [
            r"^\s*ITEM\s+4\b.*$",
        ],
    },


    # --------------------------------------------------------
    # PART I - ITEM 4
    # Controls
    # --------------------------------------------------------

    {
        "key": "part1_item4_controls",

        "title": "Controls and Procedures",

        "start":
            r"^\s*ITEM\s+4[\.\s:\-]+"
            r"CONTROLS\s+AND\s+PROCEDURES\b.*$",

        "ends": [
            r"^\s*PART\s+II\b.*$",
        ],
    },


    # --------------------------------------------------------
    # PART II - ITEM 1A
    # Risk Factors
    # --------------------------------------------------------

    {
        "key": "part2_item1a_risk_factors",

        "title": "Risk Factors",

        "start":
            r"^\s*ITEM\s+1A[\.\s:\-]+"
            r"RISK\s+FACTORS\b.*$",

        "ends": [
            r"^\s*ITEM\s+2\b.*$",
            r"^\s*ITEM\s+3\b.*$",
            r"^\s*ITEM\s+4\b.*$",
            r"^\s*ITEM\s+5\b.*$",
            r"^\s*ITEM\s+6\b.*$",
        ],
    },
]


# ============================================================
# get_database_engine()
# ============================================================

def get_database_engine():
    """
    Create a SQLAlchemy PostgreSQL engine.

    DATABASE_URL comes from .env.
    """

    database_url = os.getenv(
        "DATABASE_URL"
    )

    if not database_url:

        raise ValueError(
            "DATABASE_URL was not found in .env."
        )

    return create_engine(
        database_url,
        pool_pre_ping=True,
    )


# ============================================================
# get_parsed_filings()
# ============================================================

def get_parsed_filings(
    engine,
):
    """
    Retrieve SEC filings that already completed:

        HTML
          ↓
        clean text

    We can only extract sections if clean_text_path exists.
    """

    query = text(
        """
        SELECT
            accession_number,
            ticker,
            form_type,
            clean_text_path

        FROM filings

        WHERE parse_status = 'PARSED'
          AND clean_text_path IS NOT NULL

        ORDER BY
            ticker,
            form_type;
        """
    )

    with engine.connect() as connection:

        result = connection.execute(
            query
        )

        filings = (
            result
            .mappings()
            .all()
        )

    return filings


# ============================================================
# is_likely_toc_occurrence()
# ============================================================

def is_likely_toc_occurrence(
    filing_text: str,
    match: re.Match,
) -> bool:
    """
    Decide whether a particular heading occurrence probably
    belongs to the Table of Contents.

    This is a HEURISTIC, not an absolute rule.

    We look for multiple clues.

    Example TOC:

        TABLE OF CONTENTS

        Item 1. Business ...................... 4
        Item 1A. Risk Factors ................ 12
        Item 1B. Unresolved Staff Comments ... 30
        Item 2. Properties ................... 31


    Clues used:
        1. "TABLE OF CONTENTS" appears shortly before heading.
        2. Heading contains dot leaders + page number.
        3. Many ITEM headings occur very close together.
        4. Heading occurs very early in the filing.

    We do NOT mark something as TOC just because it is early.
    We require multiple clues.
    """

    score = 0

    document_length = len(
        filing_text
    )

    heading_start = (
        match.start()
    )

    heading_line = (
        match.group(0)
        .strip()
    )


    # --------------------------------------------------------
    # CLUE 1
    # "TABLE OF CONTENTS" nearby before heading
    # --------------------------------------------------------

    context_start = max(
        0,
        heading_start - TOC_CONTEXT_CHARS,
    )

    previous_context = (
        filing_text[
            context_start:heading_start
        ]
        .upper()
    )

    if "TABLE OF CONTENTS" in previous_context:

        # Strong clue.
        score += 3


    # --------------------------------------------------------
    # CLUE 2
    # Dot leaders + page number
    # --------------------------------------------------------
    #
    # Example:
    #
    # ITEM 1A. RISK FACTORS ............... 17
    #
    # \.{3,}
    #     means three or more literal dots.
    #
    # \s*
    #     means optional spaces.
    #
    # \d+
    #     means one or more digits.
    #
    # $
    #     means end of line.
    # --------------------------------------------------------

    if re.search(
        r"\.{3,}\s*\d+\s*$",
        heading_line,
    ):
        score += 2


    # --------------------------------------------------------
    # CLUE 3
    # Many ITEM headings very close together
    # --------------------------------------------------------
    #
    # In the real Business section, you normally do NOT see:
    #
    #     Item 1
    #     Item 1A
    #     Item 1B
    #     Item 2
    #
    # all within a few hundred characters.
    #
    # But that is common in a Table of Contents.
    # --------------------------------------------------------

    nearby_end = min(
        document_length,
        heading_start + TOC_NEARBY_CHARS,
    )

    nearby_text = filing_text[
        heading_start:nearby_end
    ]

    nearby_item_headings = re.findall(
        r"^\s*ITEM\s+\d+[A-Z]?\b.*$",
        nearby_text,
        flags=REGEX_FLAGS,
    )

    if len(nearby_item_headings) >= 3:

        score += 1


    # --------------------------------------------------------
    # CLUE 4
    # Very early in document
    # --------------------------------------------------------
    #
    # This is only a weak clue.
    #
    # Actual Item 1 can legitimately appear fairly early,
    # so this alone must NEVER classify it as TOC.
    # --------------------------------------------------------

    if document_length > 0:

        position_ratio = (
            heading_start
            / document_length
        )

        if (
            position_ratio
            < EARLY_DOCUMENT_RATIO
        ):
            score += 1


    # --------------------------------------------------------
    # Decision
    # --------------------------------------------------------
    #
    # Require at least TWO points.
    #
    # Examples:
    #
    # early only
    #     score = 1
    #     NOT TOC
    #
    # dot leaders
    #     score = 2
    #     likely TOC
    #
    # early + many nearby headings
    #     score = 3
    #     likely TOC
    # --------------------------------------------------------

    return score >= 3


# ============================================================
# find_next_heading()
# ============================================================

def find_next_heading(
    filing_text: str,
    search_from: int,
    end_patterns: list[str],
) -> int:
    """
    Find where the current section should stop.

    Example:

        ITEM 1A. RISK FACTORS
        ...
        thousands of characters
        ...
        ITEM 1B. UNRESOLVED STAFF COMMENTS


    For Risk Factors:

        start = Item 1A

        possible endings:
            Item 1B
            Item 1C
            Item 2


    We search for ALL possible endings and use whichever one
    occurs FIRST after the current heading.


    Parameters
    ----------
    filing_text:
        Entire cleaned filing.

    search_from:
        Character position AFTER the current start heading.

    end_patterns:
        Regex patterns identifying possible next sections.


    Returns
    -------
    int:
        Absolute character position where section ends.

        If no ending heading is found:
            return end of entire filing.
    """

    possible_end_positions = []


    # Only search AFTER the current section heading.
    #
    # There might be previous "ITEM 2" references earlier in
    # the filing and we absolutely don't want those.
    remaining_text = filing_text[
        search_from:
    ]


    for pattern in end_patterns:

        match = re.search(
            pattern,
            remaining_text,
            flags=REGEX_FLAGS,
        )

        if match:

            # match.start() is relative to remaining_text.
            #
            # Convert it back into an absolute character
            # position within filing_text.
            absolute_position = (
                search_from
                + match.start()
            )

            possible_end_positions.append(
                absolute_position
            )


    # No next heading found.
    #
    # In that case let this candidate extend until EOF.
    if not possible_end_positions:

        return len(
            filing_text
        )


    # Example:
    #
    # Item 1B found at position 30,000
    # Item 1C found at position 41,000
    # Item 2  found at position 43,000
    #
    # Current section should end at:
    #
    #     30,000
    #
    return min(
        possible_end_positions
    )


# ============================================================
# build_section_candidates()
# ============================================================

def build_section_candidates(
    filing_text: str,
    section_definition: dict,
) -> list[dict]:
    """
    Find EVERY occurrence of one section heading and construct
    a candidate section for each occurrence.

    Example:

        "ITEM 1A RISK FACTORS"

    appears at:

        character 12,000     ← Table of Contents

        character 87,000     ← actual Risk Factors section


    We create TWO candidates.

    Later choose_best_candidate() decides which one is real.

    Important:
        Previously, candidates smaller than MIN_SECTION_CHARS
        were discarded immediately.

        That meant DEBUG_SECTION_MATCHES could not show them.

    This version keeps every regex match first.

    Each candidate records:

        occurrence
            Which heading occurrence this is.

        text
            Text from this heading until the next section.

        length
            Number of characters extracted.

        likely_toc
            Whether our heuristic thinks the heading may be
            inside a Table of Contents.

        rejected
            Whether this candidate is too small to use.

        rejection_reason
            Why it was rejected.

    This makes debugging much clearer.
    """

    

    start_pattern = (
        section_definition["start"]
    )

    end_patterns = (
        section_definition["ends"]
    )

    # --------------------------------------------------------
    # Find ALL headings matching the start regex.
    # --------------------------------------------------------
    #
    # Example:
    #
    # Match 1:
    #     Table of Contents occurrence
    #
    # Match 2:
    #     Actual MD&A occurrence
    #
    matches = list(
        re.finditer(
            start_pattern,
            filing_text,
            flags=REGEX_FLAGS,
        )
    )

    candidates = []

    for occurrence_number, match in enumerate(
        matches,
        start=1,
    ):

        section_start = (
            match.start()
        )

        # ----------------------------------------------------
        # Find where this possible section ends.
        # ----------------------------------------------------

        section_end = find_next_heading(
            filing_text=filing_text,
            search_from=match.end(),
            end_patterns=end_patterns,
        )

        # ----------------------------------------------------
        # Extract the possible section text.
        # ----------------------------------------------------

        section_text = (
            filing_text[
                section_start:section_end
            ]
            .strip()
        )

        section_length = len(
            section_text
        )

        # ----------------------------------------------------
        # Check whether this occurrence resembles a TOC entry.
        # ----------------------------------------------------

        likely_toc = (
            is_likely_toc_occurrence(
                filing_text=filing_text,
                match=match,
            )
        )

        # ----------------------------------------------------
        # Calculate where heading appears in document.
        #
        # Example:
        #
        # 0.04 = 4% through filing
        # 0.45 = 45% through filing
        # ----------------------------------------------------

        if filing_text:

            position_ratio = (
                section_start
                / len(filing_text)
            )

        else:

            position_ratio = 0


        # ----------------------------------------------------
        # Instead of immediately using "continue",
        # keep the candidate and mark whether it is rejected.
        # ----------------------------------------------------

        rejected = False
        rejection_reason = None

        if (
            section_length
            < MIN_SECTION_CHARS
        ):

            rejected = True

            rejection_reason = (
                f"Section is smaller than "
                f"MIN_SECTION_CHARS "
                f"({MIN_SECTION_CHARS})"
            )


        candidates.append(
            {
                "occurrence":
                    occurrence_number,

                "text":
                    section_text,

                "start_position":
                    section_start,

                "end_position":
                    section_end,

                "length":
                    section_length,

                "likely_toc":
                    likely_toc,

                "position_ratio":
                    position_ratio,

                "heading":
                    match.group(0).strip(),

                # New debugging information
                "rejected":
                    rejected,

                "rejection_reason":
                    rejection_reason,
            }
        )

    return candidates

# ============================================================
# choose_best_candidate()
# ============================================================

def choose_best_candidate(
    candidates: list[dict],
):
    """
    Choose the most likely REAL section from all heading
    occurrences.

    Selection strategy
    ------------------

    STEP 1:
        If we have candidates that do NOT look like TOC,
        ignore likely-TOC candidates.

    STEP 2:
        Compare remaining candidates by section length.

        Actual sections usually contain significantly more
        text than references or accidental matches.

    STEP 3:
        If candidates have equal/similar length, prefer the
        one appearing later in the document.

        This helps because the Table of Contents normally
        occurs before the real section.


    Why don't we simply choose the last occurrence?
    -----------------------------------------------

    Because headings can also appear later in:

        references
        exhibits
        repeated navigation text

    So position alone is not enough.


    Why don't we simply choose the longest occurrence?
    --------------------------------------------------

    Because a false match could sometimes accidentally span
    a lot of text.

    Therefore we combine:

        TOC detection
        +
        size
        +
        position
    """

    if not candidates:

        return None

    # ========================================================
    # STEP 1 - Remove invalid / tiny candidates
    # ========================================================

    valid_candidates = [
        candidate
        for candidate in candidates
        if not candidate["rejected"]
    ]

    if not valid_candidates:

        return None


    # ========================================================
    # STEP 2 - Prefer candidates not identified as TOC
    # ========================================================

    non_toc_candidates = [
        candidate
        for candidate in valid_candidates
        if not candidate["likely_toc"]
    ]


    if non_toc_candidates:

        candidate_pool = (
            non_toc_candidates
        )

    else:

        # All surviving candidates were classified as possible
        # TOC entries.
        #
        # Don't automatically throw them away because the
        # heuristic could be wrong.
        candidate_pool = (
            valid_candidates
        )



    # ========================================================
    # STEP 3 - Determine largest candidate
    # ========================================================

    max_length = max(
        candidate["length"]
        for candidate in candidate_pool
    )


    # ========================================================
    # STEP 4 - Keep substantial candidates
    # ========================================================
    #
    # Suppose:
    #
    # candidate 1 = 70,000 chars
    # candidate 2 = 68,000 chars
    #
    # Both are substantial.
    #
    # But:
    #
    # candidate 1 = 70,000 chars
    # candidate 2 = 3,000 chars
    #
    # candidate 2 is much less plausible.
    #
    #
    # We keep candidates at least 40% as long as the largest
    # candidate.
    #
    # This isn't a universal SEC rule; it is a practical
    # heuristic for AlphaLens.
    # --------------------------------------------------------

    substantial_candidates = [
        candidate
        for candidate in candidate_pool

        if candidate["length"]
        >= max_length * 0.40
    ]


    # ========================================================
    # STEP 5 - Among similarly substantial candidates, prefer later.
    # ========================================================
    #
    # This specifically helps:
    #
    #     early Table of Contents occurrence
    #
    # versus
    #
    #     later actual filing occurrence
    # --------------------------------------------------------

    best_candidate = max(
        substantial_candidates,
        key=lambda candidate:
            candidate["start_position"],
    )


    return best_candidate


# ============================================================
# print_candidate_debug()
# ============================================================

def print_candidate_debug(
    section_definition: dict,
    candidates: list[dict],
    chosen_candidate,
):
    """
    Print EVERY regex match found for this section.

    This includes candidates that were rejected for being
    too small.

    This lets us understand exactly what the regex detected.
    """

    print()

    print(
        "------------------------------------------------"
    )

    print(
        f"SECTION: "
        f"{section_definition['title']}"
    )

    print(
        "------------------------------------------------"
    )


    if not candidates:

        print(
            "No regex matches found."
        )

        return


    for candidate in candidates:

        print(
            f"Candidate "
            f"{candidate['occurrence']}"
        )

        print(
            f"  heading: "
            f"{candidate['heading'][:150]}"
        )

        print(
            f"  position: "
            f"{candidate['position_ratio']:.1%}"
        )

        print(
            f"  chars: "
            f"{candidate['length']:,}"
        )

        print(
            f"  likely TOC: "
            f"{candidate['likely_toc']}"
        )

        print(
            f"  rejected: "
            f"{candidate['rejected']}"
        )


        # Only display rejection reason if there is one.
        if candidate["rejection_reason"]:

            print(
                f"  reason: "
                f"{candidate['rejection_reason']}"
            )


        print()


    print(
        "CHOSEN:"
    )

    if chosen_candidate:

        print(
            f"  Candidate "
            f"{chosen_candidate['occurrence']}"
        )

        print(
            f"  chars: "
            f"{chosen_candidate['length']:,}"
        )

        print(
            f"  likely TOC: "
            f"{chosen_candidate['likely_toc']}"
        )

    else:

        print(
            "  None"
        )


# ============================================================
# extract_best_section()
# ============================================================

def extract_best_section(
    filing_text: str,
    section_definition: dict,
):
    """
    Extract the most likely real occurrence of one SEC section.

    This function coordinates:

        find all headings
            ↓
        create candidates
            ↓
        detect possible TOC candidates
            ↓
        choose best candidate
            ↓
        return its text
    """

    candidates = build_section_candidates(
        filing_text=filing_text,
        section_definition=section_definition,
    )


    best_candidate = (
        choose_best_candidate(
            candidates
        )
    )


    if DEBUG_SECTION_MATCHES:

        print_candidate_debug(
            section_definition=
                section_definition,

            candidates=
                candidates,

            chosen_candidate=
                best_candidate,
        )


    if best_candidate is None:

        return None


    return best_candidate[
        "text"
    ]


# ============================================================
# extract_sections()
# ============================================================

def extract_sections(
    filing_text: str,
    form_type: str,
) -> list[dict]:
    """
    Extract all AlphaLens sections from one filing.

    Parameters
    ----------
    filing_text:
        Full cleaned SEC filing.

    form_type:
        10-K or 10-Q


    Returns
    -------
    list of dictionaries

    Example:

        [
            {
                "section_key":
                    "item_1a_risk_factors",

                "section_title":
                    "Risk Factors",

                "content":
                    "ITEM 1A. RISK FACTORS ...",

                "char_count":
                    68420
            }
        ]
    """

    if form_type == "10-K":

        definitions = (
            TEN_K_SECTIONS
        )

    elif form_type == "10-Q":

        definitions = (
            TEN_Q_SECTIONS
        )

    else:

        # AlphaLens currently doesn't extract other forms.
        return []


    extracted_sections = []


    for definition in definitions:

        content = extract_best_section(
            filing_text=filing_text,
            section_definition=definition,
        )


        # Heading wasn't detected.
        if content is None:

            continue


        extracted_sections.append(
            {
                "section_key":
                    definition["key"],

                "section_title":
                    definition["title"],

                "content":
                    content,

                "char_count":
                    len(content),
            }
        )


    return extracted_sections


# ============================================================
# save_sections()
# ============================================================

def save_sections(
    engine,
    accession_number: str,
    sections: list[dict],
) -> int:
    """
    Insert extracted sections into PostgreSQL.

    Unique key:

        accession_number
        +
        section_key


    Example:

        accession:
            0000320193-25-000079

        section:
            item_1a_risk_factors


    This combination is unique.

    Therefore rerunning the parser updates that section rather
    than creating duplicate records.
    """

    if not sections:

        return 0


    records = []


    for section in sections:

        records.append(
            {
                "accession_number":
                    accession_number,

                "section_key":
                    section["section_key"],

                "section_title":
                    section["section_title"],

                "content":
                    section["content"],

                "char_count":
                    section["char_count"],
            }
        )


    metadata = MetaData()


    filing_sections_table = Table(
        "filing_sections",
        metadata,
        autoload_with=engine,
    )


    insert_statement = insert(
        filing_sections_table
    )


    # --------------------------------------------------------
    # PostgreSQL UPSERT
    # --------------------------------------------------------
    #
    # New:
    #
    #     INSERT
    #
    # Already exists:
    #
    #     UPDATE
    #
    #
    # This means you can improve the extraction algorithm and
    # safely rerun this script.
    # --------------------------------------------------------

    upsert_statement = (
        insert_statement
        .on_conflict_do_update(

            index_elements=[
                "accession_number",
                "section_key",
            ],

            set_={
                "section_title":
                    insert_statement.excluded.section_title,

                "content":
                    insert_statement.excluded.content,

                "char_count":
                    insert_statement.excluded.char_count,

                "updated_at":
                    text("NOW()"),
            },
        )
    )


    with engine.begin() as connection:

        connection.execute(
            upsert_statement,
            records,
        )


    return len(
        records
    )


# ============================================================
# process_filing()
# ============================================================

def process_filing(
    engine,
    filing,
) -> int:
    """
    Process one SEC filing:

        clean text
            ↓
        extract sections
            ↓
        save sections into PostgreSQL
    """

    accession_number = (
        filing["accession_number"]
    )

    ticker = (
        filing["ticker"]
    )

    form_type = (
        filing["form_type"]
    )

    clean_text_path = Path(
        filing["clean_text_path"]
    )


    # --------------------------------------------------------
    # Validate source file
    # --------------------------------------------------------

    if not clean_text_path.exists():

        print(
            f"[FAILED] "
            f"{ticker} {form_type}: "
            "clean text file does not exist"
        )

        return 0


    # --------------------------------------------------------
    # Read complete filing
    # --------------------------------------------------------

    filing_text = (
        clean_text_path.read_text(
            encoding="utf-8"
        )
    )


    # --------------------------------------------------------
    # Extract sections
    # --------------------------------------------------------

    sections = extract_sections(
        filing_text=filing_text,
        form_type=form_type,
    )


    # --------------------------------------------------------
    # Save sections
    # --------------------------------------------------------

    stored_count = save_sections(
        engine=engine,
        accession_number=accession_number,
        sections=sections,
    )


    print(
        f"[OK] "
        f"{ticker} {form_type}: "
        f"{stored_count} sections extracted"
    )


    # Print each extracted section and its size.
    #
    # This is useful for spotting suspicious results.
    #
    # Example:
    #
    # Risk Factors: 70,000
    #
    # looks realistic.
    #
    # Risk Factors: 600
    #
    # should probably be investigated.
    for section in sections:

        print(
            f"     - "
            f"{section['section_title']}: "
            f"{section['char_count']:,} chars"
        )


    return stored_count


# ============================================================
# run_section_extraction()
# ============================================================

def run_section_extraction():
    """
    Run section extraction for every parsed SEC filing.
    """

    engine = get_database_engine()


    filings = get_parsed_filings(
        engine
    )


    print(
        f"\nParsed filings found: "
        f"{len(filings)}"
    )


    total_sections = 0


    for filing in filings:

        total_sections += process_filing(
            engine=engine,
            filing=filing,
        )


    print()

    print(
        "========================================"
    )

    print(
        "SECTION EXTRACTION COMPLETE"
    )

    print(
        "========================================"
    )

    print(
        f"Total sections stored: "
        f"{total_sections}"
    )


# ============================================================
# Script Entry Point
# ============================================================

if __name__ == "__main__":

    run_section_extraction()
