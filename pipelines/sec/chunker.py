"""
AlphaLens - SEC Filing Chunker

Purpose
-------
Split large SEC filing sections into smaller pieces suitable
for:

    embeddings
        ↓
    FAISS vector search
        ↓
    Retrieval-Augmented Generation (RAG)


Current pipeline
----------------

SEC filing HTML
    ↓
parser.py
    ↓
clean filing text
    ↓
section_extractor.py
    ↓
filing_sections
    ↓
chunker.py
    ↓
filing_chunks
    ↓
OpenAI embeddings        <- NEXT
    ↓
FAISS


Why chunk sections?
-------------------

A section such as:

    Item 1A - Risk Factors

can contain tens of thousands of tokens.

It can discuss:

    cybersecurity
    regulation
    supply chain
    competition
    litigation
    foreign exchange
    privacy
    AI
    etc.

Embedding that entire section as ONE vector mixes all those
subjects together.

Instead:

    Risk Factors
        ↓
    chunk 0
    chunk 1
    chunk 2
    ...

allows vector search to retrieve only the text that is actually
relevant to the user's question.


Chunk configuration
-------------------

Target maximum:

    700 tokens

Overlap:

    100 tokens


Example:

    chunk 0:
        tokens 0 -> 699

    chunk 1:
        tokens 600 -> 1299

    chunk 2:
        tokens 1200 -> 1899


Why overlap?
------------

Imagine an important sentence begins near the end of chunk 0
and continues conceptually into chunk 1.

Without overlap:

    chunk 0
        ...cybersecurity incidents may...

    chunk 1
        ...result in significant operational disruption...

The ideas are separated.

With overlap, some text appears in both chunks so context near
the boundary is preserved.


Important:
    These values are STARTING PARAMETERS.

    Later, once AlphaLens RAG is working, we can test retrieval
    quality and adjust:

        chunk size
        overlap
        number of retrieved chunks
"""

import os

import tiktoken

from dotenv import load_dotenv

from sqlalchemy import (
    MetaData,
    Table,
    create_engine,
    delete,
    text,
)

from sqlalchemy.dialects.postgresql import insert


# ============================================================
# Environment
# ============================================================

load_dotenv()


# ============================================================
# Chunk Configuration
# ============================================================

# Maximum number of tokens in one chunk.
#
# A value around 500-800 tokens is a reasonable starting point
# for financial-document RAG.
#
MAX_CHUNK_TOKENS = 700


# Number of tokens repeated between consecutive chunks.
#
# Example:
#
# Chunk 0:
#     token 0 -> 699
#
# Chunk 1:
#     token 600 -> 1299
#
# therefore:
#
#     tokens 600 -> 699
#
# exist in BOTH chunks.
#
CHUNK_OVERLAP_TOKENS = 100


# ============================================================
# Token Encoding
# ============================================================
#
# tiktoken converts:
#
#     normal text
#
# into:
#
#     integer token IDs
#
# Example conceptually:
#
#     "Apple reported revenue growth"
#
# becomes something like:
#
#     [12345, 678, 9012, ...]
#
#
# We use cl100k_base as a stable tokenizer for estimating
# OpenAI-style token counts.
#
# The exact embedding model can be selected later.
# ============================================================

TOKEN_ENCODING_NAME = "cl100k_base"


ENCODING = tiktoken.get_encoding(
    TOKEN_ENCODING_NAME
)


# ============================================================
# Validation
# ============================================================

if CHUNK_OVERLAP_TOKENS >= MAX_CHUNK_TOKENS:

    raise ValueError(
        "CHUNK_OVERLAP_TOKENS must be smaller than "
        "MAX_CHUNK_TOKENS."
    )


# ============================================================
# get_database_engine()
# ============================================================

def get_database_engine():
    """
    Create a SQLAlchemy PostgreSQL engine.

    DATABASE_URL comes from:

        .env

    Example:

        postgresql+psycopg2://
        alphalens:password@localhost:5432/alphalens
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
# count_tokens()
# ============================================================

def count_tokens(
    content: str,
) -> int:
    """
    Count how many tokenizer tokens are present in some text.

    Parameters
    ----------
    content:
        Normal text.

    Returns
    -------
    int:
        Number of tokens.


    Example
    -------

    Text:

        "Apple reported higher revenue."

    Characters:

        about 30

    Tokens:

        much fewer than 30


    Why do we care?
    ---------------

    Embedding models and LLM context windows are measured in
    tokens rather than characters.
    """

    return len(
        ENCODING.encode(
            content
        )
    )


# ============================================================
# chunk_text()
# ============================================================

def chunk_text(
    content: str,
) -> list[dict]:
    """
    Split one filing section into token-based chunks.

    Parameters
    ----------
    content:
        Complete text from one filing_sections row.


    Returns
    -------
    list[dict]:

        [
            {
                "chunk_index": 0,
                "content": "...",
                "token_count": 682,
                "char_count": 2910
            },

            {
                "chunk_index": 1,
                ...
            }
        ]


    Small sections
    --------------

    If a section contains fewer than MAX_CHUNK_TOKENS:

        257 tokens

    then:

        section
            ↓
        ONE chunk

    We do NOT split it unnecessarily.


    Large sections
    --------------

    Example:

        2,000 tokens

    with:

        MAX_CHUNK_TOKENS = 700
        overlap = 100

    roughly becomes:

        chunk 0:
            0 -> 699

        chunk 1:
            600 -> 1299

        chunk 2:
            1200 -> 1899

        chunk 3:
            1800 -> end


    Why split using token IDs?
    --------------------------

    It guarantees that no chunk exceeds our configured
    token limit.

    Character-based splitting cannot guarantee that.
    """

    # --------------------------------------------------------
    # Basic validation / cleanup
    # --------------------------------------------------------

    if content is None:

        return []


    content = content.strip()


    if not content:

        return []


    # ========================================================
    # Convert complete section into token IDs
    # ========================================================

    token_ids = ENCODING.encode(
        content
    )


    total_tokens = len(
        token_ids
    )


    # ========================================================
    # Small section -> one chunk
    # ========================================================

    if total_tokens <= MAX_CHUNK_TOKENS:

        return [
            {
                "chunk_index":
                    0,

                "content":
                    content,

                "token_count":
                    total_tokens,

                "char_count":
                    len(content),
            }
        ]


    # ========================================================
    # Calculate how far to move between chunks
    # ========================================================
    #
    # Example:
    #
    # max = 700
    # overlap = 100
    #
    # step:
    #
    #     700 - 100
    #     =
    #     600
    #
    # Therefore:
    #
    # chunk 0 starts at 0
    # chunk 1 starts at 600
    # chunk 2 starts at 1200
    #
    # ========================================================

    step_size = (
        MAX_CHUNK_TOKENS
        - CHUNK_OVERLAP_TOKENS
    )


    chunks = []


    # Position in the complete token array.
    start = 0


    # Each chunk receives a sequential index.
    chunk_index = 0


    # ========================================================
    # Create token windows
    # ========================================================

    while start < total_tokens:

        # ----------------------------------------------------
        # Calculate end position
        # ----------------------------------------------------

        end = min(
            start + MAX_CHUNK_TOKENS,
            total_tokens,
        )


        # ----------------------------------------------------
        # Take this portion of the token IDs
        # ----------------------------------------------------

        chunk_token_ids = token_ids[
            start:end
        ]


        # ----------------------------------------------------
        # Convert tokens back into normal human-readable text
        # ----------------------------------------------------

        chunk_content = (
            ENCODING
            .decode(
                chunk_token_ids
            )
            .strip()
        )


        # ----------------------------------------------------
        # Only keep non-empty chunks
        # ----------------------------------------------------

        if chunk_content:

            chunks.append(
                {
                    "chunk_index":
                        chunk_index,

                    "content":
                        chunk_content,

                    "token_count":
                        len(
                            chunk_token_ids
                        ),

                    "char_count":
                        len(
                            chunk_content
                        ),
                }
            )


            chunk_index += 1


        # ----------------------------------------------------
        # Stop if we reached the end of the section.
        # ----------------------------------------------------

        if end >= total_tokens:

            break


        # ----------------------------------------------------
        # Move forward while preserving overlap.
        # ----------------------------------------------------
        #
        # Example:
        #
        # previous:
        #     0 -> 699
        #
        # next:
        #     600 -> ...
        #
        start += step_size


    return chunks


# ============================================================
# get_sections()
# ============================================================

def get_sections(
    engine,
):
    """
    Retrieve every extracted SEC section that should be
    converted into RAG chunks.

    We join:

        filing_sections
            +
        filings

    because filing_sections contains the actual section text,
    while filings contains metadata such as:

        ticker
        form_type
        filing_date


    Returns
    -------
    list:
        SQLAlchemy mapping rows.
    """

    query = text(
        """
        SELECT
            fs.section_id,
            fs.accession_number,
            fs.section_key,
            fs.section_title,
            fs.content,

            f.ticker,
            f.form_type,
            f.filing_date

        FROM filing_sections fs

        JOIN filings f
            ON fs.accession_number = f.accession_number

        ORDER BY
            f.ticker,
            f.filing_date,
            fs.section_id;
        """
    )


    with engine.connect() as connection:

        result = connection.execute(
            query
        )


        sections = (
            result
            .mappings()
            .all()
        )


    return sections


# ============================================================
# save_section_chunks()
# ============================================================

def save_section_chunks(
    engine,
    chunk_table,
    section,
    chunks: list[dict],
) -> int:
    """
    Save all chunks belonging to one section.

    Very important idempotency behavior
    -----------------------------------

    We DELETE the old chunks for this section first.

    Why not only UPSERT?


    Suppose the old algorithm produced:

        chunk 0
        chunk 1
        chunk 2
        chunk 3
        chunk 4


    Then we improve the chunking algorithm and it now produces:

        chunk 0
        chunk 1
        chunk 2


    If we only UPSERT:

        chunk 3
        chunk 4

    would remain in PostgreSQL even though they are obsolete.


    Therefore:

        DELETE chunks for this section
                    ↓
        INSERT newly generated chunks

    guarantees that PostgreSQL exactly matches the current
    chunking algorithm.
    """

    section_id = section[
        "section_id"
    ]


    # ========================================================
    # Build database rows
    # ========================================================

    records = []


    for chunk in chunks:

        records.append(
            {
                "section_id":
                    section_id,

                "accession_number":
                    section[
                        "accession_number"
                    ],

                "ticker":
                    section[
                        "ticker"
                    ],

                "form_type":
                    section[
                        "form_type"
                    ],

                "filing_date":
                    section[
                        "filing_date"
                    ],

                "section_key":
                    section[
                        "section_key"
                    ],

                "section_title":
                    section[
                        "section_title"
                    ],

                "chunk_index":
                    chunk[
                        "chunk_index"
                    ],

                "content":
                    chunk[
                        "content"
                    ],

                "token_count":
                    chunk[
                        "token_count"
                    ],

                "char_count":
                    chunk[
                        "char_count"
                    ],
            }
        )


    # ========================================================
    # Transaction
    # ========================================================
    #
    # engine.begin() means:
    #
    #     start transaction
    #
    # If everything succeeds:
    #     COMMIT
    #
    # If something fails:
    #     ROLLBACK
    #
    # This is important because we don't want:
    #
    #     old chunks deleted
    #
    # but:
    #
    #     new chunks only partially inserted.
    # ========================================================

    with engine.begin() as connection:


        # ----------------------------------------------------
        # Delete previous version of this section's chunks
        # ----------------------------------------------------

        connection.execute(
            delete(
                chunk_table
            ).where(
                chunk_table.c.section_id
                == section_id
            )
        )


        # ----------------------------------------------------
        # Insert newly generated chunks
        # ----------------------------------------------------

        if records:

            connection.execute(
                insert(
                    chunk_table
                ),
                records,
            )


    return len(
        records
    )


# ============================================================
# process_section()
# ============================================================

def process_section(
    engine,
    chunk_table,
    section,
) -> int:
    """
    Chunk and save one filing section.

    Flow:

        filing_sections.content
                ↓
        tokenization
                ↓
        chunk_text()
                ↓
        chunks
                ↓
        filing_chunks
    """

    content = section[
        "content"
    ]


    # ========================================================
    # Create chunks
    # ========================================================

    chunks = chunk_text(
        content
    )


    # ========================================================
    # Save chunks
    # ========================================================

    stored_count = save_section_chunks(
        engine=engine,
        chunk_table=chunk_table,
        section=section,
        chunks=chunks,
    )


    return stored_count


# ============================================================
# run_chunking()
# ============================================================

def run_chunking():
    """
    Chunk all AlphaLens SEC filing sections.

    Current expected input:

        ~1,870 filing sections


    Output:

        several thousand filing_chunks rows


    The exact number depends on:

        section length
        chunk size
        overlap


    Important:
        Do NOT assume a specific total chunk count.

        We will inspect the resulting distribution afterward.
    """

    engine = get_database_engine()


    # ========================================================
    # Read filing_chunks table definition
    # ========================================================

    metadata = MetaData()


    chunk_table = Table(
        "filing_chunks",
        metadata,
        autoload_with=engine,
    )


    # ========================================================
    # Retrieve source sections
    # ========================================================

    sections = get_sections(
        engine
    )


    print(
        f"\nSections found: "
        f"{len(sections)}"
    )


    total_chunks = 0


    # ========================================================
    # Process each section
    # ========================================================

    for section_number, section in enumerate(
        sections,
        start=1,
    ):

        chunks_created = process_section(
            engine=engine,
            chunk_table=chunk_table,
            section=section,
        )


        total_chunks += (
            chunks_created
        )


        # ----------------------------------------------------
        # Progress output
        # ----------------------------------------------------
        #
        # Don't print every section because ~1,870 lines
        # becomes unnecessarily noisy.
        #
        # Print every 100 sections instead.
        # ----------------------------------------------------

        if (
            section_number % 100 == 0
            or section_number == len(sections)
        ):

            print(
                f"Processed "
                f"{section_number}/{len(sections)} sections "
                f"| "
                f"{total_chunks:,} chunks created"
            )


    print()

    print(
        "========================================"
    )

    print(
        "SEC CHUNKING COMPLETE"
    )

    print(
        "========================================"
    )

    print(
        f"Sections processed: "
        f"{len(sections):,}"
    )

    print(
        f"Chunks created: "
        f"{total_chunks:,}"
    )


# ============================================================
# Script Entry Point
# ============================================================

if __name__ == "__main__":

    run_chunking()