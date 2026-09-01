"""
AlphaLens - Earnings Call Transcript Chunker

Purpose:
    Splits earnings call transcripts into smaller chunks suitable for:

        embeddings
            |
            v
        vector search
            |
            v
        RAG answers over management commentary


Current transcript pipeline:

    earnings_transcripts
        |
        v
    earnings_transcript_turns
        |
        v
    chunker.py
        |
        v
    earnings_transcript_chunks


Why chunk transcripts?
----------------------

One earnings call can contain:

    - operator remarks
    - management prepared remarks
    - financial explanations
    - analyst questions
    - management answers

Embedding the whole transcript as one vector would mix all of those
topics together.

Smaller chunks let future retrieval find the specific part of the call
that matches the user's question.


Important:
    This file creates RAG-ready text chunks.

    It does NOT call OpenAI and does NOT build a FAISS index yet.

    Embedding support can be added as a later transcript embedder, similar
    to pipelines/sec/embedder.py.
"""

import os

import tiktoken

from dotenv import load_dotenv

from sqlalchemy import (
    MetaData,
    Table,
    create_engine,
    delete,
    select,
)

from sqlalchemy.dialects.postgresql import insert


# ============================================================
# Environment
# ============================================================

load_dotenv()


# ============================================================
# Chunk Configuration
# ============================================================

# Keep transcript chunks near the SEC chunk size.
#
# This makes future retrieval behavior easier to reason about because
# SEC chunks and transcript chunks have similar context sizes.
MAX_CHUNK_TOKENS = 700

# cl100k_base is the same tokenizer used by the SEC chunker.
#
# We use it to estimate OpenAI-style token counts without making an API
# call.
TOKEN_ENCODING_NAME = "cl100k_base"

ENCODING = tiktoken.get_encoding(
    TOKEN_ENCODING_NAME
)


def get_database_engine():
    """
    Create a SQLAlchemy PostgreSQL engine.
    """

    # DATABASE_URL is read from .env to avoid hard-coding credentials.
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise ValueError(
            "DATABASE_URL was not found in .env."
        )

    # pool_pre_ping=True checks stale connections before SQLAlchemy reuses
    # them.
    return create_engine(
        database_url,
        pool_pre_ping=True,
    )


def count_tokens(
    content: str,
) -> int:
    """
    Count tokens using the same tokenizer as the SEC chunker.
    """

    # Token counts matter because embedding models and LLM context windows
    # are limited by tokens, not characters.
    return len(
        ENCODING.encode(content)
    )


def get_transcripts(
    engine,
    transcript_table,
):
    """
    Retrieve ingested transcripts.
    """

    # The parent transcript table tells us which calls exist and provides
    # fallback full text if the provider did not return speaker turns.
    query = (
        select(
            transcript_table.c.transcript_id,
            transcript_table.c.ticker,
            transcript_table.c.fiscal_year,
            transcript_table.c.fiscal_quarter,
            transcript_table.c.fiscal_period,
            transcript_table.c.source_provider,
            transcript_table.c.content,
        )
        .order_by(
            transcript_table.c.ticker,
            transcript_table.c.fiscal_year,
            transcript_table.c.fiscal_quarter,
        )
    )

    with engine.connect() as connection:
        return connection.execute(
            query
        ).mappings().all()


def get_turns_for_transcript(
    engine,
    turn_table,
    transcript_id: int,
):
    """
    Retrieve speaker turns for one transcript.
    """

    # Speaker turns are ordered by turn_index so the transcript remains in
    # the original call sequence.
    query = (
        select(
            turn_table.c.speaker_name,
            turn_table.c.speaker_title,
            turn_table.c.content,
        )
        .where(
            turn_table.c.transcript_id == transcript_id
        )
        .order_by(
            turn_table.c.turn_index
        )
    )

    with engine.connect() as connection:
        return connection.execute(
            query
        ).mappings().all()


def format_turn(
    turn,
) -> str:
    """
    Format one speaker turn for chunk content.
    """

    # Include speaker metadata directly inside the chunk text.
    #
    # Later, if a RAG answer cites a chunk, the model can see whether the
    # statement came from the CEO, CFO, operator, or an unknown speaker.
    speaker_name = turn["speaker_name"] or "Unknown Speaker"
    speaker_title = turn["speaker_title"]

    if speaker_title:
        heading = f"{speaker_name} - {speaker_title}"
    else:
        heading = speaker_name

    return (
        f"{heading}\n"
        f"{turn['content']}"
    )


def chunk_turns(
    turns,
) -> list[dict]:
    """
    Group transcript turns into chunks without splitting a turn
    unless a single turn is larger than MAX_CHUNK_TOKENS.
    """

    # We prefer to keep complete speaker turns together.
    #
    # Why?
    #
    # Splitting in the middle of an answer can remove the question/answer
    # context that makes earnings calls valuable.
    chunks = []
    current_parts = []
    current_speakers = []
    current_tokens = 0
    chunk_index = 0

    def flush_current():
        """
        Save the current accumulated group of turns as one chunk.

        This small inner function avoids repeating the same "finish the
        current chunk" logic in multiple branches below.
        """

        nonlocal current_parts
        nonlocal current_speakers
        nonlocal current_tokens
        nonlocal chunk_index

        if not current_parts:
            return

        content = "\n\n".join(
            current_parts
        ).strip()

        # Store unique speaker names as metadata.
        #
        # This allows future filtering/search UI such as:
        #
        #     show chunks containing CFO commentary
        chunks.append(
            {
                "chunk_index": chunk_index,
                "speaker_names": sorted(set(current_speakers)),
                "content": content,
                "token_count": count_tokens(content),
                "char_count": len(content),
            }
        )

        chunk_index += 1
        current_parts = []
        current_speakers = []
        current_tokens = 0

    for turn in turns:
        turn_text = format_turn(
            turn
        )

        turn_tokens = count_tokens(
            turn_text
        )

        # If adding this turn would exceed the chunk limit, close the
        # current chunk before starting a new one.
        if (
            current_parts
            and current_tokens + turn_tokens > MAX_CHUNK_TOKENS
        ):
            flush_current()

        # Normal case:
        #
        # the whole speaker turn fits in one chunk.
        if turn_tokens <= MAX_CHUNK_TOKENS:
            current_parts.append(
                turn_text
            )

            if turn["speaker_name"]:
                current_speakers.append(
                    turn["speaker_name"]
                )

            current_tokens += turn_tokens
            continue

        # Rare case:
        #
        # a single speaker turn is longer than the entire chunk size.
        #
        # We must split it, otherwise one chunk would exceed the embedding
        # model input target.
        token_ids = ENCODING.encode(
            turn_text
        )

        for start in range(0, len(token_ids), MAX_CHUNK_TOKENS):
            piece = ENCODING.decode(
                token_ids[
                    start:start + MAX_CHUNK_TOKENS
                ]
            ).strip()

            if not piece:
                continue

            chunks.append(
                {
                    "chunk_index": chunk_index,
                    "speaker_names": (
                        [turn["speaker_name"]]
                        if turn["speaker_name"]
                        else []
                    ),
                    "content": piece,
                    "token_count": count_tokens(piece),
                    "char_count": len(piece),
                }
            )

            chunk_index += 1

    flush_current()

    return chunks


def chunk_transcript_content(
    content: str,
) -> list[dict]:
    """
    Fallback chunking when provider data has no speaker turns.
    """

    # Some providers may return one full transcript string but no speaker
    # turn array. This fallback still makes the transcript searchable.
    token_ids = ENCODING.encode(
        content.strip()
    )

    chunks = []

    for chunk_index, start in enumerate(
        range(0, len(token_ids), MAX_CHUNK_TOKENS)
    ):
        piece = ENCODING.decode(
            token_ids[
                start:start + MAX_CHUNK_TOKENS
            ]
        ).strip()

        if not piece:
            continue

        chunks.append(
            {
                "chunk_index": chunk_index,
                "speaker_names": [],
                "content": piece,
                "token_count": count_tokens(piece),
                "char_count": len(piece),
            }
        )

    return chunks


def save_chunks(
    engine,
    chunk_table,
    transcript,
    chunks: list[dict],
) -> int:
    """
    Replace chunks for one transcript.
    """

    transcript_id = transcript["transcript_id"]

    records = []

    # Denormalize common metadata into the chunk table.
    #
    # This mirrors filing_chunks and makes retrieval/filtering faster later
    # because we do not need to join every time we filter by ticker/period.
    for chunk in chunks:
        records.append(
            {
                "transcript_id": transcript_id,
                "ticker": transcript["ticker"],
                "fiscal_year": transcript["fiscal_year"],
                "fiscal_quarter": transcript["fiscal_quarter"],
                "fiscal_period": transcript["fiscal_period"],
                "source_provider": transcript["source_provider"],
                "chunk_index": chunk["chunk_index"],
                "speaker_names": chunk["speaker_names"],
                "content": chunk["content"],
                "token_count": chunk["token_count"],
                "char_count": chunk["char_count"],
            }
        )

    # Idempotency:
    #
    # Delete old chunks for this transcript first, then insert the current
    # chunking output.
    #
    # If the chunking algorithm changes, obsolete old chunks will not be
    # left behind.
    with engine.begin() as connection:
        connection.execute(
            delete(
                chunk_table
            ).where(
                chunk_table.c.transcript_id == transcript_id
            )
        )

        if records:
            connection.execute(
                insert(chunk_table),
                records,
            )

    return len(records)


def run_chunking():
    """
    Chunk every ingested earnings call transcript.
    """

    engine = get_database_engine()

    # Reflect the existing PostgreSQL tables instead of redefining their
    # columns in Python.
    metadata = MetaData()

    transcript_table = Table(
        "earnings_transcripts",
        metadata,
        autoload_with=engine,
    )

    turn_table = Table(
        "earnings_transcript_turns",
        metadata,
        autoload_with=engine,
    )

    chunk_table = Table(
        "earnings_transcript_chunks",
        metadata,
        autoload_with=engine,
    )

    transcripts = get_transcripts(
        engine=engine,
        transcript_table=transcript_table,
    )

    print(
        f"\nTranscripts found: {len(transcripts)}"
    )

    total_chunks = 0

    for transcript in transcripts:
        # Prefer speaker-turn chunking because it preserves call structure.
        turns = get_turns_for_transcript(
            engine=engine,
            turn_table=turn_table,
            transcript_id=transcript["transcript_id"],
        )

        if turns:
            chunks = chunk_turns(
                turns
            )
        else:
            # Fallback for providers that return only a full text blob.
            chunks = chunk_transcript_content(
                transcript["content"]
            )

        chunks_stored = save_chunks(
            engine=engine,
            chunk_table=chunk_table,
            transcript=transcript,
            chunks=chunks,
        )

        total_chunks += chunks_stored

        print(
            f"[OK] {transcript['ticker']} "
            f"{transcript['fiscal_period']}: "
            f"{chunks_stored} chunks"
        )

    print("\n========================================")
    print("TRANSCRIPT CHUNKING COMPLETE")
    print("========================================")

    print(
        f"Chunks created: {total_chunks:,}"
    )


if __name__ == "__main__":
    run_chunking()
