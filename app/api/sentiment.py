"""Read-only APIs for versioned transcript and filing sentiment."""

from fastapi import APIRouter, HTTPException, Query

from app.services.sentiment import (
    get_filing_sentiment,
    get_transcript_sentiment,
    get_transcript_sentiment_timeline,
)


router = APIRouter(
    prefix="/api/sentiment",
    tags=["Sentiment"],
)


@router.get("/transcripts")
def transcript_sentiment_timeline(
    ticker: str = Query(min_length=1, max_length=20),
):
    """Return fiscal-period sentiment history for one company."""

    return get_transcript_sentiment_timeline(ticker)


@router.get("/transcripts/{transcript_id}")
def transcript_sentiment(transcript_id: int):
    """Return management and analyst sentiment for one earnings call."""

    result = get_transcript_sentiment(transcript_id)

    if result is None:
        raise HTTPException(status_code=404, detail="Transcript not found.")

    return result


@router.get("/filings/{accession_number}")
def filing_sentiment(accession_number: str):
    """Return section-level sentiment for one SEC filing."""

    result = get_filing_sentiment(accession_number)

    if result is None:
        raise HTTPException(
            status_code=404,
            detail="Filing sentiment scope not found.",
        )

    return result
