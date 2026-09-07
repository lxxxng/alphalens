"""
Compatibility wrapper for AlphaLens semantic retrieval.

New code should import from app.rag.retrievers.router or one of the concrete
retrievers under app.rag.retrievers.
"""

from app.rag.retrievers.base import (
    DEFAULT_TOP_K,
    FAISS_DIRECTORY,
    SEC_INDEX_PATH,
    SEC_METADATA_PATH,
    TRANSCRIPT_INDEX_PATH,
    TRANSCRIPT_METADATA_PATH,
    apply_metadata_filters,
    build_query_vector,
    embed_query,
    get_database_engine,
    get_openai_client,
    load_faiss_index,
    load_index_metadata,
    normalize_query_vector,
    search_faiss,
    search_with_filters,
    validate_index_dimension,
)

from app.rag.retrievers.router import (
    normalize_corpus,
    retrieve_evidence,
    resolve_source_types,
    semantic_search,
)


INDEX_PATH = SEC_INDEX_PATH
METADATA_PATH = SEC_METADATA_PATH


def print_search_results(
    query: str,
    results,
):
    """
    Print semantic-search results in a readable format.
    """

    print()
    print("========================================")
    print("ALPHALENS SEMANTIC SEARCH")
    print("========================================")
    print(f"\nQuery:\n{query}")
    print(f"\nResults returned: {len(results)}")

    for rank, result in enumerate(
        results,
        start=1,
    ):
        print()
        print("----------------------------------------")
        print(f"RESULT {rank}")
        print("----------------------------------------")
        print(f"Similarity: {result['score']:.4f}")
        print(f"Chunk ID: {result['chunk_id']}")
        print(f"Ticker: {result['ticker']}")

        if result.get("source_type") == "transcript":
            print("Source: Earnings transcript")
            print(f"Fiscal period: {result['fiscal_period']}")
            print(f"Call date: {result['call_date']}")
            print(
                "Speakers: "
                f"{', '.join(result.get('speaker_names') or [])}"
            )
        else:
            print("Source: SEC filing")
            print(f"Form: {result['form_type']}")
            print(f"Filing date: {result['filing_date']}")
            print(f"Section: {result['section_title']}")
            print(f"Section key: {result['section_key']}")

        print(f"Chunk index: {result['chunk_index']}")
        print(f"Tokens: {result['token_count']}")
        print("\nCONTENT:")
        print(result["content"])


if __name__ == "__main__":
    test_query = "What cybersecurity risks does NVIDIA face?"

    search_results = semantic_search(
        query=test_query,
        top_k=5,
        ticker="NVDA",
        corpus="filings",
    )

    print_search_results(
        query=test_query,
        results=search_results,
    )
