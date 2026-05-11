"""
retriever.py — Retrieval strategies on top of FAISS.
Uses MMR (Maximal Marginal Relevance) by default to balance
relevance with diversity and avoid repetitive context.
"""

import logging
from typing import List, Tuple

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

logger = logging.getLogger(__name__)

# Number of chunks to retrieve
TOP_K = 5
# MMR lambda: 1.0 = pure similarity, 0.0 = pure diversity
MMR_LAMBDA = 0.7
# Fetch more candidates than top_k so MMR has room to diversify
FETCH_K_MULTIPLIER = 3


def get_retriever(vector_store: FAISS, top_k: int = TOP_K):
    """
    Return an MMR-based retriever. MMR re-ranks candidates to maximise
    coverage of the query while minimising redundancy between chunks.
    """
    return vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": top_k,
            "fetch_k": top_k * FETCH_K_MULTIPLIER,
            "lambda_mult": MMR_LAMBDA,
        },
    )


def retrieve_with_scores(
    vector_store: FAISS,
    query: str,
    top_k: int = TOP_K,
) -> List[Tuple[Document, float]]:
    """
    Similarity search with relevance scores (0–1, higher = more relevant).
    Useful for displaying source confidence in the UI.
    """
    results = vector_store.similarity_search_with_relevance_scores(query, k=top_k)
    logger.debug(f"Retrieved {len(results)} chunks for query: '{query[:60]}…'")
    return results