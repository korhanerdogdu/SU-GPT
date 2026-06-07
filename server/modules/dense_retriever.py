"""Dense (embedding) retrieval over the Chroma vector store (Section 3).

Wraps the existing ChromaDB similarity search into the same normalized result
shape used by `bm25_retriever` and `hybrid_retriever`, so `retrieval_modes`
can treat every retriever interchangeably.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("modules.dense_retriever")


def _to_similarity(distance: float) -> float:
    """Convert Chroma's L2 distance (lower = closer) into a 0-1 similarity score."""
    return 1.0 / (1.0 + max(distance, 0.0))


def search(
    vectorstore: Any,
    query: str,
    top_k: int,
    metadata_filter: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return up to `top_k` dense matches as normalized result dicts.

    Normalized shape (shared with bm25/hybrid): chunk_id, text, metadata,
    score, retriever="dense". `score` is a similarity in (0, 1] derived from
    Chroma's raw L2 distance (also kept as `distance` for transparency).
    """
    try:
        scored = vectorstore.similarity_search_with_score(query, k=top_k, filter=metadata_filter)
    except Exception:
        logger.exception("similarity_search_with_score failed; falling back to similarity_search")
        try:
            documents = vectorstore.similarity_search(query, k=top_k, filter=metadata_filter)
        except Exception:
            logger.exception("similarity_search fallback also failed for dense retrieval")
            return []
        scored = [(doc, None) for doc in documents]

    results: list[dict[str, Any]] = []
    for document, distance in scored:
        metadata = document.metadata or {}
        if distance is None:
            score: float | None = None
        else:
            distance = float(distance)
            score = _to_similarity(distance)
        results.append(
            {
                "chunk_id": metadata.get("chunk_id"),
                "text": document.page_content,
                "metadata": metadata,
                "score": score,
                "distance": distance if distance is not None else None,
                "retriever": "dense",
            }
        )
    return results
