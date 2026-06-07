"""Hybrid retrieval: merges BM25 (lexical) and dense (embedding) candidates.

Uses Reciprocal Rank Fusion (RRF) -- Cormack, Clarke & Buettcher, 2009,
"Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning
Methods" -- to merge ranked lists from retrievers whose raw scores live on
incomparable scales (BM25's unbounded lexical scores vs. Chroma's L2
distances). RRF only looks at each result's *rank* within its own list, so no
score normalization step is needed and the merge stays robust and citable.
"""

from __future__ import annotations

from typing import Any

from modules.config import BM25_TOP_K, DENSE_TOP_K

RRF_RANK_CONSTANT = 60


def _dedupe_key(result: dict[str, Any]) -> Any:
    chunk_id = result.get("chunk_id")
    if chunk_id:
        return ("chunk_id", chunk_id)
    metadata = result.get("metadata") or {}
    return (
        "fallback",
        metadata.get("source") or metadata.get("file_name"),
        metadata.get("page"),
        metadata.get("slide"),
        metadata.get("section"),
        (result.get("text") or "")[:80],
    )


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict[str, Any]]],
    top_k: int,
    rank_constant: int = RRF_RANK_CONSTANT,
) -> list[dict[str, Any]]:
    """Merge several ranked result lists into one list ordered by RRF score.

    RRF score for a document = sum over lists containing it of 1 / (k + rank),
    where `rank` is its 1-based position in that list. Each fused result keeps
    its original fields (text, metadata, ...) plus `rrf_score`,
    `retrievers_matched`, and `component_scores` for transparency/evaluation.
    """
    fused: dict[Any, dict[str, Any]] = {}

    for ranked_list in ranked_lists:
        for rank, result in enumerate(ranked_list, start=1):
            key = _dedupe_key(result)
            entry = fused.get(key)
            contribution = 1.0 / (rank_constant + rank)
            if entry is None:
                entry = {
                    **result,
                    "retriever": "hybrid",
                    "rrf_score": 0.0,
                    "retrievers_matched": [],
                    "component_scores": {},
                }
                fused[key] = entry
            entry["rrf_score"] += contribution
            retriever_name = result.get("retriever", "unknown")
            entry["retrievers_matched"].append(retriever_name)
            entry["component_scores"][retriever_name] = {
                "rank": rank,
                "score": result.get("score"),
            }

    merged = sorted(fused.values(), key=lambda item: item["rrf_score"], reverse=True)
    return merged[:top_k]


def hybrid_search(
    vectorstore: Any,
    query: str,
    *,
    top_k: int,
    bm25_k: int | None = None,
    dense_k: int | None = None,
    metadata_filter: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Retrieve candidates from BM25 and dense search, then fuse them with RRF."""
    from modules import bm25_retriever, dense_retriever

    bm25_results = bm25_retriever.search(
        vectorstore, query, top_k=bm25_k or BM25_TOP_K, metadata_filter=metadata_filter
    )
    dense_results = dense_retriever.search(
        vectorstore, query, top_k=dense_k or DENSE_TOP_K, metadata_filter=metadata_filter
    )
    return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)
