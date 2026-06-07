"""Unified retrieval-mode contract (Section 3 / Step 2).

Exposes one function, `run_retrieval_mode`, that runs the same question
through any of six retrieval configurations and returns a normalized report.
This lets `/ask`, the CLI comparison script, and (later) the Section 5
evaluation runner all compare modes on equal footing without duplicating
retrieval plumbing.

Supported modes:
    llm_only      -- no retrieval; the LLM answers from internal knowledge only
    bm25          -- lexical retrieval only (BM25 over tokenized chunk text)
    dense         -- embedding retrieval only (Chroma similarity search)
    dense_rerank  -- dense candidates re-scored by the CrossEncoder reranker
    hybrid        -- BM25 + dense candidates merged with Reciprocal Rank Fusion
    hybrid_rerank -- hybrid candidates, then CrossEncoder-reranked (production default)
"""

from __future__ import annotations

import logging
import time
from typing import Any

from langchain_core.documents import Document

from modules import bm25_retriever, dense_retriever, hybrid_retriever
from modules.config import (
    DEFAULT_RETRIEVAL_MODE,
    HYBRID_TOP_K,
    RERANK_TOP_K,
    RETRIEVAL_CANDIDATE_K,
)
from modules.reranker import rerank_documents

logger = logging.getLogger("modules.retrieval_modes")

SUPPORTED_MODES = ("llm_only", "bm25", "dense", "dense_rerank", "hybrid", "hybrid_rerank")

DEFAULT_MODE = DEFAULT_RETRIEVAL_MODE if DEFAULT_RETRIEVAL_MODE in SUPPORTED_MODES else "hybrid_rerank"

MODE_DESCRIPTIONS: dict[str, str] = {
    "llm_only": "No retrieval -- the LLM answers from its own internal knowledge only.",
    "bm25": "Lexical retrieval only (BM25 ranking over tokenized chunk text).",
    "dense": "Embedding retrieval only (Chroma similarity search, no reranking).",
    "dense_rerank": "Dense retrieval candidates re-scored by a CrossEncoder reranker.",
    "hybrid": "BM25 + dense candidates merged with Reciprocal Rank Fusion, no reranking.",
    "hybrid_rerank": "BM25 + dense candidates merged with RRF, then CrossEncoder-reranked (production default).",
}

LLM_ONLY_WARNING = (
    "LLM-only mode does not use retrieved course documents and may hallucinate course-specific facts."
)

_MODE_ALIASES = {
    "llm": "llm_only",
    "baseline": "llm_only",
    "rerank": "hybrid_rerank",
}


def normalize_mode(value: str | None) -> str:
    """Map a possibly-loose mode label to one of SUPPORTED_MODES, defaulting safely."""
    if not value:
        return DEFAULT_MODE
    candidate = value.strip().lower().replace("-", "_").replace(" ", "_")
    if candidate in SUPPORTED_MODES:
        return candidate
    if candidate in _MODE_ALIASES:
        return _MODE_ALIASES[candidate]
    logger.warning("Unknown retrieval mode %r; falling back to default %r", value, DEFAULT_MODE)
    return DEFAULT_MODE


def _result_to_document(result: dict[str, Any]) -> Document:
    """Convert a normalized retrieval result into the `Document` shape the rest
    of the app (context formatting, reranker, LLM chain) already expects."""
    metadata = dict(result.get("metadata") or {})
    return Document(page_content=result.get("text") or "", metadata=metadata)


def _attach_rerank_scores(query: str, results: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    """Re-score normalized results with the CrossEncoder reranker.

    `rerank_documents` returns the same `Document` instances it was given
    (reordered, sliced to top_k, with `.metadata` replaced to include
    `rerank_score`/`combined_rerank_score`) -- so `id(document)` reliably maps
    each reranked Document back to the normalized result it came from.
    """
    if not results:
        return []
    documents = [_result_to_document(result) for result in results]
    original_by_id = {id(document): result for document, result in zip(documents, results)}
    reranked_documents = rerank_documents(query, documents, top_k=top_k)

    reranked: list[dict[str, Any]] = []
    for document in reranked_documents:
        original = original_by_id.get(id(document), {})
        metadata = dict(document.metadata or {})
        rerank_score = metadata.pop("rerank_score", None)
        combined_rerank_score = metadata.pop("combined_rerank_score", None)
        reranked.append(
            {
                **original,
                "text": document.page_content,
                "metadata": metadata,
                "rerank_score": rerank_score,
                "combined_rerank_score": combined_rerank_score,
            }
        )
    return reranked


def run_retrieval_mode(
    mode: str,
    query: str,
    vectorstore: Any,
    *,
    top_k: int | None = None,
    metadata_filter: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run `query` through one retrieval mode and return a normalized report.

    Returns a dict shaped as:
        {
          "mode": <normalized mode name>,
          "description": <human-readable mode description>,
          "results": [normalized result dicts: chunk_id/text/metadata/score/retriever/...],
          "context_documents": [Document, ...]   (ready for source-labeled prompting),
          "timings_ms": {"retrieval_ms": ..., "rerank_ms": ..., "total_ms": ...},
          "warnings": [str, ...],
        }

    Every mode shares this contract, so the same question can be run through
    any of them and produce directly comparable reports -- this is what
    `compare_retrieval_modes.py` and the Section 5 evaluation runner rely on.
    """
    normalized_mode = normalize_mode(mode)
    final_k = top_k if top_k and top_k > 0 else RERANK_TOP_K

    warnings: list[str] = []
    results: list[dict[str, Any]] = []
    retrieval_ms = 0.0
    rerank_ms = 0.0
    total_started = time.perf_counter()

    if normalized_mode == "llm_only":
        warnings.append(LLM_ONLY_WARNING)

    elif normalized_mode == "bm25":
        retrieval_started = time.perf_counter()
        results = bm25_retriever.search(vectorstore, query, top_k=final_k, metadata_filter=metadata_filter)
        retrieval_ms = (time.perf_counter() - retrieval_started) * 1000

    elif normalized_mode == "dense":
        retrieval_started = time.perf_counter()
        results = dense_retriever.search(vectorstore, query, top_k=final_k, metadata_filter=metadata_filter)
        retrieval_ms = (time.perf_counter() - retrieval_started) * 1000

    elif normalized_mode == "dense_rerank":
        retrieval_started = time.perf_counter()
        candidates = dense_retriever.search(
            vectorstore, query, top_k=max(RETRIEVAL_CANDIDATE_K, final_k), metadata_filter=metadata_filter
        )
        retrieval_ms = (time.perf_counter() - retrieval_started) * 1000
        rerank_started = time.perf_counter()
        results = _attach_rerank_scores(query, candidates, top_k=final_k)
        rerank_ms = (time.perf_counter() - rerank_started) * 1000

    elif normalized_mode == "hybrid":
        retrieval_started = time.perf_counter()
        fused = hybrid_retriever.hybrid_search(
            vectorstore,
            query,
            top_k=max(HYBRID_TOP_K, final_k),
            metadata_filter=metadata_filter,
        )
        results = fused[:final_k]
        retrieval_ms = (time.perf_counter() - retrieval_started) * 1000

    elif normalized_mode == "hybrid_rerank":
        retrieval_started = time.perf_counter()
        candidates = hybrid_retriever.hybrid_search(
            vectorstore,
            query,
            top_k=max(HYBRID_TOP_K, RETRIEVAL_CANDIDATE_K, final_k),
            metadata_filter=metadata_filter,
        )
        retrieval_ms = (time.perf_counter() - retrieval_started) * 1000
        rerank_started = time.perf_counter()
        results = _attach_rerank_scores(query, candidates, top_k=final_k)
        rerank_ms = (time.perf_counter() - rerank_started) * 1000

    else:  # pragma: no cover -- normalize_mode() guarantees a supported value
        raise ValueError(f"Unsupported retrieval mode: {mode!r}")

    if normalized_mode != "llm_only" and not results:
        warnings.append("No chunks were retrieved for this query; an answer would be ungrounded.")

    context_documents = [_result_to_document(result) for result in results]
    total_ms = (time.perf_counter() - total_started) * 1000

    return {
        "mode": normalized_mode,
        "description": MODE_DESCRIPTIONS[normalized_mode],
        "results": results,
        "context_documents": context_documents,
        "timings_ms": {
            "retrieval_ms": round(retrieval_ms, 2),
            "rerank_ms": round(rerank_ms, 2),
            "total_ms": round(total_ms, 2),
        },
        "warnings": warnings,
    }
