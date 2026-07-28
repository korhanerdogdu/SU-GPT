from __future__ import annotations

"""
Selectable retrieval modes (CLAUDE.md Section 3).

The advising pipeline already runs one fixed path: structured `get` + dense Chroma search +
BM25 over the same scoped subset, fused in `catalog_retriever`, then CrossEncoder rerank.
That path is excellent for the product but useless for the CS 455 evaluation, which has to
report what each component *contributes*. You cannot measure a component you cannot switch off.

So this module does not reimplement retrieval. It exposes the existing retrievers as five
comparable configurations:

    llm_only       no retrieval at all (baseline; will hallucinate course facts, that's the point)
    bm25           lexical only (the shipped BM25 - the benchmark's mandatory baseline)
    dense          vector only
    hybrid         lexical + vector fused, no reranking
    hybrid_rerank  the previous production path
    hybrid_meta    BM25F + E5 dense (RRF) + metadata/record-type boosts - the benchmarked
                   winner and the current default; see docs/retrieval_benchmark_report.md

Scoping invariant: every RAG mode receives the SAME `metadata_filter` that the profile-aware
retrieval policy produced. Ablating the retriever must never widen the corpus, or the numbers
would be measuring a different (and unsafe) system than the one that ships.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from langchain_core.documents import Document

from modules.config import BM25_TOP_K, DENSE_TOP_K, ENABLE_RERANKING

RETRIEVAL_MODES: tuple[str, ...] = (
    "llm_only", "bm25", "dense", "hybrid", "hybrid_rerank", "hybrid_meta",
)
RAG_MODES: tuple[str, ...] = tuple(m for m in RETRIEVAL_MODES if m != "llm_only")
DEFAULT_MODE = "hybrid_rerank"

LLM_ONLY_WARNING = (
    "LLM-only mode does not use retrieved course documents and may hallucinate "
    "course-specific facts."
)


@dataclass
class RetrievalOutcome:
    """Normalized result of one retrieval, whatever the mode."""

    mode: str
    documents: list[Document] = field(default_factory=list)
    candidate_count: int = 0
    retrieval_ms: float = 0.0
    rerank_ms: float = 0.0
    reranked: bool = False

    @property
    def used_retrieval(self) -> bool:
        return self.mode != "llm_only"


def normalize_mode(value: str | None) -> str:
    """Map user input to a supported mode. Unknown/empty values fall back to the default."""
    candidate = (value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "llm": "llm_only",
        "llmonly": "llm_only",
        "baseline": "llm_only",
        "lexical": "bm25",
        "vector": "dense",
        "semantic": "dense",
        "hybrid_reranked": "hybrid_rerank",
        "hybrid_cross_encoder": "hybrid_rerank",
    }
    candidate = aliases.get(candidate, candidate)
    return candidate if candidate in RETRIEVAL_MODES else DEFAULT_MODE


def _tag(docs: Sequence[Document], retriever: str) -> list[Document]:
    """Give every result the same shape: metadata carries `_retriever` and `_score`."""
    tagged: list[Document] = []
    for doc in docs:
        meta = dict(doc.metadata or {})
        meta.setdefault("_retriever", retriever)
        if "_score" not in meta:
            meta["_score"] = float(meta.get("_bm25") or 0.0)
        tagged.append(Document(page_content=doc.page_content, metadata=meta))
    return tagged


def _dense_search(vectorstore: Any, query: str, k: int, where: dict[str, Any] | None) -> list[Document]:
    """Vector-only search. Uses scored search when the store supports it so `_score` is real."""
    kwargs: dict[str, Any] = {"k": max(k, 1)}
    if where:
        kwargs["filter"] = where
    try:
        scored = vectorstore.similarity_search_with_score(query, **kwargs)
        docs: list[Document] = []
        for doc, score in scored:
            meta = dict(doc.metadata or {})
            meta["_retriever"] = "dense"
            # Chroma returns a distance (lower is better); expose similarity so higher is better.
            meta["_distance"] = float(score)
            meta["_score"] = 1.0 / (1.0 + float(score))
            docs.append(Document(page_content=doc.page_content, metadata=meta))
        return docs
    except Exception:
        # Older/stub vector stores may not implement the scored variant.
        return _tag(vectorstore.similarity_search(query, **kwargs), "dense")


def _bm25_search(vectorstore: Any, query: str, k: int, where: dict[str, Any] | None) -> list[Document]:
    from modules.bm25_retriever import annotate_bm25

    docs = annotate_bm25(vectorstore, query, where, top=max(k, 1), require_narrowing=False)
    return _tag(docs, "bm25")


def retrieve(
    mode: str,
    *,
    vectorstore: Any,
    query: str,
    top_k: int,
    candidate_k: int,
    metadata_filter: dict[str, Any] | None = None,
    hybrid_search: Callable[[], list[Document]] | None = None,
) -> RetrievalOutcome:
    """Run one retrieval configuration.

    `hybrid_search` is the caller's existing fused path (catalog_retriever, already scoped and
    possibly multi-search). It is injected rather than rebuilt here so hybrid modes keep the
    exact production behaviour, including route-based multi-search.
    """
    mode = normalize_mode(mode)
    top_k = max(int(top_k or 1), 1)
    candidate_k = max(int(candidate_k or top_k), top_k)

    if mode == "llm_only":
        return RetrievalOutcome(mode=mode)

    started = time.perf_counter()
    if mode == "hybrid_meta":
        # Benchmarked winner (docs/retrieval_benchmark_report.md): field-weighted BM25F over
        # the structured curriculum corpus, fused by RRF with multilingual-E5-small, then the
        # soft metadata/record-type boost. Held-out test Recall@10 0.9419 vs 0.5992 for the
        # BM25 baseline. Runs through modules.lab_retriever, which is the SAME code the
        # benchmark scored - there is no second implementation to drift.
        #
        # If its index cannot be built the request must still be answered, so we fall back to
        # the Chroma hybrid path - but loudly. A silent downgrade would leave the app serving
        # a weaker retriever than the one it reports using.
        try:
            from modules.lab_retriever import lab_search

            candidates = lab_search(
                query, top_k=max(candidate_k, top_k), metadata_filter=metadata_filter
            )
        except Exception as exc:
            from logger import logger

            logger.error(
                "retrieval_modes: hybrid_meta unavailable (%s: %s); falling back to the Chroma "
                "hybrid path for this request", type(exc).__name__, exc,
            )
            candidates = list(hybrid_search()) if hybrid_search else _dense_search(
                vectorstore, query, candidate_k, metadata_filter
            )
            candidates = _tag(candidates, "hybrid_fallback")
    elif mode == "dense":
        candidates = _dense_search(vectorstore, query, max(candidate_k, DENSE_TOP_K), metadata_filter)
    elif mode == "bm25":
        candidates = _bm25_search(vectorstore, query, max(candidate_k, BM25_TOP_K), metadata_filter)
    else:  # hybrid, hybrid_rerank
        candidates = list(hybrid_search()) if hybrid_search else _dense_search(
            vectorstore, query, candidate_k, metadata_filter
        )
        candidates = _tag(candidates, "hybrid")
    retrieval_ms = (time.perf_counter() - started) * 1000.0

    outcome = RetrievalOutcome(
        mode=mode,
        candidate_count=len(candidates),
        retrieval_ms=round(retrieval_ms, 2),
    )

    if mode == "hybrid_rerank" and ENABLE_RERANKING and candidates:
        from modules.reranker import rerank_documents

        started = time.perf_counter()
        ranked = rerank_documents(query, candidates, top_k=top_k)
        outcome.rerank_ms = round((time.perf_counter() - started) * 1000.0, 2)
        outcome.reranked = True
        outcome.documents = ranked
    else:
        # bm25 / dense / hybrid keep their own ordering; just cut to the context budget.
        outcome.documents = candidates[:top_k]

    return outcome
