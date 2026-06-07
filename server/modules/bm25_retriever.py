"""BM25 lexical retrieval over the Chroma corpus (Section 3).

Builds an in-memory BM25 index lazily from the same persistent Chroma
collection used for dense retrieval, so the bm25/dense/hybrid modes search a
directly comparable corpus. The index is built once per process -- the first
BM25 query pays a one-time fetch+tokenize+index cost -- and cached in memory
for every later query (see `_get_corpus`).
"""

from __future__ import annotations

import logging
import re
import threading
import time
from typing import Any

from modules.config import BM25_MAX_CORPUS_SIZE

logger = logging.getLogger("modules.bm25_retriever")

_TOKEN_RE = re.compile(r"[a-z0-9çğıöşü]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokenizer that keeps Turkish letters intact."""
    return _TOKEN_RE.findall((text or "").lower())


class _Bm25Corpus:
    __slots__ = ("ids", "documents", "metadatas", "index", "size")

    def __init__(self, ids: list[str], documents: list[str], metadatas: list[dict], index: Any) -> None:
        self.ids = ids
        self.documents = documents
        self.metadatas = metadatas
        self.index = index
        self.size = len(ids)


_lock = threading.Lock()
_cache: dict[str, _Bm25Corpus] = {}


def _collection_key(vectorstore: Any) -> str:
    collection = getattr(vectorstore, "_collection", None)
    if collection is None:
        return "no-collection"
    return f"{getattr(collection, 'name', 'default')}:{collection.count()}"


# Chroma's SQLite backend rejects a single `collection.get(limit=...)` call
# once the generated query needs too many bound SQL variables -- empirically
# this corpus fails somewhere between 30k and 50k rows. Paginating with
# `offset` keeps every page well under that ceiling regardless of corpus size.
_FETCH_PAGE_SIZE = 20000


def _fetch_all_chunks(collection: Any, limit: int) -> tuple[list[str], list[str], list[dict]]:
    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict] = []
    offset = 0
    while offset < limit:
        page_size = min(_FETCH_PAGE_SIZE, limit - offset)
        page = collection.get(limit=page_size, offset=offset, include=["documents", "metadatas"])
        page_ids = page["ids"]
        if not page_ids:
            break
        ids.extend(page_ids)
        documents.extend(page["documents"])
        metadatas.extend(page["metadatas"])
        offset += len(page_ids)
    return ids, documents, metadatas


def _build_corpus(vectorstore: Any) -> _Bm25Corpus:
    from rank_bm25 import BM25Okapi

    collection = getattr(vectorstore, "_collection", None)
    if collection is None:
        raise RuntimeError("Vectorstore has no underlying Chroma collection for BM25 indexing")

    total = collection.count()
    limit = total if BM25_MAX_CORPUS_SIZE <= 0 else min(BM25_MAX_CORPUS_SIZE, total)
    if limit < total:
        logger.warning(
            "BM25_MAX_CORPUS_SIZE=%s caps the BM25 index at %s/%s chunks; "
            "bm25/hybrid results only cover this subset of the corpus.",
            BM25_MAX_CORPUS_SIZE,
            limit,
            total,
        )

    started = time.perf_counter()
    logger.info(
        "Building BM25 index over %s/%s chunks in pages of %s (one-time cost, cached in memory)...",
        limit,
        total,
        _FETCH_PAGE_SIZE,
    )
    ids, documents, metadatas = _fetch_all_chunks(collection, limit)
    tokenized_corpus = [tokenize(doc) for doc in documents]
    index = BM25Okapi(tokenized_corpus)
    logger.info(
        "BM25 index ready: %s chunks indexed in %.1fs",
        len(ids),
        time.perf_counter() - started,
    )
    return _Bm25Corpus(ids=ids, documents=documents, metadatas=metadatas, index=index)


def _get_corpus(vectorstore: Any) -> _Bm25Corpus:
    key = _collection_key(vectorstore)
    with _lock:
        corpus = _cache.get(key)
        if corpus is None:
            corpus = _build_corpus(vectorstore)
            _cache[key] = corpus
        return corpus


def _matches_filter(metadata: dict, metadata_filter: dict[str, Any] | None) -> bool:
    if not metadata_filter:
        return True
    document_type = metadata_filter.get("documentType")
    if document_type is None:
        return True
    if isinstance(document_type, dict) and "$in" in document_type:
        return metadata.get("documentType") in document_type["$in"]
    return metadata.get("documentType") == document_type


def search(
    vectorstore: Any,
    query: str,
    top_k: int,
    metadata_filter: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return up to `top_k` BM25 matches as normalized result dicts.

    Normalized shape (shared with dense/hybrid): chunk_id, text, metadata,
    score, retriever="bm25". Filtering happens after scoring -- BM25 scoring
    over the full corpus takes single-digit milliseconds, so this stays fast
    while keeping bm25 mode's metadata filtering consistent with dense mode.
    """
    query_tokens = tokenize(query)
    if not query_tokens:
        return []

    corpus = _get_corpus(vectorstore)
    if corpus.size == 0:
        return []

    scores = corpus.index.get_scores(query_tokens)
    ranked_indices = sorted(range(corpus.size), key=lambda i: scores[i], reverse=True)

    results: list[dict[str, Any]] = []
    for idx in ranked_indices:
        score = float(scores[idx])
        if score <= 0:
            break
        metadata = corpus.metadatas[idx] or {}
        if not _matches_filter(metadata, metadata_filter):
            continue
        results.append(
            {
                "chunk_id": metadata.get("chunk_id") or corpus.ids[idx],
                "text": corpus.documents[idx],
                "metadata": metadata,
                "score": score,
                "retriever": "bm25",
            }
        )
        if len(results) >= top_k:
            break
    return results
