from __future__ import annotations

"""
Production adapter for the benchmarked retrieval_lab pipeline.

This module is deliberately thin. It does not reimplement ranking - it builds the *same*
`retrieval_lab` index the benchmark scored and translates between the app's types (Chroma
`where` filters in, LangChain `Document`s out) and the lab's (plain predicates and chunk
ids). Keeping the ranking in one place is what stops the shipped system and the measured
system from drifting apart, which is the failure mode that makes a benchmark worthless.

The index is built lazily from the on-disk corpus (data/degree_requirements, data/minors) and
cached process-wide, because it is the same corpus ingest_degree_requirements.py pushes into
Chroma. Build cost is a few seconds and happens once per process.

FALLBACK POLICY
If the lab index cannot be built (missing corpus, import error), `lab_search` raises and the
caller falls back to the existing Chroma path. The failure is logged at ERROR - a silent
downgrade to a weaker retriever is exactly the kind of thing that goes unnoticed for months,
so it is never silent.
"""

import json
import os
import threading
from typing import Any

from langchain_core.documents import Document

from logger import logger

_LOCK = threading.Lock()
_STATE: dict[str, Any] = {}

# Winning configuration from docs/retrieval_benchmark_report.md, held-out test split:
# BM25F + multilingual-E5-small fused with RRF (k=60, equal weights) over a depth-200
# candidate pool, then the soft metadata/record-type boost. These constants are the ones the
# benchmark scored - changing them here silently invalidates the reported numbers.
RRF_K = 60
CANDIDATE_DEPTH = 200
DENSE_MODEL_KEY = "e5_small"
ENABLE_DENSE = os.getenv("ADVISU_LAB_DENSE", "true").strip().lower() == "true"

# Reranking is measured but OFF by default. `fuse_bge_first` passes every quality criterion on
# the held-out split (Hit@1 +0.0441, CI [+0.0180,+0.0721], p=0.0016, Hit@10 unchanged) but costs
# ~1.5 s P95 against a 1 s interactive budget, so docs/reranking_winner_selection_preregistration.md
# requires it to ship behind a flag rather than as the default. Depth stays at 10: deeper pools
# were measured to drag worse candidates into the top 10 and cost Hit@10.
RERANK_MODE = os.getenv("ADVISU_RERANK", "off").strip().lower()
RERANK_DEPTH = int(os.getenv("ADVISU_RERANK_DEPTH", "10"))
RERANKERS = ("off", "fuse_bge_first", "bge_v2m3_structq", "metadata_rule_rerank")


def _build() -> dict[str, Any]:
    """Build (once) the corpus, the BM25F index, the dense index and the metadata policy.

    The dense half is optional. If the E5 model or its cached embedding matrix is not
    available the adapter degrades to BM25F + metadata boosts, which is the second-best
    configuration measured (test Recall@10 0.9178 vs the hybrid's 0.9419) rather than a
    cliff. The degradation is logged at WARNING so it is visible in the logs.
    """
    import sys
    from pathlib import Path

    server_root = Path(__file__).resolve().parents[1]
    if str(server_root) not in sys.path:
        sys.path.insert(0, str(server_root))

    from retrieval_lab.corpus import load_corpus
    from retrieval_lab.fusion import build_program_lexicon
    from retrieval_lab.sparse import BM25FIndex
    from retrieval_lab.text import tokenize_v2

    corpus = load_corpus()
    if not len(corpus):
        raise RuntimeError("retrieval_lab corpus is empty - is data/degree_requirements present?")
    index = BM25FIndex(corpus, tokenizer=tokenize_v2)
    logger.info(
        "lab_retriever: built BM25F index over %d chunks (corpus %s)",
        len(corpus), corpus.fingerprint,
    )

    dense = None
    if ENABLE_DENSE:
        try:
            from retrieval_lab.dense import DENSE_CONFIGS, DenseIndex

            dense = DenseIndex(DENSE_CONFIGS[DENSE_MODEL_KEY], corpus)
            logger.info("lab_retriever: dense index ready (%s)", DENSE_MODEL_KEY)
        except Exception as exc:
            logger.warning(
                "lab_retriever: dense half unavailable (%s: %s); serving BM25F+metadata only "
                "(measured Recall@10 0.918 vs 0.942 for the full hybrid)",
                type(exc).__name__, exc,
            )
            dense = None

    lexicon = build_program_lexicon(corpus)
    reranker = None
    if RERANK_MODE != "off":
        if RERANK_MODE not in RERANKERS:
            logger.error("lab_retriever: unknown ADVISU_RERANK=%r (known: %s); reranking stays off",
                         RERANK_MODE, ", ".join(RERANKERS))
        else:
            try:
                reranker = _build_reranker(corpus, lexicon)
                logger.info("lab_retriever: reranking ENABLED (%s, depth %d, adds ~1.5s P95)",
                            RERANK_MODE, RERANK_DEPTH)
            except Exception as exc:
                logger.error(
                    "lab_retriever: reranker %s failed to load (%s: %s); serving the "
                    "first-stage order unchanged", RERANK_MODE, type(exc).__name__, exc,
                )

    return {
        "corpus": corpus,
        "index": index,
        "dense": dense,
        "lexicon": lexicon,
        "tokenize": tokenize_v2,
        "reranker": reranker,
    }


def _build_reranker(corpus, lexicon):
    """Construct the configured reranker from the SAME module the benchmark scored."""
    from retrieval_lab.rerank import (
        CrossEncoderReranker,
        FusionReranker,
        IdentityReranker,
        MetadataRuleReranker,
    )

    def bge():
        return CrossEncoderReranker(
            corpus, lexicon, name="bge_v2m3_structq", model_name="BAAI/bge-reranker-v2-m3",
            doc_format="contextual", query_format="structured", max_length=512, batch_size=16,
        )

    if RERANK_MODE == "fuse_bge_first":
        return FusionReranker("fuse_bge_first", [bge(), IdentityReranker()])
    if RERANK_MODE == "bge_v2m3_structq":
        return bge()
    return MetadataRuleReranker(corpus, lexicon)


def get_state() -> dict[str, Any]:
    if "index" not in _STATE:
        with _LOCK:
            if "index" not in _STATE:
                _STATE.update(_build())
    return _STATE


def reset_state() -> None:
    """Drop the cached index (used by tests that rebuild the corpus)."""
    with _LOCK:
        _STATE.clear()


# --- Chroma `where` -> in-memory predicate ------------------------------------------------

def _matches(meta: dict[str, Any], where: dict[str, Any] | None) -> bool:
    """Support the filter shapes retrieval_policy / rag_router actually emit.

    Handles {field: value}, {field: {"$in": [...]}}, {field: {"$eq"/"$ne": v}} and {"$and"/"$or": [...]}.
    An unrecognised operator is treated as "no constraint" rather than silently excluding
    everything - over-filtering would look like a retrieval miss and be hard to trace.
    """
    if not where:
        return True
    for key, cond in where.items():
        if key == "$and":
            if not all(_matches(meta, c) for c in cond):
                return False
            continue
        if key == "$or":
            if not any(_matches(meta, c) for c in cond):
                return False
            continue
        value = meta.get(key)
        if isinstance(cond, dict):
            if "$in" in cond and value not in cond["$in"]:
                return False
            if "$nin" in cond and value in cond["$nin"]:
                return False
            if "$eq" in cond and value != cond["$eq"]:
                return False
            if "$ne" in cond and value == cond["$ne"]:
                return False
        elif value != cond:
            return False
    return True


_SCOPE_CACHE: dict[str, set[int]] = {}


def _allowed_indices(corpus, where: dict[str, Any] | None) -> set[int] | None:
    """Document indices matching `where`, cached per filter.

    Returns None for "no constraint" so the hot path skips the membership test entirely.
    A scope is stable across queries (it depends only on the student's profile), so the scan
    over the corpus happens once per distinct filter rather than once per question.
    """
    if not where:
        return None
    key = json.dumps(where, sort_keys=True, default=str)
    cached = _SCOPE_CACHE.get(key)
    if cached is None:
        cached = {
            i for i, chunk in enumerate(corpus.chunks) if _matches(chunk.meta, where)
        }
        _SCOPE_CACHE[key] = cached
    return cached


def _filter_keys(where: dict[str, Any] | None) -> set[str]:
    """Every metadata field name referenced anywhere in a (possibly nested) filter."""
    out: set[str] = set()
    if not where:
        return out
    for key, cond in where.items():
        if key in ("$and", "$or"):
            for clause in cond:
                out |= _filter_keys(clause)
        else:
            out.add(key)
    return out


def _unknown_filter_keys(corpus, where: dict[str, Any] | None) -> set[str]:
    """Filter fields that no chunk in the corpus carries (a schema mismatch, not a miss)."""
    keys = _filter_keys(where)
    if not keys:
        return set()
    known: set[str] = set()
    for chunk in corpus.chunks[:512]:  # the schema is uniform; a sample settles it
        known |= set(chunk.meta)
    return keys - known


# Corpora this retriever actually indexes. The lab corpus is built from
# data/degree_requirements/** and data/minors/** only - it holds NO instructor reviews, NO
# exams and NO user-uploaded documents, all of which live solely in Chroma.
COVERED_DOCUMENT_TYPES = frozenset({"course", "minor"})
_COVERAGE_FIELDS = ("documentType", "data_role")


def _uncovered_filter_values(corpus, where: dict[str, Any] | None) -> dict[str, list]:
    """Filter constraints this corpus structurally cannot satisfy.

    `_unknown_filter_keys` catches a filter naming a field that does not exist. This catches
    the subtler and more dangerous case: the field exists, but the requested VALUE belongs to
    a corpus this retriever does not index at all - e.g. `{"documentType": "review"}`, which
    the router emits for instructor-review questions. Matching zero chunks then looks
    identical to "no such curriculum", so the request would be answered from no evidence
    instead of falling back to Chroma, which does hold reviews and exams.
    """
    if not where:
        return {}

    def collect(w: dict[str, Any], out: dict[str, set]) -> None:
        for key, cond in w.items():
            if key in ("$and", "$or"):
                for clause in cond:
                    collect(clause, out)
            elif key in _COVERAGE_FIELDS:
                vals = cond.get("$in") if isinstance(cond, dict) else [cond]
                if vals:
                    out.setdefault(key, set()).update(str(v) for v in vals)

    requested: dict[str, set] = {}
    collect(where, requested)
    if not requested:
        return {}

    present: dict[str, set] = {f: set() for f in _COVERAGE_FIELDS}
    for chunk in corpus.chunks[:512]:
        for f in _COVERAGE_FIELDS:
            v = chunk.meta.get(f)
            if v is not None:
                present[f].add(str(v))

    uncovered: dict[str, list] = {}
    for field, wanted in requested.items():
        missing = sorted(wanted - present.get(field, set()))
        # Only a filter whose values are ENTIRELY outside our coverage is a routing error.
        # A mixed filter (e.g. course + review) still has something we can legitimately serve.
        if missing and len(missing) == len(wanted):
            uncovered[field] = missing
    return uncovered


def _to_document(chunk, score: float, retriever: str) -> Document:
    """Preserve every metadata field the answer generator and citation layer rely on."""
    meta = dict(chunk.meta)
    meta.setdefault("chunk_id", chunk.chunk_id)
    meta.setdefault("source", chunk.source_document or chunk.meta.get("source", ""))
    meta["_retriever"] = retriever
    meta["_score"] = float(score)
    return Document(page_content=chunk.text, metadata=meta)


def lab_search(
    query: str,
    *,
    top_k: int = 10,
    metadata_filter: dict[str, Any] | None = None,
    candidate_depth: int = 200,
    use_metadata_boost: bool = True,
) -> list[Document]:
    """Rank with the benchmarked BM25F + metadata-boost pipeline.

    `metadata_filter` is the same Chroma `where` the profile-aware policy produces; it is
    applied as a hard scope (the student's program/term), exactly as the Chroma path does,
    so switching retriever never widens the corpus beyond what the policy allows.
    """
    from retrieval_lab.fusion import (
        apply_metadata_boost,
        extract_query_metadata,
        reciprocal_rank_fusion,
    )

    state = get_state()
    corpus, index, dense = state["corpus"], state["index"], state["dense"]

    # A blank/zero-token query must return nothing. BM25F naturally returns [] (no query terms),
    # but a dense encoder happily embeds the empty string and hands back its nearest
    # neighbours, which would surface arbitrary curriculum rows as if they were evidence.
    tokens = state["tokenize"](query)
    if not tokens:
        return []

    depth = max(candidate_depth, top_k)
    allowed = _allowed_indices(corpus, metadata_filter)
    if allowed is not None and not allowed:
        # An empty scope has two very different causes and they must not be conflated:
        #
        #  * every field in the filter is a key the corpus actually has, and simply no chunk
        #    matches (e.g. a curriculum we do not hold). Returning [] is correct - the caller
        #    surfaces a data-unavailable message.
        #  * the filter references a field the corpus does not carry at all. Then [] is a lie:
        #    the evidence exists, we just cannot see it. Raising hands the request to the
        #    caller's fallback ladder instead of silently answering from no context, which is
        #    exactly the silent downgrade this module promises never to do.
        unknown = _unknown_filter_keys(corpus, metadata_filter)
        if unknown:
            raise RuntimeError(
                f"metadata filter references field(s) absent from the retrieval_lab corpus: "
                f"{sorted(unknown)}; filter={metadata_filter}"
            )
        uncovered = _uncovered_filter_values(corpus, metadata_filter)
        if uncovered:
            raise RuntimeError(
                f"retrieval_lab does not index {uncovered}; this corpus (reviews / exams / "
                f"uploaded documents) lives only in Chroma. Routing error - falling back. "
                f"filter={metadata_filter}"
            )
        logger.info("lab_retriever: metadata filter %s matched no chunks", metadata_filter)
        return []

    hits = index.search(tokens, top=depth, allowed=allowed)
    ranking = [(h.chunk_id, h.score) for h in hits]
    retriever_name = "bm25f"

    if dense is not None:
        # Rank-based fusion: BM25F scores and cosine similarities are on incompatible scales,
        # so they are never added directly.
        dense_hits = dense.search(query, top=depth, allowed=allowed)
        ranking = reciprocal_rank_fusion([ranking, dense_hits], k=RRF_K, top=depth)
        retriever_name = "hybrid_bm25f_e5"

    if use_metadata_boost:
        meta = extract_query_metadata(query, state["lexicon"])
        ranking = apply_metadata_boost(ranking, corpus, meta, top=depth)
        retriever_name += "_meta"

    reranker = state.get("reranker")
    if reranker is not None and ranking:
        # Rerank only the shallow head, then keep the untouched tail behind it. The reranked
        # window is what the benchmark measured; splicing the tail back means a reranker can
        # never shorten the result, so Hit@10 is preserved exactly as measured.
        head = ranking[:RERANK_DEPTH]
        tail = ranking[RERANK_DEPTH:]
        try:
            reordered = reranker.rerank(query, [c for c, _ in head], head)
            if {c for c, _ in reordered} == {c for c, _ in head}:
                ranking = list(reordered) + tail
                retriever_name += f"+{RERANK_MODE}"
            else:
                logger.error("lab_retriever: %s did not return a permutation; keeping the "
                             "first-stage order", RERANK_MODE)
        except Exception as exc:
            logger.error("lab_retriever: reranking failed (%s: %s); returning the first-stage "
                         "order unchanged", type(exc).__name__, exc)

    by_id = corpus.by_id
    return [
        _to_document(by_id[cid], score, retriever_name)
        for cid, score in ranking[:top_k]
        if cid in by_id
    ]
