"""Retrieval metrics for the SU-GPT benchmark runner.

The benchmark can define gold evidence as exact chunk IDs or, when those are
not available, as source labels. Chunk-level evidence is preferred because it
is the stricter measurement; source-level scoring is the documented fallback.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def normalize_id(value: object) -> str:
    return str(value or "").strip()


def normalize_source(value: object) -> str:
    return " ".join(str(value or "").strip().lower().split())


def recall_at_k(retrieved: Sequence[object], expected: Sequence[object], k: int) -> float:
    expected_set = {normalize_id(item) for item in expected if normalize_id(item)}
    if not expected_set:
        return 0.0
    retrieved_set = {normalize_id(item) for item in retrieved[:k] if normalize_id(item)}
    return len(expected_set & retrieved_set) / len(expected_set)


def mrr_at_k(retrieved: Sequence[object], expected: Sequence[object], k: int) -> float:
    expected_set = {normalize_id(item) for item in expected if normalize_id(item)}
    if not expected_set:
        return 0.0
    for rank, item in enumerate(retrieved[:k], start=1):
        if normalize_id(item) in expected_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: Sequence[object], expected: Sequence[object], k: int) -> float:
    expected_set = {normalize_id(item) for item in expected if normalize_id(item)}
    if not expected_set:
        return 0.0

    dcg = 0.0
    for rank, item in enumerate(retrieved[:k], start=1):
        if normalize_id(item) in expected_set:
            dcg += 1.0 / math.log2(rank + 1)

    ideal_hits = min(len(expected_set), k)
    ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / ideal_dcg if ideal_dcg else 0.0


def source_recall_at_k(retrieved_sources: Sequence[object], expected_sources: Sequence[object], k: int) -> float:
    return recall_at_k(
        [normalize_source(item) for item in retrieved_sources],
        [normalize_source(item) for item in expected_sources],
        k,
    )


def source_mrr_at_k(retrieved_sources: Sequence[object], expected_sources: Sequence[object], k: int) -> float:
    return mrr_at_k(
        [normalize_source(item) for item in retrieved_sources],
        [normalize_source(item) for item in expected_sources],
        k,
    )


def source_ndcg_at_k(retrieved_sources: Sequence[object], expected_sources: Sequence[object], k: int) -> float:
    return ndcg_at_k(
        [normalize_source(item) for item in retrieved_sources],
        [normalize_source(item) for item in expected_sources],
        k,
    )


def retrieval_hits(
    retrieved_chunk_ids: Sequence[object],
    retrieved_sources: Sequence[object],
    expected_chunk_ids: Sequence[object],
    expected_sources: Sequence[object],
) -> dict[str, bool]:
    chunk_hit = bool(
        {normalize_id(item) for item in retrieved_chunk_ids if normalize_id(item)}
        & {normalize_id(item) for item in expected_chunk_ids if normalize_id(item)}
    )
    source_hit = bool(
        {normalize_source(item) for item in retrieved_sources if normalize_source(item)}
        & {normalize_source(item) for item in expected_sources if normalize_source(item)}
    )
    return {"retrieval_hit_chunk": chunk_hit, "retrieval_hit_source": source_hit}


def retrieval_metrics(
    retrieved_chunk_ids: Sequence[object],
    retrieved_sources: Sequence[object],
    expected_chunk_ids: Sequence[object],
    expected_sources: Sequence[object],
    *,
    k: int,
) -> dict[str, float | bool]:
    """Compute primary retrieval metrics.

    If expected chunk IDs exist, metrics are chunk-level. If expected chunk IDs
    are empty, metrics fall back to source-level matching as required by
    CLAUDE.md Section 5.4.
    """

    hits = retrieval_hits(retrieved_chunk_ids, retrieved_sources, expected_chunk_ids, expected_sources)
    if [item for item in expected_chunk_ids if normalize_id(item)]:
        recall = recall_at_k(retrieved_chunk_ids, expected_chunk_ids, k)
        mrr = mrr_at_k(retrieved_chunk_ids, expected_chunk_ids, k)
        ndcg = ndcg_at_k(retrieved_chunk_ids, expected_chunk_ids, k)
    else:
        recall = source_recall_at_k(retrieved_sources, expected_sources, k)
        mrr = source_mrr_at_k(retrieved_sources, expected_sources, k)
        ndcg = source_ndcg_at_k(retrieved_sources, expected_sources, k)

    return {
        **hits,
        "recall_at_k": round(float(recall), 6),
        "mrr_at_k": round(float(mrr), 6),
        "ndcg_at_k": round(float(ndcg), 6),
    }
