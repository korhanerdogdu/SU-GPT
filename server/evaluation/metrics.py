from __future__ import annotations

"""
Retrieval metrics for the adviSU benchmark (CLAUDE.md Section 5.4).

Binary relevance: a retrieved item is relevant iff its id appears in the question's ground-truth
id set. Ground truth comes from the corpus itself (see build_benchmark.py), never from a model.

Two granularities, because the brief asks for both:

  chunk-level   ids are `chunk_id`s. This is the meaningful one for this project.
  source-level  ids are `source_document`s. NOTE: for this corpus a whole program x term lives in
                ONE file (e.g. degree_requirements/CS/202401.jsonl), so once retrieval is scoped
                by program/term every hit shares the expected source and source-level scores sit
                at ceiling. It is reported for completeness and should NOT be read as evidence of
                retrieval quality. Chunk-level is the metric to quote.

No numpy dependency - these are small lists and the arithmetic is worth keeping readable.
"""

import math
from typing import Iterable, Sequence

DEFAULT_KS: tuple[int, ...] = (1, 3, 5, 10)


def _dedupe(ids: Iterable[str]) -> list[str]:
    """Preserve rank order, drop repeats (a chunk retrieved twice must not count twice)."""
    seen: set[str] = set()
    out: list[str] = []
    for value in ids:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def recall_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Fraction of the ground-truth set that appears in the top-k."""
    gold = set(relevant)
    if not gold:
        return float("nan")  # undefined, not zero - caller must skip unanswerable questions
    top = set(_dedupe(retrieved)[:k])
    return len(top & gold) / len(gold)


def precision_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    gold = set(relevant)
    if not gold:
        return float("nan")
    top = _dedupe(retrieved)[:k]
    if not top:
        return 0.0
    return sum(1 for item in top if item in gold) / len(top)


def mrr_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Reciprocal rank of the FIRST relevant hit within the top-k (0.0 if none)."""
    gold = set(relevant)
    if not gold:
        return float("nan")
    for rank, item in enumerate(_dedupe(retrieved)[:k], start=1):
        if item in gold:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Binary-relevance nDCG@k. IDCG assumes all gold items ranked first."""
    gold = set(relevant)
    if not gold:
        return float("nan")
    top = _dedupe(retrieved)[:k]
    dcg = sum(1.0 / math.log2(rank + 1) for rank, item in enumerate(top, start=1) if item in gold)
    ideal_hits = min(len(gold), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return (dcg / idcg) if idcg else 0.0


def evaluate_retrieval(
    retrieved: Sequence[str],
    relevant: Iterable[str],
    ks: Sequence[int] = DEFAULT_KS,
) -> dict[str, float]:
    """All metrics at all ks for one question. Values are NaN when ground truth is empty."""
    gold = list(relevant)
    out: dict[str, float] = {}
    for k in ks:
        out[f"recall@{k}"] = recall_at_k(retrieved, gold, k)
        out[f"precision@{k}"] = precision_at_k(retrieved, gold, k)
        out[f"mrr@{k}"] = mrr_at_k(retrieved, gold, k)
        out[f"ndcg@{k}"] = ndcg_at_k(retrieved, gold, k)
    return out


def mean_ignoring_nan(values: Iterable[float]) -> float | None:
    """Macro-average over questions, skipping undefined entries.

    Returns None (not 0.0) when nothing was measurable, so an empty result can never be
    mistaken for a measured score of zero in the report.
    """
    usable = [v for v in values if v is not None and not math.isnan(v)]
    if not usable:
        return None
    return sum(usable) / len(usable)


def aggregate(per_question: Sequence[dict[str, float]]) -> dict[str, float | None]:
    """Macro-average each metric across questions."""
    keys: list[str] = []
    for row in per_question:
        for key in row:
            if key not in keys:
                keys.append(key)
    return {key: mean_ignoring_nan([row.get(key, float("nan")) for row in per_question]) for key in keys}


# ---- answer-side signals -------------------------------------------------------------
# Deliberately conservative: these are *signals* to speed up manual labelling, not automatic
# correctness judgements. CLAUDE.md 5.5 keeps the real answer_correctness / faithfulness /
# hallucination_flag fields for human (or later RAGAS) labelling.

REFUSAL_MARKERS: tuple[str, ...] = (
    "do not provide enough information",
    "does not provide enough information",
    "not provide enough evidence",
    "no supporting source",
    "yeterli bilgi",
    "yeterli kaynak",
    "bilgi bulunmuyor",
    "bulunmamaktad",
    "kaynak bulamad",
    "veri bulunmuyor",
    "elimde",
    "sahip deg",
    "sahip değ",
)


def looks_like_refusal(answer: str) -> bool:
    """Heuristic: did the system decline for lack of evidence? Used only as a labelling hint."""
    text = (answer or "").lower()
    return any(marker in text for marker in REFUSAL_MARKERS)


def cited_expected_source(answer: str, expected_sources: Iterable[str]) -> bool | None:
    """Whether any expected source string appears in the answer text. None if nothing expected."""
    expected = [s for s in expected_sources if s]
    if not expected:
        return None
    text = (answer or "").lower()
    return any(source.lower() in text for source in expected)
