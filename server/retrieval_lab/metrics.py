from __future__ import annotations

"""
Retrieval metrics and paired significance testing.

GOLD SEMANTICS (documented here because the numbers are meaningless without it)
A benchmark item carries `expected_chunk_ids`, a set of chunks that are ALL acceptable
evidence for the question. Two different questions in adviSU need two different readings of
that set, so both are reported and never mixed:

  * "any-gold" (Recall@K, MRR, nDCG, Hit@1) - the question is answered if ANY gold chunk is
    retrieved. Correct for the common case where several rows restate the same fact (a
    category_pool row and the rule row that repeats its minimum are equally good citations).

  * "all-gold" (EvidenceSetRecall@K, AllGold@K) - the question needs the WHOLE set. Reported
    separately for the multi-evidence subset, where retrieving one of three required rows is
    a partial answer, not a win. EvidenceSetRecall@K is the *fraction* of the gold set found
    in the top K; AllGold@K is the stricter 0/1 "did we get every one of them".

For single-gold questions the two collapse to the same value, which is why the multi-evidence
subgroup is reported on its own rather than being averaged away into the headline.

Unanswerable items (empty gold) are excluded from every recall average - there is nothing to
retrieve - and are scored separately as "correctly returned nothing relevant" is a generation
concern, not a first-stage retrieval one.
"""

import math
import random
from dataclasses import dataclass, field
from typing import Sequence

K_VALUES: tuple[int, ...] = (1, 3, 5, 10, 20, 50)


# --- per-query metrics ----------------------------------------------------------------


def hit_at_k(ranked: Sequence[str], gold: set[str], k: int) -> float:
    """1.0 if AT LEAST ONE gold chunk appears in the top k."""
    return 1.0 if gold & set(ranked[:k]) else 0.0


def true_recall_at_k(ranked: Sequence[str], gold: set[str], k: int) -> float:
    """Fraction of the gold set present in the top k — the textbook Recall@K.

    Differs from `hit_at_k` only when a query has several gold chunks. Recall@1 is capped at
    1/|gold| for a multi-gold query, because one position cannot hold three chunks; that is a
    property of the metric, not a failure of the ranker, which is why Hit@1 and Recall@1 must
    be read together and never used interchangeably.
    """
    if not gold:
        return 0.0
    return len(gold & set(ranked[:k])) / len(gold)


def recall_at_k(ranked: Sequence[str], gold: set[str], k: int) -> float:
    """DEPRECATED NAME - this is any-gold Hit@K, kept so earlier runs stay reproducible.

    The retrieval benchmark (docs/retrieval_benchmark_report.md) reports its "Recall@K" column
    using this function, i.e. Hit@K semantics. For the 483 of 499 single-gold test queries the
    two coincide exactly; they diverge only on the 16 multi-evidence queries. New code should
    call `hit_at_k` or `true_recall_at_k` explicitly.
    """
    return hit_at_k(ranked, gold, k)


def evidence_set_recall_at_k(ranked: Sequence[str], gold: set[str], k: int) -> float:
    """Fraction of the required gold set present in the top k."""
    if not gold:
        return 0.0
    return len(gold & set(ranked[:k])) / len(gold)


def all_gold_at_k(ranked: Sequence[str], gold: set[str], k: int) -> float:
    """1.0 only if every gold chunk is in the top k."""
    if not gold:
        return 0.0
    return 1.0 if gold <= set(ranked[:k]) else 0.0


def reciprocal_rank(ranked: Sequence[str], gold: set[str], k: int = 10) -> float:
    for i, cid in enumerate(ranked[:k], 1):
        if cid in gold:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: Sequence[str], gold: set[str], k: int = 10) -> float:
    """Binary-relevance nDCG. Ideal DCG uses min(|gold|, k) relevant docs at the top."""
    if not gold:
        return 0.0
    dcg = sum(1.0 / math.log2(i + 1) for i, cid in enumerate(ranked[:k], 1) if cid in gold)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(gold), k) + 1))
    return dcg / ideal if ideal else 0.0


def hit_at_1(ranked: Sequence[str], gold: set[str]) -> float:
    return 1.0 if ranked and ranked[0] in gold else 0.0


# --- aggregation -----------------------------------------------------------------------


@dataclass
class QueryScore:
    query_id: str
    recall: dict[int, float] = field(default_factory=dict)
    evidence_recall: dict[int, float] = field(default_factory=dict)
    all_gold: dict[int, float] = field(default_factory=dict)
    mrr10: float = 0.0
    ndcg10: float = 0.0
    hit1: float = 0.0
    latency_ms: float = 0.0
    num_gold: int = 0
    multi_evidence: bool = False


def score_query(query_id: str, ranked: Sequence[str], gold: set[str], latency_ms: float = 0.0) -> QueryScore:
    return QueryScore(
        query_id=query_id,
        recall={k: recall_at_k(ranked, gold, k) for k in K_VALUES},
        evidence_recall={k: evidence_set_recall_at_k(ranked, gold, k) for k in (5, 10, 20)},
        all_gold={k: all_gold_at_k(ranked, gold, k) for k in (5, 10, 20)},
        mrr10=reciprocal_rank(ranked, gold, 10),
        ndcg10=ndcg_at_k(ranked, gold, 10),
        hit1=hit_at_1(ranked, gold),
        latency_ms=latency_ms,
        num_gold=len(gold),
        multi_evidence=len(gold) > 1,
    )


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def aggregate(scores: Sequence[QueryScore]) -> dict[str, float]:
    if not scores:
        return {}
    out: dict[str, float] = {"n_queries": float(len(scores))}
    for k in K_VALUES:
        out[f"recall@{k}"] = _mean([s.recall[k] for s in scores])
    for k in (5, 10, 20):
        out[f"evidence_set_recall@{k}"] = _mean([s.evidence_recall[k] for s in scores])
        out[f"all_gold@{k}"] = _mean([s.all_gold[k] for s in scores])
    out["mrr@10"] = _mean([s.mrr10 for s in scores])
    out["ndcg@10"] = _mean([s.ndcg10 for s in scores])
    out["hit@1"] = _mean([s.hit1 for s in scores])
    lat = sorted(s.latency_ms for s in scores)
    out["latency_mean_ms"] = _mean(lat)
    out["latency_p50_ms"] = lat[len(lat) // 2]
    out["latency_p95_ms"] = lat[min(len(lat) - 1, int(0.95 * len(lat)))]
    return out


# --- paired significance ----------------------------------------------------------------


@dataclass
class PairedComparison:
    metric: str
    baseline: str
    candidate: str
    n: int
    baseline_mean: float
    candidate_mean: float
    delta: float
    ci_low: float
    ci_high: float
    improved: int
    harmed: int
    tied: int
    p_value: float
    significant: bool


def paired_bootstrap(
    baseline: dict[str, float],
    candidate: dict[str, float],
    *,
    metric: str,
    baseline_name: str,
    candidate_name: str,
    n_boot: int = 10000,
    seed: int = 20260727,
    alpha: float = 0.05,
) -> PairedComparison:
    """Bootstrap the paired per-query difference (candidate - baseline).

    Resamples QUERIES (not the two systems independently) so the two systems are always
    compared on the same queries - the pairing is the whole point, since query difficulty
    dominates the variance. Also reports a two-sided permutation p-value on the same pairing.
    """
    qids = sorted(set(baseline) & set(candidate))
    diffs = [candidate[q] - baseline[q] for q in qids]
    n = len(diffs)
    if n == 0:
        raise ValueError("no overlapping queries between baseline and candidate")

    rng = random.Random(seed)
    boot: list[float] = []
    for _ in range(n_boot):
        s = 0.0
        for _ in range(n):
            s += diffs[rng.randrange(n)]
        boot.append(s / n)
    boot.sort()
    lo = boot[int((alpha / 2) * n_boot)]
    hi = boot[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]

    # Two-sided paired permutation test: flip the sign of each difference at random.
    observed = abs(sum(diffs) / n)
    rng2 = random.Random(seed + 1)
    extreme = 0
    n_perm = 10000
    for _ in range(n_perm):
        s = 0.0
        for d in diffs:
            s += d if rng2.random() < 0.5 else -d
        if abs(s / n) >= observed - 1e-12:
            extreme += 1
    p = (extreme + 1) / (n_perm + 1)

    return PairedComparison(
        metric=metric,
        baseline=baseline_name,
        candidate=candidate_name,
        n=n,
        baseline_mean=_mean([baseline[q] for q in qids]),
        candidate_mean=_mean([candidate[q] for q in qids]),
        delta=sum(diffs) / n,
        ci_low=lo,
        ci_high=hi,
        improved=sum(1 for d in diffs if d > 0),
        harmed=sum(1 for d in diffs if d < 0),
        tied=sum(1 for d in diffs if d == 0),
        p_value=p,
        significant=(lo > 0 or hi < 0),
    )
