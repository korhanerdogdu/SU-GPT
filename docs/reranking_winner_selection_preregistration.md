# Pre-registration: reranking winner-selection rule

**Frozen 2026-07-28, before the held-out test split was scored with any reranker.**

All numbers cited below are from `dev` (n=238). The test split has not been touched by any
reranking experiment at the time of writing.

## Frozen experimental setup

- **First stage:** `hybrid_bm25f_e5_meta` (the retrieval benchmark winner), candidates frozen
  in `outputs/retrieval_lab/{dev__pools,test__final_v3}/rankings/`, 100 ids persisted per query.
- **Candidates are never regenerated** during reranker comparison. Every reranker sees
  byte-identical pools; candidate depth is the only permitted variable.
- **Corpus fingerprint:** `9918ab2b1f34ebe9`. Seed 20260727, 10,000 bootstrap resamples.

## Candidate ceiling (measured, dev)

| Depth | Candidate Hit | Oracle Hit@1 | Queries with no gold |
|------:|--------------:|-------------:|---------------------:|
| 10 | 0.9580 | 0.9580 | 10 |
| 25 | 0.9622 | 0.9622 | 9 |
| 50 | 0.9664 | 0.9664 | 8 |
| 100 | 0.9706 | 0.9706 | 7 |

Depth 10 captures 98.7% of the depth-100 oracle. **Default depth is therefore 10**, and any
deeper configuration must justify its extra latency with a measured quality gain, not with the
assumption that more candidates are better. At depth 10 the pool *is* the top 10, so `Hit@10`
is preserved by construction; at depth > 10 it can regress and must be checked.

## Baseline

`original_ranking` — the first-stage order, unchanged. Dev: Hit@1 0.8109, Recall@1 0.8067,
Recall@3 0.9034, MRR@10 0.8670, nDCG@10 0.8863, Hit@10 0.9580.

## Acceptance criteria (all must hold)

1. **Hit@1 strictly greater** than `original_ranking` on the held-out test split.
2. The 95% paired bootstrap CI for the **Hit@1** difference **excludes zero**. A nominal gain
   whose CI spans zero is not a win — the same standard applied to the retrieval benchmark.
3. `Recall@3` and `MRR@10` do not regress (point estimates ≥ baseline).
4. **`Hit@10` ≥ baseline Hit@10 − 0.005.** Rationale: the answer generator receives the top
   10, so losing evidence from that window trades a visible correctness loss for an ordering
   gain. The tolerance is deliberately tight because at depth 10 a correct reranker cannot
   lose anything at all.
5. No critical-subgroup collapse: no subgroup among `has_course_code`, `has_catalog_year`,
   `hard_negatives`, `cross_language`, `multi_evidence`, `minor` may fall more than 10
   absolute Hit@1 points below baseline.
6. Reproducible from saved configuration and the documented commands.
7. **Production-feasible**: P95 warm latency ≤ 1000 ms for the interactive path. A
   configuration that is better but slower than this is reported as a research result and
   NOT integrated as the default; it may ship behind a flag.

## Ranking rule among qualifying candidates

Lexicographic, stopping at the first criterion where the paired 95% CI separates them:

1. Hit@1 · 2. Recall@1 · 3. Recall@3 · 4. MRR@10 · 5. nDCG@10 ·
6. EvidenceSetRecall@5 · 7. Hit@10 preservation · 8. P95 warm latency · 9. operational simplicity

Two candidates are **tied** on a metric when the paired 95% bootstrap CI of their difference
contains zero; ties fall through to the next criterion.

## If nothing qualifies

`original_ranking` stays in production, reranking is left disabled, and the negative result is
reported with its diagnosis. This is a live possibility: as of freezing, **no zero-shot
reranker has beaten the baseline on Hit@1** (best single model `bge_v2m3_structq` 0.8067 vs
0.8109), and the best fused candidate `fuse_bge_first` improves Hit@1 by only +0.0294 with
CI [−0.0042, +0.0630], p=0.145 — which would **fail criterion 2** if it held on test.

## Declared risks

1. **Dev-selection overfitting.** Many configurations (input formats, depths, fusion weights)
   were compared on dev, so the best dev number is optimistically biased. The test split is the
   arbiter and is scored once per finalist.
2. **Templated benchmark.** Inherited from the retrieval benchmark: question wording is
   corpus-derived, which flatters lexical/structural signals. See
   `docs/retrieval_benchmark_report.md` §10.
3. **Latency measured on one laptop (MPS)**, single process. Cold-start (model download and
   load) and warm inference are reported separately; only warm latency is compared.
4. **Scores are not calibrated probabilities.** The Qwen3 reranker's value is a two-way softmax
   over `yes`/`no` token logits — a prompt-conditioned likelihood ratio, reported as a score
   only.
