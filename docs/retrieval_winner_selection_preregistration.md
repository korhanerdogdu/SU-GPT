# Pre-registration: retrieval winner-selection rule

**Frozen on 2026-07-27, before the held-out test split was scored even once.**

This file exists so the acceptance criteria cannot be adjusted after seeing the test numbers.
Everything below was written while only `train` and `dev` results existed.

## Splits

| Split | Items | Scorable (non-empty gold) | Used for |
|-------|------:|--------------------------:|----------|
| train | 478 | 478 | reserved; not used for any tuning in this round |
| dev   | 238 | 238 | all tuning: field weights, boosts, RRF constants, tokenizer, record-type policy |
| test  | 505 | 499 | scored **once**, only for the frozen finalists |

Split policy: partition by gold-chunk connected component, so no chunk that is gold for a
train/dev question is gold for a test question. The 22 items of the original hand-built
benchmark are pinned to `test`. Leakage audit (`build_retrieval_benchmark.py --> check_leakage`)
asserts zero shared gold chunks and zero identical question strings across splits, and it
fails the build if either is violated.

## Baseline

The mandatory baseline is the shipped BM25 (`modules/bm25_retriever.py`: production
tokenizer, k1=1.5, b=0.75). It is reported in two forms:

- `bm25_original` — including the production 3000-document subset cap that applies when
  standalone `bm25` mode runs without a narrowing metadata filter.
- `bm25_full_corpus` — identical scoring, whole 30,343-chunk corpus.

**The winner must beat `bm25_full_corpus`**, the stronger of the two. Beating only the capped
variant would be beating a defect, not a baseline.

## Acceptance criteria (all must hold)

1. Held-out `Recall@10` strictly greater than `bm25_full_corpus` `Recall@10` on the test split.
2. The 95% paired bootstrap CI for the `Recall@10` difference **excludes zero**
   (10,000 resamples over queries, seed 20260727), and the paired permutation p-value < 0.05.
3. No critical-subgroup catastrophic regression. Critical subgroups for adviSU:
   `has_course_code`, `has_catalog_year`, `type:catalog_year`, `type:program_specific`,
   `hard_negatives`, `cross_language` (Turkish query over English evidence), `minor`.
   A candidate is disqualified if it falls **more than 10 absolute points below
   `bm25_full_corpus`** on any of these, even if its aggregate is higher.
4. Reproducible from the saved config and the documented commands, on the hardware recorded
   in `run_config.json`.
5. Operates within the project's resource envelope (must build and serve on the target
   laptop; no dependency that cannot be installed from `server/requirements.txt`).

## Ranking rule among qualifying candidates

Strictly in this order:

1. `Recall@10`
2. `MRR@10`
3. `nDCG@10`
4. `Hit@1`
5. Multi-evidence performance (`evidence_set_recall@10`, then `all_gold@10`)
6. P95 single-query latency (lower wins)
7. Operational simplicity (fewer models / no GPU requirement / smaller index)

`Recall@50` is explicitly **not** a selection criterion. A candidate is not selected on
answer quality without separately demonstrating first-stage retrieval quality.

### Addendum — definition of a "tie" (added 2026-07-28, still before any test-split scoring)

The original rule ranked on `Recall@10` without saying when two candidates count as tied.
Dev exposed the gap: `hybrid_bm25f_e5_meta` (0.9580) and `metadata_bm25f` (0.9454) differ by
1.3 points, but the paired 95% CI on that difference is [-0.0126, +0.0420] (p=0.55, 7 queries
improved vs 4 harmed out of 238) — indistinguishable. Reading a nominal 1.3-point lead as a
win would be selecting on noise, which criterion 2 already forbids for the baseline
comparison; the same standard has to apply between candidates.

**Rule:** two candidates are *tied* on `Recall@10` when the paired 95% bootstrap CI for their
difference contains zero. Ties fall through to the tie-breaker list. Because tie-breakers 6
and 7 (latency, operational simplicity) then decide, a candidate needing a neural embedding
model, a GPU and a cached embedding matrix loses to a pure-lexical candidate it cannot
statistically separate itself from.

This is written before the test split has been scored even once; whichever way the test
numbers fall, this rule applies unchanged.

## If nothing qualifies

BM25 remains the production default, the failure is reported with a diagnosis of the
bottleneck, and no integration is performed.

## Declared limitation, acknowledged in advance

Benchmark question wording is templated and corpus-derived (see
`server/evaluation/build_retrieval_benchmark.py`). Templated phrasing reuses corpus
vocabulary and therefore **favours lexical retrievers**, so BM25-family numbers here are an
optimistic bound and any dense/hybrid margin is a conservative estimate. ~19% of generated
items use natural student phrasing with program names and human term labels instead of raw
codes, which reduces but does not eliminate the bias. This limitation is stated in the report
regardless of which method wins.
