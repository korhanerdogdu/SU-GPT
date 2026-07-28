# adviSU reranking benchmark — measured, and deliberately not enabled by default

**Date:** 2026-07-28 · **Branch:** `advisu-v2-dev` · **First stage:** `hybrid_bm25f_e5_meta`
**Corpus fingerprint:** `9918ab2b1f34ebe9` · **Hardware:** Apple M2 (MPS), torch 2.13.0
**Runs:** `outputs/reranking/dev__{cheap,msmarco,bge,fusebge,qwenfull3,posthoc}`, `outputs/reranking/test__frozen_test`

## 1. Executive result

| | Hit@1 | Recall@1 | Recall@3 | MRR@10 | nDCG@10 | Hit@10 | P95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `original_ranking` (no reranking) | 0.7535 | 0.7505 | 0.8747 | 0.8228 | 0.8491 | 0.9419 | ~0 ms |
| `metadata_rule_rerank` | 0.7615 | 0.7585 | 0.8768 | 0.8283 | 0.8534 | 0.9419 | 1 ms |
| `bge_v2m3_structq` | 0.7936 | 0.7886 | **0.9178** | 0.8557 | 0.8742 | 0.9419 | 2,005 ms |
| **`fuse_bge_first`** | **0.7976** | **0.7946** | 0.9088 | **0.8567** | **0.8754** | 0.9419 | 1,471 ms |

Held-out test, n = 499. Paired bootstrap (10,000 resamples, seed 20260727) for
`fuse_bge_first` vs `original_ranking`:

| Metric | Δ | 95% CI | p | improved / harmed / tied |
|---|---:|---:|---:|---:|
| Hit@1 | +0.0441 | [+0.0180, +0.0721] | 0.0016 | 34 / 12 / 453 |
| Recall@1 | +0.0441 | [+0.0190, +0.0711] | 0.0015 | 34 / 12 / 453 |
| Recall@3 | +0.0341 | [+0.0140, +0.0541] | 0.0017 | 22 / 5 / 472 |
| MRR@10 | +0.0340 | [+0.0192, +0.0492] | 0.0001 | 66 / 19 / 414 |
| nDCG@10 | +0.0263 | [+0.0154, +0.0374] | 0.0001 | 67 / 19 / 413 |
| Hit@10 | 0.0000 | [0, 0] | 1.000 | 0 / 0 / 499 |

**Verdict: qualifies on quality, fails on latency.** Criteria 1–6 of the frozen rule all pass;
criterion 7 (P95 ≤ 1000 ms) fails at 1,471 ms. Per the pre-registration it ships **behind a
flag**, not as the default. `ADVISU_RERANK=fuse_bge_first` enables it.

## 2. Candidate ceiling — why depth 10

A reranker cannot add a candidate, so the pool bounds it absolutely. Measured on dev (n = 238)
by an oracle that places every relevant candidate ahead of every irrelevant one:

| Depth | Candidate Hit | Candidate AllGold | Oracle Hit@1 | Oracle Recall@3 | Oracle MRR@10 | No gold in pool |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 0.9580 | 0.9496 | 0.9580 | 0.9538 | 0.9580 | 10 |
| 20 | 0.9622 | 0.9538 | 0.9622 | 0.9580 | 0.9622 | 9 |
| 25 | 0.9622 | 0.9538 | 0.9622 | 0.9580 | 0.9622 | 9 |
| 50 | 0.9664 | 0.9580 | 0.9664 | 0.9622 | 0.9664 | 8 |
| 100 | 0.9706 | 0.9622 | 0.9706 | 0.9664 | 0.9706 | 7 |

Two consequences drove every later decision:

1. **Depth 100 recovers three queries over depth 10.** The task brief assumed
   Recall@25/50 ≈ 1.00; it is 0.962/0.966. Reranking a deeper pool is not worth its latency here.
2. **Ordering headroom is large**: achieved Hit@1 0.8109 against an oracle of 0.9580 at depth 10.
   The opportunity is real — it just is not in the pool size.

At depth 10 the pool *is* the top 10, so `Hit@10` cannot regress; at depth 25 it demonstrably does
(ms-marco: 0.9580 → 0.8529).

## 3. Development results (n = 238)

| Reranker | Hit@1 | Recall@3 | MRR@10 | Hit@10 | P95 |
|---|---:|---:|---:|---:|---:|
| `fuse_bge_first` | 0.8403 | 0.9181 | 0.8847 | 0.9580 | 1,700 ms |
| `original_ranking` | 0.8109 | 0.9034 | 0.8670 | 0.9580 | ~0 ms |
| `metadata_rule_rerank` | 0.8109 | 0.9034 | 0.8677 | 0.9580 | 1 ms |
| `bge_v2m3_structq` | 0.8067 | 0.9160 | 0.8626 | 0.9580 | 3,619 ms |
| `fuse_qwen_first` | 0.7899 | 0.9286 | 0.8625 | 0.9580 | ~3,300 ms |
| `bge_v2m3` | 0.7941 | 0.9034 | 0.8564 | 0.9580 | 3,282 ms |
| `qwen3_06b_domain` | 0.7521 | 0.9202 | 0.8401 | 0.9580 | 3,341 ms |
| `msmarco_contextual` | 0.7437 | 0.8403 | 0.8118 | 0.9580 | 102 ms |
| `bge_v2m3_raw` | 0.7227 | 0.9034 | 0.8197 | 0.9580 | 4,244 ms |
| `qwen3_06b_default` | 0.7143 | 0.8571 | 0.7982 | 0.9580 | 3,148 ms |
| `existing_msmarco` (incumbent) | 0.5504 | 0.7353 | 0.6727 | 0.9580 | 81 ms |

**Every zero-shot reranker alone lost to the first-stage order on dev.** Only fusion exceeded it,
and on dev even that was not significant (+0.0294, CI [−0.0042, +0.0630], p = 0.145).

**Dev underestimated the effect.** On test the same configuration reached +0.0441 with a CI
excluding zero. This is the ordinary reason to hold out a test split — n = 499 has roughly twice
the power of n = 238 — and it is why the finalist was carried to test despite failing on dev.

## 4. What actually drives reranking quality here

### Input format beats model choice

| Model | raw document | + metadata header | + structured query |
|---|---:|---:|---:|
| ms-marco-MiniLM-L-6-v2 | 0.5504 | 0.7437 (+0.193) | — |
| bge-reranker-v2-m3 | 0.7227 | 0.7941 (+0.071) | 0.8067 (+0.084) |

Dev Hit@1. The effect is large, consistent across two unrelated model families, and in the
predicted direction: adviSU chunks are separated by structured fields, so a reranker that never
sees `[Program: …] [Catalog year: …] [Requirement type: …]` is ranking on the one part of the text
that does *not* discriminate. This is the same mechanism that makes generic web-QA cross-encoders
actively harmful on this corpus.

### Neural rerankers improve the body and damage position 1

`qwen3_06b_domain` reaches Recall@3 0.9202 (above the 0.9034 baseline) while its Hit@1 is 0.7521
(well below 0.8109). `bge_v2m3_structq` shows the same asymmetry. That is precisely the profile
rank fusion repairs: the first stage is better at position 1, the cross-encoder is better through
the body, and RRF keeps both opinions.

### The metadata rule reranker is a no-op — and that is informative

`metadata_rule_rerank` moves dev Hit@1 by +0.000 and MRR by +0.0007, because `hybrid_meta`
*already* applies metadata and record-type boosting. The structural signal is fully consumed
upstream, which is why the remaining headroom needed a genuine relevance model and why the
structured-feature learning-to-rank direction was not pursued further.

### Domain instruction wording matters

Qwen3-Reranker-0.6B with an adviSU-specific instruction scores 0.7521 against 0.7143 with the
generic web-search instruction (+0.038 Hit@1). Selected on dev only.

## 5. Subgroup analysis (test, Hit@1, `fuse_bge_first` vs baseline)

| Subgroup | n | Baseline | Fusion | Δ |
|---|---:|---:|---:|---:|
| has_catalog_year | 498 | 0.755 | 0.799 | +0.044 |
| single_evidence | 483 | 0.772 | 0.818 | +0.046 |
| lang:en | 456 | 0.759 | 0.800 | +0.042 |
| has_course_code | 326 | 0.926 | 0.939 | +0.012 |
| type:course_requirement | 210 | 0.543 | 0.619 | +0.076 |
| type:factual_lookup | 157 | 0.924 | 0.955 | +0.032 |
| hard_negatives | 89 | 0.978 | 0.955 | −0.022 |
| minor | 75 | 0.573 | 0.573 | 0.000 |
| type:program_specific | 47 | 1.000 | 0.979 | −0.021 |
| cross_language (TR→EN) | 43 | 0.698 | 0.767 | +0.070 |
| type:catalog_year | 42 | 0.952 | 0.929 | −0.024 |
| multi_evidence | 16 | 0.188 | 0.188 | 0.000 |

No regression approaches the −0.10 disqualification threshold. The three small negatives are all
in subgroups the first stage already solves near-perfectly (0.95–1.00), where a reranker has
nothing to gain and a little to lose — the expected shape, not a warning sign.

## 6. Failure analysis

- **Unreachable queries.** 29 of 499 test queries have no gold chunk in the top 10; no reranker
  can fix those. They are the difference between Hit@10 0.9419 and 1.000.
- **Multi-evidence stays broken** (0.188, unchanged). Relevance reranking cannot help: the
  problem is that the *set* is incomplete, not that it is mis-ordered. This is where an evidence
  coverage selector (task brief §15) would apply — untested, see limitations.
- **Minor queries unchanged** (0.573). The gold is a programme-level profile row; the reranker
  neither helps nor harms.

## 7. Figures

Generated from result files by `server/evaluation/make_reranking_figures.py`; every plotted value
is read from disk. PNG + SVG + PDF in `outputs/reranking/figures/`.

| Figure | Conclusion |
|---|---|
| `fig1_rerank_recall_at_k` | Fusion lifts K = 1–3 while tracking the baseline at K = 10; oracle line shows remaining headroom. |
| `fig2_gold_rank_ecdf` | Distribution of the first relevant rank shifts left under fusion. |
| `fig3_hit1_recall3_improvement` | Hit@1 and Recall@3 deltas with paired-bootstrap CIs. |
| `fig4_rank_shift` | Per-query gold-rank movement; improvements outnumber regressions ~3:1. |
| `fig5_win_tie_loss` | 34 improved / 12 harmed / 453 tied on Hit@1. |
| `fig6_quality_latency` | Only two systems are Pareto-efficient — no reranking (~0 ms) and fusion (1.5 s). Everything else is slower *and* worse. |
| `fig7_subgroup_delta` | Diverging heatmap; no critical subgroup collapse. |
| `fig9_candidate_depth` | Oracle ceiling is flat in depth — the argument against deep reranking. |
| `fig10_oracle_gap` | Fusion captures roughly a third of the available Hit@1 headroom. |

## 8. Production integration

**Reranking is off by default.** `DEFAULT_RETRIEVAL_MODE=hybrid_meta` remains the first stage.

```bash
ADVISU_RERANK=fuse_bge_first   # enable (adds ~1.5 s P95)
ADVISU_RERANK=off              # default
ADVISU_RERANK_DEPTH=10         # candidate pool; >10 risks Hit@10 (measured)
```

The benchmark and production share `retrieval_lab/rerank.py` — there is no second
implementation. On model-load failure or timeout the first-stage order is returned unchanged and
the fallback is logged at ERROR; an empty evidence set is never returned.

**The incumbent CrossEncoder path should stay disabled.** `ENABLE_RERANKING=true` with
`hybrid_rerank` reproduces `existing_msmarco`, measured 26 Hit@1 points below no reranking at all.

## 9. Experiments executed

| Experiment | Status |
|---|---|
| `original_ranking`, `metadata_rule_rerank`, `fuse_rules_first` (depths 10/25/50) | completed |
| `existing_msmarco`, `msmarco_contextual` (10/25) | completed |
| `bge_v2m3` × {raw, contextual, structured query} (10/25) | completed |
| `fuse_bge_first`, `_w2`, `_rules` | completed |
| `qwen3_06b` × {domain, default instruction} | completed (after an MPS OOM required batch 8→2, max_len 1024→512) |
| `fuse_qwen_first` | completed via exact post-hoc RRF (validated against the live path) |
| Frozen test: baseline + rule + BGE + fusion | completed |
| Qwen3-4B / 8B | **not run** — 0.6B already ~2.3 s/query at 4× the latency budget; larger variants cannot be production-feasible on this hardware |
| Domain fine-tuning (Wave 3) | **not run** — pre-registered to escalate only if zero-shot left headroom *and* a candidate was production-viable; the binding constraint is latency, which fine-tuning does not fix |
| Listwise / setwise reranking (Wave 4) | **not run** — same reason: an LLM listwise pass costs more than the 1.5 s already over budget |
| Evidence-coverage selection (§15) | **not run** — the correct fix for multi-evidence, but out of scope once reranking was ruled out as default |

## 10. Limitations

1. **Latency measured on one laptop, single process, MPS.** A GPU server would change the
   production verdict entirely — the quality result would stand and the 1,471 ms would not.
2. **Templated benchmark**, inherited from the retrieval benchmark: wording is corpus-derived and
   favours structural signals. See `docs/retrieval_benchmark_report.md` §10.
3. **Dev-selection bias.** Input formats, depths, fusion weights and models were all compared on
   dev, so dev numbers are optimistically biased. Test was scored once, for four frozen finalists.
4. **Multi-evidence (n = 16) and minor (n = 75) subgroups are small**; their deltas are
   directional, not precise.
5. **Reranker scores are not calibrated probabilities.** The Qwen3 value is a two-way softmax over
   `yes`/`no` token logits — a prompt-conditioned likelihood ratio, reported as a score only. No
   answerability/abstention analysis (§17) was performed.
6. **Positional-bias testing (§13.2) was not needed** because no listwise reranker was run; it
   remains required before any listwise method is adopted.

## 11. Reproduction

```bash
cd server
# candidate pools (100 deep) + ceiling
../.venv/bin/python evaluation/run_retrieval_lab.py --split dev --methods hybrid_bm25f_e5_meta \
    --out-tag pools --save-depth 100
../.venv/bin/python evaluation/candidate_ceiling.py ../outputs/retrieval_lab/dev__pools \
    --method hybrid_bm25f_e5_meta --split dev

# reranker sweeps (dev)
../.venv/bin/python evaluation/run_reranking_lab.py --run ../outputs/retrieval_lab/dev__pools \
    --split dev --rerankers original_ranking,metadata_rule_rerank,bge_v2m3_structq,fuse_bge_first \
    --depths 10 --out-tag cheap

# frozen test
../.venv/bin/python evaluation/run_reranking_lab.py --run ../outputs/retrieval_lab/test__final_v3 \
    --split test --rerankers original_ranking,metadata_rule_rerank,bge_v2m3_structq,fuse_bge_first \
    --depths 10 --out-tag frozen_test

# figures
../.venv/bin/python evaluation/make_reranking_figures.py \
    ../outputs/reranking/dev__cheap ../outputs/reranking/dev__bge ../outputs/reranking/dev__fusebge \
    --ceiling ../outputs/retrieval_lab/dev__pools/reranking/candidate_ceiling.csv
```
