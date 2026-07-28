# adviSU retrieval benchmark — replacing BM25 in the advising pipeline

**Date:** 2026-07-28 · **Branch:** `advisu-v2-dev` · **Corpus fingerprint:** `9918ab2b1f34ebe9` (30,343 chunks)
**Hardware:** Apple silicon (MPS), Python 3.12.9, torch 2.13.0 · **Run:** `outputs/retrieval_lab/test__final_v3/`

## 1. Executive result

The winner is **`hybrid_bm25f_e5_meta`** — field-weighted BM25F over the structured curriculum
corpus, fused by Reciprocal Rank Fusion with `multilingual-e5-small`, followed by soft
metadata and record-type boosts.

| | Recall@10 | Recall@1 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|---:|
| `bm25_original` (shipped, incl. 3000-doc cap) | 0.0681 | 0.0501 | 0.0567 | 0.0595 |
| `bm25_full_corpus` (fair baseline) | 0.5992 | 0.3367 | 0.4029 | 0.4481 |
| **`hybrid_bm25f_e5_meta` (winner)** | **0.9419** | 0.7535 | 0.8223 | 0.8487 |

- **Absolute improvement over the fair baseline:** +0.3427 Recall@10
- **Relative improvement:** +57.2%
- **95% paired bootstrap CI:** [+0.2986, +0.3888], permutation p = 0.0001
- **Per-query:** 178 improved, 7 harmed, 314 tied (n = 499)
- **Against the baseline as actually deployed** (`bm25_original`): +0.8738 Recall@10
- **Integrated:** yes — `DEFAULT_RETRIEVAL_MODE=hybrid_meta`, with BM25 retained as a
  configuration-controlled fallback.

## 2. What was measured against what

### Corpus

30,343 pre-chunked rows from `data/degree_requirements/**` (28,082) and `data/minors/**`
(2,261), covering 10 majors and 16 minors across catalog terms 202201–202501. The shape of
this corpus drives every result below: **90.6% of chunks are `degree_requirement_pool_course`
rows** — one row per (course × pool × program × term). Thousands of them are near-identical
prose that differs *only* in structured metadata. Retrieval here is mostly a disambiguation
problem, not a semantic-similarity problem.

### Benchmark

The existing `data/benchmark/questions.jsonl` holds 22 questions, only 16 with gold chunks.
At n = 16 one query is worth 6.25 points of Recall@10, a paired bootstrap CI spans most of
[0, 1], and the 13 required subgroups hold 1–3 queries each — no honest significance claim is
possible. The benchmark was therefore widened by
`server/evaluation/build_retrieval_benchmark.py` to **1,221 items**:

| Split | Items | Scorable | Turkish | Multi-evidence | Role |
|-------|------:|---------:|--------:|---------------:|------|
| train | 478 | 478 | 36 | 23 | held in reserve; unused this round |
| dev | 238 | 238 | 24 | 4 | **all** tuning |
| test | 505 | 499 | 43 | 16 | scored once, finalists only |

Every answerable question is generated from a real corpus row: the gold `expected_chunk_ids`
are that row's own `chunk_id` and the reference answer is built from its own fields, so
nothing is fabricated and `verify()` re-checks every id against the corpus. The 22 original
items — including the 6 hand-written unanswerable/misleading ones — are preserved verbatim and
pinned to `test`.

**Leakage control.** Splits partition by gold-chunk connected component: no chunk that is gold
for a train/dev question is gold for a test question. The build asserts zero shared gold
chunks and zero identical question strings across all three split pairs and fails if either is
violated. Audited output: `0 / 0` on all pairs.

> **Limitation, stated up front.** Question wording is templated. Templated phrasing reuses
> corpus vocabulary and therefore **favours lexical retrievers**, so BM25-family numbers here
> are an optimistic bound and the dense contribution is likely understated. An earlier draft
> was worse: it restated program, term and course verbatim in every question, and BM25F scored
> a perfect 1.000 on dev — a ceiling effect with no headroom. ~19% of items were then
> re-rendered in natural student phrasing using program names and human term labels
> ("I'm a Computer Science and Engineering student who started in Fall 2024-2025…"), which
> reduces but does not remove the bias. This bounds how strongly the headline generalises to
> real student prose.

### The two baselines

`modules/bm25_retriever.py` caps its candidate pool at `_SUBSET_CAP = 3000` documents. In
standalone `bm25` mode it runs `annotate_bm25(require_narrowing=False)` with no metadata
filter, so Chroma's `.get(limit=3000)` hands BM25 an arbitrary ~10% slice of a 30,343-chunk
corpus, and any gold chunk outside that slice is unreachable at **any** K. That is a real
property of the deployed system, and it is why `bm25_original` scores 0.0681.

Beating a defect proves nothing, so `bm25_full_corpus` runs identical scoring (same
tokenizer, k1 = 1.5, b = 0.75) over the whole corpus, and **the winner is required to beat
that stronger baseline**. All headline claims use `bm25_full_corpus`.

## 3. Results (held-out test split, n = 499 scorable)

| Method | R@1 | R@5 | R@10 | R@20 | R@50 | MRR@10 | nDCG@10 | Hit@1 | P95 ms | Index |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `bm25_original` | 0.0501 | 0.0661 | 0.0681 | 0.0681 | 0.0721 | 0.0567 | 0.0595 | 0.0501 | 5.7 | in-mem |
| `bm25_full_corpus` | 0.3367 | 0.4810 | 0.5992 | 0.7034 | 0.7655 | 0.4029 | 0.4481 | 0.3367 | 64.2 | in-mem |
| `bm25_v2tok` | 0.3387 | 0.4770 | 0.5932 | 0.6954 | 0.7635 | 0.4023 | 0.4461 | 0.3387 | 69.4 | in-mem |
| `dense_minilm` | 0.0621 | 0.2004 | 0.2685 | 0.3427 | 0.4429 | 0.1206 | 0.1557 | 0.0621 | 29.4 | 44 MB |
| `dense_e5_small` | 0.2625 | 0.4850 | 0.5832 | 0.6954 | 0.8116 | 0.3586 | 0.4119 | 0.2625 | 64.7 | 44 MB |
| `bm25f` | 0.6754 | 0.7255 | 0.7355 | 0.7615 | 0.8337 | 0.6924 | 0.7024 | 0.6754 | 168.3 | in-mem |
| `contextual_bm25` | 0.6293 | 0.7295 | 0.7475 | 0.7635 | 0.8016 | 0.6694 | 0.6881 | 0.6293 | 66.9 | in-mem |
| `hybrid_meta_rerank50` | 0.4309 | 0.6152 | 0.6894 | 0.7695 | 0.9098 | 0.5129 | 0.5547 | 0.4309 | 1023.5 | in-mem |
| `metadata_filter_bm25f` | 0.6834 | 0.7595 | 0.8056 | 0.8517 | 0.9279 | 0.7161 | 0.7374 | 0.6834 | 160.5 | in-mem |
| `hybrid_bm25f_e5` | 0.6092 | 0.7455 | 0.8156 | 0.8577 | 0.9018 | 0.6771 | 0.7099 | 0.6092 | 1132.1 | in-mem |
| `metadata_bm25f` | 0.7715 | 0.8818 | 0.9178 | 0.9178 | 0.9339 | 0.8169 | 0.8398 | 0.7715 | 199.6 | in-mem |
| **`hybrid_bm25f_e5_meta`** | 0.7535 | 0.9158 | **0.9419** | 0.9419 | 0.9519 | 0.8223 | 0.8487 | 0.7535 | 178.6 | 44 MB |

"in-mem" indices are Python inverted indices rebuilt at startup in 0.6–3.5 s; the 44 MB is the
cached `float32` embedding matrix (30,343 × 384).

### Statistical comparison vs `bm25_full_corpus`

| Method | Recall@10 Δ | 95% CI | p | Improved | Harmed | Tied | Critical subgroup regression |
|---|---:|---:|---:|---:|---:|---:|---|
| `hybrid_bm25f_e5_meta` | **+0.3427** | [+0.2986, +0.3888] | 0.0001 | 178 | 7 | 314 | none |
| `metadata_bm25f` | +0.3186 | [+0.2745, +0.3627] | 0.0001 | 166 | 7 | 326 | none |
| `hybrid_bm25f_e5` | +0.2164 | [+0.1784, +0.2565] | 0.0001 | 117 | 9 | 373 | none |
| `metadata_filter_bm25f` | +0.2064 | [+0.1703, +0.2425] | 0.0001 | 107 | 4 | 388 | none |
| `contextual_bm25` | +0.1483 | [+0.1142, +0.1824] | 0.0001 | 80 | 6 | 413 | none |
| `hybrid_meta_rerank50` | +0.0902 | [+0.0501, +0.1303] | 0.0001 | 78 | 33 | 388 | none |
| `bm25f` | +0.1363 | [+0.1022, +0.1703] | 0.0001 | 76 | 8 | 415 | none |
| `bm25_v2tok` | −0.0060 | [−0.0140, +0.0000] | 0.2488 | 0 | 3 | 496 | n/a (not significant) |
| `dense_e5_small` | −0.0160 | [−0.0681, +0.0341] | 0.5908 | 80 | 88 | 331 | n/a (not significant) |
| `dense_minilm` | −0.3307 | [−0.3848, −0.2766] | 0.0001 | 36 | 201 | 262 | severe |
| `bm25_original` | −0.5311 | [−0.5812, −0.4810] | 0.0001 | 17 | 282 | 200 | severe |

10,000 bootstrap resamples over queries, seed 20260727, paired; p from a two-sided paired
permutation test on the same pairing.

### Finalist head-to-head

| Metric | `metadata_bm25f` | `hybrid_bm25f_e5_meta` | Δ | 95% CI | p | Verdict |
|---|---:|---:|---:|---:|---:|---|
| Recall@10 | 0.9178 | 0.9419 | +0.0240 | [+0.0100, +0.0401] | 0.0044 | **significant** |
| MRR@10 | 0.8169 | 0.8223 | +0.0054 | [−0.0131, +0.0237] | 0.5590 | tie |
| nDCG@10 | 0.8398 | 0.8487 | +0.0089 | [−0.0057, +0.0237] | 0.2368 | tie |
| Hit@1 | 0.7715 | 0.7535 | −0.0180 | [−0.0461, +0.0100] | 0.2590 | tie |

The hybrid's Recall@10 lead is real (14 queries improved, 2 harmed), so under the
pre-registered rule it wins on the primary metric and no tie-breaker is reached. Note the
honest wrinkle: the hybrid is *nominally worse* at Hit@1, though not significantly.

### Multi-evidence

Reported on the **16 test items that actually have more than one gold chunk** — not the whole
split. (An earlier draft of this table printed whole-split values under this heading, which
made a weak result look strong; the whole-split figures are given underneath for continuity.)

Restricted to the 16 multi-evidence questions:

| Method | EvidenceSetRecall@10 | AllGold@10 | Hit@10 |
|---|---:|---:|---:|
| `bm25_full_corpus` | 0.0312 | 0.0000 | 0.0625 |
| `metadata_bm25f` | 0.1562 | 0.0625 | 0.2500 |
| `hybrid_bm25f_e5_meta` | **0.2188** | **0.0625** | **0.3750** |

**This is the weakest area of the system.** The winner retrieves the complete evidence set for
**1 of 16** multi-evidence questions and finds *any* gold chunk for only 6 of 16. It is still
3–7× the baseline, but the absolute level is poor and should not be described as solved.

Whole-split values (all 499 queries, dominated by the 483 single-gold ones, where
EvidenceSetRecall collapses to Hit@K):

| Method | EvidenceSetRecall@10 | AllGold@10 |
|---|---:|---:|
| `bm25_full_corpus` | 0.5982 | 0.5972 |
| `metadata_bm25f` | 0.9148 | 0.9118 |
| `hybrid_bm25f_e5_meta` | 0.9369 | 0.9319 |

## 4. Subgroup analysis (Recall@10)

| Subgroup | n | BM25 | `metadata_bm25f` | winner | Δ winner |
|---|---:|---:|---:|---:|---:|
| has_course_code | 326 | 0.825 | 0.975 | 0.975 | +0.150 |
| has_catalog_year | 498 | 0.600 | 0.920 | 0.942 | +0.341 |
| type:catalog_year | 42 | 0.500 | 1.000 | 1.000 | +0.500 |
| type:program_specific | 47 | 1.000 | 1.000 | 1.000 | +0.000 |
| hard_negatives | 89 | 0.764 | 1.000 | 1.000 | +0.236 |
| cross_language (TR→EN) | 43 | 0.721 | 0.744 | 0.791 | +0.070 |
| minor | 75 | 0.573 | 0.987 | 1.000 | +0.427 |
| multi_evidence | 16 | 0.062 | 0.250 | 0.375 | +0.312 |
| single_evidence | 483 | 0.617 | 0.940 | 0.961 | +0.344 |
| original_benchmark | 16 | 0.188 | 0.812 | 1.000 | +0.812 |
| type:factual_lookup | 157 | 0.815 | 1.000 | 1.000 | +0.185 |
| type:course_requirement | 210 | 0.343 | 0.857 | 0.905 | +0.562 |
| lang:en | 456 | 0.588 | 0.934 | 0.956 | +0.368 |

**No subgroup regresses.** The winner is ≥ baseline everywhere and strictly better in 12 of
13. The weakest subgroup in absolute terms is `multi_evidence` (0.375) — see §6.

## 5. Why the winner wins

The gains are structural, not semantic. In order of contribution:

1. **Field weighting (+0.136).** Concatenating metadata into one string lets a long body
   drown a two-token course code. BM25F normalizes each field by *that field's* average
   length and weights per-field term frequency before saturation, so a `course_code` hit keeps
   its weight regardless of body length. On a corpus where 90.6% of chunks differ only in
   structured fields, this is the difference between ranking "CS 414 in CS/202401" and ranking
   a random pool row that shares body prose.
2. **Metadata + record-type boosts (+0.182 more).** Two distinct signals:
   - *Entity agreement* — chunks matching an unambiguously extracted course code, catalog
     term or program get a bounded bonus. This is what lifts `type:catalog_year` from 0.500 to
     1.000: the correct term now outranks the same course in three sibling terms.
   - *Record granularity* — error analysis on dev showed "how many SU credits of free
     electives are required" returning individual pool-course rows (NS 206, NS 209) with the
     right program and term but the **wrong granularity**, because 27,490 course rows swamp
     332 category rows. Inferring the requested record type from the question lifted dev
     Recall@10 from 0.857 to 0.945 and multi-evidence from 0.000 to 0.500.
3. **Soft boosts beat hard filters (+0.112).** `metadata_filter_bm25f` (hard filter) scores
   0.8056 against `metadata_bm25f`'s 0.9178 with the identical extractor. A hard filter on a
   *predicted* entity is unrecoverable — one wrong program guess and the correct chunk cannot
   be returned at any K. A boost degrades gracefully.
4. **The dense half contributes a small, real +0.024.** Where that gain lands was checked
   directly rather than assumed: of the 14 test queries the hybrid wins over `metadata_bm25f`,
   **12 are English and only 2 are Turkish**, and 12 of 14 are `type:course_requirement`. So
   the earlier claim in this report that the dense gain "concentrates in cross-language items"
   was wrong — the cross-language subgroup does improve (0.744 → 0.791) but supplies a small
   minority of the wins. The accurate statement is narrower: E5 recovers a handful of
   requirement-phrased English questions whose wording diverges from corpus vocabulary. Given
   14 wins against 2 losses on 499 queries, this is a real but modest effect, and it is the
   *only* thing separating the hybrid from the pure-lexical runner-up.

### Negative results (reported, not hidden)

- **Dense retrieval alone never beats BM25.** `dense_e5_small` scores 0.5832 against the
  0.5992 baseline (CI spans zero — indistinguishable, not a win), and `dense_minilm` collapses
  to 0.2685. `dense_e5_small` *did* lead on dev (0.6092 vs 0.5840) and lost that lead on test;
  this is exactly what the held-out split is for. The mechanism is the corpus: embeddings blur
  the near-identical prose of sibling course rows and the disambiguating information lives in
  metadata they compress away.
- **The Turkish tokenizer fix, alone, does nothing** (−0.0060, p = 0.2488). It genuinely
  repairs token shattering — the shipped tokenizer's `[a-z0-9]+` regex turns `müfredatında`
  into `['m','fredat','nda']`, while the new one yields `['mufredatinda']` — but these Turkish
  queries are carried by course codes and 6-digit terms, which both tokenizers preserve. The
  fix is retained in the winner because it is a precondition for the boosts, not because it
  pays off on its own.
- **Cross-encoder reranking actively hurt.** `hybrid_meta_rerank50` scores 0.6894 against its
  own first stage's 0.9419 — a 25-point *loss*. Candidate recall at depth 50 was 0.9098, so
  the evidence was present and the reranker demoted it. `cross-encoder/ms-marco-MiniLM-L-6-v2`
  is trained on English web QA; it scores query–passage prose similarity and is blind to the
  program/term/category agreement that actually decides relevance here, so it reorders by
  surface similarity and destroys the metadata signal. A curriculum-domain reranker might
  help; this one is harmful and should not be enabled for this corpus.

## 6. Error analysis

Remaining failures of the winner (`recall@10 == 0`, 29 of 499):

| Category | Mechanism |
|---|---|
| Multi-evidence (10 of 16 items) | Questions needing *both* a `category_pool` row and the `rule` row restating its minimum. The record-type boost promotes the class but not reliably both members; `AllGold@10` is 0.9319 overall but the residue concentrates here. |
| Cross-language (9 of 43 TR items) | Turkish natural-phrasing items where the student names neither a course code nor a term, leaving only paraphrased prose against an English corpus. |
| Granularity residue | Aggregate questions where the phrasing carries no aggregate cue the record-type inference recognises. |

`bm25_original`'s failures are dominated by a category the other methods do not have at all:
the gold chunk is outside its 3000-document window, so `recall@50` is also 0 — unrecoverable
by any amount of reranking.

## 7. Figures

All generated from the result files by `server/evaluation/make_retrieval_figures.py`; no metric
is typed in by hand. PNG + SVG + PDF in `outputs/retrieval_lab/test__final_v3/figures/`.

| Figure | Conclusion |
|---|---|
| `fig1_recall_at_k` | The winner dominates at every K; the baseline needs K = 50 to reach what the winner achieves at K = 1. |
| `fig2_delta_vs_bm25` | Seven methods significantly beat BM25; `bm25_v2tok` and `dense_e5_small` straddle zero. |
| `fig3_quality_latency` | The winner is Pareto-optimal: higher Recall@10 *and* lower P95 than `metadata_bm25f` and `hybrid_bm25f_e5`. |
| `fig4_subgroups` | No dark cells for the winner — the gain is broad, not one easy subgroup. |
| `fig5_ranking_metrics` | The two finalists are visually indistinguishable on MRR/nDCG/Hit@1, matching the head-to-head ties. |
| `fig6_win_tie_loss` | 178 improved vs 7 harmed — the gain is not a few large swings. |
| `fig7_error_categories` | Winner failures concentrate in multi-evidence and cross-language; baseline failures are dominated by unreachable gold. |

## 8. Production integration

| File | Change |
|---|---|
| `server/retrieval_lab/` | **New.** `corpus`, `text`, `sparse` (BM25/BM25F), `dense`, `fusion`, `metrics`, `retrievers`. Shared by benchmark and production. |
| `server/modules/lab_retriever.py` | **New.** Production adapter: Chroma `where` → in-memory scope, ranking → `Document` with citation metadata intact. |
| `server/modules/retrieval_modes.py` | Added `hybrid_meta` mode delegating to `lab_retriever`; all five previous modes unchanged. |
| `server/modules/config.py` | `DEFAULT_RETRIEVAL_MODE` default `hybrid` → `hybrid_meta`, with the measured justification in a comment. |
| `.env.example` | `DEFAULT_RETRIEVAL_MODE=hybrid_meta`. |
| `server/tests/test_retrieval_lab.py` | **New.** 46 unit + regression tests. |
| `server/tests/test_advising.py` | Updated the mode-set assertion for the new mode. |
| `server/evaluation/` | **New.** `build_retrieval_benchmark.py`, `run_retrieval_lab.py`, `analyze_retrieval_lab.py`, `make_retrieval_figures.py`. |
| `.gitignore` | Ignore `outputs/retrieval_cache/` (regenerable embeddings) and ranking dumps. |

**The benchmark and production share one implementation.** `lab_retriever` builds the same
`BM25FIndex` + `DenseIndex` + `apply_metadata_boost` the benchmark scored, with the fusion
constants (`RRF_K = 60`, `CANDIDATE_DEPTH = 200`, `e5_small`) pinned in one place.

**Fallback ladder**, each step logged, none silent:

1. `hybrid_meta` (default) — full winner.
2. `ADVISU_LAB_DENSE=false` or E5 unavailable → BM25F + metadata only (0.9178), logged WARNING.
3. Lab index unbuildable → previous Chroma hybrid path, logged ERROR.
4. `DEFAULT_RETRIEVAL_MODE=hybrid` / `=bm25` → previous default / plain BM25 baseline.

**Scoping is preserved.** The profile-aware `metadata_filter` is applied as a *pre-filter*
inside both the sparse and dense search (masking before top-k selection), matching Chroma's
`where` semantics. An early version post-filtered and returned nothing for narrow scopes; a
unit test caught it.

## 9. Validation

| Check | Result |
|---|---|
| `pytest server/tests/` | **73 passed**, 0 failed |
| Retrieval-lab unit + regression tests | 46 passed |
| Pre-existing advising tests | 27 passed |
| End-to-end `retrieval_modes.retrieve("hybrid_meta", …)` | returns correctly scoped, citation-carrying documents |
| Benchmark leakage audit | 0 shared gold chunks, 0 identical questions across all split pairs |
| Gold-id verification | all gold ids exist in the corpus (30,343 known) |
| Figures vs raw results | `fig2` values cross-checked against `significance_results.json` |

Not run: linting and type-checking (no configured linter/type-checker in the repo); the
FastAPI app was not booted end-to-end because it requires MongoDB and a Groq key.

## 10. Limitations

1. **Templated benchmark.** The dominant limitation. Questions are corpus-derived, so lexical
   methods are flattered and the absolute numbers are optimistic. Ranking *between* methods is
   more trustworthy than any single value. Real student query logs would settle this.
2. **Multi-evidence is thin** — 16 test items. The 0.375 subgroup figure is directionally
   informative, not precise.
3. **Turkish is thin** — 43 test items, all machine-templated. The +0.070 cross-language gain
   from the dense half is the least certain claim in this report.
4. **Larger dense models untested.** `e5_base`, `e5_large_instruct` and `bge-m3` were not
   completed (see `docs/retrieval_lab.md`), so "dense does not help here" is established for
   small multilingual models only. The structural argument suggests larger models would not
   close a 0.34 gap, but that is reasoning, not measurement.
5. **`train` (478 items) was never used.** Tuning used dev only; the split exists for future
   fusion-weight or reranker fine-tuning.
6. **Latency is single-process on one laptop**, and the `hybrid_bm25f_e5` P95 of 1132 ms was
   measured while another job contended for the GPU — it is not comparable to the others.
   Winner and baseline figures were measured uncontended.
7. **BM25F field weights were hand-set** from domain reasoning and validated on dev, not swept
   exhaustively.
8. **The record-type router is partly fitted to this benchmark's own generator.** Its phrase
   list (`"credit requirement"`, `"requirements of"`, `"mezun"`, …) overlaps the templates in
   `build_retrieval_benchmark.py`; train↔test template overlap modulo entity is ~57%. No test
   data was used to choose them, so this is generator-fitting rather than test-set fitting, but
   it inflates precisely the component credited with the largest single gain in §5.2. On real
   student prose that component should be expected to contribute less.
9. **The winner's headline number survived a bug, which is worth recording.** An earlier
   version of the record-type router matched `"graduate"` as a bare substring, which also fires
   inside `"Undergraduate Program"` — a string in every program name in the corpus. Fixing it
   dropped test Recall@10 from 0.9419 to 0.9078, because the accident had been promoting the
   correct record class for minor-programme questions. Explicit intent cues were then added
   (chosen on dev, from the dev failure list) and the corrected system returns to 0.9419. The
   reported figure is therefore the corrected one, but it is the *third* scoring of the test
   split (original → post-bug-fix → post-cue-fix). Every tuning decision was made on dev, yet
   test numbers were observed between iterations, which is weaker than a single frozen
   scoring. Treat the headline as reliable to roughly the nearest point, not the fourth decimal.

## 11. Reproduction

```bash
python3 -m venv .venv && ./.venv/bin/pip install -r server/requirements.txt
./.venv/bin/pip install pytest matplotlib rank-bm25

cd server
../.venv/bin/python evaluation/build_retrieval_benchmark.py
../.venv/bin/python evaluation/run_retrieval_lab.py --split test --out-tag final_v3 \
    --methods bm25_original,bm25_full_corpus,bm25_v2tok,contextual_bm25,bm25f,\
metadata_bm25f,metadata_filter_bm25f,dense_minilm,dense_e5_small,hybrid_bm25f_e5,\
hybrid_bm25f_e5_meta,hybrid_meta_rerank50 --latency-sample 30
../.venv/bin/python evaluation/analyze_retrieval_lab.py ../outputs/retrieval_lab/test__final_v3 \
    --baseline bm25_full_corpus
../.venv/bin/python evaluation/make_retrieval_figures.py ../outputs/retrieval_lab/test__final_v3 \
    --baseline bm25_full_corpus --highlight hybrid_bm25f_e5_meta
../.venv/bin/python -m pytest tests/ -q
```

See `docs/retrieval_lab.md` for single-candidate runs, runtime switching and the list of
experiments that were attempted but not completed.
