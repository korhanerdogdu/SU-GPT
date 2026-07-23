# adviSU — AI Usage / Prompt Log

CS 455 follows a transparent AI-usage policy, so this file records where generative AI was used in
building the project, what came out, and what we did to verify it.

**How to read this:** "Output used" describes what was kept. "Verified by" is the concrete check we
ran — a passing test, a real command output, or a code read. Nothing in this project is accepted on
the model's say-so; every number in `docs/experiment_log.md` comes from an executed run.

| Date | Section | Tool | Prompt summary | Output used | Verified by |
|---|---|---|---|---|---|
| 2026-07-22 | §3 Retrieval modes | Claude Code (Opus 4.8) | "Implement Section 3 — selectable retrieval modes" | `modules/retrieval_modes.py`, `require_narrowing` flag on BM25, `answer_without_context()`, `/ask/` mode params, header mode selector | Ran all 5 modes against the real 30,343-vector index: every RAG mode returned CS-only docs under a CS/202401 filter, `top_k` honoured. 4 new invariant tests added; **18 unit tests, 0 failed**. `tsc --noEmit` clean. |
| 2026-07-22 | §3 | Claude Code | Challenged the assistant's claim that BM25/hybrid were unimplemented | Correction: BM25 + hybrid fusion already existed; only the *mode switch* and `llm_only` were missing. Section 3 rescoped and the deviation recorded in CLAUDE.md | Read `catalog_retriever.py:103-112` and `bm25_retriever.py` directly |
| 2026-07-22 | Branding | Claude Code | "Use `big.png` as the logo, large on login and chat" | Trimmed lockup generated with PIL, wired into login/signup hero, chat header and empty state; old `sugptlogo.png` / `sabanci_logo.png` deleted | Measured the source PNG: only **30%** of its 2061×1152 canvas was non-transparent, so a raw swap would have rendered at half size. `tsc --noEmit` clean |
| 2026-07-22 | §5 Benchmark | Claude Code | "Build the benchmark without fabricating answers" | `evaluation/build_benchmark.py` derives every answerable question from a real corpus row; gold `chunk_id`s and reference answers are read out of the data | Builder verifies each gold id against all 30,343 chunks and exits non-zero on a miss. 22 questions generated |
| 2026-07-22 | §5 Metrics | Claude Code | "Recall@k, MRR@k, nDCG@k, chunk and source level" | `evaluation/metrics.py` | Undefined cases return NaN and are skipped by `mean_ignoring_nan`, so an unmeasurable metric can never be reported as 0.0 |
| 2026-07-22 | §5 Runner | Claude Code | "Run every question × every mode, save the Section 5.3 artefacts" | `evaluation/run_evaluation.py` | Retrieval metrics identical across **three** independent runs |
| 2026-07-22 | §5 | Claude Code | Investigated a surprising result: `hybrid` beat `hybrid_rerank` | Confirmed genuine, not a harness bug — traced individual gold chunks and found the reranker demoting them (rank 2→20, 7→20, 1→14) | Standalone diagnostic script comparing pre- and post-rerank ranks per question |
| 2026-07-22 | §5 | Claude Code | Investigated a second oddity: `hybrid` showed 0.000 answer accuracy despite the best retrieval | **Found a bug in our own aggregator.** Groq's 100k tokens/day cap returned HTTP 429; empty answers were being scored as *wrong* instead of *not measured*. Fixed to exclude failed calls and report `generation_coverage` | Re-summarised the same run: `hybrid` now correctly shows `n/a` at 0% coverage instead of a fabricated 0.000 |
| 2026-07-22 | §6 Ablations | Claude Code | "Ablation grid + failure analysis + report tables" | `ablation_configs.yaml`, `ablation_runner.py`, `make_tables.py` | 12 cells run; mode ordering stable at every top-k |
| 2026-07-22 | §6 | Claude Code | Reviewed the first failure table | Caught two bugs in our own table generator: rows double-counted when merging two runs (bm25 showed 20 misses instead of 10), and `llm_only` was charged with 32 "retrieval misses" despite having no retriever | Compared counts against the single-run output; regenerated |
| 2026-07-22 | §6 | Claude Code | Asked for chunk-size ablation per §6.2 | **Declined and documented why:** the advising corpus is pre-chunked, so `CHUNK_SIZE` has no effect on it and a sweep would produce a misleading "no effect" result | Inspected the corpus: 1 JSONL row = 1 chunk with its own stable id |
| 2026-07-22 | §7 Docs | Claude Code | "README, prompt log, experiment log, demo script" | `docs/experiment_log.md`, `docs/demo_script.md`, this file, README | Every figure cross-checked against the generated CSVs in `outputs/tables/` |

## Earlier work

Sections 1–2, the React migration, the advising data pipeline, the deterministic degree audit and
the ChromaDB/MongoDB backbone were built in earlier sessions, also with Claude Code assistance. The
history is recorded in `CLAUDE.md`. **Team: fill in the earlier rows** — this table should cover the
whole project, not only the final sessions.

## Standing rules we applied

1. **No fabricated results.** Benchmark ground truth is derived from the corpus; every reported
   number comes from an executed run. Where a run was incomplete, the table says so.
2. **Surprising results get verified before they get reported.** Both the reranker finding and the
   answer-coverage anomaly were traced to root cause before being written down — one turned out to
   be a real property of the system, the other a bug in our own code.
3. **Empty beats wrong.** Manual-label columns and the cost column are left blank rather than filled
   with plausible guesses.
