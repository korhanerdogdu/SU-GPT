# adviSU — Experiment Log

Every number below was produced by running the code in this repository. Nothing is estimated,
extrapolated or filled in by hand. Where a measurement is missing or partial, it says so.

## Configuration

| Component | Value |
|---|---|
| LLM | `llama-3.3-70b-versatile` via Groq |
| Embeddings | `sentence-transformers/all-MiniLM-L12-v2` |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Vector store | ChromaDB, collection `su_knowledge`, 30,343 vectors |
| Retrieval modes | `llm_only`, `bm25`, `dense`, `hybrid`, `hybrid_rerank` |
| Prompt strategies | `basic` only (Section 4 not implemented) |
| Candidate k | 20 |
| Context top-k | 6 |
| Chunking | corpus is **pre-chunked** (1 JSONL row = 1 chunk); `CHUNK_SIZE`/`CHUNK_OVERLAP` do not apply |
| Benchmark | 22 questions (16 answerable, 6 unanswerable/misleading) |
| Evaluation date | 2026-07-22 |

Reproduce with:

```bash
npm run benchmark:build     # regenerate the benchmark from the corpus
npm run eval:retrieval      # retrieval metrics, no LLM cost
npm run eval                # + generated answers (needs Groq quota, see caveat below)
npm run eval:ablate         # ablation grid
npm run eval:tables         # report-ready CSVs
```

## Benchmark construction

`server/evaluation/build_benchmark.py` derives every answerable question from a real row in
`data/degree_requirements/**` or `data/minors/**`. The gold `chunk_id` is that row's actual id and
the reference answer is built from its own fields, so the benchmark cannot drift from the corpus.
The builder refuses to emit a question whose gold chunk id is not present in the corpus (verified
against all 30,343 chunks at build time).

**Known bias:** question wording is templated and shares vocabulary with the chunk text. This
flatters lexical retrieval, so the BM25 and hybrid numbers below should be read as an *optimistic
bound*, not as performance on free student prose.

## Result 1 — Retrieval quality (chunk-level, 16 answerable questions)

From `outputs/tables/retrieval_metrics.csv`. Identical across three independent runs.

| Mode | Recall@1 | Recall@6 | MRR@6 | nDCG@6 | Recall@10 |
|---|---|---|---|---|---|
| `hybrid` | **0.4375** | **0.6250** | **0.5312** | **0.5558** | **0.6250** |
| `hybrid_rerank` | 0.3125 | 0.4375 | 0.3385 | 0.3617 | 0.4375 |
| `bm25` | 0.1875 | 0.3750 | 0.2604 | 0.2894 | 0.3750 |
| `dense` | 0.0625 | 0.1875 | 0.1094 | 0.1289 | 0.1875 |
| `llm_only` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

`llm_only` retrieves nothing by definition; the row is the control, not a result.

**Source-level metrics are also emitted but are not meaningful here.** A whole program × term lives
in one JSONL file, so once retrieval is scoped by program/term every hit shares the expected source
and source-level scores sit at ceiling. Quote the chunk-level numbers.

### Finding 1a: the CrossEncoder reranker *hurts* retrieval on this corpus

This was the opposite of what we expected, and it reproduces at every cutoff:

| top-k | `hybrid` Recall | `hybrid_rerank` Recall |
|---|---|---|
| 3 | 0.625 | 0.312 |
| 5 | 0.625 | 0.375 |
| 10 | 0.750 | 0.562 |

Both modes see the **same 20 candidates**; only the ordering differs. Tracing individual gold
chunks through the reranker shows it demoting them out of the context window:

| Question | rank before rerank | rank after rerank |
|---|---|---|
| q009 core-elective credits | 2 | 20 |
| q008 area-elective credits | 7 | 20 |
| q016 IE total credits | 1 | 14 |
| q007 total ECTS | 1 | 8 |
| q011 required-course credits | 1 | 7 |

The failure table quantifies the same thing: `hybrid` has 4 retrieval misses and 2 ranking losses,
`hybrid_rerank` has 4 misses and **5** ranking losses.

**Interpretation.** The gold chunks for these questions are `degree_requirement_category_pool` rows,
whose text is a long enumeration of course names with the credit minimum in one clause.
`ms-marco-MiniLM` is trained on short web passages and scores those long list-like chunks poorly
against a short question, preferring compact `pool_course` rows that mention similar words. The
structured-metadata bonus in `catalog_retriever` does not compensate, because these questions carry
no course code to match on. This is a corpus/reranker mismatch, not a bug in the ablation harness —
the harness was checked by tracing the ranks above.

**Not yet acted on.** Changing the reranker is a product decision that belongs to a later section;
Section 5/6 measure and report. Candidate fixes to evaluate next: a domain-appropriate reranker, or
splitting category-pool chunks so the requirement clause is its own chunk.

### Finding 1b: cross-lingual retrieval gap

q013 is the Turkish rendering of q006 with the *same* gold chunk. English retrieves it (rank 8);
Turkish does not retrieve it at all. The corpus text is English while the product answers in
Turkish, so Turkish questions rely entirely on the multilingual capacity of a MiniLM embedding.

## Result 2 — Answers: retrieval vs no retrieval

From `outputs/tables/answer_metrics.csv`.

**Metric definition.** `reference_number_match_rate` = fraction of scored answerable questions where
*every* ground-truth number (course codes stripped) appears in the answer. It is an objective,
reproducible signal — **not** a full correctness judgement. `answer_correctness`, `faithfulness`,
`citation_correctness` and `hallucination_rate` are deliberately left empty for manual labelling.

| Mode | Coverage | Answers scored | Reference-number match | Refusal rate on unanswerable |
|---|---|---|---|---|
| `bm25` | 100% | 16 | **0.933** (14/15) | **0.667** |
| `llm_only` | 100% | 16 | 0.267 (4/15) | **0.000** |
| `dense` | 18% | 4 | 1.000 | — |
| `hybrid_rerank` | 5% | 1 | 1.000 | — |
| `hybrid` | 0% | 0 | — | — |

**This is the headline hallucination result.** With retrieval, the system reproduces the correct
official number in 93% of answers and declines 67% of the questions it cannot support. Without
retrieval, the same model gets 27% right and **refuses nothing at all** — it answered every
unanswerable and every false-premise question with a confident fabrication. `llm_only`'s failure
profile is 8 hallucinations, 6 unsupported answers on unanswerable questions, and 4 refusals.

### Caveat — incomplete answer coverage (infrastructure, not a result)

The Groq free tier caps usage at **100,000 tokens/day**. The full sweep is 5 modes × 22 questions =
110 calls, and the shipped advisor prompt is large, so the daily budget was exhausted partway
through: `dense` completed 18%, `hybrid_rerank` 5%, `hybrid` 0%, all with HTTP 429.

The runner records `generation_error` per row and **excludes failed calls from every answer
statistic** rather than scoring them as wrong answers — an earlier version of the aggregator did
score them as wrong, which made `hybrid` look like it produced 0% correct answers when it had in
fact produced none at all. Retrieval metrics are unaffected (they need no LLM).

**To complete the table:** run one or two modes per day, e.g.
`npm run eval -- --modes hybrid_rerank llm_only` (44 calls), or move to a paid tier.

## Result 3 — Efficiency

From `outputs/tables/efficiency_metrics.csv`, mean per query over 22 questions, warm models.

| Mode | Retrieval ms | Rerank ms | Total ms | Context chunks |
|---|---|---|---|---|
| `dense` | 100.7 | 0.0 | 100.7 | 6 |
| `bm25` | 296.6 | 0.0 | 296.7 | 6 |
| `hybrid` | 439.1 | 0.0 | 439.2 | 6 |
| `hybrid_rerank` | 427.2 | 295.4 | 722.8 | 6 |
| `llm_only` | 0.0 | 0.0 | 0.0 | 0 |

Generation dominates end-to-end latency: ~7.6–8.2 s per answered question against Groq, versus
under 0.8 s for the entire retrieval stack. First call in a process additionally pays a one-off
CrossEncoder load (~5.7 s), excluded from the warm means above.

`estimated_cost_usd_per_query` is intentionally **blank**: no price-per-token is hard-coded, because
a plausible-looking wrong cost is worse than an empty cell. Set `COST_PER_1K_INPUT` /
`COST_PER_1K_OUTPUT` in `make_tables.py` to populate it.

Combined with Result 1, `hybrid` is the better configuration on this benchmark on **both** axes: it
retrieves better than `hybrid_rerank` *and* costs 283 ms less per query.

## Result 4 — Ablations

From `outputs/tables/ablation_summary.csv`, 12 cells (4 modes × 3 top-k), retrieval-only.

Recall rises with top-k for every mode, and the mode ordering
(`hybrid` > `bm25` > `hybrid_rerank` > `dense`) is stable at every cutoff — the reranker penalty is
not an artefact of one context-window size.

**No chunk-size / chunk-overlap sweep.** CLAUDE.md §6.2 asks for one, but the advising corpus is
pre-chunked offline: one JSONL row is one retrieval chunk with its own stable id.
`CHUNK_SIZE`/`CHUNK_OVERLAP` only affect the uploaded-document path, so sweeping them against this
benchmark would produce identical numbers and a misleading "no effect" conclusion. Sweeping them
honestly requires re-ingesting a differently-chunked corpus, which also invalidates every
ground-truth chunk id. Recorded as a limitation rather than faked.

## What is NOT measured

- **The per-student layer.** MongoDB course history and the deterministic degree audit are not
  exercised by this benchmark. The audit is deterministic code, covered by the 18 invariant tests in
  `server/tests/test_advising.py`.
- **Prompt-strategy comparison** (basic vs few-shot vs expert-routed) — Section 4 is not built.
- **RAGAS metrics.** RAGAS is optional and not installed; the runner records
  `ragas_available: false` and skips it.
- **Human answer labels.** Every manual column is empty. Do not quote answer quality beyond the
  reference-number signal until they are filled in.

## Run provenance

| Artefact | Run |
|---|---|
| Retrieval, efficiency, failures | `outputs/evaluation_runs/20260722T140613Z_retrieval_authoritative` |
| Answers | `outputs/evaluation_runs/20260722T134612Z_full` (partial, see caveat) |
| Ablations | `outputs/evaluation_runs/ablations.jsonl` |

`outputs/tables/run_provenance.json` records the exact configuration of the run behind the tables.
