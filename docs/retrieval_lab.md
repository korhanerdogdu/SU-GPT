# Retrieval lab — how to reproduce every number

The retrieval experiments live in `server/retrieval_lab/` (the retrievers) and
`server/evaluation/` (the benchmark, the runner, the statistics, the figures). The same
`retrieval_lab` code serves production through `server/modules/lab_retriever.py`, so the
benchmarked pipeline and the shipped pipeline cannot drift apart.

## 0. Environment

The project is developed on Apple silicon; `torch` picks MPS automatically and falls back to
CPU. Do **not** place the checkout inside an iCloud-synced folder (`~/Desktop`,
`~/Documents`) — macOS evicts file contents under disk pressure and both git and the corpus
start reading as empty files.

```bash
python3 -m venv .venv
./.venv/bin/pip install -r server/requirements.txt
./.venv/bin/pip install pytest matplotlib rank-bm25
```

## 1. Build the benchmark splits

Regenerates `data/benchmark/retrieval_{train,dev,test}.jsonl` from the corpus. Gold chunk ids
are taken from real corpus rows and re-verified; the build fails loudly on an unknown id or on
any leakage between splits.

```bash
cd server && ../.venv/bin/python evaluation/build_retrieval_benchmark.py
```

## 2. Run the benchmark

`--methods` accepts `sparse`, `all`, or a comma-separated list. Use `--limit` for a smoke test
before committing to a full sweep. Dense models download on first use and cache their corpus
embeddings under `outputs/retrieval_cache/`, keyed by the corpus fingerprint — editing the
corpus invalidates them automatically.

```bash
# sparse only (no model downloads, ~3 min)
cd server && ../.venv/bin/python evaluation/run_retrieval_lab.py \
    --split dev --methods sparse --out-tag sparse_dev --no-dense --no-rerank

# a single candidate
cd server && ../.venv/bin/python evaluation/run_retrieval_lab.py \
    --split dev --methods metadata_bm25f --out-tag one_method --no-dense --no-rerank

# everything, including dense + hybrid + reranking
cd server && ../.venv/bin/python evaluation/run_retrieval_lab.py \
    --split dev --methods all --out-tag full_dev
```

## 3. Statistics and tables

Writes `retrieval_metrics.csv/json`, `per_query_results.csv`, `subgroup_metrics.csv`,
`latency_metrics.csv` and `significance_results.json` (paired bootstrap + permutation test)
into the run directory.

```bash
cd server && ../.venv/bin/python evaluation/analyze_retrieval_lab.py \
    ../outputs/retrieval_lab/dev__sparse_dev --baseline bm25_full_corpus
```

## 4. Figures

```bash
cd server && ../.venv/bin/python evaluation/make_retrieval_figures.py \
    ../outputs/retrieval_lab/<run_dir> --baseline bm25_full_corpus --highlight metadata_bm25f
```

## 5. Tests

```bash
cd server && ../.venv/bin/python -m pytest tests/test_retrieval_lab.py -q
```

## 6. Selecting the retriever at runtime

The active retriever is configuration-controlled. `bm25f_meta` is the benchmarked winner;
every previous mode still exists and BM25 remains available as a fallback.

```bash
# run adviSU with the winning retriever
DEFAULT_RETRIEVAL_MODE=bm25f_meta ./.venv/bin/python -m uvicorn main:app --app-dir server

# revert to the previous production default
DEFAULT_RETRIEVAL_MODE=hybrid ./.venv/bin/python -m uvicorn main:app --app-dir server

# force the plain BM25 baseline
DEFAULT_RETRIEVAL_MODE=bm25 ./.venv/bin/python -m uvicorn main:app --app-dir server
```

If `bm25f_meta` cannot build its index, the request is served by the Chroma hybrid path and
the downgrade is logged at ERROR — it is never silent.

## Experiments attempted but not completed

Recorded rather than silently dropped, per the experiment protocol.

| Experiment | Status | Why |
|------------|--------|-----|
| `dense_e5_base` (`intfloat/multilingual-e5-base`) | **not completed** | Model downloaded (1.1 GB) but corpus encoding ran >34 min on MPS without finishing, against ~4 min for `e5_small`. Terminated so it would stop contending for the GPU and corrupting the latency measurements of the scored run. |
| `dense_e5_large_instruct` (`intfloat/multilingual-e5-large-instruct`) | **not run** | ~2.2 GB download plus an encode pass strictly slower than `e5_base`, which had already failed to complete. Implemented and registered in `retrieval_lab/dense.py`; runnable with `--methods dense_e5_large_instruct`. |
| `dense_bge_m3` (`BAAI/bge-m3`) | **not run** | Same reason: ~2.2 GB, 1024-dim, 512-token sequences. Implemented and registered; runnable on a machine with more headroom. |
| `bge_m3_sparse` (learned sparse) | **not run** | Requires the `FlagEmbedding` package, which is not in `server/requirements.txt`; adding a heavyweight dependency was not justified once dense E5 had been measured far below the lexical candidates. |
| Wave-3 methods (SPLADE, late interaction, domain fine-tuning) | **not run** | The protocol only escalates to Wave 3 if earlier waves fail to beat BM25. Wave 1 produced a large, significant improvement, so escalation was not triggered. |

The disk budget on the development machine was ~13 GiB free, which is the binding constraint
on the large dense models. `dense_minilm` and `dense_e5_small` **were** run to completion, and
both scored far below the field-weighted lexical candidates (see the report), which is the
evidence behind not spending the remaining budget on larger models of the same family.

## Method names

| Name | What it is |
|------|------------|
| `bm25_original` | shipped BM25 exactly as deployed, including the 3000-doc subset cap |
| `bm25_full_corpus` | same scoring over the whole corpus — the fair baseline |
| `bm25_v2tok` | BM25 with the Turkish-safe, course-code-aware tokenizer |
| `contextual_bm25` | BM25 over a metadata-prefixed view of each chunk |
| `bm25f` | field-weighted BM25F (course code / title / program / term / category / body) |
| `metadata_bm25f` | BM25F + soft metadata & record-type boosts |
| `metadata_filter_bm25f` | BM25F + **hard** metadata filter (ablation: shows why soft wins) |
| `dense_*` | MiniLM / multilingual-E5 / BGE-M3 dense retrieval |
| `hybrid_bm25f_*` | BM25F + dense fused with Reciprocal Rank Fusion |
| `hybrid_meta_rerank{20,50,100}` | hybrid + cross-encoder rerank at that candidate depth |
