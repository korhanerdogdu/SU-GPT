# Provider-evaluation data boundary

This directory contains protocol metadata, not a fabricated final benchmark.

- `development_manifest.json` maps 48 already-committed rows from
  `data/benchmark/retrieval_dev.jsonl` (24 Turkish, 24 English). It is useful only for wiring,
  scorer calibration, and dry runs. It contains quality/retrieval questions only.
- `final_manifest.json` intentionally points to no dataset and declares the default-switch verdict
  `BLOCKED`. A final claim requires an independently authored and sealed JSONL with at least 100
  Turkish and 100 English items, including all veto categories.
- `item_schema.json` is the schema for that future final JSONL.

A sealed final row must additionally contain the exact pre-provider `provider_prompt`, its
canonical `frozen_context`, and the recomputed `context_sha256`. The manifest binds the dataset,
all materialized prompts, the context aggregate, the exact preregistered materialization settings,
and a repository-bounded one-time run-ledger path. The final runner loads and verifies these itself;
callers cannot inject a passing gate/configuration or substitute an in-memory item list.

Inference is cluster-aware. Related paraphrases, rows derived from the same source, or variants of
one student-profile scenario must share `cluster_id`; a row without one defaults to its `item_id`.
Run artifacts persist only those identifiers and redacted measurements, never prompts or answers.

The repository also has no product-owner p95 latency limit, per-query cost limit, or API error-rate
non-inferiority margin. Those unresolved thresholds independently keep a default-switch decision
blocked, even after a quality win. A separately configured total-cost cap is only an operator safety
fuse for a run, not an acceptance budget.

Do not generate a “final” set by paraphrasing development items after examining Groq or DeepSeek
outputs. The final file must be authored and frozen before the first scored final call, hashed in its
manifest, and run once.
