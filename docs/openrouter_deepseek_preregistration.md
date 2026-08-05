# OpenRouter DeepSeek V4 Pro provider evaluation preregistration

Status: **preregistered protocol v2; default-switch verdict currently BLOCKED**.

Version 2 is a pre-final-seal audit amendment. Version 1 ambiguously called the app-level cutoff
the candidate depth and incorrectly said reranking was enabled. No final dataset or candidate final
outputs existed. V2 names and freezes the actual retrieval stages and adds an executable
conformance gate; this correction is not a post-final outcome adjustment.

This protocol is frozen against base commit
`5a6949c1e2e5b0603d8c1c88820f590099000f34`. The current default baseline is Groq
`llama-3.3-70b-versatile`; the sole candidate is OpenRouter
`deepseek/deepseek-v4-pro`. AWS deployment is outside this experiment and remains deferred.

## Experimental design

Every scored item/repetition is sent once to each provider with the same application prompt, retrieved
evidence, deterministic profile state, output-token ceiling, temperature, and reasoning setting.
Provider call order is randomized independently per item with seed `20260804` and recorded as
`call_order`. Two warmups per provider are excluded from every quality, latency, token, and cost
metric. A warmup authentication/configuration failure aborts scored execution and blocks any model
decision.

The frozen confirmatory generation configuration is temperature `0`, maximum output `1024`
tokens, three repetitions per item, request timeout `60` seconds, at most two retries, no fallback,
and prompt strategy `structured_lookup`. The application prompt template SHA-256 is
`29bae5340a76c1291005477132a55053d177a27ee8f58a462a7fbbc29d7028c3`; the strategy directive
SHA-256 is `15a27de0228dc32279acde9e6731d5ceb16a7aa54a4efdf0f28de47323ce9cef`.
The provider APIs do not expose a guaranteed seed, so nondeterminism is handled with stable paired
item IDs and the frozen repetition count rather than an invented provider seed.

Retrieval is frozen as `hybrid_meta`: BM25F + multilingual-E5-small, RRF `k=60`, metadata boost,
first-stage depth `200`, app/provider-context cutoff `20`, and final context top-k `6`. The optional
post-fusion reranker is explicitly off (`reranker_id=null`), matching the current default and its
latency decision. Contexts are materialized and aggregate-hashed before either provider receives a
scored item.

`run_preregistered_final_benchmark` may create a protocol-conformant research artifact. It loads the
sealed manifest and dataset itself; callers cannot inject items, a passing gate, a configuration
map, or a context hash. The runner recomputes dataset, exact materialized-provider-input, and
per-item context aggregate hashes, verifies the provider adapters' resolved runtime generation
settings, and atomically claims the manifest-named single-run ledger before provider calls. It
rejects any provider/model, seed, repetition, warmup, generation, retrieval-materialization,
reasoning, fallback, or hash mismatch.

That runner still accepts provider and scorer objects, so neither it nor the public
`default_switch_verdict` may authorize a production default change. The sole production boundary
is the argument-free `canonical_final_decision_status`, which reads the repository-fixed final
manifest and hashed gate artifacts. In this revision it always fails closed because the independent
final scorer/execution path, sealed data, external artifacts, and owner thresholds are absent. A
caller-supplied mapping or scorer cannot unlock the switch.

Every row also carries a `cluster_id`. Multiple paraphrases, checks based on the same source, or
variants of one profile/scenario must share a cluster. If no explicit cluster is supplied, the
harness conservatively defaults it to `item_id`. Prompts and answers are still excluded from run
artifacts.

Primary comparison has provider reasoning disabled. Reasoning-on experiments are development-only
ablations; their raw reasoning is never stored, and they cannot change the final configuration after
the final manifest is sealed. Both providers use the same temperature and output-token ceiling.

Provider exceptions, HTTP errors, timeouts, empty responses, missing rows, and scorer errors remain
in the planned denominator with zero quality scores. Stopping early never shrinks the denominator:
unattempted planned rows are recorded as `not_run_stop_rule` failures.

## Data boundary

The final untouched test set must be independently authored and sealed before its first provider
call. It must contain at least 100 Turkish and 100 English items. Within **each** language it must
contain at least:

| Category | Minimum |
|---|---:|
| Quality/grounded advising | 60 |
| Safety | 15 |
| Adversarial/prompt injection | 15 |
| Planner critical constraints | 10 |

The sealed manifest must name one direct JSONL path and carry its SHA-256; assembling final data via
development `source_mappings` is forbidden. After the final run begins, no prompt, rubric,
threshold, retry policy, provider configuration, or analysis rule may change. The final set is run
once. It is prohibited to generate or paraphrase final items after observing development, Groq, or
DeepSeek outputs.

The repository currently has no such set. `data/benchmark/provider_eval/final_manifest.json`
therefore intentionally fails the gate. The 24-TR/24-EN development mapping is derived from the
existing committed retrieval dev split and may be used only for integration and scorer calibration;
it has no safety/adversarial/planner coverage and cannot authorize a default switch.

## Metrics

The primary metric is binary `task_success`: an item passes only when every requirement in its
frozen scoring rubric passes. Errors and missing outputs are failures.

Secondary metrics are answer correctness, groundedness, citation support, language match, and safe
behavior. Operational metrics are API error rate, end-to-end p50/p95 latency, native-stream TTFT,
provider-reported prompt/completion/total tokens, and provider-reported cost. Token and cost values
are never estimated when the API omits them; they remain null and the completeness gate fails.

The current `/ask/stream` endpoint replays a completed answer word by word. That is fake streaming,
not provider TTFT. TTFT must be reported as unavailable unless measured from a native provider
stream's first content event.

## Statistical analysis

- First reduce rows sharing a `cluster_id` to paired provider means. Compute the
  candidate-minus-baseline delta and a 95% paired **cluster-bootstrap** interval with 10,000
  resamples and seed `20260804`; clusters, not rows, are the independent resampling units.
- Run a two-sided paired **cluster sign-flip randomization test** with 10,000 draws and fixed seed
  `20260804` for the primary endpoint overall, Turkish, and English.
- Apply Holm correction across those three clustered primary p-values at family-wise
  `alpha=0.05`. This clustered randomization result, together with the clustered bootstrap, is the
  inferential acceptance test.
- Report exact item-level McNemar tests for binary metrics as secondary diagnostics only. They are
  not used to authorize a switch because related rows may not be independent.
- Report all secondary metrics with paired intervals, but do not use post-hoc secondary wins to
  replace a failed primary endpoint.
- Report overall, TR, EN, and category strata. Do not hide a subgroup regression inside an overall
  mean.

## Vetoes and acceptance

Safety, adversarial, and planner are veto strata. Any observed candidate failure in one of these
frozen critical rubrics rejects the candidate; all three strata require 100% candidate success. A
missing veto stratum blocks the decision rather than being treated as a pass.

DeepSeek becomes only **eligible** for a default switch when all of these hold:

1. The sealed final data gate passes.
2. The primary cluster-bootstrap 95% CI lower bound is greater than zero.
3. The overall cluster sign-flip result remains significant after Holm correction.
4. Neither Turkish nor English has a statistically significant regression.
5. Candidate success is 100% in safety, adversarial, and planner veto strata.
6. API-reported token and cost fields are complete.
7. The candidate satisfies product-owner thresholds for p95 latency, per-query provider-reported
   cost, and API error-rate non-inferiority.
8. Turkish and English response-language accuracy are each at least 99%; groundedness, citation
   support, Hit@10, AllGold@10, security, and every reported subgroup do not regress.
9. The separate end-to-end, confidence-calibration, and independent blind-evaluation gates pass.
10. If reasoning is enabled, its frozen quality benefit justifies both its latency and cost impact.

Failure to prove a primary gain means keep Groq. Missing final data or an infrastructure-aborted run
means `BLOCKED`, not a tie and not a DeepSeek loss.

The repository defines **no** product p95 latency limit, per-query cost limit, or API error-rate
non-inferiority margin. Those three owner decisions are recorded as unresolved in the machine
protocol. Consequently, even a clean quality win remains `BLOCKED` until explicit values are
provided; the evaluation code does not invent them.

## Stop rules

Stop after three consecutive failures for either provider, after a provider exceeds a 25% failure
rate once 20 paired items have been attempted. An operator may also supply an explicit total-cost
safety fuse before a run, but it has no default and is not a product acceptance threshold. Remaining
planned rows stay in the denominator as failures. Rate limits may pause and resume only if the
pause/retry policy was fixed before the run; configuration and authentication failures do not
trigger adaptive retries.

## Artifact privacy

Run artifacts must never contain prompts, system messages, retrieved context, response text, raw
provider exceptions, API keys, or `reasoning_details`. They contain item IDs, provider/model IDs,
scores, safe exception class, call order, timings, exact API usage/cost, and a response SHA-256 only.
Reasoning must not be persisted or shown to scorers.
