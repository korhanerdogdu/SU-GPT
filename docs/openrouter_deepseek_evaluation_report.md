# OpenRouter DeepSeek V4 Pro integration and evaluation report

**Report date:** 2026-08-04
**Decision status:** `BLOCKED` for a DeepSeek default switch
**Production default:** Groq (`LLM_PROVIDER=groq`)
**Scope:** local implementation and evaluation only; no deployment

## 1. Executive outcome

The exact OpenRouter model ID `deepseek/deepseek-v4-pro` is integrated behind a reversible,
configuration-selected provider contract. Groq remains the default. The integration includes
exact-model preflight, bounded retry/backoff, one-hop opt-in fallback, a process-wide circuit
breaker, one shared request deadline, safe streaming, reasoning removal, usage telemetry, and
redacted typed errors.

No confirmatory provider decision is possible. The repository has no independently authored,
sealed final set with at least 100 untouched Turkish and 100 untouched English examples, no
independent blind final judge, and no owner-approved p95 latency, per-query cost, or API-error
non-inferiority thresholds. The final-manifest validator therefore returns `BLOCKED` by design.
The public statistical verdict and caller-injected benchmark runner also cannot authorize a
production switch; the argument-free canonical decision path is code-bound and deliberately has no
success branch until independent final evidence is provisioned by a reviewed change.

The only live candidate evidence is a rejected exploratory development pilot. Its least-bad
DeepSeek configuration scored 81.25% task success against 87.50% for the Groq development
baseline, with a paired clustered 95% CI of [-12.50, 0.00] percentage points. The experiment was
confounded by concurrent candidate calls and is not final evidence. A later serial run failed at
the provider boundary on all 48 planned candidate rows and is classified as an infrastructure-
aborted run, not a model-quality result.

## 2. Final model choice

Groq remains the configured default. OpenRouter/DeepSeek is retained only as an optional
experimental provider because the abstraction is small, reversible, covered by contract tests,
and does not expose provider reasoning. Fallback is disabled unless explicitly configured.

## 3. Is either configuration proven better?

No. The required final evidence does not exist, and neither model may be described as superior.
The development pilot does not support switching to DeepSeek, but it also cannot establish a
general model ranking because it is small, lookup-only, operationally confounded, and lacks the
mandatory planner, safety, adversarial, end-to-end, and blinded-judging strata.

## 4. Starting branch and commit

- Requested base branch: `advisu-v2-dev`
- Frozen starting commit: `5a6949c1e2e5b0603d8c1c88820f590099000f34`
- The local working tree began clean for task-owned files.
- Existing retrieval and reranking reports predate this work and are treated as historical
  evidence, not newly rerun provider results.

## 5. Feature branch

`feat/openrouter-deepseek-evaluation`

The implemented request path is: React client (`frontend/src/lib/api.ts`) → authenticated `/ask`
route (`server/main.py`) → deterministic input/language/resource controls → intent and retrieval or
deterministic planner → provider-neutral chain (`server/modules/llm.py`) → exact adapter
(`server/modules/llm_providers.py`) → output/citation validation → owner-qualified bounded memory.
Provider selection does not bypass planner, authorization, retrieval trust, or output gates.

## 6–8. Changed-file inventory, change, and necessity

| File | What changed and why it was necessary |
|---|---|
| `.env.example` | Added placeholder-only provider, reliability, auth, quota, retention, and production-mode settings; keeps real credentials out of Git. |
| `.gitignore` | Keeps raw/local provider dumps ignored while narrowly allowing two immutable run manifests, completion markers, and response-free redacted rows needed to reproduce the rejected pilot. |
| `frontend/package-lock.json` | Applied non-breaking dependency security updates via `npm audit fix`. |
| `frontend/src/App.tsx` | Registered the public course-review policy page. |
| `frontend/src/main.tsx` | Mounts the application-wide locale provider. |
| `frontend/src/components/LanguageToggle.tsx` | Adds the persistent, accessible TR/EN switch shared by every user-facing surface. |
| `frontend/src/components/ThemeToggle.tsx` | Localizes its accessible label. |
| `frontend/src/components/chat/ChatHeader.tsx` | Adds language switching and localizes chat actions. |
| `frontend/src/components/chat/ChatInput.tsx` | Localizes the input, stop, send, and quota affordances. |
| `frontend/src/components/chat/ChatMessages.tsx` | Localizes assistant/error/copy labels without translating model or schema data. |
| `frontend/src/components/chat/EmptyState.tsx` | Sources starter prompts and headings from the active locale. |
| `frontend/src/components/chat/Sidebar.tsx` | Localizes navigation, conversation actions, account roles, and logout. |
| `frontend/src/contexts/AuthContext.tsx` | Stores and clears signed bearer-token state rather than treating a username as authorization. |
| `frontend/src/contexts/LocaleContext.tsx` | Centralizes active locale, persistence, document language, interpolation, and typed translation lookup. |
| `frontend/src/localization/resources.ts` | Defines compile-time key-parity TR/EN resources, sample questions, starters, and weekdays. |
| `frontend/src/lib/api.ts` | Attaches bearer tokens to private requests, handles the updated auth contract, and localizes safe HTTP failures. |
| `frontend/src/lib/export-xlsx.ts` | Localizes generated workbook labels and safely passes locale-aware meeting formatting. |
| `frontend/src/lib/sample-questions.ts` | Delegates examples to the centralized locale resource. |
| `frontend/src/lib/schedule.ts` | Makes day/meeting labels locale-aware while preserving canonical schedule data. |
| `frontend/src/pages/ChatPage.tsx` | Applies deterministic locale to requests and localizes page-level failures/actions. |
| `frontend/src/pages/CoursesPage.tsx` | Localizes the course view and exposes the global language switch. |
| `frontend/src/pages/LoginPage.tsx` | Uses token login, centralized TR/EN strings, the language switch, and the review-policy link. |
| `frontend/src/pages/ProfilePage.tsx` | Captures authoritative academic year (1–6) for planner stage selection and localizes profile/audit UI. |
| `frontend/src/pages/SchedulePage.tsx` | Localizes schedule construction/export UI and exposes the language switch. |
| `frontend/src/pages/SignupPage.tsx` | Disables unsupported public signup, removes the false account-creation path, and localizes the page. |
| `frontend/src/pages/CourseReviewPolicyPage.tsx` | Adds visible TR/EN policy disclosure for the disabled, course-only review feature. |
| `server/main.py` | Enforces auth/ownership, login and request limits, safe errors/uploads, evidence confidence, localized refusals, review/WhatsApp shutdown, legacy-review quarantine, deterministic planner year propagation, exact citation/output checks, and abstention before any incomplete-profile recommendation can reach an LLM. |
| `server/modules/config.py` | Adds provider/reliability/retention configuration and production startup refusal for weak secrets/default passwords. |
| `server/modules/conversation_memory.py` | Owner-qualifies all prompt-memory reads/writes, relies on the unique session index for atomic creation, fails closed on storage errors, and bounds retention. |
| `server/modules/course_planner.py` | Makes stage inference conservative, propagates academic year, fixes known CS prerequisites, supports alternative paths, and fails closed on unknown prerequisite data. |
| `server/modules/file_lifecycle.py` | Permanently disables private-chat/instructor-review ingestion and removes the old parser; retains deletion support for legacy data. |
| `server/modules/llm.py` | Replaces direct vendor coupling with a cached provider adapter, exact-model preflight, deterministic language selection, and non-disclosing prompt strategies. |
| `server/modules/load_vectorstore.py` | Adds file-count, per-file, aggregate-size, extension, MIME, path, and magic-byte validation and writes nothing until the whole batch passes. |
| `server/modules/mongodb.py` | Adds academic year storage and a conversation TTL index. |
| `server/modules/query_handlers.py` | Maps typed provider failures to sanitized, localized application responses. |
| `server/modules/rag_router.py` | Removes legacy private-review documents from the general retrieval pool. |
| `server/modules/retrieval_modes.py` | Applies safe retrieval boundaries and preserves authorized source metadata. |
| `server/modules/schedule_planner.py` | Preserves secondary components, reports placed SU and unavoidable credit shortfall instead of inserting unsafe courses. |
| `server/modules/source_indexer.py` | Permanently excludes review, WhatsApp, private-chat, and outside-root symlink sources from general retrieval indexing. |
| `server/requirements.txt` | Adds the direct HTTP client dependency used by the provider contract. |
| `server/test.py` | Keeps the legacy evaluator compatible with the provider-neutral runtime. |
| `server/modules/auth.py` | Implements signed, expiring bearer tokens and header redaction. |
| `server/modules/confidence.py` | Computes confidence only from observable evidence and hard-abstains on missing evidence/invalid planner/schema states. |
| `server/modules/course_reviews.py` | Adds the disabled-by-default five-dimension course-only domain service with consent, moderation, pseudonymous dedupe, PII/name filtering, and k≥10 suppression. |
| `server/modules/guardrails.py` | Adds layered input/retrieval/output validation, exact normalized citation-source membership, and safe refusal classification. |
| `server/modules/language.py` | Adds deterministic TR/EN/mixed language selection robust to course codes and code-heavy text. |
| `server/modules/llm_providers.py` | Adds Groq/Mistral/OpenRouter contract, exact-model verification, retry/jitter/Retry-After, circuit breaker, shared deadline, telemetry, fallback, safe stream sanitizer, reasoning removal, and safe resolved-runtime settings for final-protocol binding. |
| `server/modules/localization.py` | Centralizes new TR/EN error, quota, refusal, confidence, and review strings. |
| `server/modules/resource_controls.py` | Adds per-user/IP request controls, account+network login-attempt limits, concurrency, token, budget, input, and timeout controls with privacy-preserving hashes. |
| `server/evaluation/confidence_calibration.py` | Implements Brier, ECE, and coverage-risk calculations without inventing a deployment threshold. |
| `server/evaluation/language_benchmark.py` | Builds the deterministic 100 TR + 100 EN + 20 mixed detector benchmark. |
| `server/evaluation/provider_benchmark.py` | Adds manifest gating, stable IDs, three clustered repetitions, paired randomized execution, failure denominators, redacted results, and a final runner that internally loads sealed inputs, recomputes dataset/prompt/context hashes, binds provider runtime settings, and atomically claims a single-run ledger. |
| `server/evaluation/provider_stats.py` | Adds clustered 10k bootstrap/sign-flip analysis, Holm correction, subgroup/veto logic, and the complete fail-closed switch verdict. |
| `server/evaluation/recompute_development_pilot.py` | Recomputes statistics directly from the committed response-free JSONL evidence. |
| `server/evaluation/run_live_provider_pilot.py` | Adds exact-model preflight, immutable legacy comparison, serial candidate execution, 3-repetition protocol, stop rules, hashes, timestamps, environment metadata, and append-only checksums. |
| `server/evaluation/security_benchmark.py` | Evaluates the versioned adversarial corpus by application security surface. |
| `server/tests/conftest.py` | Makes repository-root pytest imports deterministic. |
| `server/tests/test_advising.py` | Adds planner stage/year/prerequisite/canonical-name/credit/schedule regression cases. |
| `server/tests/test_adversarial_dataset.py` | Validates balanced adversarial data and deterministic attack metrics. |
| `server/tests/test_confidence_calibration.py` | Verifies calibration formulas and no-threshold behavior. |
| `server/tests/test_frontend_localization.py` | Prevents locale-provider, resource-parity, or surface-level language-switch regression. |
| `server/tests/test_language_and_course_reviews.py` | Covers language, localization, review privacy, moderation, dedupe, and aggregation gates. |
| `server/tests/test_language_benchmark.py` | Enforces 100/100 TR, 100/100 EN, and 20/20 mixed detector results. |
| `server/tests/test_llm_providers.py` | Covers request shape, exact model, streaming splits, reasoning stripping, retry, fallback, deadline, circuit, telemetry, errors, and chain compatibility. |
| `server/tests/test_provider_evaluation.py` | Covers manifests, alignment, clustered statistics, Holm, missing failures, stop rules, 200-item materialized final-gate tampering, single-run ledger, and artifact checksums. |
| `server/tests/test_security_controls.py` | Covers tokens, cross-user denial, atomic fail-closed ownership, incomplete-profile recommendation abstention, exact citations, login brute-force limits, review quarantine, MIME/batch-safe uploads, production config, and secret scanning. |
| `data/benchmark/provider_eval/README.md` | Documents development/final data boundaries and unresolved owner gates. |
| `data/benchmark/provider_eval/development_manifest.json` | Freezes development IDs and mapping without calling them final. |
| `data/benchmark/provider_eval/final_manifest.json` | Represents the absent final set explicitly, freezes materialization configuration/ledger location, and forces `BLOCKED` until real dataset/prompt/context hashes exist. |
| `data/benchmark/provider_eval/item_schema.json` | Defines stable final-item fields, cluster provenance, exact materialized provider prompt, canonical frozen context, and its hash. |
| `data/benchmark/provider_eval/development_pilot_summary.json` | Commits only redacted aggregate pilot evidence and all protocol deviations. |
| `data/benchmark/security_adversarial_v1.jsonl` | Adds 36 balanced TR/EN cases over input, retrieval, upload, output, planner, and benign controls. |
| `docs/conversation_data_policy.md` | Documents conversation purpose, access, 90-day TTL, bounds, deletion, and limitations. |
| `docs/course_review_data_policy.md` | Documents consent, five allowed dimensions, moderation, anonymity, suppression, and forbidden instructor/private-chat uses. |
| `docs/openrouter_deepseek_preregistration.json` | Machine-readable primary metric, hashes, generation/retrieval settings, statistics, vetoes, and unresolved gates. |
| `docs/openrouter_deepseek_preregistration.md` | Human-readable preregistration and final-test contamination rules. |
| `docs/openrouter_deepseek_evaluation_report.md` | This decision record and reproducibility handoff. |
| `outputs/provider_evaluation/dev-pilot-20260804T120357Z/FINALIZED` | Marks the first exploratory run immutable and complete. |
| `outputs/provider_evaluation/dev-pilot-20260804T120357Z/run_manifest.json` | Records base/diff hashes, retrieval settings, selected IDs, repetitions, and privacy flags. |
| `outputs/provider_evaluation/dev-pilot-20260804T120357Z/artifact_checksums.json` | Cryptographically binds every committed redacted row file and run manifest. |
| `outputs/provider_evaluation/dev-pilot-20260804T120357Z/legacy_vs_adapter_groq.redacted.jsonl` | Stores response-free incumbent adapter parity rows. |
| `outputs/provider_evaluation/dev-pilot-20260804T120357Z/deepseek_basic.redacted.jsonl` | Stores response-free basic-candidate paired rows. |
| `outputs/provider_evaluation/dev-pilot-20260804T120357Z/deepseek_structured.redacted.jsonl` | Stores response-free structured-ablation paired rows. |
| `outputs/provider_evaluation/dev-pilot-20260804T120357Z/deepseek_few_shot.redacted.jsonl` | Stores response-free few-shot-ablation paired rows. |
| `outputs/provider_evaluation/dev-pilot-20260804T120357Z/deepseek_reasoning.redacted.jsonl` | Stores response-free reasoning-ablation paired rows; no reasoning content is present. |
| `outputs/provider_evaluation/dev-pilot-20260804T215913Z/FINALIZED` | Marks the serial rerun immutable and complete despite infrastructure abort. |
| `outputs/provider_evaluation/dev-pilot-20260804T215913Z/run_manifest.json` | Records the second run’s frozen settings and privacy flags. |
| `outputs/provider_evaluation/dev-pilot-20260804T215913Z/artifact_checksums.json` | Cryptographically binds every committed serial-run evidence file. |
| `outputs/provider_evaluation/dev-pilot-20260804T215913Z/legacy_vs_adapter_groq.redacted.jsonl` | Stores response-free serial incumbent rows. |
| `outputs/provider_evaluation/dev-pilot-20260804T215913Z/deepseek_basic.redacted.jsonl` | Preserves all 48 typed candidate failures instead of dropping them. |

## 9. Groq baseline table

There is no valid frozen end-to-end confirmatory Groq baseline because the required final set was
not provisioned. The table below is the honest exploratory retrieval-fixed development reference,
not a production baseline claim.

| Track | Items × repetitions | Task success | Errors | p50 | p95 | Usage | Cost |
|---|---:|---:|---:|---:|---:|---|---|
| Groq development reference | 16 × 3 = 48 | 87.50% | 0/48 | 1.625 s | 4.793 s | 32,808 input + 1,113 output = 33,921 tokens | Not reported; not assumed zero |

Historical retrieval evidence, unchanged by this task, remains:

| Retrieval mode (held-out n=499) | R@1 | R@5 | R@10 | MRR@10 | nDCG@10 | AllGold@10 |
|---|---:|---:|---:|---:|---:|---:|
| Full-corpus BM25 fair baseline | 0.3367 | 0.4810 | 0.5992 | 0.4029 | 0.4481 | 0.5972 |
| Current `hybrid_meta` | 0.7535 | 0.9158 | 0.9419 | 0.8223 | 0.8487 | 0.9319 |

The multi-evidence subset remains weak: current AllGold@10 is 1/16 (0.0625). Provider work does
not repair that retrieval limitation.

## 10–12. DeepSeek, before/after, and prompt-ablation results

| Candidate | Task success | Δ vs Groq | Clustered 95% CI | p (10k sign flip) | Error rate | p50 | p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Basic | 81.25% | -6.25 pp | [-12.50, 0.00] | 0.2478 | 10.42% | 4.172 s | 12.554 s |
| Structured | 62.50% | -25.00 pp | [-31.25, -16.67] | 0.0006 | 22.92% | 4.691 s | 47.745 s |
| Few-shot | 64.58% | -22.92 pp | [-33.33, -12.50] | 0.0043 | 25.00% | 4.083 s | 15.419 s |
| Limited reasoning | 70.83% | -16.67 pp | [-29.17, -6.25] | 0.0317 | 14.58% | 4.326 s | 11.292 s |

Architecture before/after:

| Before | After |
|---|---|
| Direct vendor construction in `llm.py` | One provider-neutral contract with Groq unchanged as default |
| Provider reasoning/error bodies could vary by SDK | Reasoning fields stripped; errors and telemetry have safe schemas |
| Request-local provider state | Cached process-wide circuit state and one total deadline |
| Username path implied access | Signed bearer token plus ownership checks |
| Planner inferred stage from sparse upper-level courses | Authoritative year when supplied; conservative fallback otherwise |
| Missing prerequisite map entry meant “none” | Missing prerequisite data blocks recommendation |

## 13. Reasoning off/on

Reasoning did not justify itself in the rejected pilot. Compared with DeepSeek basic, limited
reasoning reduced task success from 81.25% to 70.83%, increased mean reported tokens per successful
response from 798.4 to 823.8, and increased incomplete successful-call cost from $0.03564 to
$0.03812 despite fewer successful rows (43 vs 41). Because failures have missing usage, exact
total and per-query cost cannot be certified. Reasoning remains off by default.

## 14. Retrieval-fixed comparison

The exploratory pilot used one frozen `hybrid_meta` retrieval result per item, candidate depth 10,
and context top-k 6. This isolates generation better than a full application run, but the four
candidate treatments ran concurrently against one account, so provider rate limits/concurrency are
confounded with treatment. The result is retained as a failed experiment only.

## 15. End-to-end comparison

`BLOCKED`: no sealed final dataset, no frozen end-to-end Groq artifact on that dataset, no blind
judge, and no one-time confirmatory comparison. The deterministic application suite passed, but
that is not a substitute for provider end-to-end measurement.

## 16–19. Primary statistic, intervals, permutation, and effects

- Preregistered primary metric: paired binary `task_success`, failures retained as failures.
- Unit of uncertainty: highest shared `cluster_id`; 16 clusters and 48 repeated rows in the pilot.
- Resampling: 10,000 clustered bootstrap draws and 10,000 clustered sign-flip draws.
- Basic exploratory effect: -0.0625, 95% CI [-0.1250, 0.0000], p=0.2478.
- Basic McNemar diagnostic: baseline-only wins 3, candidate-only wins 0, p=0.25.
- The CI is not entirely above zero; the central decision rule cannot pass.
- Holm correction is implemented across the overall/TR/EN primary family. No final inference is
  made because this run is development-only and protocol-confounded.

An independent audit found that preregistration v1 described the retrieval depth ambiguously and
claimed reranking despite the frozen runtime having it disabled. Before any final dataset was
sealed or any final output existed, this was transparently amended to protocol v2. A later critic
then proved the first executable v2 gate still trusted caller-supplied settings and an arbitrary
64-character hash. That implementation was rejected and replaced before any final run.

The protocol runner accepts a manifest path—not caller-supplied items, gate results, settings, or
hashes. It loads the direct, content-hashed dataset; rejects final-set `source_mappings`; recomputes
the dataset file, exact materialized provider prompts, canonical per-item contexts, and aggregate
hashes; compares provider adapters' resolved runtime generation settings; requires the frozen
prompt/language/retrieval materialization; and atomically creates the manifest-named single-run
ledger before provider calls. A 100 TR + 100 EN regression fixture proves runtime tampering,
dataset tampering, and a selective second run are rejected. Because providers and the scorer remain
caller-supplied, this artifact is research-only. The argument-free canonical status function reads
only the fixed repository manifest and hashed artifacts and is the sole production decision
boundary; it is hard-blocked until a code-bound independent scorer/execution path lands.

## 20–21. Subgroups and Turkish/English

Exploratory basic configuration only:

| Language | Groq task success | DeepSeek task success | DeepSeek language-match (failures included) |
|---|---:|---:|---:|
| Turkish (24 repeated rows) | 75.00% | 70.83% | 87.50% |
| English (24 repeated rows) | 100.00% | 91.67% | 91.67% |

Separate deterministic language-detector benchmark:

| Subset | Passed | Accuracy |
|---|---:|---:|
| Turkish | 100/100 | 100% |
| English | 100/100 | 100% |
| Mixed dominant-language | 20/20 | 100% |

The detector benchmark is synthetic and proves only deterministic selection, not end-to-end model
response-language accuracy. Final TR/EN ≥99% response-language gates remain unevaluated.

## 22. Security results by attack category

The deterministic application-layer corpus passed 36/36: attack success 0%, safe-refusal
accuracy 100%, false-refusal 0%. Every category below contains one TR and one EN item and passed
2/2: direct injection, indirect/retrieval injection, system prompt, secret extraction, role
change, emotional provocation, harmful generation, cross-user privacy, malicious upload,
oversized input, citation fabrication, and planner bypass. Six benign/control categories also
passed 2/2 each.

This does not prove provider end-to-end security non-inferiority. That hard gate remains blocked.
Private WhatsApp/group imports return 410, legacy review chunks are excluded from routing and
context formatting, and course reviews remain disabled. Login guesses are bounded atomically by
both privacy-hashed account and network identifiers. Exact citation matching rejects prefix and
invented-suffix labels that the earlier substring implementation accepted.

## 23. Planner rule results

All executed deterministic planner tests passed. Covered gates include:

| Rule | Result |
|---|---|
| Sophomore current plan contains no 4XX, including isolated transferred 3XX/4XX | PASS |
| Authoritative academic year controls stage | PASS |
| Unmet or unknown prerequisite is never recommended | PASS |
| Alternative prerequisite paths are honored | PASS |
| Completed and duplicate courses rejected | PASS |
| Canonical code/name enforced | PASS |
| University debt and required foundations prioritized | PASS |
| Future targets separated | PASS |
| Credit ceiling enforced | PASS |
| Lecture plus required lab/recitation conflicts rejected | PASS |
| Unachievable schedule load exposes SU shortfall | PASS |

No LLM decides these constraints. An authenticated complete profile takes the deterministic
planner path; incomplete-profile or anonymous course recommendations now abstain before retrieval
or generation instead of asking an LLM to turn planner prose into an unchecked course list.

## 24. Groundedness and citations

In the rejected basic pilot, the development scorer reported Groq/DeepSeek groundedness
0.5896/0.5238 and citation support 0.8750/0.8125. These are automated proxy scores: groundedness
is token/evidence overlap, and citation support checks expected retrieved chunk presence rather
than independently judging every generated claim/citation. They cannot satisfy the final
groundedness/citation gate. Production now quarantines retrieved instructions and blocks explicit
source labels that are not exact normalized members of authorized metadata; short-prefix and
invented-suffix bypasses are regression-tested and rejected.

## 25. Latency

See the prompt-ablation table for p50/p95. TTFT was unavailable because the pilot did not capture
native provider TTFT. The repository has no owner-approved p95 product limit, so the gate is
unresolved and no threshold was invented.

## 26–27. Tokens and cost

| Candidate | Successful usage rows | Input tokens | Output tokens | Total tokens | Reported partial total | Per success | Lower bound / planned row |
|---|---:|---:|---:|---:|---:|---:|---:|
| Basic | 43/48 | 28,123 | 6,207 | 34,330 | $0.03564 | $0.000829 | $0.000742 |
| Structured | 37/48 | 24,952 | 5,757 | 30,709 | $0.03579 | $0.000967 | $0.000746 |
| Few-shot | 36/48 | 24,899 | 5,826 | 30,725 | $0.03790 | $0.001053 | $0.000790 |
| Reasoning | 41/48 | 26,817 | 6,959 | 33,776 | $0.03812 | $0.000930 | $0.000794 |

These totals cover successful responses only. “Lower bound / planned row” divides the partial
reported total by all 48 rows and is explicitly not a cost estimate. Missing failed-call usage is
not estimated as zero, so neither exact total benchmark cost nor certified per-query cost is
available. Groq reported no cost, and the repository has no approved per-query budget.

## 28. API errors and fallback

Pilot error rates appear in the ablation table. Fallback was disabled during comparison. The
operational adapter supports only an explicit OpenRouter/Mistral → Groq hop, records requested and
effective provider/model plus reason, prevents loops, and preserves the original deadline. Auth
and invalid-request errors never fall back. The serial run produced 48/48 `ProviderRequestError`
and no usage; it is infrastructure-aborted. The repaired driver now stops after three consecutive
failures (or >25% after 20 rows) and records remaining rows as not run.

## 29. Confidence calibration

Brier, 10-bin ECE, and coverage-risk implementations are unit-tested. No representative labeled
development set and no product coverage/error target were supplied, so no threshold was selected.
Production abstains on hard observable failures (no evidence, planner invalid, schema invalid)
instead of using a fabricated numeric threshold. Comparative provider calibration remains
`BLOCKED`.

## 30–31. Retained and rejected experiments

Retained in runtime:

- Provider abstraction and optional exact OpenRouter model.
- Basic existing application prompt path; Groq default.
- Reasoning support behind an off-by-default flag with reasoning excluded from responses.
- Deterministic guardrails, planner enforcement, language selection, auth, and resource controls.

Rejected/reverted as product choices:

- Parallel four-treatment pilot: retained only as invalid exploratory evidence because provider
  concurrency confounded treatments.
- Structured, few-shot, and reasoning candidate variants: not selected; all had lower pilot task
  success and incomplete operational evidence.
- Serial pilot: invalid for quality because 48/48 candidate requests failed at infrastructure.
- DeepSeek default switch: blocked.
- Reranker default switch: unchanged; historical candidate failed its preregistered latency gate.
- Private WhatsApp and instructor-rating functionality: permanently disabled.

## 32. Known limitations

- Untouched final 100 TR/100 EN provider dataset and blind evaluator are absent.
- No final end-to-end, security non-inferiority, response-language, or confidence comparison.
- Owner p95, cost/query, API-error margin, daily spend limit, and calibration target are unresolved.
- Live pilot uses a small lookup-only development sample and imperfect proxy scorers.
- No TTFT data; provider usage is absent on failed requests.
- The main login, chat, profile, courses, schedule, signup, theme, error, export, and review-policy
  surfaces now share typed TR/EN resources and persistent switching. Course codes, CRNs, file
  formats, model IDs, and stored schema values intentionally remain language-neutral/English.
- Historical retrieval multi-evidence AllGold@10 is low (1/16).
- `npm audit` after non-breaking fixes still reports 5 issues: 2 high (`vite` and `xlsx`) and
  3 moderate (`esbuild`, `react-router`, `react-router-dom`). `xlsx` has no npm fix; the remaining
  path requires major upgrades. The application only writes structured workbook data and does not
  parse user-supplied workbooks, but the dependency risk remains disclosed. No critical issue.
- Python dependency compatibility passes; a Python vulnerability database audit was not run.
- The credential supplied through chat must be considered compromised and rotated after testing.

## 33. Features included in code

- Reversible provider selection and exact DeepSeek ID.
- Normal and native streaming calls with reasoning/secret sanitizer.
- Retries, jitter, Retry-After, circuit breaker, total deadline, telemetry, optional fallback.
- Preregistered paired statistics, internally verified materialized final inputs, runtime-bound
  provider settings, single-run ledger, and a non-injectable fail-closed canonical decision path.
- Deterministic planner, language, security, upload, auth, ownership, retention, and quota controls.
- Disabled course-only review domain plus public bilingual policy.
- App-wide typed TR/EN resources and persistent language switching.
- Redacted per-row pilot evidence, reproducible aggregate tooling, and versioned manifests.

## 34. Features explicitly not enabled

- DeepSeek as default; fallback; reasoning; course-review submissions; private chat ingestion;
  instructor ratings; deployment; production traffic; canary; or infrastructure-as-code.

## 35. Rollback

Runtime rollback is configuration-only:

1. Set `LLM_PROVIDER=groq` (already the default).
2. Leave `LLM_FALLBACK_PROVIDER` empty.
3. Restart the backend so the cached provider instance is rebuilt.

To remove the experiment entirely, revert the local handoff commit after preserving this report
and benchmark records. No database migration or deployment rollback is required.

## 36–37. Exact verification commands and outcomes

| Command | Outcome |
|---|---|
| `.venv/bin/pytest -q` | PASS — 207 passed, 13 warnings |
| `.venv/bin/pytest -q server/tests/test_llm_providers.py server/tests/test_provider_evaluation.py server/tests/test_security_controls.py server/tests/test_frontend_localization.py` | PASS — 73 passed, 13 warnings |
| `cd frontend && npm run build` | PASS — TypeScript and Vite production build |
| `.venv/bin/pip check` | PASS — no broken requirements |
| Authenticated exact-model preflight through ignored local `.env` | PASS — OpenRouter returned exact `deepseek/deepseek-v4-pro`; no substitution |
| Deterministic language/security evaluation Python command | PASS — 220/220 language cases; 36/36 security cases |
| `.venv/bin/python server/evaluation/provider_benchmark.py --manifest data/benchmark/provider_eval/final_manifest.json --validate-only` | Expected exit 2 — `BLOCKED`, 0 TR / 0 EN final items |
| `.venv/bin/python server/evaluation/recompute_development_pilot.py outputs/provider_evaluation/dev-pilot-20260804T120357Z --draws 500` | PASS — all five tracked configurations recomputed from redacted rows; quality deltas matched the summary |
| Check every file in both `artifact_checksums.json` manifests | PASS — all committed redacted rows and run manifests match SHA-256 |
| `cd frontend && npm audit --json` | Partial — 5 unresolved vulnerabilities (2 high, 3 moderate, 0 critical) |
| `git diff --check` | PASS at audit time |
| Tracked/untracked production-shaped secret-pattern scan plus `git check-ignore .env` | PASS — no production-shaped secret in changed files; `.env` is ignored and Groq remains default |
| Changed-file deployment/IaC path scan | PASS — no deployment or infrastructure mutation |

Warnings are deprecations for LangChain Community, FastAPI `on_event`, Starlette/httpx TestClient,
and NumPy/joblib behavior; no test was skipped or hidden.

## 38. Independent critic

Initial review: **44/100, NO-SHIP**, with four critical findings:

1. Sparse upper-level history could promote a sophomore and unknown prerequisites passed open.
2. Conversation-owner lookup failed open on MongoDB errors.
3. Private WhatsApp/instructor-review ingestion remained reachable.
4. Default-switch code omitted mandatory decision gates.

All four were repaired and regression-tested. A later full review scored **81/100** with zero
critical and four major findings: caller-self-attested final configuration, incomplete-profile
recommendations reaching an LLM, substring citation authorization, and an unbounded login route.
All four were then repaired with manifest/runtime/hash/ledger binding, pre-LLM profile abstention,
exact citation membership, and account+network login attempt limits. Checksum manifests also close
that review's pilot-integrity minor. A third audit scored **94/100** with zero critical and one
major: rich caller-supplied provenance dictionaries could still authorize the public switch
verdict because the research runner accepted provider/scorer objects. The public evaluator now
always blocks production authorization, while the sole argument-free canonical path reads the
fixed repository manifest/artifact hashes and has no success branch until the code-bound
independent scorer and real final evidence land. The final independent rescore follows in section
39; external final-data/product-threshold blockers never authorize a model switch.

## 39. Final auditor

Independent read-only delta audit: **96/100 — 0 critical, 0 major, 3 minor**. This satisfies the
requested ≥95 and zero-critical/major quality bar for the local research/evaluation handoff, not
for changing the production model or deploying it.

| Area | Score |
|---|---:|
| Scientific validity | 24/25 |
| Statistical rigor | 15/15 |
| Planner correctness | 15/15 |
| Security and privacy | 14/15 |
| Provider reliability | 10/10 |
| Reproducibility and artifact integrity | 9/10 |
| Language and course review | 5/5 |
| Documentation and rollback | 4/5 |
| **Total** | **96/100** |

The auditor adversarially confirmed that caller-supplied passing dictionaries cannot authorize the
public verdict; the canonical decision signature is argument-free and returns `BLOCKED` from the
repository-fixed manifest; final `source_mappings` and path escapes are rejected; per-call
generation overrides cannot diverge from the resolved provider configuration; and incomplete
profiles do not fall through to an LLM recommendation. Full 207/207, focused 73/73, and five
targeted adversarial tests passed, as did the frontend build, `pip check`, artifact checksums,
secret exclusion, and no-deployment scan.

Remaining minor findings are: five accurately disclosed npm vulnerabilities; process-local login
and quota state that must move to a shared atomic backend before multi-worker deployment; and the
handoff commit, which is completed after this report is finalized. External final-data, independent
scorer, hashed gate-artifact, and owner-threshold blockers remain mandatory. Auditor decision:
**ship the blocked local evaluation/reliability implementation; keep Groq; do not switch DeepSeek;
do not deploy.**

## 40. Final commit hash

A Git commit cannot contain its own content-addressed hash. The exact local handoff `HEAD` is
therefore reported by `git rev-parse HEAD` in the final response and can be verified locally.
Starting commit remains `5a6949c1e2e5b0603d8c1c88820f590099000f34`.

## 41. AWS status

Deferred — not implemented in this scope.
