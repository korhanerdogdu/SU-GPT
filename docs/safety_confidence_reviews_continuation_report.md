# SU-GPT Safety, Confidence, Resource Controls, and Course Reviews — Continuation Report

## Decision

**Primary status: `SHIP_LOCAL_BLOCKED_IMPLEMENTATION`**

The requested application paths are locally implemented and the available deterministic,
endpoint, backend, and frontend checks pass. Activation and any model/default decision remain
blocked by external product thresholds, a sealed independent final dataset/scorer, an approved
live-provider cost cap, production shared-state infrastructure, and deployment scope. This report
does not describe the branch as production-ready.

## Repository and provenance

| Field | Evidence |
|---|---|
| Selected repository | `/Users/selmanyilmaz/dev/advisu` |
| Starting branch | `feat/openrouter-deepseek-evaluation` descendant at the required checkpoint |
| Starting commit | `28a55cf4836429986350e8a1e0c6a5a6a09a631b` |
| Continuation branch | `codex/complete-safety-confidence-reviews` |
| Final commit | The local commit containing this report; the immutable hash is recorded in the final handoff because a commit cannot contain its own hash. |
| Initial worktree | Clean; no user-owned changes were present. Empty diff SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`. |
| Runtime | Python 3.12.9, pip 26.1.2, Node 20.20.0, npm 10.8.2 |

The Desktop copy was not selected. The development copy contains the required checkpoint, prior
OpenRouter/DeepSeek evaluation code, manifests, tests, and immutable pilot evidence.

## Forensic gap map

Before this continuation, three-language provider adapters, a reversible experimental OpenRouter
path, final-manifest gating, deterministic language selection, the 36-row security corpus, a basic
confidence formula, process-local resource controls, and a disabled course-review domain scaffold
already existed. The real `/ask` path did not consistently validate untouched LLM output before
persistence; `llm_only` could bypass equivalent output controls; quarantined retrieval could be
misreported as evidence; confidence lacked an explicit answer contract and intent policies; the
limiter was process-local; review persistence/routes/UI were incomplete; and no real endpoint
adversarial artifact existed. Private review ingestion entry points also remained reachable.

The two existing provider pilot directories and their checksum manifests were treated as immutable.
No existing pilot row, manifest, or checksum was changed.

## Completion matrix

`VERIFIED_COMPLETE` means locally verified, not deployed or production-ready.

| Requirement | State | Concrete implementation evidence | Test/command and observed result |
|---|---|---|---|
| Input guardrail, TR/EN matched benign cases, Unicode/zero-width/encoded attacks | `VERIFIED_COMPLETE` | `modules.guardrails.assess_input`, bounded detection variants | `pytest ...test_guardrails_v2.py`; included in 151 focused passes; v2 52/52 |
| Retrieval instruction quarantine and trust boundary | `VERIFIED_COMPLETE` | `modules.guardrails.assess_retrieval`, `retrieval_boundary`; `main._format_for_context` | endpoint benchmark: poisoned retrieval abstained; 12/12 |
| Untouched output validation before formatting/persistence | `VERIFIED_COMPLETE` | `modules.guardrails.validate_output`; both `/ask` LLM branches in `main.ask_question` | `test_chat_safety_endpoints.py`; secret output filtered before persistence |
| Exact authorized citation membership | `VERIFIED_COMPLETE` | `citations_are_authorized`, `main._source_list_is_authorized` | focused security suite passed; prefix/suffix cases covered |
| Evidence-based structured confidence and explicit `answer_allowed` | `VERIFIED_COMPLETE` | `confidence.ConfidenceSignals`, `policy_for_intent`, `assess` | `test_confidence_policies.py`; unverified claim coverage now fails closed |
| Representative confidence calibration threshold | `BLOCKED_EXTERNAL` | Calibration pipeline refuses implicit selection | No representative labeled development set or owner risk/coverage target exists |
| Deterministic planner/prerequisite/schedule validation and abstention | `VERIFIED_COMPLETE` | `main.ask_question` calls `validate_proposed_plan` and schedule conflict validation before returning | full backend 294/294; focused advising/confidence/security paths passed |
| Shared atomic backend protocol and local adapter | `VERIFIED_COMPLETE` | `rate_limit_backends.RateLimitBackend`, `SharedInMemoryRateLimitBackend` | two-controller/concurrent tests: zero over-admission |
| Redis-compatible atomic adapter and fail-closed selection | `VERIFIED_COMPLETE` | `RedisRateLimitBackend` Lua scripts; `build_rate_limit_backend`; no fallback | `test_shared_resource_controls.py`; focused suite passed; fake Redis Lua boundary covered |
| Production Redis service/operations | `BLOCKED_EXTERNAL` | Required configuration is documented and validated | No Redis or deployment work authorized |
| Independent login/chat/stream/upload/conversation/review controls and daily provider budgets | `VERIFIED_COMPLETE` | `ResourceController.check_operation`, reservation/reconciliation, real route calls in `main.py` | boundary, concurrency, TTL, backend failure, retry-after, privacy-key tests passed |
| Course-only review storage, consent, atomic duplicate prevention, moderation, delete, k=10 aggregation | `VERIFIED_COMPLETE` | `CourseReviewStore`; dedicated `courseReviews` collection | store/endpoint suites included in 151 focused passes; nine suppressed, ten visible, delete re-suppresses |
| Course-review feature disabled by default and bilingual accessible UI | `VERIFIED_COMPLETE` | config default false; policy/route gate; typed frontend resources and form | `.env.example` and config verified; frontend build passed |
| Private chat/instructor review ingestion disabled | `VERIFIED_COMPLETE` | `load_vectorstore` rejection and removed bulk-ingest target | `test_private_ingestion_disabled.py`; full suite passed |
| Endpoint adversarial benchmark with redacted immutable artifacts | `VERIFIED_COMPLETE` | `evaluation.endpoint_security_benchmark` loads a checksummed bounded manifest and calls real FastAPI `/ask` | `bounded-v1-20260805`: 12/12, ASR 0, false refusal 0 |
| Auth/cross-user, upload exhaustion, retry, planner regression coverage | `PARTIAL` | Existing `test_security_controls.py` plus new endpoint/resource suites | Applicable local paths pass; not every requested category has a dedicated persisted endpoint row |
| Live Groq vs DeepSeek v4 Pro development benchmark | `BLOCKED_EXTERNAL` | Preregistered harness, redacted rows, stop rules, statistics already exist | Not run: chat-shared credential is forbidden and no operator-approved cost cap was declared |
| Sealed final model decision | `BLOCKED_EXTERNAL` | Canonical validator fails closed | validator exit 2; 0 TR/0 EN independent final items, 21 blocking reasons |
| AWS/deployment | `NOT_APPLICABLE` | No deployment/IaC files changed | deployment diff scan empty |

## Retained implementation

### Guardrails and endpoint safety

- Added normalized, mixed-script, zero-width, spacing/punctuation, bounded Base64, role-change,
  jailbreak, harmful-generation, discrimination/harassment, secret, system-prompt, and reasoning
  detection while retaining benign academic/counterspeech examples.
- Delimited retrieved evidence, recent conversation context, and student profile context as
  untrusted data. Quarantined chunks are removed before the model and do not count as evidence.
- Validated untouched LLM output before sanitization, summarization, or persistence in both RAG and
  `llm_only` paths. Refusals never echo attack payloads.
- Exact normalized source membership prevents citation prefix/suffix spoofing. Only stable public
  confidence states are returned.

### Confidence and deterministic academic safety

- Added intent policies and observable hard-failure reasons with `answer_allowed` and statuses:
  `verified`, `limited_evidence`, `profile_required`, `curriculum_unavailable`,
  `provider_unavailable`, `cannot_verify`, and `safe_abstention`.
- Missing critical profile/curriculum data, provider failure, empty safe retrieval, unauthorized
  citations, invalid planner output, and schedule conflicts abstain deterministically.
- A correct source label alone no longer admits model prose. Evidence-required model answers
  abstain until a claim/evidence coverage verifier has actually run; a real-route fabricated-claim
  regression verifies that the unsupported text is neither returned nor persisted.
- Course eligibility, prerequisite checks, academic stage, completed-course exclusion, degree
  audit arithmetic, and schedule conflicts remain code-owned rather than LLM-owned.
- No numeric threshold was selected. The existing calibration pipeline remains development-only.

### Shared resource controls

- Introduced a backend protocol, a thread-safe shared-memory development implementation, and an
  opt-in Redis adapter whose admit/window/reconcile paths use atomic Lua scripts.
- Backend identifiers are purpose-separated, versioned HMACs. Rotation can check current and prior
  aliases without storing plaintext usernames, account IDs, or network addresses.
- Added atomic concurrency, request windows, daily provider requests, token reservations, and exact
  integer-microdollar budget accounting with idempotent reconciliation and TTL cleanup.
- Shared-backend timeout/unavailability fails closed; production validation cannot silently select
  memory state. Safe 429/503 responses and `Retry-After` are tested.
- Login account/network attempts, chat, stream, upload, conversation creation, and course-review
  submission have independent operation keys. Upload file size/count/total validation remains in
  the existing upload validators.

### Course-only reviews

- Added a dedicated Mongo-compatible store with unique `reviewId` and
  `(courseCode, authorDigest)` indexes, startup index validation, atomic insertion, compare-and-set
  moderation, and hard deletion.
- Only five integer 1–5 course dimensions are accepted. Consent/version are mandatory. Instructor
  targets, identity fields, unsupported dimensions, subgroup collection, and stale consent fail.
- Email, phone, student number, URL, social handle, instructor reference, and likely-name material
  is redacted and rejected; safe free text remains pending for human review. Raw identity is never
  placed on review documents or public responses.
- Only approved reviews aggregate. Mongo suppresses results below ten before a count reaches the
  application. Deletion can return an aggregate to suppression.
- Added authenticated submission/deletion/aggregate routes, fail-closed admin moderation, and a
  typed Turkish/English accessible form. The feature remains off by default.

### Evaluation

- Preserved security v1 and added a labeled, clustered v2 deterministic corpus with category,
  language, layer, critical, and obfuscation metrics. Prompts are not emitted in results.
- Added a real FastAPI `/ask` synthetic-provider benchmark. Cases come from a checksummed bounded
  development manifest; run evidence binds dataset, manifest, runner, configuration, HEAD, and
  working-diff hashes and persists only redacted decisions/digests under a unique run ID.
- The existing provider harness retains failures in denominators, three repetitions, randomized
  paired order, warmups, stop rows, paired cluster bootstrap (10,000 draws), clustered sign-flip,
  McNemar diagnostic, and Holm adjustment. No live rows were fabricated in this continuation.

## Measurement tables

### Guardrails

The v1 scorer changed to report richer fields, so old and new rows are **separate snapshots**, not
a valid measured improvement. V2 is a new development corpus and is not an independent final set.

| Snapshot | Dataset/scorer | N | Passed | ASR | Safe refusal | False refusal | Macro-F1 | Critical bypass | Obfuscation bypass | TR / EN |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Before | v1, prior scorer | 36 | 36 | 0.000 | 1.000 | 0.000 | not reported | not labeled | not labeled | aggregate only |
| Final compatibility | v1 SHA `e0685c...`, final scorer SHA `d9197c...` | 36 | 36 | 0.000 | 1.000 | 0.000 | not labeled | not labeled | not labeled | corpus-compatible |
| Final extended | v2 SHA `de7e5a...`, final scorer SHA `d9197c...` | 52 | 52 | 0.000 | 1.000 | 0.000 | 1.000 | 0 | 0 | 26/26 each |

Endpoint bounded snapshot: 12/12, 6 TR and 6 EN, ASR 0.000, critical bypasses 0, safe-refusal
accuracy 1.000, false-refusal rate 0.000.

### Confidence

No representative labeled set exists, so this is not an improvement claim and no threshold is
chosen. The unchanged metric implementation's small exact unit fixture yields Brier 0.055, ECE
0.200, coverage 0.500 and selective accuracy 1.000 at an illustrative 0.5 cut. That fixture is
only a math test—not product evidence. Final functional checks add hard abstention behavior for all
observable failure modes; representative TR/EN/per-intent coverage, answered-wrong, and critical
wrong-answer metrics remain `BLOCKED_EXTERNAL` pending labels.

| Measure | Before | Final | Interpretation |
|---|---:|---:|---|
| Brier / ECE | 0.055 / 0.200 on toy fixture | unchanged | Pipeline regression check only |
| Threshold | none | none | Correctly unresolved |
| Explicit answer contract | absent | present | Functional contract, not calibration evidence |
| Missing-profile/provider/curriculum/invalid-plan/unverified-claim fallthrough | incompletely enforced | regression tests pass | Observable hard gates |

### Rate limits

| Measure | Before | Final | Notes |
|---|---:|---:|---|
| Two-controller cap=1 admissions | 2 | 1 | Before controllers had isolated state; final shares one backend |
| Concurrent over-admission | 1 | 0 | Same local cap scenario |
| False rejection | 0 observed | 0 observed | Boundary/concurrency suite |
| Multi-worker consistency | failed process-local simulation | passed shared-backend simulation | Redis deployment not performed |
| Backend unavailable | local-only/not a production contract | explicit fail-closed 503 | No silent fallback |
| p50 / p95 admission+reconcile | 0.0081 / 0.0096 ms | 0.0688 / 0.1018 ms | Latest local development memory run; no predefined budget |
| TTL recovery / cleanup | limited | passed | Expired windows, leases, usage, settled IDs |
| Backend keys | process HMAC | versioned purpose HMAC | Plain identity inspection passed |

The latency figures are local microbenchmarks and not a Redis production latency claim.

### Course reviews

| Control | Before | Final local result |
|---|---|---|
| Persistence/routes/UI | disabled scaffold | dedicated store, routes, bilingual form; still disabled by default |
| Validation | domain-only | invalid 0/6/float/string/missing/extra dimensions rejected |
| PII/person target | partial | redaction plus rejected moderation state |
| Duplicate prevention | process-local | atomic unique database insert; concurrent test passes |
| Moderation | scaffold | fail-closed admin route and pending-only CAS |
| Small-group privacy | domain-only | database pipeline suppresses 9; exposes aggregate at 10 |
| Deletion | incomplete | owner-HMAC hard delete; can re-suppress aggregate |
| Cross-user/raw identity | not end-to-end | auth/HMAC projection tests pass; digest absent from API |
| Feature flag | off | off; real write/read routes return safe disabled response |

### Cybersecurity and real endpoint coverage

| Measure | Before | Final snapshot |
|---|---:|---:|
| Deterministic attack success | 0/36 | 0/36 v1; 0/52 v2 |
| Critical attack success | not separately labeled | 0 v2 |
| Secret/system/reasoning leakage bypass | unit-only aggregate | 0 in covered deterministic/endpoint cases |
| Cross-user leakage | 0 in existing route tests | 0 in existing route tests |
| Indirect prompt-injection bypass | helper coverage | 0 in real `/ask` poisoned-retrieval rows |
| Real endpoint adversarial rows | none | 12 redacted rows, all passed |
| False refusal | 0 deterministic | 0 deterministic and endpoint snapshots |

The endpoint artifact covers a manifest-bound, bounded real `/ask` subset with synthetic in-memory providers.
Authentication, uploads, review moderation, and resource controls are exercised by route/integration
tests but are not all represented as persisted endpoint benchmark rows; this is reported as partial
rather than hidden.

## Live provider development benchmark

| Field | Status |
|---|---|
| Baseline | Groq `llama-3.3-70b-versatile` remains default |
| Candidate | OpenRouter `deepseek/deepseek-v4-pro`, experimental only |
| Primary reasoning | Disabled by preregistration |
| Planned protocol | Same frozen prompts/context, two warmups/provider, three randomized paired repetitions, controlled concurrency |
| Credential preflight | Not executed with the previously chat-shared credential; using it is explicitly forbidden |
| Cost preflight | `BLOCKED_EXTERNAL`: no predeclared operator cost-safety cap |
| Live attempts/rows | 0; no fabricated `not_run` provider run was created because preflight prevented a run from starting |
| Quality, latency, TTFT, tokens, cost, CI, p-values | `BLOCKED_EXTERNAL`; no live observations exist |
| Default switch | Blocked regardless of development outcome |

The statistical implementation is covered by provider evaluation tests, but confidence intervals
or p-values are not reported for an unexecuted experiment.

## Verification record

| Command | Status and result |
|---|---|
| `.venv/bin/pytest -q` | PASS — 319 passed, 0 failed, 0 skipped, 0 xfailed; 13 deprecation warnings |
| `.venv/bin/pytest -q server/tests/test_provider_evaluation.py ... server/tests/test_course_review_endpoints.py` | PASS — 151 passed, 0 failed/skipped/xfailed; 13 warnings |
| `.venv/bin/pytest -q -s server/tests/test_shared_resource_controls.py::test_shared_memory_admission_overhead_p50_p95` | PASS — 1 passed; p50 0.0688 ms, p95 0.1018 ms |
| `PYTHONPATH=server ... evaluate(security_adversarial_v1.jsonl)` | PASS — 36/36, ASR 0, safe refusal 1, false refusal 0 |
| `PYTHONPATH=server ... evaluate(security_adversarial_v2.jsonl)` | PASS — 52/52, macro-F1 1, ASR/critical/obfuscation bypass 0 |
| `PYTHONPATH=server ... language_benchmark.evaluate(detect_language)` | PASS — 220/220: TR 100/100, EN 100/100, mixed 20/20 |
| `.venv/bin/python server/evaluation/endpoint_security_benchmark.py --run-id bounded-v1-20260805` | PASS — 12/12; manifest-bound redacted finalized artifact |
| `cd frontend && npm run build` | PASS — TypeScript and Vite build; 1,890 modules; bundle-size warning only |
| `.venv/bin/pip check` | PASS — no broken requirements |
| `cd frontend && npm audit --json` | Initial final run completed: 5 total, 2 high (`vite`, `xlsx`), 3 moderate (`esbuild`, `react-router`, `react-router-dom`), 0 critical; unchanged from baseline. Last repeat was `BLOCKED_EXTERNAL` by registry DNS, with no dependency change between runs. |
| `.venv/bin/python server/evaluation/provider_benchmark.py --manifest data/benchmark/provider_eval/final_manifest.json --validate-only` | EXPECTED BLOCK — exit 2, `not_provisioned`, not eligible, 21 reasons |
| Verify SHA-256 entries in two prior pilot and three new endpoint checksum manifests | PASS — 16 checked files total, 0 mismatches; all five directories finalized |
| `git check-ignore -q .env`; `git ls-files --error-unmatch .env` | PASS — ignored, untracked, unstaged |
| Provider-prefix/credential pattern scan of changed files and staged files | PASS after explicit synthetic-fixture review; no real credential or exposed chat key found |
| `git diff --check` and staged equivalent | PASS |
| Deployment/IaC path scan | PASS — no `render.yaml`, IaC, container, or deployment mutation |
| Default/flag inspection | PASS — `LLM_PROVIDER` defaults to `groq`; `COURSE_REVIEWS_ENABLED` defaults to `false` |

Baseline full backend was 207 passed with 13 warnings; the final suite has 319 passed and the same
warning count. This is a regression result, not proof of production security. Baseline and final
frontend build passed. npm audit stayed at 2 high, 3 moderate, 0 critical.

Dataset/scorer hashes:

- Security v1: `e0685c9e846b9538f297efb06253b32c5fcfbd3a62936de698a5cb2041cadb9f`
- Security v2: `de7e5ae15a6f53cd8a87f42cd17ff604bc120c2620126753b957c95ca1cfc43b`
- Final security scorer: `d9197cb02c61e863bcd2c486e9e7ee25f84b61d8953aa304a7daacdd6eea2bf4`
- Language scorer: `3eac562b40d05d24587074c602800c34e6bc92235c65247c394388aff7d6e751`
- Confidence calibration code: `61c611498107bdc08202906eb5ed350e7cd47637f3ca922da0c5a2264e6d4d9f`
- Bounded endpoint dataset: `c3c376271326aa9d78bc67d399b182fc83ec047dba49b0b87051af6b9775ddec`
- Bounded endpoint manifest: `10b5d37e43a4b1948e425b10305717a133a76aae321d776885ad2ba334003398`
- Bounded endpoint runner: `377106d51fb441b01ccb59ded7a5ca789a240f0ac4e2028deb1259299968afb1`

## Changed files and purpose

| Files | Purpose |
|---|---|
| `.env.example`, `server/requirements.txt`, `docs/resource_controls.md` | Placeholder-only configuration, Redis client dependency, operations/failure/rotation documentation |
| `server/modules/rate_limit_backends.py`, `server/modules/resource_controls.py` | Atomic backend protocol/adapters, quotas, concurrency, HMAC keys, reconciliation |
| `server/main.py`, `server/modules/config.py`, `server/modules/query_handlers.py`, `server/modules/localization.py` | Real route integration, safe errors, feature gates, confidence/status and provider telemetry handling |
| `server/modules/guardrails.py`, `server/evaluation/security_benchmark.py`, `data/benchmark/security_adversarial_v2.jsonl` | Three-layer controls and richer deterministic benchmark |
| `server/modules/confidence.py`, `server/tests/test_confidence_policies.py` | Observable intent policy and explicit answer decision |
| `server/modules/course_reviews.py`, `server/modules/course_review_store.py`, `server/modules/mongodb.py`, `docs/course_review_data_policy.md` | Course-only validation, privacy store, collection wiring, policy |
| `server/modules/load_vectorstore.py`, `server/modules/source_indexer.py`, `server/scripts/bulk_ingest.py`, `server/tests/test_private_ingestion_disabled.py` | One normalized private-source deny policy across file, text, bulk, and auto-index ingestion |
| `frontend/src/lib/api.ts`, `frontend/src/localization/resources.ts`, `frontend/src/pages/CourseReviewPolicyPage.tsx` | Typed bilingual feature-gated review UI/API |
| `data/benchmark/endpoint_security_v1.jsonl`, its manifest, `server/evaluation/endpoint_security_benchmark.py`, `outputs/security_endpoint/*` | Manifest-bound bounded real `/ask` benchmark, run registry, and three preserved unique redacted finalized runs |
| `server/tests/test_guardrails_v2.py`, `test_chat_safety_endpoints.py` | Guardrail and real endpoint regressions |
| `server/tests/test_shared_resource_controls.py` | Atomicity, limits, TTL, failure, Redis, privacy, overhead tests |
| `server/tests/test_course_review_store.py`, `test_course_review_endpoints.py`, `test_language_and_course_reviews.py`, `test_frontend_localization.py` | Store, route, validation, localization, flag, and UI regressions |
| This report | Evidence, limitations, verification, and rollback record |

## Rejected or deferred paths

- No provider default switch was retained. DeepSeek stays experimental.
- No provider-side reasoning capture or persistence was added.
- No private group/chat ingestion, instructor rating, subgroup review analytics, or real review data
  was added.
- No silent Redis-to-memory production fallback was added.
- No product confidence/false-refusal/overhead threshold was invented after results.
- No live provider run was started without cost authorization.
- No AWS, canary, push, PR, Redis deployment, IaC, Docker Compose, or `render.yaml` change was made.

## Independent critic

Round 1 scored **78/100: 0 critical, 4 major, 3 minor**. The critic demonstrated an authorized-
source fabricated claim, a private-chat bulk-ingest path, an under-bound endpoint artifact, and an
unenforced production OpenRouter switch. All four were repaired with new regression tests. Minor
hardening also added a separate production course-review release approval, operation limits on
review read/delete/moderation routes, and an explicit Redis Cluster limitation. The final
read-only re-audit scored **91/100: 0 critical, 1 major, 1 minor**: three majors closed, while
alternate text/auto-index ingestion aliases remained. A second repair centralized normalized
source policy across file, text, bulk, and auto-index paths and added course-relabeled text and
auto-index regressions. The remaining minor was addressed without mutating immutable evidence by
adding an endpoint run registry that marks both pre-manifest directories legacy/superseded.
Round 3 scored **97/100: 0 critical, 0 major, 1 minor** and approved
`SHIP_LOCAL_BLOCKED_IMPLEMENTATION`; the one minor was the legacy-directory ambiguity already
mitigated by that registry without rewriting the old artifacts.

## Known limitations and blockers

1. No sealed independent final set with at least 100 Turkish and 100 English items or independent
   scorer exists; the model decision remains blocked.
2. No owner-defined confidence risk/coverage target or false-refusal acceptance threshold exists.
3. No operator-approved live-provider cost cap exists, and the credential previously pasted into
   chat must not be used. No live comparative measurements were produced.
4. Redis has not been deployed or load-tested across real processes/network failure modes.
5. Course reviews have not been enabled, migrated, or tested with real user data; moderator
   operations and retention need production governance.
6. Five frontend dependency advisories remain unchanged from baseline; zero are critical.
7. The final endpoint artifact is a bounded synthetic-provider development benchmark, not an
   independent external penetration test.

## Workstream status

| Workstream | Status |
|---|---|
| Guardrails and discrimination protection | `VERIFIED_COMPLETE` locally |
| Confidence and safe abstention | `PARTIAL` — hard observable gates complete; calibration threshold blocked |
| Shared rate limits and quotas | `PARTIAL` — code/tests complete; production Redis deployment/load evidence blocked |
| Course-only reviews | `VERIFIED_COMPLETE` locally and disabled; activation blocked |
| Endpoint cybersecurity benchmark | `VERIFIED_COMPLETE` for implemented 12-row `/ask` scope; broader persisted endpoint matrix partial |
| Real-model provider benchmark | `BLOCKED_EXTERNAL` |
| Final model decision | `BLOCKED_EXTERNAL` |

## Rollback

The branch is local and has not been pushed. To preserve evidence, create a safety branch/tag first,
then revert the final local continuation commit with `git revert <final-commit-hash>`. Do not use
`git reset --hard`. Course reviews can also remain operationally unavailable by keeping
`COURSE_REVIEWS_ENABLED=false`; Groq remains selected with `LLM_PROVIDER=groq`.

Groq remains the default: **confirmed**.

Course reviews remain disabled by default: **confirmed**.

Deferred — not implemented in this scope.
