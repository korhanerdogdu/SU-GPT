# adviSU implementation guide

This file is the operational source of truth for coding agents working in this repository. It
describes the current architecture and its invariants. Historical roadmap text is intentionally
excluded; use Git history and dated reports only when archaeology is required.

## 1. Product contract

adviSU is an academic-advising system built from official curriculum data and student-owned
profile information.

Non-negotiable behavior:

1. Academic arithmetic, eligibility, prerequisites, and schedule conflicts are computed in code.
2. The model explains verified results; it is not an authority for academic facts.
3. Program and curriculum-term boundaries are applied before ranking.
4. Missing critical data causes an explicit abstention, never a guess.
5. A student's question determines answer language; the UI language does not override it.
6. Private chat and instructor-review text must not enter the public retrieval corpus.
7. Raw secrets, hidden reasoning, raw provider bodies, and sensitive identifiers are not logged or
   stored in evaluation artifacts.

## 2. Current request pipeline

`POST /ask/` and `POST /ask/stream` share the same stable answer contract.

```text
request validation and authentication
  -> HMAC-keyed atomic admission reservation
  -> category-specific content-safety classification
  -> prompt-injection / extraction guardrail
  -> language analysis
  -> canonical intent resolution
  -> profile or course-history command, if present
  -> deterministic major/planner/schedule/audit path, if applicable
  -> profile-scoped retrieval policy
  -> hybrid retrieval and optional reranking
  -> retrieval instruction-boundary quarantine
  -> bounded provider adapter
  -> output, citation, secret, language, and reasoning validation
  -> confidence admission or safe abstention
  -> usage reconciliation with actual provider telemetry
```

The order is load-bearing. In particular:

- Crisis classification must run before prompt guardrails and before provider access.
- Deterministic paths must return before retrieval or model generation when they have enough
  authoritative data.
- Retrieved documents are data, never instructions.
- Output validation runs before persistence.
- The resource reservation is reconciled in `finally`; deterministic and refused requests report
  zero provider calls.

## 3. Architecture map

### Backend orchestration

- `server/main.py`: FastAPI routes and request orchestration
- `server/modules/auth.py`: signed bearer tokens and principal checks
- `server/modules/localization.py`: server-visible Turkish/English messages
- `server/modules/language.py`: explainable TR/EN/mixed detection
- `server/modules/intents.py`: canonical English intent vocabulary and legacy boundary
- `server/modules/intent_detector.py`: benchmark-selected classifier with lazy model loading
- `server/modules/rag_router.py`: deterministic bilingual route guards

### Academic engines

- `server/modules/degree_audit.py`: deterministic graduation allocation
- `server/modules/course_planner.py`: stage-, debt-, and prerequisite-aware recommendations
- `server/modules/schedule_planner.py`: CRN bundles and conflict-free timetable construction
- `server/modules/major_advisor.py`: deterministic major-selection flow
- `server/modules/curriculum_registry.py`: official program/term availability
- `server/modules/retrieval_policy.py`: intent-to-corpus and profile gates

### Retrieval

- `server/modules/retrieval_modes.py`: evaluation-compatible retrieval contracts
- `server/modules/catalog_retriever.py`: scoped structured/dense/BM25 fusion path
- `server/modules/bm25_retriever.py`: lexical retrieval
- `server/modules/reranker.py`: optional CrossEncoder reranking
- `server/modules/load_vectorstore.py`: guarded public ingestion and Chroma access
- `server/modules/source_indexer.py`: automatic public source indexing

Do not duplicate metadata scoping inside a new retriever. Every retrieval mode must receive the
same active profile filter so ablation changes ranking rather than corpus access.

### Providers and prompting

- `server/modules/llm_providers.py`: Groq, Mistral, and OpenRouter-compatible adapters
- `server/modules/llm.py`: frozen application prompt and prompt-strategy directives
- `server/modules/query_handlers.py`: provider-chain invocation contract
- `server/modules/config.py`: provider activation and production security policy

Provider adapters enforce exact model identity, bounded retries, non-extendable deadlines, sanitized
errors, circuit state, and safe usage telemetry. Do not use a vendor SDK directly from route code.

The largest application prompt in `llm.py` is bound by the provider preregistration hash. Internal
intent identifiers are English, but `get_llm_chain` converts them to the historical prompt label at
the frozen boundary. Changing the prompt requires a new preregistration and a new development/final
evaluation protocol; never silently update the recorded hash.

### Safety and confidence

- `server/modules/content_safety.py`: self-harm, hate, directed violence, sexual harassment,
  profanity, and CS-technical exemptions
- `server/modules/guardrails.py`: prompt injection, secret extraction, encoded input, retrieval
  poisoning, output leakage, and citation authorization
- `server/modules/confidence.py`: observable-signal answer admission
- `server/modules/resource_controls.py`: application admission and reconciliation
- `server/modules/rate_limit_backends.py`: atomic in-memory/Redis state engines

Content safety and prompt guardrails are complementary; do not replace one with the other. Their
measured union is the selected pipeline.

Confidence must not use model self-reported certainty. It is derived from observable signals:
profile completeness, curriculum availability, evidence count, metadata compatibility, deterministic
validation, provider availability, output schema/language, citations, and safety checks.

### Course reviews

- `server/modules/course_reviews.py`: input validation and deterministic moderation
- `server/modules/course_review_store.py`: HMAC author identity, consent and aggregation policy
- `docs/course_review_data_policy.md`: public feature policy

This is a course-only, consent-based numeric feature. Do not revive legacy private-chat or named
instructor-review ingestion. Aggregates remain suppressed below the configured minimum cohort.

### Frontend

- `frontend/src/contexts/LocaleContext.tsx`: the only interface-locale state
- `frontend/src/localization/resources.ts`: the only typed UI translation catalogue
- `frontend/src/contexts/AuthContext.tsx`: authentication state and token lifecycle
- `frontend/src/pages/LoginPage.tsx`: branded responsive sign-in experience
- `frontend/src/pages/ChatPage.tsx`: streaming conversation surface and usage refresh
- `frontend/src/components/chat/UsageMeter.tsx`: provider-backed allowance display
- `frontend/src/pages/CoursesPage.tsx`: course history and admin upload
- `frontend/src/pages/ProfilePage.tsx`: curriculum profile and degree audit
- `frontend/src/pages/SchedulePage.tsx`: schedule workspace and export

Do not add a second language context or translation dictionary. The answer language and interface
locale are intentionally separate concepts.

## 4. Selected integrated methods

The current implementation combines independently useful methods without retaining parallel,
conflicting subsystems.

| Concern | Selected method |
|---|---|
| Crisis and abuse | Category-specific deterministic classifier with technical exemptions |
| Prompt and data attacks | Unicode/encoded input normalization plus input/retrieval/output guardrails |
| Daily allowance UI | Visible remaining-count meter |
| Daily allowance storage | HMAC-keyed atomic provider-usage backend, not plaintext Mongo counters |
| Intent naming | Canonical English application identifiers with one legacy artifact boundary |
| Localization | One typed frontend catalogue and one backend message catalogue |
| Login experience | Branded split-screen layout integrated with the existing auth/theme/locale contracts |
| Model access | Provider-neutral adapter with explicit experimental activation gates |
| Academic decisions | Deterministic engines with confidence-based abstention |

Rejected parallel implementations must stay removed:

- A second `LanguageContext` or `i18n.ts`
- A plaintext username-based question-usage collection
- A fail-open shared quota backend
- A duplicate general message catalogue
- Direct private-chat or instructor-review ingestion

## 5. Data and storage boundaries

### Committed authoritative data

- `data/degree_requirements/<PROGRAM>/<TERM>.jsonl`
- `data/minors/<CODE>/<TERM>.jsonl`
- `data/curricula/registry.jsonl`
- `data/course_catalog/current.jsonl`
- `data/schedule/*.jsonl`
- `data/benchmark/*`

### MongoDB

MongoDB stores users, academic profiles, course history, conversations, saved schedules, source
records, ingestion jobs, and feature-gated course reviews. Chroma is a reproducible derived index,
not a source of truth.

### Rate-limit backends

Use shared Redis in multi-worker deployments. Identifiers are derived with versioned HMAC keys.
Key rotation may include previous keys for coordinated migration. Never log derived keys together
with raw identifiers.

### Ingestion

All ingestion paths must call the shared public-source policy before saving or indexing. Validate
the original upload filename before adding a digest prefix. Path traversal, symlink root escape,
private source markers, invalid signatures, unsupported formats, and aggregate-size overflow fail
closed.

## 6. Configuration

Use `.env.example` as the schema. Real `.env` files and credentials remain untracked.

Important groups:

- Provider: `LLM_PROVIDER`, provider keys/models/base URLs, retries, timeout, token limit
- Experimental OpenRouter: explicit feature flag, reasoning controls, exact model
- Authentication: production passwords and token secret
- Privacy: abuse/quota HMAC keyring and course-review HMAC secret
- Retrieval: default mode, candidate depth, final top-k, embedding and reranker settings
- Resource control: request windows, daily provider/token/cost limits, concurrency, backend
- Features: course-review flag and consent version

Provider activation is checked both at application startup and at provider-settings construction.
This is required because evaluation scripts and workers can construct providers without running an
ASGI lifespan hook.

## 7. Verification gates

Minimum gate for any behavior change:

```bash
.venv/bin/pytest -q
cd frontend && npm run build
```

Current accepted baseline:

- 333 Python tests passing
- Frontend TypeScript/Vite production build passing
- 52/52 deterministic security v2 cases
- 42-case layered input benchmark: recall 1.0, F1 1.0, false-refusal rate 0.0
- Language benchmark: TR 100/100, EN 100/100, mixed 20/20
- Bounded `/ask/` endpoint benchmark: 12/12

Safety benchmark:

```bash
.venv/bin/python server/evaluation/content_safety_benchmark.py
```

Language benchmark:

```bash
.venv/bin/python -c 'import json,sys; sys.path.insert(0,"server"); from evaluation.language_benchmark import evaluate; from modules.language import detect_language; print(json.dumps(evaluate(detect_language), indent=2))'
```

Endpoint artifacts must be generated from a clean committed implementation. The manifest must bind
the exact base commit, runner, dataset, configuration, and empty working-tree diff. Artifacts are
append-only and contain redacted decisions/digests only.

Never interpret test count alone as quality. Compare the same frozen rows, expected actions, and
metrics across methods.

## 8. Change protocol

Before editing:

1. Inspect `git status`; preserve unrelated user changes.
2. Read the relevant module and its tests.
3. Identify whether the target is deterministic, retrieval, provider, UI, or policy code.
4. Check frozen benchmark or preregistration constraints.

While editing:

1. Extend an existing abstraction rather than creating a parallel subsystem.
2. Keep deterministic and provider-dependent behavior visibly separate.
3. Add a regression test for the failure being fixed.
4. Avoid raw prompts, model outputs, credentials, usernames, or private content in artifacts.
5. Keep Turkish and English behavior symmetric unless a language-specific rule is deliberate and
   tested.

Before handoff:

1. Run targeted tests while iterating.
2. Run the full Python suite and frontend build.
3. Run affected frozen benchmarks.
4. Use `git diff --check` and scan for accidentally committed secrets.
5. Document scope limits; do not label bounded synthetic testing as production certification.

## 9. Git and evidence policy

- Do not commit `.env`, credentials, raw reasoning, raw provider bodies, or private student data.
- Preserve benchmark manifests and checksums when reporting measured results.
- Do not rewrite shared history unless explicitly requested and protected with a recoverable local
  backup ref.
- When history must be rewritten, resolve the exact remote tip first and push with
  `--force-with-lease`, never an unconditional force.
- Commit messages should describe the product methods and observable behavior, not the tool used to
  produce the change.

## 10. Deployment status

AWS deployment is deferred. Do not add infrastructure, publish services, or activate production
providers without an explicit new request. Current work is optimized for local correctness,
security, reproducible evaluation, and reversible provider experiments.

## 11. Known technical debt

- Route-level frontend code splitting is still needed to remove the Vite large-chunk warning.
- FastAPI startup hooks should eventually move to lifespan handlers.
- Third-party loader and test-client deprecation warnings need dependency migration.
- The bounded endpoint security set is not a substitute for an independent penetration test.
- Live-provider quality and cost claims require the exact preregistered split and configuration.
