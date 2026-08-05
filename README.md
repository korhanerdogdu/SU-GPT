# adviSU — Retrieval-Augmented Academic Advising

adviSU is a profile-aware academic advising system for Sabancı University. It combines official
curriculum data, deterministic academic rules, retrieval-augmented generation, bounded language
models, and layered safety controls.

The core rule is simple: calculations and eligibility decisions are made in code; the language
model may explain verified results but must not invent academic facts.

## Current capabilities

- Program- and admission-term-specific degree audits
- Deterministic course recommendations with prerequisite and academic-year checks
- Conflict-free weekly schedules using official section and CRN data
- Course-history and academic-profile updates through UI or chat
- Turkish and English interface and answer routing
- Conversation history with bounded working memory
- CSV/XLSX exports
- Optional, privacy-preserving course-rating aggregates
- Groq, Mistral, and experimental OpenRouter/DeepSeek provider adapters
- Confidence-based abstention when evidence, profile data, or provider output is insufficient
- Layered input, retrieval, and output safety
- Per-user provider-request, token, cost, rate, and concurrency controls

## Runtime architecture

```text
Authenticated request
  -> privacy-preserving resource admission
  -> crisis / violence / harassment content classifier
  -> prompt-injection and secret-extraction guardrails
  -> language selection
  -> canonical English intent boundary
  -> deterministic command or planner, when available
  -> profile-scoped retrieval, when evidence is required
  -> retrieved-content quarantine
  -> bounded provider call
  -> output, citation, secret, and reasoning validation
  -> observable-signal confidence policy
  -> answer or safe abstention
  -> exact quota reconciliation
```

Deterministic paths return before the provider call whenever possible. A refused request or a
verified local calculation does not consume the daily provider-backed question allowance.

## Important design decisions

### Academic correctness

- Retrieval is filtered by `data_role`, program, and curriculum term before ranking.
- Graduation arithmetic uses SU credits and official requirement files.
- A course is counted in at most one graduation category.
- Course recommendations exclude completed courses, enforce prerequisites, and apply the
  conservative no-4XX rule for students at sophomore level or below.
- If required profile or curriculum data is missing, the system abstains instead of guessing.

### Intent and language

- Application code uses canonical English intent identifiers such as `graduation_status`,
  `course_recommendation`, and `weekly_schedule`.
- The trained classifier and frozen historical benchmark keep their legacy labels. Translation
  happens once at the boundary so recorded measurements remain reproducible.
- Response language follows the question rather than the interface setting. Course codes,
  acronyms, and Turkish proper names are treated as weak or neutral signals.

### Safety and privacy

- Self-harm messages receive a dedicated response with real help resources; they are not handled
  as generic off-topic questions.
- Hate, directed violence, sexual harassment, and profanity are classified before retrieval.
- Prompt injection, system-prompt extraction, secret extraction, encoded attacks, poisoned
  retrieval chunks, unauthorized citations, hidden reasoning, and secret-bearing outputs are
  checked independently.
- User quota keys and review-author identifiers are HMAC-derived; plaintext identities are not
  stored in rate-limit backends.
- Private chat and instructor-review corpora cannot enter public ingestion paths. The policy also
  detects private markers after generated digest prefixes.
- Raw provider reasoning and raw provider error bodies are not persisted.

### Resource limits

The default allowance is 15 provider-backed questions per user per UTC day. Shared deployments
should configure Redis so limits remain atomic across workers. A selected shared backend fails
closed; it never silently falls back to process-local counters.

## Measured state

| Verification | Result |
|---|---:|
| Python test suite | 333 passed |
| Frontend production build | passed |
| Layered input-safety benchmark | recall 100%, F1 1.000, false refusals 0% |
| Deterministic security benchmark v2 | 52/52 |
| Bounded real `/ask/` endpoint security | 12/12 |
| Turkish language routing | 100/100 |
| English language routing | 100/100 |
| Mixed-language routing | 20/20 |

The endpoint benchmark uses synthetic in-memory provider responses and production middleware
ordering. It is a bounded regression suite, not a comprehensive penetration test or live-model
quality claim.

Detailed integration evidence is in
[`docs/architecture_integration_report_20260805.md`](docs/architecture_integration_report_20260805.md).

## Repository layout

```text
frontend/                         React, Vite, TypeScript, Tailwind UI
server/main.py                    FastAPI routes and request orchestration
server/modules/                   planners, retrieval, providers, safety, persistence
server/evaluation/                deterministic and provider evaluation tools
server/tests/                     unit, integration, endpoint, and security tests
data/degree_requirements/         official program/term requirement corpus
data/minors/                      official minor requirement corpus
data/course_catalog/              course catalogue
data/schedule/                    official schedule snapshots
data/benchmark/                   frozen evaluation corpora and manifests
outputs/                          redacted, checksummed evaluation artifacts
docs/                             architecture, privacy, evaluation, and operations notes
```

## Local setup

### Requirements

- Python 3.12
- Node.js 22
- MongoDB
- Optional Redis for shared production limits
- A supported provider API key

### Environment

```bash
cp .env.example .env
```

At minimum, configure one provider and MongoDB. Never commit `.env` or real credentials.

Typical Groq configuration:

```env
LLM_PROVIDER=groq
GROQ_API_KEY=replace-me
MONGO_URI=mongodb://localhost:27017
```

Experimental OpenRouter configuration remains gated and is not the production default:

```env
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=replace-me
OPENROUTER_MODEL_NAME=deepseek/deepseek-v4-pro
OPENROUTER_EXPERIMENTAL_ENABLED=true
```

Production activation follows the repository's canonical provider decision and may reject an
experimental provider even when credentials are present.

### Backend

```bash
python3 -m venv .venv
.venv/bin/pip install -r server/requirements.txt
cd server
../.venv/bin/uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Health check: `http://127.0.0.1:8000/test`

### Frontend

```bash
cd frontend
npm install
npm run dev -- --host 127.0.0.1 --port 5173
```

Application: `http://127.0.0.1:5173`

Demo accounts are configured through environment-backed server settings. Default development
values are documented in `.env.example` and must not be used in production.

### Docker

```bash
docker compose up --build
```

## Verification commands

```bash
.venv/bin/pytest -q

cd frontend
npm run build

cd ..
.venv/bin/python server/evaluation/content_safety_benchmark.py
.venv/bin/python -c 'import json,sys; sys.path.insert(0,"server"); from evaluation.security_benchmark import evaluate; print(json.dumps(evaluate("data/benchmark/security_adversarial_v2.jsonl"), indent=2))'
```

Create a new append-only endpoint artifact only from a clean, committed implementation:

```bash
.venv/bin/python server/evaluation/endpoint_security_benchmark.py \
  --run-id <unique-run-id>
```

## Main API surfaces

- `POST /auth/login`
- `POST /ask/`
- `POST /ask/stream`
- `GET /users/{username}/usage`
- `GET|PUT /users/{username}/profile`
- `GET|PUT /users/{username}/courses`
- `GET /users/{username}/degree-audit`
- `GET|PUT /users/{username}/schedule`
- `GET /curricula/`
- `GET /courses/`
- Course-review endpoints, when the feature flag is enabled
- Admin-only document ingestion endpoints

User-scoped endpoints require a bearer token and enforce same-user or administrator access.

## Deployment status

AWS deployment is intentionally deferred. The current priority is local correctness, evaluation,
security, and provider comparison. Deployment work must not be activated without a separate,
explicit decision and production-secret review.

## Known limits

- The endpoint security artifact is deliberately small and synthetic.
- Live-provider quality is only valid for the exact preregistered model, prompt, retrieval context,
  and evaluation split.
- The default frontend bundle still triggers Vite's large-chunk warning; route-level code splitting
  remains an optimization opportunity.
- FastAPI startup hooks and some third-party packages emit deprecation warnings; these are tracked
  technical debt, not current test failures.
