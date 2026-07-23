# adviSU — Retrieval-Augmented Academic Advising System

**Sabancı University · CS 455 project**
Team: Mehmet Selman Yılmaz, Korhan Erdoğdu

adviSU is a profile-aware academic advising assistant for Sabancı University. A student sets
their **program + curriculum (admit) term** and **course history**, then asks questions like
*"does CS 455 count as an area elective for me?"* or *"how many credits do I still need?"*.
adviSU answers from the **official degree-requirement data for that exact program and term**,
computes graduation progress **deterministically in code**, and uses an LLM only to explain
the result — never to invent the numbers.

The system deliberately **never mixes requirements across programs or curriculum years**: a CS
2024 student never sees IE requirements, and if the exact official requirement file is missing
it says so instead of guessing.

---

## What's implemented

### Data corpus (in-repo, `data/`)
Scraped from the public Sabancı degree-detail pages and stored as pre-chunked JSONL (one
requirement chunk per line, with a ready `text` field, stable `chunk_id`, and flat metadata).

| Path | What | Size |
|---|---|---|
| `data/degree_requirements/<PROGRAM>/<TERM>.jsonl` | Official degree requirements — **9 majors × 4 admit terms** | 36 files |
| `data/minors/<CODE>/<TERM>.jsonl` | Minor programs — **17 minors × 4 terms** | 68 files |
| `data/curricula/registry.jsonl` | Which program × term curricula exist (drives the UI selector + missing-data checks) | 104 rows |
| `data/course_catalog/current.jsonl` | Searchable course list — **816 distinct courses** (code, title, SU, ECTS, faculty, **+ per-course engineering/basic-science ECTS**) | 816 rows |

- **Majors:** CS (BSCS), IE (BSMS), BIO (BSBIO), ME (BSME), EE (BSEE), PSY (BAPSY),
  ECON (BAECON), DSA (BSDSA), MAT (BSMAT).
- **Admit terms:** Fall 2022-23 (`202201`), 2023-24 (`202301`), 2024-25 (`202401`), 2025-26 (`202501`).
- Each requirement chunk carries `data_role` (`curriculum_requirement` for majors,
  `minor_requirement` for minors), `program`, `curriculum_term`, `requirement_category`,
  `course_id`, `su_credits`, `ects`, `faculty`, and a `document_type`
  (`..._profile` / `..._category_pool` / `..._pool_course` / `..._rule`).
- **Free/area/core classification is NOT stored on courses** — it depends on the student's
  program + admit term and is answered from the degree-requirement data.
- Offline regeneration pipeline lives in `server/scripts/degree_gen/` (see its README).

### Where the data comes from (exact source links)
Everything is scraped from the public Sabancı prospective-students site. Program codes:
CS=`BSCS`, IE=`BSMS`, BIO=`BSBIO`, ME=`BSME`, EE=`BSEE`, PSY=`BAPSY`, ECON=`BAECON`, DSA=`BSDSA`,
MAT=`BSMAT`. Terms: `202201 202301 202401 202501`.

1. **Major requirements (main page):** `.../degree-detail?SU_DEGREE.p_degree_detail?P_TERM=<TERM>&P_PROGRAM=<CODE>&P_LANG=EN&P_LEVEL=UG` — totals, University + Required lists, category minimums, pool links.
2. **Elective / faculty pools:** `.../degree-detail?SU_DEGREE.p_list_courses?P_TERM=<TERM>&P_AREA=<AREA>&P_PROGRAM=<CODE>&P_LANG=EN&P_LEVEL=UG` where `<AREA>` = `<CODE>_CEL/AEL/FEL` or `FC_FENS|FC_FASS|FC_SOM` (EE uses `_ARE/_FRE`, PSY uses `_COR/ARE/FRE`).
3. **Minor list:** `https://suis.sabanciuniv.edu/prod/SU_DEGREE.p_list_degree?P_LEVEL=UG&P_LANG=EN&P_PRG_TYPE=MINOR`; minor detail = the same `p_degree_detail` with `P_PROGRAM=<CODE>-MINOR`.
4. **Per-course engineering / basic-science ECTS:** each course's catalog page (linked from every pool row) — `.../degree-detail?sabanci_www.p_get_courses?levl_code=UG&subj_code=<SUBJ>&crse_numb=<NUM>&lang=eng`, which shows e.g. `6 ECTS (ENGINEERING:0 / BASIC:6)`. Scraped by `npm run scrape:credits` (816/816 filled).

### Vector store (ChromaDB)
The whole corpus is embedded into a single collection **`su_knowledge`** (~30,343 vectors:
28,082 degree-requirement + 2,261 minor chunks). Ingested by
`server/scripts/ingest_degree_requirements.py` (reads the pre-chunked JSONL directly,
scalarizes metadata, upserts). Chroma is a reproducible index, not the source of truth.

### Source of truth (MongoDB, db `advisu`)
- `users` — includes each student's `academic_profile` (major, degree_code, curriculum_term,
  minor_codes). `curriculum_term` is the admit term — the graduation contract that both retrieval
  scoping and the degree audit key off.
- `courses` — the ~816-course catalog (seeded from `course_catalog/current.jsonl`) that the
  course-history picker searches. Fields include `su_credits`, `ects`, and
  `engineering_ects` / `basic_science_ects` slots (see *Known limitations*).
- `user_courses` — the student's course history **with per-course status**
  (`completed` / `enrolled` / `failed` / `withdrawn` / `transfer` / `exempted`); only
  credit-eligible statuses count toward graduation.
- `conversations` — bounded session memory (last few turns) for follow-up questions.
- Plus the existing ingestion/source-of-truth collections (`sourceDocuments`,
  `ingestionJobs`, `uploadBatches`, `instructorReviews`, `exams`, `embeddingCache`).
- The Mongo client connects **lazily**, so the backend boots even if Mongo is briefly
  unreachable — only DB-backed calls fail until it's up.

### Backend intelligence
- **Profile-aware retrieval** (`server/modules/retrieval_policy.py`): each intent maps to an
  allowed `data_role` + required profile fields, and retrieval is hard-filtered by
  data_role / program / curriculum_term **before** reranking, so results never cross program
  or curriculum boundaries.
- **Missing-data safety**: an authoritative audit is only attempted when the required profile
  fields exist AND the exact official file is in the registry; otherwise adviSU returns a safe
  "authoritative audit unavailable" message.
- **Deterministic degree audit** (`server/modules/degree_audit.py`): SU-credit category
  allocation (university / required / core / area / free) with overflow (extra core → area →
  free), choice-pool handling (e.g. MATH 201 *or* MATH 212), and missing-required detection —
  all computed in code from the exact requirement file + the student's completed courses.
- **Hybrid retrieval** (`server/modules/bm25_retriever.py`): dependency-free BM25 runs over the
  same profile-scoped subset as vector search and is fused into ranking, sharpening exact
  matches like "CS 455" / "202401".
- **Conversation memory** (`server/modules/conversation_memory.py`): resolves follow-ups like
  *"can I take it next semester?"* to the course from the previous turn. Never overrides
  official data.
- **Curriculum registry** (`server/modules/curriculum_registry.py`): read-only access to
  `registry.jsonl`.
- **Selectable retrieval modes** (`server/modules/retrieval_modes.py`): the same question can be
  answered under `llm_only` / `bm25` / `dense` / `hybrid` / `hybrid_rerank`, so the evaluation can
  measure what each component contributes. Every RAG mode receives the *same* profile-scoped
  metadata filter — ablating a retriever changes ranking, never the corpus a student can see.
  The **product** always runs one mode and shows no picker: `DEFAULT_RETRIEVAL_MODE=hybrid`,
  chosen because that is what our own benchmark measured as best (see *Evaluation*).
- **Language consistency** (`llm.detect_language`): the reply language is decided in code from the
  question and injected as a top-priority directive. The system prompt is written in Turkish, which
  used to pull English questions into Turkish answers; a code-decided directive fixes that.
- **Chat history** (`server/modules/conversation_memory.py`): a durable `messages[]` transcript with
  a title per session, stored *alongside* the capped 5-turn `recent_turns` used for prompt working
  memory. The two are separate on purpose — one is browsable history, the other must stay bounded.
- Existing document RAG (uploads, exams, WhatsApp instructor reviews) is preserved.

### Frontend (React + Vite + TS + Tailwind)

Two deliberate visual registers: **dark chrome** for login and chat, **light Sabancı palette**
(`#F5F8FC` page, `#004B93` header, white cards) for the two setup pages, so they read as one section.

| Route | Screen |
|---|---|
| `/login` | Split layout — darkened campus photo, reversed logo, a word-by-word typewriter tagline, and example questions set as quotations. Sign-in panel on the right. |
| `/` | Chat. Sidebar is thread-based (New chat, titled history, delete). Empty state is spare: *"Are we graduating?"*. Gold banner prompts for a profile when one isn't set. |
| `/courses` | **Course History** — saved completed courses with a running SU total (so you can see what the system thinks you finished), plus the course picker and document upload. |
| `/profile` | **Profile & Degree Audit** — pick major + curriculum term from the live registry, save, and run the deterministic audit (per-category table, Eng/Basic ECTS, missing required). |

**Logo assets.** `adviSU-logo.png` is `big.png` trimmed of transparent padding;
`adviSU-logo-reversed.png` is a **knockout variant for dark backgrounds** — the lockup's descriptor
type and "SU" glyphs are navy and disappear on dark, so blue-dominant dark pixels are remapped to
white while the gold/teal accents are kept. Use the reversed variant on dark surfaces rather than
placing the logo on a white plate.

**No retrieval-mode picker.** Choosing a retrieval strategy isn't a student's job; the server runs
the measured-best configuration.

---

## Requirements pipeline (how a question is answered)

```text
question (+ username, session_id)
  -> resolve "this course" via session memory
  -> intent detection (TF-IDF/LogReg) + regex route guards  (+ new "minor" intent)
  -> load student academic_profile from MongoDB
  -> retrieval policy: allowed data_role + required profile fields
  -> missing-profile / missing-file gate  ->  safe limitation if not satisfiable
  -> profile-scoped retrieval (data_role + program + curriculum_term)  [vector + BM25 hybrid]
  -> CrossEncoder rerank
  -> for graduation questions: run deterministic degree_audit and inject it as an
     authoritative context source
  -> LLM (Groq Llama) explains the grounded evidence + audit  ->  answer + sources + intent
```

**What the LLM actually receives.** The prompt context is assembled from: (a) the retrieved
degree-requirement chunks from ChromaDB, (b) a `[Source: MongoDB student profile]` block — the
student's completed-course list and credit totals in natural language, and (c) a
`[Source: Deterministic degree audit]` block — the audit JSON computed in code from the MongoDB
course history. So the student's **derived academic data from MongoDB is passed to the LLM as
context**, and the model is instructed to *explain* those authoritative numbers, not recompute or
contradict them. The raw MongoDB connection string / credentials are never sent — only the derived
per-student facts.

---

## Setup

### 1. Backend
```powershell
cd server
python -m venv myenv
.\myenv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m uvicorn main:app --reload   # http://127.0.0.1:8000  (health: /test)
```

### 2. Environment (`.env` in the project root)
The backend reads a single `.env` (there is no committed `.env.example`; `.env` is gitignored).

```env
# Required for chat answers:
GROQ_API_KEY=your_groq_api_key
# Required for profile / course history / audit:
MONGO_URI=mongodb+srv://<user>:<pass>@<cluster>.mongodb.net/?retryWrites=true&w=majority
MONGO_DB_NAME=advisu

# Sensible defaults (override only if needed):
GROQ_MODEL_NAME=llama-3.3-70b-versatile
CHROMA_PERSIST_DIR=./chroma_store
CHROMA_COLLECTION_NAME=su_knowledge
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L12-v2
CROSS_ENCODER_MODEL_NAME=cross-encoder/ms-marco-MiniLM-L-6-v2
RETRIEVAL_CANDIDATE_K=20
RERANK_TOP_K=6
# DEGREE_DATA_DIR  -> leave unset; defaults to <project>/data (absolute)
```
> For MongoDB Atlas: create a DB user and add your IP (or `0.0.0.0/0` for dev) under Network Access.

### 3. Build data + index (one-time, or after regenerating the corpus)
```bash
npm run registry          # build data/curricula/registry.jsonl
npm run catalog           # build data/course_catalog/current.jsonl (816 courses)
npm run ingest:degrees:reset   # embed the corpus into ChromaDB (~30k vectors)
npm run seed:courses      # push the course catalog into MongoDB (search bar)
npm run validate:data     # (optional) validate corpus + coverage report
npm run test:advising     # (optional) run the invariant test suite
```

### 4. Frontend
```powershell
cd frontend
npm install
npm run dev               # http://localhost:5173
```
Override the API base with `frontend/.env` → `VITE_API_URL=http://127.0.0.1:8000`.

### Using it
Log in (`admin` / `admin`) → **Profile & degree audit** in the sidebar → pick your major +
curriculum term → **Save** → add completed courses in the course picker → **Run audit**, or
just ask questions in the chat.

---

## API endpoints (advising)

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/ask/` | Ask a question (form: `question`; optional `username`, `session_id`, `mode`, `top_k`, `prompt_strategy`, `expert_mode`). Posting only `question` uses `DEFAULT_RETRIEVAL_MODE` (`hybrid`). The UI never sends `mode`; the evaluation harness does. |
| `GET`  | `/users/{username}/conversations` | Chat history for the sidebar (newest first, titles only) |
| `GET` / `DELETE` | `/conversations/{session_id}` | Full transcript of one chat / delete it |
| `GET`  | `/curricula/` | All valid curricula (programs, majors, minors) for the UI selector |
| `GET`  | `/curricula/{program}` | Curriculum terms available for a program |
| `GET` / `PUT` | `/users/{username}/profile` | Read / set the student academic profile |
| `GET` / `PUT` | `/users/{username}/courses` | Read / set course history (PUT accepts per-course `statuses`) |
| `GET`  | `/users/{username}/degree-audit` | Deterministic degree audit for the saved profile |
| `GET`  | `/courses/` | Search the course catalog (`search`, `limit`) |
| `POST` | `/auth/login` | Admin login |
| `GET`  | `/test` | Health check |

Legacy document-RAG endpoints are preserved: `POST /upload_documents/`, `POST /upload_pdfs/`,
`POST /admin/whatsapp/upload`, `POST /admin/whatsapp/{batch_id}/confirm`,
`POST /admin/exams/upload`, `DELETE /sources/{source_id}`.

---

## npm scripts

| Script | Purpose |
|---|---|
| `npm run registry` | Rebuild `data/curricula/registry.jsonl` |
| `npm run catalog` | Rebuild `data/course_catalog/current.jsonl` |
| `npm run ingest:degrees[:reset]` | Embed the degree/minor corpus into ChromaDB |
| `npm run seed:courses` | Force-seed the course catalog into MongoDB |
| `npm run validate:data` | Validate corpus + print a coverage report |
| `npm run test:advising` | Run the invariant test suite (18 unit + integration) |
| `npm run ingest:all\|courses\|reviews\|exams` | Legacy document/source ingestion |
| `npm run benchmark:build` | Regenerate `data/benchmark/questions.jsonl` from the corpus |
| `npm run eval` | Run the benchmark across all retrieval modes (generates answers) |
| `npm run eval:retrieval` | Same, retrieval metrics only — no LLM calls, no API cost |
| `npm run eval:ablate` | Run the ablation grid from `ablation_configs.yaml` |
| `npm run eval:tables` | Write report-ready CSVs to `outputs/tables/` |

---

## Evaluation

The retrieval and hallucination evaluation lives under `server/evaluation/`.

```bash
npm run benchmark:build   # derive the benchmark from data/ (never hand-written answers)
npm run eval:retrieval    # Recall@k / MRR@k / nDCG@k per mode
npm run eval              # + generated answers (needs Groq quota)
npm run eval:ablate       # mode x top_k grid
npm run eval:tables       # outputs/tables/*.csv + outputs/failure_analysis/failure_cases.csv
```

**Ground truth is derived, not written.** `build_benchmark.py` reads real rows out of
`data/degree_requirements/**`, so each question's gold `chunk_id` and reference answer come from
the corpus itself and cannot drift from it. The builder aborts if any gold id is missing.

Measured results, the full method, and the caveats are in **[`docs/experiment_log.md`](docs/experiment_log.md)**.
Two findings worth flagging here:

- **Retrieval materially reduces hallucination.** With BM25 retrieval the system reproduced the
  correct official number in 93% of answers and declined 67% of unsupported questions; the same
  model with no retrieval (`llm_only`) scored 27% and **refused nothing at all**.
- **The CrossEncoder reranker hurts recall on this corpus** (Recall@6 0.44 vs 0.62 for plain
  hybrid) and adds ~295 ms per query. Reproduced at every top-k. **This measurement changed the
  product:** the default is now `hybrid`, not `hybrid_rerank`. Set
  `DEFAULT_RETRIEVAL_MODE=hybrid_rerank` to revert. Caveat: the benchmark is 16 answerable
  questions with templated wording — a real but small sample.

Answer-quality columns (`answer_correctness`, `faithfulness`, `citation_correctness`,
`hallucination_rate`) are intentionally **left empty** for manual labelling — do not quote answer
quality beyond the objective reference-number signal until they are filled in.

---

## Known limitations / next work

- **Course-status entry UI**: the backend supports failed/enrolled/transfer/exempted, but the
  Course History page currently sends plain completed IDs.
- **No long-conversation compaction**: chat history is stored in full, but only the last 5 turns
  feed the prompt as working memory.
- `frontend/src/pages/SignupPage.tsx` is **dead code** — `/signup` redirects to `/login` and
  nothing imports it. Delete it or implement real signup.
- **Schedules / program-term classifications** aren't ingested, so full course *recommendation*
  (offerings, instructors, timetable) is limited.
- **Not modeled yet**: GPA checks, course equivalencies, program transitions / double majors,
  transfer/exemption rules.
- Auth is lightweight (visual login for local development).
- **Section 4 (prompts module, expert routing, few-shot) is not implemented**, so
  `prompt_strategy` / `expert_mode` are accepted by `/ask/` but currently inert.
- **Answer-side evaluation is incomplete**: Groq's free tier caps usage at 100k tokens/day and the
  full 5-mode sweep exceeds it. Run one or two modes per day, or use a paid tier.
- **Cross-lingual retrieval is a measured weak point**: the corpus is English while the product
  answers in Turkish, and a Turkish question can miss a chunk its English twin retrieves.

---

## Project structure (advising additions)

```text
data/
  degree_requirements/<PROGRAM>/<TERM>.jsonl   # 9 majors x 4 terms
  minors/<CODE>/<TERM>.jsonl                   # 17 minors x 4 terms
  curricula/registry.jsonl
  course_catalog/current.jsonl
server/
  modules/
    curriculum_registry.py    retrieval_policy.py    degree_audit.py
    bm25_retriever.py         conversation_memory.py
    catalog_retriever.py  (hybrid + profile filters)  mongodb.py  (profile + course status)
  scripts/
    ingest_degree_requirements.py   build_curricula_registry.py   build_course_catalog.py
    seed_courses.py                 validate_degree_data.py
    degree_gen/                     # offline corpus regeneration
  evaluation/
    build_benchmark.py              # derives the benchmark from data/
    metrics.py                      # Recall@k / MRR@k / nDCG@k
    run_evaluation.py               # question x mode runner
    ablation_runner.py              # ablation grid
    ablation_configs.yaml
    make_tables.py                  # report-ready CSVs + failure analysis
  tests/test_advising.py
data/benchmark/questions.jsonl      # generated, committed
outputs/
  evaluation_runs/<timestamp>/      # results.jsonl, summary_metrics.json, run_config.json
  tables/                           # retrieval/answer/efficiency/failure/ablation CSVs
  failure_analysis/failure_cases.csv
docs/
  experiment_log.md  prompt_log.md  demo_script.md
frontend/
  public/assets/
    adviSU-logo.png                 # trimmed lockup (light grounds)
    adviSU-logo-reversed.png        # knockout variant (dark grounds)
    big.png  campus.jpg
  src/
    pages/{Login,Chat,Courses,Profile}Page.tsx
    components/chat/{Sidebar,ChatHeader,ChatMessages,ChatInput}.tsx
    lib/{api,sample-questions,utils}.ts
```
