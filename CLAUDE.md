# CLAUDE.md — adviSU Implementation Guide

> **CURRENT STATE:** the project is now **adviSU — Retrieval-Augmented Academic Advising
> System**. For what is actually built and running (data corpus, ChromaDB, MongoDB,
> profile-aware retrieval, deterministic degree audit, endpoints, frontend), read
> **"adviSU — Final Implemented State (2026-07-20)"** further down. Everything above that
> section (original SU-GPT name, section-by-section roadmap, `documentType`-only model) is
> historical context — keep it, but the Final Implemented State section is authoritative.

# Project Name

adviSU — Retrieval-Augmented Academic Advising System (Sabancı University).
Originally: "SU-GPT: An Evaluation of Retrieval, Hallucination and Efficiency for a
Course-Aware RAG Assistant".

# Team

- Mehmet Selman Yilmaz
- Korhan Erdogdu

# Track

CS 455

---

# 0. Read This First

You are working inside an existing project, not starting from scratch. The project was originally RagBot 2.0 and is now SU-GPT.

## Current State (updated 2026-07-22)

**Implemented**
- Sections 1, 2, **3, 5, 6 and 7** are complete. See each section's own Status block for detail.
- A **product/UX round** followed the evaluation work: new brand assets + design system, redesigned
  login and chat, a Course History page, chat history, language consistency, and removal of the
  retrieval picker. **See "I. Product / UX round" — it is the authoritative description of the
  current UI and supersedes older frontend notes in this file.**
- Section 4 has a partial bleed-through: the LLM prompt has been softened, made citation-aware, and the `/ask/` handler now passes source-labeled context to the model. The dedicated `prompts.py` module, query router, and few-shot/expert prompts are NOT yet implemented — those still belong to Section 4 proper.
- Retrieval pipeline was tuned (CHUNK_SIZE=1000, CHUNK_OVERLAP=150, RETRIEVAL_CANDIDATE_K=20, RERANK_TOP_K=6, source-labeled context).
- Frontend was migrated from Streamlit (deleted) to a React + Vite + shadcn/ui + Tailwind app under `frontend/`. A visual-only login/signup gate (localStorage-backed) exists at `/login` and `/signup`; the main chat lives at `/`.
- Beyond the numbered sections, the whole **adviSU advising system** was built — see "adviSU — Final Implemented State" below, which is the authoritative description.

**Not yet implemented**
- **Section 4 proper** (prompts module, query router, expert routing, few-shot). `/ask/` accepts `prompt_strategy` and `expert_mode` but they are inert, so no prompt-strategy comparison exists.

**Implemented but incomplete**
- Answer-side evaluation covers only 2 of 5 modes fully (`bm25`, `llm_only`); `dense` 18%, `hybrid_rerank` 5%, `hybrid` 0%. Cause is Groq's 100k tokens/day free-tier cap, not a code defect. Retrieval metrics are complete and reproduce exactly across three runs.
- Section 7.6 (clean-machine reproducibility check) was not performed.

## What the project currently has

- **React frontend** under `frontend/` (Vite + TS + Tailwind + shadcn/ui)
- **FastAPI backend** under `server/`
- `POST /upload_pdfs/` (legacy)
- `POST /upload_documents/` (multi-format: PDF / PPTX / DOCX / MD / TXT)
- `POST /ask/` (dense retrieval + CrossEncoder rerank + source-labeled context + Groq Llama)
- `GET /test` health endpoint
- ChromaDB persistent vector store
- HuggingFace SentenceTransformers embedding model
- CrossEncoder reranking
- Groq-hosted Llama generation via langchain-groq
- Multi-format document loaders + a conservative text cleaner
- Stable chunk IDs with rich metadata (source, document_type, page, slide, section, chunk_id)
- Environment variable configuration through `server/modules/config.py`

## Production RAG Architecture Update (2026-05-20)

This section records the new production-ready RAG backbone without deleting the older course-section roadmap below. Some older checklist items still describe the pre-production architecture; treat this update as the current implemented state for MongoDB/Chroma/ingestion/router behavior.

### Implemented Production Backbone

- MongoDB is now the source of truth for durable application data.
- ChromaDB is now a reproducible vector index, not the canonical data store.
- The active Chroma collection name is `su_knowledge`.
- Data type separation is done through Chroma metadata field `documentType`, not multiple Chroma collections.
- `documentType` values are:
  - `course`
  - `review`
  - `exam`
- Raw uploaded files currently live under local `DOCUMENT_STORAGE_DIR`, but every source document stores a `storageKey` so this can be moved to MinIO/S3 later without changing the ingestion/RAG contract.
- Upload, bulk ingest, PDF/exam ingest, and WhatsApp review ingest now share the same deterministic ingestion path as much as possible.
- `/ask/` now uses an intent-aware RAG router before vector search.
- API responses now include `source_chunk_ids` in addition to human-readable `sources`.

### New/Updated Backend Files

- `server/modules/source_of_truth.py`
  - Sync MongoDB helpers for `sourceDocuments`, `ingestionJobs`, `uploadBatches`, `instructorReviews`, `exams`, and `embeddingCache`.
  - Deterministic helpers for `contentHash`, `sourceId`, and `embeddingCache` key shape.
- `server/modules/load_vectorstore.py`
  - Unified file/text ingestion helpers.
  - Deterministic chunk IDs.
  - Chroma upsert instead of append-only writes.
  - `documentType` metadata normalization.
- `server/modules/rag_router.py`
  - Routes review/exam/graduation/recommendation/general queries to the correct metadata filter.
- `server/modules/file_lifecycle.py`
  - WhatsApp pending upload and approval flow.
  - Exam/PDF upload flow.
  - Cascade source deletion across MongoDB, file storage, and ChromaDB.
- `server/scripts/bulk_ingest.py`
  - Idempotent CLI for all/course/review/exam ingest.
- Root `package.json`
  - Adds `npm run ingest:all`, `npm run ingest:courses`, `npm run ingest:reviews`, and `npm run ingest:exams`.

### MongoDB Source-of-Truth Collections

The following collections are part of the production model:

- `users`
- `courses`
- `instructorReviews`
- `exams`
- `uploadBatches`
- `sourceDocuments`
- `ingestionJobs`
- `embeddingCache`

Important `sourceDocuments` fields:

```json
{
  "sourceId": "exam:9f0f8d1d0b77c0b7a8f2a41b",
  "type": "exam",
  "fileName": "CS412-final-2024.pdf",
  "storageKey": "server/uploaded_documents/exams/9f0f8d1d0b77c0b7-CS412-final-2024.pdf",
  "contentHash": "9f0f8d1d0b77c0b7a8f2a41b4e...",
  "status": "indexed",
  "createdBy": "admin",
  "chunksCreated": 14
}
```

Important `ingestionJobs` fields:

```json
{
  "jobId": "exam:9f0f8d1d0b77c0b7a8f2a41b:20260520001340000123",
  "sourceId": "exam:9f0f8d1d0b77c0b7a8f2a41b",
  "status": "indexed",
  "chunksCreated": 14,
  "error": "",
  "startedAt": "2026-05-20T00:13:40Z",
  "finishedAt": "2026-05-20T00:13:48Z"
}
```

Embedding cache keys should use:

```text
provider_model_textHash
```

Example:

```text
hf_sentence-transformers/all-MiniLM-L12-v2_2cf24dba5fb0a30e26e83b2ac5b9e29e1b161...
```

### Deterministic Chunk and Chroma Shape

Chunk IDs use:

```text
sourceId:chunkIndex:contentHashPrefix
```

Example deterministic chunk ID:

```text
review:1e0f3b1c9d4a5e6f7a8b9c0d:3:a7719f2db2e4c19a
```

Example ChromaDB upsert payload:

```json
{
  "collection": "su_knowledge",
  "id": "review:1e0f3b1c9d4a5e6f7a8b9c0d:3:a7719f2db2e4c19a",
  "document": "Yücel hoca projelerde zorlayabiliyor ama dersin içeriği faydalı...",
  "metadata": {
    "documentType": "review",
    "document_type": "review",
    "sourceId": "review:1e0f3b1c9d4a5e6f7a8b9c0d",
    "contentHash": "a7719f2db2e4c19a7b8f5d7d6c9f0...",
    "chunk_id": "review:1e0f3b1c9d4a5e6f7a8b9c0d:3:a7719f2db2e4c19a",
    "chunkIndex": 3,
    "source": "whatsapp-export.txt",
    "file_name": "whatsapp-export.txt",
    "storageKey": "server/uploaded_documents/whatsapp/1e0f3b1c9d4a5e6f-whatsapp-export.txt",
    "createdBy": "admin",
    "uploadBatchId": "whatsapp:1e0f3b1c9d4a5e6f7a8b9c0d",
    "reviewStatus": "approved"
  }
}
```

Course/exam chunks use the same schema and change only `documentType`, source metadata, and file-specific metadata such as page, section, courseCode, year, semester, or examType.

### Unified Ingestion Flow

```text
source file
-> sourceDocuments / uploadBatches / ingestionJobs in MongoDB
-> local file storage via storageKey (MinIO/S3-ready)
-> normalize
-> PII clean where needed
-> chunk
-> embed
-> Chroma upsert into su_knowledge
-> MongoDB status update
```

Important behavior:

- WhatsApp uploads do not go directly to Chroma.
- WhatsApp exports first become `uploadBatches.status=pending`.
- Admin confirmation updates `instructorReviews` and triggers `documentType=review` ingest.
- Exam/PDF uploads save the file, create/update `sourceDocuments` and `exams`, then ingest as `documentType=exam`.
- Cascade delete soft-deletes the Mongo source, removes the local stored file, and deletes all Chroma chunks with `where={"sourceId": sourceId}`.

### Intent Detection and Ask Pipeline

The classifier in `server/modules/intent_detector.py` is still TF-IDF + Logistic Regression for these 7 base intents:

- `mezuniyet_durumu`
- `ders_onerisi`
- `calisma_plani`
- `major_secimi`
- `alanda_ozellesme`
- `ders_ayrintisi`
- `diger`

The router in `server/modules/rag_router.py` adds deterministic regex guards:

- `Bu hoca nasıl?`, `Yücel Saygın zor mu?` -> `intent=review`, Chroma filter `{documentType: "review"}`
- `CS412 final soruları var mı?` -> `intent=exam`, Chroma filter `{documentType: "exam"}`
- `Mezuniyetime ne kadar kaldı?` -> `intent=mezuniyet_durumu`, Chroma filter `{documentType: "course"}`
- Ders önerisi / NLP / Data / Web / AI style questions -> `intent=ders_onerisi`, Chroma filter `{documentType: "course"}`
- Unclear academic questions can multi-search `course`, `exam`, and `review`, then CrossEncoder reranks the combined candidates.

Current `/ask/` flow:

```text
question
-> get_intent()
-> _resolve_intent()
-> route_query()
-> retrieve_documents(..., metadata_filter=route.metadata_filter)
-> multi-search if route is uncertain
-> rerank_documents()
-> source-labeled context docs
-> get_llm_chain()
-> query_chain()
-> response + sources + source_chunk_ids + intent
```

### Bulk Script Commands

Run from repo root:

```bash
npm run ingest:all
npm run ingest:courses
npm run ingest:reviews
npm run ingest:exams
```

The root scripts call `server/scripts/run_python.sh`, which prefers `server/.venv/bin/python` and falls back to `PYTHON` or `python3`.

### Current Known Gaps / Next Work

- Add real syllabus documents to the corpus.
- Improve final answer quality and prompt behavior for review/exam/course modes.
- Build a benchmark set and collect retrieval/answer metrics.
- Evaluation/benchmark work must not fabricate results. Only report measured results from actual runs.

The goal is to continue transforming SU-GPT section by section.

Do not rebuild the project from scratch.

Preserve the existing working functionality and extend it section by section.

---

## adviSU — Final Implemented State (2026-07-20)

**The project is now `adviSU` — "Retrieval-Augmented Academic Advising System | Sabancı
University".** This is the authoritative description of what is actually built and running.
It supersedes the older `documentType`-only production notes above where they conflict. It
follows `ACADEMIC_ADVISING_DATA_ROADMAP.md`; `GEREKLI_DATALAR.txt` was deleted. The old
document-upload / exam / instructor-review RAG flows are preserved but are not the focus.

### A. Where the data comes from — exact source links

Program-code map: CS=`BSCS`, IE=`BSMS`, BIO=`BSBIO`, ME=`BSME`, EE=`BSEE`, PSY=`BAPSY`,
ECON=`BAECON`, DSA=`BSDSA`, MAT=`BSMAT`. Admit/curriculum terms: `202201 202301 202401 202501`.
Everything is scraped from the public Sabancı prospective-students site.

1. **Major degree requirements (main page)** — totals, University + Required course lists,
   category minimums (SU/ECTS/courses), and links to the elective/faculty pools:
   `https://www.sabanciuniv.edu/en/prospective-students/degree-detail?SU_DEGREE.p_degree_detail?P_TERM=<TERM>&P_PROGRAM=<CODE>&P_SUBMIT=&P_LANG=EN&P_LEVEL=UG`
2. **Elective / faculty course pools** — the enumerated course lists behind "Click for …":
   `...?SU_DEGREE.p_list_courses?P_TERM=<TERM>&P_AREA=<AREA>&P_PROGRAM=<CODE>&P_LANG=EN&P_LEVEL=UG`
   where `<AREA>` = `<CODE>_CEL` (core), `<CODE>_AEL` (area), `<CODE>_FEL` (free), or the faculty
   pools `FC_FENS`(+`&P_FAC=E`) / `FC_FASS`(+`&P_FAC=S`) / `FC_SOM`(+`&P_FAC=M`).
   **Pool-code quirks:** EE uses `BSEE_CEL/ARE/FRE`; PSY uses `BAPSY_COR/ARE/FRE`. Each course row
   in these pages is `[marker, code, name, ECTS, SU, faculty]` and the **code links to that
   course's catalog page** (see #5).
3. **Minor list:** `https://suis.sabanciuniv.edu/prod/SU_DEGREE.p_list_degree?P_LEVEL=UG&P_LANG=EN&P_PRG_TYPE=MINOR`
   → 17 minors with codes like `MATH-MINOR`, `FIN-MINOR`, …
4. **Minor requirements:** the same `p_degree_detail` URL with `P_PROGRAM=<CODE>-MINOR`. Minors list
   their Required + Core/Area courses inline; 3 (ENTREP/MKTG/DECB) put Area electives behind an
   external `p_list_courses` pool (`P_AREA=ENTREP_ARE` / `MKTG_AEL` / `DECB_ARE`).
5. **Per-course engineering / basic-science ECTS** — the split needed for the graduation Eng≥90 /
   Basic-Science≥60 rules. Found on each course's catalog page (the link embedded in every pool row
   in #2), e.g. NS 101 shows `6 ECTS (ENGINEERING:0 / BASIC:6)`, CS 303 shows `ENGINEERING:6 / BASIC:1`:
   `https://www.sabanciuniv.edu/en/prospective-students/degree-detail?sabanci_www.p_get_courses?levl_code=UG&subj_code=<SUBJ>&crse_numb=<NUM>&lang=eng`

### B. Offline generation pipeline (`server/scripts/degree_gen/` + a few scripts)

Run once (or when re-scraping); the running app never calls these. `curl` downloads raw HTML,
Python regex-parses it (WebFetch truncates the big free-elective lists, so raw parse is used).
1. `parse_pools.py <pages_dir> <out.json>` — pool HTML → `{catalog, pools}`.
2. `gen_degree_reqs.py <config_<PROG>.json> <parsed.json> <out_dir>` — emits
   `data/degree_requirements/<PROG>/<TERM>.jsonl` (CS used the older `gen_cs_degree_reqs.py`). Each
   `config_<PROG>.json` carries the program's University/Required lists + per-term flags
   (math choices, `pool_suffix`, `hum_min_courses`, `extra_pools`, `faculty_rule`, `eng_ects`/`bsci_ects`).
3. `build_minors.py <minor_pages> data/minors <external_pools>` — minors.
4. `build_curricula_registry.py` (`npm run registry`) → `data/curricula/registry.jsonl`.
5. `build_course_catalog.py` (`npm run catalog`) → `data/course_catalog/current.jsonl` (union of all
   `*_pool_course` rows → 816 courses; eng/basic left null).
6. `scrape_course_credits.py` (`npm run scrape:credits`) — fetches source #5 for all 816 courses
   (urllib + threadpool, cached in `data/course_catalog/credits_cache.json`), fills
   `engineering_ects`/`basic_science_ects` in `current.jsonl`. **816/816 filled** (211 eng>0, 153 basic>0).

### C. What is stored WHERE (storage map)

**On disk — `data/` (pre-chunked JSONL, the offline output, committed):**
- `degree_requirements/<PROG>/<TERM>.jsonl` — 9 majors × 4 terms = **36 files**, `data_role="curriculum_requirement"`.
- `minors/<CODE>/<TERM>.jsonl` — 17 minors × 4 terms = **68 files**, `data_role="minor_requirement"`.
- `curricula/registry.jsonl` — 104 rows (which program×term exist).
- `course_catalog/current.jsonl` — **816 courses** (code/title/su_credits/ects/faculty/engineering_ects/basic_science_ects).
- Each requirement row is one retrieval chunk: `text` + stable `chunk_id` + flat metadata
  (`program`, `curriculum_term`, `requirement_category`, `course_id`, `su_credits`, `ects`, `faculty`,
  `document_type` ∈ {..._profile, ..._category_pool, ..._pool_course, ..._rule}).

**ChromaDB — collection `su_knowledge` (`server/chroma_store`):** ONLY the requirement/minor
retrieval chunks are embedded here (**~30,343 vectors**). Ingested by
`ingest_degree_requirements.py` (`npm run ingest:degrees[:reset]`), which reads the JSONL directly,
scalarizes metadata (lists→JSON string, None dropped), adds aliases (`documentType`="course"/"minor",
`term_code`=`curriculum_term`), and upserts. **NOT in Chroma:** the course catalog as standalone
docs (its pool_course rows are already in Chroma as requirement chunks), student profiles, course
history — those live only in MongoDB. Chroma is a reproducible index, not the source of truth.

**MongoDB — db `advisu` (source of truth for per-student + catalog data):**
- `users` — each has `academic_profile` (major, degree_code, admission_term, curriculum_term,
  minor_codes, profile_status). **`current_term` was removed 2026-07-22** — it was collected by the
  profile form but never read; retrieval scoping and the audit both key off `curriculum_term`.
- `courses` — the ~816-course catalog seeded from `course_catalog/current.jsonl` (`npm run seed:courses`),
  with `su_credits`, `ects`, `engineering_ects`, `basic_science_ects`, `faculty`. This is what the
  frontend course-picker searches (`/courses/`).
- `user_courses` — the student's course history **with per-course status** (completed/enrolled/
  failed/withdrawn/transfer/exempted); `get_completed_course_codes` returns only credit-eligible
  (completed/transfer/exempted).
- `conversations` — bounded session memory (last 5 turns + working context).
- Legacy source-of-truth collections (`sourceDocuments`, `ingestionJobs`, `uploadBatches`,
  `instructorReviews`, `exams`, `embeddingCache`) still exist for the upload/exam/review flows.
- **Lazy Mongo:** `source_of_truth.py` builds the pymongo client on first use (was constructed at
  import → hung ~20s on an unreachable Atlas SRV). App boots in ~12s (model load only).

### D. Runtime `/ask` flow — step by step (and exactly what the LLM receives)

`POST /ask/` (form: `question`, optional `username`, `session_id`) — `server/main.py`:
1. **Session memory:** `conversation_memory.get_working_context(session_id)`; `resolve_reference`
   rewrites "can I take it next semester?" → appends the last course code.
2. **Intent:** `get_intent` (TF-IDF/LogReg) → `_resolve_intent` regex guards (adds `minor`) → `route_query`.
3. **Profile:** `get_academic_profile(username)` from MongoDB (major, curriculum_term, …).
4. **Policy + gate:** `retrieval_policy.build_metadata_filter(intent, profile)` and `check_profile`.
   For an authoritative graduation question with no profile / no official file → return a safe
   limitation message (never a cross-program/term fallback).
5. **Retrieval (profile-scoped, hybrid):** `catalog_retriever.retrieve_documents(vs, query, k,
   metadata_filter={data_role, program, curriculum_term})` — vector (Chroma) + dependency-free BM25
   over the same scoped subset, fused — then `rerank_documents` (CrossEncoder). Result: the exact
   requirement chunks for THIS student's program+term.
6. **Deterministic audit (graduation intents):** `degree_audit.audit(program, curriculum_term,
   get_completed_course_codes(username))` — computed in code (see E). Injected as a context Document
   `[Source: Deterministic degree audit engine (authoritative)]` containing the audit JSON.
7. **Student context Document:** `get_user_course_context(username)` (or a compact recommendation
   variant) is injected as `[Source: MongoDB student profile]` — the student's completed-course list
   and credit totals in natural language.
8. **LLM:** all context Documents (retrieved requirement chunks + the audit JSON block + the MongoDB
   student-profile block + an intent-guidance block) are passed via a `StaticRetriever` into
   `get_llm_chain` (LangChain RetrievalQA, chain_type `stuff`) → Groq Llama. `query_chain` returns
   `{response, sources, source_chunk_ids, intent}`.
9. `conversation_memory.append_turn(...)` stores the turn.

**Do we give MongoDB details to the LLM?** Yes — the student's **derived academic data** (their
`academic_profile` fields, their completed-course list, credit totals, and the deterministic audit,
which itself is computed from the MongoDB course history) are embedded into the prompt as the
`[Source: MongoDB student profile]` and `[Source: Deterministic degree audit]` context blocks.
The system prompt (`modules/llm.py`) tells the model to **explain** these authoritative numbers and
**not recompute or contradict** them. The raw connection string / credentials are NEVER sent — only
the derived per-student facts. So numbers come from code + Mongo; the LLM only phrases the answer.

### E. Deterministic degree audit (`degree_audit.py`)

Reads exactly one official file `degree_requirements/<PROG>/<TERM>.jsonl` + the student's completed
codes. Computes, in code (the LLM never does this):
- **SU categories** — university / required / core / area / free with greedy overflow (extra
  core→area→free), choice pools (e.g. MATH 201 *or* MATH 212 — 212 not flagged missing when 201 done),
  and missing-required detection.
- **Engineering / Basic-Science ECTS** — loads per-course eng/basic ECTS from
  `course_catalog/current.jsonl` (`_course_credit_catalog`) and sums the student's completed courses
  vs the program's Eng≥90 / Basic≥60 requirement → `ects_requirements` (completed/required/remaining).
  Returns `engineering_basic_science: "computed"`.
- Returns `{status, reliability:"authoritative", total_min_su_credits, completed_su_credits,
  categories[], ects_requirements[], missing_required_courses[], warnings[]}`.
  (**Note:** the audit reads eng/basic from `current.jsonl`, not Mongo — restart the backend after a
  re-scrape so the lru_cache reloads.)

### F. Endpoints (advising) + Frontend

New endpoints: `GET/PUT /users/{u}/profile`, `GET /curricula/`, `GET /curricula/{program}`,
`GET /users/{u}/degree-audit`; `PUT /users/{u}/courses` now takes per-course `statuses`.
Chat history (2026-07-22): `GET /users/{u}/conversations`, `GET /conversations/{id}`,
`DELETE /conversations/{id}`. **17 paths / 24 routes total.**
Frontend: see **section I** below for the current UI (the 2026-07-22 UX round superseded the
earlier logo/header/sidebar arrangement described in older revisions of this file). Core pieces:
a Profile & Degree-Audit page (`/profile`) that loads `/curricula/`, saves the profile and renders
the audit (SU category table + Eng/Basic ECTS lines + missing required); a Course History page
(`/courses`); the chat at `/`; and a `session_id` sent with each `/ask`.

### G. Config / tooling / tests

Single root `.env` (no `.env.example`; `.env` gitignored): `GROQ_API_KEY` (chat) +
`MONGO_URI` / `MONGO_DB_NAME=advisu` (profile/audit). `DEGREE_DATA_DIR` must stay unset (absolute
`<project>/data` default — a relative value resolves to `server/data` and breaks curricula/audit).
`DEFAULT_RETRIEVAL_MODE` defaults to **`hybrid`** (evidence-based, see §I/§15); the UI does not
expose a picker.
npm: `registry`, `catalog`, `scrape:credits`, `ingest:degrees[:reset]`, `seed:courses`,
`validate:data`, `test:advising`, plus the evaluation scripts `benchmark:build`, `eval`,
`eval:retrieval`, `eval:ablate`, `eval:tables`.
`validate_degree_data.py` = validation + coverage report (0 errors).
`server/tests/test_advising.py` = **18** unit + integration invariants (scope isolation,
missing-data gate, audit choices/overflow/eng-basic, bm25, memory, status, and 4 retrieval-mode
invariants: normalization/fallback, llm_only retrieves nothing, hybrid skips rerank but keeps scope
and top_k, bm25 narrowing-gate default) — all pass.

**Python environment on the dev machine is `server/myenv/`** (not `.venv`); `run_python.sh` prefers
`server/.venv/bin/python` so npm scripts may need `PYTHON=server/myenv/Scripts/python.exe`.
On Windows, set `PYTHONIOENCODING=utf-8` before running scripts that print Turkish text, or the
cp1252 console encoder raises `UnicodeEncodeError` (the JSONL files themselves are UTF-8 and fine).

### H. Roadmap status — done vs. NOT done (honest, keep updated)

The **core advising roadmap is implemented**: clean per-program/term data + minors + course
catalog (with eng/basic ECTS), ChromaDB ingestion, MongoDB profile + course-history-with-status,
intent→policy retrieval scoping, missing-data safety, hybrid (vector+BM25) retrieval, deterministic
degree audit (SU categories + engineering/basic-science ECTS), conversation memory, endpoints,
frontend profile+audit page, validation + coverage report, and an automated invariant test suite.

**NOT done / to improve later (roadmap items still open):**

Data:
- **Schedules** (`data/schedules/<TERM>.jsonl`) and **program-term classifications**
  (`data/program_classifications/<TERM>/<PROG>.jsonl`) are NOT ingested → so schedule questions
  (offerings, CRN, instructor, timetable) and full **course recommendation** are not backed by real
  data. `main.py`'s legacy recommendation reads (`CATALOG_DATA_DIR/<term>/CS.jsonl`,
  `.../schedule/…`) find nothing and are effectively dead; recommendation currently leans on the
  requirement corpus only.
- **`data/rules/`** (course_equivalencies, category_allocation, program_transitions) not created.
- Only **9 majors + 17 minors × 4 admit terms**. More programs/terms require adding their config +
  re-running the offline pipeline.

Audit engine (`degree_audit.py`):
- **Course equivalencies / double-count prevention** not applied (needs `rules/course_equivalencies`).
- **Faculty-Courses requirement** (≥5 courses, ≥2 MATH, ≥3 FENS / FASS-based for BA) is described in
  the data as a rule but the audit does NOT verify it yet.
- **GPA checks** (min program/cumulative GPA) — no GPA data collected.
- SU-category allocation is a sound greedy heuristic; it does not solve edge cases optimally
  (a course legal in multiple pools is assigned once, greedily).

Product / UX:
- **Course-status entry UI**: backend accepts completed/failed/enrolled/transfer/exempted, but the
  Course History page still sends plain completed IDs — add status toggles.
- Real login/authorization (current auth is visual/localStorage; admin-only backend).
- ~~Replace the placeholder logo~~ — **done 2026-07-22**, real adviSU logo in place (see §I).
- ~~Chat history~~ — **done 2026-07-22** (§I). Long-conversation summarisation/compaction is still
  not implemented: `recent_turns` stays capped at 5 for prompt context.
- `frontend/src/pages/SignupPage.tsx` is **dead code** — `/signup` redirects to `/login` and
  nothing imports the component. Delete it or wire up real signup; do not "fix" its styling.

Retrieval quality (measured 2026-07-22, see `docs/experiment_log.md`):
- **The CrossEncoder reranker currently HURTS recall on this corpus** (Recall@6 0.438 vs 0.625 for
  plain hybrid, reproduced at top_k 3/5/10) and adds ~295 ms/query. Cause: `ms-marco-MiniLM` scores
  the long enumerated `category_pool` chunks poorly against short questions. Candidate fixes:
  a domain-appropriate reranker, or splitting category-pool chunks so the requirement clause is its
  own chunk. **Deliberately not changed yet** — Sections 5/6 measure and report; changing the
  retrieval stack is a separate product decision.
- **Cross-lingual retrieval gap**: the corpus is English while the product answers in Turkish. A
  Turkish question can fail to retrieve the chunk its English twin retrieves (benchmark q013 vs q006).

Advanced student cases (roadmap Phase 7): double major, minor advising audit, transfer/exemption
handling, program transitions, private per-student degree-evaluation uploads with access control.

Conversation memory: keeps the last 5 turns; no long-conversation summarization/compaction yet.

---

### I. Product / UX round (2026-07-22) — current UI, supersedes older frontend notes

**Brand assets.** `frontend/public/assets/`:
- `big.png` — the source lockup as supplied (kept, untouched).
- `adviSU-logo.png` — `big.png` trimmed of transparent padding (only **30%** of the source canvas
  was non-transparent, so a raw swap rendered at half size). Regenerate with PIL `getbbox()` + 2% margin.
- `adviSU-logo-reversed.png` — **knockout variant for dark grounds**, generated from
  `adviSU-logo.png` by remapping blue-dominant dark pixels (66% of the mark, ~#1E3060) to white
  while leaving the gold/teal accents alone; alpha preserved so edges stay clean. This exists
  because the lockup's descriptor type and "SU" glyphs are navy and vanish on dark. **Use it on
  every dark surface — do NOT put the logo on a white plate/rectangle** (tried, looked cheap).
- `campus.jpg` — login background. `sugptlogo.png` / `sabanci_logo.png` deleted.

**Design system.** Two deliberate registers:
- **Dark chrome** (`/login`, chat): `#08152b`/`#0a1830` surfaces, `white/10` hairlines,
  `sabanci-light` muted text, mono uppercase eyebrows (`font-mono text-[11px] uppercase
  tracking-[0.22em]`), gold (`#D6A13A`) as the single accent (caret, focus rings, icons).
- **Light setup pages** (`/profile`, `/courses`): `#F5F8FC` page, `#004B93` header bar, white cards
  on `#D8E6F3` borders, `#003B73` headings, `#4A5568` muted. Both use the same tokens so they read
  as one section.

**Screens.**
- `/login` — vertical split. Left: darkened `campus.jpg`, reversed lockup, a word-by-word
  typewriter tagline cycling 3 lines (respects `prefers-reduced-motion`), and 5 example questions
  set as quotations with a gold hanging quote mark. Right: navy sign-in panel. Copy lives in
  `src/lib/sample-questions.ts` (shared, so screens can't drift).
- `/` chat — sidebar is thread-based (New chat, titled history with relative times, delete);
  header carries **no logo and no retrieval picker**; empty state is spare ("Are we graduating?"
  + one explanatory line). Uploads and course picking moved OUT of the sidebar.
- `/courses` — Course History. Left panel lists **saved completed courses with a running SU total**
  (the point of the page: see what the system thinks you finished); right panel is the picker plus
  document upload.
- `/profile` — its "add courses in the chat sidebar" copy now links to `/courses`. The
  **"Current academic term" selector was removed**: `current_term` was stored but never read by
  anything (retrieval scoping and `degree_audit` both use `curriculum_term`, the admit term). A
  control that changes nothing is worse than no control. Dropped from `AcademicProfilePayload`,
  `DEFAULT_ACADEMIC_PROFILE` and the `AcademicProfile` TS interface.
  **`get_academic_profile` now filters reads to the known keys.** Removing the field from the
  default dict was not enough: the old code did `dict(DEFAULT).update(stored)`, so a user document
  written before the removal still echoed `current_term` straight back out of `GET
  /users/{u}/profile` — verified live, then fixed. Stored values are left in Mongo untouched;
  they are simply no longer served.

**Behaviour changes.**
- **Language consistency** — `llm.detect_language()` decides `tr`/`en` in code and
  `_LANGUAGE_DIRECTIVE` is injected at the top AND bottom of the prompt. The prompt already had a
  "answer in the user's language" rule, but the whole system prompt is Turkish, which biased the
  model to Turkish for English questions. A rule buried in Turkish prose does not win; a
  code-decided directive does. Verified live both ways.
- **Chat history** — `conversation_memory` now stores a durable `messages[]` transcript + `title` +
  `updatedAt` **alongside** the capped `recent_turns` (5). They are deliberately separate:
  `recent_turns` is prompt working memory and must stay bounded; `messages` is what the user browses.
- **Retrieval picker removed from the UI** and `DEFAULT_RETRIEVAL_MODE` changed
  `hybrid_rerank` → **`hybrid`** — see §12/§15: our own benchmark measured hybrid better
  (Recall@6 0.625 vs 0.438) *and* ~283 ms faster. Modes still exist on `/ask/` for the evaluation
  harness. Revert with `DEFAULT_RETRIEVAL_MODE=hybrid_rerank`.
- **Profile gate banner** — chat shows a gold banner linking to `/profile` when major/curriculum
  term is unset.
- **Provider errors sanitised** — a Groq 429 used to render the raw payload (including the
  organisation id) in the chat bubble; `/ask/` now returns a plain message and logs the detail.
- **Auth deep-link fix** — `AuthContext` reads localStorage during the first render. Loading it in
  a `useEffect` made the initial render unauthenticated, so refreshing on `/profile` bounced to
  `/login` → `/`, silently losing the route.

**Deleted:** `components/chat/CourseSelector.tsx` (superseded by `/courses`).

---

CS 455 *evaluation* track (sections 3, 5, 6, 7) — **implemented 2026-07-22.** Selectable retrieval
modes, a corpus-derived benchmark, Recall@k/MRR/nDCG, efficiency logging, ablations, failure
analysis, report tables, and the README/prompt/experiment/demo docs all exist and have been run.
See sections 12, 14, 15, 16 and `docs/experiment_log.md` for measured numbers. **Section 4
(prompts module, query router, few-shot, expert routing) is still NOT implemented** — `/ask/`
accepts `prompt_strategy` and `expert_mode` but they are inert, so no prompt-strategy comparison
exists yet. Answer-side evaluation is incomplete for 3 of 5 modes because Groq's free tier caps
usage at 100k tokens/day. Do not fabricate results.

---

# 1. Critical Incremental Implementation Rule

This file is divided into implementation sections.

You must only implement the section explicitly requested by the user.

Examples:

- If the user says: Finish Section 1
  - Implement only Section 1.
  - Do not implement Section 2, Section 3, or later sections.

- If the user says: Finish Section 2
  - Implement only Section 2.
  - Preserve Section 1.
  - Do not implement Section 3 or later sections.

- If the user says: Finish Section 4
  - Implement only Section 4.
  - Make only the smallest necessary compatibility changes to previous sections.

Never jump ahead.

Never implement all sections at once.

Always keep the project runnable after every section.

After completing a section, summarize:

- what files were changed
- what functionality was added
- how to run or test it
- any assumptions made
- any TODOs left for future sections

---

# 2. High-Level Project Goal

SU-GPT is a course-aware academic RAG assistant for Sabanci University course-related documents and selected LLM course materials.

The system should answer questions over documents such as:

- course syllabi
- course catalog descriptions
- lecture slides
- homework descriptions
- project announcements
- Markdown notes
- plain text notes
- selected LLM/NLP course materials

The system must generate answers grounded in retrieved evidence.

The project is not only a chatbot.

It must also support empirical evaluation of different retrieval and generation configurations.

Main configurations to support eventually:

- LLM-only baseline
- BM25 RAG
- Dense retrieval RAG
- Hybrid RAG
- Hybrid RAG with CrossEncoder reranking

Main evaluation dimensions to support eventually:

- retrieval quality
- answer correctness
- citation correctness
- faithfulness
- hallucination rate
- failure cases
- latency
- prompt token length
- retrieved chunk count
- estimated API cost

---

# 3. Current Codebase Assumptions

Actual current structure (after Section 1, Section 2, and the React migration):

- RagBot-2.0-main/
  - frontend/                       # React + Vite + TS + Tailwind + shadcn/ui
    - public/assets/                # campus.jpg, sabanci_logo.png, sugptlogo.png
    - src/
      - components/ui/              # Button, Input, Label, Card, Checkbox, Separator
      - components/chat/            # Sidebar, ChatHeader, ChatMessages, ChatInput
      - contexts/AuthContext.tsx    # localStorage-backed visual auth
      - lib/{api,utils}.ts          # FastAPI client + cn() helper
      - pages/{Login,Signup,Chat}Page.tsx
      - App.tsx                     # Router + auth guards
      - main.tsx
      - index.css                   # Tailwind base + theme tokens
    - package.json, vite.config.ts, tailwind.config.js, tsconfig.json
  - server/
    - main.py                       # FastAPI app + /upload_pdfs /upload_documents /ask /test
    - logger.py
    - modules/
      - config.py                   # GROQ_*, CHROMA_*, EMBEDDING_*, CHUNK_*, *_K
      - document_loaders.py         # PDF/PPTX/DOCX/MD/TXT dispatch
      - document_cleaner.py         # whitespace/unicode normalization
      - load_vectorstore.py         # ingest + chunk + Chroma upsert
      - reranker.py                 # CrossEncoder
      - llm.py                      # PromptTemplate + RetrievalQA (Groq Llama)
      - query_handlers.py           # source-formatted response
    - requirements.txt
    - uploaded_pdfs/                # legacy upload dir (PDFs from /upload_pdfs/)
    - uploaded_documents/           # multi-format upload dir (from /upload_documents/)
    - chroma_store/                 # persistent vector store

There is no longer a Streamlit `client/` directory — it was deleted after the React migration.

Before modifying code:

- inspect the folder structure
- identify the actual React entrypoints (`frontend/src/main.tsx`, `App.tsx`)
- identify the actual FastAPI entrypoint (`server/main.py`)
- identify the existing upload logic (`server/modules/load_vectorstore.py`)
- identify the existing retrieval, reranking, and generation logic
- identify the existing prompt template (`server/modules/llm.py`)
- identify the existing requirements/dependency files (`server/requirements.txt`, `frontend/package.json`)

Then make the smallest safe changes required by the requested section.

---

# 4. Existing Technology Stack

Preserve and reuse the existing stack unless a section explicitly asks for additions.

Current stack:

- Backend language: Python (3.10+)
- Backend: FastAPI + Uvicorn
- Frontend: React 18 + Vite + TypeScript (no longer Streamlit)
- Frontend styling: Tailwind CSS + shadcn/ui primitives (Button, Input, Label, Card, Checkbox, Separator)
- Frontend routing: React Router v6 with `RequireAuth` / `RedirectIfAuthed` guards
- Frontend auth state: localStorage-backed `AuthContext` (visual-only, no backend auth yet)
- Frontend HTTP: `fetch` against the FastAPI base URL (`VITE_API_URL`, default `http://127.0.0.1:8000`)
- Toasts: sonner; Icons: lucide-react
- RAG framework: LangChain
- Vector database: ChromaDB (persistent)
- Embeddings: HuggingFace SentenceTransformers
- Default embedding model: sentence-transformers/all-MiniLM-L12-v2
- Reranking: SentenceTransformers CrossEncoder
- Default reranker: cross-encoder/ms-marco-MiniLM-L-6-v2
- LLM provider: GroqCloud / Groq API
- Default LLM: llama-3.3-70b-versatile
- Document parsing: PyPDFLoader / PyPDF (PDF), python-pptx (PPTX, Section 2), python-docx (DOCX, Section 2), plain UTF-8 for MD and TXT
- Environment management: python-dotenv
- File upload support: python-multipart and aiofiles

Already added in Section 2:

- python-pptx
- python-docx

Potential future additions:

- rank-bm25 (Section 3)
- tiktoken, optional (Section 6)
- pandas, optional (Sections 5-6)
- numpy, optional (Sections 5-6)
- pyyaml, optional (Section 6)
- ragas, optional (Section 5)

Add dependencies only in the section that needs them.

---

# 5. Sabanci University Visual Theme

The UI should be visually aligned with Sabanci University colors.

Use a clean academic design.

Primary colors:

- Sabanci Blue: #004B93
- Dark Navy: #003B73
- White: #FFFFFF
- Light Background: #F5F8FC
- Soft Border Blue: #D8E6F3
- Muted Text: #4A5568
- Optional Gold Accent: #D6A13A

Design principles:

- blue and white dominant theme
- professional academic look
- clean card-based layout
- avoid clutter
- clear source and citation cards
- clear retrieval and evaluation mode indicators
- avoid old medical chatbot identity
- avoid MediBot wording
- avoid medical advice wording
- use SU-GPT everywhere visible

Good UI wording:

- SU-GPT
- Course-Aware RAG Assistant for Sabanci University Materials
- Ask questions over syllabi, lecture slides, course documents, and project materials.

Bad UI wording:

- MediBot
- Medical PDF Assistant
- Ask your medical questions

---

# 6. Safety and Grounding Rules

SU-GPT must not hallucinate university-specific facts.

The assistant must not invent:

- course policies
- grading rules
- attendance rules
- deadlines
- prerequisites
- homework requirements
- project requirements
- exam rules
- instructor statements
- private institutional information

If retrieved context is insufficient, answer with:

The uploaded documents do not provide enough information to answer this question.

or a close equivalent.

The model may explain general concepts such as Word2Vec, BERT, GPT, attention, embeddings, or RAG only when the retrieved course material supports the explanation, unless the selected mode is explicitly LLM-only.

For RAG modes, answers must be grounded in retrieved evidence.

---

# 7. Advanced Prompting Techniques

These techniques should be implemented carefully and practically.

Do not claim that the project trains a real Chain-of-Thought model or a real neural Mixture-of-Experts model.

Use these as prompting and routing strategies.

## 7.1 Structured Reasoning / Chain-of-Thought Inspired Grounding

Use chain-of-thought inspired prompting internally, but do not expose long hidden reasoning to the user.

The final answer should not include long reasoning traces such as:

- Let me think step by step...
- First I will analyze...
- My reasoning is...

Instead, prompts may instruct the model to silently check:

- Does the provided context contain enough evidence?
- Which retrieved chunks directly support the answer?
- Are there unsupported claims that should be avoided?
- Should the system answer or refuse due to insufficient context?

The visible final answer should be concise, grounded, and citation-aware.

## 7.2 Few-Shot Prompting

Few-shot prompting may be added in Section 4.

Use few-shot examples to teach the model:

- how to answer with citations
- how to refuse unsupported questions
- how to answer Turkish questions
- how to avoid unsupported Sabanci-specific claims
- how to separate grounded evidence from general explanation

Example supported-answer behavior:

Question: What does the syllabus say about AI usage?

Context: Source is CS455_555 Syllabus.pdf, page 5. Generative AI tools are permitted as coding assistants, but students must submit a Prompt Log and explain how they modified and verified the output.

Answer: The syllabus allows generative AI tools as coding assistants, but students must document their usage with a Prompt Log and be able to explain and verify the code they submit.

Sources:
1. CS455_555 Syllabus.pdf, page 5

Example unsupported-answer behavior:

Question: When is Homework 3 due?

Context: No retrieved chunk contains a Homework 3 deadline.

Answer: The uploaded documents do not provide enough information to answer this question. I could not find a supported Homework 3 deadline in the retrieved context.

Sources:
No supporting source found.

Example Turkish behavior:

Question: Derste yapay zeka kullanımı serbest mi?

Context: Source is CS455_555 Syllabus.pdf, page 5. Generative AI tools are permitted as coding assistants, but students must submit a Prompt Log and explain how they modified and verified the output.

Answer: Evet, syllabus’a göre üretken yapay zeka araçları kodlama asistanı olarak kullanılabilir. Ancak bu kullanım şeffaf şekilde belgelenmeli, Prompt Log sunulmalı ve öğrenci kullandığı kodu açıklayabilmelidir.

Sources:
1. CS455_555 Syllabus.pdf, page 5

## 7.3 Mixture-of-Experts Inspired Query Routing

Do not implement a real neural MoE model.

Implement a lightweight expert-routing layer.

The router should classify the user query and choose a prompt profile or retrieval behavior.

Possible expert types:

- syllabus_policy_expert
- lecture_content_expert
- course_catalog_expert
- homework_project_expert
- concept_explanation_expert
- evaluation_expert
- unanswerable_detector
- general_academic_rag_expert

A simple rule-based router is enough at first.

Example routing idea:

- If the query contains terms such as syllabus, grading, attendance, AI usage, make-up, policy, or academic integrity, route to syllabus_policy_expert.
- If the query contains terms such as BERT, GPT, Word2Vec, attention, RAG, embedding, transformer, reranking, or semantic search, route to lecture_content_expert.
- If the query contains terms such as prerequisite, credits, course code, course title, or requirement, route to course_catalog_expert.
- If the query contains terms such as homework, project, submission, prompt log, experiment log, or deadline, route to homework_project_expert.
- Otherwise, route to general_academic_rag_expert.

Each expert can be a prompt template or prompt prefix, not a separate model.

Example expert prompt prefix for syllabus_policy_expert:

You are using syllabus_policy_expert mode. Focus on course policies, grading rules, attendance, AI usage, make-up rules, and academic integrity. Only answer if the retrieved syllabus context supports the answer. Do not invent policies.

## 7.4 Prompt Strategy Evaluation

In evaluation sections, add prompt strategy comparison if feasible.

Compare:

- basic_grounded_prompt
- few_shot_grounded_prompt
- expert_routed_prompt

Compare these using:

- answer correctness
- faithfulness
- citation correctness
- hallucination rate
- latency
- prompt token length
- estimated cost

Few-shot and expert-routed prompts may improve answer quality but increase prompt length and API cost, so they should be treated as experimental modes.

---

# 8. Environment Variables

Preserve existing environment variables.

Existing variables likely include:

- GROQ_API_KEY
- GROQ_MODEL_NAME, default llama-3.3-70b-versatile
- CHROMA_PERSIST_DIR, default ./chroma_store
- CHROMA_COLLECTION_NAME, default ragbot_documents
- EMBEDDING_MODEL_NAME, default sentence-transformers/all-MiniLM-L12-v2
- CROSS_ENCODER_MODEL_NAME, default cross-encoder/ms-marco-MiniLM-L-6-v2
- RETRIEVAL_CANDIDATE_K, default 10
- RERANK_TOP_K, default 3

Future variables to add only when needed:

- CHUNK_SIZE, default 500
- CHUNK_OVERLAP, default 50
- DEFAULT_RETRIEVAL_MODE, default hybrid_rerank
- DEFAULT_PROMPT_STRATEGY, default basic
- DEFAULT_EXPERT_MODE, default auto
- BM25_TOP_K, default 10
- DENSE_TOP_K, default 10
- HYBRID_TOP_K, default 10
- FINAL_CONTEXT_TOP_K, default 5
- ENABLE_RERANKING, default true
- OPENAI_API_KEY, optional
- OPENAI_MODEL_NAME, optional
- ESTIMATED_INPUT_COST_PER_1K, optional
- ESTIMATED_OUTPUT_COST_PER_1K, optional

Do not require OpenAI if the current project already works with Groq.

The system may support OpenAI or a similar API later, but it must not break Groq-based generation.

---

# 9. File and Folder Guidelines

Prefer modular additions under `server/modules/`.

The frontend is React under `frontend/` (Vite + TS + Tailwind + shadcn/ui) — there is no Streamlit `client/` anymore.

Target structure (✓ = already exists; ⬜ = planned for a later section):

- server/
  - main.py                                 ✓
  - logger.py                               ✓
  - modules/
    - config.py                             ✓
    - document_loaders.py                   ✓ (Section 2)
    - document_cleaner.py                   ✓ (Section 2)
    - load_vectorstore.py                   ✓ (replaces chunking + vector_store split)
    - reranker.py                           ✓
    - llm.py                                ✓ (will be replaced by prompts.py + generation.py in Section 4)
    - query_handlers.py                     ✓
    - bm25_retriever.py                     ✓ (built during the advising work, before Section 3)
    - retrieval_modes.py                    ✓ Section 3
    - dense_retriever.py                    ✗ NOT created — deliberate. Dense/hybrid live inside
    - hybrid_retriever.py                   ✗   catalog_retriever.retrieve_documents(); the working
                                            ✗   fused path is injected into retrieval_modes.retrieve()
                                            ✗   as a callable instead of being split up. See §12.
    - prompts.py                            ⬜ Section 4
    - query_router.py                       ⬜ Section 4
    - generation.py                         ⬜ Section 4
    - response_schema.py                    ⬜ Section 4
    - logging_utils.py                      ✗ NOT created — per-query timings are carried on
                                            ✗   RetrievalOutcome and written by the eval runner.
  - evaluation/                             ✓ Sections 5-6
    - __init__.py                           ✓
    - build_benchmark.py                    ✓ (generates the benchmark from data/)
    - metrics.py                            ✓ Recall@k / Precision@k / MRR@k / nDCG@k
    - run_evaluation.py                     ✓
    - ablation_runner.py                    ✓
    - ablation_configs.yaml                 ✓
    - make_tables.py                        ✓ (report CSVs + failure analysis)
  - uploaded_pdfs/                          ✓ (legacy /upload_pdfs/ target)
  - uploaded_documents/                     ✓ (Section 2; /upload_documents/ target)
  - chroma_store/                           ✓ (persistent ChromaDB)

- frontend/                                 ✓ (React app)
  - public/assets/                          ✓
  - src/
    - App.tsx, main.tsx, index.css          ✓
    - components/ui/                        ✓ (shadcn primitives)
    - components/chat/                      ✓
    - contexts/AuthContext.tsx              ✓
    - lib/{api,utils}.ts                    ✓
    - pages/{Login,Signup,Chat,Profile}Page.tsx  ✓ (ProfilePage added by the advising work)

- data/                                     ✓
  - benchmark/questions.jsonl               ✓ Section 5 (GENERATED by build_benchmark.py, committed)
  - degree_requirements/ minors/            ✓ advising corpus — see "Final Implemented State"
  - curricula/ course_catalog/              ✓
  - sample_documents/                       ✗ not created — the advising corpus is the corpus

- outputs/                                  ✓ Sections 5-6
  - evaluation_runs/<timestamp>/            ✓ results.jsonl, summary_metrics.json,
                                            ✓   retrieved_chunks.jsonl, run_config.json
  - evaluation_runs/ablations.jsonl         ✓
  - failure_analysis/failure_cases.csv      ✓
  - tables/                                 ✓ retrieval/answer/efficiency/failure/ablation CSVs

- docs/                                     ✓ Section 7
  - prompt_log.md                           ✓
  - experiment_log.md                       ✓ (the measured numbers live here)
  - demo_script.md                          ✓ (also the video shot list)

Do not force this exact structure if the current codebase already has a better organization.

Adapt to the existing code.

---

# 10. Section 1 — Rebrand RagBot 2.0 into SU-GPT and Apply Sabanci Theme

## Status: ✅ Implemented (with later React migration)

**What was done**
- Replaced visible "RagBot" / "MediBot" / "medical assistant" wording with **SU-GPT — Course-Aware RAG Assistant for Sabancı University Materials** across the UI.
- Applied the Sabancı color palette (`#004B93`, `#003B73`, `#FFFFFF`, `#F5F8FC`, `#D8E6F3`, `#D6A13A`) in the React frontend (Tailwind theme tokens + the `bg-sky-night` utility).
- Updated the backend `PromptTemplate` in `server/modules/llm.py` — identity is now "You are SU-GPT, a course-aware academic RAG assistant for Sabanci University course materials" with grounded-answer rules and a refusal phrase for missing context.
- FastAPI app title in `server/main.py` is now "SU-GPT — Course-Aware RAG Assistant".
- Endpoints preserved: `POST /upload_pdfs/`, `POST /ask/`, `GET /test`.
- Existing PDF RAG flow preserved: upload → text extraction → chunking → ChromaDB → dense retrieval → CrossEncoder rerank → LLM answer → source metadata.

**Notes (not in the original Section 1 brief)**
- The Streamlit `client/` was deleted; its responsibilities were taken over by a React + Vite + shadcn/ui + Tailwind app under `frontend/`. The Sabancı blue/white theme now lives in Tailwind tokens (`src/index.css`) and the `sabanci.*` palette in `tailwind.config.js`.
- A visual-only login/signup gate (localStorage `AuthContext`) was added on top of the brand work — not required by Section 1 but bundled with the UI overhaul.

## Goal

Transform the visible identity of the existing app from RagBot, MediBot, or generic PDF chatbot into SU-GPT.

This section focuses only on:

- UI rebranding
- Sabanci University visual theme
- prompt identity update
- academic and course-aware wording
- preserving the current PDF RAG functionality

Do not implement:

- BM25
- multi-format ingestion
- evaluation scripts
- DOCX or PPTX support
- retrieval mode refactoring
- benchmark files

## Tasks

## 1.1 Inspect the current app

Find:

- Streamlit app file
- backend prompt template
- endpoint request and response structure
- current upload and ask flow
- current source display logic

Do not modify before understanding the current flow.

## 1.2 Rename visible identity

Replace visible old names:

- RagBot
- RagBot 2.0
- MediBot
- medical assistant
- medical advice
- PDF medical assistant

with:

- SU-GPT
- Course-Aware RAG Assistant
- Sabanci University Academic RAG Assistant
- Academic Document Assistant

## 1.3 Update Streamlit frontend

The frontend should show:

SU-GPT

Course-Aware RAG Assistant for Sabanci University Materials

Suggested page description:

Ask questions over uploaded course documents, syllabi, lecture slides, project descriptions, and LLM course materials. SU-GPT retrieves relevant evidence and generates grounded answers with sources.

Suggested sidebar:

SU-GPT

Course-Aware Academic RAG Assistant

Current implementation:
Dense RAG with CrossEncoder reranking

Current supported upload:
PDF documents

Future sections:
- Multi-format document ingestion
- BM25 retrieval
- Dense, BM25, and hybrid retrieval modes
- Evaluation pipeline
- Hallucination and efficiency analysis

## 1.4 Apply Sabanci-inspired theme

Use CSS inside Streamlit where appropriate.

Use these colors:

- #004B93
- #003B73
- #FFFFFF
- #F5F8FC
- #D8E6F3
- #D6A13A

Suggested UI improvements:

- blue header area
- white content cards
- soft blue borders
- clear upload card
- clear chat card
- source cards
- academic and professional wording

Avoid making the UI too heavy.

## 1.5 Update backend prompt identity

Replace current MediBot or medical prompt with:

You are SU-GPT, a course-aware academic RAG assistant for Sabanci University course materials.

Answer the user's question using only the provided context.

If the context does not contain enough information, say that the uploaded documents do not provide enough evidence.

Do not invent course policies, deadlines, prerequisites, grading rules, homework requirements, project requirements, or lecture content.

When possible, mention the relevant source document, page, slide, or section.

## 1.6 Preserve existing endpoints

Do not break:

- POST /upload_pdfs/
- POST /ask/
- GET /test

## 1.7 Preserve existing flow

The current flow must still work:

PDF upload → text extraction → chunking → ChromaDB storage → dense retrieval → CrossEncoder reranking → LLM answer → source metadata

## Expected Result After Section 1

The app should:

- open as SU-GPT
- use a Sabanci blue and white theme
- still upload PDFs
- still answer questions from PDFs
- still show sources if available
- no longer mention MediBot
- no longer mention medical advice
- preserve existing backend behavior

## Acceptance Checklist

- [ ] App title is SU-GPT
- [ ] Subtitle is course-aware and Sabanci-related
- [ ] Sabanci color theme is applied
- [ ] Old medical wording is removed
- [ ] Backend prompt identity is updated
- [ ] POST /upload_pdfs/ still works
- [ ] POST /ask/ still works
- [ ] GET /test still works
- [ ] No future section is implemented early

---

# 11. Section 2 — Multi-Format Academic Document Ingestion

## Status: ✅ Implemented

**What was done**
- `POST /upload_documents/` added to `server/main.py`, alongside the preserved `POST /upload_pdfs/`. Accepts PDF, PPTX, DOCX, MD, TXT; reports `{chunks, accepted_files, skipped_files, supported_extensions}` in the response.
- `server/modules/document_loaders.py` created: per-extension loaders that return normalized `{text, metadata: {source, document_type, page, slide, section}}` records. PDF page numbers are 1-indexed; PPTX slides preserved; DOCX paragraphs grouped under `Heading*` styles; Markdown blocks grouped under `#` headings; TXT falls back to UTF-8 with `errors="replace"`. PPTX/DOCX imports are lazy so PDF-only deployments still work.
- `server/modules/document_cleaner.py` created: NFC unicode normalization + whitespace collapse + line-ending fix + zero-width / NBSP / BOM stripping. Preserves code, formulas, and Turkish characters.
- `server/modules/load_vectorstore.py` refactored:
  - Legacy `load_vectorstore()` still serves `/upload_pdfs/`, saves PDFs into `./uploaded_pdfs/`.
  - New `load_vectorstore_multi()` serves `/upload_documents/`, saves into `./uploaded_documents/`.
  - Common `_ingest_paths()` runs loaders → cleaner → `RecursiveCharacterTextSplitter(CHUNK_SIZE, CHUNK_OVERLAP)`.
  - Each chunk gets a stable `chunk_id` (`{stem}::{location}::{index}::{sha1[:10]}`) + rich metadata (`source`, `document_type`, `page`, `slide`, `section`, `chunk_id`, `file_name` alias for back-compat).
- `CHUNK_SIZE` (default 1000) and `CHUNK_OVERLAP` (default 150) added to `server/modules/config.py`, overridable via env.
- `python-pptx` and `python-docx` added to `server/requirements.txt`.
- React frontend (`frontend/src/lib/api.ts` and `Sidebar.tsx`) calls the new endpoint, accepts the full extension list, and shows accepted/skipped/chunk counts via sonner toasts.

**Tuning that bled in alongside Section 2 (not required by the brief but improves answer quality)**
- Retrieval defaults bumped: `RETRIEVAL_CANDIDATE_K=20`, `RERANK_TOP_K=6` (was 10 / 3).
- Chunking defaults bumped: `CHUNK_SIZE=1000`, `CHUNK_OVERLAP=150` (was 500 / 50).
- `server/main.py` now wraps each reranked doc with a `[Source: filename, page N]` header before passing to the chain (`_format_for_context`).
- `server/modules/query_handlers.py` formats the Sources panel with `filename, page/slide/section`.

## Goal

Extend document ingestion beyond PDFs.

Supported formats:

- .pdf
- .pptx
- .docx
- .md
- .txt

This section focuses on:

- document upload
- text extraction
- cleaning
- metadata
- chunking
- storing chunks in ChromaDB

Do not implement:

- BM25 retrieval
- retrieval modes
- evaluation scripts
- benchmark metrics

## Tasks

## 2.1 Add general upload endpoint

Add:

POST /upload_documents/

It should accept multiple files.

It should support:

- .pdf
- .pptx
- .docx
- .md
- .txt

Keep this endpoint backward-compatible with current logic.

Keep POST /upload_pdfs/ working.

## 2.2 Add document loader module

Create or update:

server/modules/document_loaders.py

Implement loader responsibilities for:

- PDF loading
- PPTX loading
- DOCX loading
- Markdown loading
- TXT loading
- general document loading based on extension

Each loader should return normalized document, page, or slide records.

A normalized record should include:

- text
- metadata
  - source
  - document_type
  - page, if available
  - slide, if available
  - section, if available

## 2.3 Recommended extraction behavior

PDF:

- use existing PyPDFLoader if already working
- preserve page number when available

PPTX:

- use python-pptx
- extract text from each slide
- preserve slide number

DOCX:

- use python-docx
- extract paragraphs
- optionally detect headings as section names

Markdown:

- read as UTF-8 text
- optionally preserve headings as sections

TXT:

- read as UTF-8 text
- fallback to error-tolerant decoding if needed

## 2.4 Add dependencies only if needed

Update the requirements file with:

- python-pptx
- python-docx

Only add these in Section 2.

## 2.5 Add cleaning module

Create or update:

server/modules/document_cleaner.py

Implement a text cleaning utility.

Cleaning should:

- normalize whitespace
- remove repeated blank lines
- remove obvious encoding artifacts where possible
- strip empty text
- avoid destroying useful academic content
- avoid aggressive cleaning that removes formulas or code

## 2.6 Add metadata

Each chunk should include metadata:

- source
- document_type
- page, if available
- slide, if available
- section, if available
- language
- topic, optional
- chunk_id

chunk_id should be stable enough for evaluation.

Suggested chunk ID format:

source_name + location + chunk_index

or a hash of source, location, and text.

## 2.7 Make chunking configurable

Add environment variables:

- CHUNK_SIZE, default 500
- CHUNK_OVERLAP, default 50

Use these in the existing RecursiveCharacterTextSplitter.

Do not hardcode chunk size if configuration exists.

## 2.8 Store uploaded documents

Current PDF files may be saved under:

server/uploaded_pdfs/

For multi-format documents, create:

server/uploaded_documents/

or reuse a clean existing upload directory.

Do not remove the old PDF directory if existing code depends on it.

## Expected Result After Section 2

The app and API should support uploading academic documents in multiple formats and store chunks with richer metadata.

## Acceptance Checklist

- [ ] POST /upload_documents/ exists
- [ ] POST /upload_pdfs/ still works
- [ ] PDF upload still works
- [ ] PPTX upload works
- [ ] DOCX upload works
- [ ] MD upload works
- [ ] TXT upload works
- [ ] chunks include source metadata
- [ ] page numbers are preserved for PDFs where possible
- [ ] slide numbers are preserved for PPTX where possible
- [ ] chunk IDs exist
- [ ] chunk size and overlap are configurable
- [ ] existing POST /ask/ still works
- [ ] No BM25 or evaluation is implemented early

---

# 12. Section 3 — Retrieval Modes: LLM-only, BM25, Dense, Hybrid, Hybrid Rerank

## Status: ✅ Implemented (2026-07-22)

**Important correction to the original brief below:** by the time this section was reached, the
*retrievers* already existed — the advising work had built BM25 (`bm25_retriever.py`, dependency-free
Okapi, no `rank-bm25` needed) and fused it with dense + structured `get` inside
`catalog_retriever.retrieve_documents`, followed by CrossEncoder rerank. What was missing was the
**switch**: every query went down one hardcoded path. So this section built the mode-selection layer
over the existing retrievers, plus the genuinely absent `llm_only` baseline. `dense_retriever.py` /
`hybrid_retriever.py` were deliberately NOT created — splitting the working fused path into separate
modules would have been churn with no behavioural gain.

**What was done**
- `server/modules/retrieval_modes.py` — `RETRIEVAL_MODES`, `normalize_mode()` (aliases + safe
  fallback to default on unknown input), and `retrieve()` returning a `RetrievalOutcome`
  (`documents`, `candidate_count`, `retrieval_ms`, `rerank_ms`, `reranked`). The production fused
  path is *injected* as a `hybrid_search` callable so hybrid modes keep exact production behaviour,
  including route-based multi-search.
- **Scoping invariant:** every RAG mode receives the same `metadata_filter` the profile-aware
  retrieval policy produced. Ablating a retriever changes ranking only — it never widens the corpus
  past the student's program/term. Verified by test + live run against the 30k-vector index.
- `bm25_retriever.annotate_bm25()` gained `require_narrowing: bool = True`. Hybrid keeps the old
  narrowing-only gate; standalone `bm25` mode passes `False` (a baseline must answer any question,
  not only program/term/course-scoped ones).
- `llm.answer_without_context()` — LLM-only baseline: no retrieval, no student data, no sources,
  and it bypasses the profile gate. Expected to hallucinate Sabancı facts; that measurement is
  the reason it exists.
- `POST /ask/` now accepts optional `mode`, `top_k`, `prompt_strategy`, `expert_mode`. Posting only
  `question` behaves exactly as before (`hybrid_rerank`). Every response — including early returns
  (gate, missing interest area, non-academic fallback) — is stamped with the config that produced
  it, plus `reranked` / `num_retrieved_chunks` / `num_final_context_chunks`. `prompt_strategy` and
  `expert_mode` are accepted and echoed but **inert until Section 4**.
- Config (§8): `DEFAULT_RETRIEVAL_MODE`, `ENABLE_RERANKING`, `BM25_TOP_K`, `DENSE_TOP_K`,
  `FINAL_CONTEXT_TOP_K`, `DEFAULT_PROMPT_STRATEGY`, `DEFAULT_EXPERT_MODE`.
- Frontend: a mode selector was added to `ChatHeader` per §3.8 — **but it was removed again in the
  2026-07-22 UX round** (see §I). Picking a retrieval strategy is not a student's job, and the
  server now defaults to the mode the benchmark measured as best. `RETRIEVAL_MODES` is still
  exported from `lib/api.ts` and `/ask/` still accepts `mode`, because the evaluation harness
  depends on it. **`DEFAULT_RETRIEVAL_MODE` is now `hybrid`, not `hybrid_rerank`** — see the
  headline finding in §15.
- Tests: 4 new invariants in `server/tests/test_advising.py` (normalization/fallback, llm_only
  retrieves nothing, hybrid skips rerank but keeps scope + top_k, bm25 gate default unchanged).
  **18 unit tests, 0 failed.** `tsc --noEmit` clean.

**Verified live** against the real Chroma index (CS/202401 filter, "CS 455 area elective"): all five
modes ran, all returned CS-only documents, `top_k` honored, and `bm25` returns hits with no
narrowing filter. Cold-start CrossEncoder load dominates `hybrid_rerank` latency on first call.

**Known gap:** `llm_only` skips conversation-memory working context by design (it takes the raw
question), so follow-up pronoun resolution does not apply in that mode.

---

## Original brief (kept for reference)

The only retrieval mode currently wired is **Dense + CrossEncoder rerank** (the original RagBot pipeline). When implementing this section, refactor the inline retrieval in `server/main.py` (`vectorstore.similarity_search` + `rerank_documents`) into the modules listed below.

## Goal

Implement the retrieval configurations required by the proposal:

- LLM-only baseline
- BM25 RAG
- Dense retrieval RAG
- Hybrid RAG
- Hybrid RAG with CrossEncoder reranking

This section focuses on retrieval architecture.

Do not implement full evaluation scripts yet.

## Tasks

## 3.1 Define retrieval modes

Create or update:

server/modules/retrieval_modes.py

Supported modes:

- llm_only
- bm25
- dense
- hybrid
- hybrid_rerank

Default mode:

hybrid_rerank

If implementing hybrid_rerank immediately is risky, keep the existing dense + rerank behavior as default but expose all modes cleanly.

## 3.2 Add request schema support

Update POST /ask/ so it can accept:

- question
- mode
- top_k
- prompt_strategy
- expert_mode

For backward compatibility, if the current frontend sends only a question, it must still work.

Defaults:

- mode: DEFAULT_RETRIEVAL_MODE or hybrid_rerank
- top_k: RERANK_TOP_K or 5
- prompt_strategy: basic
- expert_mode: auto

## 3.3 Add BM25 retriever

Add dependency:

- rank-bm25

Create:

server/modules/bm25_retriever.py

Responsibilities:

- load or index available chunks
- tokenize chunk text
- tokenize query
- return top-k chunks
- return scores
- return the same normalized result shape as dense retrieval

Normalized retrieval result should include:

- chunk_id
- text
- metadata
- score
- retriever, with value bm25

## 3.4 Keep dense retrieval through Chroma

The existing ChromaDB search becomes dense mode.

Do not remove existing Chroma logic.

Wrap it into a reusable dense retrieval function if needed.

## 3.5 Add hybrid retrieval

Create:

server/modules/hybrid_retriever.py

Hybrid retrieval should:

- retrieve from BM25
- retrieve from dense Chroma search
- merge results
- deduplicate by chunk_id, source/page/slide, or content hash
- optionally normalize scores
- return candidates

Suggested default:

BM25 top 10 + dense top 10 → merge → deduplicate → top candidates

## 3.6 Add optional CrossEncoder reranking

The current project already has CrossEncoder reranking.

Refactor it into:

server/modules/reranker.py

The reranker should:

- accept query
- accept candidate chunks
- score each query and chunk pair
- sort candidates by rerank score
- return top-k candidates
- attach rerank_score to each result

Mode behavior:

- llm_only: no retrieval, no reranking
- bm25: BM25 retrieval only
- dense: Chroma dense retrieval only
- hybrid: BM25 + dense, no reranking
- hybrid_rerank: BM25 + dense + CrossEncoder reranking

## 3.7 LLM-only baseline

In llm_only mode:

- do not retrieve chunks
- do not pass document context
- generate answer only from LLM internal knowledge
- clearly mark response mode as llm_only
- do not show sources
- use this mode mainly for comparison and evaluation

The UI should warn:

LLM-only mode does not use retrieved course documents and may hallucinate course-specific facts.

## 3.8 Update frontend mode selector

Add a Streamlit selector:

Retrieval Mode:

- LLM-only baseline
- BM25 RAG
- Dense RAG
- Hybrid RAG
- Hybrid + CrossEncoder Reranking

Map labels to internal modes.

Do not make the frontend complicated.

## Expected Result After Section 3

The backend and frontend should allow asking the same question under different retrieval modes.

## Acceptance Checklist

- [x] llm_only mode works
- [x] bm25 mode works
- [x] dense mode works
- [x] hybrid mode works
- [x] hybrid_rerank mode works
- [x] POST /ask/ remains backward-compatible
- [~] frontend has retrieval mode selector — built, then **deliberately removed** in the UX round
      (§I). Students should not pick a retrieval strategy; the server runs the measured-best mode.
- [x] sources still appear for RAG modes
- [x] no sources are shown for LLM-only
- [x] no full evaluation pipeline is implemented early

---

# 13. Section 4 — Grounded Answer Generation, Citations, Few-Shot Prompting, and Expert Routing

## Status: 🟡 Partially started (only the prompt-softening and source-labeling pieces)

**Already done (bled in during Section 2 tuning)**
- The prompt in `server/modules/llm.py` was rewritten: grounded but not over-refusing, asks the model to extract partial info when present, asks for a `Sources:` block, and supports both English and Turkish answers.
- `server/main.py` formats each reranked doc with a `[Source: ...]` header (`_format_for_context`) before passing to the chain — this is the contextual half of "citation correctness".
- `server/modules/query_handlers.py` returns `filename, page/slide/section` strings in the response's `sources[]`.

**Still to do under Section 4 proper**
- `server/modules/prompts.py` (basic / few-shot / expert templates)
- `server/modules/query_router.py` (rule-based syllabus/lecture/catalog/homework/concept/general routing, `auto` vs explicit `expert_mode`)
- Frontend: prompt-strategy + expert-mode selectors
- Response schema: `{answer, mode, prompt_strategy, expert_mode, sources, latency_ms, prompt_tokens_estimate}` with per-source rerank scores and previews
- Few-shot examples (supported English, unsupported, Turkish, lecture-concept)

## Goal

Improve answer generation so SU-GPT produces grounded academic answers with clear citations.

This section focuses on:

- prompt templates
- grounded answer behavior
- citation formatting
- insufficient context refusal
- few-shot prompting
- mixture-of-experts inspired query routing
- structured reasoning without exposing hidden reasoning

Do not implement full benchmark evaluation yet.

## Tasks

## 4.1 Create prompts module

Create or update:

server/modules/prompts.py

Add templates for:

- basic grounded prompt
- few-shot grounded prompt
- LLM-only prompt
- syllabus policy expert prompt
- lecture content expert prompt
- course catalog expert prompt
- homework/project expert prompt
- concept explanation expert prompt
- unanswerable detection prompt

## 4.2 Basic grounded prompt behavior

The basic grounded prompt should instruct:

- You are SU-GPT, a course-aware academic RAG assistant for Sabanci University course materials.
- Use only the provided context to answer the question.
- Before answering, silently check whether the context contains enough evidence.
- Do not reveal hidden reasoning.
- If the context does not contain enough evidence, answer: The uploaded documents do not provide enough information to answer this question.
- Do not invent course policies, deadlines, prerequisites, grading rules, homework requirements, project requirements, or lecture content.
- When the answer is supported, provide a clear answer and cite the source documents.

## 4.3 Few-shot grounded prompt

Add few-shot examples for:

- supported English syllabus question
- unsupported question
- Turkish question
- lecture concept question

Keep examples short to avoid excessive token usage.

prompt_strategy = few_shot should use the few-shot template.

## 4.4 Query router

Create:

server/modules/query_router.py

Implement query routing responsibilities.

Supported expert modes:

- auto
- syllabus_policy_expert
- lecture_content_expert
- course_catalog_expert
- homework_project_expert
- concept_explanation_expert
- general_academic_rag_expert

If expert_mode is auto, use rule-based classification.

If user explicitly selects an expert mode, use that.

## 4.5 Expert prompts

Expert prompts should not change facts.

They should only influence style and focus.

Syllabus expert:

Focus on policies, grading, attendance, AI usage, make-up rules, academic integrity, and course requirements. Only answer if retrieved syllabus or course policy context supports the answer.

Lecture content expert:

Focus on lecture concepts such as Word2Vec, BERT, GPT, attention, embeddings, semantic search, RAG, reranking, and evaluation. Use the retrieved lecture material as the evidence.

Course catalog expert:

Focus on course code, title, prerequisites, credits, and catalog-style descriptions. Do not invent prerequisites.

Homework/project expert:

Focus on homework instructions, project requirements, submission rules, prompt logs, experiment logs, and evaluation deliverables. Do not invent deadlines.

## 4.6 Citation formatting

API response should include:

- answer
- mode
- prompt_strategy
- expert_mode
- sources
- latency_ms
- prompt_tokens_estimate

Each source should include:

- source
- document_type
- page
- slide
- section
- chunk_id
- score
- rerank_score
- preview

Citation display format:

Sources:
1. CS455_555 Syllabus.pdf, page 5
2. 3_GPT.pptx, slide 12

If page or slide is unavailable:

Sources:
1. filename.ext

If unsupported:

Sources:
No supporting source found.

## 4.7 Update frontend source display

The Streamlit frontend should show:

- answer
- retrieval mode
- prompt strategy
- expert mode
- latency if available
- source cards
- source preview
- page or slide information if available

## 4.8 Add prompt strategy selector

Frontend selector:

Prompt Strategy:

- Basic grounded prompt
- Few-shot grounded prompt
- Expert-routed prompt

Internal values:

- basic
- few_shot
- expert_routed

Default:

basic

## 4.9 Add expert mode selector

Frontend selector:

Expert Mode:

- Auto
- Syllabus / Policy
- Lecture Content
- Course Catalog / Prerequisites
- Homework / Project
- Concept Explanation

Internal values:

- auto
- syllabus_policy_expert
- lecture_content_expert
- course_catalog_expert
- homework_project_expert
- concept_explanation_expert

Default:

auto

## Expected Result After Section 4

SU-GPT should answer in a more grounded, citation-aware, academic style and support different prompt strategies.

## Acceptance Checklist

- [ ] prompts.py exists
- [ ] basic grounded prompt works
- [ ] few-shot prompt works
- [ ] query router exists
- [ ] expert-routed prompt works
- [ ] unsupported questions are refused
- [ ] sources are formatted clearly
- [ ] frontend displays source cards
- [ ] frontend supports prompt strategy selection
- [ ] frontend supports expert mode selection
- [ ] hidden reasoning is not exposed
- [ ] no benchmark evaluation is implemented early

---

# 14. Section 5 — Benchmark Dataset and Evaluation Pipeline

## Status: ✅ Implemented (2026-07-22)

**Files:** `server/evaluation/{__init__,build_benchmark,metrics,run_evaluation}.py`,
`data/benchmark/questions.jsonl` (generated + committed), `outputs/evaluation_runs/<timestamp>/`.
npm: `benchmark:build`, `eval`, `eval:retrieval`.

**Benchmark is GENERATED from the corpus, not hand-written.** `build_benchmark.py` derives every
answerable question from a real row in `data/degree_requirements/**` / `data/minors/**`: the gold
`chunk_id` is that row's actual id and the reference answer is built from its own fields. The
builder verifies each gold id against all 30,343 chunks and exits non-zero on a miss. This is the
direct implementation of §17's "do not fabricate benchmark answers". 22 questions: 16 answerable
(factual_lookup / course_requirement / turkish), 4 unanswerable targeting *documented* corpus gaps
(schedules, instructors, enrolment, grades — all real §H gaps), 2 misleading (false premise).
Known bias, recorded in the file: templated wording shares vocabulary with the chunks, so lexical
scores are an optimistic bound.

**metrics.py** — Recall@k, Precision@k, MRR@k, nDCG@k, chunk-level AND source-level. Undefined
metrics return NaN and are skipped by `mean_ignoring_nan`, so an unmeasurable metric can never be
reported as 0.0. Source-level is near-ceiling for this corpus (a program×term is one file) and is
documented as such — quote chunk-level.

**run_evaluation.py** — every question × every mode; writes `results.jsonl`, `summary_metrics.json`,
`retrieved_chunks.jsonl`, `run_config.json`. Uses the SHIPPED prompt and context formatting
(imported from `main.py`) so the evaluation cannot drift from production. Flags: `--retrieval-only`
(no LLM cost), `--modes`, `--limit`, `--summarize-only` (re-aggregate a run in place).
Section 5.5 manual fields (`answer_correctness`, `citation_correctness`, `faithfulness`,
`answer_relevancy`, `hallucination_flag`) are written as `null` and NEVER auto-filled. Two objective
signals are: `auto_refusal_detected` and `auto_reference_number_match`. RAGAS optional (§5.6):
absent here, recorded as `ragas_available: false`, skipped gracefully.

**Measured results are in `docs/experiment_log.md`.** Headlines: retrieval chunk-level Recall@6 —
hybrid 0.625 > hybrid_rerank 0.438 > bm25 0.375 > dense 0.188; and with BM25 retrieval the system
reproduced the correct official number in 93% of answers and refused 67% of unsupported questions,
versus 27% / **0%** for `llm_only`.

**Two bugs found and fixed while validating our own harness** (both recorded in `docs/prompt_log.md`):
1. Groq's 100k tokens/day cap returned 429s mid-run; the aggregator was scoring those empty answers
   as *wrong* rather than *not measured*, which made `hybrid` look like 0% correct when it had
   produced no answers at all. Now failed calls are excluded and `generation_coverage` is reported.
2. `make_tables` double-counted failures when merging two runs, and charged `llm_only` with
   "retrieval misses" despite it having no retriever.

**Known gap:** answer-side coverage is incomplete (`hybrid` 0%, `hybrid_rerank` 5%, `dense` 18%)
because of the daily token cap. Retrieval metrics are unaffected and reproduce exactly across three
independent runs. Re-run one or two modes per day to complete it.

## Goal

Create a benchmark-based evaluation framework to compare SU-GPT configurations.

This section focuses on:

- benchmark file format
- evaluation runner
- retrieval metrics
- structured outputs
- manual evaluation fields

## Tasks

## 5.1 Create benchmark structure

Create:

data/benchmark/questions.jsonl

Each benchmark item should include:

- id
- question
- language
- question_type
- answerable
- reference_answer
- expected_sources
- expected_chunk_ids
- notes

Supported question types:

- factual_lookup
- concept_explanation
- syllabus_policy
- course_requirement
- homework_project
- turkish
- english
- unanswerable
- misleading
- multi_hop

Create a small starter benchmark with 10 to 15 example questions only if source documents exist.

Do not fabricate exact answers unless supported by included sample documents.

## 5.2 Create evaluation runner

Create:

server/evaluation/run_evaluation.py

It should:

- load benchmark questions
- run each question against selected modes
- collect answers
- collect retrieved sources
- collect latency
- compute retrieval metrics where possible
- save results

Supported modes:

- llm_only
- bm25
- dense
- hybrid
- hybrid_rerank

Supported prompt strategies:

- basic
- few_shot
- expert_routed

The runner should allow command-line usage with selected modes and benchmark path.

## 5.3 Save outputs

Create output folder:

outputs/evaluation_runs/

For each run, create a timestamped folder.

Save:

- results.jsonl
- summary_metrics.json
- retrieved_chunks.jsonl
- run_config.json

## 5.4 Implement retrieval metrics

Create or update:

server/evaluation/metrics.py

Implement:

- Recall@k
- MRR@k
- nDCG@k

If expected chunk IDs are unavailable, compute source-level metrics using expected_sources.

Support both:

- chunk-level matching
- source-level matching

## 5.5 Add answer evaluation fields

Each result row should include fields for later manual or automated evaluation:

- answer_correctness
- citation_correctness
- faithfulness
- answer_relevancy
- hallucination_flag
- manual_notes

These can initially be null, empty, or false until manually labeled.

Do not overcomplicate automated grading in this section.

## 5.6 Optional RAGAS integration

If easy and safe, add optional RAGAS integration.

RAGAS should be optional.

The project must work without RAGAS installed.

If RAGAS is not installed, skip RAGAS metrics gracefully.

## Expected Result After Section 5

The project should be able to run benchmark questions across retrieval modes and save structured evaluation outputs.

## Acceptance Checklist

- [ ] benchmark file format exists
- [ ] evaluation runner exists
- [ ] evaluation runs selected modes
- [ ] Recall@k implemented
- [ ] MRR implemented
- [ ] nDCG@k implemented if feasible
- [ ] results saved as JSONL
- [ ] summary metrics saved as JSON
- [ ] retrieved chunks saved
- [ ] manual evaluation fields supported
- [ ] RAGAS is optional, not required

---

# 15. Section 6 — Efficiency Logging, Ablations, and Failure Analysis

## Status: ✅ Implemented (2026-07-22)

**Files:** `server/evaluation/{ablation_configs.yaml,ablation_runner.py,make_tables.py}`,
`outputs/tables/*.csv`, `outputs/failure_analysis/failure_cases.csv`.
npm: `eval:ablate`, `eval:tables`.

**Efficiency logging (§6.1)** — per-query `retrieval_ms` / `rerank_ms` / `generation_ms` /
`total_ms`, `num_retrieved_chunks`, `num_final_context_chunks`, `prompt_chars_estimate`,
`prompt_tokens_estimate` (clearly named an ESTIMATE, ~4 chars/token). Measured: retrieval stack is
100–723 ms while generation is ~7.6–8.2 s, so generation dominates end-to-end latency.
`estimated_cost_usd_per_query` is deliberately **blank** — no price is hard-coded, because a
plausible-looking wrong cost is worse than an empty cell. Set `COST_PER_1K_INPUT` in
`make_tables.py` to populate it.

**Ablations (§6.2/6.3)** — 12 cells (4 modes × 3 top_k) run and recorded. The runner skips cells
already present in `ablations.jsonl` (so an interrupted sweep resumes), flushes after each cell, and
defaults to retrieval-only so the grid does not multiply LLM cost. **No chunk-size/overlap sweep,
deliberately:** the advising corpus is pre-chunked (1 JSONL row = 1 chunk with its own stable id),
so `CHUNK_SIZE`/`CHUNK_OVERLAP` do not affect it and a sweep would produce identical numbers and a
misleading "no effect" conclusion. Sweeping honestly would require re-ingesting a differently-chunked
corpus, which invalidates every ground-truth chunk id. Documented in the YAML and the experiment log.

**Failure analysis (§6.4)** — `failure_cases.csv` with the required columns. `error_type` is
auto-classified by documented rules (`retrieval_miss` = gold never in candidates; `retrieval_noise`
= gold retrieved but ranked out of top-k; `generation_hallucination`; `insufficient_context`;
`unsupported_answer`). `root_cause` / `proposed_fix` / `before_after_evidence` are left BLANK —
they are human analysis, not measurements. Provider failures are excluded from generation verdicts.

**Report tables (§6.5)** — `retrieval_metrics.csv`, `answer_metrics.csv`, `efficiency_metrics.csv`,
`failure_summary.csv`, `ablation_summary.csv`, plus `run_provenance.json`. `make_tables.py` accepts
`--answers-run` so retrieval and answer metrics can come from different runs (needed because the
token cap split them).

**Headline finding:** the CrossEncoder reranker *hurts* retrieval on this corpus — Recall@6 0.438
vs 0.625 for plain hybrid, reproduced at top_k 3/5/10, and it costs ~295 ms/query. Verified by
tracing gold chunks through the reranker (rank 2→20, 7→20, 1→14), so it is a real corpus/reranker
mismatch, not a harness artefact. Failure counts show it: hybrid 4 miss + 2 ranking losses vs
hybrid_rerank 4 + 5. **Not acted on** — measuring and reporting is this section's job; changing the
reranker is a later product decision. Candidate fixes are listed in the experiment log.

## Goal

Add practical experiment logging for the CS455 final report.

This section supports:

- latency measurement
- prompt length estimation
- retrieved chunk count
- estimated API cost
- chunk-size ablations
- top-k ablations
- reranking ablations
- failure analysis tables

## Tasks

## 6.1 Add efficiency logging

Each query result should log:

- query_id
- mode
- prompt_strategy
- expert_mode
- latency_ms
- retrieval_latency_ms
- rerank_latency_ms
- generation_latency_ms
- num_retrieved_chunks
- num_final_context_chunks
- prompt_tokens_estimate
- completion_tokens_estimate
- estimated_cost_usd

If exact token counting is unavailable, use a simple estimate and clearly name it as an estimate.

## 6.2 Add ablation config

Create:

server/evaluation/ablation_configs.yaml

Include experiments such as:

- chunk_sizes: 300, 500, 600
- chunk_overlaps: 30, 50, 100
- top_k_values: 3, 5, 10
- modes: bm25, dense, hybrid, hybrid_rerank
- prompt_strategies: basic, few_shot, expert_routed

## 6.3 Add ablation runner

Create:

server/evaluation/ablation_runner.py

It should:

- load ablation config
- run selected combinations
- save metrics
- avoid rerunning expensive experiments unnecessarily if outputs already exist
- keep runtime manageable

Do not make ablations too heavy by default.

## 6.4 Add failure analysis file

Create:

outputs/failure_analysis/failure_cases.csv

Columns:

- query_id
- question
- mode
- prompt_strategy
- expert_mode
- model_output
- reference_answer
- retrieved_sources
- error_type
- root_cause
- proposed_fix
- before_after_evidence

Allowed error types:

- retrieval_miss
- irrelevant_retrieval
- retrieval_noise
- generation_hallucination
- citation_error
- insufficient_context
- multi_hop_reasoning_failure
- other

## 6.5 Add report-ready outputs

Generate simple CSV or JSON summaries that can be used in the final report:

- outputs/tables/retrieval_metrics.csv
- outputs/tables/answer_metrics.csv
- outputs/tables/efficiency_metrics.csv
- outputs/tables/failure_summary.csv
- outputs/tables/ablation_summary.csv

## Expected Result After Section 6

The project should produce experiment artifacts suitable for the final CS455 report.

## Acceptance Checklist

- [ ] latency is logged
- [ ] retrieval latency is logged
- [ ] reranking latency is logged
- [ ] generation latency is logged
- [ ] prompt token estimate is logged
- [ ] retrieved chunk count is logged
- [ ] estimated cost is logged
- [ ] ablation config exists
- [ ] ablation runner exists
- [ ] failure analysis CSV exists
- [ ] report-ready tables are generated

---

# 16. Section 7 — Final Cleanup, README, Prompt Log, Experiment Log, and Demo Flow

## Status: ✅ Implemented (2026-07-22)

- **README** — already strong; extended with the retrieval-modes description, the new `/ask/`
  parameters, an Evaluation section (how to reproduce + the two headline findings), the evaluation
  npm scripts, the `server/evaluation/` + `outputs/` + `docs/` tree, and an honest limitations list
  (Section 4 inert, answer-side coverage capped by Groq quota, cross-lingual retrieval gap).
- **`docs/experiment_log.md`** — configuration table, benchmark method + its known bias, and the
  four measured results (retrieval quality, answers-vs-no-retrieval, efficiency, ablations), each
  traced to the CSV it came from, plus an explicit "what is NOT measured" section and run provenance.
- **`docs/prompt_log.md`** — AI-usage table with a "Verified by" column for every row, including the
  two occasions where reviewing the model's output caught bugs in our own harness. Earlier sessions
  are flagged for the team to fill in.
- **`docs/demo_script.md`** — 6–8 minute shot list doubling as the video script: grounded answer →
  deterministic audit → refusal/missing-data safety → retrieval-mode comparison including the
  `llm_only` control → measured results. Ends with a "questions to avoid on camera" list tied to the
  §H gaps.
- **`.gitignore`** — evaluation outputs handled: the small report artefacts (`summary_metrics.json`,
  `run_config.json`, `outputs/tables/`, `failure_cases.csv`) are **committed on purpose** as the
  evidence behind the experiment log; only the bulky per-run `results.jsonl` /
  `retrieved_chunks.jsonl` dumps are excluded.

**Not done:** a from-scratch reproducibility check on a clean machine (§7.6) — the pipeline was only
exercised on the development machine with an already-built Chroma index.

## Goal

Prepare the project for final CS455 submission.

This section focuses on:

- reproducibility
- documentation
- prompt log
- experiment log
- final demo readiness
- cleanup

## Tasks

## 7.1 Update README

The README should include:

- project title
- team members
- CS455 track
- project description
- architecture overview
- setup instructions
- environment variables
- how to run backend
- how to run frontend
- how to upload documents
- how to ask questions
- how to run evaluation
- supported retrieval modes
- supported prompt strategies
- example queries
- known limitations

## 7.2 Add prompt log

Create:

docs/prompt_log.md

Include a table with:

- date
- section
- tool used
- prompt summary
- output used
- what was changed or verified

This is required because the course follows a transparent AI usage policy.

## 7.3 Add experiment log

Create:

docs/experiment_log.md

Include:

- embedding model
- reranking model
- LLM model
- retrieval modes
- prompt strategies
- chunk size
- chunk overlap
- top-k
- benchmark size
- evaluation date
- important observations

## 7.4 Add demo script

Create:

docs/demo_script.md

The demo should show:

- upload course documents
- ask a syllabus or policy question
- ask a lecture-content question
- ask an unanswerable question
- compare LLM-only vs RAG
- show evaluation table
- show failure analysis example

## 7.5 Update .gitignore

Do not commit:

- .env
- __pycache__/
- server/chroma_store/
- server/uploaded_pdfs/
- server/uploaded_documents/
- outputs/evaluation_runs/raw_large_files/
- temporary files
- local cache files

Add or update .gitignore accordingly.

## 7.6 Final reproducibility check

Make sure a new user can:

- install dependencies
- configure environment variables
- run the FastAPI backend
- run the Streamlit frontend
- upload sample documents
- ask questions
- run evaluation scripts
- inspect outputs

## Expected Result After Section 7

The project should be clean, reproducible, documented, and ready for final CS455 submission and demo.

## Acceptance Checklist

- [ ] README is complete
- [ ] prompt log exists
- [ ] experiment log exists
- [ ] demo script exists
- [ ] .gitignore is updated
- [ ] setup instructions are clear
- [ ] evaluation instructions are clear
- [ ] final app runs end-to-end
- [ ] project is ready for demo

---

# 17. Final Implementation Rules

Do not implement future sections early.

If the user asks to finish Section 1, only finish Section 1.

If a future section requires small compatibility changes to previous code, make the smallest safe change and explain it clearly.

Always preserve existing working behavior.

Always keep the project runnable after each section.

Always prefer simple, understandable, CS455-appropriate engineering over overly complex abstractions.

Do not hide errors silently.

Log meaningful errors.

Do not expose API keys.

Do not commit private data.

Do not include private student records, grades, or confidential institutional data.

Do not hallucinate Sabanci-specific facts.

Do not invent evaluation results.

Do not fabricate benchmark answers unless they are supported by available documents.

The final contribution is not a new LLM model. The final contribution is a complete, reproducible, course-aware RAG application with careful evaluation of retrieval, hallucination, citations, and efficiency.
