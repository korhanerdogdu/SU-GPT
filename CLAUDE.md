# CLAUDE.md — SU-GPT Incremental Implementation Guide

# Project Name

SU-GPT: An Evaluation of Retrieval, Hallucination and Efficiency for a Course-Aware RAG Assistant

# Team

- Mehmet Selman Yilmaz
- Korhan Erdogdu

# Track

CS 455

---

# 0. Read This First

You are working inside an existing project, not starting from scratch. The project was originally RagBot 2.0 and is now SU-GPT.

## Current State (as of last update)

**Implemented**
- Sections 1 and 2 are complete.
- Section 4 has a partial bleed-through: the LLM prompt has been softened, made citation-aware, and the `/ask/` handler now passes source-labeled context to the model. The dedicated `prompts.py` module, query router, and few-shot/expert prompts are NOT yet implemented — those still belong to Section 4 proper.
- Retrieval pipeline was tuned (CHUNK_SIZE=1000, CHUNK_OVERLAP=150, RETRIEVAL_CANDIDATE_K=20, RERANK_TOP_K=6, source-labeled context).
- Frontend was migrated from Streamlit (deleted) to a React + Vite + shadcn/ui + Tailwind app under `frontend/`. A visual-only login/signup gate (localStorage-backed) exists at `/login` and `/signup`; the main chat lives at `/`.

**Not yet implemented**
- Section 3 (retrieval modes: LLM-only, BM25, Dense, Hybrid, Hybrid+Rerank)
- Section 4 proper (prompts module, query router, expert routing, few-shot)
- Section 5 (benchmark + evaluation pipeline)
- Section 6 (efficiency logging, ablations, failure analysis)
- Section 7 (README, prompt log, experiment log, demo script)

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
    - bm25_retriever.py                     ⬜ Section 3
    - dense_retriever.py                    ⬜ Section 3 (currently inline in main.py)
    - hybrid_retriever.py                   ⬜ Section 3
    - retrieval_modes.py                    ⬜ Section 3
    - prompts.py                            ⬜ Section 4
    - query_router.py                       ⬜ Section 4
    - generation.py                         ⬜ Section 4
    - response_schema.py                    ⬜ Section 4
    - metrics.py                            ⬜ Section 5
    - logging_utils.py                      ⬜ Section 6
  - evaluation/                             ⬜ Sections 5-6
    - run_evaluation.py
    - metrics.py
    - ablation_runner.py
    - ablation_configs.yaml
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
    - pages/{Login,Signup,Chat}Page.tsx     ✓

- data/                                     ⬜ Section 5
  - benchmark/
    - questions.jsonl
  - sample_documents/

- outputs/                                  ⬜ Sections 5-6
  - evaluation_runs/
  - failure_analysis/
  - tables/

- docs/                                     ⬜ Section 7
  - prompt_log.md
  - experiment_log.md
  - demo_script.md

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

## Status: ⬜ Not yet implemented

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

- [ ] llm_only mode works
- [ ] bm25 mode works
- [ ] dense mode works
- [ ] hybrid mode works
- [ ] hybrid_rerank mode works
- [ ] POST /ask/ remains backward-compatible
- [ ] frontend has retrieval mode selector
- [ ] sources still appear for RAG modes
- [ ] no sources are shown for LLM-only
- [ ] no full evaluation pipeline is implemented early

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

## Status: ⬜ Not yet implemented

No benchmark file, evaluation runner, or metrics module exists yet.

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

## Status: ⬜ Not yet implemented

No latency / token / cost logging, ablation config, ablation runner, or failure-analysis artifacts exist yet. Depends on Sections 3-5 being in place.

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

## Status: ⬜ Not yet implemented

Only `frontend/README.md` exists (covers the React app's setup). The project-level README, `docs/prompt_log.md`, `docs/experiment_log.md`, and `docs/demo_script.md` are not written. `.gitignore` is partial (`frontend/.gitignore` exists; project-root `.gitignore` should still be reviewed against the list in this section).

When implementing this section, remember the Streamlit `client/` no longer exists — README setup instructions should describe running `frontend/` (npm install + npm run dev) instead.

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
