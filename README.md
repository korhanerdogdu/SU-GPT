# SU-GPT

SU-GPT is a course-aware Retrieval-Augmented Generation (RAG) assistant for Sabanci University course materials. It lets users upload academic documents, indexes them into a local ChromaDB vector store, retrieves relevant chunks for a question, reranks the retrieved evidence, and asks a Groq-hosted Llama model to answer with source grounding.

The project was originally based on RagBot, but the current codebase is structured as a React frontend plus a FastAPI backend.

## Current Features

- React + Vite + TypeScript frontend
- Tailwind CSS and shadcn/ui-style primitives
- Visual-only login and signup flow backed by `localStorage`
- FastAPI backend with open local-development CORS
- Multi-format document upload: PDF, PPTX, DOCX, Markdown, TXT, HTML, IPYNB, and JSON
- Persistent ChromaDB vector store
- Single ChromaDB collection: `su_knowledge`
- MongoDB source-of-truth collections for users, courses, source documents, upload batches, ingestion jobs, reviews, exams, and embedding cache
- Unified ingestion pipeline with deterministic chunk IDs and idempotent upserts
- Intent-aware RAG router for course, review, exam, graduation, and recommendation queries
- HuggingFace SentenceTransformers embeddings
- CrossEncoder reranking
- Groq Llama answer generation through LangChain
- Source-aware responses using page, slide, section, and Chroma chunk ID metadata
- Admin lifecycle flows for pending WhatsApp review uploads, exam/PDF ingest, and cascade source delete
- Bulk ingest scripts for all, courses, reviews, and exams

## Project Structure

```text
RagBot-SUGPT/
+-- frontend/                  # React + Vite frontend
|   +-- public/assets/          # Campus and logo assets
|   +-- src/
|   |   +-- components/chat/    # Chat UI components
|   |   +-- components/ui/      # Shared UI primitives
|   |   +-- contexts/           # AuthContext
|   |   +-- lib/                # API client and utilities
|   |   +-- pages/              # Login, signup, chat pages
|   |   +-- App.tsx             # Router and auth guards
|   |   +-- main.tsx            # React entrypoint
|   +-- package.json
|   +-- vite.config.ts
+-- server/                     # FastAPI backend
|   +-- main.py                 # API routes
|   +-- logger.py
|   +-- modules/
|   |   +-- config.py           # Environment-based configuration
|   |   +-- source_of_truth.py  # MongoDB source-of-truth helpers
|   |   +-- document_cleaner.py
|   |   +-- document_loaders.py # PDF/PPTX/DOCX/MD/TXT/HTML/IPYNB/JSON loaders
|   |   +-- load_vectorstore.py # Ingestion, chunking, Chroma upsert
|   |   +-- rag_router.py       # Intent-aware retrieval routing
|   |   +-- file_lifecycle.py   # Upload approval, exam ingest, cascade delete
|   |   +-- llm.py              # Groq LLM chain and prompt
|   |   +-- query_handlers.py   # Response and source formatting
|   |   +-- reranker.py         # CrossEncoder reranking
|   +-- scripts/
|   |   +-- bulk_ingest.py      # Idempotent all/course/review/exam ingest CLI
|   |   +-- run_python.sh       # Uses server/.venv when available
|   +-- requirements.txt
|   +-- uploaded_documents/     # Runtime uploads from /upload_documents/
|   +-- uploaded_pdfs/          # Legacy PDF upload directory
|   +-- chroma_store/           # Persistent ChromaDB data
+-- assets/                     # Project assets and reports
+-- package.json                # Root npm scripts for bulk ingest
+-- CLAUDE.md                   # Internal implementation guide
+-- README.md
```

## Backend Setup

Open a terminal in the project root and run:

Recommended, with a virtual environment:

```powershell
cd server
python -m venv myenv
.\myenv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m uvicorn main:app --reload
```

You can also install and run without a virtual environment:

```powershell
cd server
pip install -r requirements.txt
python -m uvicorn main:app --reload
```

If your system Python does not allow global installs, use the user install option:

```powershell
cd server
pip install --user -r requirements.txt
python -m uvicorn main:app --reload
```

The backend runs at:

```text
http://127.0.0.1:8000
```

Health check:

```text
http://127.0.0.1:8000/test
```

## Frontend Setup

Open a second terminal in the project root and run:

```powershell
cd frontend
npm install
npm run dev
```

The frontend runs at:

```text
http://localhost:5173
```

The frontend uses `http://127.0.0.1:8000` as the default backend URL. To override it, create `frontend/.env`:

```env
VITE_API_URL=http://127.0.0.1:8000
```

## Environment Variables

The backend reads configuration from `server/.env`.

Required:

```env
GROQ_API_KEY=your_groq_api_key
```

Optional:

```env
MONGO_URI=mongodb://localhost:27017
MONGO_DB_NAME=sugpt
GROQ_MODEL_NAME=llama-3.3-70b-versatile
CHROMA_PERSIST_DIR=./chroma_store
CHROMA_COLLECTION_NAME=su_knowledge
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L12-v2
CROSS_ENCODER_MODEL_NAME=cross-encoder/ms-marco-MiniLM-L-6-v2
RETRIEVAL_CANDIDATE_K=20
RERANK_TOP_K=6
CHUNK_SIZE=1000
CHUNK_OVERLAP=150
SOURCES_DIR=../sources
REVIEWS_DIR=../sources/reviews
EXAMS_DIR=../sources/exams
DOCUMENT_STORAGE_DIR=./uploaded_documents
CATALOG_DATA_DIR=/Users/selmanyilmaz/data
```

`DOCUMENT_STORAGE_DIR` is the current local storage adapter for raw uploaded files. The architecture keeps this behind `storageKey`, so the same MongoDB source document records can later point to MinIO/S3 object keys without changing the RAG contract.

## API Endpoints

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/test` | Basic backend health check |
| `POST` | `/upload_documents/` | Upload PDF, PPTX, DOCX, MD, TXT, HTML, IPYNB, or JSON files |
| `POST` | `/upload_pdfs/` | Legacy PDF-only upload endpoint |
| `POST` | `/ask/` | Ask a question over the indexed documents |
| `POST` | `/admin/whatsapp/upload` | Save WhatsApp export as a pending review upload batch |
| `POST` | `/admin/whatsapp/{batch_id}/confirm` | Approve/reject pending WhatsApp review data and trigger ingest |
| `POST` | `/admin/exams/upload` | Store exam PDF metadata and ingest it as `documentType=exam` |
| `DELETE` | `/sources/{source_id}` | Cascade delete one source from MongoDB, file storage, and ChromaDB |

The frontend currently uses:

- `POST /upload_documents/`
- `POST /ask/`
- `GET /test`

## RAG Pipeline

The production-ready direction is:

```text
source file
-> MongoDB sourceDocuments/uploadBatches/ingestionJobs
-> local storage path today, MinIO/S3 storageKey later
-> normalize text
-> PII clean where applicable
-> chunk
-> embed
-> ChromaDB su_knowledge upsert
-> MongoDB status update
-> ask router
-> metadata-filtered vector search
-> CrossEncoder rerank
-> prompt build
-> LLM answer with source chunk IDs
```

MongoDB is the source of truth. ChromaDB is treated as a reproducible vector index. If Chroma is deleted, it should be rebuilt from MongoDB records plus the raw files referenced by `storageKey`.

### Ingestion Pipeline

1. A source is created in MongoDB with `sourceId`, `type`, `fileName`, `storageKey`, `contentHash`, `status`, and `createdBy`.
2. An `ingestionJobs` record is started with `jobId`, `sourceId`, `status=processing`, `startedAt`, and `chunksCreated=0`.
3. The source file is parsed by `server/modules/document_loaders.py`.
4. Text is cleaned by `server/modules/document_cleaner.py`.
5. Chunks are created in `server/modules/load_vectorstore.py`.
6. Each chunk receives deterministic IDs in this format:

```text
sourceId:chunkIndex:contentHashPrefix
```

Example:

```text
course:9f0f8d1d0b77c0b7a8f2a41b:0:2cf24dba5fb0a30e
```

7. Chunks are upserted into the single Chroma collection `su_knowledge`.
8. MongoDB `sourceDocuments` and `ingestionJobs` are updated to `indexed` or `failed`.

### ChromaDB Storage Shape

All vectors go into one collection:

```text
collection_name = su_knowledge
```

Document type separation is done with metadata, not separate Chroma collections:

```json
{
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

Course and exam chunks use the same shape, with `documentType=course` or `documentType=exam`.

### Intent Detection and RAG Router

`server/modules/intent_detector.py` keeps the TF-IDF + Logistic Regression classifier for the original 7 intents:

- `mezuniyet_durumu`
- `ders_onerisi`
- `calisma_plani`
- `major_secimi`
- `alanda_ozellesme`
- `ders_ayrintisi`
- `diger`

`server/modules/rag_router.py` adds deterministic routing guards around that classifier:

- Hoca/course review questions, such as `Yücel Saygın zor mu?`, route to `documentType=review`.
- Exam questions, such as `CS412 final soruları var mı?`, route to `documentType=exam`.
- Graduation and course recommendation questions route to `documentType=course`.
- Unclear or general academic questions can multi-search `course`, `exam`, and `review`, then use CrossEncoder reranking.

The `/ask/` flow is now:

```text
question
-> TF-IDF/logistic intent + regex route guards
-> Chroma metadata filter
-> dense vector search
-> CrossEncoder rerank
-> source-labeled context documents
-> LangChain RetrievalQA prompt
-> answer + sources + source_chunk_ids + intent
```

## Supported Document Types

- `.pdf`
- `.pptx`
- `.docx`
- `.md`
- `.txt`
- `.html`
- `.htm`
- `.ipynb`
- `.json`

Metadata may include filename, document type, page number, slide number, section heading, and chunk ID.

## Bulk Ingest Scripts

The root `package.json` exposes idempotent ingestion commands. They use `server/.venv/bin/python` when it exists, otherwise `PYTHON` or `python3`.

```bash
npm run ingest:all
npm run ingest:courses
npm run ingest:reviews
npm run ingest:exams
```

The scripts are designed to be re-runnable. They use deterministic `sourceId`, `contentHash`, and chunk IDs, then upsert instead of blindly appending duplicate Chroma entries.

## File Lifecycle

- WhatsApp exports first enter MongoDB as `uploadBatches.status=pending`.
- Admin confirmation updates `instructorReviews` and triggers review ingestion into Chroma.
- Exam PDFs are saved through `storageKey`, recorded in `sourceDocuments` and `exams`, then ingested as `documentType=exam`.
- Deleting a source cascades through MongoDB source status, physical file deletion, and `collection.delete(where={"sourceId": sourceId})` in Chroma.

## Frontend Routes

| Route | Page | Notes |
| --- | --- | --- |
| `/login` | Login page | Visual-only authentication |
| `/signup` | Signup page | Visual-only authentication |
| `/` | Chat page | Requires local auth state |

Authentication is still lightweight for local development. The frontend stores `su-gpt-auth` in `localStorage`, while the backend now keeps admin/user and selected-course data in MongoDB for personalization and graduation/recommendation context.

## Development Notes

- Run backend and frontend in separate terminals.
- The first backend run may take longer because embedding and reranking models can be downloaded.
- Uploaded files and ChromaDB data are local runtime artifacts.
- CORS is currently permissive for local development.
- The current implemented retrieval mode is dense retrieval with CrossEncoder reranking.
- Future planned work includes BM25, hybrid retrieval, evaluation scripts, efficiency logging, ablations, and prompt experiments.
- Known next gaps: add real syllabus documents, improve final answer quality/prompt behavior, and collect benchmark results for retrieval and answer correctness.
