# SU-GPT

SU-GPT is a course-aware Retrieval-Augmented Generation (RAG) assistant for Sabanci University course materials. It lets users upload academic documents, indexes them into a local ChromaDB vector store, retrieves relevant chunks for a question, reranks the retrieved evidence, and asks a Groq-hosted Llama model to answer with source grounding.

The project was originally based on RagBot, but the current codebase is structured as a React frontend plus a FastAPI backend.

## Current Features

- React + Vite + TypeScript frontend
- Tailwind CSS and shadcn/ui-style primitives
- Visual-only login and signup flow backed by `localStorage`
- FastAPI backend with open local-development CORS
- Multi-format document upload: PDF, PPTX, DOCX, Markdown, and TXT
- Persistent ChromaDB vector store
- HuggingFace SentenceTransformers embeddings
- CrossEncoder reranking
- Groq Llama answer generation through LangChain
- Source-aware responses using page, slide, or section metadata

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
|   |   +-- document_cleaner.py
|   |   +-- document_loaders.py # PDF/PPTX/DOCX/MD/TXT loaders
|   |   +-- load_vectorstore.py # Ingestion, chunking, Chroma upsert
|   |   +-- llm.py              # Groq LLM chain and prompt
|   |   +-- query_handlers.py   # Response and source formatting
|   |   +-- reranker.py         # CrossEncoder reranking
|   +-- requirements.txt
|   +-- uploaded_documents/     # Runtime uploads from /upload_documents/
|   +-- uploaded_pdfs/          # Legacy PDF upload directory
|   +-- chroma_store/           # Persistent ChromaDB data
+-- assets/                     # Project assets and reports
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
GROQ_MODEL_NAME=llama-3.3-70b-versatile
CHROMA_PERSIST_DIR=./chroma_store
CHROMA_COLLECTION_NAME=ragbot_documents
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L12-v2
CROSS_ENCODER_MODEL_NAME=cross-encoder/ms-marco-MiniLM-L-6-v2
RETRIEVAL_CANDIDATE_K=20
RERANK_TOP_K=6
CHUNK_SIZE=1000
CHUNK_OVERLAP=150
```

## API Endpoints

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/test` | Basic backend health check |
| `POST` | `/upload_documents/` | Upload PDF, PPTX, DOCX, MD, or TXT files |
| `POST` | `/upload_pdfs/` | Legacy PDF-only upload endpoint |
| `POST` | `/ask/` | Ask a question over the indexed documents |

The frontend currently uses:

- `POST /upload_documents/`
- `POST /ask/`
- `GET /test`

## RAG Pipeline

1. The user uploads one or more supported documents from the frontend.
2. The frontend sends files to `POST /upload_documents/`.
3. The backend saves accepted files under `server/uploaded_documents/`.
4. Documents are parsed by file type.
5. Text is cleaned and split into chunks.
6. Chunks are embedded with the configured SentenceTransformers model.
7. Chunks and metadata are stored in ChromaDB.
8. For each question, dense similarity search retrieves candidate chunks.
9. A CrossEncoder reranks the candidates.
10. Top reranked chunks are formatted with source labels.
11. Groq Llama generates a concise answer grounded in the retrieved context.
12. The API returns both the answer and source labels.

## Supported Document Types

- `.pdf`
- `.pptx`
- `.docx`
- `.md`
- `.txt`

Metadata may include filename, document type, page number, slide number, section heading, and chunk ID.

## Frontend Routes

| Route | Page | Notes |
| --- | --- | --- |
| `/login` | Login page | Visual-only authentication |
| `/signup` | Signup page | Visual-only authentication |
| `/` | Chat page | Requires local auth state |

Authentication is currently frontend-only. It stores a local `su-gpt-auth` value in `localStorage`; there is no backend user system yet.

## Development Notes

- Run backend and frontend in separate terminals.
- The first backend run may take longer because embedding and reranking models can be downloaded.
- Uploaded files and ChromaDB data are local runtime artifacts.
- CORS is currently permissive for local development.
- The current implemented retrieval mode is dense retrieval with CrossEncoder reranking.
- Future planned work includes BM25, hybrid retrieval, evaluation scripts, efficiency logging, ablations, and prompt experiments.
