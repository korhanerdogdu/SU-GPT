import os
from pathlib import Path
from dotenv import load_dotenv


SERVER_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SERVER_ROOT.parent if SERVER_ROOT.name == "server" else SERVER_ROOT

load_dotenv(PROJECT_ROOT / ".env")
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL_NAME = os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").strip().lower()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_MODEL_NAME = os.getenv("MISTRAL_MODEL_NAME", "mistral-small-latest")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_store")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "su_knowledge")
CHROMA_HNSW_BATCH_SIZE = int(os.getenv("CHROMA_HNSW_BATCH_SIZE", "1000"))
CHROMA_HNSW_SYNC_THRESHOLD = int(
    os.getenv("CHROMA_HNSW_SYNC_THRESHOLD", "50000")
)
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "advisu")
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin")
SOURCES_DIR = os.getenv("SOURCES_DIR", str(PROJECT_ROOT / "sources"))
REVIEWS_DIR = os.getenv("REVIEWS_DIR", str(PROJECT_ROOT / "sources" / "reviews"))
EXAMS_DIR = os.getenv("EXAMS_DIR", str(PROJECT_ROOT / "sources" / "exams"))
CATALOG_DATA_DIR = os.getenv("CATALOG_DATA_DIR", str(Path.home() / "data"))
# Pre-chunked degree-requirement corpus that ships inside the repo (data/degree_requirements,
# data/minors, data/curricula). Defaults to <project>/data so it no longer depends on a
# machine-specific CATALOG_DATA_DIR.
DEGREE_DATA_DIR = os.getenv("DEGREE_DATA_DIR", str(PROJECT_ROOT / "data"))
CATALOG_TERM_CODES = [
    value.strip()
    for value in os.getenv("CATALOG_TERM_CODES", "202502").split(",")
    if value.strip()
]
AUTO_INGEST_SOURCES = os.getenv("AUTO_INGEST_SOURCES", "false").lower() == "true"
AUTO_SEED_COURSES = os.getenv("AUTO_SEED_COURSES", "true").lower() == "true"
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "sentence-transformers/all-MiniLM-L12-v2",
)
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "auto").strip().lower()
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "64"))
CROSS_ENCODER_MODEL_NAME = os.getenv(
    "CROSS_ENCODER_MODEL_NAME",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
)
CROSS_ENCODER_DEVICE = os.getenv("CROSS_ENCODER_DEVICE", "auto").strip().lower()
RETRIEVAL_CANDIDATE_K = int(os.getenv("RETRIEVAL_CANDIDATE_K", "20"))
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "6"))

# Retrieval modes (Section 3). The other modes exist so the evaluation track can ablate one
# component at a time; the product always runs this default and no longer exposes a picker.
#
# Default is `hybrid` (vector + BM25 fused), NOT `hybrid_rerank`, because that is what our own
# benchmark measured as best — chunk-level Recall@6 0.625 vs 0.438, reproduced at top-k 3/5/10,
# while also being ~283 ms/query faster because it skips the CrossEncoder. See
# docs/experiment_log.md "Finding 1a". Set DEFAULT_RETRIEVAL_MODE=hybrid_rerank to revert.
DEFAULT_RETRIEVAL_MODE = os.getenv("DEFAULT_RETRIEVAL_MODE", "hybrid").strip().lower()
ENABLE_RERANKING = os.getenv("ENABLE_RERANKING", "true").strip().lower() == "true"
BM25_TOP_K = int(os.getenv("BM25_TOP_K", "25"))
DENSE_TOP_K = int(os.getenv("DENSE_TOP_K", "20"))
FINAL_CONTEXT_TOP_K = int(os.getenv("FINAL_CONTEXT_TOP_K", str(RERANK_TOP_K)))
# Accepted by /ask/ from Section 3 onward; only wired to real behaviour in Section 4.
DEFAULT_PROMPT_STRATEGY = os.getenv("DEFAULT_PROMPT_STRATEGY", "basic").strip().lower()
DEFAULT_EXPERT_MODE = os.getenv("DEFAULT_EXPERT_MODE", "auto").strip().lower()

# Chunking (Section 2)
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
DOCUMENT_STORAGE_DIR = os.getenv(
    "DOCUMENT_STORAGE_DIR",
    str(SERVER_ROOT / "uploaded_documents"),
)


def require_env(name: str, value: str | None) -> str:
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value
