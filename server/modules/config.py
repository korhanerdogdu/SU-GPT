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
STUDENT_USERNAME = os.getenv("STUDENT_USERNAME", "student")
STUDENT_PASSWORD = os.getenv("STUDENT_PASSWORD", "student")
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
# Default is `hybrid_meta`: BM25F over the structured curriculum corpus, fused by RRF with
# multilingual-E5-small, then soft metadata/record-type boosts. On the 499-query held-out
# test split it reaches Recall@10 0.9419 against 0.5992 for the BM25 baseline
# (+0.343, 95% CI [+0.299, +0.389], p=0.0001) with no subgroup regression, and it replaced
# `hybrid` after the 2026-07-28 benchmark. See docs/retrieval_benchmark_report.md.
#
# Fallbacks, in order: set DEFAULT_RETRIEVAL_MODE=hybrid for the previous Chroma path,
# =bm25 for the plain lexical baseline, or ADVISU_LAB_DENSE=false to run the winner's
# lexical half only (Recall@10 0.9178) when the E5 model is unavailable.
DEFAULT_RETRIEVAL_MODE = os.getenv("DEFAULT_RETRIEVAL_MODE", "hybrid_meta").strip().lower()
ENABLE_RERANKING = os.getenv("ENABLE_RERANKING", "true").strip().lower() == "true"
BM25_TOP_K = int(os.getenv("BM25_TOP_K", "25"))
DENSE_TOP_K = int(os.getenv("DENSE_TOP_K", "20"))
FINAL_CONTEXT_TOP_K = int(os.getenv("FINAL_CONTEXT_TOP_K", str(RERANK_TOP_K)))
# `structured_lookup` tied for best exact-match accuracy (6/6) with all candidates in the
# 2026-07-24 strategy benchmark, and wins the documented tie-break because it also produces
# the table-oriented UI contract. Set `basic` to restore the historical prompt.
DEFAULT_PROMPT_STRATEGY = os.getenv("DEFAULT_PROMPT_STRATEGY", "structured_lookup").strip().lower()
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
