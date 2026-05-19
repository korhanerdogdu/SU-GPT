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
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "sugpt")
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin")
SOURCES_DIR = os.getenv("SOURCES_DIR", str(PROJECT_ROOT / "sources"))
REVIEWS_DIR = os.getenv("REVIEWS_DIR", str(PROJECT_ROOT / "sources" / "reviews"))
EXAMS_DIR = os.getenv("EXAMS_DIR", str(PROJECT_ROOT / "sources" / "exams"))
CATALOG_DATA_DIR = os.getenv("CATALOG_DATA_DIR", str(Path.home() / "data"))
AUTO_INGEST_SOURCES = os.getenv("AUTO_INGEST_SOURCES", "false").lower() == "true"
AUTO_SEED_COURSES = os.getenv("AUTO_SEED_COURSES", "true").lower() == "true"
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "sentence-transformers/all-MiniLM-L12-v2",
)
CROSS_ENCODER_MODEL_NAME = os.getenv(
    "CROSS_ENCODER_MODEL_NAME",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
)
RETRIEVAL_CANDIDATE_K = int(os.getenv("RETRIEVAL_CANDIDATE_K", "20"))
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "6"))

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
