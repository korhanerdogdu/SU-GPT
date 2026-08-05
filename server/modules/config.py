import os
from pathlib import Path
from dotenv import load_dotenv


SERVER_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SERVER_ROOT.parent if SERVER_ROOT.name == "server" else SERVER_ROOT

load_dotenv(PROJECT_ROOT / ".env")
load_dotenv()

APP_ENV = os.getenv("APP_ENV", "development").strip().lower()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL_NAME = os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").strip().lower()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_MODEL_NAME = os.getenv("MISTRAL_MODEL_NAME", "mistral-small-latest")
MISTRAL_BASE_URL = os.getenv("MISTRAL_BASE_URL", "https://api.mistral.ai/v1")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
# OPENROUTER_MODEL is the public configuration name. Keep the *_MODEL_NAME alias so provider
# code follows the same naming convention as the existing Groq and Mistral settings.
OPENROUTER_MODEL_NAME = os.getenv(
    "OPENROUTER_MODEL",
    os.getenv("OPENROUTER_MODEL_NAME", "deepseek/deepseek-v4-pro"),
)
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_HTTP_REFERER = os.getenv("OPENROUTER_HTTP_REFERER", "").strip()
OPENROUTER_APP_TITLE = os.getenv("OPENROUTER_APP_TITLE", "adviSU").strip()
OPENROUTER_REASONING_ENABLED = os.getenv("OPENROUTER_REASONING_ENABLED", "false").lower() == "true"
OPENROUTER_REASONING_EFFORT = os.getenv("OPENROUTER_REASONING_EFFORT", "medium").strip().lower()
OPENROUTER_EXPERIMENTAL_ENABLED = (
    os.getenv("OPENROUTER_EXPERIMENTAL_ENABLED", "false").strip().lower() == "true"
)

# Provider reliability. Fallback is intentionally opt-in: setting LLM_FALLBACK_PROVIDER=groq
# enables one OpenRouter/Mistral -> Groq hop. A flat provider list (rather than recursive calls)
# makes fallback loops impossible.
LLM_FALLBACK_PROVIDER = os.getenv("LLM_FALLBACK_PROVIDER", "").strip().lower()
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "60"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "2"))
LLM_RETRY_BASE_SECONDS = float(os.getenv("LLM_RETRY_BASE_SECONDS", "0.25"))
LLM_MAX_RETRY_AFTER_SECONDS = float(os.getenv("LLM_MAX_RETRY_AFTER_SECONDS", "10"))
LLM_CIRCUIT_FAILURE_THRESHOLD = int(os.getenv("LLM_CIRCUIT_FAILURE_THRESHOLD", "3"))
LLM_CIRCUIT_RECOVERY_SECONDS = float(os.getenv("LLM_CIRCUIT_RECOVERY_SECONDS", "30"))
# One output ceiling for both the provider adapter and application resource policy. The legacy
# LLM_MAX_TOKENS name remains the explicit override for backwards compatibility.
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", os.getenv("MAX_OUTPUT_TOKENS", "1024")))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))
CONVERSATION_RETENTION_DAYS = max(1, int(os.getenv("CONVERSATION_RETENTION_DAYS", "90")))
MAX_TRANSCRIPT_MESSAGES = max(2, int(os.getenv("MAX_TRANSCRIPT_MESSAGES", "200")))
COURSE_REVIEWS_ENABLED = os.getenv("COURSE_REVIEWS_ENABLED", "false").strip().lower() == "true"
COURSE_REVIEWS_RELEASE_APPROVED = (
    os.getenv("COURSE_REVIEWS_RELEASE_APPROVED", "false").strip().lower() == "true"
)
COURSE_REVIEW_HMAC_SECRET = os.getenv("COURSE_REVIEW_HMAC_SECRET", "")
COURSE_REVIEW_CONSENT_VERSION = os.getenv(
    "COURSE_REVIEW_CONSENT_VERSION", "course-review-v1"
).strip()
COURSE_REVIEW_DIGEST_NAMESPACE = os.getenv(
    "COURSE_REVIEW_DIGEST_NAMESPACE", "course-review-author-v1"
).strip()
RATE_LIMIT_BACKEND = os.getenv("RATE_LIMIT_BACKEND", "memory").strip().lower()
RATE_LIMIT_REDIS_URL = os.getenv("RATE_LIMIT_REDIS_URL", "").strip()
RATE_LIMIT_REDIS_NAMESPACE = os.getenv(
    "RATE_LIMIT_REDIS_NAMESPACE", "advisu:rate-limit:v1"
).strip()
DAILY_SPEND_LIMIT_USD = float(os.getenv("DAILY_SPEND_LIMIT_USD", "0") or "0")
PROVIDER_REQUEST_COST_RESERVATION_MICROUSD = int(
    os.getenv("PROVIDER_REQUEST_COST_RESERVATION_MICROUSD", "0") or "0"
)
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


def validate_provider_activation() -> None:
    """Keep the experimental provider separate from the production authorization boundary."""

    if LLM_PROVIDER != "openrouter":
        return
    if not OPENROUTER_EXPERIMENTAL_ENABLED:
        raise RuntimeError(
            "OpenRouter requires the explicit development-only experimental feature flag"
        )
    if APP_ENV in {"production", "prod"}:
        # This repository-fixed function accepts no caller-supplied manifest, scorer, threshold,
        # or verdict. In the current revision it deliberately has no PASS branch.
        from evaluation.provider_benchmark import canonical_final_decision_status

        decision = canonical_final_decision_status()
        if decision.get("verdict") != "PASS":
            raise RuntimeError(
                "OpenRouter production activation is blocked by the canonical final decision"
            )


def validate_production_security() -> None:
    """Refuse to boot production with development credentials or token secrets."""
    if APP_ENV not in {"production", "prod"}:
        return
    token_secret = os.getenv("AUTH_TOKEN_SECRET", "")
    abuse_secret = os.getenv("ABUSE_HASH_SECRET", "")
    unsafe = []
    if len(token_secret) < 32 or token_secret.startswith("replace_"):
        unsafe.append("AUTH_TOKEN_SECRET")
    if len(abuse_secret) < 32 or abuse_secret.startswith("replace_"):
        unsafe.append("ABUSE_HASH_SECRET")
    if ADMIN_PASSWORD in {"", "admin", "replace_with_a_strong_admin_password"}:
        unsafe.append("ADMIN_PASSWORD")
    if STUDENT_PASSWORD in {"", "student", "replace_with_a_strong_student_password"}:
        unsafe.append("STUDENT_PASSWORD")
    if COURSE_REVIEWS_ENABLED and len(COURSE_REVIEW_HMAC_SECRET.encode()) < 32:
        unsafe.append("COURSE_REVIEW_HMAC_SECRET")
    if COURSE_REVIEWS_ENABLED and not COURSE_REVIEWS_RELEASE_APPROVED:
        unsafe.append("COURSE_REVIEWS_RELEASE_APPROVED")
    if RATE_LIMIT_BACKEND != "redis":
        unsafe.append("RATE_LIMIT_BACKEND")
    if RATE_LIMIT_BACKEND == "redis" and not RATE_LIMIT_REDIS_URL:
        unsafe.append("RATE_LIMIT_REDIS_URL")
    if DAILY_SPEND_LIMIT_USD > 0 and PROVIDER_REQUEST_COST_RESERVATION_MICROUSD < 1:
        unsafe.append("PROVIDER_REQUEST_COST_RESERVATION_MICROUSD")
    if unsafe:
        raise RuntimeError(
            "Unsafe production security configuration: " + ", ".join(sorted(unsafe))
        )
