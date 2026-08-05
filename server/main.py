import asyncio
import json
import os
import re
import time
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI,UploadFile,File,Form,Request,HTTPException
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Literal
from langchain_core.retrievers import BaseRetriever
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from modules.load_vectorstore import (
    get_vectorstore,
    load_vectorstore,
    load_vectorstore_multi,
)
from modules.file_lifecycle import (
    cascade_delete_source,
    ingest_exam_upload,
)
from modules.intent_detector import get_intent
from modules.llm import answer_without_context_with_telemetry, detect_language, get_llm_chain
from modules.auth import AuthenticationError, Principal, bearer_token, issue_token, verify_token
from modules.confidence import ConfidenceSignals, assess as assess_confidence
from modules.guardrails import (
    assess_input,
    refusal_message,
    retrieval_boundary,
    retrieved_content_is_safe,
    validate_output,
)
from modules import content_safety, intents
from modules.language import analyze_language
from modules.resource_controls import (
    CONVERSATION_CREATIONS_PER_MINUTE,
    COURSE_REVIEW_REQUESTS_PER_MINUTE,
    MAX_INPUT_CHARS,
    MAX_OUTPUT_TOKENS,
    PROVIDER_REQUEST_COST_RESERVATION_MICROUSD,
    REQUEST_TIMEOUT_SECONDS,
    ResourceLimitError,
    STREAM_REQUESTS_PER_MINUTE,
    UPLOAD_REQUESTS_PER_MINUTE,
    DAILY_PROVIDER_REQUEST_QUOTA,
    controller as resource_controller,
)
from modules.rate_limit_backends import UsageReservation
from modules.localization import message as localized_message
from modules.query_handlers import query_chain
from modules.config import (
    ADMIN_PASSWORD,
    ADMIN_USERNAME,
    APP_ENV,
    AUTO_INGEST_SOURCES,
    AUTO_SEED_COURSES,
    CATALOG_DATA_DIR,
    COURSE_REVIEWS_ENABLED,
    COURSE_REVIEW_CONSENT_VERSION,
    COURSE_REVIEW_DIGEST_NAMESPACE,
    COURSE_REVIEW_HMAC_SECRET,
    DEFAULT_EXPERT_MODE,
    DEFAULT_PROMPT_STRATEGY,
    DEFAULT_RETRIEVAL_MODE,
    LLM_MAX_RETRIES,
    RERANK_TOP_K,
    RETRIEVAL_CANDIDATE_K,
    SOURCES_DIR,
    STUDENT_PASSWORD,
    STUDENT_USERNAME,
    validate_provider_activation,
    validate_production_security,
)
from modules.course_review_store import (
    CourseReviewPersistencePolicy,
    CourseReviewStorageError,
    CourseReviewStore,
    ModerationConflictError,
)
from modules.course_reviews import DuplicateReviewError, ReviewValidationError
from modules import retrieval_modes
from modules.catalog_retriever import retrieve_documents
from modules.mongodb import (
    ensure_database,
    ensure_user,
    get_academic_profile,
    get_completed_course_codes,
    get_user_course_context,
    get_user_courses,
    get_user_schedule,
    list_courses,
    mutate_user_courses_by_codes,
    seed_courses_from_catalog,
    set_academic_profile,
    set_user_courses,
    set_user_schedule,
    ScheduleRevisionConflict,
    course_reviews as course_review_collection,
)
from modules.course_commands import parse_course_history_command
from modules.profile_commands import parse_academic_profile_command, resolve_profile_update
from modules.export_utils import audit_rows, course_rows, rows_to_csv, rows_to_xlsx
from modules.response_formatter import (
    audit_structured_content,
    course_update_structured_content,
    audit_answer,
    graduation_plan_answer,
    ensure_summary_section,
    merge_structured_content,
    sanitize_student_answer,
)
from modules.rag_router import route_query
from modules import curriculum_registry
from modules import degree_audit
from modules import course_planner
from modules import major_advisor
from modules import schedule_planner
from modules import conversation_memory
from modules.retrieval_policy import build_metadata_filter, check_profile
from modules.source_indexer import ensure_sources_indexed
from logger import logger

app = FastAPI(title="adviSU — Retrieval-Augmented Academic Advising System")


@lru_cache(maxsize=1)
def _course_review_store() -> CourseReviewStore:
    """Construct the dedicated course-only store without touching legacy review collections."""

    return CourseReviewStore(
        collection=course_review_collection,
        hmac_secret=COURSE_REVIEW_HMAC_SECRET.encode("utf-8"),
        policy=CourseReviewPersistencePolicy(
            consent_policy_version=COURSE_REVIEW_CONSENT_VERSION,
            digest_namespace=COURSE_REVIEW_DIGEST_NAMESPACE,
            minimum_reviews=10,
        ),
    )


class StaticRetriever(BaseRetriever):
    documents: List[Document] = Field(default_factory=list)

    def _get_relevant_documents(self, query: str) -> List[Document]:
        return self.documents


class LoginPayload(BaseModel):
    username: str
    password: str


class CourseSelectionPayload(BaseModel):
    course_ids: list[str] = Field(default_factory=list)
    # optional per-course status: completed | enrolled | failed | withdrawn | transfer | exempted
    statuses: dict[
        str,
        Literal["completed", "enrolled", "failed", "withdrawn", "transfer", "exempted"],
    ] = Field(default_factory=dict)


class AcademicProfilePayload(BaseModel):
    major: str | None = None
    degree_code: str | None = None
    admission_term: str | None = None
    curriculum_term: str | None = None
    academic_year: int | None = Field(default=None, ge=1, le=6)
    minor_codes: list[str] | None = None


class ConversationUpdatePayload(BaseModel):
    title: str | None = None
    pinned: bool | None = None


class CourseReviewModerationPayload(BaseModel):
    state: Literal["approved", "rejected"]
    reason_codes: list[str] = Field(min_length=1, max_length=8)


class ScheduleSourcePayload(BaseModel):
    name: str = Field(default="Sabanci University SUIS (BannerWeb)", max_length=200)
    authority: str = Field(default="official", max_length=40)
    snapshot_date: str = Field(default="", max_length=40)
    scraped_at: str = Field(default="", max_length=60)


class ScheduleMeetingPayload(BaseModel):
    day_codes: list[str] = Field(default_factory=list, max_length=7)
    day_names_tr: list[str] = Field(default_factory=list, max_length=7)
    day_names_en: list[str] = Field(default_factory=list, max_length=7)
    start_time: str | None = Field(default=None, max_length=5)
    end_time: str | None = Field(default=None, max_length=5)
    location: str = Field(default="", max_length=300)
    instructors: str = Field(default="", max_length=300)
    status: str = Field(default="tba", max_length=40)
    start_date: str | None = Field(default=None, max_length=20)
    end_date: str | None = Field(default=None, max_length=20)


class ScheduleSectionPayload(BaseModel):
    course_id: str = Field(default="", max_length=30)
    title: str = Field(default="", max_length=300)
    section_title: str = Field(default="", max_length=300)
    crn: str = Field(default="", max_length=20)
    section: str = Field(default="", max_length=30)
    component: str = Field(default="Primary", max_length=60)
    component_code: str = Field(default="", max_length=20)
    component_label: str = Field(default="", max_length=60)
    instructors: str = Field(default="", max_length=300)
    locations: str = Field(default="", max_length=500)
    tba: bool = False
    meetings: list[ScheduleMeetingPayload] = Field(default_factory=list, max_length=14)
    source_url: str = Field(default="", max_length=500)


class ScheduleCoursePayload(BaseModel):
    course_id: str = Field(min_length=1, max_length=30)
    title: str = Field(default="", max_length=300)
    sections: list[ScheduleSectionPayload] = Field(min_length=1, max_length=10)


class WeeklySchedulePayload(BaseModel):
    schema_version: int = Field(default=1, ge=1, le=1)
    term: str = Field(min_length=6, max_length=6)
    term_label: str = Field(default="", max_length=100)
    generated_at: str = Field(default="", max_length=60)
    origin: str = Field(default="manual", max_length=40)
    # The string variant is accepted for the first schedule-page client, which sent
    # source="manual". Persistence canonicalizes it to origin + the official source object.
    source: ScheduleSourcePayload | str = Field(default_factory=ScheduleSourcePayload)
    courses: list[ScheduleCoursePayload] = Field(default_factory=list, max_length=30)
    crns: list[str] = Field(default_factory=list, max_length=100)
    not_offered: list[str] = Field(default_factory=list, max_length=50)
    unplaced: list[str] = Field(default_factory=list, max_length=50)
    conflicts: list[dict] = Field(default_factory=list, max_length=100)


class ScheduleSavePayload(BaseModel):
    schedule: WeeklySchedulePayload
    expected_revision: int | None = Field(default=None, ge=0)


GRADUATION_INTENT_RE = re.compile(
    r"\b("
    r"mezuniyet|mezun|kredi|credit|credits|degree evaluation|degree audit|audit|"
    r"kalan ders|kalan kredi|requirements?|requirement|kategori|dağılım|dagilim|"
    r"hangi derslerim sayıldı|hangi derslerim sayildi"
    r")",
    re.IGNORECASE,
)

RECOMMENDATION_INTENT_RE = re.compile(
    r"\b("
    r"hangi dersleri alayım|hangi dersleri alayim|hangi dersi alayım|hangi dersi alayim|"
    r"ders öner|ders oner|öner|oner|recommend|recommendation|"
    r"gelecek dönem|gelecek donem|next semester|ders programı|ders programi|"
    r"program öner|program oner|program oluştur|program olustur|schedule|"
    r"kolay|rahat|zor|ağır|agir|yoğun|yogun"
    r")",
    re.IGNORECASE,
)

EXPLICIT_RECOMMENDATION_RE = re.compile(
    r"\b("
    r"hangi dersleri alayım|hangi dersleri alayim|hangi dersi alayım|hangi dersi alayim|"
    r"ders öner|ders oner|öner|oner|recommend|recommendation|"
    r"gelecek dönem|gelecek donem|next semester|ders programı|ders programi|"
    r"program öner|program oner|program oluştur|program olustur|schedule"
    r")",
    re.IGNORECASE,
)

# Menu option 5: a real, time-tabled, conflict-free weekly schedule (with CRNs), as opposed to the
# times-less course list (ders_onerisi / "Hangi dersleri alayım"). Keyed off the word "program" /
# "schedule" / "takvim" / "çakışmasız" — which the times-less list phrasing deliberately avoids.
SCHEDULE_INTENT_RE = re.compile(
    r"\b("
    r"ders programı|ders programi|haftalık program|haftalik program|haftalık ders|haftalik ders|"
    r"saatli program|saatli ders|çakışmasız|cakismasiz|program yap|program oluştur|program olustur|"
    r"program çıkar|program cikar|program öner|program oner|takvim|weekly schedule|class schedule|"
    r"schedule"
    r")",
    re.IGNORECASE,
)

# These two starters deliberately use natural student language instead of implementation labels.
# They must remain deterministic even though their wording overlaps graduation/recommendation
# keywords: "this term" opens the CRN-bearing timetable, while "until I graduate" keeps the
# existing prerequisite-aware course-plan flow.
CURRENT_TERM_SCHEDULE_RE = re.compile(
    r"\b(?:bu|içinde bulunduğum|icinde bulundugum)\s+dönem\s+hangi\s+dersleri?\s+"
    r"(?:alayım|alayim|almalıyım|almaliyim)\b|\bwhat\s+courses?\s+should\s+i\s+take\s+this\s+term\b",
    re.IGNORECASE,
)
UNTIL_GRAD_RECOMMENDATION_RE = re.compile(
    r"\bmezun\s+olana\s+kadar\s+hangi\s+dersleri?\s+(?:alayım|alayim|almalıyım|almaliyim)\b|"
    r"\bwhat\s+courses?\s+should\s+i\s+take\s+until\s+i\s+graduate\b",
    re.IGNORECASE,
)

HEAVY_LOAD_RE = re.compile(
    r"\b(yoğun|yogun|ağır|agir|yüksek\s+tempo|yuksek\s+tempo|heavy|intensive)\b",
    re.IGNORECASE,
)
HEAVY_NEGATION_RE = re.compile(
    r"\b(?:yoğun|yogun|ağır|agir|heavy|intensive)\s+(?:olmasın|olmasin|istemiyorum|değil|degil|not)\b|"
    r"\b(?:not|don't|do\s+not)\s+(?:heavy|intensive)\b",
    re.IGNORECASE,
)
COURSE_COUNT_RE = re.compile(r"\b([1-8])\s*(?:tane\s*)?(?:ders|courses?)\b", re.IGNORECASE)
COURSE_COUNT_WORD_RE = re.compile(
    r"\b(bir|iki|üç|uc|dört|dort|beş|bes|altı|alti|yedi|sekiz|"
    r"one|two|three|four|five|six|seven|eight)\s+(?:tane\s*)?(?:ders|courses?)\b",
    re.IGNORECASE,
)
COURSE_COUNT_WORDS = {
    "bir": 1, "iki": 2, "üç": 3, "uc": 3, "dört": 4, "dort": 4,
    "beş": 5, "bes": 5, "altı": 6, "alti": 6, "yedi": 7, "sekiz": 8,
    "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8,
}
PAST_COURSE_COUNT_RE = re.compile(
    r"^\s*(?:aldım|aldim|tamamladım|tamamladim|geçtim|gectim|bitirdim|"
    r"completed|finished|passed|took)\b",
    re.IGNORECASE,
)


def _recommendation_preferences(question: str) -> tuple[int, int, bool]:
    """Return (soft course target, minimum SU credits, strict course-count limit).

    Product invariant: a normal plan is at least 15 SU; an explicitly heavy plan is at least
    18 SU. Only an explicit request for four-or-fewer courses is allowed to override the credit
    floor, because the student's requested course-count cap is then the stronger constraint.
    """
    text = question or ""
    numeric = COURSE_COUNT_RE.search(text)
    word = COURSE_COUNT_WORD_RE.search(text)
    count_match = numeric or word
    requested_count = int(numeric.group(1)) if numeric else (
        COURSE_COUNT_WORDS.get(word.group(1).lower()) if word else None
    )
    # "4 ders aldım" describes history, not a four-course request. Only the request form is
    # allowed to relax the 15-SU product floor.
    if count_match and PAST_COURSE_COUNT_RE.search(text[count_match.end():]):
        requested_count = None
    heavy = bool(HEAVY_LOAD_RE.search(text)) and not HEAVY_NEGATION_RE.search(text)
    if requested_count is not None and requested_count <= 4:
        return requested_count, 0, True
    return requested_count or (6 if heavy else 5), 18 if heavy else 15, False

STUDY_PLAN_INTENT_RE = re.compile(
    r"\b("
    r"nasıl çalış|nasil calis|çalışma planı|calisma plani|çalışma plan|calisma plan|"
    r"sınavlarına nasıl|sinavlarina nasil|hazırlanılır|hazirlanilir|"
    r"A ile geç|a ile gec|geçmek için|gecmek icin|study plan"
    r")",
    re.IGNORECASE,
)

MAJOR_SELECTION_INTENT_RE = re.compile(
    r"\b("
    r"hangi bölüm|hangi bolum|bölümü seç|bolumu sec|major seç|major sec|"
    r"ana dal|anadal|cs mi|bilgisayar bilimleri seçmek|endüstri mi|endustri mi"
    r")",
    re.IGNORECASE,
)

SPECIALIZATION_INTENT_RE = re.compile(
    r"\b("
    r"özelleş|ozelles|uzmanlaş|uzmanlas|alt dal|yönel|yonel|"
    r"nlp mi|security mi|data alanında|data alaninda|yapay zeka alanında|yapay zeka alaninda"
    r")",
    re.IGNORECASE,
)

MINOR_INTENT_RE = re.compile(
    r"\b(minor|yandal|yan dal|yan-dal)\b",
    re.IGNORECASE,
)

# "Kalan üniversite derslerim neler?" — a personal listing of still-missing first-year University
# Courses. Requires both the "university course" phrase AND a listing/remaining cue so a generic
# "üniversite dersleri nedir?" definition question does not trigger the deterministic list.
_UNIVERSITY_PHRASE_RE = re.compile(r"(üniversite|universite|university)\s*(ders|course)", re.IGNORECASE)
_LISTING_CUE_RE = re.compile(
    r"\b(kalan|kalanlar|kalanları|eksik|neler|nedir listele|listele|hangileri|hangi|kaldı|"
    r"remaining|missing|left|which|what)\b",
    re.IGNORECASE,
)


def _is_university_courses_query(question: str) -> bool:
    q = question or ""
    return bool(_UNIVERSITY_PHRASE_RE.search(q) and _LISTING_CUE_RE.search(q))

COURSE_DETAIL_INTENT_RE = re.compile(
    r"\b("
    r"kim veriyor|hoca|hocanın|hocanin|syllabus|içeriği|icerigi|"
    r"dersin içeriği|dersin icerigi|notlandırması|notlandirmasi|"
    r"prerequisite|önkoşul|onkosul|workload|zor mu"
    r")",
    re.IGNORECASE,
)

INTEREST_ALIASES = {
    "ai": (
        "AI",
        (
            "ai",
            "artificial intelligence",
            "yapay zeka",
            "yapay zekâ",
            "machine learning",
            "ml",
            "deep learning",
            "dl",
        ),
    ),
    "data": (
        "Data Science",
        ("data", "data science", "veri bilimi", "veri analitiği", "veri analitigi"),
    ),
    "nlp": (
        "NLP",
        (
            "nlp",
            "natural language",
            "doğal dil",
            "dogal dil",
            "chatbot",
            "duygu analizi",
            "sentiment",
            "language model",
            "dil model",
            "llm",
            "llms",
            "metin işleme",
            "metin isleme",
        ),
    ),
    "security": (
        "Security",
        ("security", "cybersecurity", "siber güvenlik", "siber guvenlik", "güvenlik", "guvenlik"),
    ),
    "systems": (
        "Systems",
        ("systems", "sistem", "distributed", "cloud", "operating systems", "os"),
    ),
    "web": (
        "Web",
        ("web", "frontend", "backend", "full stack", "full-stack", "uygulama geliştirme", "uygulama gelistirme"),
    ),
}

INTEREST_AREAS = {
    alias
    for _, aliases in INTEREST_ALIASES.values()
    for alias in aliases
}

COURSE_CODE_RE = re.compile(r"\b[A-Z]{2,5}\s*\d{3,5}\b", re.IGNORECASE)
RECOMMENDATION_TERM = "202601"

INTEREST_COURSE_HINTS = {
    "nlp": ["CS445", "CS455", "CS412", "CS415", "DSA440", "EE417", "ECON494", "CS460", "CS48004"],
    "ai": ["CS404", "CS412", "CS415", "CS455", "DSA440", "EE417", "CS48011", "ECON495"],
    "data": ["CS412", "CS445", "DSA301", "DSA428", "DSA440", "DSA473", "ECON494", "ECON495", "OPIM390"],
    "security": ["CS432", "CS437", "CS438", "CS48008", "CS411", "CS408"],
    "systems": ["CS307", "CS401", "CS403", "CS406", "CS408", "CS436", "CS460"],
    "web": ["CS306", "CS308", "CS310", "CS442", "CS449", "CS48004", "VA325"],
}

NON_ACADEMIC_FALLBACK = (
    "Bu konuda size yardımcı olamıyorum, akademik konularda sorular sorabilirsiniz."
)


def _is_graduation_intent(question: str) -> bool:
    return bool(GRADUATION_INTENT_RE.search(question or ""))


def _is_recommendation_intent(question: str) -> bool:
    return (
        bool(RECOMMENDATION_INTENT_RE.search(question or ""))
        or _is_short_interest_area(question)
        or _is_short_interest_phrase(question)
    )


def _is_short_interest_area(question: str) -> bool:
    normalized = re.sub(r"\s+", " ", (question or "").strip().lower())
    return normalized in INTEREST_AREAS


def _extract_interest_key(question: str) -> str | None:
    normalized = re.sub(r"\s+", " ", (question or "").strip().lower())
    for key, (_, aliases) in INTEREST_ALIASES.items():
        if normalized in aliases:
            return key
    for key, (_, aliases) in INTEREST_ALIASES.items():
        if any(re.search(rf"\b{re.escape(alias)}\b", normalized) for alias in aliases):
            return key
    return None


def _interest_label(question: str) -> str | None:
    key = _extract_interest_key(question)
    if not key:
        return None
    return INTEREST_ALIASES[key][0]


def _is_short_interest_phrase(question: str) -> bool:
    normalized = re.sub(r"\s+", " ", (question or "").strip().lower())
    if len(normalized) > 120 or not _extract_interest_key(normalized):
        return False
    non_recommendation_terms = re.compile(
        r"\b(nedir|ne demek|açıkla|acikla|kim veriyor|hoca|nasıl çalış|nasil calis|"
        r"çalışmalıyım|calismaliyim|syllabus|içeriği|icerigi)\b",
        re.IGNORECASE,
    )
    return not non_recommendation_terms.search(normalized)


def _has_interest_area(question: str) -> bool:
    return _extract_interest_key(question) is not None


def _is_course_detail_like(question: str) -> bool:
    if COURSE_DETAIL_INTENT_RE.search(question or ""):
        return True
    if not COURSE_CODE_RE.search(question or ""):
        return False
    if EXPLICIT_RECOMMENDATION_RE.search(question or ""):
        return False
    graduation_terms = re.compile(
        r"\b(mezuniyet|mezun|kalan|eksik|degree audit|degree evaluation|kategori|dağılım|dagilim)\b",
        re.IGNORECASE,
    )
    return not graduation_terms.search(question or "")


def _resolve_intent(
    question: str,
    detected_intent: str,
    working_context: dict | None = None,
) -> str:
    previous_intent = (working_context or {}).get("last_intent")
    normalized = re.sub(r"\s+", " ", (question or "").strip().lower())
    graduation_followups = (
        "yani ne almam lazım",
        "yani ne almaliyim",
        "ne almam lazım",
        "ne almaliyim",
        "what do i need",
        "what should i take then",
    )
    if previous_intent in {intents.GRADUATION_STATUS, intents.GRADUATION_PLAN, "mezuniyet_durumu"} and any(
        phrase in normalized for phrase in graduation_followups
    ):
        return intents.GRADUATION_PLAN
    if previous_intent in {intents.GRADUATION_STATUS, intents.GRADUATION_PLAN, "mezuniyet_durumu"} and re.search(
        r"\b(?:sonraki|gelecek)\s+dönem\b|\bnext semester\b", normalized
    ):
        return intents.GRADUATION_PLAN
    if MINOR_INTENT_RE.search(question or ""):
        return intents.MINOR
    if CURRENT_TERM_SCHEDULE_RE.search(question or ""):
        return intents.WEEKLY_SCHEDULE
    if UNTIL_GRAD_RECOMMENDATION_RE.search(question or ""):
        return intents.COURSE_RECOMMENDATION
    if _is_graduation_intent(question) and not _is_course_detail_like(question):
        return intents.GRADUATION_STATUS
    if _is_course_detail_like(question):
        return intents.COURSE_DETAIL
    if STUDY_PLAN_INTENT_RE.search(question or ""):
        return intents.STUDY_PLAN
    if MAJOR_SELECTION_INTENT_RE.search(question or ""):
        return intents.MAJOR_SELECTION
    if SPECIALIZATION_INTENT_RE.search(question or ""):
        return intents.SPECIALIZATION
    # A timetabled "program/schedule" request (option 5) is checked before the times-less course
    # list (option 2) so the word "program" routes to the weekly-schedule builder.
    if SCHEDULE_INTENT_RE.search(question or ""):
        return intents.WEEKLY_SCHEDULE
    if _is_recommendation_intent(question):
        return intents.COURSE_RECOMMENDATION
    return intents.to_canonical(detected_intent)


def _recommendation_question_for_llm(question: str, taken_codes: list[str]) -> str:
    taken = ", ".join(taken_codes) if taken_codes else "none"
    interest = _interest_label(question) or "the requested area"
    return (
        "COURSE_RECOMMENDATION_MODE. The user wants course recommendations, not a graduation "
        "audit. Do not open with graduation status or 125/125 credits. "
        f"Interest area already given: {interest}; do not ask again — produce the plan now.\n"
        "DETERMINISTIC PLANNER IS AUTHORITATIVE. The context contains a block "
        "'[Source: Deterministic academic-stage planner (authoritative)]' with the student's "
        "academic stage, a CANDIDATE POOL, and FUTURE TARGETS. You MUST obey it:\n"
        "- Recommend ONLY courses from that CANDIDATE POOL. Never introduce a course that is not "
        "listed there.\n"
        "- Write each course EXACTLY as its 'CODE — Official Name' from that block. Never rename, "
        "translate, or invent a course title.\n"
        "- Place missing University Courses first, then missing 2XX required foundations, then "
        "eligible electives that fit the interest.\n"
        "- If the block says the student is sophomore level or below, put NO 4XX course in the "
        "current-semester plan. List advanced interest courses under a separate 'Gelecek hedefler' "
        "(future targets) section only, and briefly say which prerequisites are needed first.\n"
        "- Do NOT output any instructor, weekday, time, room or section information, and do not "
        "write placeholders for them.\n"
        f"- Never recommend an already-taken course: {taken}.\n"
        "Give the current-semester plan as a short markdown list of 'CODE — Name — one-line why', "
        "then the 'Gelecek hedefler' section if any. Keep the explanation in the user's language.\n"
        f"Original user question: {question}"
    )


def _expanded_recommendation_question(question: str) -> str:
    interest = _interest_label(question)
    if not interest:
        return question
    if _is_short_interest_area(question) or _is_short_interest_phrase(question):
        return (
            f"{interest} alanında gelecek dönem için 5 derslik ders programı öner. "
            "Öğrencinin aldığı dersleri tekrar önerme. "
            f"Kullanıcının ilgi ayrıntısı: {question}"
        )
    return (
        f"{question}\n\n"
        f"Çıkarılan ilgi alanı: {interest}. Bu alan için 5 derslik program öner; "
        "alınmış dersleri tekrar önerme."
    )


def _recommendation_retrieval_query(question: str) -> str:
    interest_key = _extract_interest_key(question)
    hints = " ".join(INTEREST_COURSE_HINTS.get(interest_key or "", []))
    return (
        f"{_expanded_recommendation_question(question)} {hints} course recommendation "
        "Sabanci CS BSCS course catalog core electives area electives free electives "
        f"schedule instructors {RECOMMENDATION_TERM} AI NLP Data Web Systems Security"
    )


def _retrieval_query_for_intent(question: str, intent: str, program: str | None = None) -> str:
    if intent == intents.REVIEW:
        return question
    if intent == intents.EXAM:
        return f"{question} exam final midterm quiz past questions solutions assessment"
    if intent == intents.MINOR:
        return f"{question} minor program required courses core electives area electives"
    if intent == intents.COURSE_RECOMMENDATION:
        return _recommendation_retrieval_query(question)
    if intent == intents.GRADUATION_STATUS:
        prog = (program or "").strip()
        return (
            f"{question} {prog} degree requirements graduation "
            "university courses required courses core electives area electives free electives"
        )
    if intent == intents.STUDY_PLAN:
        return f"{question} course syllabus assignments exams study plan workload"
    if intent == intents.COURSE_DETAIL:
        return f"{question} course detail instructor syllabus schedule prerequisite workload"
    if intent in {intents.MAJOR_SELECTION, intents.SPECIALIZATION}:
        return f"{question} Sabanci program requirements course catalog specialization career"
    return question


def _intent_context_document(intent: str, taken_codes: list[str] | None = None) -> Document:
    if intent == intents.COURSE_RECOMMENDATION:
        taken = ", ".join(taken_codes or []) if taken_codes else "none"
        text = (
            "[Source: Request intent]\n"
            "Detected intent: course_recommendation / schedule planning. "
            "Do not produce graduation audit. Do not summarize completed credits. "
            "Never start with any sentence about checking graduation status. "
            f"Already taken course codes, strictly forbidden to recommend: {taken}. "
            "Use MongoDB profile only as a taken-course exclusion list and for personalization."
        )
    elif intent == intents.GRADUATION_STATUS:
        text = (
            "[Source: Request intent]\n"
            "Detected intent: graduation_status / graduation audit. Use official degree evaluation "
            "and degree requirement RAG sources with MongoDB student profile."
        )
    elif intent == intents.STUDY_PLAN:
        text = (
            "[Source: Request intent]\n"
            "Detected intent: study_plan. Give course-specific study guidance. "
            "Do not produce graduation audit unless explicitly requested."
        )
    elif intent == intents.COURSE_DETAIL:
        text = (
            "[Source: Request intent]\n"
            "Detected intent: course_detail. Answer the requested course detail "
            "such as instructor, schedule, syllabus, prerequisite or workload. Do not produce graduation audit."
        )
    elif intent == intents.REVIEW:
        text = (
            "[Source: Request intent]\n"
            "Detected intent: course review. The consent-based course-only feature is disabled; "
            "never use instructor ratings or private-chat material."
        )
    elif intent == intents.EXAM:
        text = (
            "[Source: Request intent]\n"
            "Detected intent: exam / past exam or assessment question. Use exam chunks and course context; "
            "do not invent unavailable questions or answers."
        )
    elif intent == intents.MAJOR_SELECTION:
        text = (
            "[Source: Request intent]\n"
            "Detected intent: major_selection. Compare programs and fit using catalog context. "
            "Do not produce graduation audit."
        )
    elif intent == intents.SPECIALIZATION:
        text = (
            "[Source: Request intent]\n"
            "Detected intent: specialization guidance. Use catalog and course context "
            "to guide subfield choice. Do not produce graduation audit."
        )
    else:
        text = (
            "[Source: Request intent]\n"
            f"Detected intent: {intent}. Answer only this academic intent using the retrieved context."
        )
    return Document(
        page_content=text,
        metadata={"source": "Request intent", "document_type": "request_intent", "intent": intent},
    )


def _audit_context_document(audit_result: dict) -> Document:
    """Wrap the deterministic degree-audit object as an authoritative, non-negotiable context source."""
    text = (
        "[Source: Deterministic degree audit engine (authoritative)]\n"
        "The following graduation audit was computed in backend code from the official "
        "requirement file and the student's completed courses. Treat these numbers as final; "
        "do NOT recompute or contradict them. Explain them clearly to the student.\n"
        + json.dumps(audit_result, ensure_ascii=False, indent=2)
    )
    return Document(
        page_content=text,
        metadata={"source": "Deterministic degree audit", "document_type": "degree_audit_result"},
    )


def _conversation_context_document(turns: list[dict[str, str]]) -> Document:
    compact = []
    for turn in turns[-3:]:
        compact.append(
            {
                "user": str(turn.get("user") or "")[:600],
                "assistant": str(turn.get("assistant") or "")[:900],
            }
        )
    serialized = json.dumps(compact, ensure_ascii=False)
    return Document(
        page_content=(
            "[Source: Recent conversation context]\n"
            "Use these turns only to resolve follow-up meaning. Official curriculum data and "
            "the deterministic audit remain authoritative.\n"
            + retrieval_boundary(serialized, "recent-conversation")
        ),
        metadata={
            "source": "Recent conversation",
            "document_type": "conversation_context",
        },
    )


def _course_mutation_answer(mutation: dict, *, language: str) -> str:
    updates = mutation.get("updates") or []
    missing = mutation.get("missing") or []
    if language == "en":
        lines = [
            (
                f"Updated your course history. You now have {mutation.get('course_count', 0)} "
                f"stored courses; {mutation.get('credit_eligible_course_count', 0)} currently "
                "count toward graduation credit."
            )
        ]
        if updates:
            lines.extend(
                f"- **{row.get('code')}**: {row.get('action')} ({row.get('status') or 'removed'})"
                for row in updates
            )
        if missing:
            lines.append("Not found in the course catalog: " + ", ".join(missing))
        lines.append(
            f"Current credit-eligible SU total: {mutation.get('total_su_credits', 0):g}."
        )
        return "\n".join(lines)

    lines = [
        (
            f"Ders geçmişin güncellendi. Şu anda {mutation.get('course_count', 0)} kayıtlı dersin "
            f"var; bunların {mutation.get('credit_eligible_course_count', 0)} tanesi mezuniyet "
            "kredisine sayılabilecek durumda."
        )
    ]
    if updates:
        lines.extend(
            f"- **{row.get('code')}**: {row.get('action')} ({row.get('status') or 'silindi'})"
            for row in updates
        )
    if missing:
        lines.append("Ders kataloğunda bulunamadı: " + ", ".join(missing))
    lines.append(
        f"Güncel, krediye sayılabilir SU toplamı: {mutation.get('total_su_credits', 0):g}."
    )
    return "\n".join(lines)


def _clean_code(code: str) -> str:
    return re.sub(r"\s+", "", code or "").upper()


def _display_code(code: str) -> str:
    cleaned = _clean_code(code)
    match = re.match(r"^([A-Z]+)(\d.+)$", cleaned)
    if not match:
        return cleaned
    return f"{match.group(1)} {match.group(2)}"


@lru_cache(maxsize=8)
def _catalog_rows_for_term(term: str) -> dict[str, dict]:
    catalog_path = Path(CATALOG_DATA_DIR) / term / "CS.jsonl"
    rows: dict[str, dict] = {}
    if not catalog_path.exists():
        return rows
    with catalog_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            code = _clean_code(f"{row.get('Major', '')}{row.get('Code', '')}")
            if code:
                rows[code] = row
    return rows


@lru_cache(maxsize=8)
def _schedule_rows_for_term(term: str) -> dict[str, list[dict]]:
    schedule_path = Path(CATALOG_DATA_DIR) / "schedule" / f"{term}.jsonl"
    rows: dict[str, list[dict]] = {}
    if not schedule_path.exists():
        return rows
    with schedule_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            course_id = _clean_code(str(row.get("course_id", "")))
            if course_id:
                rows.setdefault(course_id, []).append(row)
    return rows


def _meeting_summary(schedule_rows: list[dict]) -> str:
    parts: list[str] = []
    for row in schedule_rows[:2]:
        meetings = row.get("meetings") or []
        instructors = []
        meeting_bits = []
        for meeting in meetings:
            instructor = str(meeting.get("instructors") or "").replace("( P )", "").strip()
            if instructor and instructor not in instructors:
                instructors.append(instructor)
            days = str(meeting.get("days") or "").strip()
            time = str(meeting.get("time") or "").strip()
            if days or time:
                meeting_bits.append(" ".join(part for part in [days, time] if part))
        section = row.get("section")
        crn = row.get("crn")
        parts.append(
            "; ".join(
                part
                for part in [
                    f"section {section}" if section else "",
                    f"CRN {crn}" if crn else "",
                    f"instructor(s): {', '.join(instructors)}" if instructors else "",
                    f"meetings: {', '.join(meeting_bits)}" if meeting_bits else "",
                ]
                if part
            )
        )
    return " | ".join(part for part in parts if part)


def _recommendation_support_documents(question: str, taken_codes: list[str]) -> list[Document]:
    interest_key = _extract_interest_key(question)
    if not interest_key:
        return []

    taken_set = {_clean_code(code) for code in taken_codes}
    catalog_rows = _catalog_rows_for_term(RECOMMENDATION_TERM)
    schedule_rows = _schedule_rows_for_term(RECOMMENDATION_TERM)
    docs: list[Document] = []

    strategy_text = (
        "[Source: Course recommendation strategy]\n"
        f"Interest area: {INTEREST_ALIASES[interest_key][0]}.\n"
        f"Primary/adjacent course hints for this area: {', '.join(_display_code(code) for code in INTEREST_COURSE_HINTS[interest_key])}.\n"
        f"Already taken course codes, never recommend again: {', '.join(_display_code(code) for code in taken_set) if taken_set else 'none'}.\n"
        "If primary courses are already taken, explicitly say so and recommend adjacent/supportive untaken courses. "
        "Do not ask the user to choose a narrower subarea; produce the program now."
    )
    docs.append(
        Document(
            page_content=strategy_text,
            metadata={
                "source": "Course recommendation strategy",
                "document_type": "course_recommendation_strategy",
            },
        )
    )

    for code in INTEREST_COURSE_HINTS[interest_key]:
        cleaned = _clean_code(code)
        catalog = catalog_rows.get(cleaned)
        schedules = schedule_rows.get(cleaned, [])
        if not catalog and not schedules:
            continue
        status = "already_taken_do_not_recommend" if cleaned in taken_set else "eligible_candidate"
        title = (catalog or schedules[0]).get("Course_Name") or (catalog or schedules[0]).get("title") or ""
        summary = _meeting_summary(schedules)
        catalog_bits = []
        if catalog:
            catalog_bits.extend(
                [
                    f"major/list source: CS {RECOMMENDATION_TERM}",
                    f"category: {catalog.get('EL_Type')}",
                    f"SU credits: {catalog.get('SU_credit')}",
                    f"ECTS: {catalog.get('ECTS')}",
                    f"faculty: {catalog.get('Faculty')}",
                ]
            )
        text = (
            f"[Source: CS recommendation candidate {_display_code(cleaned)}]\n"
            f"Course: {_display_code(cleaned)} - {title}\n"
            f"Recommendation status: {status}\n"
            + ("\n".join(catalog_bits) + "\n" if catalog_bits else "")
            + (f"Schedule {RECOMMENDATION_TERM}: {summary}\n" if summary else f"Schedule {RECOMMENDATION_TERM}: not found in loaded schedule context\n")
        )
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": f"CS recommendation candidate {_display_code(cleaned)}",
                    "document_type": "course_recommendation_candidate",
                    "course_code": _display_code(cleaned),
                    "recommendation_status": status,
                },
            )
        )
    return docs


def _course_codes(courses: list[dict]) -> list[str]:
    codes = []
    for course in courses:
        code = str(course.get("code", "")).strip()
        if code and code not in codes:
            codes.append(code)
    return codes


def _compact_recommendation_user_context(username: str | None, selected: list[dict]) -> str:
    if not username:
        return ""
    if not selected:
        return (
            f"User profile from MongoDB: {username}.\n"
            "No taken/completed courses are stored. Use general recommendation rules."
        )
    lines = []
    for course in selected:
        code = str(course.get("code", "")).strip()
        title = str(course.get("title", "")).strip()
        su_credits = course.get("su_credits")
        detail = " - ".join(part for part in [code, title] if part)
        if su_credits is not None:
            detail = f"{detail} (SU {su_credits})"
        if detail:
            lines.append(detail)
    return (
        f"User profile from MongoDB: {username}.\n"
        "Intent: course recommendation only, not graduation audit.\n"
        "Taken/completed courses that must not be recommended again:\n"
        + "\n".join(f"- {line}" for line in lines)
    )


def _clean_recommendation_response(result: dict) -> dict:
    text = result.get("response", "")
    text = re.sub(
        r"^\s*Öncelikle mezuniyet durumunu[^.\n]*(?:\.|\n)\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"^\s*Mezuniyet durumunu[^.\n]*(?:\.|\n)\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    result["response"] = text.strip()
    return result


def _output_language_matches(text: str, expected: str) -> bool:
    """Treat signal-poor codes/tables as neutral; otherwise require the requested language."""

    analysis = analyze_language(text)
    return analysis.signal_token_count < 3 or analysis.response_language == expected


def _document_source_label(document: Document) -> str:
    metadata = document.metadata or {}
    source = str(metadata.get("source") or metadata.get("file_name") or "").strip()
    parts = [source] if source else []
    if metadata.get("page") is not None:
        parts.append(f"page {metadata['page']}")
    if metadata.get("slide") is not None:
        parts.append(f"slide {metadata['slide']}")
    if metadata.get("section"):
        parts.append(f"section '{metadata['section']}'")
    return ", ".join(parts)


def _source_list_is_authorized(returned: list[object], authorized: list[str]) -> bool:
    allowed = {" ".join(value.split()).casefold() for value in authorized if value}
    return all(
        " ".join(str(value).split()).casefold() in allowed
        for value in returned
        if str(value).strip()
    )


def _confidence_public(status: str) -> dict[str, str]:
    """Expose only a stable user-facing state, never scores, thresholds, or internal reasons."""

    return {"status": status}


def _confidence_abstention_message(status: str, language: str) -> str:
    key = status if status in {
        "profile_required",
        "curriculum_unavailable",
        "provider_unavailable",
        "cannot_verify",
        "safe_abstention",
    } else "cannot_verify"
    return localized_message(f"confidence.{key}", language)


def _format_for_context(docs: List[Document]) -> List[Document]:
    """Prepend a [Source: ...] header to each chunk so the LLM can ground & cite."""
    formatted: List[Document] = []
    for doc in docs:
        meta = doc.metadata or {}
        document_type = str(meta.get("document_type") or meta.get("documentType") or "").lower()
        if document_type in {"review", "whatsapp", "instructor_review"}:
            logger.warning("legacy private-review chunk quarantined")
            continue
        if not retrieved_content_is_safe(doc.page_content):
            logger.warning("retrieved chunk quarantined by instruction-boundary policy")
            continue
        source = meta.get("source") or meta.get("file_name") or "unknown"
        location_parts = []
        if meta.get("page") is not None:
            location_parts.append(f"page {meta['page']}")
        if meta.get("slide") is not None:
            location_parts.append(f"slide {meta['slide']}")
        if meta.get("section"):
            location_parts.append(f"section '{meta['section']}'")
        location = ", ".join(location_parts)
        header = f"[Source: {source}" + (f", {location}" if location else "") + "]"
        formatted.append(
            Document(
                page_content=f"{header}\n{retrieval_boundary(doc.page_content, str(meta.get('chunk_id') or source))}",
                metadata=meta,
            )
        )
    return formatted


def _dedupe_documents(docs: List[Document]) -> List[Document]:
    seen = set()
    deduped: List[Document] = []
    for doc in docs:
        meta = doc.metadata or {}
        key = meta.get("chunk_id") or meta.get("chunkId") or (meta.get("source"), doc.page_content[:120])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(doc)
    return deduped


def _retrieve_for_route(vectorstore, route, retrieval_query: str) -> List[Document]:
    if route.use_multi_search:
        docs: List[Document] = []
        for document_type in route.document_types:
            docs.extend(
                retrieve_documents(
                    vectorstore,
                    retrieval_query,
                    k=RETRIEVAL_CANDIDATE_K,
                    metadata_filter={"documentType": document_type},
                )
            )
        return _dedupe_documents(docs)
    return retrieve_documents(
        vectorstore,
        retrieval_query,
        k=RETRIEVAL_CANDIDATE_K,
        metadata_filter=route.metadata_filter,
    )

# allow frontend

_cors_origins = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ALLOWED_ORIGINS", "http://127.0.0.1:5173,http://localhost:5173"
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.on_event("startup")
async def startup_event():
    validate_provider_activation()
    validate_production_security()
    try:
        await ensure_database()
        if AUTO_SEED_COURSES:
            await seed_courses_from_catalog()
    except Exception:
        logger.exception("MongoDB startup initialization failed")
        if APP_ENV in {"production", "prod"}:
            # Authentication and conversation ownership rely on the unique database indexes.
            # Production must never accept traffic when those controls could not be established.
            raise

    if COURSE_REVIEWS_ENABLED:
        try:
            await _course_review_store().ensure_indexes()
        except Exception as exc:
            # An enabled privacy feature without its unique/suppression indexes is unsafe in
            # every environment.  Never silently downgrade to the in-memory prototype.
            logger.error(
                "course-review startup validation failed (error_class=%s)",
                type(exc).__name__,
            )
            raise

    if AUTO_INGEST_SOURCES:
        try:
            indexed = ensure_sources_indexed(SOURCES_DIR)
            if indexed:
                logger.info("indexed %d source material chunks", indexed)
        except Exception:
            logger.exception("source material indexing failed")

@app.middleware("http")
async def catch_exception_middleware(request:Request,call_next):
    try:
        return await asyncio.wait_for(call_next(request), timeout=REQUEST_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        return JSONResponse(status_code=504, content={"error": "Request timed out"})
    except ResourceLimitError as exc:
        status = 503 if exc.reason.startswith("resource_backend_") else 429
        headers = (
            {"Retry-After": str(exc.retry_after_seconds)}
            if status == 429 and exc.retry_after_seconds is not None
            else None
        )
        return JSONResponse(
            status_code=status,
            content={"error": "Request temporarily unavailable"},
            headers=headers,
        )
    except Exception as exc:
        logger.error("unhandled request failure (error_class=%s)", type(exc).__name__)
        return JSONResponse(status_code=500,content={"error":"Internal server error"})


def _principal(request: Request) -> Principal:
    try:
        return verify_token(bearer_token(request.headers.get("authorization")))
    except AuthenticationError as exc:
        raise HTTPException(status_code=401, detail="Authentication required") from exc


def _authorize_user(request: Request, username: str) -> Principal:
    principal = _principal(request)
    if principal.role != "admin" and principal.username != username:
        raise HTTPException(status_code=403, detail="Access denied")
    return principal


def _authorize_admin(request: Request) -> Principal:
    principal = _principal(request)
    if principal.role != "admin":
        raise HTTPException(status_code=403, detail="Administrator access required")
    return principal


def _resource_limit_http_exception(
    exc: ResourceLimitError,
    *,
    language: str = "en",
) -> HTTPException:
    backend_failure = exc.reason.startswith("resource_backend_")
    headers = None
    if not backend_failure and exc.retry_after_seconds is not None:
        headers = {"Retry-After": str(max(1, int(exc.retry_after_seconds)))}
    return HTTPException(
        status_code=503 if backend_failure else 429,
        detail=(
            localized_message("errors.internal", language)
            if backend_failure
            else localized_message("limits.request", language)
        ),
        headers=headers,
    )


def _check_operation_limit(
    request: Request,
    *,
    kind: str,
    identity: str | None,
    limit: int,
    language: str = "en",
) -> None:
    try:
        resource_controller.check_operation(
            kind,
            identity=identity,
            client_address=request.client.host if request.client else None,
            limit=limit,
        )
    except ResourceLimitError as exc:
        raise _resource_limit_http_exception(exc, language=language) from exc


async def _authorize_conversation(request: Request, session_id: str) -> Principal:
    principal = _principal(request)
    owner = await conversation_memory.conversation_owner(session_id)
    # Missing ownership metadata is not proof of access. Legacy/unowned conversations remain
    # available to administrators for repair, but fail closed for student identities.
    if principal.role != "admin" and owner != principal.username:
        raise HTTPException(status_code=403, detail="Access denied")
    return principal
    
@app.post("/upload_pdfs/")
async def upload_pdfs(request: Request, files:List[UploadFile]=File(...)):
    principal = _authorize_admin(request)
    _check_operation_limit(
        request, kind="upload", identity=principal.username, limit=UPLOAD_REQUESTS_PER_MINUTE
    )
    try:
        logger.info(f"recieved {len(files)} files")
        chunk_count = load_vectorstore(files)
        logger.info("documents added to chroma")
        return {"message":"Files processed and vectorstore updated","chunks":chunk_count}
    except Exception:
        logger.exception("Error during pdf upload")
        return JSONResponse(status_code=400, content={"error": "Upload rejected"})


@app.post("/upload_documents/")
async def upload_documents(request: Request, files: List[UploadFile] = File(...)):
    """Multi-format upload endpoint (Section 2).

    Accepts PDF, PPTX, DOCX, MD, and TXT files. Unsupported types are
    skipped and reported in the response.
    """
    principal = _authorize_admin(request)
    _check_operation_limit(
        request, kind="upload", identity=principal.username, limit=UPLOAD_REQUESTS_PER_MINUTE
    )
    try:
        logger.info(f"received {len(files)} document(s) for multi-format ingest")
        result = load_vectorstore_multi(files)
        logger.info(
            "multi-format ingest complete: %d chunks, %d accepted, %d skipped",
            result["chunks"],
            len(result["accepted_files"]),
            len(result["skipped_files"]),
        )
        return {
            "message": "Files processed and vectorstore updated",
            **result,
        }
    except Exception:
        logger.exception("Error during multi-format document upload")
        return JSONResponse(status_code=400, content={"error": "Upload rejected"})


@app.post("/admin/whatsapp/upload")
async def upload_whatsapp_chat(
    request: Request,
    file: UploadFile = File(...),
    username: str = Form("admin"),
    language: str = Form("en"),
):
    del file, username
    principal = _authorize_admin(request)
    _check_operation_limit(
        request,
        kind="upload",
        identity=principal.username,
        limit=UPLOAD_REQUESTS_PER_MINUTE,
        language=language,
    )
    raise HTTPException(
        status_code=410,
        detail=localized_message("course_reviews.private_chat_disabled", language),
    )


@app.post("/admin/whatsapp/{batch_id}/confirm")
async def confirm_whatsapp_upload(
    batch_id: str,
    request: Request,
    approved: bool = Form(True),
    username: str = Form("admin"),
    language: str = Form("en"),
):
    del batch_id, approved, username
    _authorize_admin(request)
    raise HTTPException(
        status_code=410,
        detail=localized_message("course_reviews.private_chat_disabled", language),
    )


@app.post("/admin/exams/upload")
async def upload_exam_pdf(
    request: Request,
    file: UploadFile = File(...),
    course_code: str = Form(""),
    year: str = Form(""),
    semester: str = Form(""),
    exam_type: str = Form(""),
    username: str = Form("admin"),
):
    principal = _authorize_admin(request)
    username = principal.username
    _check_operation_limit(
        request, kind="upload", identity=username, limit=UPLOAD_REQUESTS_PER_MINUTE
    )
    try:
        return ingest_exam_upload(
            file,
            course_code=course_code,
            year=year,
            semester=semester,
            exam_type=exam_type,
            uploaded_by=username,
        )
    except Exception:
        logger.exception("Error during exam upload")
        return JSONResponse(status_code=400, content={"error": "Upload rejected"})


@app.delete("/sources/{source_id}")
async def delete_source_document(source_id: str, request: Request, hard: bool = False):
    _authorize_admin(request)
    try:
        return cascade_delete_source(source_id, hard=hard)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception:
        logger.exception("Error during source cascade delete")
        return JSONResponse(status_code=500, content={"error": "Source operation failed"})


@app.post("/auth/login")
async def login(payload: LoginPayload, request: Request):
    try:
        resource_controller.check_login_attempt(
            username=payload.username,
            client_address=request.client.host if request.client else None,
        )
    except ResourceLimitError as exc:
        raise _resource_limit_http_exception(exc, language="en") from exc
    if payload.username == ADMIN_USERNAME and payload.password == ADMIN_PASSWORD:
        role = "admin"
    elif payload.username == STUDENT_USERNAME and payload.password == STUDENT_PASSWORD:
        role = "student"
    else:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    try:
        await ensure_user(payload.username, role=role)
    except Exception:
        logger.exception("Could not sync user to MongoDB")
    token, expires_at = issue_token(payload.username, role)
    return {
        "username": payload.username,
        "role": role,
        "access_token": token,
        "token_type": "bearer",
        "expires_at": expires_at,
    }


@app.get("/course-reviews/policy")
async def course_review_policy(language: str = "en"):
    """Public, identity-free policy metadata; the feature remains off by default."""
    selected = "tr" if language == "tr" else "en"
    return {
        "enabled": COURSE_REVIEWS_ENABLED,
        "course_only": True,
        "instructor_ratings_allowed": False,
        "private_chat_ingestion_allowed": False,
        "explicit_consent_required": True,
        "minimum_aggregate_reviews": 10,
        "rating_dimensions": [
            "difficulty",
            "workload",
            "learning_value",
            "organization",
            "overall_satisfaction",
        ],
        "message": localized_message(
            "course_reviews.enabled" if COURSE_REVIEWS_ENABLED else "course_reviews.disabled",
            selected,
        ),
    }


def _require_course_review_feature(language: str) -> CourseReviewStore:
    if not COURSE_REVIEWS_ENABLED:
        raise HTTPException(
            status_code=503,
            detail=localized_message("course_reviews.disabled", language),
        )
    try:
        return _course_review_store()
    except Exception:
        raise HTTPException(
            status_code=503,
            detail=localized_message("course_reviews.storage_unavailable", language),
        ) from None


@app.post("/course-reviews")
async def submit_course_review(payload: dict, request: Request, language: str = "en"):
    principal = _principal(request)
    _check_operation_limit(
        request,
        kind="course-review",
        identity=principal.username,
        limit=COURSE_REVIEW_REQUESTS_PER_MINUTE,
        language=language,
    )
    store = _require_course_review_feature(language)
    try:
        review = await store.submit(payload, author_token=principal.username)
    except DuplicateReviewError:
        raise HTTPException(
            status_code=409,
            detail=localized_message("course_reviews.duplicate", language),
        ) from None
    except ReviewValidationError:
        raise HTTPException(
            status_code=400,
            detail=localized_message("course_reviews.invalid", language),
        ) from None
    except CourseReviewStorageError:
        raise HTTPException(
            status_code=503,
            detail=localized_message("course_reviews.storage_unavailable", language),
        ) from None
    state = review.moderation_state
    return {
        "review": review.to_public_mapping(),
        "message": localized_message(f"course_reviews.{state}", language),
    }


@app.delete("/course-reviews/{course_code}")
async def delete_my_course_review(course_code: str, request: Request, language: str = "en"):
    principal = _principal(request)
    _check_operation_limit(
        request,
        kind="course-review-delete",
        identity=principal.username,
        limit=COURSE_REVIEW_REQUESTS_PER_MINUTE,
        language=language,
    )
    store = _require_course_review_feature(language)
    try:
        deleted = await store.hard_delete_mine(course_code, author_token=principal.username)
    except ReviewValidationError:
        raise HTTPException(
            status_code=400,
            detail=localized_message("course_reviews.invalid", language),
        ) from None
    except CourseReviewStorageError:
        raise HTTPException(
            status_code=503,
            detail=localized_message("course_reviews.storage_unavailable", language),
        ) from None
    if not deleted:
        raise HTTPException(
            status_code=404,
            detail=localized_message("course_reviews.not_found", language),
        )
    return {"deleted": True, "message": localized_message("course_reviews.deleted", language)}


@app.get("/course-reviews/{course_code}/aggregate")
async def get_course_review_aggregate(course_code: str, request: Request, language: str = "en"):
    principal = _principal(request)
    _check_operation_limit(
        request,
        kind="course-review-aggregate",
        identity=principal.username,
        limit=COURSE_REVIEW_REQUESTS_PER_MINUTE,
        language=language,
    )
    store = _require_course_review_feature(language)
    try:
        aggregate = await store.aggregate(course_code)
    except ReviewValidationError:
        raise HTTPException(
            status_code=400,
            detail=localized_message("course_reviews.invalid", language),
        ) from None
    except CourseReviewStorageError:
        raise HTTPException(
            status_code=503,
            detail=localized_message("course_reviews.storage_unavailable", language),
        ) from None
    if not aggregate.available:
        return {
            "available": False,
            "message": localized_message(
                "course_reviews.aggregate_suppressed",
                language,
                minimum=aggregate.minimum_required,
            ),
        }
    return {
        "available": True,
        "courseCode": aggregate.course_code,
        "reviewCount": aggregate.review_count,
        "averages": dict(aggregate.averages),
        "distributions": {
            name: dict(values) for name, values in aggregate.distributions.items()
        },
    }


@app.post("/course-reviews/{review_id}/moderation")
async def moderate_course_review(
    review_id: str,
    payload: CourseReviewModerationPayload,
    request: Request,
    language: str = "en",
):
    principal = _authorize_admin(request)
    _check_operation_limit(
        request,
        kind="course-review-moderate",
        identity=principal.username,
        limit=COURSE_REVIEW_REQUESTS_PER_MINUTE,
        language=language,
    )
    store = _require_course_review_feature(language)
    try:
        review = await store.moderate(
            review_id,
            state=payload.state,
            reason_codes=payload.reason_codes,
        )
    except (ReviewValidationError, ModerationConflictError):
        raise HTTPException(
            status_code=409,
            detail=localized_message("course_reviews.invalid", language),
        ) from None
    except CourseReviewStorageError:
        raise HTTPException(
            status_code=503,
            detail=localized_message("course_reviews.storage_unavailable", language),
        ) from None
    return {"review": review.to_public_mapping()}


@app.get("/courses/")
async def get_courses(search: str = "", limit: int = 200):
    return {"courses": await list_courses(search=search, limit=limit)}


@app.get("/schedule/sections")
async def search_schedule_sections(
    term: str | None = None,
    search: str = "",
    limit: int = 80,
    language: str = "tr",
):
    """Bounded search over the immutable official SUIS schedule snapshot."""
    if len(search) > 120:
        raise HTTPException(status_code=400, detail="search cannot exceed 120 characters")
    if language not in {"tr", "en"}:
        raise HTTPException(status_code=400, detail="language must be tr or en")
    try:
        return schedule_planner.search_sections_payload(
            term=term,
            search=search,
            limit=limit,
            language=language,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/schedule/courses/{course_code}/sections")
async def get_schedule_course_sections(
    course_code: str,
    term: str | None = None,
    language: str = "tr",
):
    """Return every lecture/lab/recitation option for one exact course."""
    if len(course_code) > 30:
        raise HTTPException(status_code=400, detail="course_code cannot exceed 30 characters")
    if language not in {"tr", "en"}:
        raise HTTPException(status_code=400, detail="language must be tr or en")
    try:
        return schedule_planner.course_sections_payload(
            course_code,
            term=term,
            language=language,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (FileNotFoundError, LookupError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/users/{username}/courses")
async def get_selected_courses(username: str, request: Request):
    _authorize_user(request, username)
    return {"courses": await get_user_courses(username)}


@app.put("/users/{username}/courses")
async def save_selected_courses(username: str, payload: CourseSelectionPayload, request: Request):
    _authorize_user(request, username)
    return {"courses": await set_user_courses(username, payload.course_ids, payload.statuses)}


@app.get("/users/{username}/schedule")
async def get_saved_schedule(username: str, request: Request):
    _authorize_user(request, username)
    return await get_user_schedule(username)


@app.put("/users/{username}/schedule")
async def save_user_schedule(username: str, payload: ScheduleSavePayload, request: Request):
    _authorize_user(request, username)
    try:
        schedule = schedule_planner.validate_schedule_payload(payload.schedule.model_dump())
        return await set_user_schedule(
            username,
            schedule,
            expected_revision=payload.expected_revision,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ScheduleRevisionConflict as exc:
        raise HTTPException(
            status_code=409,
            detail={"message": str(exc), "current": exc.current},
        ) from exc


@app.get("/users/{username}/profile")
async def get_profile(username: str, request: Request):
    _authorize_user(request, username)
    return {"profile": await get_academic_profile(username)}


@app.get("/users/{username}/usage")
async def get_usage(username: str, request: Request):
    """Expose the privacy-preserving provider quota used by the chat header.

    The underlying controller stores only versioned HMAC identifiers. Deterministic answers and
    refused requests reconcile their reservation back to zero, so the displayed number measures
    actual provider-backed questions instead of penalising free or blocked paths.
    """
    _authorize_user(request, username)
    try:
        state = resource_controller.usage(username=username)
    except ResourceLimitError as exc:
        raise _resource_limit_http_exception(exc) from exc
    now = datetime.now(timezone.utc)
    next_midnight = datetime.combine(
        now.date() + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc
    )
    used = int(state.provider_requests)
    return {
        "limit": DAILY_PROVIDER_REQUEST_QUOTA,
        "used": used,
        "remaining": max(0, DAILY_PROVIDER_REQUEST_QUOTA - used),
        "resets_at": next_midnight.isoformat().replace("+00:00", "Z"),
        "exempt": False,
        "allowed": used < DAILY_PROVIDER_REQUEST_QUOTA,
    }


@app.put("/users/{username}/profile")
async def put_profile(username: str, payload: AcademicProfilePayload, request: Request):
    _authorize_user(request, username)
    updated = await set_academic_profile(username, payload.model_dump(exclude_none=True))
    return {"profile": updated}


@app.get("/curricula/")
async def get_curricula():
    """Valid curriculum choices for the frontend selector (roadmap section 20)."""
    return {
        "programs": curriculum_registry.list_major_programs(),
        "majors": curriculum_registry.majors(),
        "minors": curriculum_registry.minors(),
    }


@app.get("/curricula/{program}")
async def get_program_curricula(program: str):
    return {"program": program.upper(), "curricula": curriculum_registry.curricula_for_program(program)}


@app.get("/users/{username}/degree-audit")
async def degree_audit_endpoint(username: str, request: Request):
    """Deterministic degree audit for the student's confirmed profile (roadmap section 14)."""
    _authorize_user(request, username)
    return await _degree_audit(username)


async def _degree_audit(username: str):
    profile = await get_academic_profile(username)
    program = (profile.get("major") or "").strip().upper()
    term = (profile.get("curriculum_term") or "").strip()
    if not program or not term:
        raise HTTPException(status_code=400, detail="Academic profile (major + curriculum_term) is not set.")
    completed = await get_completed_course_codes(username)
    return degree_audit.audit(program, term, completed)


def _tabular_download(
    rows: list[dict],
    *,
    file_format: str,
    filename: str,
    sheet_name: str,
) -> Response:
    normalized = file_format.strip().lower()
    if normalized == "csv":
        content = rows_to_csv(rows)
        media_type = "text/csv; charset=utf-8"
        extension = "csv"
    elif normalized == "xlsx":
        content = rows_to_xlsx(rows, sheet_name=sheet_name)
        media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        extension = "xlsx"
    else:
        raise HTTPException(status_code=400, detail="format must be csv or xlsx")
    return Response(
        content=content,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}.{extension}"'},
    )


@app.get("/users/{username}/courses/export")
async def export_selected_courses(username: str, request: Request, format: str = "csv"):
    _authorize_user(request, username)
    selected = await get_user_courses(username)
    return _tabular_download(
        course_rows(selected),
        file_format=format,
        filename=f"advisu-{username}-courses",
        sheet_name="Course History",
    )


@app.get("/users/{username}/degree-audit/export")
async def export_degree_audit(username: str, request: Request, format: str = "xlsx"):
    _authorize_user(request, username)
    audit_result = await _degree_audit(username)
    if audit_result.get("reliability") == "unavailable":
        raise HTTPException(status_code=404, detail=audit_result.get("message"))
    return _tabular_download(
        audit_rows(audit_result),
        file_format=format,
        filename=f"advisu-{username}-degree-audit",
        sheet_name="Degree Audit",
    )


@app.post("/ask/")
async def ask_question(
    request: Request,
    question: str = Form(...),
    username: str | None = Form(None),
    session_id: str | None = Form(None),
    # Section 3. All four are optional: a client that posts only `question` keeps the
    # previous behaviour (hybrid_rerank). prompt_strategy/expert_mode are accepted and
    # echoed now, but stay inert until Section 4 implements the prompts module.
    mode: str | None = Form(None),
    top_k: int | None = Form(None),
    prompt_strategy: str | None = Form(None),
    expert_mode: str | None = Form(None),
):
    retrieval_mode = retrieval_modes.normalize_mode(mode or DEFAULT_RETRIEVAL_MODE)
    context_top_k = min(max(int(top_k), 1), RETRIEVAL_CANDIDATE_K) if top_k else RERANK_TOP_K
    prompt_strategy = (prompt_strategy or DEFAULT_PROMPT_STRATEGY).strip().lower()
    expert_mode = (expert_mode or DEFAULT_EXPERT_MODE).strip().lower()
    original_question = question
    language = detect_language(original_question)
    course_mutation: dict | None = None
    course_update_content: dict | None = None
    usage_total_tokens = 0
    usage_cost_usd: float | None = None
    provider_attempted = False
    admission = None

    def _stamp(payload: dict) -> dict:
        """Every /ask response reports the configuration that produced it (needed by Section 5)."""
        payload.setdefault("mode", retrieval_mode)
        payload.setdefault("top_k", context_top_k)
        payload.setdefault("prompt_strategy", prompt_strategy)
        payload.setdefault("expert_mode", expert_mode)
        return payload

    if username:
        _authorize_user(request, username)
        owner = await conversation_memory.conversation_owner(session_id)
        if owner is not None and owner != username:
            raise HTTPException(status_code=403, detail="Access denied")
        if session_id and owner is None:
            _check_operation_limit(
                request,
                kind="conversation-create",
                identity=username,
                limit=CONVERSATION_CREATIONS_PER_MINUTE,
                language=language,
            )
    else:
        # Generic unauthenticated questions remain available, but they must be stateless;
        # otherwise a bearer-less caller could append to or later claim a guessed session ID.
        session_id = None

    try:
        admission = resource_controller.begin(
            username=username,
            client_address=request.client.host if request.client else None,
            reservation=UsageReservation(
                provider_requests=1,
                # Character counts intentionally overestimate tokens.  The bounded retrieval
                # context and retry ceiling prevent in-flight requests from crossing the daily
                # token budget before provider telemetry can be reconciled.
                tokens=(
                    min(len(original_question), MAX_INPUT_CHARS)
                    + context_top_k * 1_600
                    + MAX_OUTPUT_TOKENS
                ) * (LLM_MAX_RETRIES + 1),
                cost_microusd=PROVIDER_REQUEST_COST_RESERVATION_MICROUSD,
            ),
        )
    except ResourceLimitError as exc:
        raise _resource_limit_http_exception(exc, language=language) from exc

    try:
        logger.info("user query received (mode=%s, top_k=%s)", retrieval_mode, context_top_k)
        content_verdict = content_safety.classify(original_question)
        if content_verdict.blocked:
            return _stamp({
                "response": content_safety.response_for(content_verdict.category, language),
                "sources": [],
                "source_chunk_ids": [],
                "intent": intents.SAFETY_BLOCKED,
                "safety_category": content_verdict.category,
                "confidence": _confidence_public("safe_abstention"),
            })
        assessment = assess_input(original_question)
        if not assessment.allowed:
            message = refusal_message(assessment.category or "prompt_injection", language)
            return _stamp({
                "response": message,
                "sources": [],
                "source_chunk_ids": [],
                "intent": "safety_refusal",
                "refusal_reason": assessment.category,
                "confidence": _confidence_public("safe_abstention"),
            })

        # Major and curriculum term can be changed conversationally and are persisted
        # immediately, just like the profile screen.
        profile_command = parse_academic_profile_command(original_question)
        if profile_command and username:
            current_profile = await get_academic_profile(username)
            profile_update, profile_error = resolve_profile_update(profile_command, current_profile)
            if profile_error:
                return _stamp({
                    "response": profile_error,
                    "summary": profile_error,
                    "sources": [],
                    "source_chunk_ids": [],
                    "intent": "academic_profile_update",
                })
            saved_profile = await set_academic_profile(username, profile_update)
            if profile_command.standalone:
                answer = (
                    f"Akademik profilin güncellendi: **{saved_profile.get('major')} / "
                    f"{saved_profile.get('degree_code')}**, müfredat dönemi "
                    f"**{saved_profile.get('curriculum_term')}**."
                )
                summary = (
                    f"Bölümün {saved_profile.get('major')}, müfredat dönemin "
                    f"{saved_profile.get('curriculum_term')} olarak kaydedildi."
                )
                answer = f"{answer}\n\n## Kısa Özet\n\n{summary}"
                await conversation_memory.append_turn(
                    session_id,
                    username=username,
                    question=original_question,
                    answer=answer,
                    intent="academic_profile_update",
                    term_code=saved_profile.get("curriculum_term"),
                    working_context_updates={"active_topic": "academic_profile"},
                )
                return _stamp({
                    "response": answer,
                    "summary": summary,
                    "structured_content": {
                        "kind": "academic_profile_update",
                        "tables": [{
                            "id": "academic-profile-update",
                            "title": "Akademik Profil",
                            "columns": [
                                {"key": "major", "label": "Bölüm"},
                                {"key": "degree_code", "label": "Program"},
                                {"key": "curriculum_term", "label": "Müfredat dönemi"},
                            ],
                            "rows": [saved_profile],
                        }],
                    },
                    "sources": [],
                    "source_chunk_ids": [],
                    "intent": "academic_profile_update",
                    "profile_updated": True,
                })

        # Explicit course-history statements are deterministic commands, not LLM guesses.
        # They update only the mentioned courses and preserve the rest of the profile.
        course_command = parse_course_history_command(original_question)
        if course_command and username:
            course_mutation = await mutate_user_courses_by_codes(
                username,
                course_command.course_codes,
                status=course_command.status,
                remove=course_command.is_remove,
            )
            course_update_content = course_update_structured_content(
                course_mutation.get("updates"),
                language=language,
            )
            if course_command.standalone:
                answer = _course_mutation_answer(course_mutation, language=language)
                answer, summary = ensure_summary_section(answer, language=language)
                await conversation_memory.append_turn(
                    session_id,
                    username=username,
                    question=original_question,
                    answer=answer,
                    intent="course_history_update",
                    course_id=course_command.course_codes[-1] if course_command.course_codes else None,
                    working_context_updates={"active_topic": "course_history"},
                )
                return _stamp(
                    {
                        "response": answer,
                        "summary": summary,
                        "structured_content": course_update_content,
                        "sources": [],
                        "source_chunk_ids": [],
                        "intent": "course_history_update",
                        "profile_updated": True,
                        "course_history": course_mutation,
                        "export_links": {
                            "csv": f"/users/{username}/courses/export?format=csv",
                            "xlsx": f"/users/{username}/courses/export?format=xlsx",
                        },
                    }
                )

        # LLM-only baseline: no retrieval, no student data, no sources, no profile gate.
        # It answers from model knowledge alone so the evaluation can measure what RAG adds.
        if retrieval_mode == "llm_only":
            # Provider I/O is synchronous internally. Keep it off the event loop; the provider
            # adapter enforces its own non-extendable wall-clock deadline inside this worker.
            usage_total_tokens = (len(question) + MAX_OUTPUT_TOKENS) * (LLM_MAX_RETRIES + 1)
            provider_attempted = True
            answer, provider_telemetry = await asyncio.to_thread(
                answer_without_context_with_telemetry, question
            )
            if provider_telemetry:
                if provider_telemetry.get("total_tokens") is not None:
                    usage_total_tokens = provider_telemetry.get("total_tokens")
                usage_cost_usd = provider_telemetry.get("cost_usd")
            raw_answer = str(answer or "")
            validation = validate_output(raw_answer, [])
            language_match = _output_language_matches(raw_answer, language)
            confidence = assess_confidence(
                ConfidenceSignals(
                    intent="llm_only",
                    evidence_required=False,
                    metadata_compatible=True,
                    output_schema_valid=bool(raw_answer.strip()),
                    output_language_match=language_match,
                    guardrail_passed=validation.safe,
                    citations_authorized=validation.safe,
                    provider_response_available=bool(raw_answer.strip()),
                    fallback_used=bool((provider_telemetry or {}).get("fallback_used")),
                )
            )
            if confidence.answer_allowed:
                answer = sanitize_student_answer(raw_answer)
            else:
                answer = (
                    refusal_message("unsafe_output", language)
                    if confidence.status == "safe_abstention"
                    else _confidence_abstention_message(confidence.status, language)
                )
            answer, summary = ensure_summary_section(answer, language=language)
            await conversation_memory.append_turn(
                session_id,
                username=username,
                question=original_question,
                answer=answer,
                intent="llm_only",
            )
            return _stamp({
                "response": answer,
                "summary": summary,
                "sources": [],
                "source_chunk_ids": [],
                "intent": "llm_only",
                "warning": retrieval_modes.LLM_ONLY_WARNING,
                "security_filtered": not validation.safe,
                "confidence": _confidence_public(confidence.status),
            })

        # Bounded session memory: resolve "this course" style follow-ups (roadmap section 19).
        working_context = await conversation_memory.get_working_context(
            session_id, username=username
        )
        question = conversation_memory.resolve_reference(question, working_context)
        prior_turns = await conversation_memory.recent_turns(
            session_id, username=username, limit=3
        )

        detected_intent = get_intent(question)
        resolved_intent = _resolve_intent(question, detected_intent, working_context)
        route = route_query(question, resolved_intent)
        intent = resolved_intent if resolved_intent == intents.MINOR else route.intent
        logger.info(
            "detected intent=%s resolved intent=%s route intent=%s document_types=%s confidence=%.3f",
            detected_intent,
            resolved_intent,
            intent,
            route.document_types,
            route.confidence,
        )

        # Student academic profile (roadmap section 8) drives profile-scoped retrieval.
        profile = await get_academic_profile(username) if username else {}
        program = (profile.get("major") or "").strip().upper()

        # ---- Major-selection mini-test (deterministic; never mixes RAG or graduation audit) ----
        # "Which major should I choose?" runs a bounded questionnaire and returns ONE definitive
        # best-fit program. Turn 1 asks the questions; the student's reply is scored in code.
        wc = working_context or {}
        if wc.get("major_quiz_pending") and major_advisor.looks_like_answers(question):
            rec = major_advisor.evaluate(question, current_major=program, language=language)
            await conversation_memory.append_turn(
                session_id, username=username, question=original_question, answer=rec.body,
                intent=intents.MAJOR_SELECTION,
                working_context_updates={"major_quiz_pending": False, "recommended_major": rec.best},
            )
            return _stamp({
                "response": rec.body, "summary": rec.summary,
                "sources": [], "source_chunk_ids": [], "intent": intents.MAJOR_SELECTION,
            })
        if (
            resolved_intent == intents.MAJOR_SELECTION
            and major_advisor.is_major_question(original_question)
            and not wc.get("major_quiz_pending")
        ):
            quiz = major_advisor.build_quiz(language)
            await conversation_memory.append_turn(
                session_id, username=username, question=original_question, answer=quiz,
                intent=intents.MAJOR_SELECTION,
                working_context_updates={"major_quiz_pending": True},
            )
            return _stamp({
                "response": quiz, "sources": [], "source_chunk_ids": [], "intent": intents.MAJOR_SELECTION,
            })

        graduation_plan_intent = resolved_intent == intents.GRADUATION_PLAN
        if graduation_plan_intent:
            intent = "graduation_plan"
        schedule_intent = resolved_intent == intents.WEEKLY_SCHEDULE
        if schedule_intent:
            intent = intents.WEEKLY_SCHEDULE
        graduation_intent = intent == intents.GRADUATION_STATUS
        recommendation_intent = intent == intents.COURSE_RECOMMENDATION
        if (
            recommendation_intent
            and not _has_interest_area(question)
            and not UNTIL_GRAD_RECOMMENDATION_RE.search(question or "")
        ):
            return _stamp({
                "response": localized_message("recommendation.ask_interest", language),
                "sources": [],
                "source_chunk_ids": [],
                "intent": intent,
            })
        if recommendation_intent and not (
            username and program and profile.get("curriculum_term")
        ):
            # A prose-only planner context is not an enforcement layer. Without an authenticated,
            # complete profile there is no safe completed-course/prerequisite/year basis, so no
            # model-authored course list may be returned.
            return _stamp({
                "response": localized_message("confidence.profile_required", language),
                "sources": [],
                "source_chunk_ids": [],
                "intent": intent,
                "profile_required": True,
                "confidence": _confidence_public("profile_required"),
            })
        if intent == "review":
            # Legacy instructor-review/chat-group data is outside the consent and product boundary.
            # Until the course-only review feature is explicitly released, do not retrieve or
            # summarize that corpus.
            return _stamp({
                "response": localized_message("course_reviews.disabled", language),
                "sources": [],
                "source_chunk_ids": [],
                "intent": "review_disabled",
            })

        # Missing-data / missing-profile safety for authoritative intents (roadmap section 15).
        gate = check_profile(intents.GRADUATION_STATUS if graduation_plan_intent else intent, profile)
        if not gate.ok:
            gate_status = "curriculum_unavailable" if gate.data_unavailable else "profile_required"
            return _stamp({
                "response": _confidence_abstention_message(gate_status, language),
                "sources": [],
                "source_chunk_ids": [],
                "intent": intent,
                "profile_required": bool(gate.missing_fields),
                "curriculum_unavailable": gate.data_unavailable,
                "confidence": _confidence_public(gate_status),
            })

        # Deterministic, balanced course recommendation. The recommendation must be correct and
        # student-specific, not LLM-invented: the planner reads the student's completed courses +
        # the official requirement file and returns a prerequisite-eligible, difficulty-balanced
        # set (no hallucinated courses, canonical names, choice pools de-duplicated). Only runs
        # only when the profile is complete enough to be authoritative. Incomplete profiles have
        # already abstained at the fail-closed profile gate above and never reach the LLM.
        if recommendation_intent and program and profile.get("curriculum_term") and username:
            completed = await get_completed_course_codes(username)
            interest_key = _extract_interest_key(question)
            interest_codes = list(INTEREST_COURSE_HINTS.get(interest_key or "", []))
            target_courses, minimum_su, strict_count = _recommendation_preferences(question)
            plan = course_planner.build_plan(
                program,
                profile["curriculum_term"],
                completed,
                interest_codes,
                target=target_courses,
                minimum_su_credits=minimum_su,
                exact_course_count=strict_count,
                academic_year=profile.get("academic_year"),
            )
            plan_errors = course_planner.validate_proposed_plan(
                [{"code": item.code} for item in plan.recommended],
                completed_codes=completed,
                stage=plan.stage,
            )
            if not plan.has_official_data:
                return _stamp({
                    "response": _confidence_abstention_message("curriculum_unavailable", language),
                    "sources": [],
                    "source_chunk_ids": [],
                    "intent": intents.COURSE_RECOMMENDATION,
                    "confidence": _confidence_public("curriculum_unavailable"),
                })
            if plan_errors:
                logger.error("deterministic recommendation validation failed (%s errors)", len(plan_errors))
                return _stamp({
                    "response": _confidence_abstention_message("cannot_verify", language),
                    "sources": [],
                    "source_chunk_ids": [],
                    "intent": intents.COURSE_RECOMMENDATION,
                    "confidence": _confidence_public("cannot_verify"),
                })
            body, summary = course_planner.render_plan(
                plan, language=language, interest_label=_interest_label(question)
            )
            rec_result = _stamp({
                "response": body,
                "summary": summary,
                "structured_content": course_planner.plan_structured_content(plan, language=language),
                "sources": ["Deterministic academic-stage planner"],
                "source_chunk_ids": [],
                "intent": intents.COURSE_RECOMMENDATION,
                "confidence": _confidence_public("verified"),
            })
            await conversation_memory.append_turn(
                session_id, username=username, question=original_question, answer=body,
                intent=intents.COURSE_RECOMMENDATION, term_code=profile.get("curriculum_term"),
                working_context_updates={"active_topic": "course_recommendation"},
            )
            return rec_result

        # Menu option 5: a REAL, conflict-free weekly timetable with CRNs. Builds the same balanced
        # course set as the recommendation, then attaches official SUIS section/time data and
        # resolves overlaps in code. The response carries the CRN list for copy + Excel export.
        if schedule_intent and username:
            if not (program and profile.get("curriculum_term")):
                return _stamp({
                    "response": _confidence_abstention_message("profile_required", language),
                    "sources": [], "source_chunk_ids": [], "intent": intents.WEEKLY_SCHEDULE,
                    "profile_required": True,
                    "confidence": _confidence_public("profile_required"),
                })
            completed = await get_completed_course_codes(username)
            interest_key = _extract_interest_key(question)
            interest_codes = list(INTEREST_COURSE_HINTS.get(interest_key or "", []))
            target_courses, minimum_su, strict_count = _recommendation_preferences(question)
            # Build a bounded candidate pool, then let the timetable solver choose the smallest
            # conflict-free subset that still satisfies the requested 15/18-SU load. This avoids
            # silently losing credits when one otherwise-good course combination has no compatible
            # section assignment.
            plan = course_planner.build_plan(
                program,
                profile["curriculum_term"],
                completed,
                interest_codes,
                target=max(target_courses, 8),
                minimum_su_credits=minimum_su,
                exact_course_count=False,
                academic_year=profile.get("academic_year"),
            )
            plan_errors = course_planner.validate_proposed_plan(
                [{"code": item.code} for item in plan.recommended],
                completed_codes=completed,
                stage=plan.stage,
            )
            if not plan.has_official_data:
                return _stamp({
                    "response": _confidence_abstention_message("curriculum_unavailable", language),
                    "sources": [], "source_chunk_ids": [], "intent": intents.WEEKLY_SCHEDULE,
                    "confidence": _confidence_public("curriculum_unavailable"),
                })
            if plan_errors:
                logger.error("deterministic schedule proposal validation failed (%s errors)", len(plan_errors))
                return _stamp({
                    "response": _confidence_abstention_message("cannot_verify", language),
                    "sources": [], "source_chunk_ids": [], "intent": intents.WEEKLY_SCHEDULE,
                    "confidence": _confidence_public("cannot_verify"),
                })
            timetable = schedule_planner.build_timetable_for_load(
                [i.code for i in plan.recommended],
                target_courses=target_courses,
                minimum_su_credits=minimum_su,
            )
            body, summary = schedule_planner.render_timetable(timetable, language=language)
            schedule_payload = schedule_planner.timetable_payload(
                timetable,
                language=language,
            )
            if schedule_payload.get("conflicts"):
                logger.error("deterministic timetable conflict validation failed")
                return _stamp({
                    "response": _confidence_abstention_message("cannot_verify", language),
                    "sources": [], "source_chunk_ids": [], "intent": intents.WEEKLY_SCHEDULE,
                    "confidence": _confidence_public("cannot_verify"),
                })
            saved_schedule: dict | None = None
            try:
                # Persist the same canonical object returned to the client. The schedule page can
                # therefore load a chat-created timetable even after a refresh or on another tab.
                saved_schedule = await set_user_schedule(username, schedule_payload)
            except Exception:
                # The response payload still lets the frontend keep a local fallback when MongoDB
                # is temporarily unavailable; schedule generation itself must remain usable.
                logger.exception("Could not persist chat-generated weekly schedule")
            sched_result = _stamp({
                "response": body,
                "summary": summary,
                "schedule": schedule_payload,
                "structured_content": schedule_planner.timetable_structured_content(
                    timetable, language=language
                ),
                "sources": ["Sabancı SUIS course schedule (official)"],
                "source_chunk_ids": [],
                "intent": intents.WEEKLY_SCHEDULE,
                "confidence": _confidence_public("verified"),
            })
            if saved_schedule:
                sched_result["schedule_revision"] = saved_schedule["revision"]
                sched_result["schedule_updated_at"] = saved_schedule["updated_at"]
            await conversation_memory.append_turn(
                session_id, username=username, question=original_question, answer=body,
                intent=intents.WEEKLY_SCHEDULE, term_code=profile.get("curriculum_term"),
                working_context_updates={"active_topic": "weekly_schedule"},
            )
            return sched_result

        # Deterministic academic follow-up (Failure A / "Kalan üniversite derslerim neler?"):
        # list the exact still-missing first-year University Courses from code — never inferred
        # from a credit gap, and never sent to the out-of-domain fallback.
        if _is_university_courses_query(original_question) and username:
            if not (program and profile.get("curriculum_term")):
                return _stamp({
                    "response": _confidence_abstention_message("profile_required", language),
                    "sources": [], "source_chunk_ids": [], "intent": intents.UNIVERSITY_COURSES,
                    "profile_required": True,
                    "confidence": _confidence_public("profile_required"),
                })
            completed = await get_completed_course_codes(username)
            missing = course_planner.missing_university_courses(program, completed)
            if missing:
                body = (
                    "Tamamlaman gereken birinci sınıf Üniversite Dersleri:\n"
                    + "\n".join(f"- {m}" for m in missing)
                )
            else:
                body = "Birinci sınıf Üniversite Derslerinin tamamını tamamlamışsın."
            uni_result = _stamp({
                "response": body,
                "sources": ["Deterministic academic-stage planner"],
                "source_chunk_ids": [],
                "intent": intents.UNIVERSITY_COURSES,
                "confidence": _confidence_public("verified"),
            })
            await conversation_memory.append_turn(
                session_id, username=username, question=original_question,
                answer=body, intent=intents.UNIVERSITY_COURSES,
                term_code=profile.get("curriculum_term"),
            )
            return uni_result

        # Graduation arithmetic and its immediate follow-ups are deterministic. This keeps
        # context stable and prevents implementation narration from leaking into the answer.
        if (graduation_intent or graduation_plan_intent) and program and profile.get("curriculum_term"):
            completed = await get_completed_course_codes(username) if username else []
            audit_result = degree_audit.audit(program, profile["curriculum_term"], completed)
            if audit_result.get("reliability") == "unavailable":
                return _stamp({
                    "response": _confidence_abstention_message("curriculum_unavailable", language),
                    "sources": [], "source_chunk_ids": [], "intent": intent,
                    "confidence": _confidence_public("curriculum_unavailable"),
                })
            if graduation_plan_intent:
                # "What should I take next?" -> the balanced, prerequisite-eligible plan computed
                # by the planner, NOT the raw missing-required list (which contains unreachable
                # 3XX/4XX courses a lower-year student cannot take yet).
                plan = course_planner.build_plan(
                    program,
                    profile["curriculum_term"],
                    completed,
                    academic_year=profile.get("academic_year"),
                )
                plan_errors = course_planner.validate_proposed_plan(
                    [{"code": item.code} for item in plan.recommended],
                    completed_codes=completed,
                    stage=plan.stage,
                )
                if not plan.has_official_data or plan_errors:
                    logger.error(
                        "deterministic graduation plan validation failed (%s errors)",
                        len(plan_errors),
                    )
                    return _stamp({
                        "response": _confidence_abstention_message("cannot_verify", language),
                        "sources": [], "source_chunk_ids": [], "intent": intent,
                        "confidence": _confidence_public("cannot_verify"),
                    })
                rendered_answer, summary = course_planner.render_plan(plan, language=language)
                structured = course_planner.plan_structured_content(plan, language=language)
                plan_sources = ["Deterministic academic-stage planner"]
            else:
                rendered_answer, summary = audit_answer(audit_result, language=language)
                structured = audit_structured_content(audit_result, language=language)
                plan_sources = ["Deterministic degree audit"]
            result = _stamp({
                "response": rendered_answer,
                "summary": summary,
                "structured_content": structured,
                "sources": plan_sources,
                "source_chunk_ids": [],
                "intent": intent,
                "confidence": _confidence_public("verified"),
            })
            await conversation_memory.append_turn(
                session_id,
                username=username,
                question=original_question,
                answer=rendered_answer,
                intent=intent,
                term_code=profile.get("curriculum_term"),
                sources=result["sources"],
                working_context_updates={
                    "active_topic": "graduation_planning",
                    "last_audit_status": audit_result.get("status"),
                    "last_remaining_su": audit_result.get("remaining_su_credits"),
                },
            )
            return result

        vectorstore = get_vectorstore()
        effective_question = _expanded_recommendation_question(question) if recommendation_intent else question
        retrieval_query = _retrieval_query_for_intent(effective_question, intent, program=program)

        # Profile-aware retrieval: scope by data_role (+ program/curriculum_term) BEFORE rerank,
        # so e.g. a CS student never sees IE requirements. Falls back to the documentType route
        # for review/exam/diger where no policy applies.
        policy_filter = build_metadata_filter(intent, profile)
        active_filter = policy_filter if policy_filter is not None else getattr(route, "metadata_filter", None)

        def _hybrid_search() -> List[Document]:
            """The production fused path (structured get + dense + BM25), already scoped."""
            if policy_filter is not None:
                return retrieve_documents(
                    vectorstore, retrieval_query, k=RETRIEVAL_CANDIDATE_K, metadata_filter=policy_filter
                )
            return _retrieve_for_route(vectorstore, route, retrieval_query)

        # Section 3: the selected mode decides which retrievers run. Every mode gets the same
        # `active_filter`, so ablating a retriever never widens the corpus beyond the student's
        # program/term scope.
        outcome = retrieval_modes.retrieve(
            retrieval_mode,
            vectorstore=vectorstore,
            query=retrieval_query,
            top_k=context_top_k,
            candidate_k=RETRIEVAL_CANDIDATE_K,
            metadata_filter=active_filter,
            hybrid_search=_hybrid_search,
        )
        if intent == intents.OTHER and not outcome.documents:
            return _stamp({
                "response": localized_message("fallback.non_academic", language),
                "sources": [],
                "source_chunk_ids": [],
                "intent": intent,
            })
        if not outcome.documents:
            return _stamp({
                "response": localized_message("confidence.insufficient_evidence", language),
                "sources": [],
                "source_chunk_ids": [],
                "intent": intent,
                "confidence": _confidence_public("cannot_verify"),
            })
        context_docs = _format_for_context(outcome.documents)
        safe_evidence_docs = list(context_docs)
        if not safe_evidence_docs:
            return _stamp({
                "response": localized_message("confidence.insufficient_evidence", language),
                "sources": [],
                "source_chunk_ids": [],
                "intent": intent,
                "confidence": _confidence_public("cannot_verify"),
                "security_filtered": True,
            })

        # Deterministic degree audit: compute the numbers in code, inject as authoritative context.
        audit_result: dict | None = None
        if graduation_intent and program and profile.get("curriculum_term"):
            try:
                completed = await get_completed_course_codes(username) if username else []
                audit_result = degree_audit.audit(program, profile["curriculum_term"], completed)
                context_docs.insert(0, _audit_context_document(audit_result))
            except Exception:
                logger.exception("degree audit failed")

        taken_codes: list[str] = []
        if recommendation_intent and username:
            selected_courses = await get_user_courses(username)
            taken_codes = _course_codes(selected_courses)
            user_context = _compact_recommendation_user_context(username, selected_courses)
        else:
            user_context = await get_user_course_context(username)
        if user_context:
            context_docs.insert(
                0,
                Document(
                    page_content=(
                        "[Source: MongoDB student profile]\n"
                        + retrieval_boundary(user_context, "student-profile")
                    ),
                    metadata={
                        "source": "MongoDB student profile",
                        "document_type": "user_course_history",
                    },
                ),
            )
        if recommendation_intent:
            # Deterministic, student-specific planner: canonical course names + an ordered,
            # prerequisite-checked candidate pool + hard rules (University-course debt first,
            # sophomores get no current 4XX, advanced interest courses become future targets).
            # Injected first so it outranks the softer interest-hint documents below.
            interest_key = _extract_interest_key(effective_question)
            interest_codes = list(INTEREST_COURSE_HINTS.get(interest_key or "", []))
            planner_doc = Document(
                page_content=course_planner.build_context(
                    program,
                    taken_codes,
                    interest_codes,
                    academic_year=profile.get("academic_year"),
                ),
                metadata={
                    "source": "Academic-stage planner",
                    "document_type": "academic_stage_plan",
                },
            )
            context_docs = (
                [planner_doc]
                + _recommendation_support_documents(effective_question, taken_codes)
                + context_docs
            )
        if prior_turns:
            context_docs.insert(0, _conversation_context_document(prior_turns))
        context_docs.insert(0, _intent_context_document(intent, taken_codes))

        retriever = StaticRetriever(documents=context_docs)
        # Answer in the language the student wrote in. Decided here rather than left to a rule
        # inside the (Turkish) system prompt, which the model was not reliably honouring.
        chain = await asyncio.to_thread(
            get_llm_chain,
            retriever,
            intent,
            language,
            prompt_strategy,
        )
        llm_question = _recommendation_question_for_llm(effective_question, taken_codes) if recommendation_intent else question
        # If the provider fails before returning usage, reserve a conservative worst-case token
        # charge so retry/error traffic cannot bypass the daily quota. Successful calls replace
        # this with provider-reported usage below.
        estimated_prompt_tokens_upper_bound = len(llm_question) + sum(
            len(document.page_content) for document in context_docs
        )
        usage_total_tokens = (
            estimated_prompt_tokens_upper_bound + MAX_OUTPUT_TOKENS
        ) * (LLM_MAX_RETRIES + 1)
        provider_attempted = True
        result = await asyncio.to_thread(query_chain, chain, llm_question)
        provider_telemetry = result.pop("_provider_telemetry", None)
        if provider_telemetry:
            if provider_telemetry.get("total_tokens") is not None:
                usage_total_tokens = provider_telemetry.get("total_tokens")
            usage_cost_usd = provider_telemetry.get("cost_usd")
            logger.info(
                "llm call provider=%s model=%s effective_provider=%s effective_model=%s "
                "fallback=%s latency_ms=%s total_tokens=%s cost_usd=%s",
                provider_telemetry.get("requested_provider"),
                provider_telemetry.get("requested_model"),
                provider_telemetry.get("effective_provider"),
                provider_telemetry.get("effective_model"),
                provider_telemetry.get("fallback_used"),
                provider_telemetry.get("latency_ms"),
                provider_telemetry.get("total_tokens"),
                provider_telemetry.get("cost_usd"),
            )
        # Validate the untouched provider output before any formatter can remove a leaked marker.
        # The allow-list is derived from the server-owned context, never from model text.
        raw_answer = str(result.get("response") or "")
        authorized_sources = [
            label for label in (_document_source_label(doc) for doc in context_docs) if label
        ]
        output_validation = validate_output(raw_answer, authorized_sources)
        returned_sources_authorized = _source_list_is_authorized(
            list(result.get("sources") or []),
            authorized_sources,
        )
        guardrail_passed = output_validation.safe and returned_sources_authorized
        evidence_sources = {
            str(doc.metadata.get("source") or doc.metadata.get("documentType") or "").strip()
            for doc in safe_evidence_docs
            if str(doc.metadata.get("source") or doc.metadata.get("documentType") or "").strip()
        }
        retrieval_scores = [
            float(doc.metadata["_score"])
            for doc in safe_evidence_docs
            if isinstance(doc.metadata.get("_score"), (int, float))
        ]
        confidence = assess_confidence(
            ConfidenceSignals(
                intent=intent,
                retrieval_score=max(retrieval_scores) if retrieval_scores else None,
                evidence_count=len(safe_evidence_docs),
                independent_source_count=len(evidence_sources),
                metadata_compatible=True,
                evidence_required=True,
                citations_required=True,
                citations_present=bool(result.get("sources") or result.get("source_chunk_ids")),
                citations_authorized=output_validation.safe and returned_sources_authorized,
                claim_coverage_checked=False,
                output_schema_valid=bool(raw_answer.strip()),
                output_language_match=_output_language_matches(raw_answer, language),
                guardrail_passed=guardrail_passed,
                provider_response_available=bool(raw_answer.strip()),
                fallback_used=bool((provider_telemetry or {}).get("fallback_used")),
            )
        )
        if confidence.answer_allowed:
            if recommendation_intent:
                result = _clean_recommendation_response(result)
                raw_answer = str(result.get("response") or "")
            cleaned_answer = sanitize_student_answer(raw_answer)
            rendered_answer, summary = ensure_summary_section(cleaned_answer, language=language)
        else:
            safe_message = (
                refusal_message("unsafe_output", language)
                if confidence.status == "safe_abstention"
                else _confidence_abstention_message(confidence.status, language)
            )
            rendered_answer, summary = ensure_summary_section(safe_message, language=language)
            result["sources"] = []
            result["source_chunk_ids"] = []
        result["security_filtered"] = not guardrail_passed
        # Expose only the stable user-safe state, never formula inputs, score, or reason codes.
        result["confidence"] = _confidence_public(confidence.status)
        result["response"] = rendered_answer
        result["summary"] = summary
        result["structured_content"] = merge_structured_content(
            audit_structured_content(audit_result, language=language),
            course_update_content,
        )
        if username:
            result["export_links"] = {
                "courses_csv": f"/users/{username}/courses/export?format=csv",
                "courses_xlsx": f"/users/{username}/courses/export?format=xlsx",
            }
            if audit_result and audit_result.get("reliability") != "unavailable":
                result["export_links"].update(
                    {
                        "audit_csv": f"/users/{username}/degree-audit/export?format=csv",
                        "audit_xlsx": f"/users/{username}/degree-audit/export?format=xlsx",
                    }
                )
        if course_mutation:
            result["profile_updated"] = True
            result["course_history"] = course_mutation
        result["intent"] = intent
        result = _stamp(result)
        result["reranked"] = outcome.reranked
        result["num_retrieved_chunks"] = outcome.candidate_count
        result["num_final_context_chunks"] = len(safe_evidence_docs)

        # Update bounded session memory (best-effort).
        await conversation_memory.append_turn(
            session_id,
            username=username,
            question=original_question,
            answer=result.get("response", ""),
            intent=intent,
            course_id=conversation_memory.extract_course_code(question)
            or (working_context.get("last_course_id") if working_context else None),
            term_code=profile.get("curriculum_term"),
            sources=result.get("sources") or [],
        )

        logger.info("query successful")
        return result

    except Exception as e:
        # Exception messages may contain provider response bodies.  Record only the class and
        # return a localized safe error; raw payloads and authorization details are never logged.
        logger.error("question processing failed (error_class=%s)", type(e).__name__)
        detail = str(e)
        if "rate_limit" in detail or "429" in detail:
            message = localized_message("errors.rate_limited", language)
        else:
            message = localized_message("errors.internal", language)
        return _stamp({
            "response": message,
            "sources": [],
            "source_chunk_ids": [],
            "intent": "error",
            "error": True,
            "confidence": _confidence_public(
                "provider_unavailable" if provider_attempted else "cannot_verify"
            ),
        })
    finally:
        resource_controller.finish(
            username=username,
            total_tokens=usage_total_tokens,
            cost_usd=usage_cost_usd,
            admission=admission,
            provider_requests=1 if provider_attempted else 0,
        )


@app.post("/ask/stream")
async def ask_question_stream(
    request: Request,
    question: str = Form(...),
    username: str | None = Form(None),
    session_id: str | None = Form(None),
    mode: str | None = Form(None),
    top_k: int | None = Form(None),
    prompt_strategy: str | None = Form(None),
    expert_mode: str | None = Form(None),
):
    """NDJSON response that renders progressively while retaining the stable /ask contract."""
    _check_operation_limit(
        request,
        kind="stream",
        identity=username,
        limit=STREAM_REQUESTS_PER_MINUTE,
        language=detect_language(question),
    )
    payload = await ask_question(
        request=request,
        question=question,
        username=username,
        session_id=session_id,
        mode=mode,
        top_k=top_k,
        prompt_strategy=prompt_strategy,
        expert_mode=expert_mode,
    )
    if isinstance(payload, Response):
        return payload

    response_text = str(payload.get("response") or "")
    metadata = {key: value for key, value in payload.items() if key != "response"}

    async def events():
        deadline = time.monotonic() + REQUEST_TIMEOUT_SECONDS
        yield json.dumps({"type": "metadata", "data": metadata}, ensure_ascii=False) + "\n"
        for chunk in re.findall(r"\S+\s*", response_text):
            if time.monotonic() >= deadline:
                yield json.dumps({"type": "error", "code": "stream_deadline"}) + "\n"
                return
            yield json.dumps({"type": "token", "text": chunk}, ensure_ascii=False) + "\n"
            await asyncio.sleep(0.012)
        yield json.dumps({"type": "done"}, ensure_ascii=False) + "\n"

    return StreamingResponse(
        events(),
        media_type="application/x-ndjson",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )


@app.get("/users/{username}/conversations")
async def list_conversations(username: str, request: Request, limit: int = 50):
    """Chat history for the sidebar: newest first, titles only."""
    _authorize_user(request, username)
    return {"conversations": await conversation_memory.list_conversations(username, limit)}


@app.get("/conversations/{session_id}")
async def get_conversation(session_id: str, request: Request):
    principal = await _authorize_conversation(request, session_id)
    owner_filter = None if principal.role == "admin" else principal.username
    conversation = await conversation_memory.get_conversation(
        session_id, username=owner_filter
    )
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation


@app.patch("/conversations/{session_id}")
async def update_conversation(session_id: str, payload: ConversationUpdatePayload, request: Request):
    principal = await _authorize_conversation(request, session_id)
    owner_filter = None if principal.role == "admin" else principal.username
    conversation = None
    if payload.title is not None:
        conversation = await conversation_memory.rename_conversation(
            session_id, payload.title, username=owner_filter
        )
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
    if payload.pinned is not None:
        conversation = await conversation_memory.set_conversation_pinned(
            session_id,
            payload.pinned,
            username=owner_filter,
        )
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
    if conversation is None:
        conversation = await conversation_memory.get_conversation(
            session_id, username=owner_filter
        )
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation


@app.delete("/conversations/{session_id}")
async def delete_conversation(session_id: str, request: Request):
    principal = await _authorize_conversation(request, session_id)
    owner_filter = None if principal.role == "admin" else principal.username
    deleted = await conversation_memory.delete_conversation(
        session_id, username=owner_filter
    )
    if not deleted:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"deleted": True}


@app.get("/test")
async def test():
    return {"message":"Testing successfull..."}
