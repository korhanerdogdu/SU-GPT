import asyncio
import json
import re
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI,UploadFile,File,Form,Request,HTTPException
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List
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
    confirm_whatsapp_batch,
    create_pending_whatsapp_batch,
    ingest_exam_upload,
)
from modules.intent_detector import get_intent
from modules.llm import answer_without_context, detect_language, get_llm_chain
from modules.query_handlers import query_chain
from modules.config import (
    ADMIN_PASSWORD,
    ADMIN_USERNAME,
    AUTO_INGEST_SOURCES,
    AUTO_SEED_COURSES,
    CATALOG_DATA_DIR,
    DEFAULT_EXPERT_MODE,
    DEFAULT_PROMPT_STRATEGY,
    DEFAULT_RETRIEVAL_MODE,
    RERANK_TOP_K,
    RETRIEVAL_CANDIDATE_K,
    SOURCES_DIR,
    STUDENT_PASSWORD,
    STUDENT_USERNAME,
)
from modules import retrieval_modes
from modules.catalog_retriever import retrieve_documents
from modules.mongodb import (
    ensure_database,
    ensure_user,
    get_academic_profile,
    get_completed_course_codes,
    get_user_course_context,
    get_user_courses,
    list_courses,
    mutate_user_courses_by_codes,
    seed_courses_from_catalog,
    set_academic_profile,
    set_user_courses,
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
from modules import conversation_memory
from modules.retrieval_policy import build_metadata_filter, check_profile
from modules.source_indexer import ensure_sources_indexed
from logger import logger

app = FastAPI(title="adviSU — Retrieval-Augmented Academic Advising System")


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
    statuses: dict[str, str] = Field(default_factory=dict)


class AcademicProfilePayload(BaseModel):
    major: str | None = None
    degree_code: str | None = None
    admission_term: str | None = None
    curriculum_term: str | None = None
    minor_codes: list[str] | None = None


class ConversationUpdatePayload(BaseModel):
    title: str | None = None
    pinned: bool | None = None


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
RECOMMENDATION_TERM = "202502"

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
    if previous_intent in {"mezuniyet_durumu", "graduation_plan"} and any(
        phrase in normalized for phrase in graduation_followups
    ):
        return "graduation_plan"
    if previous_intent in {"mezuniyet_durumu", "graduation_plan"} and re.search(
        r"\b(?:sonraki|gelecek)\s+dönem\b|\bnext semester\b", normalized
    ):
        return "graduation_plan"
    if MINOR_INTENT_RE.search(question or ""):
        return "minor"
    if _is_graduation_intent(question) and not _is_course_detail_like(question):
        return "mezuniyet_durumu"
    if _is_course_detail_like(question):
        return "ders_ayrintisi"
    if STUDY_PLAN_INTENT_RE.search(question or ""):
        return "calisma_plani"
    if MAJOR_SELECTION_INTENT_RE.search(question or ""):
        return "major_secimi"
    if SPECIALIZATION_INTENT_RE.search(question or ""):
        return "alanda_ozellesme"
    if _is_recommendation_intent(question):
        return "ders_onerisi"
    return detected_intent or "diger"


def _recommendation_question_for_llm(question: str, taken_codes: list[str]) -> str:
    taken = ", ".join(taken_codes) if taken_codes else "none"
    interest = _interest_label(question) or "the requested area"
    return (
        "COURSE_RECOMMENDATION_MODE. The user is asking for course recommendations, "
        "not graduation audit. Do not mention graduation status, 125/125 credits, or "
        "category audit unless the user explicitly asks for it in this same message. "
        "Never start with any sentence about checking graduation status. "
        f"The user already provided the interest area: {interest}. Do not ask another "
        "clarifying question about the area; produce the course program now. "
        f"Already taken course codes, do not recommend any of these: {taken}. "
        "If the obvious courses in the requested area are already taken, say that clearly "
        "and recommend adjacent untaken courses only. "
        "Unless the user asks for a different count, recommend exactly 5 untaken courses. "
        "For each course, include: course code, title, instructor if available in context, "
        "why it fits the interest area, and the schedule if available. "
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
    if intent == "review":
        return f"{question} instructor professor review workload grading difficulty course experience"
    if intent == "exam":
        return f"{question} exam final midterm quiz past questions solutions assessment"
    if intent == "minor":
        return f"{question} minor program required courses core electives area electives"
    if intent == "ders_onerisi":
        return _recommendation_retrieval_query(question)
    if intent == "mezuniyet_durumu":
        prog = (program or "").strip()
        return (
            f"{question} {prog} degree requirements graduation "
            "university courses required courses core electives area electives free electives"
        )
    if intent == "calisma_plani":
        return f"{question} course syllabus assignments exams study plan workload"
    if intent == "ders_ayrintisi":
        return f"{question} course detail instructor syllabus schedule prerequisite workload"
    if intent in {"major_secimi", "alanda_ozellesme"}:
        return f"{question} Sabanci program requirements course catalog specialization career"
    return question


def _intent_context_document(intent: str, taken_codes: list[str] | None = None) -> Document:
    if intent == "ders_onerisi":
        taken = ", ".join(taken_codes or []) if taken_codes else "none"
        text = (
            "[Source: Request intent]\n"
            "Detected intent: ders_onerisi / course recommendation / schedule planning. "
            "Do not produce graduation audit. Do not summarize completed credits. "
            "Never start with any sentence about checking graduation status. "
            f"Already taken course codes, strictly forbidden to recommend: {taken}. "
            "Use MongoDB profile only as a taken-course exclusion list and for personalization."
        )
    elif intent == "mezuniyet_durumu":
        text = (
            "[Source: Request intent]\n"
            "Detected intent: mezuniyet_durumu / graduation audit. Use official degree evaluation "
            "and degree requirement RAG sources with MongoDB student profile."
        )
    elif intent == "calisma_plani":
        text = (
            "[Source: Request intent]\n"
            "Detected intent: calisma_plani / study plan. Give course-specific study guidance. "
            "Do not produce graduation audit unless explicitly requested."
        )
    elif intent == "ders_ayrintisi":
        text = (
            "[Source: Request intent]\n"
            "Detected intent: ders_ayrintisi / course detail. Answer the requested course detail "
            "such as instructor, schedule, syllabus, prerequisite or workload. Do not produce graduation audit."
        )
    elif intent == "review":
        text = (
            "[Source: Request intent]\n"
            "Detected intent: review / instructor or course review. Use only instructor review chunks "
            "and clearly separate retrieved student sentiment from official course facts."
        )
    elif intent == "exam":
        text = (
            "[Source: Request intent]\n"
            "Detected intent: exam / past exam or assessment question. Use exam chunks and course context; "
            "do not invent unavailable questions or answers."
        )
    elif intent == "major_secimi":
        text = (
            "[Source: Request intent]\n"
            "Detected intent: major_secimi / major selection. Compare programs and fit using catalog context. "
            "Do not produce graduation audit."
        )
    elif intent == "alanda_ozellesme":
        text = (
            "[Source: Request intent]\n"
            "Detected intent: alanda_ozellesme / specialization guidance. Use catalog and course context "
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
    return Document(
        page_content=(
            "[Source: Recent conversation context]\n"
            "Use these turns only to resolve follow-up meaning. Official curriculum data and "
            "the deterministic audit remain authoritative.\n"
            + json.dumps(compact, ensure_ascii=False)
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


def _format_for_context(docs: List[Document]) -> List[Document]:
    """Prepend a [Source: ...] header to each chunk so the LLM can ground & cite."""
    formatted: List[Document] = []
    for doc in docs:
        meta = doc.metadata or {}
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
                page_content=f"{header}\n{doc.page_content}",
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.on_event("startup")
async def startup_event():
    try:
        await ensure_database()
        if AUTO_SEED_COURSES:
            await seed_courses_from_catalog()
    except Exception:
        logger.exception("MongoDB startup initialization failed")

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
        return await call_next(request)
    except Exception as exc:
        logger.exception("UNHANDLED EXCEPTION")
        return JSONResponse(status_code=500,content={"error":str(exc)})
    
@app.post("/upload_pdfs/")
async def upload_pdfs(files:List[UploadFile]=File(...)):
    try:
        logger.info(f"recieved {len(files)} files")
        chunk_count = load_vectorstore(files)
        logger.info("documents added to chroma")
        return {"message":"Files processed and vectorstore updated","chunks":chunk_count}
    except Exception as e:
        logger.exception("Error during pdf upload")
        return JSONResponse(status_code=500,content={"error":str(e)})


@app.post("/upload_documents/")
async def upload_documents(files: List[UploadFile] = File(...)):
    """Multi-format upload endpoint (Section 2).

    Accepts PDF, PPTX, DOCX, MD, and TXT files. Unsupported types are
    skipped and reported in the response.
    """
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
    except Exception as e:
        logger.exception("Error during multi-format document upload")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/admin/whatsapp/upload")
async def upload_whatsapp_chat(
    file: UploadFile = File(...),
    username: str = Form("admin"),
):
    try:
        return create_pending_whatsapp_batch(file, uploaded_by=username)
    except Exception as e:
        logger.exception("Error during WhatsApp pending upload")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/admin/whatsapp/{batch_id}/confirm")
async def confirm_whatsapp_upload(
    batch_id: str,
    approved: bool = Form(True),
    username: str = Form("admin"),
):
    try:
        return confirm_whatsapp_batch(batch_id, approved=approved, approved_by=username)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Error during WhatsApp confirmation")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/admin/exams/upload")
async def upload_exam_pdf(
    file: UploadFile = File(...),
    course_code: str = Form(""),
    year: str = Form(""),
    semester: str = Form(""),
    exam_type: str = Form(""),
    username: str = Form("admin"),
):
    try:
        return ingest_exam_upload(
            file,
            course_code=course_code,
            year=year,
            semester=semester,
            exam_type=exam_type,
            uploaded_by=username,
        )
    except Exception as e:
        logger.exception("Error during exam upload")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.delete("/sources/{source_id}")
async def delete_source_document(source_id: str, hard: bool = False):
    try:
        return cascade_delete_source(source_id, hard=hard)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Error during source cascade delete")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/auth/login")
async def login(payload: LoginPayload):
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
    return {"username": payload.username, "role": role}


@app.get("/courses/")
async def get_courses(search: str = "", limit: int = 200):
    return {"courses": await list_courses(search=search, limit=limit)}


@app.get("/users/{username}/courses")
async def get_selected_courses(username: str):
    return {"courses": await get_user_courses(username)}


@app.put("/users/{username}/courses")
async def save_selected_courses(username: str, payload: CourseSelectionPayload):
    return {"courses": await set_user_courses(username, payload.course_ids, payload.statuses)}


@app.get("/users/{username}/profile")
async def get_profile(username: str):
    return {"profile": await get_academic_profile(username)}


@app.put("/users/{username}/profile")
async def put_profile(username: str, payload: AcademicProfilePayload):
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
async def degree_audit_endpoint(username: str):
    """Deterministic degree audit for the student's confirmed profile (roadmap section 14)."""
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
async def export_selected_courses(username: str, format: str = "csv"):
    selected = await get_user_courses(username)
    return _tabular_download(
        course_rows(selected),
        file_format=format,
        filename=f"advisu-{username}-courses",
        sheet_name="Course History",
    )


@app.get("/users/{username}/degree-audit/export")
async def export_degree_audit(username: str, format: str = "xlsx"):
    audit_result = await degree_audit_endpoint(username)
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
    context_top_k = max(int(top_k), 1) if top_k else RERANK_TOP_K
    prompt_strategy = (prompt_strategy or DEFAULT_PROMPT_STRATEGY).strip().lower()
    expert_mode = (expert_mode or DEFAULT_EXPERT_MODE).strip().lower()
    original_question = question
    language = detect_language(original_question)
    course_mutation: dict | None = None
    course_update_content: dict | None = None

    def _stamp(payload: dict) -> dict:
        """Every /ask response reports the configuration that produced it (needed by Section 5)."""
        payload.setdefault("mode", retrieval_mode)
        payload.setdefault("top_k", context_top_k)
        payload.setdefault("prompt_strategy", prompt_strategy)
        payload.setdefault("expert_mode", expert_mode)
        return payload

    try:
        logger.info(f"user query: {question} (mode={retrieval_mode}, top_k={context_top_k})")

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
            answer = answer_without_context(question)
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
            })

        # Bounded session memory: resolve "this course" style follow-ups (roadmap section 19).
        working_context = await conversation_memory.get_working_context(session_id)
        question = conversation_memory.resolve_reference(question, working_context)
        prior_turns = await conversation_memory.recent_turns(session_id, limit=3)

        detected_intent = get_intent(question)
        resolved_intent = _resolve_intent(question, detected_intent, working_context)
        route = route_query(question, resolved_intent)
        intent = resolved_intent if resolved_intent == "minor" else route.intent
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

        graduation_plan_intent = resolved_intent == "graduation_plan"
        if graduation_plan_intent:
            intent = "graduation_plan"
        graduation_intent = intent == "mezuniyet_durumu"
        recommendation_intent = intent == "ders_onerisi"
        if recommendation_intent and not _has_interest_area(question):
            return _stamp({
                "response": "Hangi alana ilgilisin? Örn: NLP, Web, Data, Systems, AI, Security.",
                "sources": [],
                "source_chunk_ids": [],
                "intent": intent,
            })

        # Missing-data / missing-profile safety for authoritative intents (roadmap section 15).
        gate = check_profile("mezuniyet_durumu" if graduation_plan_intent else intent, profile)
        if not gate.ok:
            return _stamp({
                "response": gate.message,
                "sources": [],
                "source_chunk_ids": [],
                "intent": intent,
                "profile_required": bool(gate.missing_fields),
                "curriculum_unavailable": gate.data_unavailable,
            })

        # Graduation arithmetic and its immediate follow-ups are deterministic. This keeps
        # context stable and prevents implementation narration from leaking into the answer.
        if (graduation_intent or graduation_plan_intent) and program and profile.get("curriculum_term"):
            completed = await get_completed_course_codes(username) if username else []
            audit_result = degree_audit.audit(program, profile["curriculum_term"], completed)
            if graduation_plan_intent:
                next_term = bool(re.search(
                    r"\b(?:sonraki|gelecek)\s+dönem\b|\bnext semester\b",
                    original_question,
                    re.IGNORECASE,
                ))
                rendered_answer, summary = graduation_plan_answer(
                    audit_result, next_term=next_term, language=language
                )
            else:
                rendered_answer, summary = audit_answer(audit_result, language=language)
            result = _stamp({
                "response": rendered_answer,
                "summary": summary,
                "structured_content": audit_structured_content(audit_result, language=language),
                "sources": ["Deterministic degree audit"],
                "source_chunk_ids": [],
                "intent": intent,
                "export_links": {
                    "courses_csv": f"/users/{username}/courses/export?format=csv",
                    "courses_xlsx": f"/users/{username}/courses/export?format=xlsx",
                    "audit_csv": f"/users/{username}/degree-audit/export?format=csv",
                    "audit_xlsx": f"/users/{username}/degree-audit/export?format=xlsx",
                },
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
        if intent == "diger" and not outcome.documents:
            return _stamp({
                "response": NON_ACADEMIC_FALLBACK,
                "sources": [],
                "source_chunk_ids": [],
                "intent": intent,
            })
        context_docs = _format_for_context(outcome.documents)

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
                    page_content=f"[Source: MongoDB student profile]\n{user_context}",
                    metadata={
                        "source": "MongoDB student profile",
                        "document_type": "user_course_history",
                    },
                ),
            )
        if recommendation_intent:
            context_docs = _recommendation_support_documents(effective_question, taken_codes) + context_docs
        if prior_turns:
            context_docs.insert(0, _conversation_context_document(prior_turns))
        context_docs.insert(0, _intent_context_document(intent, taken_codes))

        retriever = StaticRetriever(documents=context_docs)
        # Answer in the language the student wrote in. Decided here rather than left to a rule
        # inside the (Turkish) system prompt, which the model was not reliably honouring.
        chain = get_llm_chain(
            retriever,
            intent=intent,
            language=language,
            prompt_strategy=prompt_strategy,
        )
        llm_question = _recommendation_question_for_llm(effective_question, taken_codes) if recommendation_intent else question
        result = query_chain(chain, llm_question)
        if recommendation_intent:
            result = _clean_recommendation_response(result)
        cleaned_answer = sanitize_student_answer(result.get("response", ""))
        rendered_answer, summary = ensure_summary_section(
            cleaned_answer,
            language=language,
        )
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
        result["num_final_context_chunks"] = len(outcome.documents)

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
        logger.exception("Error processing question")
        # Never surface the raw provider payload: it carries the organisation id and quota
        # internals, and reads as a crash to the student. The full error is in the server log.
        detail = str(e)
        if "rate_limit" in detail or "429" in detail:
            message = (
                "The language model is temporarily rate limited, so I can't answer right now. "
                "Please try again in a few minutes — your profile and course history are unaffected."
            )
        else:
            message = (
                "Something went wrong while answering that. Please try again; if it keeps "
                "happening, check the backend logs."
            )
        return _stamp({
            "response": message,
            "sources": [],
            "source_chunk_ids": [],
            "intent": "error",
            "error": True,
        })


@app.post("/ask/stream")
async def ask_question_stream(
    question: str = Form(...),
    username: str | None = Form(None),
    session_id: str | None = Form(None),
    mode: str | None = Form(None),
    top_k: int | None = Form(None),
    prompt_strategy: str | None = Form(None),
    expert_mode: str | None = Form(None),
):
    """NDJSON response that renders progressively while retaining the stable /ask contract."""
    payload = await ask_question(
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
        yield json.dumps({"type": "metadata", "data": metadata}, ensure_ascii=False) + "\n"
        for chunk in re.findall(r"\S+\s*", response_text):
            yield json.dumps({"type": "token", "text": chunk}, ensure_ascii=False) + "\n"
            await asyncio.sleep(0.012)
        yield json.dumps({"type": "done"}, ensure_ascii=False) + "\n"

    return StreamingResponse(
        events(),
        media_type="application/x-ndjson",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )


@app.get("/users/{username}/conversations")
async def list_conversations(username: str, limit: int = 50):
    """Chat history for the sidebar: newest first, titles only."""
    return {"conversations": await conversation_memory.list_conversations(username, limit)}


@app.get("/conversations/{session_id}")
async def get_conversation(session_id: str):
    conversation = await conversation_memory.get_conversation(session_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation


@app.patch("/conversations/{session_id}")
async def update_conversation(session_id: str, payload: ConversationUpdatePayload):
    conversation = None
    if payload.title is not None:
        conversation = await conversation_memory.rename_conversation(session_id, payload.title)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
    if payload.pinned is not None:
        conversation = await conversation_memory.set_conversation_pinned(
            session_id,
            payload.pinned,
        )
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
    if conversation is None:
        conversation = await conversation_memory.get_conversation(session_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation


@app.delete("/conversations/{session_id}")
async def delete_conversation(session_id: str):
    deleted = await conversation_memory.delete_conversation(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"deleted": True}


@app.get("/test")
async def test():
    return {"message":"Testing successfull..."}
