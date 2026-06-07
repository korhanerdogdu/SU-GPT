import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI,UploadFile,File,Form,Request,HTTPException
from fastapi.responses import JSONResponse
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
from modules.llm import get_llm_chain
from modules.query_handlers import query_chain
from modules.config import (
    ADMIN_PASSWORD,
    ADMIN_USERNAME,
    AUTO_INGEST_SOURCES,
    AUTO_SEED_COURSES,
    CATALOG_DATA_DIR,
    RERANK_TOP_K,
    RETRIEVAL_CANDIDATE_K,
    SOURCES_DIR,
)
from modules.catalog_retriever import retrieve_documents
from modules.mongodb import (
    ensure_database,
    ensure_user,
    get_user_course_context,
    get_user_courses,
    list_courses,
    seed_courses_from_catalog,
    set_user_courses,
)
from modules.reranker import rerank_documents
from modules.rag_router import route_query
from modules.source_indexer import ensure_sources_indexed
from logger import logger

app = FastAPI(title="SU-GPT — Course-Aware RAG Assistant")


class StaticRetriever(BaseRetriever):
    documents: List[Document] = Field(default_factory=list)

    def _get_relevant_documents(self, query: str) -> List[Document]:
        return self.documents


class LoginPayload(BaseModel):
    username: str
    password: str


class CourseSelectionPayload(BaseModel):
    course_ids: list[str] = Field(default_factory=list)


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
DEFAULT_DEGREE_PROGRAM = "CS"

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


def _resolve_intent(question: str, detected_intent: str) -> str:
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


def _retrieval_query_for_intent(question: str, intent: str) -> str:
    if intent == "review":
        return f"{question} instructor professor review workload grading difficulty course experience"
    if intent == "exam":
        return f"{question} exam final midterm quiz past questions solutions assessment"
    if intent == "ders_onerisi":
        return _recommendation_retrieval_query(question)
    if intent == "mezuniyet_durumu":
        return (
            f"{question} official degree evaluation BSCS graduation requirements "
            "university courses required courses core electives area electives free electives"
        )
    if intent == "calisma_plani":
        return f"{question} course syllabus assignments exams study plan workload"
    if intent == "ders_ayrintisi":
        return f"{question} course detail instructor syllabus schedule prerequisite workload"
    if intent in {"major_secimi", "alanda_ozellesme"}:
        return f"{question} Sabanci program requirements course catalog specialization career"
    return question


def _structured_context_documents(
    intent: str,
    question: str,
    taken_codes: list[str],
) -> list[Document] | None:
    """Fast paths for intents that are better served by structured data.

    These modes avoid broad Chroma scans over the full course/source corpus.
    Chroma remains available for source-heavy questions, but graduation,
    recommendation, and missing-source review/exam intents get compact context
    directly from Mongo/catalog files or from an explicit general-knowledge guard.
    """

    if intent == "mezuniyet_durumu":
        return _graduation_support_documents(DEFAULT_DEGREE_PROGRAM)
    if intent == "ders_onerisi":
        return _recommendation_support_documents(question, taken_codes)
    if intent == "review":
        return [
            *_instructor_schedule_support_documents(question),
            _general_knowledge_mode_document(intent),
        ]
    if intent == "exam":
        return [
            *_course_detail_support_documents(question),
            _general_knowledge_mode_document(intent),
        ]
    if intent == "ders_ayrintisi" and _syllabus_like(question):
        return [
            *_course_detail_support_documents(question),
            _general_knowledge_mode_document("syllabus"),
        ]
    return None


@lru_cache(maxsize=4)
def _degree_requirement_rows(program: str) -> tuple[dict, ...]:
    path = Path(CATALOG_DATA_DIR) / "degree_requirements" / f"{program.upper()}.jsonl"
    if not path.exists():
        return ()
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return tuple(rows)


def _graduation_support_documents(program: str) -> list[Document]:
    rows = _degree_requirement_rows(program)
    docs: list[Document] = []
    for index, row in enumerate(rows):
        record_type = str(row.get("record_type") or f"record_{index}")
        source = _degree_source_label(row, program)
        docs.append(
            Document(
                page_content=_format_degree_requirement_row(row),
                metadata={
                    "source": source,
                    "document_type": record_type,
                    "program": program,
                    "chunk_id": f"structured:degree_requirements:{program}:{record_type}:{index}",
                    "source_authority": row.get("source_authority") or "official_degree_requirements",
                },
            )
        )
    if not docs:
        docs.append(
            Document(
                page_content=(
                    "No local structured degree requirement file is configured. "
                    "Use the MongoDB student profile if present; otherwise say the "
                    "official degree requirement source is missing."
                ),
                metadata={
                    "source": f"{program} degree requirements missing",
                    "document_type": "degree_requirement_missing",
                    "chunk_id": f"structured:degree_requirements:{program}:missing",
                },
            )
        )
    return docs


def _degree_source_label(row: dict, program: str) -> str:
    record_type = str(row.get("record_type") or "degree_requirement")
    if record_type == "official_degree_evaluation_projection":
        return f"{program}/BSCS official degree evaluation projection"
    if record_type == "degree_requirement_profile":
        return f"{program}/BSCS degree requirement profile"
    category = row.get("category_label_tr") or row.get("category")
    if category:
        return f"{program}/BSCS degree requirement pool {category}"
    return f"{program}/BSCS {record_type}"


def _format_degree_requirement_row(row: dict) -> str:
    record_type = row.get("record_type")
    if record_type == "degree_requirement_profile":
        minima = "; ".join(
            _format_category_minimum(item)
            for item in row.get("category_minima", [])
            if isinstance(item, dict)
        )
        rules = "; ".join(str(rule) for rule in row.get("rules", []))
        return (
            "Record type: degree requirement profile.\n"
            f"Program: {row.get('program')} / {row.get('degree_code')} - {row.get('program_name_tr') or row.get('program_name_en')}.\n"
            f"Requirement term: {row.get('program_requirements_term_label')}.\n"
            f"Total minimum: {row.get('total_min_su_credits')} SU credits, {row.get('total_min_ects')} ECTS.\n"
            f"Category minima: {minima}.\n"
            f"Rules: {rules}."
        )
    if record_type == "degree_requirement_category_pool":
        category = row.get("category_label_tr") or row.get("category")
        courses = _degree_course_codes(row.get("courses") or row.get("courses_representative") or [])
        choice_pools = []
        for pool in row.get("choice_pools", []) or []:
            if not isinstance(pool, dict):
                continue
            choice_pools.append(
                f"{pool.get('name')} choose {pool.get('choose')}: {_degree_course_codes(pool.get('courses') or [])}"
            )
        return (
            "Record type: degree requirement category pool.\n"
            f"Program: {row.get('program')} / {row.get('degree_code')}.\n"
            f"Category: {category}.\n"
            f"Minimum: {_format_category_minimum(row)}.\n"
            f"Selection rule: {row.get('selection_rule') or row.get('full_pool_note') or ''}\n"
            f"Courses: {courses}.\n"
            f"Choice pools: {'; '.join(choice_pools) if choice_pools else 'none'}."
        )
    if record_type == "official_degree_evaluation_projection":
        categories = []
        for category in row.get("categories", []) or []:
            if not isinstance(category, dict):
                continue
            courses = _degree_course_codes(category.get("courses") or [])
            categories.append(
                f"{category.get('label_tr') or category.get('category')}: "
                f"{category.get('completed_su_credits')}/{category.get('min_su_credits')} SU, "
                f"{category.get('course_count')} courses, courses [{courses}]"
            )
        return (
            "Record type: official degree evaluation projection.\n"
            f"Authority: {row.get('source_authority')}.\n"
            f"Applies to: {row.get('applies_to')}.\n"
            f"Evaluation term: {row.get('evaluation_term_label')}.\n"
            f"Result: {row.get('result')}.\n"
            f"Completed total: {row.get('completed_total_su_credits')}/125 SU credits, {row.get('completed_total_ects')} ECTS.\n"
            f"GPA: program {row.get('program_gpa')}, cumulative {row.get('cumulative_gpa')}.\n"
            f"Category allocations: {'; '.join(categories)}."
        )
    return json.dumps(row, ensure_ascii=False, separators=(",", ":"))


def _format_category_minimum(item: dict) -> str:
    pieces = [str(item.get("label_tr") or item.get("category") or "category")]
    if item.get("min_su_credits") is not None:
        pieces.append(f"min {item.get('min_su_credits')} SU")
    if item.get("min_courses") is not None:
        pieces.append(f"min {item.get('min_courses')} courses")
    if item.get("min_ects") is not None:
        pieces.append(f"min {item.get('min_ects')} ECTS")
    return " ".join(pieces)


def _degree_course_codes(courses: list[dict]) -> str:
    codes: list[str] = []
    for course in courses:
        if not isinstance(course, dict):
            continue
        code = course.get("code")
        credits = course.get("su_credits")
        if code and credits is not None:
            codes.append(f"{code} ({credits} SU)")
        elif code:
            codes.append(str(code))
    return ", ".join(codes) if codes else "none"


def _syllabus_like(question: str) -> bool:
    return bool(
        re.search(
            r"\b(syllabus|outline|weekly plan|haftalık plan|haftalik plan|içerik|icerik|notlandırma|notlandirma)\b",
            question or "",
            re.IGNORECASE,
        )
    )


def _course_detail_support_documents(question: str) -> list[Document]:
    docs: list[Document] = []
    for raw_code in COURSE_CODE_RE.findall(question or "")[:3]:
        cleaned = _clean_code(raw_code)
        display = _display_code(cleaned)
        catalog = _course_catalog_row(cleaned)
        schedules = _schedule_rows_for_term(RECOMMENDATION_TERM).get(cleaned, [])
        if not catalog and not schedules:
            continue
        title = (catalog or schedules[0]).get("title") or (catalog or schedules[0]).get("Course_Name") or ""
        details = [
            f"Course: {display} - {title}",
            f"Schedule {RECOMMENDATION_TERM}: {_meeting_summary(schedules) if schedules else 'not found in loaded schedule context'}",
        ]
        if catalog:
            details.extend(
                [
                    f"SU credits: {catalog.get('su_credits')}",
                    f"ECTS: {catalog.get('ects')}",
                    f"Prerequisites: {catalog.get('prerequisites') or 'not listed'}",
                    f"Corequisites: {catalog.get('corequisites') or 'not listed'}",
                    f"Catalog description: {catalog.get('description') or 'not listed'}",
                    f"Source URL: {catalog.get('source_url') or 'not listed'}",
                ]
            )
        docs.append(
            Document(
                page_content="\n".join(details),
                metadata={
                    "source": f"structured course detail {display}",
                    "document_type": "structured_course_detail",
                    "course_code": display,
                    "chunk_id": f"structured:course_detail:{cleaned}",
                },
            )
        )
    return docs


def _instructor_schedule_support_documents(question: str) -> list[Document]:
    tokens = _instructor_query_tokens(question)
    if not tokens:
        return []

    docs: list[Document] = []
    seen: set[str] = set()
    for course_rows in _schedule_rows_for_term(RECOMMENDATION_TERM).values():
        for row in course_rows:
            instructor_text = " ".join(
                str(meeting.get("instructors") or "")
                for meeting in row.get("meetings") or []
                if isinstance(meeting, dict)
            )
            normalized_instructor = _normalize_match_text(instructor_text)
            if not all(token in normalized_instructor for token in tokens):
                continue
            course_id = _display_code(_clean_code(str(row.get("course_id") or "")))
            key = f"{course_id}:{row.get('section')}:{row.get('crn')}"
            if key in seen:
                continue
            seen.add(key)
            text = (
                f"Term: {RECOMMENDATION_TERM}.\n"
                f"Instructor match: {re.sub(r'\\s+', ' ', instructor_text).replace('( P )', '').strip()}.\n"
                f"Course: {course_id} - {row.get('title') or ''}.\n"
                f"Component/section: {row.get('component') or ''} {row.get('section') or ''}.\n"
                f"Schedule: {_meeting_summary([row]) or 'not listed'}.\n"
                "This is official schedule/catalog context only, not a student review."
            )
            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": f"structured instructor schedule {course_id}",
                        "document_type": "structured_instructor_schedule",
                        "course_code": course_id,
                        "term_code": RECOMMENDATION_TERM,
                        "chunk_id": f"structured:instructor_schedule:{RECOMMENDATION_TERM}:{_clean_code(course_id)}:{row.get('section') or 'section'}",
                    },
                )
            )
            if len(docs) >= 6:
                return docs
    return docs


def _instructor_query_tokens(question: str) -> list[str]:
    stop_words = {
        "hoca",
        "hocam",
        "prof",
        "professor",
        "zor",
        "kolay",
        "nasil",
        "nasıl",
        "mi",
        "mu",
        "mı",
        "mü",
        "yorum",
        "review",
        "ders",
        "course",
    }
    tokens = []
    for token in re.findall(r"[A-Za-zÇĞİÖŞÜçğıöşü]+", question or ""):
        normalized = _normalize_match_text(token)
        if len(normalized) > 2 and normalized not in stop_words:
            tokens.append(normalized)
    return tokens[:3]


def _normalize_match_text(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value or "")
    ascii_text = decomposed.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", ascii_text.lower()).strip()


@lru_cache(maxsize=1024)
def _course_catalog_row(cleaned_code: str) -> dict:
    path = Path(CATALOG_DATA_DIR) / "all_coursepage_info.jsonl"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            row_code = _clean_code(row.get("course_id") or f"{row.get('subj_code', '')}{row.get('crse_numb', '')}")
            if row_code == cleaned_code:
                return row
    return {}


def _general_knowledge_mode_document(intent: str) -> Document:
    practical_next_steps = (
        "Practical next steps to suggest (in the user's language): (1) check course/department "
        "WhatsApp or Telegram groups for what current/past students say, and (2) email the "
        "instructor directly with a short note about the student's background and ask whether "
        "the course is a good fit -- offer to draft a short sample email with placeholders "
        "such as [DERS KODU] / [Hoca Adı] / [Adınız]. Be upfront that this system is still in "
        "development and has no local archive for this, and frame any general observation as a "
        "general, non-official, person-dependent opinion -- never as a Sabanci-specific fact."
    )
    if intent == "review":
        text = (
            "No local instructor review archive is configured for this deployment "
            "(development-stage system without a review dataset). Answer from general academic "
            "advising knowledge and any course facts in the question. Do not claim that the answer "
            "is based on Sabanci student review data, and avoid precise claims about a specific "
            "instructor's grading, workload, personality, or past sentiment. " + practical_next_steps
        )
        source = "general knowledge review mode"
    elif intent == "exam":
        text = (
            "No local past exam/PDF archive is configured for this deployment "
            "(development-stage system without an exam dataset). Answer from general academic and "
            "course-topic knowledge about what kinds of topics or question styles such a course "
            "typically covers. Do not invent exact past exam questions, dates, or official "
            "solutions; clearly say that no local exam archive was used. " + practical_next_steps
        )
        source = "general knowledge exam mode"
    else:
        text = (
            "No local syllabus archive is configured for this deployment "
            "(development-stage system without a syllabus dataset). Use available catalog/schedule "
            "details if present, then answer from general academic knowledge about what such a "
            "course typically covers. Do not claim access to an official syllabus unless it "
            "appears in context. " + practical_next_steps
        )
        source = "general knowledge syllabus mode"
    return Document(
        page_content=text,
        metadata={
            "source": source,
            "document_type": "general_knowledge_mode",
            "intent": intent,
            "chunk_id": f"structured:general_knowledge:{intent}",
        },
    )


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
            "Detected intent: review / instructor or course review. If local review chunks are present, "
            "lead with them as grounded student sentiment, and you may add a brief (1-2 sentence) "
            "supplementary tip to also check WhatsApp/Telegram groups or email the instructor. If the "
            "context says no local review archive is configured, follow the general-knowledge mode "
            "guidance: be transparent, then suggest WhatsApp/Telegram groups and offer a short draft "
            "email to the instructor."
        )
    elif intent == "exam":
        text = (
            "[Source: Request intent]\n"
            "Detected intent: exam / past exam or assessment question. If local exam chunks are present, "
            "lead with them as grounded evidence, and you may add a brief (1-2 sentence) supplementary "
            "tip to also check WhatsApp/Telegram groups or email the instructor. If the context says no "
            "local exam archive is configured, follow the general-knowledge mode guidance: be "
            "transparent, then suggest WhatsApp/Telegram groups and offer a short draft email to the "
            "instructor, without inventing exact past questions."
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
                "chunk_id": f"structured:recommendation:{interest_key}:strategy",
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
                    "chunk_id": f"structured:recommendation:{interest_key}:{cleaned}",
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
    if payload.username != ADMIN_USERNAME or payload.password != ADMIN_PASSWORD:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    try:
        await ensure_user(ADMIN_USERNAME)
    except Exception:
        logger.exception("Could not sync admin user to MongoDB")
    return {"username": ADMIN_USERNAME, "role": "admin"}


@app.get("/courses/")
async def get_courses(search: str = "", limit: int = 200):
    return {"courses": await list_courses(search=search, limit=limit)}


@app.get("/users/{username}/courses")
async def get_selected_courses(username: str):
    return {"courses": await get_user_courses(username)}


@app.put("/users/{username}/courses")
async def save_selected_courses(username: str, payload: CourseSelectionPayload):
    return {"courses": await set_user_courses(username, payload.course_ids)}


@app.post("/ask/")
async def ask_question(question: str = Form(...), username: str | None = Form(None)):
    try:
        logger.info(f"user query: {question}")

        detected_intent = get_intent(question)
        resolved_intent = _resolve_intent(question, detected_intent)
        route = route_query(question, resolved_intent)
        intent = route.intent
        logger.info(
            "detected intent=%s resolved intent=%s route intent=%s document_types=%s confidence=%.3f",
            detected_intent,
            resolved_intent,
            intent,
            route.document_types,
            route.confidence,
        )

        graduation_intent = intent == "mezuniyet_durumu"
        recommendation_intent = intent == "ders_onerisi"
        if recommendation_intent and not _has_interest_area(question):
            return {
                "response": "Hangi alana ilgilisin? Örn: NLP, Web, Data, Systems, AI, Security.",
                "sources": [],
                "source_chunk_ids": [],
                "intent": intent,
            }

        effective_question = _expanded_recommendation_question(question) if recommendation_intent else question
        retrieval_query = _retrieval_query_for_intent(effective_question, intent)

        taken_codes: list[str] = []
        if recommendation_intent and username:
            selected_courses = await get_user_courses(username)
            taken_codes = _course_codes(selected_courses)
            user_context = _compact_recommendation_user_context(username, selected_courses)
        else:
            user_context = await get_user_course_context(username)

        structured_docs = _structured_context_documents(intent, effective_question, taken_codes)
        if structured_docs is None:
            vectorstore = get_vectorstore()
            candidate_docs = _retrieve_for_route(vectorstore, route, retrieval_query)
            if intent == "diger" and not candidate_docs:
                return {"response": NON_ACADEMIC_FALLBACK, "sources": [], "source_chunk_ids": [], "intent": intent}
            reranked_docs = rerank_documents(retrieval_query, candidate_docs, top_k=RERANK_TOP_K)
        else:
            reranked_docs = structured_docs

        context_docs = _format_for_context(reranked_docs)
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
        if recommendation_intent and structured_docs is None:
            context_docs = _recommendation_support_documents(effective_question, taken_codes) + context_docs
        context_docs.insert(0, _intent_context_document(intent, taken_codes))

        retriever = StaticRetriever(documents=context_docs)
        chain = get_llm_chain(retriever, intent=intent)
        llm_question = _recommendation_question_for_llm(effective_question, taken_codes) if recommendation_intent else question
        result = query_chain(chain, llm_question)
        if recommendation_intent:
            result = _clean_recommendation_response(result)
        result["intent"] = intent

        logger.info("query successful")
        return result

    except Exception as e:
        logger.exception("Error processing question")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/test")
async def test():
    return {"message":"Testing successfull..."}
