from __future__ import annotations

"""Deterministic course-to-curriculum requirement lookup.

Questions such as "Is CS306 required based on my admit term?" are not open-ended RAG
questions. The answer is in one official JSONL row for the student's program/curriculum term,
so the backend should answer it directly instead of asking the LLM to infer the category from a
mixed retrieval context.
"""

import re
from dataclasses import dataclass

from modules import degree_audit, intents


COURSE_REQUIREMENT_LOOKUP = intents.COURSE_REQUIREMENT_LOOKUP


_COURSE_SUBJECT_PATTERN = (
    r"ACC|ACCA|ANTH|BIO|CHEM|CIP|CONF|CS|CULT|DSA|ECON|EE|ENS|ENT|ENRG|FILM|FIN|"
    r"FRE|GEN|GER|HART|HIST|HUM|IE|IF|IR|LAW|LIT|MATH|MAT|ME|MGMT|MKTG|NS|OPIM|"
    r"ORG|PHIL|PHYS|POLS|PROJ|PSIR|PSY|SOC|SPA|SPS|TLL|TS|TUR|VA|VIS|XM"
)
_COURSE_CODE_RE = re.compile(
    rf"\b({_COURSE_SUBJECT_PATTERN})\s*-?\s*(\d{{3,5}})\b",
    re.IGNORECASE,
)
_LOOKUP_CUE_RE = re.compile(
    r"\b("
    r"required|requirement|requirements|requirement\s+(?:group|category)|category|heading|pool|"
    r"count(?:s|ed)?\s+(?:toward|towards|as|for|in)|fit(?:s)?\s+(?:in|into)|"
    r"based\s+on\s+my\s+(?:admit|admission|curriculum)\s+term|for\s+my\s+curriculum|"
    r"zorunlu|gereklilik|gereklilik\s+grubu|kategori|başlık|baslik|havuz|"
    r"say(?:ı|i)l(?:ı|i)r|say(?:ı|i)l(?:ı|i)yor|"
    r"m[üu]fredat(?:ı|i|ım|im|ımda|imda|ında|inda)?|"
    r"hangi\s+(?:kategori|başlık|baslik|gereklilik|grup)"
    r")\b",
    re.IGNORECASE,
)
_PREREQ_CUE_RE = re.compile(
    r"\b(prereq(?:uisite)?|pre-?req|ön\s*koşul|onkosul|before\s+taking|"
    r"complete\s+before|must\s+(?:i\s+)?(?:complete|take)\s+before)\b",
    re.IGNORECASE,
)


_CATEGORY_PRIORITY = {
    "university_courses_mandatory": 0,
    "university_courses_hum_pool": 1,
    "university_courses": 2,
    "required_courses": 3,
    "core_electives": 4,
    "area_electives": 5,
    "free_electives": 6,
}

_CATEGORY_LABELS_EN = {
    "university_courses_mandatory": "university courses",
    "university_courses_hum_pool": "university courses HUM pool",
    "university_courses": "university courses",
    "required_courses": "required courses",
    "core_electives": "core electives",
    "area_electives": "area electives",
    "free_electives": "free electives",
    "faculty_courses_fens": "FENS faculty courses",
}
_CATEGORY_LABELS_TR = {
    "university_courses_mandatory": "universite dersleri",
    "university_courses_hum_pool": "universite dersleri HUM havuzu",
    "university_courses": "universite dersleri",
    "required_courses": "zorunlu dersler",
    "core_electives": "cekirdek secmeliler",
    "area_electives": "alan secmelileri",
    "free_electives": "serbest secmeliler",
    "faculty_courses_fens": "FENS fakultesi dersleri",
}


@dataclass(frozen=True)
class RequirementLookup:
    course_code: str
    course_title: str
    category: str
    category_label: str
    is_required_course: bool
    source: str
    confidence_status: str = "verified"


@dataclass(frozen=True)
class RequirementLookupAnswer:
    body: str
    summary: str
    sources: list[str]
    source_chunk_ids: list[str]
    found: bool


def normalize_course_code(value: str) -> str:
    cleaned = re.sub(r"[\s-]+", "", str(value or "")).upper()
    match = re.match(r"^([A-Z]{2,5})(\d{3,5})$", cleaned)
    return f"{match.group(1)} {match.group(2)}" if match else cleaned


def extract_course_code(question: str) -> str | None:
    match = _COURSE_CODE_RE.search(question or "")
    if not match:
        return None
    return normalize_course_code("".join(match.groups()))


def is_lookup_question(question: str) -> bool:
    text = question or ""
    return bool(
        extract_course_code(text)
        and _LOOKUP_CUE_RE.search(text)
        and not _PREREQ_CUE_RE.search(text)
    )


def _label(category: str, language: str) -> str:
    labels = _CATEGORY_LABELS_EN if language == "en" else _CATEGORY_LABELS_TR
    return labels.get(category, category.replace("_", " "))


def _candidate_categories(model: dict, course_code: str) -> list[str]:
    found: list[str] = []
    for category, courses in (model.get("pools") or {}).items():
        if course_code in set(courses or []):
            found.append(str(category))
    return sorted(found, key=lambda c: (_CATEGORY_PRIORITY.get(c, 99), c))


def lookup(
    *,
    program: str,
    curriculum_term: str,
    course_code: str,
    language: str = "tr",
) -> RequirementLookup | None:
    normalized_program = str(program or "").strip().upper()
    normalized_term = str(curriculum_term or "").strip()
    normalized_course = normalize_course_code(course_code)
    model = degree_audit.load_requirements(normalized_program, normalized_term)
    if not model:
        return None
    categories = _candidate_categories(model, normalized_course)
    if not categories:
        return RequirementLookup(
            course_code=normalized_course,
            course_title=model.get("title_of", {}).get(normalized_course, ""),
            category="not_listed",
            category_label="not listed" if language == "en" else "listelenmemis",
            is_required_course=False,
            source=f"degree_requirements/{normalized_program}/{normalized_term}.jsonl",
            confidence_status="verified",
        )
    category = categories[0]
    return RequirementLookup(
        course_code=normalized_course,
        course_title=model.get("title_of", {}).get(normalized_course, ""),
        category=category,
        category_label=_label(category, language),
        is_required_course=category == "required_courses",
        source=f"degree_requirements/{normalized_program}/{normalized_term}.jsonl",
    )


def render_answer(
    result: RequirementLookup,
    *,
    program: str,
    curriculum_term: str,
    language: str = "tr",
) -> RequirementLookupAnswer:
    code_title = (
        f"{result.course_code} - {result.course_title}"
        if result.course_title else result.course_code
    )
    source = result.source
    if language == "en":
        if result.category == "not_listed":
            body = (
                f"For {program.upper()} curriculum/admit term {curriculum_term}, "
                f"{code_title} is not listed in the official requirement pools I have. "
                f"So I cannot verify that it counts toward that curriculum from the local "
                f"official data.\n\nSource: {source}"
            )
            summary = f"{result.course_code} is not listed for {program.upper()} {curriculum_term}."
        elif result.is_required_course:
            body = (
                f"Yes. For {program.upper()} curriculum/admit term {curriculum_term}, "
                f"{code_title} is listed under {result.category_label}; that means it is a "
                f"required course for that curriculum.\n\nSource: {source}"
            )
            summary = f"{result.course_code} is required for {program.upper()} {curriculum_term}."
        else:
            body = (
                f"No. For {program.upper()} curriculum/admit term {curriculum_term}, "
                f"{code_title} is not listed under required courses; it is listed under "
                f"{result.category_label}.\n\nSource: {source}"
            )
            summary = (
                f"{result.course_code} is {result.category_label}, not required, "
                f"for {program.upper()} {curriculum_term}."
            )
    else:
        if result.category == "not_listed":
            body = (
                f"{program.upper()} {curriculum_term} mufredati icin {code_title}, elimdeki "
                f"resmi gereklilik havuzlarinda listelenmiyor. Bu yuzden bu mufredatta "
                f"sayildigini yerel resmi veriden dogrulayamiyorum.\n\nKaynak: {source}"
            )
            summary = f"{result.course_code}, {program.upper()} {curriculum_term} icin listelenmiyor."
        elif result.is_required_course:
            body = (
                f"Evet. {program.upper()} {curriculum_term} mufredati icin {code_title}, "
                f"{result.category_label} kategorisinde listeleniyor; yani bu mufredatta "
                f"zorunlu derstir.\n\nKaynak: {source}"
            )
            summary = f"{result.course_code}, {program.upper()} {curriculum_term} icin zorunlu derstir."
        else:
            body = (
                f"Hayir. {program.upper()} {curriculum_term} mufredati icin {code_title}, "
                f"zorunlu dersler altinda degil; {result.category_label} kategorisinde "
                f"listeleniyor.\n\nKaynak: {source}"
            )
            summary = (
                f"{result.course_code}, {program.upper()} {curriculum_term} icin zorunlu degil; "
                f"{result.category_label} kategorisinde."
            )
    return RequirementLookupAnswer(
        body=body,
        summary=summary,
        sources=[source],
        source_chunk_ids=[],
        found=result.category != "not_listed",
    )
