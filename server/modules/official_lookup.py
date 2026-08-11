from __future__ import annotations

"""Small deterministic official-data lookups for common academic questions.

These are intentionally outside the LLM/RAG answer path.  If a question can be answered from one
bounded official JSONL snapshot -- a course's catalog card, current scheduled sections, or the
current instructor-to-course listing -- the backend should do that directly.  This prevents
ordinary academic questions from becoming opaque "cannot verify" failures just because a provider
answer did not pass the generic claim-coverage gate.
"""

import json
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from modules import intents
from modules.config import DEGREE_DATA_DIR
from modules.requirement_lookup import normalize_course_code


_COURSE_SUBJECT_PATTERN = (
    r"ACC|ACCA|ANTH|BIO|CHEM|CIP|CONF|CS|CULT|DSA|ECON|EE|ENS|ENT|ENRG|FILM|FIN|"
    r"FRE|GEN|GER|HART|HIST|HUM|IE|IF|IR|LAW|LIT|MATH|MAT|ME|MGMT|MKTG|NS|OPIM|"
    r"ORG|PHIL|PHYS|POLS|PROJ|PSIR|PSY|SOC|SPA|SPS|TLL|TS|TUR|VA|VIS|XM"
)
_COURSE_CODE_RE = re.compile(
    rf"\b({_COURSE_SUBJECT_PATTERN})\s*-?\s*(\d{{3,5}}[A-Z]?)"
    r"(?:['’]?(?:i|ı|u|ü|yi|yı|yu|yü|e|a|ye|ya|de|da|te|ta|den|dan|ten|tan|nin|nın|nun|nün|in|ın|un|ün))?\b",
    re.IGNORECASE,
)
_COURSE_DETAIL_CUE_RE = re.compile(
    r"\b("
    r"neyi\s+anlat|ne\s+anlat|ne\s+öğret|ne\s+ogret|nedir|hakkında|hakkinda|"
    r"içeriği|icerigi|ders\s+içeri|ders\s+iceri|course\s+content|learning\s+outcomes?|"
    r"course\s+outcomes?|öğrenme\s+çıktı|ogrenme\s+cikti|kısa\s+özet|kisa\s+ozet|"
    r"short\s+summary|summary|summari[sz]e|özet|ozet|what\s+is|what\s+are|"
    r"what\s+does\s+.+\s+cover|understand|explain|about|kim\s+veriyor|hangi\s+hoca|"
    r"instructor|who\s+teaches"
    r")",
    re.IGNORECASE,
)
_PREREQUISITE_CUE_RE = re.compile(
    r"\b(prereq(?:uisite)?s?|pre-?reqs?|ön\s*koşul|onkosul|önkoşul|"
    r"before\s+taking|almak\s+için\s+ne\s+gerek|almak\s+icin\s+ne\s+gerek)\b",
    re.IGNORECASE,
)
_OFFERING_CUE_RE = re.compile(
    r"\b(offered|open|available|this\s+(?:fall|semester|term)|fall\s+\d{4}|"
    r"açılıyor|aciliyor|açılır|acilir|bu\s+dönem|bu\s+donem|güz|guz|take)\b",
    re.IGNORECASE,
)
_TEACHING_CUE_RE = re.compile(
    r"\b("
    r"hangi\s+dersleri?|ne\s+dersleri?|dersleri?\s+(?:veriyor|verir|vermekte)|"
    r"verdiği\s+dersler|verdigi\s+dersler|"
    r"which\s+courses?|what\s+courses?|teaches|is\s+teaching"
    r")\b",
    re.IGNORECASE,
)
_TEACHING_QUESTION_RE = re.compile(
    r"\b("
    r"hangi\s+dersleri?\s+(?:veriyor|verir|vermekte)|"
    r"ne\s+dersleri?\s+(?:veriyor|verir|vermekte)|"
    r"verdiği\s+dersler|verdigi\s+dersler|"
    r"which\s+courses?\s+does\s+.+\s+teach|"
    r"what\s+courses?\s+does\s+.+\s+teach|"
    r".+\s+teaches\s+which\s+courses?"
    r")\b",
    re.IGNORECASE,
)
_PROFILE_QUESTION_RE = re.compile(
    r"\b(?:"
    r"kim(?:dir)?|who\s+is|biograph\w*|bio\b|profile|research|research\s+areas?|"
    r"araştırma\s+alan\w*|arastirma\s+alan\w*|ne\s+çalışır|ne\s+calisir|"
    r"hakkında\s+bilgi|hakkinda\s+bilgi"
    r")\b",
    re.IGNORECASE,
)
_INSTRUCTOR_OPINION_RE = re.compile(
    r"\b(?:nasıl\s+biri|nasil\s+biri|zor\s+mu|kolay\s+m[ıi]|yorum|review|rate|"
    r"iyi\s+mi|kötü\s+m[üu]|kotu\s+mu|sevil|seviliyor|önerir\s+misin|onerir\s+misin)\b",
    re.IGNORECASE,
)
_TURKISH_FOLD = str.maketrans({
    "ı": "i", "İ": "i", "ş": "s", "Ş": "s", "ğ": "g", "Ğ": "g",
    "ü": "u", "Ü": "u", "ö": "o", "Ö": "o", "ç": "c", "Ç": "c",
})


@dataclass(frozen=True)
class OfficialAnswer:
    body: str
    summary: str
    sources: list[str]
    source_chunk_ids: list[str]
    intent: str
    course_id: str | None = None
    confidence_status: str = "verified"


def _data_root() -> Path:
    return Path(DEGREE_DATA_DIR)


def _norm(value: str) -> str:
    folded = str(value or "").translate(_TURKISH_FOLD)
    folded = unicodedata.normalize("NFKD", folded).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", " ", folded.lower()).strip()


def extract_course_code(question: str) -> str | None:
    match = _COURSE_CODE_RE.search(question or "")
    if not match:
        return None
    return normalize_course_code("".join(match.groups()))


def is_course_overview_question(question: str) -> bool:
    text = question or ""
    if re.search(r"\bwhat\s+are\s+the\s+details\s+of\b", text, re.IGNORECASE):
        return False
    return bool(extract_course_code(text) and _COURSE_DETAIL_CUE_RE.search(text))


def is_instructor_teaching_question(question: str) -> bool:
    text = question or ""
    if _INSTRUCTOR_OPINION_RE.search(text):
        return False
    return bool(_TEACHING_QUESTION_RE.search(text) and _extract_instructor_name(text))


def is_instructor_fact_question(question: str) -> bool:
    return is_instructor_teaching_question(question)


def is_instructor_profile_question(question: str) -> bool:
    text = question or ""
    if _INSTRUCTOR_OPINION_RE.search(text):
        return False
    return bool(_PROFILE_QUESTION_RE.search(text) and _extract_instructor_name(text))


def is_course_fact_question(question: str) -> bool:
    return is_course_overview_question(question)


def is_prerequisite_question(question: str) -> bool:
    return bool(extract_course_code(question) and _PREREQUISITE_CUE_RE.search(question or ""))


def is_offering_question(question: str) -> bool:
    text = question or ""
    return bool(extract_course_code(text) and _OFFERING_CUE_RE.search(text))


def _jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    if not path.exists():
        return ()
    rows: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    except (OSError, json.JSONDecodeError):
        return ()
    return tuple(rows)


@lru_cache(maxsize=1)
def _catalog_rows() -> tuple[dict[str, Any], ...]:
    return _jsonl(_data_root() / "course_catalog" / "current.jsonl")


@lru_cache(maxsize=4)
def _schedule_rows(term: str) -> tuple[dict[str, Any], ...]:
    return _jsonl(_data_root() / "schedule" / f"{term}.jsonl")


@lru_cache(maxsize=4)
def _syllabus_rows(term: str) -> tuple[dict[str, Any], ...]:
    return _jsonl(_data_root() / "syllabi" / f"{term}.jsonl")


@lru_cache(maxsize=1)
def _all_syllabus_rows() -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    root = _data_root() / "syllabi"
    if not root.exists():
        return ()
    for path in sorted(root.glob("20*.jsonl")):
        for row in _jsonl(path):
            copied = dict(row)
            copied["_source_path"] = f"syllabi/{path.name}"
            rows.append(copied)
    return tuple(rows)


@lru_cache(maxsize=2)
def _latest_term(kind: str) -> str | None:
    pointer = _data_root() / kind / "latest.json"
    if not pointer.exists():
        return None
    try:
        payload = json.loads(pointer.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    term = str(payload.get("term") or payload.get("latest_term") or "").strip()
    return term or None


def _catalog_by_code(course_code: str) -> dict[str, Any] | None:
    normalized = normalize_course_code(course_code)
    return next((row for row in _catalog_rows() if row.get("course_id") == normalized), None)


def _course_schedule_rows(course_code: str, term: str) -> list[dict[str, Any]]:
    normalized = normalize_course_code(course_code)
    return [row for row in _schedule_rows(term) if row.get("course_id") == normalized]


def _course_syllabus_rows(course_code: str, term: str) -> list[dict[str, Any]]:
    normalized = normalize_course_code(course_code)
    return [row for row in _syllabus_rows(term) if row.get("course_id") == normalized]


def _requested_or_latest_term(question: str) -> str | None:
    """Resolve the small set of demo/user-facing term phrasings to the local schedule term.

    The schedule snapshot's ``latest.json`` is authoritative for "this term" and for the
    Fall 2026-2027 demo corpus.  If a future snapshot is loaded, the latest pointer keeps this
    helper from hard-coding a stale term into the product path.
    """
    latest = _latest_term("schedule")
    text = question or ""
    if re.search(r"\bfall\s+2026(?:\s*[-/]\s*2027)?\b|2026\s*[-/]\s*2027|\bgüz\b|\bguz\b", text, re.I):
        return latest
    return latest


def _prerequisite_text(course_code: str) -> str | None:
    """Return deterministic prerequisite text from the local academic rules layer."""
    try:
        from modules import course_planner
    except Exception:
        return None
    code = normalize_course_code(course_code)
    normalized = course_planner.normalize_code(code)
    minimum = getattr(course_planner, "MINIMUM_CREDIT_PREREQS", {}).get(normalized)
    if minimum is not None:
        return f"minimum {minimum} completed SU credits"
    alternatives = getattr(course_planner, "ALTERNATIVE_PREREQUISITE_PATHS", {}).get(normalized)
    if alternatives:
        paths = [
            " + ".join(course_planner.display_code(item) for item in path)
            for path in alternatives
        ]
        return " OR ".join(paths)
    direct = getattr(course_planner, "PREREQS", {}).get(normalized)
    if direct:
        return " + ".join(course_planner.display_code(item) for item in direct)
    return None


def _latest_published_syllabus_row(course_code: str) -> dict[str, Any] | None:
    normalized = normalize_course_code(course_code)
    candidates = [
        row
        for row in _all_syllabus_rows()
        if row.get("course_id") == normalized
        and str(row.get("publication_state") or "").strip().lower().startswith("published")
    ]
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda row: (
            str(row.get("term_code") or row.get("term") or ""),
            str(row.get("component") or "").lower() == "primary",
            str(row.get("section") or "") in {"0", "A"},
        ),
        reverse=True,
    )[0]


def _clean_sentence(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip(" -;:.")


def _learning_outcomes(row: dict[str, Any] | None, *, limit: int = 5) -> list[str]:
    text = str((row or {}).get("text") or "")
    if "Learning Outcomes:" not in text:
        return []
    after = text.split("Learning Outcomes:", 1)[1]
    after = re.split(
        r"\b(?:Course\s+Objective|Course\s+Content|Assessment|Grading|Weekly\s+Schedule|"
        r"Course\s+Material|Textbook|Prerequisite)s?:",
        after,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    outcomes: list[str] = []
    for match in re.finditer(r"(?:^|\s)(\d+)\.\s*(.*?)(?=\s+\d+\.\s+|$)", after, re.DOTALL):
        value = _clean_sentence(match.group(2))
        if value:
            outcomes.append(value[:320])
        if len(outcomes) >= limit:
            break
    return outcomes


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(v for v in values if v))


def _component_label(component: str, language: str) -> str:
    comp = str(component or "Primary")
    if language == "tr":
        return {
            "Primary": "Ders",
            "Recitation": "Problem saati",
            "Laboratory": "Laboratuvar",
            "Discussion": "Tartışma",
        }.get(comp, comp)
    return comp


def _course_source(term: str | None, *, schedule: bool = False, syllabus: bool = False) -> list[str]:
    sources = ["course_catalog/current.jsonl"]
    if term and schedule:
        sources.append(f"schedule/{term}.jsonl")
    if term and syllabus:
        sources.append(f"syllabi/{term}.jsonl")
    return sources


def course_overview(question: str, *, language: str = "tr") -> OfficialAnswer | None:
    code = extract_course_code(question)
    if not code or not is_course_overview_question(question):
        return None
    catalog = _catalog_by_code(code)
    schedule_term = _latest_term("schedule")
    syllabus_term = _latest_term("syllabi") or schedule_term
    schedule_rows = _course_schedule_rows(code, schedule_term) if schedule_term else []
    syllabus_rows = _course_syllabus_rows(code, syllabus_term) if syllabus_term else []
    published_syllabus_row = _latest_published_syllabus_row(code)
    outcomes = _learning_outcomes(published_syllabus_row)
    if not catalog and not schedule_rows and not syllabus_rows and not published_syllabus_row:
        return None

    title = (
        str((catalog or {}).get("title") or "")
        or str((schedule_rows[0] if schedule_rows else {}).get("section_title") or "")
        or str((syllabus_rows[0] if syllabus_rows else {}).get("course_title") or "")
    )
    display = f"{code} — {title}" if title else code
    primary_rows = [row for row in schedule_rows if str(row.get("component") or "") == "Primary"]
    instructors = _unique([
        str(row.get("instructors") or "").replace("( P )", "").replace("(P)", "").strip()
        for row in primary_rows or schedule_rows
    ])
    sections = _unique([str(row.get("section") or "").strip() for row in primary_rows])
    term_label = str((schedule_rows[0] if schedule_rows else {}).get("term_label") or schedule_term or "")
    published_syllabus = any(
        str(row.get("publication_state") or "").strip().lower().startswith("published")
        for row in syllabus_rows
    )
    chunk_ids = _unique([
        str(row.get("chunk_id") or "") for row in (schedule_rows[:4] + syllabus_rows[:4])
    ])
    sources = _course_source(
        schedule_term,
        schedule=bool(schedule_rows),
        syllabus=bool(syllabus_rows),
    )
    published_source = str((published_syllabus_row or {}).get("_source_path") or "")
    if published_source and published_source not in sources:
        sources.append(published_source)
    if published_syllabus_row and published_syllabus_row.get("chunk_id"):
        chunk_ids.append(str(published_syllabus_row["chunk_id"]))

    if language == "en":
        lines = [f"From the official local data I can verify this for **{display}**:"]
        if catalog:
            lines.append(
                f"- Catalog: {catalog.get('su_credits', '—')} SU / {catalog.get('ects', '—')} ECTS"
                + (f", faculty {catalog.get('faculty')}." if catalog.get("faculty") else ".")
            )
        if schedule_rows:
            instructor_text = f" Instructor(s): {', '.join(instructors)}." if instructors else ""
            section_text = f" Primary section(s): {', '.join(sections)}." if sections else ""
            lines.append(f"- Current schedule ({term_label}): offered.{section_text}{instructor_text}")
        if outcomes:
            published_term = str(
                published_syllabus_row.get("term_code") or published_syllabus_row.get("term") or ""
            )
            suffix = (
                f" (latest published syllabus term: {published_term})"
                if published_term and published_term != str(schedule_term or "")
                else ""
            )
            lines.append(f"- Course coverage{suffix}:")
            lines.extend(f"  - {item}" for item in outcomes)
        elif syllabus_rows and not published_syllabus:
            lines.append(
                "- A detailed published syllabus/course description is not available in the "
                "captured official syllabus data yet, so I am not inventing extra content beyond "
                "the catalog title and schedule facts."
            )
        summary = f"{code} is {title or 'listed in official course data'}."
        lines.append("")
        lines.append("Sources: " + ", ".join(sources))
    else:
        lines = [f"Resmi yerel veriden **{display}** için şunları doğrulayabiliyorum:"]
        if catalog:
            lines.append(
                f"- Katalog: {catalog.get('su_credits', '—')} SU / {catalog.get('ects', '—')} ECTS"
                + (f", fakülte {catalog.get('faculty')}." if catalog.get("faculty") else ".")
            )
        if schedule_rows:
            instructor_text = f" Öğretim üyesi: {', '.join(instructors)}." if instructors else ""
            section_text = f" Ana section(lar): {', '.join(sections)}." if sections else ""
            lines.append(f"- Güncel schedule ({term_label}): açılıyor.{section_text}{instructor_text}")
        if outcomes:
            published_term = str(
                published_syllabus_row.get("term_code") or published_syllabus_row.get("term") or ""
            )
            suffix = (
                f" (son yayımlanmış syllabus dönemi: {published_term})"
                if published_term and published_term != str(schedule_term or "")
                else ""
            )
            lines.append(f"- Dersin kapsadığı ana başlıklar/öğrenme çıktıları{suffix}:")
            lines.extend(f"  - {item}" for item in outcomes)
        elif syllabus_rows and not published_syllabus:
            lines.append(
                "- Yakalanan resmi syllabus verisinde ayrıntılı ders açıklaması henüz yayımlı "
                "görünmüyor; bu yüzden katalog başlığı ve schedule bilgisi dışında konu icat etmiyorum."
            )
        summary = f"{code}, resmi veride {title or 'kayıtlı'} olarak görünüyor."
        lines.append("")
        lines.append("Kaynaklar: " + ", ".join(sources))
    return OfficialAnswer(
        body="\n".join(lines).strip(),
        summary=summary,
        sources=sources,
        source_chunk_ids=chunk_ids,
        intent=intents.COURSE_DETAIL,
        course_id=code,
    )


def prerequisite_lookup(question: str, *, language: str = "tr") -> OfficialAnswer | None:
    if not is_prerequisite_question(question):
        return None
    code = extract_course_code(question)
    if not code:
        return None
    catalog = _catalog_by_code(code)
    title = str((catalog or {}).get("title") or "").strip()
    prereq = _prerequisite_text(code)
    if not prereq:
        body = (
            f"I could not verify official prerequisite data for **{code}** in the local academic rules."
            if language == "en"
            else f"Yerel akademik kurallarda **{code}** için resmi önkoşul verisini doğrulayamadım."
        )
        return OfficialAnswer(
            body=body,
            summary=body,
            sources=["Deterministic prerequisite rules"],
            source_chunk_ids=[],
            intent=intents.COURSE_PREREQUISITE_LOOKUP,
            course_id=code,
            confidence_status="cannot_verify",
        )
    display = f"{code} — {title}" if title else code
    if language == "en":
        body = f"**{display}** prerequisite: **{prereq}**.\n\nSource: Deterministic prerequisite rules"
        summary = f"{code} requires {prereq}."
    else:
        body = f"**{display}** önkoşulu: **{prereq}**.\n\nKaynak: Deterministic prerequisite rules"
        summary = f"{code} için önkoşul: {prereq}."
    return OfficialAnswer(
        body=body,
        summary=summary,
        sources=["Deterministic prerequisite rules", "course_catalog/current.jsonl"],
        source_chunk_ids=[],
        intent=intents.COURSE_PREREQUISITE_LOOKUP,
        course_id=code,
    )


def offering_lookup(question: str, *, language: str = "tr") -> OfficialAnswer | None:
    if not is_offering_question(question):
        return None
    code = extract_course_code(question)
    term = _requested_or_latest_term(question)
    if not code or not term:
        return None
    rows = _course_schedule_rows(code, term)
    catalog = _catalog_by_code(code)
    title = str((catalog or {}).get("title") or "").strip()
    term_label = str((rows[0] if rows else {}).get("term_label") or term)
    primary = [row for row in rows if str(row.get("component") or "") == "Primary"]
    crns = _unique([str(row.get("crn") or "") for row in (primary or rows)])
    display = f"{code} — {title}" if title else code
    if rows:
        if language == "en":
            body = (
                f"Yes — **{display}** is offered in **{term_label}**."
                + (f" Primary CRN(s): {', '.join(crns)}." if crns else "")
                + f"\n\nSource: schedule/{term}.jsonl"
            )
            summary = f"{code} is offered in {term_label}."
        else:
            body = (
                f"Evet — **{display}**, **{term_label}** döneminde açılıyor."
                + (f" Ana CRN(ler): {', '.join(crns)}." if crns else "")
                + f"\n\nKaynak: schedule/{term}.jsonl"
            )
            summary = f"{code}, {term_label} döneminde açılıyor."
        status = "verified"
    else:
        if language == "en":
            body = f"No offered section for **{display}** appears in **{term_label}**.\n\nSource: schedule/{term}.jsonl"
            summary = f"{code} is not offered in {term_label}."
        else:
            body = f"**{display}** için **{term_label}** döneminde açılan section görünmüyor.\n\nKaynak: schedule/{term}.jsonl"
            summary = f"{code}, {term_label} döneminde açılıyor görünmüyor."
        status = "verified"
    return OfficialAnswer(
        body=body,
        summary=summary,
        sources=[f"schedule/{term}.jsonl"],
        source_chunk_ids=_unique([str(row.get("chunk_id") or "") for row in rows[:4]]),
        intent=intents.COURSE_OFFERING_LOOKUP,
        course_id=code,
        confidence_status=status,
    )


@lru_cache(maxsize=1)
def _known_instructor_names() -> dict[str, str]:
    names: dict[str, str] = {}
    for row in _faculty_profile_rows():
        cleaned = str(row.get("name") or "").strip()
        if len(cleaned.split()) >= 2:
            key = _norm(cleaned)
            if key and key not in names:
                names[key] = cleaned
    for term in filter(None, {_latest_term("schedule"), _latest_term("syllabi"), "202502"}):
        for row in (*_schedule_rows(str(term)), *_syllabus_rows(str(term))):
            raw_values: list[str] = []
            if row.get("instructors"):
                raw_values.extend(str(row.get("instructors") or "").split("|"))
            for name in row.get("instructor_names") or []:
                raw_values.append(str(name))
            for raw in raw_values:
                cleaned = re.sub(r"\([^)]*\)", "", raw).strip()
                if len(cleaned.split()) < 2:
                    continue
                key = _norm(cleaned)
                if key and key not in names:
                    names[key] = cleaned
    return names


def _extract_instructor_name(question: str) -> str | None:
    q = _norm(question)
    matches = [
        (key, original)
        for key, original in _known_instructor_names().items()
        if key and re.search(rf"(?:^| ){re.escape(key)}(?: |$)", q)
    ]
    if not matches:
        return None
    # Prefer the most specific full-name match.
    return max(matches, key=lambda item: len(item[0]))[1]


@lru_cache(maxsize=1)
def _faculty_profile_rows() -> tuple[dict[str, Any], ...]:
    return _jsonl(_data_root() / "faculty_profiles" / "current.jsonl")


def _faculty_profile_by_name(instructor: str) -> dict[str, Any] | None:
    key = _norm(instructor)
    for row in _faculty_profile_rows():
        candidates = [str(row.get("name") or ""), *[str(alias) for alias in (row.get("aliases") or [])]]
        if any(_norm(candidate) == key for candidate in candidates):
            return row
    return None


def _instructor_rows(instructor: str, term: str) -> list[dict[str, Any]]:
    instructor_key = _norm(instructor)
    return [
        row for row in _schedule_rows(term)
        if instructor_key in _norm(str(row.get("instructors") or ""))
    ]


def _group_teaching_rows(rows: list[dict[str, Any]], *, language: str) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        course_id = str(row.get("course_id") or "").strip()
        if not course_id:
            continue
        item = grouped.setdefault(course_id, {
            "title": str(row.get("section_title") or "").split(" -")[0].strip(),
            "components": {},
            "chunk_ids": [],
        })
        component = _component_label(str(row.get("component") or "Primary"), language)
        item["components"].setdefault(component, [])
        section = str(row.get("section") or "").strip()
        if section and section not in item["components"][component]:
            item["components"][component].append(section)
        chunk_id = str(row.get("chunk_id") or "")
        if chunk_id:
            item["chunk_ids"].append(chunk_id)
    return grouped


def instructor_profile_lookup(question: str, *, language: str = "tr") -> OfficialAnswer | None:
    if not is_instructor_profile_question(question):
        return None
    instructor = _extract_instructor_name(question)
    if not instructor:
        return None
    term = _latest_term("schedule")
    schedule_rows = _instructor_rows(instructor, term) if term else []
    profile = _faculty_profile_by_name(instructor)
    syllabus_hits = [
        row for row in _all_syllabus_rows()
        if _norm(instructor) in " ".join(_norm(str(name)) for name in (row.get("instructor_names") or []))
    ][:4]
    if not profile and not schedule_rows and not syllabus_hits:
        return None

    sources: list[str] = []
    if profile:
        sources.append("faculty_profiles/current.jsonl")
        sources.extend(
            str(url)
            for url in (profile.get("source_urls") or profile.get("sources") or [])
            if str(url).strip()
        )
    if term and schedule_rows:
        sources.append(f"schedule/{term}.jsonl")
    for row in syllabus_hits:
        source_path = row.get("_source_path") or f"syllabi/{row.get('term_code') or row.get('term')}.jsonl"
        if source_path:
            sources.append(str(source_path))
    sources = _unique(sources)

    grouped = _group_teaching_rows(schedule_rows, language=language)
    term_label = str((schedule_rows[0] if schedule_rows else {}).get("term_label") or term or "")
    chunk_ids = _unique(
        [str(row.get("chunk_id") or "") for row in schedule_rows[:4]]
        + [str(row.get("chunk_id") or "") for row in syllabus_hits]
    )

    if language == "en":
        lines = [f"Here is what I can verify from official Sabancı University data for **{instructor}**:"]
        if profile:
            role_bits = [
                str(profile.get("role") or profile.get("display_title_en") or "").strip(),
                str(profile.get("unit") or "").strip(),
            ]
            role = " — ".join(bit for bit in role_bits if bit)
            if role:
                lines.append(f"- Role: {role}.")
            programs = [str(item) for item in profile.get("programs") or [] if str(item).strip()]
            if programs:
                lines.append(f"- Programs/teaching context: {', '.join(programs)}.")
            areas = [
                str(item)
                for item in (profile.get("research_areas_en") or profile.get("research_areas") or [])
                if str(item).strip()
            ]
            if areas:
                lines.append(f"- Research areas: {', '.join(areas)}.")
            education = [str(item) for item in profile.get("education") or [] if str(item).strip()]
            if education:
                lines.append(f"- Education: {'; '.join(education)}.")
            publications = [
                str(item)
                for item in (profile.get("selected_publications") or profile.get("publications") or [])
                if str(item).strip()
            ]
            if publications:
                lines.append("- Selected verified publications/projects:")
                lines.extend(f"  - {item}" for item in publications[:5])
            if profile.get("email"):
                lines.append(f"- Official email: {profile['email']}.")
        else:
            lines.append("- I found this person in official course data, but I do not have a local official personnel biography record.")
        if grouped:
            lines.append(f"- Current official teaching records ({term_label}):")
            for course_id, item in sorted(grouped.items()):
                title = item["title"]
                lines.append(f"  - **{course_id} — {title}**" if title else f"  - **{course_id}**")
        lines.append("")
        lines.append("I am not adding unsourced biography details beyond these official records.")
        lines.append("Sources: " + ", ".join(sources))
        summary = f"{instructor} is matched in official Sabancı data; profile details are source-limited."
    else:
        lines = [f"**{instructor}** için resmi Sabancı verilerinden doğrulayabildiklerim:"]
        if profile:
            role_bits = [
                str(profile.get("display_title_tr") or profile.get("role") or "").strip(),
                str(profile.get("unit") or "").strip(),
            ]
            role = " — ".join(bit for bit in role_bits if bit)
            if role:
                lines.append(f"- Görev: {role}.")
            programs = [str(item) for item in profile.get("programs") or [] if str(item).strip()]
            if programs:
                lines.append(f"- Program/ders bağlamı: {', '.join(programs)}.")
            areas = [
                str(item)
                for item in (profile.get("research_areas_tr") or profile.get("research_areas") or [])
                if str(item).strip()
            ]
            if areas:
                lines.append(f"- Araştırma alanları: {', '.join(areas)}.")
            education = [str(item) for item in profile.get("education") or [] if str(item).strip()]
            if education:
                lines.append(f"- Eğitim: {'; '.join(education)}.")
            publications = [
                str(item)
                for item in (profile.get("selected_publications") or profile.get("publications") or [])
                if str(item).strip()
            ]
            if publications:
                lines.append("- Kaynaklarda doğrulanan seçilmiş yayın/projeler:")
                lines.extend(f"  - {item}" for item in publications[:5])
            if profile.get("email"):
                lines.append(f"- Resmi e-posta: {profile['email']}.")
        else:
            lines.append("- Kişiyi resmi ders verisinde buldum; ancak yerel veride resmi personel biyografisi yok.")
        if grouped:
            lines.append(f"- Güncel resmi ders programı kayıtları ({term_label}):")
            for course_id, item in sorted(grouped.items()):
                title = item["title"]
                lines.append(f"  - **{course_id} — {title}**" if title else f"  - **{course_id}**")
        lines.append("")
        lines.append("Bu resmi kayıtların dışına çıkıp kaynaklanmamış biyografi detayı eklemiyorum.")
        lines.append("Kaynaklar: " + ", ".join(sources))
        summary = f"{instructor}, resmi Sabancı verilerinde eşleşiyor; profil bilgisi kaynaklarla sınırlı verildi."

    return OfficialAnswer(
        body="\n".join(lines).strip(),
        summary=summary,
        sources=sources,
        source_chunk_ids=chunk_ids,
        intent=intents.INSTRUCTOR_PROFILE_LOOKUP,
        confidence_status="verified" if profile else "insufficient_evidence",
    )


def instructor_teaching_lookup(question: str, *, language: str = "tr") -> OfficialAnswer | None:
    if not is_instructor_teaching_question(question):
        return None
    instructor = _extract_instructor_name(question)
    if not instructor:
        return None
    term = _latest_term("schedule")
    if not term:
        return None
    rows = _instructor_rows(instructor, term)
    if not rows:
        return None

    grouped = _group_teaching_rows(rows, language=language)

    term_label = str(rows[0].get("term_label") or term)
    source = f"schedule/{term}.jsonl"
    lines: list[str] = []
    if language == "en":
        lines.append(
            f"I have official course-schedule records, not a personnel biography. "
            f"According to the official {term_label} schedule, **{instructor}** appears in:"
        )
    else:
        lines.append(
            f"Elimde personel biyografisi değil, resmi ders programı kayıtları var. "
            f"Resmi {term_label} schedule verisine göre **{instructor}** şu derslerde görünüyor:"
        )
    for course_id, item in sorted(grouped.items()):
        title = item["title"]
        display = f"{course_id} — {title}" if title else course_id
        details = []
        for component, sections in item["components"].items():
            if sections:
                details.append(f"{component}: {', '.join(sections[:8])}" + ("…" if len(sections) > 8 else ""))
            else:
                details.append(component)
        lines.append(f"- **{display}** ({'; '.join(details)})")
    lines.append("")
    lines.append(("Source: " if language == "en" else "Kaynak: ") + source)
    course_summary = ", ".join(sorted(grouped.keys())[:5])
    summary = (
        f"{instructor} appears in {course_summary} in {term_label}."
        if language == "en"
        else f"{instructor}, {term_label} için {course_summary} derslerinde görünüyor."
    )
    chunk_ids = _unique([cid for item in grouped.values() for cid in item["chunk_ids"][:2]])
    return OfficialAnswer(
        body="\n".join(lines).strip(),
        summary=summary,
        sources=[source],
        source_chunk_ids=chunk_ids,
        intent=intents.INSTRUCTOR_TEACHING_LOOKUP,
        confidence_status="verified",
    )


def answer(question: str, *, language: str = "tr") -> OfficialAnswer | None:
    return (
        prerequisite_lookup(question, language=language)
        or offering_lookup(question, language=language)
        or instructor_teaching_lookup(question, language=language)
        or instructor_profile_lookup(question, language=language)
        or course_overview(
        question,
        language=language,
        )
    )


def render_instructor_fact_answer(question: str, *, language: str = "tr") -> OfficialAnswer | None:
    return instructor_teaching_lookup(question, language=language) or instructor_profile_lookup(
        question,
        language=language,
    )


def render_course_fact_answer(question: str, *, language: str = "tr") -> OfficialAnswer | None:
    return course_overview(question, language=language)
