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
    r"içeriği|icerigi|ders\s+içeri|ders\s+iceri|course\s+content|what\s+is|"
    r"what\s+does\s+.+\s+cover|about|kim\s+veriyor|hangi\s+hoca|"
    r"instructor|who\s+teaches"
    r")",
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
    return bool(extract_course_code(question) and _COURSE_DETAIL_CUE_RE.search(question or ""))


def is_instructor_teaching_question(question: str) -> bool:
    text = question or ""
    if _INSTRUCTOR_OPINION_RE.search(text):
        return False
    return bool(_TEACHING_QUESTION_RE.search(text) and _extract_instructor_name(text))


def is_instructor_fact_question(question: str) -> bool:
    # Deliberately narrow: "X kim?" is a biography/person question and stays out of scope.
    # "X hangi dersleri veriyor?" is an official academic schedule lookup.
    return is_instructor_teaching_question(question)


def is_course_fact_question(question: str) -> bool:
    return is_course_overview_question(question)


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


@lru_cache(maxsize=1)
def _known_instructor_names() -> dict[str, str]:
    names: dict[str, str] = {}
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


def instructor_teaching_lookup(question: str, *, language: str = "tr") -> OfficialAnswer | None:
    if not is_instructor_fact_question(question):
        return None
    instructor = _extract_instructor_name(question)
    if not instructor:
        return None
    term = _latest_term("schedule")
    if not term:
        return None
    instructor_key = _norm(instructor)
    rows = [
        row for row in _schedule_rows(term)
        if instructor_key in _norm(str(row.get("instructors") or ""))
    ]
    if not rows:
        return None

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
    return instructor_teaching_lookup(question, language=language) or course_overview(
        question,
        language=language,
    )


def render_instructor_fact_answer(question: str, *, language: str = "tr") -> OfficialAnswer | None:
    return instructor_teaching_lookup(question, language=language)


def render_course_fact_answer(question: str, *, language: str = "tr") -> OfficialAnswer | None:
    return course_overview(question, language=language)
