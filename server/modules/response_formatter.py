from __future__ import annotations

"""Presentation helpers for backwards-compatible, structured chat responses.

The API keeps returning the historical ``response`` string, while newer clients can
also consume ``summary`` and ``structured_content``.  The helpers are deliberately
deterministic: formatting a degree audit or a course-history mutation must not require
another LLM call.
"""

import re
from typing import Any, Iterable


SUMMARY_HEADINGS = (
    "## Kısa Özet",
    "## Kisa Ozet",
    "## Short Summary",
)


def _plain_text(markdown: str) -> str:
    text = re.sub(r"```.*?```", " ", markdown or "", flags=re.DOTALL)
    text = re.sub(r"^\s*(?:#{1,6}|[-*+]|\d+\.)\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\|[-: |]+\|", " ", text)
    text = re.sub(r"[`*_>#]", "", text)
    text = re.sub(r"\[(.*?)\]\([^)]*\)", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def split_summary(answer: str) -> tuple[str, str]:
    """Return ``(body, summary)`` when the model already emitted a summary section."""
    for heading in SUMMARY_HEADINGS:
        index = answer.lower().find(heading.lower())
        if index >= 0:
            body = answer[:index].rstrip()
            summary = _plain_text(answer[index + len(heading) :])
            return body, summary
    return answer.rstrip(), ""


def compact_summary(answer: str, *, language: str = "tr", max_chars: int = 420) -> str:
    """Create a short extractive fallback when the model omitted its summary.

    This is intentionally conservative.  It reuses claims already present in the answer
    instead of asking a second model to generate another potentially inconsistent result.
    """
    _body, existing = split_summary(answer)
    if existing:
        return existing[:max_chars].rstrip()

    cleaned = _plain_text(re.sub(r"(?:^|\n)Sources?:.*$", "", answer or "", flags=re.DOTALL | re.IGNORECASE))
    if not cleaned:
        return (
            "Bu cevap için kısa özet üretilemedi."
            if language == "tr"
            else "A short summary could not be generated for this answer."
        )
    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    chosen: list[str] = []
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        candidate = " ".join([*chosen, sentence])
        if len(candidate) > max_chars and chosen:
            break
        chosen.append(sentence)
        if len(chosen) >= 2:
            break
    summary = " ".join(chosen) or cleaned[:max_chars]
    if len(summary) > max_chars:
        summary = summary[: max_chars - 1].rstrip() + "…"
    return summary


def ensure_summary_section(answer: str, *, language: str = "tr") -> tuple[str, str]:
    """Ensure the legacy markdown response ends with a compact summary section."""
    body, summary = split_summary(answer)
    summary = summary or compact_summary(body, language=language)
    heading = "## Kısa Özet" if language == "tr" else "## Short Summary"
    rendered = f"{body.rstrip()}\n\n{heading}\n\n{summary}".strip()
    return rendered, summary


def sanitize_student_answer(answer: str) -> str:
    """Remove implementation narration while preserving the actual academic result."""
    body, _summary = split_summary(answer or "")
    body = re.sub(
        r"(?im)^\s*(?:öncelikle\s+)?(?:mezuniyet durumunuzu?|mezuniyet durumunu)"
        r".*?(?:kullanacağım|hesaplayacağım|kontrol edelim)\.?\s*$",
        "",
        body,
    )
    body = re.sub(
        r"(?im)^\s*(?:kalan kredi hesabını\s+)?latex.*?(?:açıklamak|göstermek).*?:?\s*$",
        "",
        body,
    )
    body = re.sub(r"(?is)\n*\s*(?:kaynaklar|sources)\s*:\s*(?:\n.*)*$", "", body)
    body = re.sub(r"(?im)^\s*\[Source:[^\n]*\]\s*$", "", body)
    return re.sub(r"\n{3,}", "\n\n", body).strip()


def audit_summary(audit: dict[str, Any], *, language: str = "tr") -> str:
    completed = audit.get("completed_su_credits", 0)
    required = audit.get("total_min_su_credits", 0)
    remaining = audit.get("remaining_su_credits", 0)
    complete = audit.get("status") == "complete"
    if language == "en":
        status = "You currently meet the graduation requirements." if complete else "You do not yet meet all graduation requirements."
        return f"You have completed {completed} of {required} SU credits; {remaining} SU credits remain. {status}"
    status = "Mezuniyet koşullarını şu anda tamamlıyorsun." if complete else "Mezuniyet koşullarının tamamı henüz karşılanmadı."
    return f"Gerekli {required} SU kredisinin {completed} SU’sunu tamamladın; {remaining} SU kredisi kaldı. {status}"


def audit_answer(audit: dict[str, Any], *, language: str = "tr") -> tuple[str, str]:
    """Render the authoritative audit without exposing storage/retrieval internals."""
    if audit.get("reliability") == "unavailable":
        message = str(audit.get("message") or "Bu müfredat için güvenilir hesaplama yapılamıyor.")
        return ensure_summary_section(message, language=language)
    if language == "en":
        headline = (
            f"**Graduation status:** {audit.get('completed_su_credits', 0)}/"
            f"{audit.get('total_min_su_credits', 0)} SU completed; "
            f"{audit.get('remaining_su_credits', 0)} SU remaining."
        )
    else:
        headline = (
            f"**Mezuniyet durumu:** {audit.get('completed_su_credits', 0)}/"
            f"{audit.get('total_min_su_credits', 0)} SU tamamlandı; "
            f"{audit.get('remaining_su_credits', 0)} SU kaldı."
        )
    summary = audit_summary(audit, language=language)
    heading = "## Kısa Özet" if language == "tr" else "## Short Summary"
    return f"{headline}\n\n{heading}\n\n{summary}", summary


def graduation_plan_answer(audit: dict[str, Any], *, next_term: bool, language: str = "tr") -> tuple[str, str]:
    missing = list(audit.get("missing_required_courses") or [])
    categories = [
        item for item in audit.get("categories") or []
        if (item.get("remaining_su_credits") or 0) > 0
    ]
    first_courses = missing[:5]
    if language == "en":
        intro = "For next term, prioritize these missing required courses:" if next_term else "You still need to complete these required courses:"
        lines = [intro]
        lines.extend(f"- **{code}**" for code in first_courses)
        if not first_courses:
            lines.append("- No individually missing required course was found; focus on the remaining elective categories.")
        if categories:
            lines.append("\nRemaining category credits: " + "; ".join(
                f"{_category_label(str(c.get('category')), language)}: {c.get('remaining_su_credits')} SU"
                for c in categories
            ))
    else:
        intro = "Gelecek dönem önceliğin şu eksik zorunlu dersler olmalı:" if next_term else "Öncelikle şu eksik zorunlu dersleri tamamlamalısın:"
        lines = [intro]
        lines.extend(f"- **{code}**" for code in first_courses)
        if not first_courses:
            lines.append("- Tekil eksik zorunlu ders görünmüyor; kalan seçmeli kategorilerine odaklanmalısın.")
        if categories:
            lines.append("\nKalan kategori kredileri: " + "; ".join(
                f"{_category_label(str(c.get('category')), language)}: {c.get('remaining_su_credits')} SU"
                for c in categories
            ))
        if next_term:
            lines.append("\nDerslerin açılma ve önkoşul durumuna göre bu listeyi dönem programına dönüştürebilirim.")
    summary = audit_summary(audit, language=language)
    heading = "## Kısa Özet" if language == "tr" else "## Short Summary"
    return "\n".join(lines) + f"\n\n{heading}\n\n{summary}", summary


def _category_label(value: str, language: str) -> str:
    labels = {
        "university_courses": ("Üniversite dersleri", "University courses"),
        "required_courses": ("Zorunlu dersler", "Required courses"),
        "core_electives": ("Çekirdek seçmeliler", "Core electives"),
        "area_electives": ("Alan seçmelileri", "Area electives"),
        "free_electives": ("Serbest seçmeliler", "Free electives"),
        "engineering": ("Mühendislik ECTS", "Engineering ECTS"),
        "basic_science": ("Temel bilim ECTS", "Basic-science ECTS"),
    }
    tr, en = labels.get(value, (value.replace("_", " ").title(), value.replace("_", " ").title()))
    return tr if language == "tr" else en


def audit_structured_content(audit: dict[str, Any] | None, *, language: str = "tr") -> dict[str, Any] | None:
    if not audit or audit.get("reliability") == "unavailable":
        return None

    category_rows = []
    for category in audit.get("categories") or []:
        category_rows.append(
            {
                "category": _category_label(str(category.get("category", "")), language),
                "completed": category.get("completed_su_credits"),
                "required": category.get("required_su_credits"),
                "remaining": category.get("remaining_su_credits"),
                "courses": ", ".join(category.get("completed_courses") or []),
            }
        )

    ects_rows = [
        {
            "category": _category_label(str(item.get("category", "")), language),
            "completed": item.get("completed_ects"),
            "required": item.get("required_ects"),
            "remaining": item.get("remaining_ects"),
        }
        for item in audit.get("ects_requirements") or []
    ]

    tables: list[dict[str, Any]] = []
    if category_rows:
        tables.append(
            {
                "id": "degree-audit-su",
                "title": "Mezuniyet Kategori Dağılımı" if language == "tr" else "Degree Audit by Category",
                "columns": [
                    {"key": "category", "label": "Kategori" if language == "tr" else "Category"},
                    {"key": "completed", "label": "Tamamlanan" if language == "tr" else "Completed"},
                    {"key": "required", "label": "Gerekli" if language == "tr" else "Required"},
                    {"key": "remaining", "label": "Kalan" if language == "tr" else "Remaining"},
                    {"key": "courses", "label": "Sayılan dersler" if language == "tr" else "Counted courses"},
                ],
                "rows": category_rows,
                "exportable": True,
            }
        )
    if ects_rows:
        tables.append(
            {
                "id": "degree-audit-ects",
                "title": "ECTS Koşulları" if language == "tr" else "ECTS Requirements",
                "columns": [
                    {"key": "category", "label": "Kategori" if language == "tr" else "Category"},
                    {"key": "completed", "label": "Tamamlanan" if language == "tr" else "Completed"},
                    {"key": "required", "label": "Gerekli" if language == "tr" else "Required"},
                    {"key": "remaining", "label": "Kalan" if language == "tr" else "Remaining"},
                ],
                "rows": ects_rows,
                "exportable": True,
            }
        )

    return {
        "kind": "degree_audit",
        "status": audit.get("status"),
        "headline": {
            "completed_su_credits": audit.get("completed_su_credits"),
            "total_min_su_credits": audit.get("total_min_su_credits"),
            "remaining_su_credits": audit.get("remaining_su_credits"),
        },
        "tables": tables,
        "missing_required_courses": audit.get("missing_required_courses") or [],
        "warnings": audit.get("warnings") or [],
    }


def course_update_structured_content(
    updates: Iterable[dict[str, Any]] | None,
    *,
    language: str = "tr",
) -> dict[str, Any] | None:
    rows = list(updates or [])
    if not rows:
        return None
    return {
        "kind": "course_history_update",
        "tables": [
            {
                "id": "course-history-update",
                "title": "Ders Geçmişi Güncellemesi" if language == "tr" else "Course History Update",
                "columns": [
                    {"key": "code", "label": "Ders" if language == "tr" else "Course"},
                    {"key": "title", "label": "Ders adı" if language == "tr" else "Title"},
                    {"key": "status", "label": "Durum" if language == "tr" else "Status"},
                    {"key": "action", "label": "İşlem" if language == "tr" else "Action"},
                ],
                "rows": rows,
                "exportable": True,
            }
        ],
    }


def merge_structured_content(*items: dict[str, Any] | None) -> dict[str, Any] | None:
    present = [item for item in items if item]
    if not present:
        return None
    if len(present) == 1:
        return present[0]
    tables = [table for item in present for table in item.get("tables", [])]
    return {"kind": "composite", "tables": tables, "sections": present}
