from __future__ import annotations

"""
Deterministic "what would my GPA become" projections (roadmap section 14 companion).

Two question shapes are recognised and answered in code -- never guessed by the model:

1. Specific per-course hypotheticals: "CS 455'ten B+ alsam, ENS 211'den C alsam ne olur?"
   -> parsed as [(course_code, grade), ...] and projected using each course's real SU credit
   from the course catalog (a course's credit is the same for every section, so the catalog is
   already authoritative -- no need to also consult a saved schedule).

2. A bulk scenario: "6 ders alıp hepsinden A alırsam GPA'm kaça çıkar?"
   -> parsed as (course_count, grade) and projected assuming a 3-SU-per-course default (the same
   fallback course_planner and degree_audit already use for a course with no credit on file).

Anything else (including the plain starter "Bu dönem GPA'mı ne kadar yükseltebilirim?", which
names no specific courses or grades) falls back to a small fixed set of common-grade scenarios
over the student's current term load, so the answer is still concrete rather than a request for
clarification.
"""

import re

from modules.transcript_parser import GRADE_POINTS

DEFAULT_COURSE_SU = 3
DEFAULT_TERM_COURSE_COUNT = 5  # matches the app's own 15-SU-normal-term-load convention

#  Deliberately case-sensitive: real grades are always written uppercase ("A", "B+"), and a
#  case-insensitive single-letter alternation like "D" or "F" would otherwise match stray lower-
#  case letters buried inside ordinary Turkish/English words (e.g. the "d" in "hepsinden", or the
#  "f" in "if"). A trailing `\b` doesn't work here either -- "B+" ends in a non-word character, so
#  there is no word/non-word transition for `\b` to find; a lookahead that just rules out another
#  letter or digit immediately after covers both the plain-letter and +/- grades correctly.
_GRADE_TOKEN = r"(?P<grade>A-|A|B\+|B-|B|C\+|C-|C|D\+|D|F)(?![A-Za-z0-9])"
_CODE_TOKEN = r"[A-Z]{2,6}\s?\d{3,4}[A-Z]?"

# "CS 455'ten B+ alsam" / "CS 455'ten B+ alırsam" -- grade follows the code+suffix in Turkish.
_SPECIFIC_RE = re.compile(
    rf"(?P<code>{_CODE_TOKEN})(?:'?(?:d[ae]n|t[ae]n))?[^.,;\n]{{0,15}}?{_GRADE_TOKEN}",
)
# "I get a B+ in CS 455" -- grade precedes "in/for COURSE" in English. Only "in"/"for" are
# case-insensitive (scoped inline flag); the grade token itself stays case-sensitive so "if"
# can never be misread as an "F" grade.
_SPECIFIC_RE_EN = re.compile(
    rf"{_GRADE_TOKEN}[^.,;\n]{{0,15}}?(?i:in|for)\s+(?P<code>{_CODE_TOKEN})",
)

# "6 ders alıp hepsinden A alırsam" / "6 dersin hepsinden A" / "6 courses all A"
_GRADE_ALT = r"A-|A|B\+|B-|B|C\+|C-|C|D\+|D|F"
_GRADE_BOUNDARY = r"(?![A-Za-z0-9])"  # see _GRADE_TOKEN above for why not `\b`
_BULK_RE = re.compile(
    rf"(?P<count>\d{{1,2}})\s*(?:[Dd]ers|[Cc]ourse)[^.,;\n]{{0,30}}?[Hh]epsi[^.,;\n]{{0,15}}?(?P<grade1>{_GRADE_ALT}){_GRADE_BOUNDARY}"
    rf"|(?P<count2>\d{{1,2}})\s*(?:[Dd]ers|[Cc]ourse)s?[^.,;\n]{{0,20}}?\b[Aa]ll\b[^.,;\n]{{0,10}}?(?P<grade2>{_GRADE_ALT}){_GRADE_BOUNDARY}",
)


def _normalize_code(code: str) -> str:
    return re.sub(r"\s+", " ", code.strip()).upper()


def parse_specific_scenarios(question: str) -> list[tuple[str, str]]:
    """[(course_code, grade), ...] for every recognisable "COURSE'dan GRADE alsam" mention."""
    text = question or ""
    found: list[tuple[str, str]] = []
    for pattern in (_SPECIFIC_RE, _SPECIFIC_RE_EN):
        for m in pattern.finditer(text):
            code = _normalize_code(m.group("code"))
            grade = m.group("grade")
            if grade in GRADE_POINTS:
                found.append((code, grade))
    # de-duplicate while preserving first mention order
    seen: set[tuple[str, str]] = set()
    unique = []
    for pair in found:
        if pair not in seen:
            seen.add(pair)
            unique.append(pair)
    return unique


def parse_bulk_scenario(question: str) -> tuple[int, str] | None:
    """(course_count, grade) for "N ders(imin) hepsinden GRADE alırsam" style questions."""
    m = _BULK_RE.search(question or "")
    if not m:
        return None
    count = m.group("count") or m.group("count2")
    grade = m.group("grade1") or m.group("grade2")
    if not count or not grade:
        return None
    return int(count), grade


def course_su_credits(code: str, catalog_lookup) -> float:
    """SU credit is a per-course catalog property (every section of a course carries the same
    credit), so the catalog is already authoritative -- falling back to the 3-SU convention every
    other deterministic planner in this codebase uses when a code isn't on file at all."""
    record = catalog_lookup(code)
    if record and isinstance(record.get("su_credits"), (int, float)):
        return float(record["su_credits"])
    return float(DEFAULT_COURSE_SU)


def project_gpa(current_points: float, current_credit: float, additions: list[tuple[float, str]]) -> float | None:
    total_points = current_points + sum(credit * GRADE_POINTS[grade] for credit, grade in additions)
    total_credit = current_credit + sum(credit for credit, _ in additions)
    return round(total_points / total_credit, 2) if total_credit else None


def render_projection(
    *,
    current_gpa: float | None,
    current_points: float,
    current_credit: float,
    specific: list[tuple[str, str, float]],  # (code, grade, su_credits)
    bulk: tuple[int, str] | None,
    default_scenarios: bool,
    language: str = "tr",
    display_credit: float | None = None,
) -> str:
    """``current_points``/``current_credit`` may already have an already-completed course's old
    grade pulled back out (a re-grade replaces, it doesn't stack) -- that adjusted pair is what
    the projection math must use, but the student's own "current GPA" line should still show
    their real, unadjusted total. ``display_credit`` lets a caller show the true figure; it
    defaults to ``current_credit`` for callers that never made an adjustment."""
    tr = language == "tr"
    shown_credit = current_credit if display_credit is None else display_credit
    lines: list[str] = []
    current_text = f"{current_gpa:.2f}" if current_gpa is not None else ("yok" if tr else "none yet")
    lines.append(
        f"Şu anki GPA'n: **{current_text}** ({shown_credit:.0f} SU üzerinden)." if tr else
        f"Your current GPA: **{current_text}** (over {shown_credit:.0f} SU)."
    )
    lines.append("")

    if specific:
        additions = [(su, grade) for _, grade, su in specific]
        projected = project_gpa(current_points, current_credit, additions)
        lines.append("**Senaryo:**" if tr else "**Scenario:**")
        lines.append("")
        lines.append(f"| {'Ders' if tr else 'Course'} | {'Not' if tr else 'Grade'} | SU |")
        lines.append("|---|---|---|")
        for code, grade, su in specific:
            lines.append(f"| {code} | {grade} | {su:.0f} |")
        lines.append("")
        lines.append(
            f"Bu notlarla yeni GPA'n: **{projected:.2f}**." if tr else
            f"With these grades, your new GPA would be **{projected:.2f}**."
        )
        return "\n".join(lines)

    if bulk:
        count, grade = bulk
        su_each = DEFAULT_COURSE_SU
        additions = [(su_each, grade)] * count
        projected = project_gpa(current_points, current_credit, additions)
        lines.append(
            f"{count} ders (varsayılan {su_each} SU/ders) alıp hepsinden **{grade}** alırsan:" if tr else
            f"If you take {count} courses (assuming {su_each} SU each) and get **{grade}** in all of them:"
        )
        lines.append("")
        lines.append(f"| {'Durum' if tr else 'Scenario'} | GPA |")
        lines.append("|---|---|")
        lines.append(f"| {'Şu an' if tr else 'Now'} | {current_text} |")
        lines.append(f"| {count} × {grade} | **{projected:.2f}** |")
        return "\n".join(lines)

    # No specific numbers in the question -- give a concrete, useful default spread instead of
    # asking a clarifying question (the recommendation floor's own convention: a normal term is
    # ~15 SU, i.e. DEFAULT_TERM_COURSE_COUNT courses at DEFAULT_COURSE_SU each).
    count = DEFAULT_TERM_COURSE_COUNT
    su_each = DEFAULT_COURSE_SU
    lines.append(
        f"Bu dönem {count} ders (varsayılan {su_each} SU/ders, {count * su_each} SU) alırsan olası senaryolar:"
        if tr else
        f"If you take {count} courses this term (assuming {su_each} SU each, {count * su_each} SU total), here are a few scenarios:"
    )
    lines.append("")
    lines.append(f"| {'Not' if tr else 'Grade'} | {'Yeni GPA' if tr else 'New GPA'} |")
    lines.append("|---|---|")
    # The full A-through-D spread, not just the top grades: a student weighing whether a rough
    # term still helps or hurts needs to see the downside scenarios too, not only the upside.
    for grade in ("A", "A-", "B+", "B", "B-", "C+", "C", "C-", "D+", "D"):
        additions = [(su_each, grade)] * count
        projected = project_gpa(current_points, current_credit, additions)
        lines.append(f"| {'Hepsinden ' + grade if tr else 'All ' + grade} | **{projected:.2f}** |")
    lines.append("")
    lines.append(
        "Belirli dersler ve notlar için \"CS 455'ten B+ alsam ne olur?\" gibi sorabilirsin." if tr else
        'Ask about specific courses and grades, e.g. "What if I get a B+ in CS 455?"'
    )
    return "\n".join(lines)


GPA_PROJECTION_RE = re.compile(
    r"\b("
    r"gpa'?m[ıi]?\s*(ne kadar|kaça|kac[ae])|gpa\s*(hesapla|yükselt|yukselt|projek)|"
    r"notum\s*ne\s*olur|not\s*ortalamam|"
    r"alsam\s*ne\s*olur|alırsam\s*(gpa|not)|"
    r"raise\s*my\s*gpa|what.*gpa|gpa.*(project|scenario|calculat)"
    r")",
    re.IGNORECASE,
)


def is_gpa_projection_query(question: str) -> bool:
    # A recognisable per-course or bulk hypothetical is strong enough evidence on its own (e.g.
    # "What if I get a B+ in CS 455?" never says the word "GPA"); the keyword regex catches the
    # rest, including the plain starter that names no specific courses or grades at all.
    if parse_specific_scenarios(question) or parse_bulk_scenario(question):
        return True
    return bool(GPA_PROJECTION_RE.search(question or ""))
