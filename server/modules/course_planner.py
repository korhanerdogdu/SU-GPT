from __future__ import annotations

"""
Deterministic, student-specific academic-stage planner (recommendation support).

This is the "deterministic-first" half of course recommendation. It does NOT let the LLM
invent the plan: it computes, in code, from the student's completed courses + their program:

  * canonical course identity (display_code + official name) from data/course_catalog,
  * first-year "University Course" debt (fall vs spring), which is mandatory and comes first,
  * the student's effective academic stage (freshman / sophomore / junior / senior),
  * the still-missing program sophomore-foundation courses whose prerequisites are met,
  * a verified candidate pool, ordered by academic priority, with a hard rule that a sophomore
    never gets a 4XX course in the CURRENT plan (advanced interest courses become future targets).

The LLM then only *selects among and explains* these verified candidates — it may not add a
course, rename one, or promote a future target into the current semester. Names are always
rendered from the canonical catalog, never written by the model. Instructor / timetable data is
deliberately out of scope and never produced here.
"""

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from modules.config import DEGREE_DATA_DIR


# ---------------------------------------------------------------------------------------------
# Canonical course identity (names come from here, never from the LLM)
# ---------------------------------------------------------------------------------------------

def normalize_code(code: str) -> str:
    """'cs 201' / 'CS201' / 'CIP 101N' -> 'CS201' / 'CIP101N' (no space, upper)."""
    return re.sub(r"\s+", "", (code or "")).upper()


def display_code(code: str) -> str:
    cleaned = normalize_code(code)
    m = re.match(r"^([A-Z]+)(\d.*)$", cleaned)
    return f"{m.group(1)} {m.group(2)}" if m else cleaned


def course_level(code: str) -> int:
    """CS 445 -> 400, CS 201 -> 200, IF 100 -> 100. 0 when no number is found."""
    m = re.search(r"(\d)\d{2}", normalize_code(code))
    return int(m.group(1)) * 100 if m else 0


@lru_cache(maxsize=2)
def _catalog(data_dir: str) -> dict[str, dict]:
    """normalized_code -> canonical record from course_catalog/current.jsonl."""
    path = Path(data_dir) / "course_catalog" / "current.jsonl"
    out: dict[str, dict] = {}
    if not path.exists():
        return out
    for line in path.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        code = normalize_code(row.get("course_id", ""))
        if not code:
            continue
        out[code] = {
            "course_id": code,
            "display_code": display_code(code),
            "official_name": row.get("title") or "",
            "su_credits": row.get("su_credits"),
            "ects": row.get("ects"),
            "faculty": row.get("faculty"),
            "level": course_level(code),
        }
    return out


def resolve(code: str, data_dir: str | None = None) -> dict | None:
    """Canonical record for a code, or None if it does not resolve (COURSE_IDENTITY_UNRESOLVED)."""
    return _catalog(str(data_dir or DEGREE_DATA_DIR)).get(normalize_code(code))


def official_name(code: str, data_dir: str | None = None) -> str:
    rec = resolve(code, data_dir)
    return rec["official_name"] if rec else ""


# ---------------------------------------------------------------------------------------------
# First-year University Courses (mandatory foundations, same for every program)
# ---------------------------------------------------------------------------------------------
# semester 1 = fall, semester 2 = spring. su carried so the debt can be summed.
UNIVERSITY_SEM1 = ["MATH101", "NS101", "CIP101N", "HIST191", "SPS101", "TLL101", "IF100"]
UNIVERSITY_SEM2 = ["MATH102", "NS102", "AL102", "HIST192", "SPS102", "TLL102"]
UNIVERSITY_FRESHMAN = UNIVERSITY_SEM1 + UNIVERSITY_SEM2
# University courses taken later, after their own prerequisites (not first-year):
UNIVERSITY_LATER = ["PROJ201", "SPS303"]  # + one/two HUM "Major Works" (handled via HUM pool)
HUM_MAJOR_WORKS = ["HUM201", "HUM202", "HUM207"]  # a representative eligible set

# Key prerequisites for the courses this planner reasons about. Only what is needed for
# eligibility of University Courses + sophomore foundations (+ the CS chain). Not the whole
# catalog — the deterministic audit remains the authority for graduation arithmetic.
PREREQS: dict[str, list[str]] = {
    "MATH102": ["MATH101"], "NS102": ["NS101"], "HIST192": ["HIST191"], "SPS102": ["SPS101"],
    "TLL102": ["TLL101"],
    "PHYS113": ["NS101", "MATH101"],
    "CS201": ["IF100"], "CS204": ["CS201"], "CS300": ["CS204"], "CS301": ["CS300", "MATH204"],
    "CS306": ["CS204"], "CS308": ["CS204"],
    "MATH212": ["MATH102"], "MATH203": ["MATH102"], "MATH306": ["MATH203"],
    "DSA210": ["MATH203", "IF100"], "DSA201": ["IF100"],
    "ENS208": ["IF100", "MATH102"], "IE311": ["ENS208", "MATH201"],
    "ENS203": ["MATH102"], "ENS204": ["MATH102", "NS101"], "ENS206": ["MATH102"],
    "ENS211": ["MATH101"], "ENS201": ["MATH102", "NS101"], "ENS202": ["NS102"],
    "EE202": ["ENS203"], "EE200": ["ENS203"],
    "BIO301": ["NS201"], "NS216": ["NS201"], "MAT204": ["MATH101", "NS101"], "NS218": ["ENS202"],
    "ECON202": [], "ECON201": [], "ECON204": [],
    "MAT206": ["ENS202", "ENS205"],
}


@dataclass(frozen=True)
class PlanCourse:
    code: str            # normalized, e.g. CS204
    role: str            # PROGRAMME_REQUIRED | CORE | UNIVERSITY | FOUNDATION
    note: str = ""


# Program sophomore-foundation courses (2XX required / core), per major. These are the courses
# the recommended-program plans place in the sophomore year; missing ones are prioritized (after
# University-course debt) once their prerequisites are satisfied.
PROGRAM_SOPHOMORE: dict[str, tuple[PlanCourse, ...]] = {
    "CS": (
        PlanCourse("CS201", "PROGRAMME_REQUIRED"), PlanCourse("CS204", "PROGRAMME_REQUIRED"),
        PlanCourse("MATH203", "PROGRAMME_REQUIRED"), PlanCourse("MATH204", "PROGRAMME_REQUIRED"),
        PlanCourse("MATH212", "PROGRAMME_REQUIRED"), PlanCourse("PHYS113", "PROGRAMME_REQUIRED"),
        PlanCourse("DSA210", "CORE"),
    ),
    "IE": (
        PlanCourse("ENS208", "PROGRAMME_REQUIRED"), PlanCourse("DSA201", "PROGRAMME_REQUIRED"),
        PlanCourse("MATH212", "PROGRAMME_REQUIRED"), PlanCourse("MATH203", "PROGRAMME_REQUIRED"),
        PlanCourse("MATH306", "PROGRAMME_REQUIRED"), PlanCourse("PHYS113", "PROGRAMME_REQUIRED"),
        PlanCourse("IE311", "PROGRAMME_REQUIRED"),
    ),
    "ME": (
        PlanCourse("CS201", "PROGRAMME_REQUIRED"), PlanCourse("ENS203", "PROGRAMME_REQUIRED"),
        PlanCourse("ENS204", "PROGRAMME_REQUIRED"), PlanCourse("MATH212", "PROGRAMME_REQUIRED"),
        PlanCourse("ENS206", "PROGRAMME_REQUIRED"), PlanCourse("ENS214", "PROGRAMME_REQUIRED"),
    ),
    "EE": (
        PlanCourse("ENS203", "PROGRAMME_REQUIRED"), PlanCourse("ENS211", "PROGRAMME_REQUIRED"),
        PlanCourse("EE202", "PROGRAMME_REQUIRED"), PlanCourse("EE200", "PROGRAMME_REQUIRED"),
        PlanCourse("MATH212", "PROGRAMME_REQUIRED"), PlanCourse("MATH203", "PROGRAMME_REQUIRED"),
        PlanCourse("MATH204", "PROGRAMME_REQUIRED"), PlanCourse("ENS201", "CORE"),
        PlanCourse("CS201", "CORE"),
    ),
    "BIO": (
        PlanCourse("CHEM212", "PROGRAMME_REQUIRED"), PlanCourse("ENS210", "PROGRAMME_REQUIRED"),
        PlanCourse("NS201", "PROGRAMME_REQUIRED"), PlanCourse("BIO301", "PROGRAMME_REQUIRED"),
        PlanCourse("NS216", "PROGRAMME_REQUIRED"), PlanCourse("NS207", "PROGRAMME_REQUIRED"),
        PlanCourse("MATH201", "PROGRAMME_REQUIRED"), PlanCourse("MATH203", "PROGRAMME_REQUIRED"),
    ),
    "DSA": (
        PlanCourse("DSA201", "PROGRAMME_REQUIRED"), PlanCourse("DSA210", "PROGRAMME_REQUIRED"),
        PlanCourse("MATH201", "PROGRAMME_REQUIRED"), PlanCourse("MATH203", "PROGRAMME_REQUIRED"),
        PlanCourse("MATH306", "PROGRAMME_REQUIRED"),
    ),
    "PSY": (
        PlanCourse("PSY201", "PROGRAMME_REQUIRED"), PlanCourse("PSY202", "PROGRAMME_REQUIRED"),
    ),
    "ECON": (
        PlanCourse("ECON201", "PROGRAMME_REQUIRED"), PlanCourse("ECON204", "PROGRAMME_REQUIRED"),
        PlanCourse("ECON202", "PROGRAMME_REQUIRED"), PlanCourse("MATH203", "PROGRAMME_REQUIRED"),
        PlanCourse("MATH306", "PROGRAMME_REQUIRED"),
    ),
    "MAT": (
        PlanCourse("ENS205", "PROGRAMME_REQUIRED"), PlanCourse("ENS202", "PROGRAMME_REQUIRED"),
        PlanCourse("CHEM212", "PROGRAMME_REQUIRED"), PlanCourse("MAT204", "PROGRAMME_REQUIRED"),
        PlanCourse("NS218", "PROGRAMME_REQUIRED"), PlanCourse("MATH212", "PROGRAMME_REQUIRED"),
        PlanCourse("PHYS113", "PROGRAMME_REQUIRED"), PlanCourse("MATH203", "CORE"),
        PlanCourse("MAT206", "CORE"),
    ),
}

# A short, program-specific freshman-additional recommendation (Section 13). CS-oriented
# freshmen with IF 100 done are steered to CS 201 first; other engineering programs similar.
FRESHMAN_EXTRA: dict[str, str] = {
    "CS": "CS201", "ME": "CS201", "EE": "CS201", "IE": "DSA201", "DSA": "DSA201",
}


@dataclass
class StageAnalysis:
    program: str
    stage: str
    completed: set[str]
    freshman_done: int
    missing_university_sem1: list[str] = field(default_factory=list)
    missing_university_sem2: list[str] = field(default_factory=list)
    missing_university_later: list[str] = field(default_factory=list)
    eligible_foundations: list[PlanCourse] = field(default_factory=list)
    blocked_foundations: list[PlanCourse] = field(default_factory=list)

    @property
    def freshman_complete(self) -> bool:
        return not (self.missing_university_sem1 or self.missing_university_sem2)


def _prereqs_met(code: str, completed: set[str]) -> bool:
    return all(p in completed for p in PREREQS.get(code, []))


def analyze(program: str, completed_codes: list[str]) -> StageAnalysis:
    program = (program or "").strip().upper()
    completed = {normalize_code(c) for c in completed_codes if c}
    freshman_done = sum(1 for c in UNIVERSITY_FRESHMAN if c in completed)

    missing_sem1 = [c for c in UNIVERSITY_SEM1 if c not in completed]
    missing_sem2 = [c for c in UNIVERSITY_SEM2 if c not in completed]
    missing_later = [c for c in UNIVERSITY_LATER if c not in completed]
    if not any(h in completed for h in HUM_MAJOR_WORKS):
        missing_later.append("HUM2XX")  # a Major Works course is still owed

    foundations = PROGRAM_SOPHOMORE.get(program, ())
    eligible, blocked = [], []
    for pc in foundations:
        if pc.code in completed:
            continue
        (eligible if _prereqs_met(pc.code, completed) else blocked).append(pc)

    foundations_done = sum(1 for pc in foundations if pc.code in completed)
    if freshman_done < 10:
        stage = "freshman_foundation"
    elif foundations and foundations_done < max(1, len(foundations) // 2):
        stage = "sophomore_foundation"
    elif any(course_level(c) >= 400 for c in completed):
        stage = "senior_completion"
    elif any(course_level(c) >= 300 for c in completed):
        stage = "junior_progression"
    else:
        stage = "sophomore_foundation"

    return StageAnalysis(
        program=program, stage=stage, completed=completed, freshman_done=freshman_done,
        missing_university_sem1=missing_sem1, missing_university_sem2=missing_sem2,
        missing_university_later=missing_later,
        eligible_foundations=eligible, blocked_foundations=blocked,
    )


def _fmt(code: str) -> str:
    """'CS204' -> 'CS 204 — Advanced Programming' (or just the display code if unresolved)."""
    rec = resolve(code)
    if not rec:
        return display_code(code)
    return f"{rec['display_code']} — {rec['official_name']}"


def build_context(program: str, completed_codes: list[str], interest_codes: list[str] | None = None,
                  data_dir: str | None = None) -> str:
    """Authoritative planning context for the LLM: canonical names + an ordered candidate pool
    + the hard rules. The LLM selects among and explains these; it never invents a course."""
    a = analyze(program, completed_codes)
    sophomore_or_below = a.stage in {"freshman_foundation", "sophomore_foundation"}

    lines: list[str] = [
        "[Source: Deterministic academic-stage planner (authoritative)]",
        f"Student program: {program or 'unknown'}. Effective academic stage: {a.stage}.",
        f"First-year University Courses completed: {a.freshman_done}/{len(UNIVERSITY_FRESHMAN)}.",
        "",
        "HARD RULES (obey exactly):",
        "- Recommend ONLY courses listed in the CANDIDATE POOL below. Do not add any other course.",
        "- Render every course as its exact 'CODE — Official Name' from this block. Never rewrite a "
        "course name, never translate the official title, never invent a name.",
        "- Do NOT output any instructor, day, time, room or section information.",
        "- Priority order: missing University Courses first, then missing 2XX required foundations, "
        "then eligible core/area/free electives aligned with interest.",
    ]
    if sophomore_or_below:
        lines.append(
            "- The student is at sophomore level or below: do NOT put any 4XX course in the current "
            "plan. Advanced interest courses go under 'FUTURE TARGETS' only."
        )

    # 1) University-course debt
    uni_missing = a.missing_university_sem1 + a.missing_university_sem2
    if uni_missing:
        lines.append("")
        lines.append("MANDATORY UNIVERSITY COURSES STILL MISSING (place these first):")
        if a.missing_university_sem1:
            lines.append("  Fall (1st-semester sequence): " + "; ".join(_fmt(c) for c in a.missing_university_sem1))
        if a.missing_university_sem2:
            lines.append("  Spring (2nd-semester sequence): " + "; ".join(_fmt(c) for c in a.missing_university_sem2))
    if a.missing_university_later:
        later = [c for c in a.missing_university_later if c != "HUM2XX"]
        lines.append("  Other University obligations still owed: "
                     + "; ".join(_fmt(c) for c in later)
                     + ("; one HUM 'Major Works' course (e.g. " + ", ".join(_fmt(c) for c in HUM_MAJOR_WORKS) + ")"
                        if "HUM2XX" in a.missing_university_later else ""))

    # 2) Verified candidate pool
    lines.append("")
    lines.append("CANDIDATE POOL (verified: not yet completed, prerequisites satisfied):")
    pool: list[str] = []
    for c in uni_missing:
        pool.append(f"  - {_fmt(c)}  [UNIVERSITY, mandatory]")
    for pc in a.eligible_foundations:
        pool.append(f"  - {_fmt(pc.code)}  [{pc.role}, {course_level(pc.code)}-level foundation]")

    # freshman single-extra suggestion
    if not a.freshman_complete:
        extra = FRESHMAN_EXTRA.get(program)
        if extra and extra not in a.completed and _prereqs_met(extra, a.completed):
            pool.append(f"  - {_fmt(extra)}  [optional single extra for this program if a slot remains]")

    # 3) interest-aligned electives, split by level for the stage
    future: list[str] = []
    for code in (interest_codes or []):
        n = normalize_code(code)
        if n in a.completed or resolve(n) is None:
            continue
        lvl = course_level(n)
        if sophomore_or_below and lvl >= 400:
            future.append(f"  - {_fmt(n)}  [advanced; prerequisites needed first]")
        elif _prereqs_met(n, a.completed):
            pool.append(f"  - {_fmt(n)}  [interest-aligned elective]")
        else:
            future.append(f"  - {_fmt(n)}  [interest-aligned but prerequisites not yet met]")

    lines.extend(pool if pool else ["  (no verified candidates — the student may have completed the foundations)"])

    if future:
        lines.append("")
        lines.append("FUTURE TARGETS (do NOT put in the current-semester plan; mention as later goals):")
        lines.extend(future)

    if a.blocked_foundations:
        lines.append("")
        lines.append("Foundations still blocked by prerequisites (explain the prereq path, do not "
                     "place now): " + "; ".join(_fmt(pc.code) for pc in a.blocked_foundations))

    return "\n".join(lines)


def missing_university_courses(program: str, completed_codes: list[str]) -> list[str]:
    """Canonical 'CODE — Name' list of still-missing first-year University Courses (for the
    'kalan üniversite derslerim neler?' follow-up)."""
    a = analyze(program, completed_codes)
    return [_fmt(c) for c in (a.missing_university_sem1 + a.missing_university_sem2)]
