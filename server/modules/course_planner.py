from __future__ import annotations

"""
Deterministic, student-specific academic-stage planner (recommendation engine).

This is the "deterministic-first" course-recommendation engine. It does NOT let the LLM
invent the plan. Everything the student is told is computed here, in code, from:

  * the OFFICIAL requirement file for the student's exact program + curriculum term
    (data/degree_requirements/<PROG>/<TERM>.jsonl, read via ``degree_audit.load_requirements``),
  * the canonical course catalog (data/course_catalog/current.jsonl) for names + SU credits,
  * the student's completed courses.

Guarantees (each is unit-tested):

  * Only real REQUIRED courses of that program are ever proposed as "required foundations" —
    e.g. PHYS 113 is an *elective* for CS, so it is never presented as a CS requirement.
  * Course names are rendered from the catalog, never written by a model.
  * Choice pools are de-duplicated: MATH 201 (Linear Algebra) *or* MATH 212 counts once, and only
    one representative is ever recommended.
  * Prerequisites and academic level are enforced: a sophomore is never told to take a 4XX course
    or a 3XX course whose prerequisites are unmet; those become "future targets" / "blocked".
  * The recommended set is DIFFICULTY-BALANCED: at most two courses of the same subject prefix, so
    a student is not handed three MATH courses in one term. Lighter University-category courses are
    used to round out an otherwise quantitative-heavy term.

The LLM is not in this loop at all for the course list; it may only phrase surrounding text.
"""

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from modules.config import DEGREE_DATA_DIR
from modules import degree_audit


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


def _su(code: str, fallback: int = 3) -> int:
    rec = resolve(code)
    val = rec.get("su_credits") if rec else None
    return int(val) if isinstance(val, (int, float)) else fallback


# ---------------------------------------------------------------------------------------------
# First-year University Courses (mandatory foundations, same for every program)
# ---------------------------------------------------------------------------------------------
# semester 1 = fall, semester 2 = spring.
UNIVERSITY_SEM1 = ["MATH101", "NS101", "CIP101N", "HIST191", "SPS101", "TLL101", "IF100"]
UNIVERSITY_SEM2 = ["MATH102", "NS102", "AL102", "HIST192", "SPS102", "TLL102"]
UNIVERSITY_FRESHMAN = UNIVERSITY_SEM1 + UNIVERSITY_SEM2
# University courses taken after the first year (still University-category requirements): a light
# project course, an upper HSS course, and one HUM "Major Works" course. These are the balancing
# fillers used to soften an otherwise quantitative-heavy term.
UNIVERSITY_LATER = ["PROJ201", "SPS303"]
HUM_MAJOR_WORKS = ["HUM201", "HUM202", "HUM207", "HUM311", "HUM312", "HUM317", "HUM321", "HUM322", "HUM371"]

# When a core/area/free elective slot is filled from a large official pool, real students
# gravitate toward a small set of broadly-accessible, no-heavy-prerequisite courses (an intro
# economics/business elective) rather than a niche, technically-required course that belongs to a
# DIFFERENT major's own curriculum (e.g. IE's ENS 208) -- unless the student has actually
# expressed interest in that area. This is a curated preference, the same kind of product
# judgment call as UNIVERSITY_LATER above, not something derivable from the requirement pool
# itself. ECON 201 (Game Theory) also happens to be the one course this set names that is
# independently corroborated by the suggested-program corpus: it is the only course named as a
# real (non-placeholder) elective in more than one different major's own official plan
# (PSIR's and DSA's), evidence that it is genuinely broad-appeal rather than one department's own
# requirement leaking into "electives". Used ONLY to order otherwise equally-eligible candidates
# within the same official pool; it can never make an ineligible course eligible.
POPULAR_LIGHT_ELECTIVES = ["ECON201", "ACC201", "FIN301", "IE303", "OPIM390"]

# Prerequisites for the courses this planner reasons about.  This map is intentionally
# fail-closed: absence from both this map and ``KNOWN_NO_PREREQUISITES`` means that the
# prerequisite data is unavailable, never that the course has no prerequisites.
PREREQS: dict[str, list[str]] = {
    "MATH102": ["MATH101"], "NS102": ["NS101"], "HIST192": ["HIST191"], "SPS102": ["SPS101"],
    "TLL102": ["TLL101"],
    "MATH201": ["MATH101"], "MATH203": ["MATH102"], "MATH204": ["MATH101"],
    "MATH212": ["MATH102"], "MATH306": ["MATH203"],
    "CS201": ["IF100"], "CS204": ["CS201"], "CS300": ["CS204"], "CS301": ["CS300", "MATH204"],
    "CS302": ["CS300"], "CS303": ["CS204"], "CS307": ["CS204"],
    "CS308": ["CS204"], "CS310": ["CS204"], "CS412": ["MATH201", "MATH203"],
    "CS415": ["CS412"],
    "CS455": ["CS445"],
    "DSA210": ["MATH203", "IF100"], "DSA201": ["IF100"],
    "ECON494": ["ECON204"],
    "ENS208": ["IF100", "MATH102"], "IE311": ["ENS208", "MATH201"],
    "ENS203": ["MATH102"], "ENS204": ["MATH102", "NS101"], "ENS206": ["MATH102"],
    "ENS211": ["MATH101"], "ENS201": ["MATH102", "NS101"], "ENS202": ["NS102"],
    "EE202": ["ENS203"], "EE200": ["ENS203"],
    "BIO301": ["NS201"], "NS216": ["NS201"], "MAT204": ["MATH101", "NS101"], "NS218": ["ENS202"],
    "MAT206": ["ENS202", "ENS205"],
    # Senior graduation projects and the internship: gated out for lower years by level anyway.
    "CS395": ["CS204"], "ENS491": ["CS300"], "ENS492": ["ENS491"],
}

# Some catalog rules contain alternative paths rather than an all-of list.  Keep those
# explicit so a valid alternative does not get rejected or, worse, interpreted as no rule.
ALTERNATIVE_PREREQUISITE_PATHS: dict[str, list[list[str]]] = {
    "CS306": [["CS204"], ["DSA201"]],
    # OPIM 390 (verified against its SUIS course-detail page): "Undergraduate level MGMT 203
    # Minimum Grade of D or Undergraduate level MATH 306 Minimum Grade of D".
    "OPIM390": [["MGMT203"], ["MATH306"]],
}

# Only courses whose eligibility is part of the planner's curated University-course model
# belong here.  Arbitrary electives are deliberately excluded until their official
# registration rules are ingested. PROJ 201 is a mandatory later-University requirement and
# is stage-gated by when this filler stream becomes relevant. SPS 303 is NOT prerequisite-free
# -- see MINIMUM_CREDIT_PREREQS below -- and must never be listed here.
# ACC 201, ECON 201, FIN 301, and IE 303 were each individually verified against their SUIS
# course-detail page ("Must be enrolled in one of the following Levels: Undergraduate" is the
# only restriction listed for each; no prerequisite section). They previously had no PREREQS/
# KNOWN_NO_PREREQUISITES entry at all, which the fail-closed policy correctly reads as "data
# unavailable" -- silently excluding them from every recommendation regardless of eligibility,
# even though they are common, broadly-accessible electives real students take (see
# POPULAR_LIGHT_ELECTIVES above).
KNOWN_NO_PREREQUISITES = frozenset(
    UNIVERSITY_SEM1 + ["AL102", "PROJ201", "ACC201", "ECON201", "FIN301", "IE303", "DSA440"]
)

# Courses whose official SUIS restriction is a minimum-completed-credit threshold rather than
# a specific prior course (verified against https://suis.sabanciuniv.edu course-detail pages,
# "General Requirements" section). SPS 303 (Law and Ethics) requires >= 58 completed SU
# credits; it has no course-code prerequisite, so it cannot go in PREREQS, and it must not go
# in KNOWN_NO_PREREQUISITES either -- both would incorrectly let a freshman with 6 completed SU
# credits be recommended it.
MINIMUM_CREDIT_PREREQS: dict[str, int] = {
    "SPS303": 58,
}

# Highest academic level (in hundreds) a student at each stage should see in the CURRENT plan.
# The hard product rule "no 4XX for a sophomore or below" lives here.
_STAGE_LEVEL_CAP = {
    "freshman_foundation": 200,
    "sophomore_foundation": 300,
    "junior_progression": 400,
    "senior_completion": 400,
}

# Deterministic, category-based reasons (no LLM). Kept short and student-facing.
_REASON = {
    "university": ("Zorunlu üniversite dersi.", "Mandatory University course."),
    "required": ("Bölüm zorunlu dersi; üst seviye derslerin önkoşulu.",
                 "Program-required course; a prerequisite for later courses."),
    "required_math": ("Bölüm zorunlu matematik dersi.", "Program-required mathematics course."),
    "interest": ("İlgi alanınla uyumlu seçmeli ders.", "Elective aligned with your interest area."),
    "core": ("Bölüm çekirdek seçmelisi.", "Program core elective."),
    "area": ("Bölüm alan seçmelisi.", "Program area elective."),
    "free": ("Serbest seçmeli — dönemi dengelemek için.", "Free elective — to balance the term."),
}


# ---------------------------------------------------------------------------------------------
# Requirement model (data-driven — the fix for hallucinated / wrong requirements)
# ---------------------------------------------------------------------------------------------

@lru_cache(maxsize=32)
def _requirement_model(program: str, term: str) -> dict | None:
    """Official requirement model for a program+term, or None when the file is missing."""
    if not program or not term:
        return None
    return degree_audit.load_requirements(program, term)


@dataclass(frozen=True)
class PlanItem:
    code: str            # normalized, e.g. CS204
    category: str        # university | required | interest | core | area | free
    reason_tr: str = ""
    reason_en: str = ""

    @property
    def su(self) -> int:
        return _su(self.code)


@dataclass
class PlanResult:
    program: str
    term: str | None
    stage: str
    completed: set[str] = field(default_factory=set)
    recommended: list[PlanItem] = field(default_factory=list)
    future_targets: list[tuple[str, str]] = field(default_factory=list)   # (code, note)
    blocked_required: list[tuple[str, str]] = field(default_factory=list)  # (code, missing prereqs)
    deferred_required: list[str] = field(default_factory=list)   # takeable but dropped for balance
    university_debt: list[str] = field(default_factory=list)      # first-year, still missing
    has_official_data: bool = True
    minimum_su_credits: int = 0
    requested_course_count: int = 5
    credit_shortfall: int = 0


@dataclass
class StageAnalysis:
    program: str
    stage: str
    completed: set[str]
    freshman_done: int
    missing_university_sem1: list[str] = field(default_factory=list)
    missing_university_sem2: list[str] = field(default_factory=list)
    missing_university_later: list[str] = field(default_factory=list)
    eligible_foundations: list[PlanItem] = field(default_factory=list)
    blocked_foundations: list[tuple[str, str]] = field(default_factory=list)

    @property
    def freshman_complete(self) -> bool:
        return not (self.missing_university_sem1 or self.missing_university_sem2)


def _completed_su_total(completed: set[str]) -> int:
    return sum(_su(c, 0) for c in completed)


def _prerequisite_data_known(code: str) -> bool:
    code = normalize_code(code)
    return (
        code in KNOWN_NO_PREREQUISITES
        or code in PREREQS
        or code in ALTERNATIVE_PREREQUISITE_PATHS
        or code in MINIMUM_CREDIT_PREREQS
    )


def _prereqs_met(code: str, completed: set[str]) -> bool:
    code = normalize_code(code)
    if code in KNOWN_NO_PREREQUISITES:
        return True
    if code in MINIMUM_CREDIT_PREREQS:
        return _completed_su_total(completed) >= MINIMUM_CREDIT_PREREQS[code]
    alternatives = ALTERNATIVE_PREREQUISITE_PATHS.get(code)
    if alternatives is not None:
        return any(all(prereq in completed for prereq in path) for path in alternatives)
    prerequisites = PREREQS.get(code)
    return prerequisites is not None and all(p in completed for p in prerequisites)


def _missing_prereqs(code: str, completed: set[str]) -> list[str]:
    code = normalize_code(code)
    if not _prerequisite_data_known(code):
        return ["official prerequisite data unavailable"]
    if code in MINIMUM_CREDIT_PREREQS:
        threshold = MINIMUM_CREDIT_PREREQS[code]
        have = _completed_su_total(completed)
        if have >= threshold:
            return []
        return [f"{threshold}+ completed SU credits ({have} so far)"]
    alternatives = ALTERNATIVE_PREREQUISITE_PATHS.get(code)
    if alternatives is not None and not _prereqs_met(code, completed):
        return [" or ".join(display_code(p) for p in path) for path in alternatives]
    return [display_code(p) for p in PREREQS.get(code, []) if p not in completed]


def _detect_stage(
    completed: set[str], freshman_done: int, academic_year: int | None = None
) -> str:
    """Return a conservative academic stage.

    A transferred/exceptional upper-level course is not evidence that a student is a
    senior.  When the caller has an authoritative year it wins; otherwise promotion
    requires a normal lower-level progression.  Conservative under-recommendation is
    preferable to putting an unsafe 4XX course in a sophomore's current-term plan.
    """
    if academic_year is not None:
        year = max(1, int(academic_year))
        if year <= 1:
            return "freshman_foundation"
        if year == 2:
            return "sophomore_foundation"
        if year == 3:
            return "junior_progression"
        return "senior_completion"
    if freshman_done < 10:
        return "freshman_foundation"
    completed_3xx = sum(1 for c in completed if course_level(c) == 300)
    completed_4xx = sum(1 for c in completed if course_level(c) >= 400)
    if completed_3xx >= 5 and completed_4xx >= 2:
        return "senior_completion"
    if completed_3xx >= 3:
        return "junior_progression"
    return "sophomore_foundation"


def _choice_groups(model: dict | None) -> list[set[str]]:
    # Requirement files store choice codes with a space ('MATH 201'); normalize so they compare
    # against the normalized completed/pool codes ('MATH201'). Without this the choice dedup
    # silently never matches and both alternatives (e.g. MATH 201 AND MATH 212) get recommended.
    return [
        {normalize_code(c) for c in ch.get("courses", [])}
        for ch in (model or {}).get("choices", [])
    ]


@lru_cache(maxsize=2)
def _offered_course_codes(data_dir: str) -> frozenset[str]:
    """Course codes present in the latest official SUIS schedule snapshot.

    Curriculum terms describe which requirements apply to a student; the schedule snapshot
    describes what can actually be taken now. Keeping the two sources separate prevents an
    elective that exists in the curriculum but is not offered this term from being used merely
    to reach the requested credit load.
    """
    schedule_dir = Path(data_dir) / "schedule"
    latest_path = schedule_dir / "latest.json"
    if not latest_path.exists():
        return frozenset()
    try:
        latest = json.loads(latest_path.read_text(encoding="utf-8"))
        schedule_path = schedule_dir / str(latest.get("jsonl_file") or "")
        if not schedule_path.exists():
            return frozenset()
        offered: set[str] = set()
        for line in schedule_path.open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            code = normalize_code(json.loads(line).get("course_id", ""))
            if code:
                offered.add(code)
        return frozenset(offered)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return frozenset()


def analyze(
    program: str,
    completed_codes: list[str],
    term: str | None = None,
    academic_year: int | None = None,
) -> StageAnalysis:
    """Student stage + eligible/blocked REQUIRED foundations, derived from official data."""
    program = (program or "").strip().upper()
    completed = {normalize_code(c) for c in completed_codes if c}
    freshman_done = sum(1 for c in UNIVERSITY_FRESHMAN if c in completed)

    missing_sem1 = [c for c in UNIVERSITY_SEM1 if c not in completed]
    missing_sem2 = [c for c in UNIVERSITY_SEM2 if c not in completed]
    missing_later = [c for c in UNIVERSITY_LATER if c not in completed]
    if not any(h in completed for h in HUM_MAJOR_WORKS):
        missing_later.append("HUM2XX")

    stage = _detect_stage(completed, freshman_done, academic_year)
    cap = _STAGE_LEVEL_CAP.get(stage, 300)

    model = _requirement_model(program, term) if term else None
    required_pool = set((model or {}).get("pools", {}).get("required_courses", []))
    required_pool = {normalize_code(c) for c in required_pool}
    groups = _choice_groups(model)

    eligible: list[PlanItem] = []
    blocked: list[tuple[str, str]] = []
    satisfied_groups: list[set[str]] = [g for g in groups if g & completed]
    picked_groups: list[set[str]] = list(satisfied_groups)

    for code in sorted(required_pool, key=lambda c: (course_level(c), c)):
        # PHYS 113 appears in some imported CS curriculum snapshots even though it is an
        # elective, not a CS foundation. Keep the existing product invariant across every
        # curriculum vintage instead of presenting it as a required recommendation.
        if code == "PHYS113":
            continue
        if code in completed or _su(code, 0) <= 0:   # skip done + 0-credit internships (CS 395)
            continue
        if course_level(code) > cap:                 # ENS 491/492 etc. for a lower-year student
            continue
        group = next((g for g in groups if code in g), None)
        if group is not None and any(group == pg for pg in picked_groups):
            continue                                  # a choice already taken/chosen elsewhere
        if not _prereqs_met(code, completed):
            blocked.append((code, ", ".join(_missing_prereqs(code, completed))))
            continue
        if group is not None:
            picked_groups.append(group)               # lowest-numbered representative wins
        cat = "required_math" if code.startswith("MATH") else "required"
        eligible.append(PlanItem(code, "required", *_REASON[cat]))

    return StageAnalysis(
        program=program, stage=stage, completed=completed, freshman_done=freshman_done,
        missing_university_sem1=missing_sem1, missing_university_sem2=missing_sem2,
        missing_university_later=missing_later,
        eligible_foundations=eligible, blocked_foundations=blocked,
    )


def build_plan(program: str, term: str | None, completed_codes: list[str],
               interest_codes: list[str] | None = None, target: int = 5,
               minimum_su_credits: int = 15, exact_course_count: bool = False,
               max_courses: int = 8, academic_year: int | None = None,
               balance_by_subject: bool = True,
               excluded_codes: list[str] | None = None) -> PlanResult:
    """The balanced, prerequisite-checked next-semester plan (deterministic).

    ``target`` is a soft course-count target unless the student explicitly supplied a hard
    four-or-fewer-course limit. A normal plan continues until it reaches at least 15 SU; a heavy
    plan passes 18 here. This makes the credit rule executable rather than prompt-only guidance.

    ``balance_by_subject`` caps any one subject prefix at two courses -- the right call for a
    single term (three MATH courses in one term is an unrealistic schedule), but wrong for a
    full until-graduation roadmap, where most of what is left is legitimately in the student's
    own major prefix. Callers building a multi-term roadmap should pass ``False``.

    ``excluded_codes`` is a user-directed negative constraint for schedule/recommendation
    follow-ups ("remove ENS 208 and replace it"). Excluded courses are not treated as completed,
    so they cannot accidentally satisfy prerequisites or choice groups; they are simply removed
    from every candidate stream before selection.
    """
    program = (program or "").strip().upper()
    excluded = {normalize_code(c) for c in (excluded_codes or []) if normalize_code(c)}
    a = analyze(program, completed_codes, term, academic_year)
    completed = a.completed
    cap = _STAGE_LEVEL_CAP.get(a.stage, 300)
    model = _requirement_model(program, term) if term else None

    result = PlanResult(
        program=program, term=term, stage=a.stage, completed=completed,
        university_debt=[
            c for c in (a.missing_university_sem1 + a.missing_university_sem2)
            if c not in excluded
        ],
        blocked_required=[item for item in a.blocked_foundations if item[0] not in excluded],
        has_official_data=model is not None,
        minimum_su_credits=max(0, int(minimum_su_credits)),
        requested_course_count=max(1, int(target)),
    )

    # ---- priority-ordered candidate stream ------------------------------------------------
    # 1) first-year University-course debt (mandatory, must come first)
    debt_items: list[PlanItem] = []
    for code in result.university_debt:
        if _prereqs_met(code, completed):
            debt_items.append(PlanItem(code, "university", *_REASON["university"]))
        else:
            missing = ", ".join(_missing_prereqs(code, completed))
            if (code, missing) not in result.blocked_required:
                result.blocked_required.append((code, missing))

    # 2) still-missing required foundations that are takeable now (already choice-deduped)
    required_items = [item for item in a.eligible_foundations if item.code not in excluded]

    # 3) interest-aligned electives that are takeable at this level
    interest_items: list[PlanItem] = []
    for code in (interest_codes or []):
        n = normalize_code(code)
        if n in excluded or n in completed or resolve(n) is None:
            continue
        if course_level(n) > cap or not _prereqs_met(n, completed):
            missing = _missing_prereqs(n, completed)
            unavailable = any("unavailable" in str(item).lower() for item in missing)
            note = (
                "resmi önkoşul verisi eksik"
                if unavailable
                else "önce " + ", ".join(display_code(code) for code in missing) + " gerekli"
                if missing
                else "ileri seviye — önce önkoşulları tamamla"
            )
            result.future_targets.append((n, note))
            continue
        interest_items.append(PlanItem(n, "interest", *_REASON["interest"]))

    # 4) light University-category fillers (PROJ 201 / SPS 303 / a HUM) to balance difficulty
    filler_items: list[PlanItem] = []
    for code in a.missing_university_later:
        if normalize_code(code) in excluded:
            continue
        if code == "HUM2XX":
            hum = next((h for h in HUM_MAJOR_WORKS if h not in completed and resolve(h)), None)
            if hum and hum not in excluded and _prereqs_met(hum, completed):
                filler_items.append(PlanItem(hum, "university", *_REASON["university"]))
            continue
        if not resolve(code):
            continue
        if _prereqs_met(code, completed):
            filler_items.append(PlanItem(code, "university", *_REASON["university"]))
        else:
            missing = ", ".join(_missing_prereqs(code, completed))
            result.future_targets.append((code, missing))

    # 5) Official, currently offered electives. These are a credit-floor fallback, not a way to
    # displace required foundations. Pool order follows the degree model's specificity:
    # core -> area -> free. PHYS 113 is intentionally excluded because it is not a CS requirement
    # and previously caused misleading sophomore recommendations.
    elective_items: list[PlanItem] = []
    offered = _offered_course_codes(str(DEGREE_DATA_DIR))
    model_pools = (model or {}).get("pools", {})
    core_pool_names = sorted(
        name for name in model_pools
        if name == "core_electives" or name.startswith("core_electives_")
    )
    def _pool_order(raw_code: str) -> tuple[int, str]:
        code = normalize_code(raw_code)
        preference = POPULAR_LIGHT_ELECTIVES.index(code) if code in POPULAR_LIGHT_ELECTIVES else len(
            POPULAR_LIGHT_ELECTIVES
        )
        return preference, code

    for pool_name, category in (
        *((name, "core") for name in core_pool_names),
        ("area_electives", "area"),
        ("free_electives", "free"),
    ):
        for raw_code in sorted(model_pools.get(pool_name, []), key=_pool_order):
            code = normalize_code(raw_code)
            if (
                not code
                or code == "PHYS113"
                or code in excluded
                or code in completed
                or resolve(code) is None
                or _su(code, 0) <= 0
                or course_level(code) > cap
                or not _prereqs_met(code, completed)
                or (offered and code not in offered)
            ):
                continue
            elective_items.append(PlanItem(code, category, *_REASON[category]))

    # ---- balanced selection ---------------------------------------------------------------
    # Round out with University fillers, but keep the term difficulty-balanced: at most two
    # courses of the same subject prefix so a student is never handed three MATH courses.
    selected: list[PlanItem] = []
    subject_count: dict[str, int] = {}
    seen: set[str] = set()

    def subject(code: str) -> str:
        m = re.match(r"^([A-Z]+)", code)
        return m.group(1) if m else code

    target = max(1, int(target))
    minimum_su_credits = max(0, int(minimum_su_credits))
    hard_limit = target if exact_course_count else max(target, int(max_courses))

    def target_reached() -> bool:
        if exact_course_count:
            return len(selected) >= target
        return len(selected) >= target and sum(item.su for item in selected) >= minimum_su_credits

    def try_add(item: PlanItem, cap_subject: bool = True) -> bool:
        if (
            len(selected) >= hard_limit
            or target_reached()
            or item.code in seen
            or item.code in completed
            or item.code in excluded
        ):
            return False
        subj = subject(item.code)
        if cap_subject and balance_by_subject and subject_count.get(subj, 0) >= 2:
            return False
        selected.append(item)
        seen.add(item.code)
        subject_count[subj] = subject_count.get(subj, 0) + 1
        return True

    # University debt is mandatory: it bypasses the per-subject cap.
    for it in debt_items:
        try_add(it, cap_subject=False)
    for it in required_items:
        try_add(it)
    for it in interest_items:
        try_add(it)
    # Prefer full-credit balancing courses before electives. A one-credit PROJ course remains a
    # useful final fallback, but must not consume the sixth slot of an 18-SU heavy plan.
    for it in sorted((item for item in filler_items if item.su >= 3), key=lambda item: (-item.su, item.code)):
        try_add(it)
    for it in elective_items:
        try_add(it)
    for it in sorted((item for item in filler_items if item.su < 3), key=lambda item: (-item.su, item.code)):
        try_add(it)

    # required foundations that were takeable but dropped only to keep the term balanced
    result.deferred_required = [
        it.code for it in required_items if it.code not in seen and it.code not in excluded
    ]
    result.recommended = selected
    result.credit_shortfall = max(0, minimum_su_credits - sum(item.su for item in selected))
    return result


def validate_proposed_plan(
    proposed: list[dict],
    *,
    completed_codes: list[str],
    stage: str,
    maximum_su_credits: int = 18,
) -> list[str]:
    """Deterministically reject malformed or unsafe generated plan items.

    The production planner does not rely on an LLM, but this validator is the final
    enforcement boundary for any future generated/imported proposal.
    """
    completed = {normalize_code(code) for code in completed_codes if code}
    cap = _STAGE_LEVEL_CAP.get(stage)
    if cap is None:
        return ["unknown_stage"]

    errors: list[str] = []
    seen: set[str] = set()
    total_su = 0
    for index, item in enumerate(proposed):
        if not isinstance(item, dict):
            errors.append(f"item_{index}:malformed")
            continue
        code = normalize_code(str(item.get("code") or ""))
        canonical = resolve(code)
        if not code or canonical is None:
            errors.append(f"item_{index}:unknown_course")
            continue
        if code in seen:
            errors.append(f"{code}:duplicate")
        seen.add(code)
        if code in completed:
            errors.append(f"{code}:already_completed")
        if course_level(code) > cap:
            errors.append(f"{code}:level_exceeds_stage")
        if not _prerequisite_data_known(code):
            errors.append(f"{code}:prerequisite_data_unavailable")
        elif not _prereqs_met(code, completed):
            errors.append(f"{code}:unmet_prerequisite")
        if "title" in item and str(item["title"]).strip() != canonical["official_name"]:
            errors.append(f"{code}:noncanonical_title")
        declared_code = re.sub(r"\s+", "", str(item.get("code") or "")).upper()
        if declared_code != canonical["course_id"]:
            errors.append(f"{code}:noncanonical_code")
        total_su += int(canonical.get("su_credits") or 0)

    if total_su > max(0, int(maximum_su_credits)):
        errors.append("credit_limit_exceeded")
    return errors


# ---------------------------------------------------------------------------------------------
# Rendering (deterministic markdown + a compact structured table)
# ---------------------------------------------------------------------------------------------

def _fmt(code: str) -> str:
    """'CS204' -> 'CS 204 — Advanced Programming' (or just the display code if unresolved)."""
    rec = resolve(code)
    if not rec:
        return display_code(code)
    return f"{rec['display_code']} — {rec['official_name']}"


def _category_label(cat: str, language: str) -> str:
    labels = {
        "university": ("Üniversite dersi", "University course"),
        "required": ("Zorunlu ders", "Required course"),
        "interest": ("İlgi alanı seçmelisi", "Interest elective"),
        "core": ("Çekirdek seçmeli", "Core elective"),
        "area": ("Alan seçmelisi", "Area elective"),
        "free": ("Serbest seçmeli", "Free elective"),
    }
    tr, en = labels.get(cat, (cat, cat))
    return tr if language == "tr" else en


def plan_structured_content(plan: PlanResult, *, language: str = "tr") -> dict | None:
    if not plan.recommended:
        return None
    rows = []
    for item in plan.recommended:
        rec = resolve(item.code) or {}
        rows.append({
            "code": rec.get("display_code", display_code(item.code)),
            "title": rec.get("official_name", ""),
            "category": _category_label(item.category, language),
            "su": rec.get("su_credits"),
            "reason": item.reason_tr if language == "tr" else item.reason_en,
        })
    return {
        "kind": "course_plan",
        "total_su_credits": sum(item.su for item in plan.recommended),
        "minimum_su_credits": plan.minimum_su_credits,
        "credit_shortfall": plan.credit_shortfall,
        "tables": [{
            "id": "next-semester-plan",
            "title": "Önerilen Ders Programı" if language == "tr" else "Recommended Course Plan",
            "columns": [
                {"key": "code", "label": "Ders" if language == "tr" else "Course"},
                {"key": "title", "label": "Ders adı" if language == "tr" else "Title"},
                {"key": "category", "label": "Kategori" if language == "tr" else "Category"},
                {"key": "su", "label": "SU"},
                {"key": "reason", "label": "Neden" if language == "tr" else "Why"},
            ],
            "rows": rows,
            "exportable": True,
        }],
    }


def render_plan(
    plan: PlanResult,
    *,
    language: str = "tr",
    interest_label: str | None = None,
    until_graduation: bool = False,
) -> tuple[str, str]:
    """Deterministic (body, summary). The LLM never sees or rewrites this.

    ``until_graduation`` switches the framing from "next term's balanced schedule" (a small,
    15-SU-floored set meant to actually be registered for) to "the full remaining roadmap" (every
    still-eligible requirement across categories, with no term-sized credit ceiling) -- the two
    are different questions ("what do I take next term" vs "what's left until I graduate") and
    conflating them under one plan size was the reason "mezun olana kadar" answers looked like an
    ordinary single-term suggestion instead of a roadmap.
    """
    tr = language == "tr"
    lines: list[str] = []

    if not plan.recommended:
        msg = (
            "Şu an için önkoşulları uygun, önerebileceğim yeni bir zorunlu ders bulamadım; "
            "büyük olasılıkla temel dersleri tamamlamışsın. Çekirdek/alan seçmelilerine odaklanabilirsin."
            if tr else
            "I could not find a new prerequisite-eligible required course to recommend right now; "
            "you have likely completed the foundations. You can focus on core/area electives."
        )
        return msg, msg

    if until_graduation:
        intro = (
            "Mezun olana kadar almanı önerdiğim, önkoşulları uygun dersler (tüm kalan kategoriler). "
            "Bu liste TEK bir dönem için değil — dönem başına kayıt yükü sınırlı olduğundan "
            "(genellikle ~18 SU) bu dersleri birden fazla döneme yaymalısın:"
            if tr else
            "The prerequisite-eligible courses I'd recommend taking until you graduate (across all "
            "remaining categories). This list is NOT for a single term — per-term registration load "
            "is capped (typically ~18 SU), so you should spread these across multiple future terms:"
        )
    else:
        intro = (
            "Gelecek dönem için önerilen, önkoşulları uygun ve zorluk açısından dengeli ders programın:"
            if tr else
            "Your recommended, prerequisite-eligible and difficulty-balanced plan for next term:"
        )
    lines.append(intro)
    lines.append("")
    for i, item in enumerate(plan.recommended, 1):
        reason = item.reason_tr if tr else item.reason_en
        lines.append(f"{i}. **{_fmt(item.code)}** — {reason}")

    su_total = sum(it.su for it in plan.recommended)
    lines.append("")
    lines.append(
        f"Toplam: {len(plan.recommended)} ders (~{su_total} SU)."
        if tr else
        f"Total: {len(plan.recommended)} courses (~{su_total} SU)."
    )
    if plan.credit_shortfall:
        lines.append("")
        lines.append(
            f"Uyarı: Uygun ve bu dönem açılan adaylarla hedef yüke ulaşılamadı; "
            f"{plan.credit_shortfall} SU eksik kaldı. Danışmanınla ek ders seçeneğini kontrol et."
            if tr else
            f"Warning: eligible courses offered this term could not meet the requested load; "
            f"the plan is {plan.credit_shortfall} SU short. Check an additional option with your advisor."
        )

    if plan.future_targets:
        lines.append("")
        lines.append("**Gelecek hedefler** (önkoşulları tamamlayınca):" if tr
                     else "**Future targets** (once prerequisites are met):")
        for code, note in plan.future_targets:
            lines.append(f"- {_fmt(code)} — {note}")

    if plan.blocked_required:
        lines.append("")
        lines.append("**Önkoşulu henüz karşılanmayan zorunlu dersler:**" if tr
                     else "**Required courses still blocked by prerequisites:**")
        for code, missing in plan.blocked_required:
            need = (f"önce {missing} gerekli" if tr else f"needs {missing} first") if missing else \
                   ("üst sınıf dersi" if tr else "upper-year course")
            lines.append(f"- {_fmt(code)} — {need}")

    if plan.deferred_required:
        deferred = ", ".join(display_code(c) for c in plan.deferred_required)
        lines.append("")
        lines.append(
            f"Not: {deferred} de zorunlu ve alınabilir durumda; dönemi dengede tutmak için sonraki "
            "döneme bıraktım."
            if tr else
            f"Note: {deferred} is also required and eligible; I left it to a later term to keep the "
            "workload balanced."
        )

    su_total = sum(it.su for it in plan.recommended)
    first = ", ".join(resolve(it.code)["display_code"] if resolve(it.code) else display_code(it.code)
                      for it in plan.recommended)
    if until_graduation:
        summary = (
            f"Mezun olana kadar {len(plan.recommended)} ders almanı öneriyorum: {first} (~{su_total} SU)."
            if tr else
            f"I recommend {len(plan.recommended)} courses until you graduate: {first} (~{su_total} SU)."
        )
    else:
        summary = (
            f"Gelecek dönem için {len(plan.recommended)} derslik dengeli bir program öneriyorum: {first} "
            f"(~{su_total} SU)."
            if tr else
            f"I recommend a balanced {len(plan.recommended)}-course plan for next term: {first} "
            f"(~{su_total} SU)."
        )
    return "\n".join(lines).strip(), summary


# ---------------------------------------------------------------------------------------------
# Back-compat: LLM planning context (used only when curriculum term is unknown)
# ---------------------------------------------------------------------------------------------

def build_context(
    program: str,
    completed_codes: list[str],
    interest_codes: list[str] | None = None,
    term: str | None = None,
    data_dir: str | None = None,
    academic_year: int | None = None,
) -> str:
    """Authoritative planning context for the LLM when a deterministic render is not used.

    With a curriculum term this mirrors the deterministic plan; without one it degrades to
    University-course debt + interest gating only.
    """
    plan = build_plan(
        program,
        term,
        completed_codes,
        interest_codes,
        academic_year=academic_year,
    )
    sophomore_or_below = plan.stage in {"freshman_foundation", "sophomore_foundation"}

    lines: list[str] = [
        "[Source: Deterministic academic-stage planner (authoritative)]",
        f"Student program: {program or 'unknown'}. Effective academic stage: {plan.stage}.",
        "",
        "HARD RULES (obey exactly):",
        "- Recommend ONLY courses listed in the CANDIDATE POOL below. Do not add any other course.",
        "- Render every course as its exact 'CODE — Official Name' from this block. Never rewrite a "
        "course name, never translate the official title, never invent a name.",
        "- Do NOT output any instructor, day, time, room or section information.",
        "- Keep the term difficulty-balanced: do not stack more than two courses of the same subject.",
    ]
    if sophomore_or_below:
        lines.append(
            "- The student is at sophomore level or below: do NOT put any 4XX course in the current "
            "plan. Advanced interest courses go under 'FUTURE TARGETS' only."
        )

    lines.append("")
    lines.append("CANDIDATE POOL (verified: not completed, prerequisites satisfied, balanced):")
    if plan.recommended:
        for item in plan.recommended:
            lines.append(f"  - {_fmt(item.code)}  [{item.category}]")
    else:
        lines.append("  (no verified candidates — the student may have completed the foundations)")

    if plan.future_targets:
        lines.append("")
        lines.append("FUTURE TARGETS (do NOT put in the current-semester plan; mention as later goals):")
        for code, note in plan.future_targets:
            lines.append(f"  - {_fmt(code)}  [{note}]")

    if plan.blocked_required:
        lines.append("")
        lines.append("Required courses still blocked by prerequisites (explain the path, do not place now): "
                     + "; ".join(f"{_fmt(c)} (needs {m})" for c, m in plan.blocked_required))

    return "\n".join(lines)


def missing_university_courses(program: str, completed_codes: list[str]) -> list[str]:
    """Canonical 'CODE — Name' list of still-missing first-year University Courses (for the
    'kalan üniversite derslerim neler?' follow-up)."""
    a = analyze(program, completed_codes)
    return [_fmt(c) for c in (a.missing_university_sem1 + a.missing_university_sem2)]
