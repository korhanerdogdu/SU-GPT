from __future__ import annotations

"""
Deterministic degree-audit engine (roadmap section 14).

The audit arithmetic is computed HERE, in normal code, from the exact official
requirement file (data/degree_requirements/<PROGRAM>/<CURRICULUM_TERM>.jsonl) plus the
student's completed courses. The LLM only *explains* the returned object; it never
invents the numbers.

This is an SU-credit category audit (university / required / core / area / free) with
greedy overflow (extra core -> area -> free), choice-pool handling, and missing-required
detection. Engineering / Basic-Science ECTS are reported as "not computed" because the
corpus does not carry per-course engineering/basic-science ECTS yet.
"""

import json
import re
from functools import lru_cache
from pathlib import Path

from modules.config import DEGREE_DATA_DIR


@lru_cache(maxsize=4)
def _course_credit_catalog(data_dir: str) -> dict:
    """course_id -> {engineering_ects, basic_science_ects, su_credits} from course_catalog/current.jsonl."""
    path = Path(data_dir) / "course_catalog" / "current.jsonl"
    catalog: dict[str, dict] = {}
    if path.exists():
        for line in path.open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            catalog[_norm(row.get("course_id", ""))] = {
                "engineering_ects": row.get("engineering_ects"),
                "basic_science_ects": row.get("basic_science_ects"),
                "su_credits": row.get("su_credits"),
            }
    return catalog

# Highest-priority first. A completed course is allocated to the most specific category
# whose pool contains it and still needs credits; leftovers overflow down the elective chain.
_UNIVERSITY_CATS = ("university_courses_mandatory", "university_courses_hum_pool")
_BASE_ELECTIVE_CHAIN = ("core_electives", "area_electives", "free_electives")


def _elective_chain(model: dict) -> tuple[str, ...]:
    """Return every independently enforced core pool before area/free overflow.

    Most programs publish a single ``core_electives`` minimum. PSIR publishes two
    independent minima (Political Science and International Relations); collapsing them
    into one 24-SU bucket would incorrectly let one subject area satisfy the other. Keep
    every ``core_electives_*`` category separate, then apply the documented overflow into
    area and free electives.
    """
    categories = set(model.get("category_min", {})) | set(model.get("pools", {}))
    split_core = sorted(
        category for category in categories if category.startswith("core_electives_")
    )
    core = ["core_electives"] if "core_electives" in categories else []
    return tuple([*core, *split_core, "area_electives", "free_electives"])


def _norm(code: str) -> str:
    """Normalize 'cs455' / 'CS  455' -> 'CS 455'."""
    cleaned = re.sub(r"\s+", "", (code or "")).upper()
    m = re.match(r"^([A-Z]+)(\d.+)$", cleaned)
    return f"{m.group(1)} {m.group(2)}" if m else cleaned


def requirement_path(program: str, curriculum_term: str, data_dir: str | Path | None = None) -> Path:
    root = Path(data_dir or DEGREE_DATA_DIR)
    return root / "degree_requirements" / str(program).strip().upper() / f"{str(curriculum_term).strip()}.jsonl"


def load_requirements(program: str, curriculum_term: str, data_dir: str | Path | None = None) -> dict | None:
    """Parse one official requirement file into a compact model, or None if missing."""
    path = requirement_path(program, curriculum_term, data_dir)
    if not path.exists():
        return None

    profile: dict = {}
    pools: dict[str, list[str]] = {}       # requirement_category -> [course_id]
    su_of: dict[str, int] = {}             # course_id -> su_credits
    title_of: dict[str, str] = {}
    category_min: dict[str, dict] = {}     # category -> {min_su, min_courses}
    choices: list[dict] = []               # {courses:[...], max_count:int}

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            dtype = row.get("document_type", "")
            if dtype == "degree_requirement_profile":
                profile = row
                for c in row.get("categories", []):
                    category_min[c["category"]] = {
                        "min_su": c.get("min_su_credits"),
                        "min_courses": c.get("min_courses"),
                        "min_ects": c.get("min_ects"),
                    }
            elif dtype == "degree_requirement_pool_course":
                cat = row.get("requirement_category", "")
                code = _norm(row.get("course_id", ""))
                pools.setdefault(cat, []).append(code)
                if row.get("su_credits") is not None:
                    su_of[code] = int(row["su_credits"])
                if row.get("course_title"):
                    title_of[code] = row["course_title"]
            elif dtype == "degree_requirement_rule" and row.get("choice_courses"):
                choices.append({
                    "courses": [_norm(c) for c in row["choice_courses"]],
                    "max_count": int(row.get("max_count", 1)),
                })

    return {
        "program": profile.get("program", str(program).upper()),
        "degree_code": profile.get("degree_code"),
        "program_name": profile.get("program_name"),
        "curriculum_term": profile.get("curriculum_term", str(curriculum_term)),
        "total_min_su_credits": profile.get("total_min_su_credits"),
        "total_min_ects": profile.get("total_min_ects"),
        "category_min": category_min,
        "pools": {cat: list(dict.fromkeys(codes)) for cat, codes in pools.items()},
        "su_of": su_of,
        "title_of": title_of,
        "choices": choices,
    }


def audit(program: str, curriculum_term: str, completed_codes: list[str],
          data_dir: str | Path | None = None) -> dict:
    """Return a structured degree-audit object (authoritative arithmetic in code)."""
    model = load_requirements(program, curriculum_term, data_dir)
    if model is None:
        return {
            "status": "unavailable",
            "reliability": "unavailable",
            "program": str(program).upper(),
            "curriculum_term": str(curriculum_term),
            "message": (
                f"Official degree requirement data for {str(program).upper()} curriculum "
                f"{curriculum_term} is not available; a reliable audit cannot be produced."
            ),
        }

    su_of = model["su_of"]
    pools = model["pools"]
    completed = list(dict.fromkeys(_norm(c) for c in completed_codes if c))
    completed_set = set(completed)

    def pool_set(cat: str) -> set[str]:
        return set(pools.get(cat, []))

    university_pool = set().union(*(pool_set(c) for c in _UNIVERSITY_CATS)) if any(
        c in pools for c in _UNIVERSITY_CATS) else set()
    required_pool = pool_set("required_courses")
    elective_chain = _elective_chain(model)

    catalog = _course_credit_catalog(str(data_dir or DEGREE_DATA_DIR))

    def su(code: str) -> int:
        if code in su_of:
            return int(su_of[code])
        # Not in this curriculum term's pool file (e.g. an elective offered outside the scraped
        # pools). Fall back to the general course catalog before guessing, so a course with a
        # real (possibly zero) credit value — like a 0-SU civic-involvement course — never gets
        # counted as a generic 3 SU by default.
        catalog_credits = catalog.get(code, {}).get("su_credits")
        if catalog_credits is not None:
            return int(catalog_credits)
        return 3

    warnings: list[str] = []
    used: set[str] = set()

    # ---- University + Required (specific, must-have categories) --------------------
    def fixed_category(cat: str, pool: set[str]) -> dict:
        min_su = (model["category_min"].get(cat) or {}).get("min_su")
        done = [c for c in completed if c in pool and c not in used]
        for c in done:
            used.add(c)
        completed_su = sum(su(c) for c in done)
        return {
            "category": cat,
            "required_su_credits": min_su,
            "completed_su_credits": completed_su,
            "remaining_su_credits": max((min_su or 0) - completed_su, 0) if min_su is not None else None,
            "completed_courses": done,
        }

    uni = fixed_category("university_courses", university_pool)
    # required: honor choice pools (only one of a choice set is needed)
    req_min = (model["category_min"].get("required_courses") or {}).get("min_su")
    req_done = [c for c in completed if c in required_pool and c not in used]
    for c in req_done:
        used.add(c)
    req_completed_su = sum(su(c) for c in req_done)
    missing_required: list[str] = []
    for code in sorted(required_pool):
        if code in completed_set:
            continue
        # is this code an alternative in a choice the student already satisfied?
        satisfied_by_choice = any(
            code in ch["courses"] and sum(1 for x in ch["courses"] if x in completed_set) >= ch["max_count"]
            for ch in model["choices"]
        )
        if not satisfied_by_choice:
            missing_required.append(code)
    required = {
        "category": "required_courses",
        "required_su_credits": req_min,
        "completed_su_credits": req_completed_su,
        "remaining_su_credits": max((req_min or 0) - req_completed_su, 0) if req_min is not None else None,
        "completed_courses": req_done,
        "missing_required_courses": missing_required,
    }

    # ---- Electives with greedy overflow (core -> area -> free) ---------------------
    elective_result = {}
    for cat in elective_chain:
        elective_result[cat] = {
            "category": cat,
            "required_su_credits": (model["category_min"].get(cat) or {}).get("min_su"),
            "completed_su_credits": 0,
            "completed_courses": [],
        }
    chain_pools = {cat: pool_set(cat) for cat in elective_chain}
    for c in completed:
        if c in used:
            continue
        for cat in elective_chain:
            need = elective_result[cat]["required_su_credits"]
            # Free electives are the true catch-all: any completed course not claimed by a more
            # specific category counts here, not just ones the scraped free-electives pool happens
            # to enumerate. A closed list under-counts real courses (e.g. an overflow course that
            # only appears in the area-electives pool once that pool is already full) and produces
            # exactly the "extra credits with no home" mismatch this fixes.
            eligible = c in chain_pools[cat] or cat == "free_electives"
            if eligible and (need is None or elective_result[cat]["completed_su_credits"] < need):
                elective_result[cat]["completed_su_credits"] += su(c)
                elective_result[cat]["completed_courses"].append(c)
                used.add(c)
                break
    for cat in elective_chain:
        need = elective_result[cat]["required_su_credits"]
        elective_result[cat]["remaining_su_credits"] = (
            max((need or 0) - elective_result[cat]["completed_su_credits"], 0) if need is not None else None
        )

    categories = [uni, required, *(elective_result[c] for c in elective_chain)]
    total_completed = sum(su(c) for c in completed)
    total_min = model["total_min_su_credits"]
    unused = [c for c in completed if c not in used]
    if unused:
        warnings.append("Some completed courses did not map to a requirement category: " + ", ".join(unused))

    # Engineering / Basic-Science ECTS requirements, from the course catalog (roadmap section 14).
    catalog_has_credits = any(v.get("engineering_ects") is not None for v in catalog.values())
    ects_requirements = []
    if catalog_has_credits:
        for cat_key, field in (("engineering", "engineering_ects"), ("basic_science", "basic_science_ects")):
            min_ects = (model["category_min"].get(cat_key) or {}).get("min_ects")
            if min_ects is None:
                continue
            done = sum(catalog.get(c, {}).get(field) or 0 for c in completed)
            ects_requirements.append({
                "category": cat_key,
                "required_ects": min_ects,
                "completed_ects": round(done, 1),
                "remaining_ects": round(max(min_ects - done, 0), 1),
            })

    incomplete = bool(missing_required) or any(
        (cat.get("remaining_su_credits") or 0) > 0 for cat in categories
    ) or any((e.get("remaining_ects") or 0) > 0 for e in ects_requirements)
    return {
        "status": "incomplete" if incomplete else "complete",
        "reliability": "authoritative",
        "program": model["program"],
        "degree_code": model["degree_code"],
        "program_name": model["program_name"],
        "curriculum_term": model["curriculum_term"],
        "total_min_su_credits": total_min,
        "completed_su_credits": total_completed,
        "remaining_su_credits": max((total_min or 0) - total_completed, 0) if total_min is not None else None,
        "categories": categories,
        "ects_requirements": ects_requirements,
        "missing_required_courses": missing_required,
        "engineering_basic_science": "computed" if catalog_has_credits else "not_computed (per-course ECTS not yet scraped)",
        "warnings": warnings,
    }
