from __future__ import annotations

"""
Validate the degree-requirement corpus and print a coverage report (roadmap section 16).

Checks each data/degree_requirements/<PROG>/<TERM>.jsonl and data/minors/<CODE>/<TERM>.jsonl:
  - required metadata present (data_role, document_type, program, curriculum_term, chunk_id)
  - pool_course rows carry course_id + numeric su_credits/ects
  - chunk_ids unique within a file
  - category names use the controlled vocabulary
  - a profile record exists
  - every file is represented in data/curricula/registry.jsonl
Then prints a Program x Term coverage table. Exits non-zero if any hard error is found.

Usage: python server/scripts/validate_degree_data.py [--data-dir DIR]
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.config import DEGREE_DATA_DIR

CONTROLLED_CATEGORIES = {
    "university_courses", "university_courses_mandatory", "university_courses_hum_pool",
    "required_courses", "core_electives", "core_electives_political_science",
    "core_electives_international_relations", "area_electives", "free_electives",
    "faculty_courses", "faculty_courses_fens", "faculty_courses_fass", "faculty_courses_sbs",
    "engineering", "basic_science", "philosophy_requirement", "mathematics_requirement",
    "electives",
}
COURSE_ID_RE = re.compile(r"^[A-Z]{2,6}\s\d{3,5}[0-9A-Z]?$")
REQUIRED_KEYS = ("data_role", "document_type", "program", "curriculum_term", "chunk_id")


def _load_registry(data_dir: Path) -> set[tuple[str, str, bool]]:
    path = data_dir / "curricula" / "registry.jsonl"
    entries: set[tuple[str, str, bool]] = set()
    if path.exists():
        for line in path.open(encoding="utf-8"):
            line = line.strip()
            if line:
                r = json.loads(line)
                entries.add((r.get("program"), r.get("curriculum_term"), bool(r.get("is_minor"))))
    return entries


def validate_file(path: Path, is_minor: bool) -> tuple[list[str], dict]:
    errors: list[str] = []
    warnings: list[str] = []
    seen_ids: set[str] = set()
    has_profile = False
    categories: set[str] = set()
    program = term = None
    n = 0
    for i, line in enumerate(path.open(encoding="utf-8"), 1):
        line = line.strip()
        if not line:
            continue
        n += 1
        row = json.loads(line)
        program = program or row.get("program")
        term = term or row.get("curriculum_term")
        for key in REQUIRED_KEYS:
            if not row.get(key):
                errors.append(f"{path.name}:{i} missing {key}")
        cid = row.get("chunk_id")
        if cid in seen_ids:
            errors.append(f"{path.name}:{i} duplicate chunk_id {cid}")
        seen_ids.add(cid)
        dtype = row.get("document_type", "")
        if dtype.endswith("_profile"):
            has_profile = True
        if dtype.endswith("_category_pool") and row.get("requirement_category"):
            categories.add(row["requirement_category"])
        if dtype.endswith("_pool_course"):
            code = row.get("course_id", "")
            if not COURSE_ID_RE.match(code):
                warnings.append(f"{path.name}:{i} unnormalized course_id '{code}'")
            for num_key in ("su_credits", "ects"):
                if not isinstance(row.get(num_key), (int, float)):
                    warnings.append(f"{path.name}:{i} {code} non-numeric {num_key}")
        cat = row.get("requirement_category")
        if cat and cat not in CONTROLLED_CATEGORIES:
            warnings.append(f"{path.name}:{i} uncontrolled category '{cat}'")
    if not has_profile:
        errors.append(f"{path.name} has no *_profile record")
    return errors, {
        "program": program, "term": term, "records": n,
        "categories": sorted(categories), "is_minor": is_minor, "warnings": warnings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(DEGREE_DATA_DIR))
    parser.add_argument("--strict", action="store_true", help="treat warnings as failures too")
    args = parser.parse_args()
    data_dir = Path(args.data_dir).expanduser().resolve()
    registry = _load_registry(data_dir)

    all_errors: list[str] = []
    all_warnings: list[str] = []
    coverage: list[dict] = []

    for is_minor, root in ((False, data_dir / "degree_requirements"), (True, data_dir / "minors")):
        for path in sorted(root.rglob("*.jsonl")) if root.is_dir() else []:
            errs, info = validate_file(path, is_minor)
            all_errors.extend(errs)
            all_warnings.extend(info.pop("warnings"))
            key = (info["program"], info["term"], is_minor)
            info["in_registry"] = key in registry
            if not info["in_registry"]:
                all_warnings.append(f"{info['program']}/{info['term']} not in registry.jsonl")
            coverage.append(info)

    print("=== COVERAGE (majors) ===")
    print(f"{'Program':10} {'Term':8} {'Records':8} {'Registry':9} Categories")
    for c in coverage:
        if c["is_minor"]:
            continue
        cats = ",".join(x.replace("_electives", "").replace("university_courses_", "uni-") for x in c["categories"])
        print(f"{c['program']:10} {c['term']:8} {c['records']:<8} {'yes' if c['in_registry'] else 'NO':9} {cats[:70]}")

    minors = [c for c in coverage if c["is_minor"]]
    print(f"\n=== MINORS === {len(minors)} files, "
          f"{len({c['program'] for c in minors})} minors x {len({c['term'] for c in minors})} terms")

    print(f"\nFiles checked: {len(coverage)}")
    print(f"Hard errors:  {len(all_errors)}")
    print(f"Warnings:     {len(all_warnings)}")
    for e in all_errors[:20]:
        print("  ERROR:", e)
    for w in all_warnings[:10]:
        print("  warn:", w)
    if all_errors or (args.strict and all_warnings):
        sys.exit(1)
    print("\nVALIDATION PASSED.")


if __name__ == "__main__":
    main()
