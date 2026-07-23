from __future__ import annotations

"""
Build data/course_catalog/current.jsonl — one record per distinct course across the whole
degree-requirement + minor corpus (roadmap section 5.3).

This is the searchable course list the student picks from to build their course history in
MongoDB. It intentionally carries only *intrinsic* course facts (title, SU credits, ECTS,
faculty) plus engineering / basic-science ECTS slots — NOT free/area/core classifications,
because whether a course is "area" or "free" depends on the student's program + admit term
and is answered from the degree-requirement data, not the catalog.

engineering_ects / basic_science_ects are left null: the corpus does not carry per-course
engineering/basic-science ECTS yet (that lives in the SU course catalog and needs a separate
scrape). Populate them later and graduation eng/basic-science audit becomes computable.

Usage: python server/scripts/build_course_catalog.py [--data-dir DIR]
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.config import DEGREE_DATA_DIR


def _split(code: str) -> tuple[str, str]:
    m = re.match(r"^([A-Z]+)\s*(\d.*)$", (code or "").strip().upper())
    return (m.group(1), m.group(2)) if m else (code, "")


def build(data_dir: Path) -> list[dict]:
    catalog: dict[str, dict] = {}
    roots = [data_dir / "degree_requirements", data_dir / "minors"]
    files = sorted(p for root in roots if root.is_dir() for p in root.rglob("*.jsonl"))
    for path in files:
        for line in path.open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not row.get("document_type", "").endswith("_pool_course"):
                continue
            code = (row.get("course_id") or "").strip()
            if not code:
                continue
            existing = catalog.get(code)
            # keep the richest record (prefer one that has a title / higher credits)
            if existing and (existing.get("title") and not row.get("course_title")):
                continue
            subject, number = _split(code)
            catalog[code] = {
                "course_id": code,
                "subject": subject,
                "number": number,
                "title": row.get("course_title") or (existing or {}).get("title") or "",
                "su_credits": row.get("su_credits"),
                "ects": row.get("ects"),
                "faculty": row.get("faculty") or (existing or {}).get("faculty") or "",
                "engineering_ects": None,      # TODO: populate from SU course catalog scrape
                "basic_science_ects": None,    # TODO: populate from SU course catalog scrape
                "data_role": "course_catalog",
                "source_authority": "Sabanci University (derived from degree requirements)",
            }
    return sorted(catalog.values(), key=lambda r: (r["subject"], r["number"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(DEGREE_DATA_DIR))
    args = parser.parse_args()
    data_dir = Path(args.data_dir).expanduser().resolve()

    records = build(data_dir)
    out_dir = data_dir / "course_catalog"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "current.jsonl"
    with out_path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    faculties = {}
    for r in records:
        faculties[r["faculty"] or "?"] = faculties.get(r["faculty"] or "?", 0) + 1
    print(f"Wrote {out_path} ({len(records)} distinct courses)")
    print("By faculty: " + ", ".join(f"{k}={v}" for k, v in sorted(faculties.items())))


if __name__ == "__main__":
    main()
