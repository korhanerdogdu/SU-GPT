from __future__ import annotations

"""
Build data/curricula/registry.jsonl from the degree-requirement + minor corpus.

One record per (program, degree_code, curriculum_term) major curriculum and per
(minor_code, term) minor. The backend uses this to (a) offer valid curriculum choices
in the UI and (b) decide whether an *authoritative* audit is possible for a given
program/term before answering (missing-data safety, roadmap section 15/24).

Usage: python server/scripts/build_curricula_registry.py [--data-dir DIR]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.config import DEGREE_DATA_DIR


def _first_profile(path: Path) -> dict | None:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("document_type", "").endswith("_profile"):
                return row
    return None


def build(data_dir: Path) -> list[dict]:
    records: list[dict] = []

    # Majors: data/degree_requirements/<PROGRAM>/<TERM>.jsonl
    major_root = data_dir / "degree_requirements"
    for path in sorted(major_root.rglob("*.jsonl")) if major_root.is_dir() else []:
        prof = _first_profile(path)
        if not prof:
            continue
        term = prof.get("curriculum_term") or path.stem
        records.append({
            "program": prof.get("program"),
            "degree_code": prof.get("degree_code"),
            "program_name": prof.get("program_name"),
            "program_type": "major",
            "curriculum_term": term,
            "admit_term": prof.get("admit_term", term),
            "admit_term_label": prof.get("admit_term_label", ""),
            "total_min_su_credits": prof.get("total_min_su_credits"),
            "total_min_ects": prof.get("total_min_ects"),
            "is_minor": False,
            "status": "active",
            "source_authority": prof.get("source_authority", "Sabanci University"),
            "source_document": prof.get("source_document") or str(path.relative_to(data_dir)),
        })

    # Minors: data/minors/<CODE>/<TERM>.jsonl
    minor_root = data_dir / "minors"
    for path in sorted(minor_root.rglob("*.jsonl")) if minor_root.is_dir() else []:
        prof = _first_profile(path)
        if not prof:
            continue
        term = prof.get("curriculum_term") or path.stem
        records.append({
            "program": prof.get("program"),
            "degree_code": prof.get("program"),
            "program_name": prof.get("program_name"),
            "program_type": "minor",
            "curriculum_term": term,
            "admit_term": prof.get("admit_term", term),
            "admit_term_label": prof.get("admit_term_label", ""),
            "total_min_su_credits": prof.get("total_min_su_credits"),
            "total_min_ects": prof.get("total_min_ects"),
            "is_minor": True,
            "status": "active",
            "source_authority": prof.get("source_authority", "Sabanci University"),
            "source_document": prof.get("source_document") or str(path.relative_to(data_dir)),
        })

    records.sort(key=lambda r: (r["program_type"], r["program"] or "", r["curriculum_term"] or ""))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(DEGREE_DATA_DIR))
    args = parser.parse_args()
    data_dir = Path(args.data_dir).expanduser().resolve()

    records = build(data_dir)
    out_dir = data_dir / "curricula"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "registry.jsonl"
    with out_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    majors = sum(1 for r in records if not r["is_minor"])
    minors = len(records) - majors
    programs = sorted({r["program"] for r in records if not r["is_minor"]})
    print(f"Wrote {out_path} ({len(records)} records: {majors} major-curricula, {minors} minor-curricula)")
    print(f"Major programs: {', '.join(programs)}")


if __name__ == "__main__":
    main()
