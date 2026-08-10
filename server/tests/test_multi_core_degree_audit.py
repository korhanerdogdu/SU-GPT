from __future__ import annotations

import json
from pathlib import Path

from modules.degree_audit import audit, load_requirements


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_psir_core_pools_keep_independent_minimums(tmp_path: Path) -> None:
    rows = [
        {
            "document_type": "degree_requirement_profile",
            "program": "PSIR",
            "degree_code": "BAPSIR",
            "curriculum_term": "202501",
            "program_name": "Political Science and International Relations",
            "total_min_su_credits": 24,
            "total_min_ects": 48,
            "categories": [
                {"category": "core_electives_political_science", "min_su_credits": 12},
                {"category": "core_electives_international_relations", "min_su_credits": 12},
                {"category": "area_electives", "min_su_credits": 0},
                {"category": "free_electives", "min_su_credits": 0},
            ],
        }
    ]
    for category, codes in {
        "core_electives_political_science": ["POLS 301", "POLS 302", "POLS 303", "POLS 304"],
        "core_electives_international_relations": ["IR 301", "IR 302", "IR 303", "IR 304"],
    }.items():
        for code in codes:
            rows.append(
                {
                    "document_type": "degree_requirement_pool_course",
                    "requirement_category": category,
                    "course_id": code,
                    "course_title": code,
                    "su_credits": 3,
                    "ects": 6,
                }
            )

    _write_jsonl(tmp_path / "degree_requirements" / "PSIR" / "202501.jsonl", rows)
    model = load_requirements("PSIR", "202501", tmp_path)

    assert model is not None
    assert set(model["pools"]) == {
        "core_electives_political_science",
        "core_electives_international_relations",
    }

    result = audit(
        "PSIR",
        "202501",
        ["POLS 301", "POLS 302", "POLS 303", "POLS 304"],
        tmp_path,
    )
    categories = {row["category"]: row for row in result["categories"]}

    assert categories["core_electives_political_science"]["remaining_su_credits"] == 0
    assert categories["core_electives_international_relations"]["remaining_su_credits"] == 12
    assert result["status"] == "incomplete"

