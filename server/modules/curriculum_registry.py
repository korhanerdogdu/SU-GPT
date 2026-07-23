from __future__ import annotations

"""
Read-only access to data/curricula/registry.jsonl.

The registry lists every major curriculum (program x degree_code x curriculum_term) and
every minor curriculum that actually exists in the corpus. The backend uses it to (a)
offer valid curriculum choices to the frontend and (b) decide whether an *authoritative*
degree audit is even possible for a student's program/term (missing-data safety, roadmap
sections 15 & 24). No MongoDB dependency, so ingestion/tests can import it freely.
"""

import json
from functools import lru_cache
from pathlib import Path

from modules.config import DEGREE_DATA_DIR

REGISTRY_PATH = Path(DEGREE_DATA_DIR) / "curricula" / "registry.jsonl"


@lru_cache(maxsize=1)
def _load() -> list[dict]:
    if not REGISTRY_PATH.exists():
        return []
    records: list[dict] = []
    with REGISTRY_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def reload() -> None:
    _load.cache_clear()


def all_curricula() -> list[dict]:
    return list(_load())


def majors() -> list[dict]:
    return [r for r in _load() if not r.get("is_minor")]


def minors() -> list[dict]:
    return [r for r in _load() if r.get("is_minor")]


def list_major_programs() -> list[str]:
    return sorted({r["program"] for r in majors() if r.get("program")})


def curricula_for_program(program: str) -> list[dict]:
    program = (program or "").strip().upper()
    return [r for r in majors() if (r.get("program") or "").upper() == program]


def has_major_curriculum(program: str, curriculum_term: str) -> bool:
    program = (program or "").strip().upper()
    curriculum_term = (curriculum_term or "").strip()
    return any(
        (r.get("program") or "").upper() == program
        and (r.get("curriculum_term") or "") == curriculum_term
        for r in majors()
    )


def get_major_curriculum(program: str, curriculum_term: str) -> dict | None:
    program = (program or "").strip().upper()
    curriculum_term = (curriculum_term or "").strip()
    for r in majors():
        if (r.get("program") or "").upper() == program and (r.get("curriculum_term") or "") == curriculum_term:
            return r
    return None


def latest_term_for_program(program: str) -> str | None:
    terms = sorted(r.get("curriculum_term") for r in curricula_for_program(program) if r.get("curriculum_term"))
    return terms[-1] if terms else None
