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
import re
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


_TRAILING_FACULTY_CODE_RE = re.compile(r"\s*\([A-Z]{2,6}\)\s*$")
_PREVIOUS_NAME_RE = re.compile(r"\s*\(Previous Name:\s*(?P<name>[^)]+)\)\s*")
_LEADING_PROGRAMS_OF_RE = re.compile(r"^Programs?\s+of\s+", re.IGNORECASE)


def _normalize_program_name(name: str) -> str:
    """'Computer Science and Engineering (FENS)' / 'Computer Science and Engineering
    Undergraduate Program' / 'Programs of Management (SBS)' -> a comparable lowercase form,
    stripping faculty-code parentheticals, the registry's fixed 'Undergraduate Program' suffix,
    and a transcript's 'Programs of ' prefix (used only for a handful of school-scoped majors,
    e.g. Management)."""
    cleaned = _TRAILING_FACULTY_CODE_RE.sub("", name or "")
    cleaned = _LEADING_PROGRAMS_OF_RE.sub("", cleaned)
    cleaned = re.sub(r"\s+Undergraduate Program\s*$", "", cleaned, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", cleaned).strip().lower()


@lru_cache(maxsize=1)
def _program_name_aliases() -> dict[str, str]:
    """normalized program name (current AND any '(Previous Name: ...)' alias) -> program code.

    Built from the registry's own ``program_name`` field, so it can never drift from the
    curriculum data itself and never needs a hand-maintained list.
    """
    aliases: dict[str, str] = {}
    for record in majors():
        code = (record.get("program") or "").strip().upper()
        raw_name = record.get("program_name") or ""
        if not code or not raw_name:
            continue
        previous = _PREVIOUS_NAME_RE.search(raw_name)
        current = _PREVIOUS_NAME_RE.sub("", raw_name)
        aliases.setdefault(_normalize_program_name(current), code)
        if previous:
            aliases.setdefault(_normalize_program_name(previous.group("name")), code)
    return aliases


def resolve_program_code(name_hint: str | None) -> str | None:
    """Map a free-text program name (e.g. from a transcript's "Program : X" line) to one of our
    known program codes, or None when it cannot be resolved with confidence.

    Fail-closed by design (roadmap section 1's "missing critical data causes an explicit
    abstention, never a guess"): an unresolved or ambiguous hint returns None rather than a
    best-effort code, since acting on a wrong major would silently corrupt a student's profile
    and every downstream degree-audit/recommendation computation.
    """
    if not name_hint or not name_hint.strip():
        return None
    normalized = _normalize_program_name(name_hint)
    if not normalized:
        return None
    aliases = _program_name_aliases()
    exact = aliases.get(normalized)
    if exact:
        return exact
    # Fail-closed substring fallback: only accept it when exactly one known program's normalized
    # name contains (or is contained by) the hint -- ambiguity between two programs must abstain.
    candidates = {
        code for known_name, code in aliases.items()
        if known_name in normalized or normalized in known_name
    }
    return next(iter(candidates)) if len(candidates) == 1 else None
