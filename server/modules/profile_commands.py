from __future__ import annotations

"""Deterministic academic-profile updates expressed in chat."""

import re
from dataclasses import dataclass

from modules import curriculum_registry


_MAJOR_PATTERNS = (
    re.compile(r"\b(?:major(?:ım|im|um)?|bölüm(?:üm|um)?|program(?:ım|im)?)\s*(?:=|:|olarak)?\s*([A-Z]{2,5})\b", re.I),
    re.compile(r"\b([A-Z]{2,5})\s+(?:bölümündeyim|bolumundeyim|major(?:ı|i)?yim)\b", re.I),
)
_TERM_PATTERN = re.compile(
    r"\b(?:(?:curriculum|cirriculum)\s*term(?:im)?|müfredat\s*dönem(?:im|i)?|mufredat\s*donem(?:im|i)?|"
    r"giriş\s*dönem(?:im|i)?|giris\s*donem(?:im|i)?)\s*(?:=|:)?\s*(?:fall\s*)?(20\d{2}(?:0[123])?)\b",
    re.I,
)
_QUESTION_RE = re.compile(r"\?|(?:\b(?:hesapla|öner|oner|nedir|ne almalıyım|ne almaliyim)\b)", re.I)


@dataclass(frozen=True)
class AcademicProfileCommand:
    major: str | None
    curriculum_term: str | None
    standalone: bool


def parse_academic_profile_command(text: str) -> AcademicProfileCommand | None:
    raw = text or ""
    major = None
    for pattern in _MAJOR_PATTERNS:
        match = pattern.search(raw)
        if match:
            candidate = match.group(1).upper()
            if candidate in curriculum_registry.list_major_programs():
                major = candidate
            break
    term_match = _TERM_PATTERN.search(raw)
    term = term_match.group(1) if term_match else None
    if term and len(term) == 4:
        term = f"{term}01"
    if not major and not term:
        return None
    return AcademicProfileCommand(major, term, not bool(_QUESTION_RE.search(raw)))


def resolve_profile_update(
    command: AcademicProfileCommand,
    current_profile: dict,
) -> tuple[dict, str | None]:
    major = command.major or str(current_profile.get("major") or "").upper() or None
    term = command.curriculum_term or str(current_profile.get("curriculum_term") or "") or None
    if not major or not term:
        return {}, "Bölüm ve müfredat dönemini birlikte belirtmelisin (ör. “bölümüm CS, müfredat dönemim 202401”)."
    record = curriculum_registry.get_major_curriculum(major, term)
    if not record:
        return {}, f"{major} için {term} müfredatı mevcut resmi müfredatlar arasında bulunamadı."
    return {
        "major": major,
        "degree_code": record.get("degree_code"),
        "curriculum_term": term,
        "profile_status": "confirmed",
    }, None
