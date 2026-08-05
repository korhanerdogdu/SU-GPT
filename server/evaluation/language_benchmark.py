from __future__ import annotations

"""Deterministic 100-TR/100-EN language-selection regression benchmark."""

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class LanguageCase:
    case_id: str
    text: str
    expected: str
    subset: str


_TR_STEMS = (
    "Bu dersin ön koşulu nedir",
    "Gelecek dönem hangi dersi almalıyım",
    "Mezuniyet için kaç kredi kaldı",
    "Ders programımı hazırlar mısın",
    "Bu seçmeli ders zor mu",
    "Müfredat gereksinimlerini açıklar mısın",
    "Hangi derslerimi tamamladım",
    "Bu dönem için ders öner",
    "Programımda bu ders sayılır mı",
    "Haftalık programda çakışma var mı",
)
_EN_STEMS = (
    "What is the prerequisite for this course",
    "Which course should I take next term",
    "How many credits remain for graduation",
    "Can you build my weekly schedule",
    "Is this elective course difficult",
    "Please explain the curriculum requirements",
    "Which courses have I completed",
    "Recommend courses for this term",
    "Does this course count toward my program",
    "Is there a conflict in the weekly schedule",
)
_NEUTRAL_SUFFIXES = (
    "CS 201",
    "CS 412 NLP",
    "MATH 203 ECTS",
    "IF 100 API",
    "DSA 210 Python",
    "ENS 491 CRN 12345",
    "202401 BSCS",
    "CS 300 SQL",
    "EE 417 AI",
    "CS 455 URL https://example.invalid",
)


def build_cases() -> list[LanguageCase]:
    cases: list[LanguageCase] = []
    for language, stems in (("tr", _TR_STEMS), ("en", _EN_STEMS)):
        for stem_index, stem in enumerate(stems):
            for suffix_index, suffix in enumerate(_NEUTRAL_SUFFIXES):
                cases.append(
                    LanguageCase(
                        case_id=f"{language}-{stem_index:02d}-{suffix_index:02d}",
                        text=f"{stem}? {suffix}",
                        expected=language,
                        subset="course_code_heavy" if suffix_index < 8 else "code_url_heavy",
                    )
                )
    for index in range(10):
        cases.append(LanguageCase(
            case_id=f"mixed-tr-{index:02d}",
            text=f"Can you explain bu dersin ön koşulu nedir ve nasıl alabilirim? CS {201 + index}",
            expected="tr",
            subset="mixed_dominant_tr",
        ))
        cases.append(LanguageCase(
            case_id=f"mixed-en-{index:02d}",
            text=f"Bu course için what prerequisite do I need and when should I take it? CS {301 + index}",
            expected="en",
            subset="mixed_dominant_en",
        ))
    return cases


def evaluate(detector: Callable[[str], str]) -> dict:
    cases = build_cases()
    by_language: dict[str, dict] = {}
    for language in ("tr", "en"):
        selected = [case for case in cases if case.expected == language and not case.subset.startswith("mixed")]
        passed = sum(detector(case.text) == language for case in selected)
        by_language[language] = {
            "n": len(selected),
            "passed": passed,
            "accuracy": passed / len(selected),
        }
    return {
        "schema_version": 1,
        "benchmark_id": "language-selection-v1",
        "synthetic_deterministic": True,
        "total": len(cases),
        "by_language": by_language,
        "mixed": {
            "n": sum(case.subset.startswith("mixed") for case in cases),
            "passed": sum(
                detector(case.text) == case.expected
                for case in cases
                if case.subset.startswith("mixed")
            ),
        },
    }
