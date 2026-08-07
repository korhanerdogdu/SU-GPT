from __future__ import annotations

"""Deterministic routing guards on top of the statistical intent classifier.

Identifiers, intent names and comments are English (see `modules.intents`). The regex bodies
are deliberately bilingual: they match what a student types, and adviSU answers Turkish
questions in Turkish, so dropping the Turkish alternatives would break the product rather
than "translate" it.
"""

import re
from dataclasses import dataclass
from typing import Any

from modules import intents
from modules.intent_detector import get_intent_with_confidence


INSTRUCTOR_REVIEW_RE = re.compile(
    r"\b(hoca|hocanın|hocanin|prof|professor|instructor|öğretmen|ogretmen|"
    r"zor mu|kolay mı|kolay mi|nasıl biri|nasil biri|yorum|review|"
    r"yücel|yucel|saygın|saygin|ercan|solak|cem say)\b",
    re.IGNORECASE,
)

EXAM_RE = re.compile(
    r"\b(final|midterm|quiz|sınav|sinav|exam|soruları|sorulari|"
    r"çıkmış|cikmis|ne çıkar|ne cikar|past paper)\b",
    re.IGNORECASE,
)

GRADUATION_RE = re.compile(
    r"\b(mezuniyet\w*|mezun\w*|kredi\w*|credit\w*|degree evaluation|degree audit|audit|"
    r"kalan ders|kalan kredi|requirements?|requirement|hangi derslerim sayıldı|"
    r"hangi derslerim sayildi)\b",
    re.IGNORECASE,
)

COURSE_RECOMMENDATION_RE = re.compile(
    r"\b(ders öner|ders oner|hangi ders|program öner|program oner|schedule|"
    r"next semester|gelecek dönem|gelecek donem|nlp|web|data|ai|security|systems)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RagRoute:
    intent: str
    base_intent: str
    confidence: float
    document_types: list[str]
    use_multi_search: bool = False

    @property
    def metadata_filter(self) -> dict[str, Any] | None:
        if not self.document_types:
            return None
        if len(self.document_types) == 1:
            return {"documentType": self.document_types[0]}
        return {"documentType": {"$in": self.document_types}}


def route_query(question: str, resolved_intent: str | None = None) -> RagRoute:
    base_intent, confidence = get_intent_with_confidence(question)
    intent = resolved_intent or base_intent or intents.OTHER
    q = question or ""

    if INSTRUCTOR_REVIEW_RE.search(q):
        return RagRoute(
            intent=intents.REVIEW,
            base_intent=base_intent,
            confidence=confidence,
            document_types=["review"],
        )

    if EXAM_RE.search(q):
        return RagRoute(
            intent=intents.EXAM,
            base_intent=base_intent,
            confidence=confidence,
            document_types=["exam"],
        )

    # A caller-supplied resolved_intent already went through main.py's more specific, carefully
    # ordered regex chain (e.g. UNTIL_GRAD_RECOMMENDATION_RE before the bare GRADUATION_INTENT_RE
    # keyword match, so "mezun olana kadar hangi dersleri almalıyım" resolves to
    # COURSE_RECOMMENDATION rather than GRADUATION_STATUS even though it contains "mezun"). This
    # module's own GRADUATION_RE/COURSE_RECOMMENDATION_RE below are coarser bare-keyword guards
    # for when no such resolution happened -- they must not silently re-derive a *different*,
    # less specific intent and discard a resolution that was already made correctly.
    if intent in intents.CURRICULUM_INTENTS:
        return RagRoute(
            intent=intent,
            base_intent=base_intent,
            confidence=confidence,
            document_types=["course"],
        )

    if GRADUATION_RE.search(q):
        return RagRoute(
            intent=intents.GRADUATION_STATUS,
            base_intent=base_intent,
            confidence=confidence,
            document_types=["course"],
        )

    if COURSE_RECOMMENDATION_RE.search(q):
        return RagRoute(
            intent=intents.COURSE_RECOMMENDATION,
            base_intent=base_intent,
            confidence=confidence,
            document_types=["course"],
        )

    return RagRoute(
        intent=intent or intents.OTHER,
        base_intent=base_intent,
        confidence=confidence,
        # Legacy review/private-chat chunks are never part of the general retrieval pool.
        document_types=["course", "exam"],
        use_multi_search=True,
    )
