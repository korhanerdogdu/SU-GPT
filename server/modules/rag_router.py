from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from modules.intent_detector import get_intent_with_confidence


REVIEW_RE = re.compile(
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

RECOMMENDATION_RE = re.compile(
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
    intent = resolved_intent or base_intent or "diger"
    q = question or ""

    if REVIEW_RE.search(q):
        return RagRoute(
            intent="review",
            base_intent=base_intent,
            confidence=confidence,
            document_types=["review"],
        )

    if EXAM_RE.search(q):
        return RagRoute(
            intent="exam",
            base_intent=base_intent,
            confidence=confidence,
            document_types=["exam"],
        )

    if GRADUATION_RE.search(q):
        return RagRoute(
            intent="mezuniyet_durumu",
            base_intent=base_intent,
            confidence=confidence,
            document_types=["course"],
        )

    if RECOMMENDATION_RE.search(q):
        return RagRoute(
            intent="ders_onerisi",
            base_intent=base_intent,
            confidence=confidence,
            document_types=["course"],
        )

    if intent in {
        "mezuniyet_durumu",
        "ders_onerisi",
        "ders_ayrintisi",
        "calisma_plani",
        "major_secimi",
        "alanda_ozellesme",
    }:
        return RagRoute(
            intent=intent,
            base_intent=base_intent,
            confidence=confidence,
            document_types=["course"],
        )

    return RagRoute(
        intent=intent or "diger",
        base_intent=base_intent,
        confidence=confidence,
        document_types=["course", "exam", "review"],
        use_multi_search=True,
    )
