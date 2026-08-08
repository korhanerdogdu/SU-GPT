from __future__ import annotations

"""Small, deterministic language routing for user-facing Turkish/English text.

The detector deliberately ignores course codes, URLs, e-mail addresses, code blocks, and
all-uppercase technical acronyms.  Those tokens are common in advising questions but carry no
useful language signal (``CS 412``, ``NLP``, and ``ECTS`` must not force an English answer).

``detect_language`` is the compatibility entry point for the existing LLM code and always
returns ``"tr"`` or ``"en"``.  Call ``analyze_language`` when the UI or telemetry needs to know
that an input is genuinely mixed-language.
"""

import re
import unicodedata
from dataclasses import dataclass
from typing import Literal


LanguageTag = Literal["tr", "en", "mixed"]
ResponseLanguage = Literal["tr", "en"]


_COURSE_CODE_RE = re.compile(r"\b[A-Za-z]{2,6}\s*-?\s*\d{3,5}[A-Za-z]?\b")
_URL_EMAIL_RE = re.compile(r"(?:https?://|www\.)\S+|[\w.+-]+@[\w-]+(?:\.[\w-]+)+", re.I)
_CODE_BLOCK_RE = re.compile(r"```[\s\S]*?```|`[^`]*`")
_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
_TURKISH_CHARS = frozenset("çğıöşüÇĞİÖŞÜ")

# Function words and common advising verbs are much more reliable than subject nouns.  A token
# contributes once; large copied passages therefore do not swamp a short user question.
_TR_WORDS = frozenset(
    {
        "acaba", "aldım", "aldim", "almalıyım", "almaliyim", "bana", "ben", "benim",
        "bir", "bu", "çok", "cok", "ders", "dersi", "dersler", "dönem", "donem", "durum",
        "durumu", "durumum", "durumumu", "gerekiyor", "gerekli", "gibi", "hangi", "için",
        "icin", "ile", "kaç", "kac", "kaldı", "kaldi", "kredi", "kim", "mı", "mi", "mu",
        "mü", "mufredat", "müfredat", "nasıl", "nasil", "ne", "nedir", "öner", "önerir",
        "oner", "program", "sayılır", "sayilir", "seçmeli", "secmeli", "şu", "su", "ve",
        "veya", "zorunlu",
        # High-frequency academic-advising verbs/nouns this app's own starters and follow-up
        # flows use, none of which were caught by the inflection-suffix heuristic below (their
        # possessive/case suffixes don't end in those exact strings) -- "Mezuniyet durumumu
        # hesapla" (a literal main-menu starter) scored zero Turkish signal without these and
        # silently defaulted to English.
        "mezuniyet", "mezun", "hesapla", "hesap", "kalan", "tamamla", "tamamlar",
        "yükselt", "yukselt", "değiştir", "degistir", "ekle", "çıkar", "cikar",
        "yerine", "yaz", "seç", "sec",
    }
)
_EN_WORDS = frozenset(
    {
        "a", "about", "and", "are", "can", "completed", "course", "courses", "credit",
        "credits", "curriculum", "do", "does", "elective", "for", "from", "graduation",
        "how", "i", "in", "is", "it", "major", "minor", "my", "need", "of", "or", "plan",
        "prerequisite", "recommend", "requirements", "should", "student", "take", "term",
        "teacher", "teaches", "the", "this", "to", "what", "which", "who", "with", "would", "year",
    }
)


@dataclass(frozen=True)
class LanguageAnalysis:
    """Explainable result without retaining the original user text."""

    classification: LanguageTag
    response_language: ResponseLanguage
    turkish_score: int
    english_score: int
    signal_token_count: int


def _strip_diacritics(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def _signal_tokens(text: str) -> list[str]:
    cleaned = _CODE_BLOCK_RE.sub(" ", text or "")
    cleaned = _URL_EMAIL_RE.sub(" ", cleaned)
    cleaned = _COURSE_CODE_RE.sub(" ", cleaned)
    tokens: list[str] = []
    for raw in _WORD_RE.findall(cleaned):
        folded = raw.casefold()
        ascii_folded = _strip_diacritics(folded)
        # Unknown technical acronyms such as NLP/ECTS/API are neutral, while an all-caps user
        # message ("DERS ÖNER" / "WHAT COURSE") still retains its real language signals.
        if (
            len(raw) > 1
            and raw.isupper()
            and folded not in _TR_WORDS
            and ascii_folded not in _TR_WORDS
            and folded not in _EN_WORDS
        ):
            continue
        tokens.append(folded)
    return tokens


def analyze_language(text: str) -> LanguageAnalysis:
    """Classify text as Turkish, English, or mixed and choose its response language.

    Mixed requires meaningful evidence for both languages, not a single borrowed word.  For a
    mixed query, the stronger language wins; an exact tie follows the legacy behaviour and uses
    Turkish when the text contains an explicit Turkish character, otherwise English.
    """

    raw = text or ""
    tokens = _signal_tokens(raw)
    tr_score = 0
    en_score = 0

    for token in set(tokens):
        ascii_token = _strip_diacritics(token)
        if token in _TR_WORDS or ascii_token in _TR_WORDS:
            tr_score += 2
        if token in _EN_WORDS:
            en_score += 2
    # Turkish letters are strong evidence, but count them once for the sentence. Otherwise a
    # Turkish instructor's first and last name can outweigh an otherwise clear English question.
    if any(ch in _TURKISH_CHARS for ch in raw):
        tr_score += 2

    # Inflected Turkish words often are not in a compact lexicon.  These suffixes only count on
    # reasonably long words, limiting false positives on English technical terms.
    for token in set(tokens):
        ascii_token = _strip_diacritics(token)
        if len(ascii_token) >= 6 and ascii_token.endswith(
            ("lar", "ler", "lik", "luk", "mek", "mak", "dan", "den", "dir", "dır", "miyim", "misin")
        ):
            tr_score += 1

    if tr_score >= 2 and en_score >= 2 and min(tr_score, en_score) * 2 >= max(tr_score, en_score):
        classification: LanguageTag = "mixed"
    elif tr_score > en_score:
        classification = "tr"
    else:
        # Empty, code-only and acronym-only messages keep the existing English default.
        classification = "en"

    if classification == "tr":
        response: ResponseLanguage = "tr"
    elif classification == "en":
        response = "en"
    elif tr_score > en_score:
        response = "tr"
    elif en_score > tr_score:
        response = "en"
    else:
        response = "tr" if any(ch in _TURKISH_CHARS for ch in raw) else "en"

    return LanguageAnalysis(
        classification=classification,
        response_language=response,
        turkish_score=tr_score,
        english_score=en_score,
        signal_token_count=len(tokens),
    )


def classify_language(text: str) -> LanguageTag:
    """Return the three-way classification used by UI/telemetry."""

    return analyze_language(text).classification


def detect_language(text: str) -> ResponseLanguage:
    """Compatibility entry point matching the existing ``modules.llm`` return contract."""

    return analyze_language(text).response_language
