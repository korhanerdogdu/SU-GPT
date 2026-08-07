from __future__ import annotations

"""Deterministic, non-ML claim-coverage check (confidence.py's missing verifier).

``confidence.assess`` fails an evidence-bound answer closed unless ``claim_coverage_checked``
is True -- an authorized citation label only proves a document was retrieved, never that its
content actually backs every sentence the model wrote. This module is that check: cheap lexical
grounding, not semantic entailment, but enough to catch a model answer that rides on a real
citation while stating something the cited text never says (e.g. a fabricated number, a claim
about an unrelated course). It is intentionally conservative in the other direction too -- short,
non-factual sentences (headers, connective filler) are skipped so paraphrase alone never trips it.
"""

import re

_WORD_RE = re.compile(r"[a-zA-Z0-9çğıöşüÇĞİÖŞÜ]+")

# Deliberately small: only high-frequency function words in each language. Anything borderline
# is left IN as a significant token -- an over-inclusive stopword list is what would let a
# fabricated claim slip through by accident.
_STOPWORDS = frozenset({
    # English
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "of", "in", "on", "to",
    "for", "and", "or", "with", "this", "that", "it", "as", "by", "at", "from", "your", "you",
    "i", "we", "they", "he", "she", "not", "no", "yes", "if", "then", "so", "but", "which",
    "who", "what", "when", "where", "how", "do", "does", "did", "has", "have", "had", "will",
    "would", "can", "could", "should", "may", "might", "these", "those", "there", "here",
    # Turkish
    "bir", "bu", "şu", "o", "ve", "ile", "için", "de", "da", "mi", "mı", "mu", "mü", "ne",
    "gibi", "ama", "fakat", "ise", "olan", "olarak", "var", "yok", "değil", "evet", "hayır",
    "göre", "kadar", "daha", "çok", "az", "her", "hiç", "ki", "ya", "veya", "sen", "ben", "biz",
    "siz", "onlar", "bu", "şu", "olduğu", "olduğunu", "olan", "üzerinden",
})

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")

# A sentence saying "this isn't in the evidence" or "I need more information from you" is not a
# domain claim to ground -- it is the model correctly being honest about a gap, which is the
# behavior this whole pipeline wants (CLAUDE.md section 1: "missing critical data causes an
# explicit abstention, never a guess"). Penalizing that sentence for having low lexical overlap
# with evidence about a *different* fact would punish the exact honesty this check exists to
# protect. These phrases are checked case-insensitively as substrings.
_HEDGE_PHRASES = (
    # English
    "not specified", "not mentioned", "not provided", "not available", "no information",
    "cannot determine", "unable to determine", "does not specify", "is not clear",
    "please provide", "more information is needed", "to give a precise answer",
    "for a more accurate answer", "we don't have", "we do not have", "cannot verify",
    "not found in", "no specific information",
    # Turkish
    "bilgi bulunmamaktadır", "bilgi verilmemiştir", "bilgi yer almamaktadır",
    "belirtilmemiştir", "belirtilmediği için", "bulunmamaktadır", "netleştirmek için",
    "daha doğru bir cevap için", "kesin bir cevap için", "lütfen belirtin",
    "doğrulayamıyorum", "bilgi bulunmuyor",
)


def _is_hedge_sentence(sentence: str) -> bool:
    lowered = sentence.lower()
    return any(phrase in lowered for phrase in _HEDGE_PHRASES)

# Below this many significant tokens, a "sentence" is a header/label/connective fragment, not an
# independently checkable factual claim -- e.g. "## Kısa Özet", "CS 201:", a lone course code.
_MIN_CLAIM_TOKENS = 4

# Fraction of a claim sentence's significant tokens that must also appear in the retrieved
# evidence's vocabulary. Not calibrated against a labeled set (matching confidence.py's own
# "no numeric threshold is selected without calibration" stance elsewhere) -- picked loosely
# enough that real paraphrased answers over short, dense catalog text clear it comfortably,
# while a claim built from vocabulary absent from the evidence entirely (the failure mode this
# exists to catch) does not.
_MIN_OVERLAP = 0.3


def _significant_tokens(text: str) -> set[str]:
    return {
        token
        for token in (match.group(0).lower() for match in _WORD_RE.finditer(text or ""))
        if len(token) > 2 and token not in _STOPWORDS
    }


def claim_coverage_supported(answer: str, evidence_texts: list[str]) -> bool:
    """True if every claim-bearing sentence in ``answer`` shares enough vocabulary with
    ``evidence_texts`` to be considered grounded. An answer with no evidence text at all is
    never supported -- there is nothing to check it against."""

    evidence_vocab = _significant_tokens(" ".join(evidence_texts))
    if not evidence_vocab:
        return False
    sentences = _SENTENCE_SPLIT_RE.split(answer or "")
    for sentence in sentences:
        if _is_hedge_sentence(sentence):
            continue
        tokens = _significant_tokens(sentence)
        if len(tokens) < _MIN_CLAIM_TOKENS:
            continue
        overlap = len(tokens & evidence_vocab) / len(tokens)
        if overlap < _MIN_OVERLAP:
            return False
    # An answer with no checkable claim sentences at all (e.g. just a course code and a
    # citation) has nothing this check can confirm or deny -- treat as supported rather than
    # penalizing terse, already-narrow answers that citations_authorized already covers.
    return True
