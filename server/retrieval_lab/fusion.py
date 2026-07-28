from __future__ import annotations

"""
Rank fusion and query-metadata handling.

BM25 scores, cosine similarities and cross-encoder logits live on incompatible scales, so
they are never summed directly. Fusion here is rank-based (Reciprocal Rank Fusion) or uses
explicitly min-max normalized scores, which is the one thing the task brief calls out as a
correctness trap.

`extract_query_metadata` + `apply_metadata_boost` implement the soft-filter policy: an
entity that is unambiguously present in the query (a course code, a 6-digit term, a program
name) boosts matching chunks instead of hard-filtering them. A hard filter on a *predicted*
entity is unrecoverable - one bad program guess and the correct chunk can never be returned
at any K - whereas a boost degrades gracefully. Hard filtering is available for the ablation
that measures exactly that risk.
"""

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from .corpus import Corpus
from .text import extract_course_codes, extract_term_codes, fold

Ranking = Sequence[tuple[str, float]]


# --- fusion ----------------------------------------------------------------------------


def reciprocal_rank_fusion(
    rankings: Iterable[Ranking],
    *,
    k: int = 60,
    weights: Sequence[float] | None = None,
    top: int = 100,
) -> list[tuple[str, float]]:
    """RRF: score(d) = sum_i w_i / (k + rank_i(d)). Scale-free, so it needs no normalization."""
    rankings = list(rankings)
    w = list(weights) if weights else [1.0] * len(rankings)
    scores: dict[str, float] = defaultdict(float)
    for wi, ranking in zip(w, rankings):
        for rank, (cid, _) in enumerate(ranking, 1):
            scores[cid] += wi / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:top]


def minmax_fusion(
    rankings: Iterable[Ranking],
    *,
    weights: Sequence[float] | None = None,
    top: int = 100,
) -> list[tuple[str, float]]:
    """Weighted sum of per-ranking min-max normalized scores (score-based alternative to RRF)."""
    rankings = list(rankings)
    w = list(weights) if weights else [1.0] * len(rankings)
    scores: dict[str, float] = defaultdict(float)
    for wi, ranking in zip(w, rankings):
        if not ranking:
            continue
        vals = [s for _, s in ranking]
        lo, hi = min(vals), max(vals)
        span = (hi - lo) or 1.0
        for cid, s in ranking:
            scores[cid] += wi * (s - lo) / span
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:top]


# --- query metadata ---------------------------------------------------------------------


@dataclass
class QueryMetadata:
    course_codes: list[str] = field(default_factory=list)
    terms: list[str] = field(default_factory=list)
    programs: list[str] = field(default_factory=list)
    is_minor: bool = False
    record_types: list[str] = field(default_factory=list)

    @property
    def has_any(self) -> bool:
        return bool(
            self.course_codes or self.terms or self.programs or self.is_minor or self.record_types
        )


_MINOR_HINTS = ("minor", "yandal", "yan dal")

# Record-type (granularity) intent.
#
# Error analysis on the dev split: for "how many SU credits of free electives are required",
# BM25F returned the right program and the right term but the wrong *granularity* - individual
# pool_course rows (NS 206, NS 209) instead of the category_pool row that states the minimum.
# That is structural, not lexical: the corpus holds 27,490 pool_course rows against 332
# category_pool rows, so the course rows dominate any category-level query on sheer mass.
#
# A query asking for an aggregate ("minimum", "total", "how many credits of X") wants a
# category_pool / rule / profile row; a query naming a specific course wants a pool_course row.
# Substring hints are safe only for multi-word phrases. Single words must be matched on word
# boundaries: a bare "graduate" substring also fires inside "Undergraduate Program", which is
# part of every program_name in the corpus, so a natural-phrasing question that merely names
# the student's programme was being misread as an aggregate/graduation question.
# "What does the Mathematics minor require?" asks about a *programme*, so the answer lives in
# a profile/rule row, not in any one course row. These phrasings were previously catching the
# aggregate class only by accident, via "graduate" matching inside "Undergraduate Program";
# removing that accident exposed the gap, so the intent is now stated explicitly. The signal
# is domain logic - a question about what a programme requires wants the programme-level
# record - but note the surface forms come from this project's own question templates, so
# their coverage of real student phrasing is unverified (see report Limitation 1).
_AGGREGATE_PHRASES = (
    "how many su credits of", "how many credits of", "in total",
    "required in the", "credit requirement", "requirement for",
    "kac kredi",
    "requirements of", "require for", "conditions must be met",
    "what does the", "requirements for",
)
_AGGREGATE_WORDS = ("minimum", "total", "graduate", "graduation", "toplam", "mezun")
_RULE_PHRASES = ("count as",)
_RULE_WORDS = ("rule", "policy", "allowed", "excluded", "kural")

_WORD_BOUNDARY_CACHE: dict[str, re.Pattern[str]] = {}


def _has_word(folded: str, word: str) -> bool:
    pattern = _WORD_BOUNDARY_CACHE.get(word)
    if pattern is None:
        pattern = re.compile(rf"\b{re.escape(word)}\b")
        _WORD_BOUNDARY_CACHE[word] = pattern
    return bool(pattern.search(folded))

AGGREGATE_TYPES = (
    "degree_requirement_category_pool",
    "degree_requirement_rule",
    "degree_requirement_profile",
    "minor_requirement_category_pool",
    "minor_requirement_rule",
    "minor_requirement_profile",
)
COURSE_TYPES = ("degree_requirement_pool_course", "minor_requirement_pool_course")


def infer_record_types(query: str, has_course_code: bool) -> list[str]:
    """Which record granularity the question is asking for. Empty = no strong signal."""
    folded = fold(query)
    aggregate = (
        any(p in folded for p in _AGGREGATE_PHRASES)
        or any(_has_word(folded, w) for w in _AGGREGATE_WORDS)
        or any(p in folded for p in _RULE_PHRASES)
        or any(_has_word(folded, w) for w in _RULE_WORDS)
    )
    # A named course code is the strongest evidence for a course-level row, and it overrides
    # a generic "how many credits" phrasing ("how many credits is CS 414" is course-level).
    if has_course_code:
        return list(COURSE_TYPES)
    if aggregate:
        return list(AGGREGATE_TYPES)
    return []


def build_program_lexicon(corpus: Corpus) -> dict[str, str]:
    """Surface form -> program code. Includes codes and full program names."""
    lex: dict[str, str] = {}
    for code, name in corpus.program_names.items():
        lex[fold(code)] = code
        if name:
            lex[fold(name)] = code
        # "CS" from "CS-MINOR" style codes, and the bare stem of a minor code
        if code.endswith("-MINOR"):
            lex[fold(code[: -len("-MINOR")])] = code
    return lex


def extract_query_metadata(query: str, lexicon: dict[str, str]) -> QueryMetadata:
    """Pull only *unambiguous* entities out of the query. Nothing is guessed."""
    folded = fold(query)
    codes = extract_course_codes(query)
    terms = extract_term_codes(query)
    is_minor = any(h in folded for h in _MINOR_HINTS)

    programs: list[str] = []
    # Longest surface forms first so "computer science" wins over a stray "cs" substring.
    for surface in sorted(lexicon, key=len, reverse=True):
        if len(surface) < 2:
            continue
        # Word-boundary match; short codes must stand alone to avoid matching inside words.
        padded = f" {folded} "
        if f" {surface} " in padded or f" {surface}?" in padded or f" {surface}," in padded:
            code = lexicon[surface]
            if code not in programs:
                programs.append(code)
    # A course code like "CS 414" mentions a subject prefix that is not a program request.
    if codes and programs:
        subjects = {c.split()[0].upper() for c in codes}
        if len(programs) == 1 and programs[0] in subjects and not any(
            f" {fold(programs[0])} " in f" {folded} ".replace(fold(c), " ") for c in codes
        ):
            programs = []
    if is_minor:
        programs = [p for p in programs if p.endswith("-MINOR")] or programs
    return QueryMetadata(
        course_codes=codes,
        terms=terms,
        programs=programs,
        is_minor=is_minor,
        record_types=infer_record_types(query, has_course_code=bool(codes)),
    )


def apply_metadata_boost(
    ranking: Ranking,
    corpus: Corpus,
    meta: QueryMetadata,
    *,
    course_boost: float = 1.0,
    term_boost: float = 0.6,
    program_boost: float = 0.6,
    role_boost: float = 0.4,
    record_type_boost: float = 1.2,
    top: int = 100,
) -> list[tuple[str, float]]:
    """Soft re-scoring: add a bounded bonus per metadata field the chunk agrees with.

    Operates on RRF scores, so the bonus is expressed in the same units. A chunk that
    disagrees is never removed - it just loses the bonus - which is what keeps a wrong
    entity guess recoverable.
    """
    if not meta.has_any or not ranking:
        return list(ranking)[:top]
    by_id = corpus.by_id
    vals = [s for _, s in ranking]
    unit = (max(vals) - min(vals)) or (max(vals) or 1.0)
    wanted_codes = {c.replace(" ", "").upper() for c in meta.course_codes}

    rescored: list[tuple[str, float]] = []
    for cid, score in ranking:
        chunk = by_id.get(cid)
        if chunk is None:
            rescored.append((cid, score))
            continue
        bonus = 0.0
        if wanted_codes and chunk.course_id:
            if chunk.course_id.replace(" ", "").upper() in wanted_codes:
                bonus += course_boost
        if meta.terms and chunk.curriculum_term:
            if chunk.curriculum_term in meta.terms:
                bonus += term_boost
        if meta.programs and chunk.program:
            if chunk.program in meta.programs:
                bonus += program_boost
        if meta.is_minor and chunk.is_minor:
            bonus += role_boost
        elif not meta.is_minor and not chunk.is_minor:
            bonus += role_boost * 0.5
        if meta.record_types and chunk.document_type in meta.record_types:
            bonus += record_type_boost
        rescored.append((cid, score + bonus * unit))
    return sorted(rescored, key=lambda kv: (-kv[1], kv[0]))[:top]


def apply_metadata_filter(
    ranking: Ranking,
    corpus: Corpus,
    meta: QueryMetadata,
    *,
    top: int = 100,
) -> list[tuple[str, float]]:
    """Hard filter (ablation only): drop chunks that contradict an extracted entity.

    Kept so the report can quantify the failure mode that motivates using boosts instead.
    Falls back to the unfiltered ranking when the filter would empty the result.
    """
    if not meta.has_any:
        return list(ranking)[:top]
    by_id = corpus.by_id
    kept: list[tuple[str, float]] = []
    for cid, score in ranking:
        chunk = by_id.get(cid)
        if chunk is None:
            continue
        if meta.terms and chunk.curriculum_term and chunk.curriculum_term not in meta.terms:
            continue
        if meta.programs and chunk.program and chunk.program not in meta.programs:
            continue
        if meta.is_minor and not chunk.is_minor:
            continue
        kept.append((cid, score))
    return (kept or list(ranking))[:top]
