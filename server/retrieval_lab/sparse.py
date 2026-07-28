from __future__ import annotations

"""
Sparse retrievers: Okapi BM25 and field-weighted BM25F.

Both are built on one inverted index so that every sparse configuration is scored by the
same engine and their latencies are directly comparable. The shipped baseline scores every
document in a Python loop (O(N x |q|) over a 3000-doc slice); scoring only the postings of
query terms is algebraically identical but does not penalize the baseline on latency for an
implementation detail rather than a retrieval decision.

BM25 here reproduces the production IDF exactly:
    idf(t) = log(1 + (N - df + 0.5) / (df + 0.5))
so `bm25_original` (original tokenizer, k1=1.5, b=0.75) returns the shipped ranking.

BM25F is not "BM25 with the metadata pasted onto the body". Concatenating fields into one
string lets a long body drown a two-token course code. Proper BM25F normalizes each field by
that field's own average length, weights the per-field term frequency, sums to a pseudo-tf,
and only then applies saturation - so a hit in `course_code` keeps its weight no matter how
long the body is. That distinction is the whole reason BM25F is a candidate here: adviSU
chunks are ~90% pool-course rows that differ *only* in their structured fields.
"""

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Sequence

from .corpus import Chunk, Corpus
from .text import tokenize_original, tokenize_v2

Tokenizer = Callable[[str], list[str]]


@dataclass
class SparseHit:
    chunk_id: str
    score: float
    index: int


# --- plain BM25 -----------------------------------------------------------------------


class BM25Index:
    """Okapi BM25 over a single text field, via an inverted index."""

    def __init__(
        self,
        docs: Sequence[list[str]],
        ids: Sequence[str],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.k1, self.b = float(k1), float(b)
        self.ids = list(ids)
        self.N = len(docs)
        self.doc_len = [len(d) for d in docs]
        self.avgdl = (sum(self.doc_len) / self.N) if self.N else 0.0

        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, doc in enumerate(docs):
            counts: dict[str, int] = {}
            for tok in doc:
                counts[tok] = counts.get(tok, 0) + 1
            for tok, tf in counts.items():
                postings[tok].append((i, tf))
        self.postings = dict(postings)
        self.idf = {
            term: math.log(1 + (self.N - len(plist) + 0.5) / (len(plist) + 0.5))
            for term, plist in self.postings.items()
        }

    def search(
        self,
        query_tokens: Sequence[str],
        top: int = 50,
        allowed: set[int] | None = None,
    ) -> list[SparseHit]:
        """`allowed` restricts scoring to a subset of document indices (a metadata scope).

        This is a PRE-filter, not a post-filter: scoping has to happen while ranking, or a
        narrow scope returns nothing whenever the global top-N happens to contain none of
        its documents. That is how the Chroma `where` clause behaves, and the adapter has to
        match it.
        """
        scores: dict[int, float] = defaultdict(float)
        avgdl = self.avgdl or 1.0
        for term in query_tokens:
            plist = self.postings.get(term)
            if not plist:
                continue
            idf = self.idf[term]
            for i, tf in plist:
                if allowed is not None and i not in allowed:
                    continue
                denom = tf + self.k1 * (1 - self.b + self.b * self.doc_len[i] / avgdl)
                scores[i] += idf * (tf * (self.k1 + 1)) / denom
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:top]
        return [SparseHit(self.ids[i], s, i) for i, s in ranked if s > 0]


# --- BM25F ----------------------------------------------------------------------------

# Which chunk attribute feeds each field. `body` is the prose the dense retriever also sees.
FIELD_EXTRACTORS: dict[str, Callable[[Chunk], str]] = {
    "course_code": lambda c: c.course_id,
    "course_title": lambda c: c.course_title,
    "program": lambda c: f"{c.program} {c.meta.get('program_name') or ''}",
    "curriculum_term": lambda c: f"{c.curriculum_term} {c.meta.get('admit_term_label') or ''}",
    "requirement_category": lambda c: c.requirement_category.replace("_", " "),
    "document_type": lambda c: c.document_type.replace("_", " "),
    "body": lambda c: c.text,
}

# Hand-set from the corpus structure, then validated on the dev split - NOT swept. The
# ordering is the claim: an exact course-code hit is the strongest evidence a chunk is the
# right one, program and catalog term are the next-strongest disambiguators (they are what
# separates otherwise-identical sibling rows), and the body is the weakest because thousands
# of pool-course rows share almost identical prose. A systematic sweep is future work.
DEFAULT_FIELD_WEIGHTS: dict[str, float] = {
    "course_code": 3.0,
    "course_title": 1.5,
    "program": 2.0,
    "curriculum_term": 2.0,
    "requirement_category": 1.5,
    "document_type": 1.0,
    "body": 1.0,
}

# Short structured fields should barely be length-normalized; prose should be.
DEFAULT_FIELD_B: dict[str, float] = {
    "course_code": 0.0,
    "course_title": 0.3,
    "program": 0.0,
    "curriculum_term": 0.0,
    "requirement_category": 0.3,
    "document_type": 0.3,
    "body": 0.75,
}


class BM25FIndex:
    """Field-weighted BM25F with per-field length normalization."""

    def __init__(
        self,
        corpus: Corpus,
        tokenizer: Tokenizer = tokenize_v2,
        weights: dict[str, float] | None = None,
        field_b: dict[str, float] | None = None,
        k1: float = 1.5,
    ) -> None:
        self.k1 = float(k1)
        self.weights = dict(weights or DEFAULT_FIELD_WEIGHTS)
        self.field_b = dict(field_b or DEFAULT_FIELD_B)
        self.tokenizer = tokenizer
        self.ids = [c.chunk_id for c in corpus.chunks]
        self.N = len(corpus.chunks)
        self.fields = [f for f, w in self.weights.items() if w > 0]

        # per-field token counts and lengths
        field_len: dict[str, list[int]] = {f: [0] * self.N for f in self.fields}
        # postings[term] -> list of (doc_idx, {field: tf})
        postings: dict[str, dict[int, dict[str, int]]] = defaultdict(dict)
        for i, chunk in enumerate(corpus.chunks):
            for f in self.fields:
                toks = self.tokenizer(FIELD_EXTRACTORS[f](chunk))
                field_len[f][i] = len(toks)
                if not toks:
                    continue
                counts: dict[str, int] = {}
                for t in toks:
                    counts[t] = counts.get(t, 0) + 1
                for t, tf in counts.items():
                    postings[t].setdefault(i, {})[f] = tf

        self.field_len = field_len
        self.avg_field_len = {
            f: (sum(field_len[f]) / self.N if self.N else 0.0) or 1.0 for f in self.fields
        }
        self.postings = {t: list(d.items()) for t, d in postings.items()}
        # Document frequency is over the whole chunk (a term counts once per document,
        # not once per field) so IDF keeps its usual meaning.
        self.idf = {
            t: math.log(1 + (self.N - len(plist) + 0.5) / (len(plist) + 0.5))
            for t, plist in self.postings.items()
        }

    def search(
        self,
        query_tokens: Sequence[str],
        top: int = 50,
        allowed: set[int] | None = None,
    ) -> list[SparseHit]:
        """`allowed` pre-filters to a metadata scope; see BM25Index.search for why."""
        scores: dict[int, float] = defaultdict(float)
        for term in query_tokens:
            plist = self.postings.get(term)
            if not plist:
                continue
            idf = self.idf[term]
            for i, per_field in plist:
                if allowed is not None and i not in allowed:
                    continue
                pseudo_tf = 0.0
                for f, tf in per_field.items():
                    b = self.field_b.get(f, 0.75)
                    norm = 1 - b + b * (self.field_len[f][i] / self.avg_field_len[f])
                    pseudo_tf += self.weights[f] * tf / (norm or 1.0)
                if pseudo_tf > 0:
                    scores[i] += idf * pseudo_tf / (self.k1 + pseudo_tf)
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:top]
        return [SparseHit(self.ids[i], s, i) for i, s in ranked if s > 0]


# --- builders -------------------------------------------------------------------------


def build_bm25(
    corpus: Corpus,
    tokenizer: Tokenizer = tokenize_original,
    k1: float = 1.5,
    b: float = 0.75,
    contextual: bool = False,
) -> BM25Index:
    """BM25 over the chunk body. `contextual=True` prepends verified metadata to the body."""
    texts = [contextual_text(c) if contextual else c.text for c in corpus.chunks]
    return BM25Index(
        [tokenizer(t) for t in texts],
        [c.chunk_id for c in corpus.chunks],
        k1=k1,
        b=b,
    )


def contextual_text(chunk: Chunk) -> str:
    """Retrieval-time view of a chunk: verified metadata header + the original body.

    Nothing here is invented - every line is a field that already exists on the row. The
    original `text` is untouched and is still what the answer generator cites.
    """
    head = [
        f"[Program: {chunk.program} {chunk.meta.get('program_name') or ''}]".strip(),
        f"[Catalog term: {chunk.curriculum_term} {chunk.meta.get('admit_term_label') or ''}]".strip(),
    ]
    if chunk.requirement_category:
        head.append(f"[Requirement category: {chunk.requirement_category.replace('_', ' ')}]")
    if chunk.course_id:
        head.append(f"[Course: {chunk.course_id} {chunk.course_title}]".strip())
    if chunk.document_type:
        head.append(f"[Record type: {chunk.document_type.replace('_', ' ')}]")
    return "\n".join(head) + "\n" + chunk.text
