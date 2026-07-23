from __future__ import annotations

"""
Dependency-free BM25 lexical retrieval for hybrid search (roadmap section 12).

Dense (vector) search is strong on semantic similarity but can miss exact strings like
"CS 455", "MATH201", or "202401". BM25 is the opposite. We run BM25 over the SAME
profile/data_role-scoped subset that vector search uses (never the whole corpus), so hybrid
recall never crosses program/curriculum boundaries. Implemented in-process (Okapi BM25, no
external package) so it works even without rank-bm25 installed.
"""

import math
import re
from collections import Counter
from typing import Any

from langchain_core.documents import Document

# Only run BM25 when the metadata filter actually narrows the corpus (a program / term /
# course / minor scope). Otherwise the subset is arbitrary and BM25 adds noise.
_NARROWING_KEYS = {"program", "curriculum_term", "term_code", "course_id"}
_SUBSET_CAP = 3000


def _tokenize(text: str) -> list[str]:
    text = (text or "").lower()
    tokens = re.findall(r"[a-z0-9]+", text)
    # add merged course-code tokens so "cs 455" also matches the exact string "cs455"
    for subj, num in re.findall(r"\b([a-z]{2,5})\s+(\d{3,5}[a-z]?)\b", text):
        tokens.append(f"{subj}{num}")
    return tokens


class _BM25:
    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.docs = corpus
        self.N = len(corpus)
        self.doc_len = [len(d) for d in corpus]
        self.avgdl = (sum(self.doc_len) / self.N) if self.N else 0.0
        df: Counter[str] = Counter()
        for d in corpus:
            for term in set(d):
                df[term] += 1
        self.idf = {
            term: math.log(1 + (self.N - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }
        self.tf = [Counter(d) for d in corpus]

    def scores(self, query_tokens: list[str]) -> list[float]:
        q = [t for t in query_tokens if t in self.idf]
        out = [0.0] * self.N
        for i in range(self.N):
            tf_i, dl = self.tf[i], self.doc_len[i]
            s = 0.0
            for term in q:
                f = tf_i.get(term, 0)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                s += self.idf[term] * (f * (self.k1 + 1)) / denom
            out[i] = s
        return out


def _is_narrowing(where: dict[str, Any] | None) -> bool:
    if not where:
        return False
    for clause in (where.get("$and") or [where]):
        if isinstance(clause, dict) and any(k in _NARROWING_KEYS for k in clause):
            return True
    return False


def annotate_bm25(
    vectorstore: Any,
    query: str,
    where: dict[str, Any] | None,
    top: int = 25,
    require_narrowing: bool = True,
) -> list[Document]:
    """Return the BM25 top-`top` docs from the scoped subset, each tagged metadata['_bm25'] in [0,1].

    Returns [] when the filter is not narrowing (keeps hybrid cheap and scoped). Standalone
    `bm25` retrieval mode (Section 3) passes require_narrowing=False: as a baseline it must
    return lexical hits for any question, not only program/term/course-scoped ones."""
    if require_narrowing and not _is_narrowing(where):
        return []
    collection = getattr(vectorstore, "_collection", None)
    if collection is None:
        return []
    try:
        res = collection.get(where=where, limit=_SUBSET_CAP, include=["documents", "metadatas"])
    except Exception:
        return []
    docs = res.get("documents") or []
    metas = res.get("metadatas") or []
    if not docs:
        return []

    bm25 = _BM25([_tokenize(d) for d in docs])
    scores = bm25.scores(_tokenize(query))
    peak = max(scores) or 1.0
    order = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)[:top]
    out: list[Document] = []
    for i in order:
        if scores[i] <= 0:
            continue
        meta = dict(metas[i] or {})
        meta["_bm25"] = round(scores[i] / peak, 4)
        out.append(Document(page_content=docs[i], metadata=meta))
    return out
