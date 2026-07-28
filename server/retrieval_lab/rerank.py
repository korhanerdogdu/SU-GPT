from __future__ import annotations

"""
Rerankers: reorder a fixed candidate pool to move correct evidence into the first positions.

Every reranker here takes the SAME frozen candidate pool and returns a permutation of it, so
comparisons isolate ordering quality and never confound it with candidate generation. None of
them can add or remove a candidate — `Hit@10` can only be preserved or destroyed, never
improved, which is exactly why the candidate ceiling is computed first.

WHY A GENERIC CROSS-ENCODER IS NOT THE DEFAULT ANSWER HERE
The first thing measured on this corpus was `cross-encoder/ms-marco-MiniLM-L-6-v2`, which
dropped Hit@10 from 0.94 to 0.74 despite candidate recall of 0.95. adviSU chunks are
near-identical prose separated by structured fields (program, catalog term, requirement
category); a model trained on English web QA scores prose similarity and is blind to exactly
the fields that decide relevance, so it confidently promotes the wrong catalog year. That
result is the reason `document_format` is a first-class experimental variable below: a
reranker that never sees the metadata cannot rank on it.
"""

import math
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from .corpus import Chunk, Corpus
from .fusion import extract_query_metadata
from .text import extract_course_codes, extract_term_codes, fold

Ranking = list[tuple[str, float]]


# --- input construction -------------------------------------------------------------------

def doc_raw(chunk: Chunk) -> str:
    return chunk.text


def doc_contextual(chunk: Chunk) -> str:
    """Metadata header before the body: critical fields survive right-truncation."""
    head = [
        f"[Course: {chunk.course_id} {chunk.course_title}]".strip() if chunk.course_id else "",
        f"[Program: {chunk.program} {chunk.meta.get('program_name') or ''}]".strip(),
        f"[Catalog year: {chunk.curriculum_term}]",
        f"[Requirement type: {chunk.requirement_category.replace('_', ' ')}]" if chunk.requirement_category else "",
        f"[Record: {chunk.document_type.replace('_', ' ')}]",
    ]
    return "\n".join(h for h in head if h) + "\n" + chunk.text


def query_raw(query: str, _lex: dict[str, str]) -> str:
    return query


def query_structured(query: str, lex: dict[str, str]) -> str:
    """Prefix the entities the query states explicitly. Nothing is inferred or overwritten."""
    meta = extract_query_metadata(query, lex)
    parts = []
    if meta.course_codes:
        parts.append(f"[Course codes: {', '.join(meta.course_codes)}]")
    if meta.terms:
        parts.append(f"[Catalog year: {', '.join(meta.terms)}]")
    if meta.programs:
        parts.append(f"[Program: {', '.join(meta.programs)}]")
    if meta.is_minor:
        parts.append("[Scope: minor programme]")
    return ("\n".join(parts) + "\n" + query) if parts else query


DOC_FORMATS: dict[str, Callable[[Chunk], str]] = {"raw": doc_raw, "contextual": doc_contextual}
QUERY_FORMATS: dict[str, Callable[[str, dict], str]] = {"raw": query_raw, "structured": query_structured}


# --- truncation accounting -----------------------------------------------------------------

@dataclass
class TruncationStats:
    """How often the tokenizer cut away fields the ranking decision depends on."""

    n_docs: int = 0
    truncated: int = 0
    lost_course_code: int = 0
    lost_catalog_year: int = 0

    def as_dict(self) -> dict[str, float]:
        n = self.n_docs or 1
        return {
            "docs_scored": self.n_docs,
            "truncated_frac": self.truncated / n,
            "lost_course_code_frac": self.lost_course_code / n,
            "lost_catalog_year_frac": self.lost_catalog_year / n,
        }


# --- base -----------------------------------------------------------------------------------

class Reranker:
    name: str

    def rerank(self, query: str, candidates: Sequence[str], base: Ranking) -> Ranking:
        raise NotImplementedError


class IdentityReranker(Reranker):
    """The first-stage order, unchanged. The baseline every reranker must beat."""

    def __init__(self, name: str = "original_ranking") -> None:
        self.name = name

    def rerank(self, query: str, candidates: Sequence[str], base: Ranking) -> Ranking:
        return list(base)


# --- deterministic metadata rules -------------------------------------------------------------

class MetadataRuleReranker(Reranker):
    """Interpretable non-neural reranker: score agreement on the fields that decide relevance.

    This exists as an honest baseline. If a 2 GB cross-encoder cannot beat a handful of exact
    -match rules on this corpus, that is the finding, and it should not be hidden behind a
    neural result.
    """

    def __init__(self, corpus: Corpus, lexicon: dict[str, str], name: str = "metadata_rule_rerank",
                 base_weight: float = 0.35) -> None:
        self.name = name
        self.corpus = corpus
        self.lexicon = lexicon
        self.base_weight = base_weight

    def score(self, query: str, chunk: Chunk, base_rank: int, n: int) -> float:
        meta = extract_query_metadata(query, self.lexicon)
        s = 0.0
        if meta.course_codes and chunk.course_id:
            want = {c.replace(" ", "").upper() for c in meta.course_codes}
            s += 3.0 if chunk.course_id.replace(" ", "").upper() in want else -0.5
        if meta.terms and chunk.curriculum_term:
            s += 2.0 if chunk.curriculum_term in meta.terms else -1.0
        if meta.programs and chunk.program:
            s += 2.0 if chunk.program in meta.programs else -1.0
        if meta.is_minor:
            s += 1.5 if chunk.is_minor else -1.5
        elif chunk.is_minor:
            s -= 0.75
        if meta.record_types and chunk.document_type in meta.record_types:
            s += 1.5
        # Keep the first-stage opinion as a tiebreaker rather than discarding it.
        s += self.base_weight * (1.0 - base_rank / max(n, 1))
        return s

    def rerank(self, query: str, candidates: Sequence[str], base: Ranking) -> Ranking:
        by_id = self.corpus.by_id
        n = len(candidates)
        out = [
            (cid, self.score(query, by_id[cid], i, n))
            for i, cid in enumerate(candidates) if cid in by_id
        ]
        return sorted(out, key=lambda kv: (-kv[1], kv[0]))


# --- cross-encoder -------------------------------------------------------------------------------

class CrossEncoderReranker(Reranker):
    """Pointwise sequence-classification reranker (ms-marco, bge-reranker-v2-m3, ...)."""

    def __init__(
        self,
        corpus: Corpus,
        lexicon: dict[str, str],
        *,
        name: str,
        model_name: str,
        doc_format: str = "contextual",
        query_format: str = "raw",
        max_length: int = 512,
        batch_size: int = 32,
    ) -> None:
        self.name = name
        self.corpus = corpus
        self.lexicon = lexicon
        self.model_name = model_name
        self.doc_fn = DOC_FORMATS[doc_format]
        self.query_fn = QUERY_FORMATS[query_format]
        self.max_length = max_length
        self.batch_size = batch_size
        self.doc_format = doc_format
        self.query_format = query_format
        self.trunc = TruncationStats()
        self._model = None
        self.load_seconds = 0.0

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            from .dense import _resolve_device

            t0 = time.perf_counter()
            self._model = CrossEncoder(self.model_name, device=_resolve_device(),
                                       max_length=self.max_length)
            self.load_seconds = time.perf_counter() - t0
        return self._model

    def _account(self, text: str, doc: str) -> None:
        """Record whether the tokenizer's cut removed the fields ranking depends on."""
        tok = self.model.tokenizer
        ids = tok(text, truncation=False)["input_ids"]
        self.trunc.n_docs += 1
        if len(ids) <= self.max_length:
            return
        self.trunc.truncated += 1
        kept = tok.decode(ids[: self.max_length])
        if extract_course_codes(doc) and not extract_course_codes(kept):
            self.trunc.lost_course_code += 1
        if extract_term_codes(doc) and not extract_term_codes(kept):
            self.trunc.lost_catalog_year += 1

    def rerank(self, query: str, candidates: Sequence[str], base: Ranking) -> Ranking:
        by_id = self.corpus.by_id
        q = self.query_fn(query, self.lexicon)
        pairs, ids = [], []
        for cid in candidates:
            chunk = by_id.get(cid)
            if chunk is None:
                continue
            doc = self.doc_fn(chunk)
            pairs.append((q, doc))
            ids.append(cid)
        if not pairs:
            return list(base)
        if self.trunc.n_docs < 400:  # sample truncation accounting; it is not free
            for _, d in pairs[:4]:
                self._account(f"{q} {d}", d)
        scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        return sorted(((cid, float(s)) for cid, s in zip(ids, scores)),
                      key=lambda kv: (-kv[1], kv[0]))


# --- Qwen3 reranker (causal LM yes/no) -------------------------------------------------------------

QWEN_DEFAULT_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query"
)
QWEN_DOMAIN_INSTRUCTION = (
    "Determine whether the document provides the authoritative Sabanci University curriculum "
    "or course evidence required to answer the query. Prefer exact agreement on course code, "
    "program, catalog year, requirement type, and the direction of prerequisite relationships."
)


class Qwen3Reranker(Reranker):
    """Qwen3-Reranker scores relevance as P(yes) vs P(no) on a chat-formatted judgement prompt.

    It is a causal LM, not a sequence classifier, so the score is the normalized two-way
    softmax over the "yes"/"no" token logits at the final position. That number is a
    likelihood ratio under the model's prompt, NOT a calibrated probability of relevance, and
    it is never reported as one.
    """

    def __init__(
        self,
        corpus: Corpus,
        lexicon: dict[str, str],
        *,
        name: str,
        model_name: str = "Qwen/Qwen3-Reranker-0.6B",
        instruction: str = QWEN_DOMAIN_INSTRUCTION,
        doc_format: str = "contextual",
        query_format: str = "raw",
        max_length: int = 512,
        batch_size: int = 2,
    ) -> None:
        self.name = name
        self.corpus = corpus
        self.lexicon = lexicon
        self.model_name = model_name
        self.instruction = instruction
        self.doc_fn = DOC_FORMATS[doc_format]
        self.query_fn = QUERY_FORMATS[query_format]
        self.max_length = max_length
        self.batch_size = batch_size
        self.doc_format = doc_format
        self.query_format = query_format
        self.trunc = TruncationStats()
        self._tok = self._model = None
        self.load_seconds = 0.0

    def _load(self):
        if self._model is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            from .dense import _resolve_device

            t0 = time.perf_counter()
            self._tok = AutoTokenizer.from_pretrained(self.model_name, padding_side="left")
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name, dtype=torch.float16
            ).to(_resolve_device()).eval()
            self._yes = self._tok.convert_tokens_to_ids("yes")
            self._no = self._tok.convert_tokens_to_ids("no")
            self.load_seconds = time.perf_counter() - t0
        return self._tok, self._model

    def _prompt(self, query: str, doc: str) -> str:
        return (
            "<|im_start|>system\nJudge whether the Document meets the requirements based on the "
            "Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."
            "<|im_end|>\n<|im_start|>user\n"
            f"<Instruct>: {self.instruction}\n<Query>: {query}\n<Document>: {doc}"
            "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )

    def rerank(self, query: str, candidates: Sequence[str], base: Ranking) -> Ranking:
        import torch

        tok, model = self._load()
        by_id = self.corpus.by_id
        q = self.query_fn(query, self.lexicon)
        prompts, ids = [], []
        for cid in candidates:
            chunk = by_id.get(cid)
            if chunk is None:
                continue
            prompts.append(self._prompt(q, self.doc_fn(chunk)))
            ids.append(cid)
        if not prompts:
            return list(base)

        scores: list[float] = []
        for i in range(0, len(prompts), self.batch_size):
            batch = prompts[i:i + self.batch_size]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                      max_length=self.max_length).to(model.device)
            self.trunc.n_docs += len(batch)
            self.trunc.truncated += int((enc["attention_mask"].sum(1) >= self.max_length).sum())
            with torch.inference_mode():
                logits = model(**enc).logits[:, -1, :]
            pair = torch.stack([logits[:, self._no], logits[:, self._yes]], dim=1).float()
            scores.extend(torch.softmax(pair, dim=1)[:, 1].tolist())
            # Release the MPS command-buffer allocations eagerly. Without this the allocator
            # grows across a long evaluation until Metal reports
            # kIOGPUCommandBufferCallbackErrorOutOfMemory and the process wedges in
            # uninterruptible IO wait - observed after ~2h on an M2.
            del enc, logits, pair
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        return sorted(((cid, float(s)) for cid, s in zip(ids, scores)),
                      key=lambda kv: (-kv[1], kv[0]))


# --- fusion of reranker score with first-stage signal ------------------------------------------------

class FusionReranker(Reranker):
    """Rank-fuse a reranker's ordering with the first-stage ordering (and optional rules).

    A reranker that ignores the first stage throws away the metadata evidence that produced
    the pool. RRF over ranks avoids adding a cross-encoder logit to a BM25F score, which are
    on incomparable scales.
    """

    def __init__(self, name: str, components: Sequence[Reranker], *, rrf_k: int = 60,
                 weights: Sequence[float] | None = None) -> None:
        self.name = name
        self.components = list(components)
        self.rrf_k = rrf_k
        self.weights = list(weights) if weights else [1.0] * len(components)

    def rerank(self, query: str, candidates: Sequence[str], base: Ranking) -> Ranking:
        from .fusion import reciprocal_rank_fusion

        rankings = [c.rerank(query, candidates, base) for c in self.components]
        return reciprocal_rank_fusion(rankings, k=self.rrf_k, weights=self.weights,
                                      top=len(candidates))
