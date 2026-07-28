from __future__ import annotations

"""
The experiment registry: every retrieval configuration the benchmark can run.

All retrievers expose the same tiny interface (`name`, `search_batch`) so the runner, the
production adapter and the figures all speak to them identically - that is what keeps the
benchmarked system and the shipped system from diverging.

ON THE TWO BM25 BASELINES
`bm25_original` is the shipped configuration: the production tokenizer, k1=1.5, b=0.75, and
crucially the 3000-document subset cap from modules/bm25_retriever.py. In standalone `bm25`
mode the production code calls annotate_bm25(require_narrowing=False) with no metadata
filter, so Chroma's `.get(limit=3000)` hands BM25 an arbitrary ~10% slice of the 30,343-chunk
corpus and gold chunks outside that slice are unreachable at any K.

That is a real property of the shipped system and it is reported as-is. But beating a
crippled baseline would prove nothing, so `bm25_full_corpus` runs the identical scoring over
the whole corpus, and the winner is required to beat the STRONGER of the two. Reporting only
the capped number would be the cherry-pick the brief forbids.
"""

import time
from dataclasses import dataclass, field
from typing import Callable, Protocol, Sequence

from .corpus import Corpus
from .dense import DENSE_CONFIGS, DenseIndex
from .fusion import (
    QueryMetadata,
    apply_metadata_boost,
    apply_metadata_filter,
    build_program_lexicon,
    extract_query_metadata,
    minmax_fusion,
    reciprocal_rank_fusion,
)
from .sparse import BM25FIndex, BM25Index, build_bm25
from .text import tokenize_original, tokenize_v2

Ranking = list[tuple[str, float]]

# Production BM25 reads at most this many documents from Chroma (modules/bm25_retriever.py).
PRODUCTION_SUBSET_CAP = 3000


class Retriever(Protocol):
    name: str

    def search_batch(self, queries: Sequence[str], top: int) -> list[Ranking]: ...


@dataclass
class BuildInfo:
    build_seconds: float = 0.0
    index_bytes: int = 0
    notes: str = ""


# --- sparse ------------------------------------------------------------------------------


class SparseRetriever:
    """BM25 / BM25F, optionally restricted to the production subset cap."""

    def __init__(
        self,
        name: str,
        corpus: Corpus,
        *,
        tokenizer=tokenize_original,
        k1: float = 1.5,
        b: float = 0.75,
        contextual: bool = False,
        subset_cap: int | None = None,
        bm25f: bool = False,
        field_weights: dict[str, float] | None = None,
        field_b: dict[str, float] | None = None,
    ) -> None:
        self.name = name
        self.tokenizer = tokenizer
        self.corpus = corpus
        started = time.perf_counter()

        sub = corpus
        if subset_cap is not None and subset_cap < len(corpus.chunks):
            # Chroma's .get(limit=N) returns documents in internal insertion order, which is
            # the order ingest_degree_requirements.py walks the sorted JSONL files - the same
            # order load_corpus() produces. Taking the first N reproduces that slice.
            sub = Corpus(chunks=corpus.chunks[:subset_cap])
        self.subset = sub

        if bm25f:
            self.index: BM25Index | BM25FIndex = BM25FIndex(
                sub, tokenizer=tokenizer, weights=field_weights, field_b=field_b, k1=k1
            )
        else:
            self.index = build_bm25(sub, tokenizer=tokenizer, k1=k1, b=b, contextual=contextual)
        self.build = BuildInfo(
            build_seconds=time.perf_counter() - started,
            notes=f"docs={len(sub.chunks)}",
        )

    def search_batch(self, queries: Sequence[str], top: int) -> list[Ranking]:
        out: list[Ranking] = []
        for q in queries:
            hits = self.index.search(self.tokenizer(q), top=top)
            out.append([(h.chunk_id, h.score) for h in hits])
        return out


# --- dense --------------------------------------------------------------------------------


class DenseRetriever:
    def __init__(self, name: str, corpus: Corpus, config_key: str) -> None:
        self.name = name
        started = time.perf_counter()
        self.index = DenseIndex(DENSE_CONFIGS[config_key], corpus)
        self.build = BuildInfo(
            build_seconds=getattr(self.index, "build_seconds", time.perf_counter() - started),
            index_bytes=int(self.index.embeddings.nbytes) if hasattr(self.index.embeddings, "nbytes") else 0,
            notes=DENSE_CONFIGS[config_key].model_name,
        )

    def search_batch(self, queries: Sequence[str], top: int) -> list[Ranking]:
        return [list(r) for r in self.index.search_many(list(queries), top=top)]


# --- hybrid / metadata ----------------------------------------------------------------------


class FusionRetriever:
    """Fuse several component retrievers, then optionally apply the metadata policy."""

    def __init__(
        self,
        name: str,
        corpus: Corpus,
        components: Sequence[Retriever],
        *,
        method: str = "rrf",
        rrf_k: int = 60,
        weights: Sequence[float] | None = None,
        candidate_depth: int = 200,
        metadata_policy: str = "none",  # none | boost | filter
        boost_kwargs: dict | None = None,
    ) -> None:
        self.name = name
        self.corpus = corpus
        self.components = list(components)
        self.method = method
        self.rrf_k = rrf_k
        self.weights = list(weights) if weights else None
        self.candidate_depth = candidate_depth
        self.metadata_policy = metadata_policy
        self.boost_kwargs = boost_kwargs or {}
        self.lexicon = build_program_lexicon(corpus)
        self.build = BuildInfo(
            build_seconds=sum(getattr(c, "build", BuildInfo()).build_seconds for c in self.components),
            notes=f"components={[c.name for c in self.components]}",
        )

    def search_batch(self, queries: Sequence[str], top: int) -> list[Ranking]:
        depth = max(self.candidate_depth, top)
        per_component = [c.search_batch(queries, depth) for c in self.components]
        out: list[Ranking] = []
        for qi, query in enumerate(queries):
            rankings = [pc[qi] for pc in per_component]
            if self.method == "rrf":
                fused = reciprocal_rank_fusion(rankings, k=self.rrf_k, weights=self.weights, top=depth)
            else:
                fused = minmax_fusion(rankings, weights=self.weights, top=depth)
            if self.metadata_policy != "none":
                meta = extract_query_metadata(query, self.lexicon)
                if self.metadata_policy == "boost":
                    fused = apply_metadata_boost(fused, self.corpus, meta, top=depth, **self.boost_kwargs)
                else:
                    fused = apply_metadata_filter(fused, self.corpus, meta, top=depth)
            out.append(fused[:top])
        return out


class MetadataRetriever:
    """Single component + metadata policy (no fusion). Used for metadata_filtered_bm25f."""

    def __init__(
        self,
        name: str,
        corpus: Corpus,
        component: Retriever,
        *,
        policy: str = "boost",
        candidate_depth: int = 200,
        boost_kwargs: dict | None = None,
    ) -> None:
        self.name = name
        self.corpus = corpus
        self.component = component
        self.policy = policy
        self.candidate_depth = candidate_depth
        self.boost_kwargs = boost_kwargs or {}
        self.lexicon = build_program_lexicon(corpus)
        self.build = getattr(component, "build", BuildInfo())

    def search_batch(self, queries: Sequence[str], top: int) -> list[Ranking]:
        depth = max(self.candidate_depth, top)
        base = self.component.search_batch(queries, depth)
        out: list[Ranking] = []
        for qi, query in enumerate(queries):
            meta = extract_query_metadata(query, self.lexicon)
            ranking = base[qi]
            if self.policy == "boost":
                ranking = apply_metadata_boost(ranking, self.corpus, meta, top=depth, **self.boost_kwargs)
            elif self.policy == "filter":
                ranking = apply_metadata_filter(ranking, self.corpus, meta, top=depth)
            out.append(ranking[:top])
        return out


# --- reranking -------------------------------------------------------------------------------


class RerankRetriever:
    """Cross-encoder reranking of a first-stage candidate set.

    A reranker can only reorder what it is given, so the runner also records the first-stage
    candidate recall at `candidate_depth` - the ceiling this retriever cannot exceed.
    """

    def __init__(
        self,
        name: str,
        corpus: Corpus,
        first_stage: Retriever,
        *,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        candidate_depth: int = 50,
        batch_size: int = 64,
    ) -> None:
        self.name = name
        self.corpus = corpus
        self.first_stage = first_stage
        self.model_name = model_name
        self.candidate_depth = candidate_depth
        self.batch_size = batch_size
        self._model = None
        self.build = getattr(first_stage, "build", BuildInfo())

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            from .dense import _resolve_device

            self._model = CrossEncoder(self.model_name, device=_resolve_device(), max_length=512)
        return self._model

    def search_batch(self, queries: Sequence[str], top: int) -> list[Ranking]:
        depth = max(self.candidate_depth, top)
        base = self.first_stage.search_batch(queries, depth)
        by_id = self.corpus.by_id
        out: list[Ranking] = []
        for qi, query in enumerate(queries):
            cands = base[qi][:depth]
            if not cands:
                out.append([])
                continue
            pairs = [(query, by_id[cid].text if cid in by_id else "") for cid, _ in cands]
            scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
            ranked = sorted(
                ((cid, float(s)) for (cid, _), s in zip(cands, scores)),
                key=lambda kv: (-kv[1], kv[0]),
            )
            out.append(ranked[:top])
        return out


# --- registry -----------------------------------------------------------------------------------

RetrieverFactory = Callable[[Corpus], Retriever]


def sparse_factories() -> dict[str, RetrieverFactory]:
    """Wave-1 sparse/structured configurations (no model downloads required)."""
    return {
        # --- mandatory baselines -------------------------------------------------------
        "bm25_original": lambda c: SparseRetriever(
            "bm25_original", c, tokenizer=tokenize_original, k1=1.5, b=0.75,
            subset_cap=PRODUCTION_SUBSET_CAP,
        ),
        "bm25_full_corpus": lambda c: SparseRetriever(
            "bm25_full_corpus", c, tokenizer=tokenize_original, k1=1.5, b=0.75,
        ),
        # --- candidates ----------------------------------------------------------------
        "bm25_v2tok": lambda c: SparseRetriever(
            "bm25_v2tok", c, tokenizer=tokenize_v2, k1=1.5, b=0.75,
        ),
        "contextual_bm25": lambda c: SparseRetriever(
            "contextual_bm25", c, tokenizer=tokenize_v2, k1=1.5, b=0.75, contextual=True,
        ),
        "bm25f": lambda c: SparseRetriever("bm25f", c, tokenizer=tokenize_v2, bm25f=True),
    }
