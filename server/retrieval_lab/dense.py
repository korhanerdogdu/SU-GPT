from __future__ import annotations

"""
Dense retrieval with a persistent embedding cache.

Corpus embeddings are the expensive part of every dense experiment (30,343 chunks per
model), so they are computed once and memory-mapped from
outputs/retrieval_cache/<model>__<corpus_fingerprint>.npy. The fingerprint is a hash of every
chunk id + body, so editing the corpus invalidates the cache automatically instead of
silently scoring stale vectors - you cannot accidentally benchmark against an old index.

Model-specific input formatting is handled here rather than by the caller, because getting it
wrong is a silent accuracy loss, not an error:
  * E5 models require the "query: " / "passage: " prefixes; without them retrieval quality
    drops noticeably and the drop looks like "dense retrieval is bad for this corpus".
  * E5 *-instruct* variants want a one-line task instruction on the query side only.
  * BGE-M3 wants no prefix at all.
Similarity is cosine throughout (vectors are L2-normalized at encode time), so scores are
comparable across models and safe to feed to rank fusion.
"""

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .corpus import Corpus
from .sparse import contextual_text

# See corpus.py's SERVER_ROOT/DATA_DIR comment: a production container build (server/Dockerfile)
# flattens server/ directly into /app, so a hardcoded parents[2] climb lands on "/" in the image
# instead of the project root, and writing the cache there fails with a permission error that
# silently degrades every request to BM25F-only retrieval. RETRIEVAL_CACHE_DIR lets it be pinned
# explicitly (e.g. in docker-compose.yml) the same way DEGREE_DATA_DIR pins the corpus dir.
_SERVER_ROOT = Path(__file__).resolve().parents[1]
_PROJECT_ROOT = _SERVER_ROOT.parent if _SERVER_ROOT.name == "server" else _SERVER_ROOT
CACHE_DIR = Path(os.getenv("RETRIEVAL_CACHE_DIR", str(_PROJECT_ROOT / "outputs" / "retrieval_cache")))

# Query-side instruction for the E5 instruct family.
E5_INSTRUCT_TASK = (
    "Instruct: Given a university curriculum question, retrieve the curriculum "
    "requirement record that answers it\nQuery: "
)


@dataclass
class DenseConfig:
    name: str
    model_name: str
    query_prefix: str = ""
    passage_prefix: str = ""
    normalize: bool = True
    batch_size: int = 64
    max_seq_length: int | None = None
    contextual: bool = False  # embed the metadata-prefixed body instead of the raw body


# Registry of dense models. Sizes are the practical reason for the smaller defaults:
# this runs on an Apple-silicon laptop with ~13 GiB free disk, so the large variants are
# opt-in rather than automatic.
DENSE_CONFIGS: dict[str, DenseConfig] = {
    "minilm": DenseConfig(
        name="minilm",
        model_name="sentence-transformers/all-MiniLM-L12-v2",
    ),
    "e5_small": DenseConfig(
        name="e5_small",
        model_name="intfloat/multilingual-e5-small",
        query_prefix="query: ",
        passage_prefix="passage: ",
    ),
    "e5_base": DenseConfig(
        name="e5_base",
        model_name="intfloat/multilingual-e5-base",
        query_prefix="query: ",
        passage_prefix="passage: ",
    ),
    "e5_large_instruct": DenseConfig(
        name="e5_large_instruct",
        model_name="intfloat/multilingual-e5-large-instruct",
        query_prefix=E5_INSTRUCT_TASK,
        passage_prefix="",
        batch_size=32,
    ),
    "bge_m3": DenseConfig(
        name="bge_m3",
        model_name="BAAI/bge-m3",
        batch_size=16,
        max_seq_length=512,
    ),
    # Same model as `e5_small`, but embedding the metadata-prefixed view of each chunk.
    "e5_small_contextual": DenseConfig(
        name="e5_small_contextual",
        model_name="intfloat/multilingual-e5-small",
        query_prefix="query: ",
        passage_prefix="passage: ",
        contextual=True,
    ),
}


def _resolve_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class DenseIndex:
    """Cosine-similarity dense index over the corpus, backed by a cached embedding matrix."""

    def __init__(self, cfg: DenseConfig, corpus: Corpus, *, cache_dir: Path | None = None) -> None:
        self.cfg = cfg
        self.ids = [c.chunk_id for c in corpus.chunks]
        self.cache_dir = cache_dir or CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        slug = cfg.name
        self.cache_path = self.cache_dir / f"{slug}__{corpus.fingerprint}.npy"
        self._model = None
        self.embeddings = self._load_or_build(corpus)

    # --- model ------------------------------------------------------------------------
    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.cfg.model_name, device=_resolve_device())
            if self.cfg.max_seq_length:
                self._model.max_seq_length = self.cfg.max_seq_length
        return self._model

    def _encode(self, texts: Sequence[str], prefix: str) -> np.ndarray:
        payload = [prefix + t for t in texts] if prefix else list(texts)
        vecs = self.model.encode(
            payload,
            batch_size=self.cfg.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.cfg.normalize,
            show_progress_bar=False,
        )
        return np.asarray(vecs, dtype=np.float32)

    # --- corpus embeddings -------------------------------------------------------------
    def _load_or_build(self, corpus: Corpus) -> np.ndarray:
        if self.cache_path.exists():
            arr = np.load(self.cache_path, mmap_mode="r")
            if arr.shape[0] == len(self.ids):
                return arr
            # Shape mismatch means the cache predates a corpus change; rebuild rather than
            # score against vectors that no longer line up with the ids.
            self.cache_path.unlink()

        texts = [
            contextual_text(c) if self.cfg.contextual else c.text
            for c in corpus.chunks
        ]
        step = max(self.cfg.batch_size * 16, 256)

        # A cold build over the full corpus can take tens of minutes; previously the only
        # np.save() happened after the entire loop finished, so any interruption (container
        # restart, OOM, deploy) lost 100% of the work and the next attempt started at batch 0
        # again. This makes the build resumable: progress is checkpointed to disk after every
        # batch, and a restart picks up from the last completed batch instead of from scratch.
        checkpoint_path = self.cache_path.with_suffix(".partial.npy")
        checkpoint_meta_path = self.cache_path.with_suffix(".partial.json")
        start_index = 0
        chunks: list[np.ndarray] = []
        if checkpoint_path.exists() and checkpoint_meta_path.exists():
            try:
                meta = json.loads(checkpoint_meta_path.read_text(encoding="utf-8"))
                if meta.get("total") == len(texts) and meta.get("model") == self.cfg.name:
                    chunks = [np.load(checkpoint_path)]
                    start_index = int(meta["completed"])
            except (OSError, ValueError, json.JSONDecodeError, KeyError):
                start_index = 0
                chunks = []

        started = time.perf_counter()
        for i in range(start_index, len(texts), step):
            chunks.append(self._encode(texts[i:i + step], self.cfg.passage_prefix))
            completed = min(i + step, len(texts))
            np.save(checkpoint_path, np.vstack(chunks))
            checkpoint_meta_path.write_text(
                json.dumps({"total": len(texts), "completed": completed, "model": self.cfg.name}),
                encoding="utf-8",
            )
        out = np.vstack(chunks) if chunks else np.empty((0, 0), dtype=np.float32)
        self.build_seconds = time.perf_counter() - started
        np.save(self.cache_path, out)
        checkpoint_path.unlink(missing_ok=True)
        checkpoint_meta_path.unlink(missing_ok=True)
        return out

    # --- search -------------------------------------------------------------------------
    def search(
        self, query: str, top: int = 50, allowed: set[int] | None = None
    ) -> list[tuple[str, float]]:
        """`allowed` pre-filters to a metadata scope by masking out-of-scope similarities.

        Masking before the top-k selection (rather than filtering the top-k afterwards) is
        what makes a narrow scope still return its best `top` documents instead of whatever
        survives from the global top-k - the same reason BM25Index.search takes `allowed`.
        """
        q = self._encode([query], self.cfg.query_prefix)[0]
        sims = np.asarray(self.embeddings) @ q
        if allowed is not None:
            if not allowed:
                return []
            mask = np.full(sims.shape[0], -np.inf, dtype=np.float32)
            idx_allowed = np.fromiter(allowed, dtype=np.int64, count=len(allowed))
            mask[idx_allowed] = 0.0
            sims = sims + mask
        k = min(top, sims.shape[0])
        idx = np.argpartition(-sims, k - 1)[:k]
        idx = idx[np.argsort(-sims[idx])]
        return [(self.ids[i], float(sims[i])) for i in idx if np.isfinite(sims[i])]

    def search_many(self, queries: Sequence[str], top: int = 50) -> list[list[tuple[str, float]]]:
        """Batch-encode queries: one forward pass for the whole benchmark split."""
        qs = self._encode(list(queries), self.cfg.query_prefix)
        emb = np.asarray(self.embeddings)
        out: list[list[tuple[str, float]]] = []
        for row in qs:
            sims = emb @ row
            k = min(top, sims.shape[0])
            idx = np.argpartition(-sims, k - 1)[:k]
            idx = idx[np.argsort(-sims[idx])]
            out.append([(self.ids[i], float(sims[i])) for i in idx])
        return out
