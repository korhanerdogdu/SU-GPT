from __future__ import annotations

"""
Loads the adviSU curriculum corpus once, into a form every retriever in the lab shares.

The corpus on disk (data/degree_requirements/**, data/minors/**) is already one chunk per
JSONL line, with a stable `chunk_id` and flat metadata. That is exactly what
server/scripts/ingest_degree_requirements.py pushes into Chroma, so loading the same files
here means the benchmark ranks the same units the production index stores - no re-chunking,
no drift between what we measure and what ships.

Everything downstream (BM25, BM25F, dense, fusion) indexes `Chunk` objects from here, so a
single corpus fingerprint identifies every experiment run against it.
"""

import hashlib
import json
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any, Iterable

SERVER_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SERVER_ROOT.parent
DATA_DIR = PROJECT_ROOT / "data"


@dataclass(slots=True)
class Chunk:
    """One retrievable unit. `text` is the body; everything else is structured metadata."""

    chunk_id: str
    text: str
    meta: dict[str, Any] = field(default_factory=dict)

    # --- convenience accessors used by field-weighted / metadata-aware retrievers ----
    @property
    def program(self) -> str:
        return str(self.meta.get("program") or "")

    @property
    def curriculum_term(self) -> str:
        return str(self.meta.get("curriculum_term") or "")

    @property
    def course_id(self) -> str:
        return str(self.meta.get("course_id") or "")

    @property
    def course_title(self) -> str:
        return str(self.meta.get("course_title") or "")

    @property
    def requirement_category(self) -> str:
        return str(self.meta.get("requirement_category") or "")

    @property
    def document_type(self) -> str:
        return str(self.meta.get("document_type") or "")

    @property
    def data_role(self) -> str:
        return str(self.meta.get("data_role") or "")

    @property
    def source_document(self) -> str:
        return str(self.meta.get("source_document") or "")

    @property
    def is_minor(self) -> bool:
        return self.data_role == "minor_requirement"


@dataclass
class Corpus:
    chunks: list[Chunk]

    def __len__(self) -> int:
        return len(self.chunks)

    def __iter__(self) -> Iterable[Chunk]:
        return iter(self.chunks)

    @cached_property
    def by_id(self) -> dict[str, Chunk]:
        return {c.chunk_id: c for c in self.chunks}

    @cached_property
    def index_of(self) -> dict[str, int]:
        return {c.chunk_id: i for i, c in enumerate(self.chunks)}

    @cached_property
    def fingerprint(self) -> str:
        """Stable hash of (ids + text). Any corpus edit invalidates every cached artifact."""
        h = hashlib.sha256()
        for c in self.chunks:
            h.update(c.chunk_id.encode())
            h.update(b"\x00")
            h.update(c.text.encode())
            h.update(b"\x01")
        return h.hexdigest()[:16]

    @cached_property
    def programs(self) -> set[str]:
        return {c.program for c in self.chunks if c.program}

    @cached_property
    def terms(self) -> set[str]:
        return {c.curriculum_term for c in self.chunks if c.curriculum_term}

    @cached_property
    def program_names(self) -> dict[str, str]:
        """program code -> human program name, for query metadata extraction."""
        out: dict[str, str] = {}
        for c in self.chunks:
            name = str(c.meta.get("program_name") or "")
            if c.program and name:
                out.setdefault(c.program, name)
        return out


def load_corpus(data_dir: Path | str | None = None) -> Corpus:
    """Read every degree-requirement / minor chunk, in a deterministic (sorted) order."""
    root = Path(data_dir) if data_dir else DATA_DIR
    roots = [root / "degree_requirements", root / "minors"]
    files = sorted(p for r in roots if r.is_dir() for p in r.rglob("*.jsonl"))
    chunks: list[Chunk] = []
    seen: set[str] = set()
    for path in files:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text, chunk_id = row.get("text"), row.get("chunk_id")
            if not text or not chunk_id:
                continue
            chunk_id = str(chunk_id)
            if chunk_id in seen:  # defensive: duplicate ids would corrupt gold matching
                continue
            seen.add(chunk_id)
            meta = {k: v for k, v in row.items() if k != "text"}
            meta.setdefault("source", meta.get("source_document") or path.name)
            _add_chroma_aliases(meta)
            chunks.append(Chunk(chunk_id=chunk_id, text=str(text), meta=meta))
    return Corpus(chunks=chunks)


# data_role -> coarse documentType, mirroring scripts/ingest_degree_requirements.py.
_DOCUMENT_TYPE_ALIAS = {
    "curriculum_requirement": "course",
    "minor_requirement": "minor",
}


def _add_chroma_aliases(meta: dict[str, Any]) -> None:
    """Add the compatibility keys the ingester writes into Chroma.

    The raw JSONL rows do not carry `documentType` or `term_code`; those are synthesized by
    scripts/ingest_degree_requirements.py when it upserts into Chroma. `rag_router` builds its
    filters as `{"documentType": ...}`, so a corpus loaded straight from JSONL would match
    zero chunks for every routed query - retrieval would silently return nothing rather than
    fail. Mirroring the aliases here keeps the in-memory corpus filterable by exactly the same
    `where` clauses as the Chroma collection.
    """
    role = str(meta.get("data_role") or "")
    meta.setdefault("documentType", _DOCUMENT_TYPE_ALIAS.get(role, "course"))
    if meta.get("curriculum_term"):
        meta.setdefault("term_code", meta["curriculum_term"])
