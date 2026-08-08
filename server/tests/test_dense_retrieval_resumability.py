from __future__ import annotations

"""Regression: a cold dense-embedding build over the full corpus can take tens of minutes, and
previously the only ``np.save`` happened after the *entire* loop finished -- any interruption
(container restart, OOM, deploy) lost 100% of the work, and the next attempt started at batch 0
again. This is why re-enabling dense retrieval kept costing another full 20-30 minute build every
time the earlier attempts got interrupted. DenseIndex._load_or_build now checkpoints to disk
after every batch and resumes from the last completed one.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retrieval_lab.corpus import Chunk, Corpus
from retrieval_lab.dense import DenseConfig, DenseIndex


def _corpus(n: int) -> Corpus:
    return Corpus(chunks=[Chunk(chunk_id=f"c{i}", text=f"chunk body {i}") for i in range(n)])


def _tiny_config(name: str = "test-model", batch_size: int = 2) -> DenseConfig:
    return DenseConfig(name=name, model_name="unused", batch_size=batch_size)


def test_interrupted_build_resumes_instead_of_restarting_from_batch_zero(tmp_path, monkeypatch):
    # step = max(batch_size * 16, 256), so the corpus must exceed 256 items to span more than
    # one batch -- otherwise there is nothing mid-build to interrupt.
    corpus = _corpus(600)
    cfg = _tiny_config()
    encode_calls: list[int] = []

    def flaky_encode(self, texts, prefix):
        encode_calls.append(len(texts))
        if len(encode_calls) == 2:
            raise RuntimeError("simulated interruption (container restart mid-build)")
        return np.ones((len(texts), 4), dtype=np.float32) * len(encode_calls)

    monkeypatch.setattr(DenseIndex, "_encode", flaky_encode)

    # First attempt: fails partway through. Interrupted build must not raise on the exception
    # escaping __init__ without ever having written the (correct) final cache file.
    try:
        DenseIndex(cfg, corpus, cache_dir=tmp_path)
    except RuntimeError:
        pass

    checkpoint_path = (tmp_path / f"{cfg.name}__{corpus.fingerprint}.npy").with_suffix(".partial.npy")
    checkpoint_meta_path = checkpoint_path.with_suffix(".json")
    assert checkpoint_path.exists(), "first batch's progress must survive the interruption"
    assert checkpoint_meta_path.exists()
    calls_before_resume = len(encode_calls)
    assert calls_before_resume == 2  # one successful batch, one that raised

    # Second attempt (a fresh process would call this the same way after a restart): must pick
    # up from the checkpoint, not re-encode batch 0 again.
    index = DenseIndex(cfg, corpus, cache_dir=tmp_path)
    calls_during_resume = len(encode_calls) - calls_before_resume
    # 600 items with step=256 spans 3 batches; the point under test is that the resumed run
    # only redoes the failed batch, not the one that already succeeded before the interruption.
    assert index.embeddings.shape[0] == 600
    assert calls_during_resume <= 2

    final_cache_path = tmp_path / f"{cfg.name}__{corpus.fingerprint}.npy"
    assert final_cache_path.exists()
    assert not checkpoint_path.exists(), "checkpoint must be cleaned up once the build completes"
    assert not checkpoint_meta_path.exists()


def test_completed_build_is_reused_without_any_encode_calls(tmp_path, monkeypatch):
    corpus = _corpus(5)
    cfg = _tiny_config()
    calls = []
    monkeypatch.setattr(
        DenseIndex, "_encode",
        lambda self, texts, prefix: (calls.append(len(texts)), np.zeros((len(texts), 4), dtype=np.float32))[1],
    )
    DenseIndex(cfg, corpus, cache_dir=tmp_path)
    assert len(calls) == 1
    DenseIndex(cfg, corpus, cache_dir=tmp_path)
    assert len(calls) == 1  # second construction must hit the on-disk cache, not re-encode
