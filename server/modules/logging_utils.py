"""Efficiency-logging helpers (Section 6).

Provides small, dependency-free utilities for measuring and estimating the
per-query efficiency signals listed in CLAUDE.md Section 6.1:

    - latency_ms / retrieval_latency_ms / rerank_latency_ms / generation_latency_ms
      (via the `Timer` context manager / helper)
    - prompt_tokens_estimate / completion_tokens_estimate
      (via `estimate_prompt_tokens` / `estimate_completion_tokens`)
    - estimated_cost_usd
      (via `estimate_cost_usd`)

Everything here is a *deliberate estimate*, not an exact measurement:

    - We do not have an exact tokenizer for the Groq-hosted Llama models
      (no `tiktoken` dependency is installed for this project -- see
      CLAUDE.md Section 4: "tiktoken, optional (Section 6)" was never added).
      Token counts are approximated from character length using a simple
      chars-per-token heuristic (~4 characters/token), which is a commonly
      cited rough average for English text with BPE-style tokenizers. This
      is intentionally named "_estimate" everywhere so it is never confused
      with a real token count.
    - Costs are estimated from `ESTIMATED_INPUT_COST_PER_1K` /
      `ESTIMATED_OUTPUT_COST_PER_1K` in `modules.config` if those are
      defined there; otherwise we fall back to small documented per-1K-token
      USD defaults (see `_DEFAULT_INPUT_COST_PER_1K` / `_DEFAULT_OUTPUT_COST_PER_1K`
      below) so the function never raises just because the env var is unset.

This module is intentionally standalone: it must never import
`server.evaluation.run_evaluation` (or anything else that pulls in the full
RAG stack), so it can be unit-tested in isolation and reused from
`/ask`, the evaluation runner, and the ablation runner alike.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

# Rough average for English text tokenized with BPE-style tokenizers
# (GPT/Llama families typically land somewhere around 3.5-4.5 chars/token for
# English prose). 4.0 is a commonly cited midpoint estimate. This constant is
# deliberately simple and documented as an estimate, per CLAUDE.md Section 6.1
# ("If exact token counting is unavailable, use a simple estimate and clearly
# name it as an estimate").
_CHARS_PER_TOKEN_ESTIMATE = 4.0


def _estimate_tokens_from_text(text: Optional[str]) -> int:
    """Shared char-count-based token estimator.

    Returns 0 for falsy/empty input. Always rounds up (ceil-like via integer
    division trick) so that any non-empty text is counted as at least 1 token.
    """
    if not text:
        return 0
    char_count = len(text)
    if char_count <= 0:
        return 0
    estimated = math.ceil(char_count / _CHARS_PER_TOKEN_ESTIMATE)
    # Ensure non-empty text never rounds down to 0 tokens.
    return max(estimated, 1)


def estimate_prompt_tokens(text: str) -> int:
    """Estimate the number of prompt tokens for `text`.

    This is a character-length-based ESTIMATE (see module docstring), not an
    exact tokenizer count. Suitable for rough efficiency comparisons across
    retrieval modes / prompt strategies (e.g. comparing `basic` vs `few_shot`
    vs `expert_routed` prompt lengths), not for billing-accurate reporting.

    Args:
        text: The full prompt text that would be sent to the LLM (e.g. the
            system prompt + source-labeled context + question).

    Returns:
        Estimated token count (>= 0). Empty/None input returns 0.
    """
    return _estimate_tokens_from_text(text)


def estimate_completion_tokens(text: str) -> int:
    """Estimate the number of completion tokens for `text`.

    Same character-length-based ESTIMATE as `estimate_prompt_tokens`, applied
    to the model's generated answer text.

    Args:
        text: The generated answer text returned by the LLM.

    Returns:
        Estimated token count (>= 0). Empty/None input returns 0.
    """
    return _estimate_tokens_from_text(text)


# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------

# Documented fallback defaults (USD per 1K tokens), used only when
# `modules.config` does not define ESTIMATED_INPUT_COST_PER_1K /
# ESTIMATED_OUTPUT_COST_PER_1K (they are listed in CLAUDE.md Section 8 as
# "Future variables to add only when needed" and have not been added yet).
#
# These numbers are deliberately small, round, ballpark figures in the
# neighborhood of published Groq/Llama-3.3-70b per-token pricing at the time
# of writing -- they exist so `estimate_cost_usd` always returns a sensible,
# clearly-labeled-as-an-estimate number instead of raising or returning None.
# They are NOT a guarantee of actual billed cost. Override via
# ESTIMATED_INPUT_COST_PER_1K / ESTIMATED_OUTPUT_COST_PER_1K in the
# environment (and, when Section 6 wiring lands in config.py, in
# `modules.config`) for a more accurate per-deployment estimate.
_DEFAULT_INPUT_COST_PER_1K = 0.00059
_DEFAULT_OUTPUT_COST_PER_1K = 0.00079


def _resolve_cost_per_1k() -> tuple[float, float]:
    """Resolve (input_cost_per_1k, output_cost_per_1k) in USD.

    Prefers `ESTIMATED_INPUT_COST_PER_1K` / `ESTIMATED_OUTPUT_COST_PER_1K`
    from `modules.config` when present and not None; otherwise falls back to
    the documented defaults above. Imported lazily (inside the function) so
    that this module never fails to import just because `modules.config`
    changes shape, and so unit tests can run this file standalone without a
    fully configured environment.
    """
    input_cost = _DEFAULT_INPUT_COST_PER_1K
    output_cost = _DEFAULT_OUTPUT_COST_PER_1K
    try:
        from modules import config as _config  # local import: keep this module standalone

        configured_input = getattr(_config, "ESTIMATED_INPUT_COST_PER_1K", None)
        configured_output = getattr(_config, "ESTIMATED_OUTPUT_COST_PER_1K", None)
        if configured_input is not None:
            input_cost = float(configured_input)
        if configured_output is not None:
            output_cost = float(configured_output)
    except Exception:
        # Config import failing (e.g. missing .env, partial test environment)
        # must never break cost estimation -- fall back to documented defaults.
        pass
    return input_cost, output_cost


def estimate_cost_usd(
    prompt_tokens: int,
    completion_tokens: int,
    model_name: Optional[str] = None,
) -> float:
    """Estimate USD cost for one LLM call from estimated token counts.

    This is an ESTIMATE built on top of already-estimated token counts (see
    `estimate_prompt_tokens` / `estimate_completion_tokens`), multiplied by
    either:

      1. `ESTIMATED_INPUT_COST_PER_1K` / `ESTIMATED_OUTPUT_COST_PER_1K` from
         `modules.config`, if defined there, or
      2. small documented per-1K-token USD defaults (see
         `_DEFAULT_INPUT_COST_PER_1K` / `_DEFAULT_OUTPUT_COST_PER_1K`).

    `model_name` is accepted for forward compatibility (e.g. a future
    per-model pricing table) and is currently included only for clarity in
    logs/derived tables -- it does not change the computed estimate. Passing
    None is fine.

    Args:
        prompt_tokens: Estimated prompt token count (>= 0).
        completion_tokens: Estimated completion token count (>= 0).
        model_name: Optional model identifier (e.g. "llama-3.3-70b-versatile").
            Reserved for future per-model pricing; currently informational.

    Returns:
        Estimated cost in USD, rounded to 6 decimal places. Always >= 0.0.
    """
    prompt_tokens = max(int(prompt_tokens or 0), 0)
    completion_tokens = max(int(completion_tokens or 0), 0)

    input_cost_per_1k, output_cost_per_1k = _resolve_cost_per_1k()

    cost = (prompt_tokens / 1000.0) * input_cost_per_1k
    cost += (completion_tokens / 1000.0) * output_cost_per_1k

    # `model_name` is intentionally unused beyond documentation/logging hooks
    # for now -- see docstring. Referenced here only so linters/tests can
    # confirm the parameter is accepted without affecting behavior.
    _ = model_name

    return round(cost, 6)


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


@dataclass
class Timer:
    """Lightweight stage-latency stopwatch for efficiency logging.

    Usage as a context manager (records `elapsed_ms` on exit):

        with Timer() as retrieval_timer:
            results = run_retrieval_mode(...)
        retrieval_latency_ms = retrieval_timer.elapsed_ms

    Usage as a manual stopwatch (e.g. across non-contiguous code regions):

        timer = Timer()
        timer.start()
        ... do work ...
        latency_ms = timer.stop()   # also stored in timer.elapsed_ms

    All timings use `time.perf_counter()`, the standard high-resolution
    monotonic clock recommended for measuring short durations in Python; it
    is immune to wall-clock adjustments (NTP sync, DST, etc).

    Attributes:
        elapsed_ms: Elapsed time in milliseconds, rounded to 2 decimals.
            `None` until the timer has been stopped (or the `with` block has
            exited) at least once.
    """

    _start: Optional[float] = field(default=None, repr=False, compare=False)
    elapsed_ms: Optional[float] = None

    def start(self) -> "Timer":
        """Start (or restart) the stopwatch. Returns self for chaining."""
        self._start = time.perf_counter()
        self.elapsed_ms = None
        return self

    def stop(self) -> float:
        """Stop the stopwatch and return the elapsed milliseconds.

        Raises:
            RuntimeError: if `start()` (or entering the `with` block) was
                never called -- calling `stop()` on a never-started Timer is
                almost certainly a logging bug, so we surface it loudly
                rather than silently returning 0.0 (per CLAUDE.md Section 17:
                "Do not hide errors silently").
        """
        if self._start is None:
            raise RuntimeError("Timer.stop() called before Timer.start() (or `with Timer()`).")
        elapsed_seconds = time.perf_counter() - self._start
        self.elapsed_ms = round(elapsed_seconds * 1000.0, 2)
        return self.elapsed_ms

    def __enter__(self) -> "Timer":
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        # Always record elapsed time, even if the block raised -- partial
        # latency information is still useful for diagnosing slow failures.
        self.stop()
        return None


def time_block() -> Timer:
    """Convenience factory: `with time_block() as t: ...; t.elapsed_ms`.

    Equivalent to `Timer()` but reads slightly more naturally at call sites
    that only ever use the context-manager form, e.g.:

        with time_block() as gen_timer:
            answer = generate(...)
        generation_latency_ms = gen_timer.elapsed_ms
    """
    return Timer()


# ---------------------------------------------------------------------------
# Self-test (run directly: `python -m modules.logging_utils` from `server/`)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Minimal standalone smoke test -- intentionally avoids any test framework
    # so this file can be sanity-checked with a bare `python` invocation in
    # constrained environments. Exercises every public symbol in this module.
    import sys

    failures: list[str] = []

    def _check(label: str, condition: bool) -> None:
        status = "ok" if condition else "FAIL"
        print(f"[{status}] {label}")
        if not condition:
            failures.append(label)

    # --- token estimation -------------------------------------------------
    _check("estimate_prompt_tokens('') == 0", estimate_prompt_tokens("") == 0)
    _check("estimate_prompt_tokens(None) == 0", estimate_prompt_tokens(None) == 0)  # type: ignore[arg-type]
    _check("estimate_prompt_tokens('hi') >= 1", estimate_prompt_tokens("hi") >= 1)
    sample_text = "x" * 400  # 400 chars / 4.0 chars-per-token => ~100 tokens
    sample_estimate = estimate_prompt_tokens(sample_text)
    _check(
        f"estimate_prompt_tokens('x'*400) ~= 100 (got {sample_estimate})",
        90 <= sample_estimate <= 110,
    )
    _check(
        "estimate_completion_tokens mirrors estimate_prompt_tokens",
        estimate_completion_tokens(sample_text) == estimate_prompt_tokens(sample_text),
    )
    longer_estimate = estimate_prompt_tokens(sample_text * 2)
    _check(
        "longer text => >= tokens (monotonic)",
        longer_estimate >= sample_estimate,
    )

    # --- cost estimation ---------------------------------------------------
    zero_cost = estimate_cost_usd(0, 0, "llama-3.3-70b-versatile")
    _check(f"estimate_cost_usd(0, 0, ...) == 0.0 (got {zero_cost})", zero_cost == 0.0)

    cost_1k_in_only = estimate_cost_usd(1000, 0)
    input_per_1k, output_per_1k = _resolve_cost_per_1k()
    _check(
        f"estimate_cost_usd(1000, 0) == input_cost_per_1k (got {cost_1k_in_only} vs {input_per_1k})",
        abs(cost_1k_in_only - input_per_1k) < 1e-9,
    )

    cost_1k_out_only = estimate_cost_usd(0, 1000)
    _check(
        f"estimate_cost_usd(0, 1000) == output_cost_per_1k (got {cost_1k_out_only} vs {output_per_1k})",
        abs(cost_1k_out_only - output_per_1k) < 1e-9,
    )

    combined = estimate_cost_usd(1000, 1000)
    _check(
        f"estimate_cost_usd(1000, 1000) == input + output per-1k (got {combined})",
        abs(combined - (input_per_1k + output_per_1k)) < 1e-9,
    )
    _check("estimate_cost_usd never returns negative", estimate_cost_usd(5, 5) >= 0.0)
    _check(
        "estimate_cost_usd accepts model_name=None",
        estimate_cost_usd(10, 10, None) >= 0.0,
    )

    # --- Timer (context-manager form) -------------------------------------
    with Timer() as t:
        time.sleep(0.01)
    _check(
        f"Timer (with-block) records elapsed_ms ~= 10ms (got {t.elapsed_ms})",
        t.elapsed_ms is not None and t.elapsed_ms >= 8.0,
    )

    # --- Timer (manual start/stop form) ------------------------------------
    manual_timer = Timer()
    manual_timer.start()
    time.sleep(0.01)
    manual_elapsed = manual_timer.stop()
    _check(
        f"Timer.start()/stop() returns elapsed_ms ~= 10ms (got {manual_elapsed})",
        manual_elapsed >= 8.0 and manual_timer.elapsed_ms == manual_elapsed,
    )

    # --- Timer (misuse: stop() before start()) -----------------------------
    raised = False
    try:
        Timer().stop()
    except RuntimeError:
        raised = True
    _check("Timer().stop() without start() raises RuntimeError", raised)

    # --- Timer (exception inside with-block still records elapsed) ---------
    boom_timer = Timer()
    try:
        with boom_timer:
            time.sleep(0.005)
            raise ValueError("boom")
    except ValueError:
        pass
    _check(
        "Timer records elapsed_ms even when the with-block raises",
        boom_timer.elapsed_ms is not None and boom_timer.elapsed_ms >= 3.0,
    )

    # --- time_block() convenience factory -----------------------------------
    with time_block() as tb:
        time.sleep(0.005)
    _check(
        "time_block() behaves like Timer()",
        tb.elapsed_ms is not None and tb.elapsed_ms >= 3.0,
    )

    print()
    if failures:
        print(f"{len(failures)} check(s) FAILED: {failures}")
        sys.exit(1)
    else:
        print("All logging_utils self-checks passed.")
