"""Provider-independent paired benchmark harness.

The harness accepts injected provider and scorer objects; it never imports Groq, OpenRouter or the
application's legacy evaluation runner.  Output rows deliberately omit prompts, response text,
system messages and reasoning details.  Only item IDs, aggregate scores, safe error classes,
latency, provider-reported usage/cost and response digests are persisted.

This file is also a manifest validator::

    python server/evaluation/provider_benchmark.py \
        --manifest data/benchmark/provider_eval/final_manifest.json --validate-only

The checked-in final manifest is intentionally unprovisioned.  Validation therefore returns a
BLOCKED verdict until an independently authored, sealed set with >=100 Turkish and >=100 English
items is supplied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

try:
    from .provider_stats import analyze_paired_results, default_switch_verdict
except ImportError:  # direct script execution from server/evaluation
    from provider_stats import analyze_paired_results, default_switch_verdict


REPO_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_FINAL_MANIFEST = REPO_ROOT / "data/benchmark/provider_eval/final_manifest.json"
ALLOWED_LANGUAGES = frozenset({"tr", "en"})
ALLOWED_CATEGORIES = frozenset({"quality", "safety", "adversarial", "planner"})
FINAL_MIN_PER_LANGUAGE = 100
BASE_COMMIT = "5a6949c1e2e5b0603d8c1c88820f590099000f34"
BASELINE_MODEL = "llama-3.3-70b-versatile"
CANDIDATE_MODEL = "deepseek/deepseek-v4-pro"
FINAL_PROTOCOL_ID = "advisu-openrouter-deepseek-v4-pro-prereg-v2"
FINAL_ORDER_SEED = 20260804
FINAL_REPETITIONS = 3
FINAL_WARMUPS_PER_PROVIDER = 2
FINAL_CONFIGURATION = {
    "protocol_id": FINAL_PROTOCOL_ID,
    "registered_base_commit": BASE_COMMIT,
    "run_ordinal": 1,
    "retrieval_mode": "hybrid_meta",
    "retrieval_first_stage_depth": 200,
    "retrieval_provider_context_cutoff": 20,
    "final_context_top_k": 6,
    "ranker_id": "bm25f_e5_small_rrf_k60_metadata_boost",
    "reranker_id": None,
    "contexts_frozen_before_provider_calls": True,
    "prompt_strategy": "structured_lookup",
    "application_prompt_template_sha256": "29bae5340a76c1291005477132a55053d177a27ee8f58a462a7fbbc29d7028c3",
    "strategy_directive_sha256": "15a27de0228dc32279acde9e6731d5ceb16a7aa54a4efdf0f28de47323ce9cef",
    "language_directive_sha256": {
        "tr": "68f7a37eda460c9958b44ed348b5445fc6f6baf411ddb595508729be5acdb404",
        "en": "d21b1a7294aa05d34288fcd7d4d0d1807988997c66a5832b8c7c9e41b51df44c",
    },
    "reasoning_enabled": False,
    "temperature": 0.0,
    "maximum_output_tokens": 1024,
    "request_timeout_seconds": 60,
    "maximum_retries": 2,
    "fallback_provider": None,
}
FINAL_PROVIDER_CONFIGURATION = {
    key: FINAL_CONFIGURATION[key]
    for key in (
        "reasoning_enabled",
        "temperature",
        "maximum_output_tokens",
        "request_timeout_seconds",
        "maximum_retries",
        "fallback_provider",
    )
}
CANONICAL_EXTERNAL_GATES = (
    "retrieval_hit_at_10_non_regression",
    "retrieval_all_gold_at_10_non_regression",
    "security_noninferiority",
    "confidence_calibration",
    "end_to_end",
    "independent_blind_evaluation",
)
CANONICAL_OWNER_THRESHOLDS = (
    "latency_p95_limit_ms",
    "cost_per_query_limit_usd",
    "api_error_rate_noninferiority_margin",
)


@dataclass(frozen=True)
class EvaluationItem:
    item_id: str
    prompt: str
    language: str
    category: str = "quality"
    cluster_id: str | None = None
    scoring_data: Mapping[str, Any] = field(default_factory=dict)
    context_sha256: str | None = None
    provider_prompt_materialized: bool = False
    frozen_context: Any | None = None


@dataclass(frozen=True)
class ProviderResponse:
    """Normalized response returned by an injected provider adapter.

    ``streaming_mode`` must be ``native``, ``fake`` or ``none``.  Fake streaming is the current
    application behaviour and is never accepted as TTFT evidence.
    """

    status: str
    text: str = ""
    streaming_mode: str = "none"
    ttft_ms: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    cost_usd: float | None = None
    error_type: str | None = None


class Provider(Protocol):
    name: str
    model_id: str
    resolved_configuration: Mapping[str, Any]

    def generate(self, prompt: str, *, item_id: str, language: str) -> ProviderResponse:
        ...


class Scorer(Protocol):
    metric_names: Sequence[str]

    def score(self, item: EvaluationItem, response_text: str) -> Mapping[str, float]:
        ...


@dataclass(frozen=True)
class StopPolicy:
    max_consecutive_failures: int = 3
    max_failure_rate: float = 0.25
    min_items_before_rate_stop: int = 20
    # Optional operator safety fuse, not a product acceptance threshold.  There is deliberately no
    # default: the repository does not define a cost budget and evaluation code must not invent one.
    max_total_cost_usd: float | None = None


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_error_type(exc: BaseException) -> str:
    """Persist only the exception class, never a provider payload or secret-bearing message."""

    return type(exc).__name__[:120]


def _safe_error_label(value: str | None) -> str | None:
    if not value:
        return None
    clean = re.sub(r"[^A-Za-z0-9_.:-]+", "_", str(value)).strip("_")
    return clean[:120] or "provider_error"


def _coerce_item(row: Mapping[str, Any], *, default_category: str = "quality") -> EvaluationItem:
    item_id = str(row.get("id") or row.get("item_id") or "").strip()
    materialized_prompt = str(row.get("provider_prompt") or "").strip()
    prompt = materialized_prompt or str(row.get("prompt") or row.get("question") or "").strip()
    language = str(row.get("language") or "").strip().lower()
    category = str(row.get("category") or default_category).strip().lower()
    cluster_id = str(row.get("cluster_id") or item_id).strip()
    if not item_id:
        raise ValueError("evaluation item is missing id")
    if not prompt:
        raise ValueError(f"evaluation item {item_id} is missing prompt/question")
    if language not in ALLOWED_LANGUAGES:
        raise ValueError(f"evaluation item {item_id} has unsupported language {language!r}")
    if category not in ALLOWED_CATEGORIES:
        raise ValueError(f"evaluation item {item_id} has unsupported category {category!r}")
    scoring_data = {
        key: value
        for key, value in row.items()
        if key not in {
            "id", "item_id", "prompt", "provider_prompt", "question", "language", "category",
            "cluster_id", "context_sha256", "frozen_context",
        }
    }
    return EvaluationItem(
        item_id,
        prompt,
        language,
        category,
        cluster_id,
        scoring_data,
        str(row.get("context_sha256") or "").strip() or None,
        bool(materialized_prompt),
        row.get("frozen_context"),
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
    return rows


def _resolve_manifest_path(manifest_path: str | Path, relative_path: str) -> Path:
    manifest = Path(manifest_path).expanduser().resolve()
    resolved = (manifest.parent / relative_path).resolve()
    if not (resolved.is_relative_to(manifest.parent) or resolved.is_relative_to(REPO_ROOT)):
        raise ValueError("manifest path escapes the allowed repository/dataset boundary")
    return resolved


def load_items_from_manifest(manifest_path: str | Path) -> tuple[dict[str, Any], list[EvaluationItem]]:
    """Load a direct dataset or a reproducible mapping over existing benchmark files."""

    path = Path(manifest_path).expanduser().resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    items: list[EvaluationItem] = []

    dataset_path = manifest.get("dataset_path")
    if dataset_path:
        source = _resolve_manifest_path(path, str(dataset_path))
        items.extend(_coerce_item(row) for row in _read_jsonl(source))
    for mapping in manifest.get("source_mappings") or []:
        source = (REPO_ROOT / str(mapping["source_path"])).resolve()
        if not source.is_relative_to(REPO_ROOT):
            raise ValueError("source mapping escapes the repository boundary")
        selected_ids = [str(value) for value in mapping.get("selected_ids") or []]
        by_id = {
            str(row.get("id") or row.get("item_id")): row
            for row in _read_jsonl(source)
        }
        missing = [item_id for item_id in selected_ids if item_id not in by_id]
        if missing:
            raise ValueError(f"manifest mapping references missing IDs in {source}: {missing}")
        for item_id in selected_ids:
            row = dict(by_id[item_id])
            row["category"] = str(mapping.get("category") or row.get("category") or "quality")
            items.append(_coerce_item(row))

    seen: set[str] = set()
    duplicates: list[str] = []
    for item in items:
        if item.item_id in seen:
            duplicates.append(item.item_id)
        seen.add(item.item_id)
    if duplicates:
        raise ValueError(f"duplicate evaluation item IDs: {sorted(set(duplicates))}")
    return manifest, items


def materialization_evidence(items: Sequence[EvaluationItem]) -> dict[str, Any]:
    """Hash the exact provider prompts and their pre-materialized retrieval contexts."""

    records: list[dict[str, str]] = []
    reasons: list[str] = []
    for item in sorted(items, key=lambda value: value.item_id):
        if item.frozen_context is None:
            computed_context_hash = ""
            reasons.append(f"item {item.item_id} is missing frozen_context")
        else:
            canonical_context = json.dumps(
                item.frozen_context,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            computed_context_hash = _sha256_text(canonical_context)
        context_hash = str(item.context_sha256 or "")
        if not item.provider_prompt_materialized:
            reasons.append(f"item {item.item_id} is missing provider_prompt")
        if not re.fullmatch(r"[0-9a-f]{64}", context_hash):
            reasons.append(f"item {item.item_id} is missing a valid context_sha256")
        elif context_hash != computed_context_hash:
            reasons.append(f"item {item.item_id} context_sha256 does not match frozen_context")
        records.append({
            "item_id": item.item_id,
            "provider_prompt_sha256": _sha256_text(item.prompt),
            "context_sha256": context_hash,
        })
    canonical_records = json.dumps(records, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    canonical_contexts = json.dumps(
        [{"item_id": record["item_id"], "context_sha256": record["context_sha256"]} for record in records],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        "valid": not reasons and bool(records),
        "reasons": reasons or ([] if records else ["materialized final inputs are empty"]),
        "item_count": len(records),
        "materialized_inputs_sha256": _sha256_text(canonical_records),
        "frozen_contexts_sha256": _sha256_text(canonical_contexts),
    }


def dataset_gate(
    manifest: Mapping[str, Any],
    items: Sequence[EvaluationItem],
    *,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Hard gate for any claim that a candidate may replace the default provider."""

    reasons: list[str] = []
    counts = {language: sum(item.language == language for item in items) for language in ALLOWED_LANGUAGES}
    category_counts = {
        language: {
            category: sum(
                item.language == language and item.category == category for item in items
            )
            for category in sorted(ALLOWED_CATEGORIES)
        }
        for language in sorted(ALLOWED_LANGUAGES)
    }
    role = str(manifest.get("dataset_role") or "")
    status = str(manifest.get("status") or "")
    if role != "final_untouched":
        reasons.append("dataset_role must be final_untouched")
    if status != "sealed":
        reasons.append("final dataset status must be sealed")
    if not bool(manifest.get("untouched")):
        reasons.append("final dataset must be explicitly marked untouched")
    if role == "final_untouched" and manifest.get("source_mappings"):
        reasons.append("final dataset must use one directly hashed dataset_path, not source_mappings")
    if role == "final_untouched" and not manifest.get("dataset_path"):
        reasons.append("final dataset requires a directly hashed dataset_path")
    for language in sorted(ALLOWED_LANGUAGES):
        if counts[language] < FINAL_MIN_PER_LANGUAGE:
            reasons.append(
                f"final dataset requires at least {FINAL_MIN_PER_LANGUAGE} {language} items; found {counts[language]}"
            )
    required_categories = set(manifest.get("required_categories") or ALLOWED_CATEGORIES)
    present_categories = {item.category for item in items}
    for category in sorted(required_categories - present_categories):
        reasons.append(f"final dataset is missing required category: {category}")
    per_language_minimums = manifest.get("required_category_minimums_per_language") or {}
    for language in sorted(ALLOWED_LANGUAGES):
        for category, raw_minimum in sorted(per_language_minimums.items()):
            minimum = int(raw_minimum)
            found = category_counts[language].get(category, 0)
            if found < minimum:
                reasons.append(
                    f"final dataset requires at least {minimum} {language}/{category} items; found {found}"
                )

    expected_sha = str(manifest.get("content_sha256") or "")
    dataset_path = manifest.get("dataset_path")
    if not expected_sha:
        reasons.append("sealed final dataset requires content_sha256")
    elif manifest_path and dataset_path:
        source = _resolve_manifest_path(manifest_path, str(dataset_path))
        if not source.exists() or _file_sha256(source) != expected_sha:
            reasons.append("final dataset content hash does not match manifest")

    materialization = materialization_evidence(items)
    reasons.extend(materialization["reasons"])
    expected_materialized_hash = str(manifest.get("materialized_inputs_sha256") or "")
    expected_context_hash = str(manifest.get("frozen_contexts_sha256") or "")
    if expected_materialized_hash != materialization["materialized_inputs_sha256"]:
        reasons.append("materialized provider-input hash does not match manifest")
    if expected_context_hash != materialization["frozen_contexts_sha256"]:
        reasons.append("frozen retrieval-context hash does not match manifest")
    if manifest.get("materialization_configuration") != FINAL_CONFIGURATION:
        reasons.append("materialization configuration does not match preregistration")
    run_ledger_path = str(manifest.get("run_ledger_path") or "").strip()
    if not run_ledger_path:
        reasons.append("sealed final dataset requires run_ledger_path")
    elif manifest_path:
        try:
            _resolve_manifest_path(manifest_path, run_ledger_path)
        except ValueError:
            reasons.append("final run ledger path is outside the allowed boundary")

    return {
        "eligible_for_default_switch": not reasons,
        "verdict": "PASS" if not reasons else "BLOCKED",
        "dataset_role": role,
        "status": status,
        "counts": counts,
        "category_counts_by_language": category_counts,
        "required_minimum_per_language": FINAL_MIN_PER_LANGUAGE,
        "dataset_content_sha256": expected_sha or None,
        "materialized_inputs_sha256": materialization["materialized_inputs_sha256"],
        "frozen_contexts_sha256": materialization["frozen_contexts_sha256"],
        "reasons": reasons,
    }


def validate_final_protocol(
    *,
    dataset_gate_result: Mapping[str, Any],
    baseline: Provider,
    candidate: Provider,
    warmup_items: Sequence[EvaluationItem],
) -> dict[str, Any]:
    """Fail-closed conformance gate for the one allowed confirmatory execution.

    The generic runner remains useful for unit/development work. A default-switch verdict may use
    only this stricter wrapper, which freezes exact providers, generation/retrieval settings,
    warmups, repetitions, order seed, and an aggregate hash of the pre-materialized contexts.
    """

    reasons: list[str] = []
    if not dataset_gate_result.get("eligible_for_default_switch"):
        reasons.append("sealed final dataset gate did not pass")
    if (baseline.name, baseline.model_id) != ("groq", BASELINE_MODEL):
        reasons.append("baseline provider/model does not match preregistration")
    if (candidate.name, candidate.model_id) != ("openrouter", CANDIDATE_MODEL):
        reasons.append("candidate provider/model does not match preregistration")
    for label, provider in (("baseline", baseline), ("candidate", candidate)):
        resolved = getattr(provider, "resolved_configuration", None)
        if not isinstance(resolved, Mapping):
            reasons.append(f"{label} provider does not expose resolved runtime configuration")
            continue
        for key, expected in FINAL_PROVIDER_CONFIGURATION.items():
            if resolved.get(key) != expected:
                reasons.append(f"{label} runtime configuration mismatch: {key}")
    if len(warmup_items) != FINAL_WARMUPS_PER_PROVIDER:
        reasons.append(f"exactly {FINAL_WARMUPS_PER_PROVIDER} warmup items are required")
    if len({item.item_id for item in warmup_items}) != len(warmup_items):
        reasons.append("warmup item IDs must be unique")
    context_hash = str(dataset_gate_result.get("frozen_contexts_sha256") or "")
    if not re.fullmatch(r"[0-9a-f]{64}", context_hash):
        reasons.append("verified frozen_contexts_sha256 is required")
    return {
        "conformant": not reasons,
        "protocol_id": FINAL_PROTOCOL_ID,
        "reasons": reasons,
        "frozen_contexts_sha256": context_hash or None,
    }


def run_preregistered_final_benchmark(
    *,
    manifest_path: str | Path,
    baseline: Provider,
    candidate: Provider,
    scorer: Scorer,
    warmup_items: Sequence[EvaluationItem],
    operator_cost_safety_cap_usd: float | None = None,
) -> dict[str, Any]:
    """Execute a protocol-conformant research run from repository-bound, sealed inputs.

    Callers cannot inject items, a passing gate, a config map, or an arbitrary context hash. All
    such evidence is recomputed from the manifest and the actual providers expose the generation
    settings used by ``generate``. Providers and the scorer are still caller-supplied, however, so
    this runner cannot authorize a production default switch; only
    :func:`canonical_final_decision_status` is the production decision boundary.
    """

    manifest, items = load_items_from_manifest(manifest_path)
    dataset_gate_result = dataset_gate(manifest, items, manifest_path=manifest_path)
    protocol_gate = validate_final_protocol(
        dataset_gate_result=dataset_gate_result,
        baseline=baseline,
        candidate=candidate,
        warmup_items=warmup_items,
    )
    if not protocol_gate["conformant"]:
        raise ValueError("final protocol is not conformant: " + "; ".join(protocol_gate["reasons"]))

    ledger_path = _resolve_manifest_path(manifest_path, str(manifest["run_ledger_path"]))
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger = {
        "schema_version": 1,
        "protocol_id": FINAL_PROTOCOL_ID,
        "status": "started",
        "dataset_content_sha256": dataset_gate_result["dataset_content_sha256"],
        "materialized_inputs_sha256": dataset_gate_result["materialized_inputs_sha256"],
        "frozen_contexts_sha256": dataset_gate_result["frozen_contexts_sha256"],
        "started_at_unix": time.time(),
    }
    try:
        descriptor = os.open(str(ledger_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise RuntimeError("final confirmatory run ledger already exists; selective rerun forbidden") from exc
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(ledger, handle, indent=2, sort_keys=True)
        handle.write("\n")

    try:
        run = run_paired_benchmark(
            items,
            baseline=baseline,
            candidate=candidate,
            scorer=scorer,
            seed=FINAL_ORDER_SEED,
            repetitions=FINAL_REPETITIONS,
            warmup_items=warmup_items,
            stop_policy=StopPolicy(max_total_cost_usd=operator_cost_safety_cap_usd),
        )
        ledger.update({
            "status": "completed" if not run.get("stop_reason") else "stopped",
            "stop_reason": run.get("stop_reason"),
            "planned_items": run.get("planned_items"),
            "completed_at_unix": time.time(),
        })
        protocol_gate.update({
            "evidence_source": "run_preregistered_final_benchmark",
            "manifest_loaded_internally": True,
            "runtime_configuration_verified": True,
            "single_run_ledger_claimed": True,
            "run_ledger_status": ledger["status"],
            "dataset_content_sha256": dataset_gate_result["dataset_content_sha256"],
            "materialized_inputs_sha256": dataset_gate_result["materialized_inputs_sha256"],
        })
        run["protocol_gate"] = protocol_gate
        run["dataset_gate"] = dataset_gate_result
        run["protocol"]["configuration"] = dict(FINAL_CONFIGURATION)
        return run
    except Exception as exc:
        ledger.update({
            "status": "failed",
            "safe_error_type": type(exc).__name__,
            "completed_at_unix": time.time(),
        })
        raise
    finally:
        temporary_ledger = ledger_path.with_suffix(ledger_path.suffix + ".tmp")
        temporary_ledger.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary_ledger.replace(ledger_path)


def _verify_canonical_artifact_descriptor(
    descriptor: Any,
    *,
    gate_name: str,
) -> list[str]:
    """Verify a repository-bound external-gate artifact without trusting a boolean assertion."""

    if not isinstance(descriptor, Mapping):
        return [f"canonical external gate artifact is missing: {gate_name}"]
    relative_path = str(descriptor.get("path") or "").strip()
    expected_sha = str(descriptor.get("sha256") or "").strip()
    if not relative_path or not re.fullmatch(r"[0-9a-f]{64}", expected_sha):
        return [f"canonical external gate artifact descriptor is incomplete: {gate_name}"]
    try:
        artifact_path = _resolve_manifest_path(CANONICAL_FINAL_MANIFEST, relative_path)
    except ValueError:
        return [f"canonical external gate artifact escapes repository boundary: {gate_name}"]
    if not artifact_path.is_file() or _file_sha256(artifact_path) != expected_sha:
        return [f"canonical external gate artifact hash mismatch: {gate_name}"]
    return []


def canonical_final_decision_status() -> dict[str, Any]:
    """Return the non-injectable production switch status from repository-fixed evidence.

    This function deliberately accepts no manifest, provider, scorer, threshold, or gate
    arguments. The current repository has no sealed independent final set/scorer, so it must
    return ``BLOCKED``. A future authorization requires a reviewed code change that provisions
    and binds those artifacts; a caller-created mapping cannot unlock the switch.
    """

    manifest, items = load_items_from_manifest(CANONICAL_FINAL_MANIFEST)
    final_dataset_gate = dataset_gate(
        manifest,
        items,
        manifest_path=CANONICAL_FINAL_MANIFEST,
    )
    reasons = list(final_dataset_gate["reasons"])

    scorer_id = str(manifest.get("independent_scorer_id") or "").strip()
    if scorer_id != "advisu-independent-blind-scorer-v1":
        reasons.append("canonical independent blind scorer is not provisioned")

    owner_thresholds = manifest.get("owner_thresholds")
    if not isinstance(owner_thresholds, Mapping):
        owner_thresholds = {}
    for threshold in CANONICAL_OWNER_THRESHOLDS:
        value = owner_thresholds.get(threshold)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or float(value) < 0:
            reasons.append(f"canonical owner threshold is unresolved: {threshold}")

    external_artifacts = manifest.get("external_gate_artifacts")
    if not isinstance(external_artifacts, Mapping):
        external_artifacts = {}
    for gate_name in CANONICAL_EXTERNAL_GATES:
        reasons.extend(
            _verify_canonical_artifact_descriptor(
                external_artifacts.get(gate_name),
                gate_name=gate_name,
            )
        )

    # There is intentionally no success branch in this revision. Even if a local caller edits all
    # JSON fields, production authorization still requires a reviewed code-bound scorer/runner.
    reasons.append("canonical code-bound final scorer and execution path are not implemented")
    return {
        "verdict": "BLOCKED",
        "authorization_boundary": "canonical_final_decision_status",
        "manifest": str(CANONICAL_FINAL_MANIFEST.relative_to(REPO_ROOT)),
        "dataset_gate": final_dataset_gate,
        "reasons": list(dict.fromkeys(reasons)),
    }


def _normalize_response(response: ProviderResponse, elapsed_ms: float) -> dict[str, Any]:
    streaming_mode = response.streaming_mode if response.streaming_mode in {"native", "fake", "none"} else "none"
    ttft = response.ttft_ms if streaming_mode == "native" else None
    text = response.text or ""
    status = response.status if response.status else "error"
    if status == "ok" and not text.strip():
        status = "missing_response"
    return {
        "status": status,
        "latency_ms": round(elapsed_ms, 3),
        "streaming": {
            "mode": streaming_mode,
            "ttft_ms": ttft,
            "ttft_available": streaming_mode == "native" and ttft is not None,
            "ttft_unavailable_reason": (
                "fake_stream_is_not_provider_ttft" if streaming_mode == "fake" else
                (None if streaming_mode == "native" and ttft is not None else "native_ttft_not_reported")
            ),
        },
        "usage": {
            "prompt_tokens": response.prompt_tokens,
            "completion_tokens": response.completion_tokens,
            "total_tokens": response.total_tokens,
            "cost_usd": response.cost_usd,
            "source": "provider_api_usage_only_no_estimates",
        },
        "error_type": _safe_error_label(response.error_type),
        "response_sha256": _sha256_text(text) if text else None,
        "_response_text": text,
    }


def _failed_outcome(
    *, provider_name: str, model_id: str, error_type: str, metric_names: Sequence[str]
) -> dict[str, Any]:
    return {
        "provider": provider_name,
        "model_id": model_id,
        "status": "error",
        "latency_ms": None,
        "streaming": {
            "mode": "none",
            "ttft_ms": None,
            "ttft_available": False,
            "ttft_unavailable_reason": "request_failed",
        },
        "usage": {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "cost_usd": None,
            "source": "provider_api_usage_only_no_estimates",
        },
        "error_type": error_type,
        "response_sha256": None,
        "metrics": {metric: 0.0 for metric in metric_names},
    }


def _run_one(
    provider: Provider,
    item: EvaluationItem,
    scorer: Scorer,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        response = provider.generate(item.prompt, item_id=item.item_id, language=item.language)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        normalized = _normalize_response(response, elapsed_ms)
    except Exception as exc:
        return _failed_outcome(
            provider_name=provider.name,
            model_id=provider.model_id,
            error_type=_safe_error_type(exc),
            metric_names=scorer.metric_names,
        )

    text = normalized.pop("_response_text")
    normalized["provider"] = provider.name
    normalized["model_id"] = provider.model_id
    if normalized["status"] != "ok":
        normalized["metrics"] = {metric: 0.0 for metric in scorer.metric_names}
        return normalized
    try:
        raw_scores = scorer.score(item, text)
        scores: dict[str, float] = {}
        for metric in scorer.metric_names:
            value = float(raw_scores.get(metric, 0.0))
            scores[metric] = min(max(value, 0.0), 1.0)
        normalized["metrics"] = scores
    except Exception as exc:
        normalized["status"] = "scoring_error"
        normalized["error_type"] = _safe_error_type(exc)
        normalized["metrics"] = {metric: 0.0 for metric in scorer.metric_names}
    return normalized


def _stop_reason(
    rows: Sequence[Mapping[str, Any]], policy: StopPolicy
) -> str | None:
    if not rows:
        return None
    for provider_key in ("baseline", "candidate"):
        statuses = [
            str((((row.get("outcomes") or {}).get(provider_key) or {}).get("status") or "missing"))
            for row in rows
        ]
        consecutive = 0
        for status in reversed(statuses):
            if status == "ok":
                break
            consecutive += 1
        if consecutive >= policy.max_consecutive_failures:
            return f"{provider_key}_consecutive_failures"
        if len(statuses) >= policy.min_items_before_rate_stop:
            failure_rate = sum(status != "ok" for status in statuses) / len(statuses)
            if failure_rate > policy.max_failure_rate:
                return f"{provider_key}_failure_rate"

    if policy.max_total_cost_usd is not None:
        cost = 0.0
        for row in rows:
            for provider_key in ("baseline", "candidate"):
                value = (((row.get("outcomes") or {}).get(provider_key) or {}).get("usage") or {}).get("cost_usd")
                if isinstance(value, (int, float)):
                    cost += float(value)
        if cost >= policy.max_total_cost_usd:
            return "cost_budget_reached"
    return None


def run_paired_benchmark(
    items: Sequence[EvaluationItem],
    *,
    baseline: Provider,
    candidate: Provider,
    scorer: Scorer,
    seed: int = 20260804,
    repetitions: int = 3,
    warmup_items: Sequence[EvaluationItem] = (),
    stop_policy: StopPolicy = StopPolicy(),
) -> dict[str, Any]:
    """Run both providers for every preregistered item/repetition in paired randomized order.

    Warmups are called but never emitted as scored rows.  If a stop rule fires, every remaining
    item is still emitted with zero-scored ``not_run_stop_rule`` outcomes so the planned denominator
    cannot shrink after observing failures.
    """

    if baseline.model_id == candidate.model_id and baseline.name == candidate.name:
        raise ValueError("baseline and candidate must be distinct")
    if repetitions < 1:
        raise ValueError("repetitions must be at least one")
    if not scorer.metric_names or "task_success" not in scorer.metric_names:
        raise ValueError("scorer.metric_names must include preregistered primary metric task_success")

    rng = random.Random(seed)
    providers = {"baseline": baseline, "candidate": candidate}
    warmup_calls = 0
    warmup_failures: list[dict[str, str]] = []
    for item in warmup_items:
        order = ["baseline", "candidate"]
        rng.shuffle(order)
        for provider_key in order:
            warmup_calls += 1
            try:
                providers[provider_key].generate(
                    item.prompt, item_id=f"warmup:{item.item_id}", language=item.language
                )
            except Exception as exc:
                warmup_failures.append(
                    {"provider": provider_key, "error_type": _safe_error_type(exc)}
                )

    rows: list[dict[str, Any]] = []
    stop_reason: str | None = "warmup_failure" if warmup_failures else None
    for item in items:
        for repetition in range(repetitions):
            row_item_id = f"{item.item_id}:r{repetition}" if repetitions > 1 else item.item_id
            if stop_reason:
                outcomes = {
                    key: _failed_outcome(
                        provider_name=provider.name,
                        model_id=provider.model_id,
                        error_type="not_run_stop_rule",
                        metric_names=scorer.metric_names,
                    )
                    for key, provider in providers.items()
                }
                rows.append(
                    {
                        "schema_version": 1,
                        "item_id": row_item_id,
                        "source_item_id": item.item_id,
                        "repetition": repetition,
                        "cluster_id": item.cluster_id or item.item_id,
                        "language": item.language,
                        "category": item.category,
                        "call_order": [],
                        "outcomes": outcomes,
                    }
                )
                continue

            order = ["baseline", "candidate"]
            rng.shuffle(order)
            outcomes: dict[str, Any] = {}
            repeated_item = EvaluationItem(
                item_id=row_item_id,
                prompt=item.prompt,
                language=item.language,
                category=item.category,
                cluster_id=item.cluster_id or item.item_id,
                scoring_data=item.scoring_data,
                context_sha256=item.context_sha256,
                provider_prompt_materialized=item.provider_prompt_materialized,
                frozen_context=item.frozen_context,
            )
            for provider_key in order:
                outcomes[provider_key] = _run_one(
                    providers[provider_key], repeated_item, scorer
                )
            rows.append(
                {
                    "schema_version": 1,
                    "item_id": row_item_id,
                    "source_item_id": item.item_id,
                    "repetition": repetition,
                    "cluster_id": item.cluster_id or item.item_id,
                    "language": item.language,
                    "category": item.category,
                    "call_order": order,
                    "outcomes": outcomes,
                }
            )
            stop_reason = _stop_reason(rows, stop_policy)

    return {
        "schema_version": 1,
        "protocol": {
            "base_commit": BASE_COMMIT,
            "baseline": {"provider": baseline.name, "model_id": baseline.model_id},
            "candidate": {"provider": candidate.name, "model_id": candidate.model_id},
            "paired_order_randomization": True,
            "repetitions_per_configuration": repetitions,
            "seed": seed,
            "warmup_calls_excluded": warmup_calls,
            "failures_in_denominator": True,
            "prompt_logged": False,
            "response_text_logged": False,
            "reasoning_logged": False,
        },
        "warmup_failures": warmup_failures,
        "stop_reason": stop_reason,
        "planned_unique_items": len(items),
        "planned_items": len(items) * repetitions,
        "rows": rows,
    }


def write_redacted_results(run: Mapping[str, Any], output_path: str | Path) -> None:
    """Persist one JSONL row per planned item; prompts and responses never enter ``run``."""

    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in run.get("rows") or []:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the preregistered provider-evaluation dataset gate.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest, items = load_items_from_manifest(args.manifest)
    gate = dataset_gate(manifest, items, manifest_path=args.manifest)
    print(json.dumps(gate, ensure_ascii=False, indent=2, sort_keys=True))
    # Live provider execution is intentionally injected by an external driver.  This CLI cannot
    # accidentally spend quota or expose secrets.
    if not args.validate_only:
        print("No live provider adapter is bundled; import run_paired_benchmark and inject adapters.")
    return 0 if gate["eligible_for_default_switch"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
