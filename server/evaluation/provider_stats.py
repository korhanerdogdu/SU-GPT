"""Paired statistics for provider/model evaluation.

This module intentionally has no provider SDK imports.  It consumes the compact, redacted
per-item rows emitted by :mod:`provider_benchmark` and treats every missing/error outcome as a
zero-quality observation.  That invariant keeps provider failures in the denominator instead of
silently making a flaky model look better by dropping its failed requests.
"""

from __future__ import annotations

import math
import random
import re
import statistics
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


PROVIDER_KEYS = ("baseline", "candidate")
VETO_CATEGORIES = frozenset({"safety", "adversarial", "planner"})
FINAL_PROTOCOL_ID = "advisu-openrouter-deepseek-v4-pro-prereg-v2"
# A caller cannot authorize a production-default change by assembling provenance dictionaries.
# Only the repository-fixed canonical orchestration may ever receive this in-process capability,
# after it has constructed providers/scoring and linked every external artifact itself.
@dataclass(frozen=True)
class BootstrapResult:
    metric: str
    n: int
    n_clusters: int
    baseline_mean: float
    candidate_mean: float
    delta: float
    ci_low: float
    ci_high: float
    confidence: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "n": self.n,
            "n_clusters": self.n_clusters,
            "baseline_mean": self.baseline_mean,
            "candidate_mean": self.candidate_mean,
            "delta": self.delta,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "confidence": self.confidence,
        }


def _mean(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def percentile(values: Sequence[float], q: float) -> float | None:
    """Linear-interpolated percentile without a NumPy dependency."""

    if not values:
        return None
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must be between 0 and 1")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def outcome_metric(outcome: Mapping[str, Any] | None, metric: str) -> float:
    """Return one metric, forcing missing/error/non-finite values to zero."""

    if not outcome or outcome.get("status") != "ok":
        return 0.0
    raw = (outcome.get("metrics") or {}).get(metric)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(value):
        return 0.0
    return min(max(value, 0.0), 1.0)


def rows_for_stratum(
    rows: Sequence[Mapping[str, Any]],
    *,
    language: str | None = None,
    category: str | None = None,
) -> list[Mapping[str, Any]]:
    return [
        row
        for row in rows
        if (language is None or row.get("language") == language)
        and (category is None or row.get("category") == category)
    ]


def paired_values(
    rows: Sequence[Mapping[str, Any]], metric: str
) -> tuple[list[float], list[float]]:
    baseline: list[float] = []
    candidate: list[float] = []
    for row in rows:
        outcomes = row.get("outcomes") or {}
        baseline.append(outcome_metric(outcomes.get("baseline"), metric))
        candidate.append(outcome_metric(outcomes.get("candidate"), metric))
    return baseline, candidate


def cluster_ids_for_rows(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Stable cluster ID per row; older rows safely default to their item ID."""

    return [str(row.get("cluster_id") or row.get("item_id") or f"row-{index}") for index, row in enumerate(rows)]


def _cluster_means(
    baseline: Sequence[float],
    candidate: Sequence[float],
    cluster_ids: Sequence[str] | None,
) -> tuple[list[float], list[float], list[str]]:
    if len(baseline) != len(candidate):
        raise ValueError("paired vectors must have equal length")
    ids = list(cluster_ids) if cluster_ids is not None else [str(index) for index in range(len(baseline))]
    if len(ids) != len(baseline):
        raise ValueError("cluster_ids must align one-to-one with paired values")
    grouped: dict[str, tuple[list[float], list[float]]] = {}
    order: list[str] = []
    for cluster_id, base_value, candidate_value in zip(ids, baseline, candidate):
        if cluster_id not in grouped:
            grouped[cluster_id] = ([], [])
            order.append(cluster_id)
        grouped[cluster_id][0].append(float(base_value))
        grouped[cluster_id][1].append(float(candidate_value))
    cluster_baseline = [_mean(grouped[cluster_id][0]) for cluster_id in order]
    cluster_candidate = [_mean(grouped[cluster_id][1]) for cluster_id in order]
    return cluster_baseline, cluster_candidate, order


def paired_bootstrap(
    baseline: Sequence[float],
    candidate: Sequence[float],
    *,
    metric: str,
    cluster_ids: Sequence[str] | None = None,
    draws: int = 10_000,
    confidence: float = 0.95,
    seed: int = 20260804,
) -> BootstrapResult:
    """Cluster-weighted paired bootstrap CI for ``candidate - baseline``.

    Rows sharing ``cluster_id`` are reduced to one paired cluster mean before resampling, so
    paraphrases or multiple checks derived from the same source do not masquerade as independent
    evidence.  With no IDs supplied, each row is its own cluster for backwards compatibility.
    """

    if len(baseline) != len(candidate):
        raise ValueError("paired bootstrap requires equal-length vectors")
    if not baseline:
        raise ValueError("paired bootstrap requires at least one pair")
    if draws < 100:
        raise ValueError("draws must be at least 100")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")

    item_b = [float(value) for value in baseline]
    item_c = [float(value) for value in candidate]
    b, c, clusters = _cluster_means(item_b, item_c, cluster_ids)
    n = len(b)
    rng = random.Random(seed)
    sampled_deltas: list[float] = []
    for _ in range(draws):
        delta_sum = 0.0
        for _index in range(n):
            selected = rng.randrange(n)
            delta_sum += c[selected] - b[selected]
        sampled_deltas.append(delta_sum / n)

    alpha = 1.0 - confidence
    baseline_mean = _mean(b)
    candidate_mean = _mean(c)
    return BootstrapResult(
        metric=metric,
        n=len(item_b),
        n_clusters=len(clusters),
        baseline_mean=baseline_mean,
        candidate_mean=candidate_mean,
        delta=candidate_mean - baseline_mean,
        ci_low=float(percentile(sampled_deltas, alpha / 2.0)),
        ci_high=float(percentile(sampled_deltas, 1.0 - alpha / 2.0)),
        confidence=confidence,
    )


def clustered_sign_flip_test(
    baseline: Sequence[float],
    candidate: Sequence[float],
    *,
    cluster_ids: Sequence[str] | None = None,
    draws: int = 10_000,
    seed: int = 20260804,
) -> dict[str, Any]:
    """Two-sided paired randomization test, flipping one sign per independent cluster."""

    if draws < 100:
        raise ValueError("draws must be at least 100")
    cluster_baseline, cluster_candidate, clusters = _cluster_means(
        baseline, candidate, cluster_ids
    )
    if not clusters:
        raise ValueError("clustered sign-flip test requires at least one cluster")
    deltas = [candidate_value - baseline_value for baseline_value, candidate_value in zip(cluster_baseline, cluster_candidate)]
    observed = _mean(deltas)
    if all(delta == 0.0 for delta in deltas):
        p_value = 1.0
    else:
        rng = random.Random(seed)
        extreme = 0
        absolute_observed = abs(observed)
        for _ in range(draws):
            permuted = _mean([delta if rng.getrandbits(1) else -delta for delta in deltas])
            if abs(permuted) >= absolute_observed - 1e-15:
                extreme += 1
        # Add-one correction prevents a finite Monte Carlo test from reporting p=0.
        p_value = (extreme + 1) / (draws + 1)
    return {
        "n": len(baseline),
        "n_clusters": len(clusters),
        "observed_delta": observed,
        "draws": draws,
        "seed": seed,
        "alternative": "two_sided",
        "p_value": p_value,
    }


def exact_mcnemar(
    baseline: Sequence[float], candidate: Sequence[float], *, threshold: float = 0.5
) -> dict[str, Any]:
    """Two-sided exact McNemar test using the conditional binomial distribution."""

    if len(baseline) != len(candidate):
        raise ValueError("McNemar requires equal-length vectors")
    baseline_only = 0
    candidate_only = 0
    both_pass = 0
    both_fail = 0
    for base_value, candidate_value in zip(baseline, candidate):
        base_pass = float(base_value) >= threshold
        candidate_pass = float(candidate_value) >= threshold
        if base_pass and candidate_pass:
            both_pass += 1
        elif base_pass:
            baseline_only += 1
        elif candidate_pass:
            candidate_only += 1
        else:
            both_fail += 1

    discordant = baseline_only + candidate_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(
            math.comb(discordant, index)
            for index in range(0, min(baseline_only, candidate_only) + 1)
        ) / (2**discordant)
        p_value = min(1.0, 2.0 * tail)
    return {
        "n": len(baseline),
        "both_pass": both_pass,
        "baseline_only": baseline_only,
        "candidate_only": candidate_only,
        "both_fail": both_fail,
        "discordant": discordant,
        "p_value": p_value,
    }


def holm_adjust(p_values: Mapping[str, float], *, alpha: float = 0.05) -> dict[str, dict[str, Any]]:
    """Holm family-wise correction with monotone adjusted p-values."""

    ordered = sorted((name, min(max(float(value), 0.0), 1.0)) for name, value in p_values.items())
    ordered.sort(key=lambda pair: pair[1])
    total = len(ordered)
    previous = 0.0
    adjusted: dict[str, dict[str, Any]] = {}
    continue_rejecting = True
    for rank, (name, p_value) in enumerate(ordered, start=1):
        multiplier = total - rank + 1
        adjusted_p = min(1.0, max(previous, multiplier * p_value))
        previous = adjusted_p
        threshold = alpha / multiplier
        reject = continue_rejecting and p_value <= threshold
        if not reject:
            continue_rejecting = False
        adjusted[name] = {
            "raw_p": p_value,
            "adjusted_p": adjusted_p,
            "holm_threshold": threshold,
            "reject": reject,
            "rank": rank,
        }
    return adjusted


def latency_summary(rows: Sequence[Mapping[str, Any]], provider_key: str) -> dict[str, Any]:
    if provider_key not in PROVIDER_KEYS:
        raise ValueError(f"unknown provider key: {provider_key}")
    latencies: list[float] = []
    ttfts: list[float] = []
    fake_stream_count = 0
    for row in rows:
        outcome = ((row.get("outcomes") or {}).get(provider_key) or {})
        latency = outcome.get("latency_ms")
        if isinstance(latency, (int, float)) and math.isfinite(float(latency)):
            latencies.append(float(latency))
        streaming = outcome.get("streaming") or {}
        if streaming.get("mode") == "fake":
            fake_stream_count += 1
        ttft = streaming.get("ttft_ms")
        if streaming.get("mode") == "native" and isinstance(ttft, (int, float)):
            ttfts.append(float(ttft))
    return {
        "n_latency": len(latencies),
        "p50_ms": percentile(latencies, 0.50),
        "p95_ms": percentile(latencies, 0.95),
        "ttft_available": bool(ttfts) and fake_stream_count == 0,
        "n_ttft": len(ttfts),
        "ttft_p50_ms": percentile(ttfts, 0.50) if ttfts and fake_stream_count == 0 else None,
        "ttft_p95_ms": percentile(ttfts, 0.95) if ttfts and fake_stream_count == 0 else None,
        "ttft_unavailable_reason": (
            "fake_stream_is_not_provider_ttft" if fake_stream_count else
            (None if ttfts else "provider_did_not_report_native_ttft")
        ),
    }


def usage_summary(rows: Sequence[Mapping[str, Any]], provider_key: str) -> dict[str, Any]:
    fields = ("prompt_tokens", "completion_tokens", "total_tokens", "cost_usd")
    totals: dict[str, float] = {field: 0.0 for field in fields}
    present: dict[str, int] = {field: 0 for field in fields}
    for row in rows:
        usage = ((((row.get("outcomes") or {}).get(provider_key) or {}).get("usage") or {}))
        for field in fields:
            value = usage.get(field)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                totals[field] += float(value)
                present[field] += 1
    n = len(rows)
    return {
        "n": n,
        "totals": totals,
        "reported_counts": present,
        "complete": {field: present[field] == n for field in fields},
        "source": "provider_api_usage_only_no_estimates",
    }


def failure_summary(rows: Sequence[Mapping[str, Any]], provider_key: str) -> dict[str, Any]:
    statuses: dict[str, int] = {}
    for row in rows:
        status = str((((row.get("outcomes") or {}).get(provider_key) or {}).get("status") or "missing"))
        statuses[status] = statuses.get(status, 0) + 1
    failures = len(rows) - statuses.get("ok", 0)
    return {
        "n": len(rows),
        "failures": failures,
        "failure_rate": failures / len(rows) if rows else 0.0,
        "statuses": statuses,
        "failures_in_denominator": True,
    }


def analyze_paired_results(
    rows: Sequence[Mapping[str, Any]],
    *,
    primary_metric: str = "task_success",
    secondary_metrics: Sequence[str] = (
        "answer_correctness",
        "groundedness",
        "citation_support",
        "language_match",
        "safe_behavior",
    ),
    binary_metrics: Iterable[str] = ("task_success", "language_match", "safe_behavior"),
    draws: int = 10_000,
    seed: int = 20260804,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Return paired overall/stratified quality, latency, usage and failure statistics."""

    metric_names = tuple(dict.fromkeys((primary_metric, *secondary_metrics)))
    binary = set(binary_metrics)
    strata: dict[str, list[Mapping[str, Any]]] = {"overall": list(rows)}
    for language in ("tr", "en"):
        strata[f"language:{language}"] = rows_for_stratum(rows, language=language)
    for category in sorted({str(row.get("category") or "quality") for row in rows}):
        strata[f"category:{category}"] = rows_for_stratum(rows, category=category)

    quality: dict[str, Any] = {}
    primary_permutation_p: dict[str, float] = {}
    for stratum_name, stratum_rows in strata.items():
        if not stratum_rows:
            quality[stratum_name] = {"n": 0, "metrics": {}}
            continue
        metric_results: dict[str, Any] = {}
        cluster_ids = cluster_ids_for_rows(stratum_rows)
        for metric_index, metric in enumerate(metric_names):
            baseline, candidate = paired_values(stratum_rows, metric)
            boot = paired_bootstrap(
                baseline,
                candidate,
                metric=metric,
                cluster_ids=cluster_ids,
                draws=draws,
                seed=seed + metric_index,
            ).as_dict()
            boot["clustered_sign_flip"] = clustered_sign_flip_test(
                baseline,
                candidate,
                cluster_ids=cluster_ids,
                draws=draws,
                seed=seed + metric_index,
            )
            if metric in binary:
                boot["mcnemar_exact"] = exact_mcnemar(baseline, candidate)
            metric_results[metric] = boot
        quality[stratum_name] = {"n": len(stratum_rows), "metrics": metric_results}
        if stratum_name in {"overall", "language:tr", "language:en"}:
            primary_permutation_p[stratum_name] = metric_results[primary_metric]["clustered_sign_flip"]["p_value"]

    holm = holm_adjust(primary_permutation_p, alpha=alpha)
    for stratum_name, correction in holm.items():
        quality[stratum_name]["metrics"][primary_metric]["clustered_sign_flip"]["holm"] = correction

    vetoes: dict[str, Any] = {}
    for category in sorted(VETO_CATEGORIES):
        veto_rows = rows_for_stratum(rows, category=category)
        baseline, candidate = paired_values(veto_rows, primary_metric)
        vetoes[category] = {
            "n": len(veto_rows),
            "baseline_failures": sum(value < 0.5 for value in baseline),
            "candidate_failures": sum(value < 0.5 for value in candidate),
            "candidate_new_failures": sum(
                base >= 0.5 and cand < 0.5 for base, cand in zip(baseline, candidate)
            ),
        }

    return {
        "schema_version": 1,
        "paired_items": len(rows),
        "primary_metric": primary_metric,
        "secondary_metrics": list(secondary_metrics),
        "quality": quality,
        "familywise_correction": {
            "method": "holm",
            "alpha": alpha,
            "family": ["overall", "language:tr", "language:en"],
            "applied_to": "clustered_sign_flip_primary_p_values",
        },
        "mcnemar_role": "secondary_item_level_binary_diagnostic_not_an_acceptance_test",
        "vetoes": vetoes,
        "latency": {key: latency_summary(rows, key) for key in PROVIDER_KEYS},
        "usage": {key: usage_summary(rows, key) for key in PROVIDER_KEYS},
        "failures": {key: failure_summary(rows, key) for key in PROVIDER_KEYS},
    }


def default_switch_verdict(
    analysis: Mapping[str, Any],
    *,
    dataset_gate: Mapping[str, Any],
    protocol_gate: Mapping[str, Any] | None = None,
    run_status: Mapping[str, Any] | None = None,
    alpha: float = 0.05,
    latency_p95_limit_ms: float | None = None,
    cost_per_query_limit_usd: float | None = None,
    api_error_rate_noninferiority_margin: float | None = None,
    external_gates: Mapping[str, bool] | None = None,
    reasoning_enabled: bool = False,
    reasoning_cost_benefit_pass: bool | None = None,
) -> dict[str, Any]:
    """Apply every hard acceptance gate without inventing missing evidence.

    Provider-generation rows can prove response quality, language consistency, critical vetoes,
    latency, usage, and failures. Retrieval, calibrated confidence, blind judging, and end-to-end
    behavior are separate tracks; callers must provide their frozen pass/fail results explicitly.
    Missing required evidence blocks eligibility rather than silently counting as a pass.
    """

    reasons: list[str] = []
    if not dataset_gate.get("eligible_for_default_switch"):
        reasons.extend(str(reason) for reason in dataset_gate.get("reasons") or ["final dataset gate failed"])
        return {"verdict": "BLOCKED", "reasons": reasons}

    if run_status:
        if run_status.get("warmup_failures"):
            reasons.append("provider warmup failed; scored comparison is invalid")
        if run_status.get("stop_reason"):
            reasons.append(f"run stopped by preregistered rule: {run_status['stop_reason']}")
        if reasons:
            return {"verdict": "BLOCKED", "reasons": reasons}

    paired_items = int(analysis.get("paired_items") or 0)
    if paired_items <= 0:
        return {"verdict": "BLOCKED", "reasons": ["no paired scored items"]}

    missing_vetoes: list[str] = []
    observed_veto_failures: list[str] = []
    veto_results = analysis.get("vetoes") or {}
    for category in sorted(VETO_CATEGORIES):
        result = veto_results.get(category) or {}
        if int(result.get("n") or 0) == 0:
            missing_vetoes.append(category)
        candidate_failures = int(result.get("candidate_failures") or 0)
        if candidate_failures > 0:
            observed_veto_failures.append(
                f"candidate has {candidate_failures} critical {category} veto failure(s)"
            )
    if observed_veto_failures:
        return {"verdict": "REJECT_CANDIDATE", "reasons": observed_veto_failures}
    if missing_vetoes:
        return {
            "verdict": "BLOCKED",
            "reasons": [f"missing required veto category: {category}" for category in missing_vetoes],
        }

    if not protocol_gate or not protocol_gate.get("conformant"):
        reasons.extend(
            str(reason)
            for reason in ((protocol_gate or {}).get("reasons") or [
                "preregistered final-run protocol conformance evidence is missing"
            ])
        )
        return {"verdict": "BLOCKED", "reasons": reasons}
    required_protocol_evidence = {
        "protocol_id": FINAL_PROTOCOL_ID,
        "evidence_source": "run_preregistered_final_benchmark",
        "manifest_loaded_internally": True,
        "runtime_configuration_verified": True,
        "single_run_ledger_claimed": True,
        "run_ledger_status": "completed",
    }
    for key, expected in required_protocol_evidence.items():
        if protocol_gate.get(key) != expected:
            reasons.append(f"verified final protocol evidence mismatch: {key}")
    for key in (
        "dataset_content_sha256",
        "materialized_inputs_sha256",
        "frozen_contexts_sha256",
    ):
        if not re.fullmatch(r"[0-9a-f]{64}", str(protocol_gate.get(key) or "")):
            reasons.append(f"verified final protocol evidence missing: {key}")
    if reasons:
        return {"verdict": "BLOCKED", "reasons": reasons}

    primary = str(analysis.get("primary_metric") or "task_success")
    overall = (((analysis.get("quality") or {}).get("overall") or {}).get("metrics") or {}).get(primary) or {}
    if float(overall.get("ci_low") or 0.0) <= 0.0:
        reasons.append("primary paired-bootstrap 95% CI does not exclude zero")
    permutation = overall.get("clustered_sign_flip") or {}
    holm = permutation.get("holm") or {}
    if not holm.get("reject") or float(holm.get("adjusted_p") or 1.0) > alpha:
        reasons.append("primary clustered sign-flip result is not significant after Holm correction")

    for language in ("tr", "en"):
        language_metric = (
            (((analysis.get("quality") or {}).get(f"language:{language}") or {}).get("metrics") or {}).get(primary)
            or {}
        )
        # A CI entirely below zero is a statistically supported subgroup regression.  A wide CI
        # crossing zero is reported but does not become an invented win or loss.
        if language_metric.get("n", 0) and float(language_metric.get("ci_high") or 0.0) < 0.0:
            reasons.append(f"candidate has a significant primary regression in language:{language}")

        response_language = (
            (((analysis.get("quality") or {}).get(f"language:{language}") or {}).get("metrics") or {}).get(
                "language_match"
            )
            or {}
        )
        if not response_language or int(response_language.get("n") or 0) == 0:
            reasons.append(f"response-language evidence is missing for language:{language}")
        elif float(response_language.get("candidate_mean") or 0.0) < 0.99:
            reasons.append(f"candidate response-language accuracy is below 99% for language:{language}")

    for metric in ("groundedness", "citation_support"):
        measured = (
            (((analysis.get("quality") or {}).get("overall") or {}).get("metrics") or {}).get(metric)
            or {}
        )
        if not measured or int(measured.get("n") or 0) == 0:
            reasons.append(f"required non-regression metric is missing: {metric}")
        elif float(measured.get("delta") or 0.0) < 0.0:
            reasons.append(f"candidate regresses on required metric: {metric}")

    for stratum_name, stratum in (analysis.get("quality") or {}).items():
        if stratum_name == "overall" or not int((stratum or {}).get("n") or 0):
            continue
        stratum_primary = ((stratum or {}).get("metrics") or {}).get(primary) or {}
        if stratum_primary and float(stratum_primary.get("delta") or 0.0) < 0.0:
            reasons.append(f"candidate primary metric regresses in subgroup {stratum_name}")

    if reasons:
        return {"verdict": "KEEP_BASELINE", "reasons": reasons}

    required_external_gates = (
        "retrieval_hit_at_10_non_regression",
        "retrieval_all_gold_at_10_non_regression",
        "security_noninferiority",
        "confidence_calibration",
        "end_to_end",
        "independent_blind_evaluation",
    )
    supplied_external = dict(external_gates or {})
    missing_external = [name for name in required_external_gates if name not in supplied_external]
    external_blockers = [
        f"required external gate is unevaluated: {name}" for name in missing_external
    ]
    failed_external = [
        name for name in required_external_gates
        if name in supplied_external and not supplied_external[name]
    ]
    if failed_external:
        return {
            "verdict": "KEEP_BASELINE",
            "reasons": [f"required external gate failed: {name}" for name in failed_external],
        }
    if reasoning_enabled:
        if reasoning_cost_benefit_pass is None:
            external_blockers.append("reasoning cost/latency/quality benefit is unevaluated")
        elif not reasoning_cost_benefit_pass:
            return {
                "verdict": "KEEP_BASELINE",
                "reasons": ["reasoning does not pass the preregistered cost/latency/quality gate"],
            }

    unresolved: list[str] = []
    if latency_p95_limit_ms is None:
        unresolved.append("product owner has not set the p95 latency limit")
    if cost_per_query_limit_usd is None:
        unresolved.append("product owner has not set the per-query cost limit")
    if api_error_rate_noninferiority_margin is None:
        unresolved.append("product owner has not set the API error non-inferiority margin")

    operational_blockers: list[str] = []
    candidate_latency = ((analysis.get("latency") or {}).get("candidate") or {})
    p95 = candidate_latency.get("p95_ms")
    if p95 is None:
        operational_blockers.append("candidate p95 latency is unavailable")

    usage = analysis.get("usage") or {}
    for provider_key in PROVIDER_KEYS:
        provider_usage = (usage.get(provider_key) or {})
        completeness = provider_usage.get("complete") or {}
        if not completeness.get("prompt_tokens") or not completeness.get("completion_tokens"):
            operational_blockers.append(f"{provider_key} API token usage is incomplete")
        if not completeness.get("cost_usd"):
            operational_blockers.append(f"{provider_key} provider-reported cost is incomplete")

    if unresolved or operational_blockers or external_blockers:
        return {
            "verdict": "BLOCKED",
            "reasons": [*unresolved, *operational_blockers, *external_blockers],
        }

    operational_failures: list[str] = []
    if float(p95) > float(latency_p95_limit_ms):
        operational_failures.append(
            f"candidate p95 latency exceeds owner limit {float(latency_p95_limit_ms):.3f} ms"
        )

    candidate_usage = usage.get("candidate") or {}
    candidate_n = int(candidate_usage.get("n") or 0)
    candidate_total_cost = float((candidate_usage.get("totals") or {}).get("cost_usd") or 0.0)
    candidate_cost_per_query = candidate_total_cost / candidate_n if candidate_n else math.inf
    if candidate_cost_per_query > float(cost_per_query_limit_usd):
        operational_failures.append(
            "candidate provider-reported cost per query exceeds owner limit"
        )

    failures = analysis.get("failures") or {}
    baseline_failure_rate = float((failures.get("baseline") or {}).get("failure_rate") or 0.0)
    candidate_failure_rate = float((failures.get("candidate") or {}).get("failure_rate") or 0.0)
    if candidate_failure_rate > baseline_failure_rate + float(api_error_rate_noninferiority_margin):
        operational_failures.append("candidate API failure rate exceeds owner non-inferiority margin")

    if operational_failures:
        return {"verdict": "KEEP_BASELINE", "reasons": operational_failures}
    # This generic evaluator intentionally cannot authorize a production default change. It is
    # public and accepts caller-supplied analysis/gate mappings, so even a fully passing result is
    # research evidence only. The repository-fixed, argument-free canonical status entrypoint is
    # the sole production decision boundary and currently fails closed until real final artifacts
    # and an independent scorer are provisioned.
    return {
        "verdict": "BLOCKED",
        "reasons": [
            "only the repository-fixed canonical final-decision entrypoint may authorize a default switch"
        ],
    }
