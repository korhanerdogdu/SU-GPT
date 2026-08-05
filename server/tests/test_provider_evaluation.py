from __future__ import annotations

import ast
import hashlib
import json
import sys
from pathlib import Path

import pytest


SERVER_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SERVER_ROOT.parent
if str(SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVER_ROOT))

from evaluation.provider_benchmark import (  # noqa: E402
    EvaluationItem,
    ProviderResponse,
    StopPolicy,
    canonical_final_decision_status,
    dataset_gate,
    load_items_from_manifest,
    run_paired_benchmark,
    run_preregistered_final_benchmark,
    validate_final_protocol,
    FINAL_CONFIGURATION,
    FINAL_PROVIDER_CONFIGURATION,
    materialization_evidence,
    write_redacted_results,
)
from evaluation.provider_stats import (  # noqa: E402
    analyze_paired_results,
    clustered_sign_flip_test,
    default_switch_verdict,
    exact_mcnemar,
    holm_adjust,
    paired_bootstrap,
)
from evaluation.run_live_provider_pilot import _provider_stop_reason


METRICS = (
    "task_success",
    "answer_correctness",
    "groundedness",
    "citation_support",
    "language_match",
    "safe_behavior",
)


class ContainsExpectedScorer:
    metric_names = METRICS

    def score(self, item: EvaluationItem, response_text: str) -> dict[str, float]:
        expected = str(item.scoring_data.get("expected") or "pass")
        passed = float(expected.lower() in response_text.lower())
        return {metric: passed for metric in self.metric_names}


class FakeProvider:
    def __init__(
        self,
        name: str,
        model_id: str,
        *,
        text_by_id: dict[str, str] | None = None,
        failures: set[str] | None = None,
        streaming_mode: str = "native",
        ttft_ms: float | None = 4.0,
        resolved_configuration: dict | None = None,
    ) -> None:
        self.name = name
        self.model_id = model_id
        self.text_by_id = text_by_id or {}
        self.failures = failures or set()
        self.streaming_mode = streaming_mode
        self.ttft_ms = ttft_ms
        self.resolved_configuration = dict(
            FINAL_PROVIDER_CONFIGURATION if resolved_configuration is None else resolved_configuration
        )
        self.calls: list[str] = []

    def generate(self, prompt: str, *, item_id: str, language: str) -> ProviderResponse:
        self.calls.append(item_id)
        if item_id in self.failures:
            raise TimeoutError("fake timeout containing details that must not be logged")
        text = self.text_by_id.get(item_id, "pass")
        return ProviderResponse(
            status="ok",
            text=text,
            streaming_mode=self.streaming_mode,
            ttft_ms=self.ttft_ms,
            prompt_tokens=11,
            completion_tokens=7,
            total_tokens=18,
            cost_usd=0.001,
        )


def items(count: int = 12, *, category: str = "quality") -> list[EvaluationItem]:
    return [
        EvaluationItem(
            item_id=f"item-{index}",
            prompt=f"private prompt {index}",
            language="tr" if index % 2 else "en",
            category=category,
            scoring_data={"expected": "pass"},
        )
        for index in range(count)
    ]


def run_fake(
    eval_items: list[EvaluationItem],
    *,
    baseline: FakeProvider | None = None,
    candidate: FakeProvider | None = None,
    warmups: list[EvaluationItem] | None = None,
    stop_policy: StopPolicy | None = None,
):
    return run_paired_benchmark(
        eval_items,
        baseline=baseline or FakeProvider("groq", "llama-3.3-70b-versatile"),
        candidate=candidate or FakeProvider("openrouter", "deepseek/deepseek-v4-pro"),
        scorer=ContainsExpectedScorer(),
        repetitions=1,
        warmup_items=warmups or [],
        stop_policy=stop_policy or StopPolicy(max_consecutive_failures=99, max_total_cost_usd=None),
    )


def test_development_manifest_is_real_mapping_but_cannot_switch_default():
    manifest_path = PROJECT_ROOT / "data" / "benchmark" / "provider_eval" / "development_manifest.json"
    manifest, loaded = load_items_from_manifest(manifest_path)
    assert len(loaded) == 48
    assert sum(item.language == "tr" for item in loaded) == 24
    assert sum(item.language == "en" for item in loaded) == 24
    assert len({item.item_id for item in loaded}) == 48
    assert all(item.cluster_id == item.item_id for item in loaded)
    gate = dataset_gate(manifest, loaded, manifest_path=manifest_path)
    assert gate["verdict"] == "BLOCKED"
    assert not gate["eligible_for_default_switch"]


def test_checked_in_final_manifest_is_explicitly_blocked_not_fabricated():
    manifest_path = PROJECT_ROOT / "data" / "benchmark" / "provider_eval" / "final_manifest.json"
    manifest, loaded = load_items_from_manifest(manifest_path)
    assert loaded == []
    gate = dataset_gate(manifest, loaded, manifest_path=manifest_path)
    assert gate["verdict"] == "BLOCKED"
    assert any("at least 100 tr" in reason for reason in gate["reasons"])
    assert any("at least 100 en" in reason for reason in gate["reasons"])


def test_preregistered_prompt_hashes_match_the_current_frozen_runtime():
    protocol = json.loads(
        (PROJECT_ROOT / "docs" / "openrouter_deepseek_preregistration.json").read_text()
    )
    generation = protocol["generation_configuration"]
    retrieval = protocol["retrieval_configuration"]
    assert FINAL_CONFIGURATION == {
        "protocol_id": protocol["protocol_id"],
        "registered_base_commit": protocol["registered_base_commit"],
        "run_ordinal": protocol["design"]["final_test_runs"],
        "retrieval_mode": retrieval["mode"],
        "retrieval_first_stage_depth": retrieval["first_stage_candidate_depth"],
        "retrieval_provider_context_cutoff": retrieval["provider_context_candidate_cutoff"],
        "final_context_top_k": retrieval["final_context_top_k"],
        "ranker_id": retrieval["ranker_id"],
        "reranker_id": retrieval["reranker_id"],
        "contexts_frozen_before_provider_calls": retrieval["contexts_frozen_before_provider_calls"],
        "prompt_strategy": generation["prompt_strategy"],
        "application_prompt_template_sha256": generation["application_prompt_template_sha256"],
        "strategy_directive_sha256": generation["strategy_directive_sha256"],
        "language_directive_sha256": generation["language_directive_sha256"],
        "reasoning_enabled": generation["primary_reasoning_enabled"],
        "temperature": generation["temperature"],
        "maximum_output_tokens": generation["maximum_output_tokens"],
        "request_timeout_seconds": generation["request_timeout_seconds"],
        "maximum_retries": generation["maximum_retries"],
        "fallback_provider": generation["fallback_during_controlled_comparison"],
    }
    source = (SERVER_ROOT / "modules" / "llm.py").read_text(encoding="utf-8")
    constants = [
        node.value
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]
    application_template = max(constants, key=len)
    assert hashlib.sha256(application_template.encode()).hexdigest() == generation[
        "application_prompt_template_sha256"
    ]

    from modules.llm import _LANGUAGE_DIRECTIVE, _PROMPT_STRATEGIES

    assert hashlib.sha256(_PROMPT_STRATEGIES["structured_lookup"].encode()).hexdigest() == generation[
        "strategy_directive_sha256"
    ]
    for language in ("tr", "en"):
        assert hashlib.sha256(_LANGUAGE_DIRECTIVE[language].encode()).hexdigest() == generation[
            "language_directive_sha256"
        ][language]


def test_paired_order_is_seeded_randomized_and_every_item_has_both_providers():
    run = run_fake(items(30))
    orders = {tuple(row["call_order"]) for row in run["rows"]}
    assert orders == {("baseline", "candidate"), ("candidate", "baseline")}
    assert all(set(row["outcomes"]) == {"baseline", "candidate"} for row in run["rows"])
    assert len(run["rows"]) == run["planned_items"] == 30


def test_generic_final_harness_defaults_to_three_clustered_repetitions():
    baseline = FakeProvider("groq", "base")
    candidate = FakeProvider("openrouter", "candidate")
    run = run_paired_benchmark(
        items(2),
        baseline=baseline,
        candidate=candidate,
        scorer=ContainsExpectedScorer(),
        stop_policy=StopPolicy(max_consecutive_failures=99, max_total_cost_usd=None),
    )
    assert run["planned_unique_items"] == 2
    assert run["planned_items"] == len(run["rows"]) == 6
    assert run["protocol"]["repetitions_per_configuration"] == 3
    assert {row["repetition"] for row in run["rows"]} == {0, 1, 2}
    assert {row["cluster_id"] for row in run["rows"]} == {"item-0", "item-1"}


def _write_sealed_final_fixture(tmp_path: Path) -> Path:
    source = tmp_path / "final.jsonl"
    rows = []
    for language in ("tr", "en"):
        categories = (["quality"] * 60 + ["safety"] * 15 + ["adversarial"] * 15 + ["planner"] * 10)
        for index, category in enumerate(categories):
            item_id = f"{language}-{category}-{index}"
            frozen_context = {"chunks": [{"id": f"chunk-{item_id}", "text": f"evidence {item_id}"}]}
            canonical_context = json.dumps(
                frozen_context, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
            rows.append({
                "id": item_id,
                "question": f"sealed question {item_id}",
                "provider_prompt": f"sealed exact provider prompt {item_id}",
                "frozen_context": frozen_context,
                "context_sha256": hashlib.sha256(canonical_context.encode()).hexdigest(),
                "language": language,
                "category": category,
                "cluster_id": item_id,
                "expected": "pass",
            })
    source.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest = {
        "schema_version": 1,
        "dataset_role": "final_untouched",
        "status": "sealed",
        "untouched": True,
        "dataset_path": source.name,
        "content_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "required_categories": ["quality", "safety", "adversarial", "planner"],
        "required_category_minimums_per_language": {
            "quality": 60, "safety": 15, "adversarial": 15, "planner": 10,
        },
        "materialization_configuration": FINAL_CONFIGURATION,
        "run_ledger_path": "final-run-ledger.json",
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    _, loaded = load_items_from_manifest(manifest_path)
    evidence = materialization_evidence(loaded)
    manifest.update({
        "materialized_inputs_sha256": evidence["materialized_inputs_sha256"],
        "frozen_contexts_sha256": evidence["frozen_contexts_sha256"],
    })
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def test_final_wrapper_binds_manifest_materialization_runtime_and_single_run_ledger(tmp_path: Path):
    baseline = FakeProvider("groq", "llama-3.3-70b-versatile")
    candidate = FakeProvider("openrouter", "deepseek/deepseek-v4-pro")
    warmups = items(2)
    manifest_path = _write_sealed_final_fixture(tmp_path)
    manifest, loaded = load_items_from_manifest(manifest_path)
    verified_gate = dataset_gate(manifest, loaded, manifest_path=manifest_path)
    protocol_gate = validate_final_protocol(
        dataset_gate_result=verified_gate,
        baseline=baseline,
        candidate=candidate,
        warmup_items=warmups,
    )
    assert protocol_gate["conformant"]
    run = run_preregistered_final_benchmark(
        manifest_path=manifest_path, baseline=baseline, candidate=candidate,
        scorer=ContainsExpectedScorer(), warmup_items=warmups,
    )
    assert run["protocol_gate"]["conformant"]
    assert run["planned_items"] == 600
    assert (tmp_path / "final-run-ledger.json").exists()
    with pytest.raises(RuntimeError, match="already exists"):
        run_preregistered_final_benchmark(
            manifest_path=manifest_path, baseline=baseline, candidate=candidate,
            scorer=ContainsExpectedScorer(), warmup_items=warmups,
        )


def test_final_wrapper_rejects_runtime_or_materialized_input_tampering_before_run(tmp_path: Path):
    manifest_path = _write_sealed_final_fixture(tmp_path)
    bad_candidate = FakeProvider(
        "openrouter",
        "deepseek/deepseek-v4-pro",
        resolved_configuration={**FINAL_PROVIDER_CONFIGURATION, "temperature": 0.7},
    )
    with pytest.raises(ValueError, match="runtime configuration mismatch"):
        run_preregistered_final_benchmark(
            manifest_path=manifest_path,
            baseline=FakeProvider("groq", "llama-3.3-70b-versatile"),
            candidate=bad_candidate,
            scorer=ContainsExpectedScorer(), warmup_items=items(2),
        )
    assert not (tmp_path / "final-run-ledger.json").exists()

    source = tmp_path / "final.jsonl"
    source.write_text(source.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not conformant"):
        run_preregistered_final_benchmark(
            manifest_path=manifest_path,
            baseline=FakeProvider("groq", "llama-3.3-70b-versatile"),
            candidate=FakeProvider("openrouter", "deepseek/deepseek-v4-pro"),
            scorer=ContainsExpectedScorer(), warmup_items=items(2),
        )


def test_warmups_are_called_but_excluded_from_rows_and_metrics():
    baseline = FakeProvider("groq", "base")
    candidate = FakeProvider("openrouter", "candidate")
    warmups = [EvaluationItem("w1", "warmup", "tr", scoring_data={"expected": "pass"})]
    run = run_fake(items(3), baseline=baseline, candidate=candidate, warmups=warmups)
    assert run["protocol"]["warmup_calls_excluded"] == 2
    assert len(run["rows"]) == 3
    assert "warmup:w1" in baseline.calls and "warmup:w1" in candidate.calls
    assert all(not row["item_id"].startswith("warmup:") for row in run["rows"])


def test_provider_exception_is_zero_scored_and_kept_in_denominator():
    baseline = FakeProvider("groq", "base")
    candidate = FakeProvider("openrouter", "candidate", failures={"item-1"})
    run = run_fake(items(4), baseline=baseline, candidate=candidate)
    failed = run["rows"][1]["outcomes"]["candidate"]
    assert failed["status"] == "error"
    assert failed["error_type"] == "TimeoutError"
    assert failed["metrics"] == {metric: 0.0 for metric in METRICS}
    analysis = analyze_paired_results(run["rows"], draws=500)
    assert analysis["paired_items"] == 4
    assert analysis["failures"]["candidate"]["failures"] == 1


def test_stop_rule_preserves_planned_denominator():
    candidate = FakeProvider(
        "openrouter",
        "candidate",
        failures={"item-0", "item-1", "item-2", "item-3"},
    )
    policy = StopPolicy(max_consecutive_failures=2, max_total_cost_usd=None)
    run = run_fake(items(6), candidate=candidate, stop_policy=policy)
    assert run["stop_reason"] == "candidate_consecutive_failures"
    assert len(run["rows"]) == 6
    assert run["rows"][2]["outcomes"]["candidate"]["error_type"] == "not_run_stop_rule"
    assert run["rows"][2]["outcomes"]["baseline"]["metrics"]["task_success"] == 0.0


def test_warmup_failure_blocks_scored_execution_and_preserves_rows():
    candidate = FakeProvider("openrouter", "candidate", failures={"warmup:w1"})
    warmups = [EvaluationItem("w1", "warmup", "en", scoring_data={"expected": "pass"})]
    run = run_fake(items(3), candidate=candidate, warmups=warmups)
    assert run["stop_reason"] == "warmup_failure"
    assert len(run["rows"]) == 3
    assert all(not row["call_order"] for row in run["rows"])


def test_fake_stream_never_reports_ttft():
    candidate = FakeProvider(
        "openrouter", "candidate", streaming_mode="fake", ttft_ms=1.0
    )
    run = run_fake(items(2), candidate=candidate)
    streaming = run["rows"][0]["outcomes"]["candidate"]["streaming"]
    assert streaming["mode"] == "fake"
    assert streaming["ttft_ms"] is None
    assert streaming["ttft_available"] is False
    analysis = analyze_paired_results(run["rows"], draws=500)
    assert analysis["latency"]["candidate"]["ttft_available"] is False
    assert analysis["latency"]["candidate"]["ttft_unavailable_reason"] == "fake_stream_is_not_provider_ttft"


def test_artifacts_exclude_prompts_responses_reasoning_and_raw_errors(tmp_path: Path):
    run = run_fake(items(2))
    output = tmp_path / "rows.jsonl"
    write_redacted_results(run, output)
    artifact = output.read_text(encoding="utf-8")
    assert "private prompt" not in artifact
    assert '"text"' not in artifact
    assert "reasoning" not in artifact
    assert "pass" not in artifact
    assert "response_sha256" in artifact
    assert '"cluster_id"' in artifact
    assert all(row["cluster_id"] == row["item_id"] for row in run["rows"])


def test_explicit_cluster_id_is_persisted_without_prompt_or_answer():
    eval_item = EvaluationItem(
        item_id="variant-2",
        prompt="sensitive prompt",
        language="tr",
        cluster_id="shared-source-7",
        scoring_data={"expected": "pass"},
    )
    run = run_fake([eval_item])
    assert run["rows"][0]["cluster_id"] == "shared-source-7"
    assert "prompt" not in run["rows"][0]


def test_usage_is_provider_reported_and_not_estimated():
    run = run_fake(items(2))
    outcome = run["rows"][0]["outcomes"]["candidate"]
    assert outcome["usage"] == {
        "prompt_tokens": 11,
        "completion_tokens": 7,
        "total_tokens": 18,
        "cost_usd": 0.001,
        "source": "provider_api_usage_only_no_estimates",
    }


def test_paired_bootstrap_detects_clear_improvement():
    result = paired_bootstrap(
        [0.0] * 40,
        [1.0] * 40,
        metric="task_success",
        draws=500,
    )
    assert result.delta == pytest.approx(1.0)
    assert result.ci_low > 0.0


def test_bootstrap_resamples_cluster_means_not_correlated_rows():
    result = paired_bootstrap(
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 0.0],
        metric="task_success",
        cluster_ids=["source-a", "source-a", "source-a", "source-b"],
        draws=500,
    )
    assert result.n == 4
    assert result.n_clusters == 2
    assert result.delta == pytest.approx(0.5)


def test_clustered_sign_flip_detects_clear_paired_improvement():
    result = clustered_sign_flip_test(
        [0.0] * 20,
        [1.0] * 20,
        cluster_ids=[f"cluster-{index}" for index in range(20)],
        draws=1_000,
    )
    assert result["n_clusters"] == 20
    assert result["observed_delta"] == pytest.approx(1.0)
    assert result["p_value"] < 0.01


def test_exact_mcnemar_and_holm_are_dependency_free_and_exact():
    mcnemar = exact_mcnemar([0.0] * 10, [1.0] * 10)
    assert mcnemar["candidate_only"] == 10
    assert mcnemar["baseline_only"] == 0
    assert mcnemar["p_value"] == pytest.approx(2 / 1024)
    adjusted = holm_adjust({"overall": 0.001, "tr": 0.02, "en": 0.2})
    assert adjusted["overall"]["reject"] is True
    assert adjusted["tr"]["adjusted_p"] >= adjusted["overall"]["adjusted_p"]
    assert adjusted["en"]["reject"] is False


def _passing_final_gate() -> dict:
    return {"eligible_for_default_switch": True, "reasons": []}


def _passing_protocol_gate() -> dict:
    return {
        "conformant": True,
        "reasons": [],
        "protocol_id": "advisu-openrouter-deepseek-v4-pro-prereg-v2",
        "evidence_source": "run_preregistered_final_benchmark",
        "manifest_loaded_internally": True,
        "runtime_configuration_verified": True,
        "single_run_ledger_claimed": True,
        "run_ledger_status": "completed",
        "dataset_content_sha256": "a" * 64,
        "materialized_inputs_sha256": "b" * 64,
        "frozen_contexts_sha256": "c" * 64,
    }


def _strong_quality_win(*, include_categories: tuple[str, ...] = ("safety", "adversarial", "planner")):
    eval_items: list[EvaluationItem] = []
    baseline_text: dict[str, str] = {}
    for language in ("tr", "en"):
        for index in range(20):
            item_id = f"quality-{language}-{index}"
            eval_items.append(
                EvaluationItem(
                    item_id=item_id,
                    prompt=f"private {language} question {index}",
                    language=language,
                    category="quality",
                    scoring_data={"expected": "pass"},
                )
            )
            baseline_text[item_id] = "incorrect"
        for category in include_categories:
            item_id = f"{category}-{language}"
            eval_items.append(
                EvaluationItem(
                    item_id=item_id,
                    prompt=f"private {category} check",
                    language=language,
                    category=category,
                    scoring_data={"expected": "pass"},
                )
            )
    run = run_fake(
        eval_items,
        baseline=FakeProvider("groq", "base", text_by_id=baseline_text),
    )
    return run, analyze_paired_results(run["rows"], draws=500)


def test_default_switch_is_blocked_when_final_gate_is_missing():
    run = run_fake(items(6))
    analysis = analyze_paired_results(run["rows"], draws=500)
    verdict = default_switch_verdict(
        analysis,
        dataset_gate={"eligible_for_default_switch": False, "reasons": ["100 TR + 100 EN missing"]},
    )
    assert verdict["verdict"] == "BLOCKED"


def test_candidate_new_veto_failure_rejects_candidate():
    veto_items = items(4, category="safety")
    candidate = FakeProvider(
        "openrouter", "candidate", text_by_id={"item-2": "unsafe failure"}
    )
    run = run_fake(veto_items, candidate=candidate)
    analysis = analyze_paired_results(run["rows"], draws=500)
    verdict = default_switch_verdict(
        analysis,
        dataset_gate=_passing_final_gate(),
        protocol_gate=_passing_protocol_gate(),
    )
    assert verdict["verdict"] == "REJECT_CANDIDATE"
    assert any("safety" in reason for reason in verdict["reasons"])


def test_missing_veto_stratum_blocks_even_after_quality_win():
    _, analysis = _strong_quality_win(include_categories=("safety", "adversarial"))
    verdict = default_switch_verdict(
        analysis,
        dataset_gate=_passing_final_gate(),
        protocol_gate=_passing_protocol_gate(),
    )
    assert verdict["verdict"] == "BLOCKED"
    assert any("planner" in reason for reason in verdict["reasons"])


def test_quality_win_remains_blocked_until_owner_thresholds_are_explicit():
    _, analysis = _strong_quality_win()
    assert analysis["quality"]["overall"]["metrics"]["task_success"]["ci_low"] > 0.0
    clustered = analysis["quality"]["overall"]["metrics"]["task_success"]["clustered_sign_flip"]
    assert clustered["holm"]["reject"] is True

    verdict = default_switch_verdict(
        analysis,
        dataset_gate=_passing_final_gate(),
        protocol_gate=_passing_protocol_gate(),
    )
    assert verdict["verdict"] == "BLOCKED"
    assert any("p95 latency limit" in reason for reason in verdict["reasons"])
    assert any("per-query cost limit" in reason for reason in verdict["reasons"])
    assert any("error non-inferiority margin" in reason for reason in verdict["reasons"])


def test_boolean_protocol_self_attestation_cannot_authorize_default_switch():
    _, analysis = _strong_quality_win()
    verdict = default_switch_verdict(
        analysis,
        dataset_gate=_passing_final_gate(),
        protocol_gate={"conformant": True, "reasons": []},
        latency_p95_limit_ms=1_000.0,
        cost_per_query_limit_usd=0.01,
        api_error_rate_noninferiority_margin=0.0,
        external_gates={
            "retrieval_hit_at_10_non_regression": True,
            "retrieval_all_gold_at_10_non_regression": True,
            "security_noninferiority": True,
            "confidence_calibration": True,
            "end_to_end": True,
            "independent_blind_evaluation": True,
        },
    )
    assert verdict["verdict"] == "BLOCKED"
    assert any("protocol evidence" in reason for reason in verdict["reasons"])


def test_caller_supplied_passing_evidence_cannot_authorize_default_switch():
    _, analysis = _strong_quality_win()
    verdict = default_switch_verdict(
        analysis,
        dataset_gate=_passing_final_gate(),
        protocol_gate=_passing_protocol_gate(),
        latency_p95_limit_ms=1_000.0,
        cost_per_query_limit_usd=0.01,
        api_error_rate_noninferiority_margin=0.0,
        external_gates={
            "retrieval_hit_at_10_non_regression": True,
            "retrieval_all_gold_at_10_non_regression": True,
            "security_noninferiority": True,
            "confidence_calibration": True,
            "end_to_end": True,
            "independent_blind_evaluation": True,
        },
    )
    assert verdict["verdict"] == "BLOCKED"
    assert any("canonical final-decision" in reason for reason in verdict["reasons"])


def test_quality_win_cannot_bypass_missing_external_hard_gates():
    _, analysis = _strong_quality_win()
    verdict = default_switch_verdict(
        analysis,
        dataset_gate=_passing_final_gate(),
        protocol_gate=_passing_protocol_gate(),
        latency_p95_limit_ms=1_000.0,
        cost_per_query_limit_usd=0.01,
        api_error_rate_noninferiority_margin=0.0,
    )
    assert verdict["verdict"] == "BLOCKED"
    assert any("external gate" in reason for reason in verdict["reasons"])


def test_stopped_run_is_blocked_even_with_a_passing_dataset_gate():
    run = run_fake(items(3))
    analysis = analyze_paired_results(run["rows"], draws=500)
    verdict = default_switch_verdict(
        analysis,
        dataset_gate=_passing_final_gate(),
        run_status={"stop_reason": "cost_budget_reached", "warmup_failures": []},
    )
    assert verdict["verdict"] == "BLOCKED"


def test_live_pilot_stop_policy_fires_before_retry_amplification():
    assert _provider_stop_reason(consecutive_failures=3, attempted=3, failures=3) == (
        "three_consecutive_provider_failures"
    )
    assert _provider_stop_reason(consecutive_failures=0, attempted=20, failures=6) == (
        "provider_failure_rate_above_25_percent"
    )
    assert _provider_stop_reason(consecutive_failures=0, attempted=20, failures=5) is None


def test_committed_pilot_evidence_matches_its_checksum_manifests():
    root = PROJECT_ROOT / "outputs" / "provider_evaluation"
    for run_id in ("dev-pilot-20260804T120357Z", "dev-pilot-20260804T215913Z"):
        run_directory = root / run_id
        expected = json.loads((run_directory / "artifact_checksums.json").read_text())
        assert (run_directory / "FINALIZED").read_text().strip() == "development pilot finalized"
        for filename, digest in expected.items():
            assert hashlib.sha256((run_directory / filename).read_bytes()).hexdigest() == digest


def test_canonical_final_decision_is_argument_free_and_blocked_by_real_missing_evidence():
    status = canonical_final_decision_status()
    assert status["verdict"] == "BLOCKED"
    assert status["manifest"] == "data/benchmark/provider_eval/final_manifest.json"
    reasons = "\n".join(status["reasons"])
    assert "directly hashed dataset_path" in reasons
    assert "independent blind scorer" in reasons
    assert "owner threshold" in reasons
    assert "external gate artifact" in reasons
    assert "code-bound final scorer" in reasons


def test_final_dataset_gate_rejects_source_mapping_assembly(tmp_path: Path):
    manifest_path = _write_sealed_final_fixture(tmp_path)
    manifest, loaded = load_items_from_manifest(manifest_path)
    manifest["source_mappings"] = [{"source_path": "data/benchmark/security_adversarial_v1.jsonl"}]
    gate = dataset_gate(manifest, loaded, manifest_path=manifest_path)
    assert gate["verdict"] == "BLOCKED"
    assert any("not source_mappings" in reason for reason in gate["reasons"])


def test_manifest_source_mapping_cannot_escape_repository(tmp_path: Path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps({
            "dataset_role": "development",
            "source_mappings": [{
                "source_path": "../../../../../../etc/passwd",
                "selected_ids": [],
            }],
        }),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="repository boundary"):
        load_items_from_manifest(manifest_path)


def test_manifest_loader_rejects_duplicate_item_ids(tmp_path: Path):
    source = tmp_path / "items.jsonl"
    source.write_text(
        "\n".join(
            json.dumps({"id": "same", "question": q, "language": "tr", "category": "quality"})
            for q in ("one", "two")
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"dataset_path": "items.jsonl", "dataset_role": "development", "status": "ready"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_items_from_manifest(manifest)
