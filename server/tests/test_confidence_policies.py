from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.confidence import ConfidenceSignals, assess, policy_for_intent


@pytest.mark.parametrize(
    ("signals", "status", "reason"),
    [
        (
            ConfidenceSignals(intent="ders_onerisi", profile_complete=False),
            "profile_required",
            "critical_profile_missing",
        ),
        (
            ConfidenceSignals(intent="graduation_plan", curriculum_available=False),
            "curriculum_unavailable",
            "curriculum_unavailable",
        ),
        (
            ConfidenceSignals(provider_error=True),
            "provider_unavailable",
            "provider_error",
        ),
        (
            ConfidenceSignals(guardrail_passed=False),
            "safe_abstention",
            "guardrail_failed",
        ),
        (
            ConfidenceSignals(
                intent="ders_programi",
                evidence_count=1,
                metadata_compatible=True,
                schedule_conflicts_valid=False,
            ),
            "cannot_verify",
            "schedule_conflict_validation_failed",
        ),
        (
            ConfidenceSignals(
                evidence_count=1,
                metadata_compatible=True,
                citations_required=True,
                citations_present=True,
                citations_authorized=False,
            ),
            "cannot_verify",
            "citations_unauthorized",
        ),
    ],
)
def test_hard_observable_failures_map_to_structured_status(signals, status, reason):
    result = assess(signals)
    assert result.status == status
    assert result.should_abstain
    assert not result.answer_allowed
    assert reason in result.reason_codes


def test_optional_evidence_path_is_limited_but_allowed_without_numeric_threshold():
    result = assess(
        ConfidenceSignals(
            intent="llm_only",
            evidence_required=False,
            metadata_compatible=True,
            output_schema_valid=True,
        )
    )
    assert result.status == "limited_evidence"
    assert result.answer_allowed and not result.should_abstain


def test_complete_authorized_evidence_is_verified():
    result = assess(
        ConfidenceSignals(
            intent="course_details",
            retrieval_score=0.9,
            evidence_count=2,
            independent_source_count=2,
            metadata_compatible=True,
            citations_required=True,
            citations_present=True,
            citations_authorized=True,
            claims_supported=True,
            claim_coverage_checked=True,
        )
    )
    assert result.status == "verified"
    assert result.answer_allowed and not result.should_abstain


def test_evidence_bound_model_answer_abstains_until_claim_coverage_was_checked():
    result = assess(
        ConfidenceSignals(
            intent="course_details",
            evidence_count=1,
            metadata_compatible=True,
            citations_required=True,
            citations_present=True,
            citations_authorized=True,
            claims_supported=True,
            claim_coverage_checked=False,
        )
    )
    assert result.status == "cannot_verify"
    assert not result.answer_allowed
    assert "claim_coverage_unverified" in result.reason_codes


def test_intent_policy_is_explicit_and_unknown_intents_remain_evidence_bound():
    assert policy_for_intent("ders_onerisi").profile_required
    assert policy_for_intent("ders_programi").deterministic_validation_required
    assert policy_for_intent("unknown").evidence_required
