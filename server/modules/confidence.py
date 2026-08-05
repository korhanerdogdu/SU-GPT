from __future__ import annotations

"""Observable-signal confidence and deterministic answer-admission policy.

The numeric score is retained for offline calibration work.  Runtime admission never trusts
model self-reporting and never invents a numeric threshold: hard observable failures are mapped
to a user-meaningful status, while an optional threshold may only be supplied by an external,
pre-registered calibration process.
"""

from dataclasses import dataclass
from modules import intents


@dataclass(frozen=True)
class ConfidenceSignals:
    retrieval_score: float | None = None
    evidence_count: int = 0
    independent_source_count: int = 0
    metadata_compatible: bool = False
    planner_valid: bool = True
    citations_present: bool = False
    output_schema_valid: bool = True
    contradictions: int = 0
    unsupported_claims: int = 0
    provider_error: bool = False
    fallback_used: bool = False
    intent: str = "general_academic"
    evidence_required: bool | None = None
    citations_required: bool = False
    citations_authorized: bool = True
    claims_supported: bool = True
    claim_coverage_checked: bool = False
    profile_complete: bool = True
    curriculum_available: bool = True
    provider_response_available: bool = True
    provider_timeout: bool = False
    output_language_match: bool = True
    guardrail_passed: bool = True
    prerequisites_valid: bool = True
    deterministic_rules_valid: bool = True
    schedule_conflicts_valid: bool = True


@dataclass(frozen=True)
class ConfidenceResult:
    score: float
    should_abstain: bool
    reason_codes: tuple[str, ...]
    status: str = "cannot_verify"
    answer_allowed: bool = False


@dataclass(frozen=True)
class IntentPolicy:
    evidence_required: bool
    profile_required: bool = False
    curriculum_required: bool = False
    deterministic_validation_required: bool = False


_DEFAULT_POLICY = IntentPolicy(evidence_required=True)
_INTENT_POLICIES = {
    intents.COURSE_RECOMMENDATION: IntentPolicy(True, True, True, True),
    "ders_onerisi": IntentPolicy(True, True, True, True),
    "graduation_planning": IntentPolicy(True, True, True, True),
    "graduation_plan": IntentPolicy(True, True, True, True),
    intents.GRADUATION_STATUS: IntentPolicy(True, True, True, True),
    "prerequisite_lookup": IntentPolicy(True, False, True, True),
    "course_details": IntentPolicy(True),
    intents.COURSE_DETAIL: IntentPolicy(True),
    "schedule_planning": IntentPolicy(True, True, True, True),
    intents.WEEKLY_SCHEDULE: IntentPolicy(True, True, True, True),
    "profile_dependent": IntentPolicy(True, True, False, True),
    "security_sensitive": IntentPolicy(False),
    "llm_only": IntentPolicy(False),
}


def policy_for_intent(intent: str | None) -> IntentPolicy:
    """Return the explicit policy for an intent without consulting an LLM."""

    normalized = intents.to_canonical(str(intent or "").strip().lower())
    return _INTENT_POLICIES.get(normalized, _DEFAULT_POLICY)


def _result(score: float, status: str, reasons: list[str]) -> ConfidenceResult:
    allowed = status in {"verified", "limited_evidence"}
    return ConfidenceResult(score, not allowed, tuple(dict.fromkeys(reasons)), status, allowed)


def assess(signals: ConfidenceSignals, *, threshold: float | None = None) -> ConfidenceResult:
    """Score observable evidence and apply only an explicitly calibrated threshold.

    No numeric threshold is selected by default because the repository has no labeled
    development calibration set or owner-defined coverage/risk target. Provider failure, absent
    evidence, and an invalid deterministic plan remain hard abstention conditions.
    """
    score = 0.0
    reasons: list[str] = []
    policy = policy_for_intent(signals.intent)
    evidence_required = (
        policy.evidence_required
        if signals.evidence_required is None
        else bool(signals.evidence_required)
    )
    if signals.provider_error or signals.provider_timeout or not signals.provider_response_available:
        reasons = [
            "provider_timeout" if signals.provider_timeout else "provider_error"
        ]
        if not signals.provider_response_available:
            reasons.append("provider_response_unavailable")
        return _result(0.0, "provider_unavailable", reasons)
    if signals.retrieval_score is not None:
        score += 0.25 * min(1.0, max(0.0, signals.retrieval_score))
    else:
        reasons.append("retrieval_score_missing")
    score += min(signals.evidence_count, 3) / 3 * 0.20
    score += min(signals.independent_source_count, 2) / 2 * 0.15
    score += 0.10 if signals.metadata_compatible else 0.0
    score += 0.10 if signals.planner_valid else 0.0
    score += 0.10 if signals.citations_present else 0.0
    score += 0.10 if signals.output_schema_valid else 0.0
    score -= min(0.3, max(0, signals.contradictions) * 0.15)
    score -= min(0.4, max(0, signals.unsupported_claims) * 0.2)
    score -= 0.10 if signals.fallback_used else 0.0
    score = round(min(1.0, max(0.0, score)), 4)
    if signals.evidence_count == 0:
        reasons.append("no_evidence")
    if not signals.metadata_compatible:
        reasons.append("metadata_mismatch")
    if not signals.planner_valid:
        reasons.append("planner_invalid")
    if signals.contradictions:
        reasons.append("source_contradiction")
    if signals.unsupported_claims:
        reasons.append("unsupported_claims")
    if signals.fallback_used:
        reasons.append("fallback_used")
    if not signals.guardrail_passed:
        reasons.append("guardrail_failed")
        return _result(score, "safe_abstention", reasons)
    if policy.profile_required and not signals.profile_complete:
        reasons.append("critical_profile_missing")
        return _result(score, "profile_required", reasons)
    if policy.curriculum_required and not signals.curriculum_available:
        reasons.append("curriculum_unavailable")
        return _result(score, "curriculum_unavailable", reasons)

    verification_failures: list[str] = []
    if evidence_required and signals.evidence_count == 0:
        verification_failures.append("no_evidence")
    if policy.deterministic_validation_required and not signals.deterministic_rules_valid:
        verification_failures.append("deterministic_rules_invalid")
    if not signals.planner_valid:
        verification_failures.append("planner_invalid")
    if not signals.prerequisites_valid:
        verification_failures.append("prerequisites_invalid")
    if not signals.schedule_conflicts_valid:
        verification_failures.append("schedule_conflict_validation_failed")
    if not signals.output_schema_valid:
        verification_failures.append("output_schema_invalid")
    if not signals.output_language_match:
        verification_failures.append("output_language_mismatch")
    if signals.citations_required and not signals.citations_present:
        verification_failures.append("citations_missing")
    if not signals.citations_authorized:
        verification_failures.append("citations_unauthorized")
    # A source label proves only that a document was retrieved; it does not prove that the
    # document supports every model-authored factual claim.  Until an independently calibrated
    # claim/evidence verifier has actually run, evidence-bound answers must fail closed instead of
    # presenting an unsupported answer under the weaker ``limited_evidence`` label.
    if evidence_required and not signals.claim_coverage_checked:
        verification_failures.append("claim_coverage_unverified")
    if (signals.claim_coverage_checked and not signals.claims_supported) or signals.unsupported_claims:
        verification_failures.append("unsupported_claims")
    if signals.contradictions:
        verification_failures.append("source_contradiction")
    if threshold is not None and score < threshold:
        verification_failures.append("below_calibrated_threshold")
    if verification_failures:
        reasons.extend(verification_failures)
        return _result(score, "cannot_verify", reasons)

    fully_verified = bool(
        signals.evidence_count
        and signals.metadata_compatible
        and signals.claims_supported
        and signals.claim_coverage_checked
        and signals.citations_authorized
        and (not signals.citations_required or signals.citations_present)
        and not signals.fallback_used
    )
    return _result(score, "verified" if fully_verified else "limited_evidence", reasons)
