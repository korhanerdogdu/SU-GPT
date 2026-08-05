from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.confidence_calibration import (
    brier_score,
    calibration_report,
    coverage_risk_curve,
    expected_calibration_error,
)
from modules.confidence import ConfidenceSignals, assess


def test_calibration_metrics_and_coverage_risk_are_exact():
    scores = [0.1, 0.4, 0.8, 0.9]
    labels = [0, 0, 1, 1]
    assert brier_score(scores, labels) == pytest.approx(0.055)
    assert expected_calibration_error(scores, labels, bins=2) == pytest.approx(0.2)
    point = coverage_risk_curve(scores, labels, thresholds=[0.5])[0]
    assert point.coverage == 0.5
    assert point.selective_accuracy == 1.0 and point.error_rate == 0.0


def test_report_refuses_to_invent_a_threshold_without_product_target():
    report = calibration_report([0.2, 0.8], [0, 1])
    assert report["threshold_selected"] is False
    assert "owner" in report["selection_note"]


def test_numeric_abstention_threshold_is_never_implicit():
    signals = ConfidenceSignals(
        retrieval_score=0.1,
        evidence_count=1,
        independent_source_count=1,
        metadata_compatible=True,
        citations_present=False,
        claim_coverage_checked=True,
        claims_supported=True,
    )
    assert not assess(signals).should_abstain
    assert assess(signals, threshold=0.9).should_abstain


@pytest.mark.parametrize(
    "scores,labels",
    [([], []), ([1.1], [1]), ([0.5], [2]), ([0.1, 0.2], [1])],
)
def test_invalid_calibration_inputs_are_rejected(scores, labels):
    with pytest.raises(ValueError):
        brier_score(scores, labels)
