from __future__ import annotations

"""Development-only calibration metrics for observable confidence scores."""

import math
from dataclasses import dataclass
from typing import Sequence


def _validated(scores: Sequence[float], labels: Sequence[int]) -> tuple[list[float], list[int]]:
    if len(scores) != len(labels) or not scores:
        raise ValueError("scores and labels must be non-empty and aligned")
    clean_scores = [float(value) for value in scores]
    clean_labels = [int(value) for value in labels]
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in clean_scores):
        raise ValueError("confidence scores must be finite values in [0, 1]")
    if any(value not in {0, 1} for value in clean_labels):
        raise ValueError("labels must be binary")
    return clean_scores, clean_labels


def brier_score(scores: Sequence[float], labels: Sequence[int]) -> float:
    predicted, observed = _validated(scores, labels)
    return sum((score - label) ** 2 for score, label in zip(predicted, observed)) / len(observed)


def expected_calibration_error(
    scores: Sequence[float], labels: Sequence[int], *, bins: int = 10
) -> float:
    predicted, observed = _validated(scores, labels)
    if bins < 2:
        raise ValueError("bins must be at least 2")
    total = len(observed)
    error = 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        selected = [
            position
            for position, score in enumerate(predicted)
            if low <= score < high or (index == bins - 1 and score == 1.0)
        ]
        if not selected:
            continue
        average_score = sum(predicted[position] for position in selected) / len(selected)
        accuracy = sum(observed[position] for position in selected) / len(selected)
        error += len(selected) / total * abs(average_score - accuracy)
    return error


@dataclass(frozen=True)
class CoverageRiskPoint:
    threshold: float
    coverage: float
    selective_accuracy: float | None
    error_rate: float | None


def coverage_risk_curve(
    scores: Sequence[float],
    labels: Sequence[int],
    *,
    thresholds: Sequence[float] = tuple(index / 20 for index in range(21)),
) -> list[CoverageRiskPoint]:
    predicted, observed = _validated(scores, labels)
    points: list[CoverageRiskPoint] = []
    for raw_threshold in thresholds:
        threshold = float(raw_threshold)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("thresholds must be in [0, 1]")
        retained = [index for index, score in enumerate(predicted) if score >= threshold]
        accuracy = (
            sum(observed[index] for index in retained) / len(retained) if retained else None
        )
        points.append(
            CoverageRiskPoint(
                threshold=threshold,
                coverage=len(retained) / len(observed),
                selective_accuracy=accuracy,
                error_rate=(1.0 - accuracy) if accuracy is not None else None,
            )
        )
    return points


def calibration_report(scores: Sequence[float], labels: Sequence[int]) -> dict:
    curve = coverage_risk_curve(scores, labels)
    return {
        "n": len(scores),
        "brier_score": brier_score(scores, labels),
        "expected_calibration_error_10_bins": expected_calibration_error(scores, labels),
        "coverage_risk": [point.__dict__ for point in curve],
        "threshold_selected": False,
        "selection_note": (
            "No product coverage/error target is defined; select a threshold only on "
            "development data after the owner supplies that target."
        ),
    }
