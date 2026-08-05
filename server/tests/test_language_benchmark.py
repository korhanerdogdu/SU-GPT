from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.language_benchmark import build_cases, evaluate
from modules.language import detect_language


def test_language_benchmark_has_100_per_language_and_meets_gate():
    cases = build_cases()
    assert sum(case.expected == "tr" and not case.subset.startswith("mixed") for case in cases) == 100
    assert sum(case.expected == "en" and not case.subset.startswith("mixed") for case in cases) == 100
    result = evaluate(detect_language)
    assert result["by_language"]["tr"]["accuracy"] >= 0.99
    assert result["by_language"]["en"]["accuracy"] >= 0.99
    assert result["mixed"]["n"] == 20
    assert result["mixed"]["passed"] >= 19
