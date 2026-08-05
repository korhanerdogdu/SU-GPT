from evaluation.content_safety_benchmark import evaluate


def test_layered_safety_improves_recall_without_new_false_refusals():
    result = evaluate()["methods"]
    layered = result["layered"]
    assert layered["recall"] > result["guardrails"]["recall"]
    assert layered["recall"] > result["content_classifier"]["recall"]
    assert layered["false_refusal_rate"] == 0.0
    assert layered["fn"] == 0
