import os
import random
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from evaluate import f05_entity, macro_f05


def independent_f05(pred, true):
    pred, true = set(pred), set(true)
    if not pred and not true:
        return 1.0
    precision = len(pred & true) / len(pred) if pred else 0.0
    recall = len(pred & true) / len(true) if true else 1.0
    denominator = 0.25 * precision + recall
    return 1.25 * precision * recall / denominator if denominator else 0.0


def independent_macro(preds, truths):
    keys = set(preds) | set(truths)
    return sum(independent_f05(preds.get(k, ()), truths.get(k, ())) for k in keys) / len(keys)


def test_worked_and_empty_edge_cases():
    assert f05_entity(["a", "b", "d"], ["a", "b", "c"]) == 2 / 3
    assert f05_entity([], []) == 1.0
    assert f05_entity(["x"], []) == 0.0
    assert f05_entity([], ["x"]) == 0.0


def test_three_entity_macro_case_matches_hand_calculation():
    preds = {"e1": ["a", "b", "d"], "e2": [], "e3": ["z"]}
    truths = {"e1": ["a", "b", "c"], "e2": [], "e3": ["y"]}
    expected = ((2 / 3) + 1.0 + 0.0) / 3
    assert abs(macro_f05(preds, truths) - expected) < 1e-12


def test_macro_rejects_mismatched_entity_populations():
    with pytest.raises(ValueError, match="populations differ"):
        macro_f05({"e1": []}, {"e1": [], "e2": []})


def test_randomized_monte_carlo_matches_independent_formula():
    rng = random.Random(20260925)
    universe = [f"S{n}" for n in range(80)]
    preds, truths = {}, {}
    for i in range(1000):
        preds[f"e{i}"] = [x for x in universe if rng.random() < 0.08]
        truths[f"e{i}"] = [x for x in universe if rng.random() < 0.08]
    assert abs(macro_f05(preds, truths) - independent_macro(preds, truths)) < 1e-12


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
