# -*- coding: utf-8 -*-
"""Macro F_0.5 evaluator — standalone, reusable (imported by train.py; runnable directly).

Per S1 entity:
  precision = |pred ∩ true| / |pred|   (1.0 if pred and true both empty;
                                        0.0 if pred non-empty and true empty)
  recall    = |pred ∩ true| / |true|   (1.0 if true empty)
  F_0.5     = (1.25 * P * R) / (0.25 * P + R), 0 if denominator is 0
Final = macro average over ALL S1 entities (singletons count fully).

Self-check against the worked example is in demo().
"""
import numpy as np


def f05_entity(pred, true):
    p = set(pred); t = set(true)
    if not p and not t:
        return 1.0
    inter = len(p & t)
    prec = 1.0 if not p else inter / len(p)
    rec = 1.0 if not t else inter / len(t)
    den = 0.25 * prec + rec
    return (1.25 * prec * rec) / den if den else 0.0


def macro_f05(preds_by_entity, truths_by_entity):
    """preds/truths: dict entity_id -> iterable of matched ids (or lists aligned by key)."""
    keys = set(preds_by_entity) | set(truths_by_entity)
    scores = [f05_entity(preds_by_entity.get(k, ()), truths_by_entity.get(k, ()))
              for k in keys]
    return float(np.mean(scores))


def bucket_report(preds_by_entity, truths_by_entity, thresholds=None):
    """Macro F_0.5 broken out by true match-count bucket: 0 / 1 / 2-3 / 4+."""
    buckets = {"0": [], "1": [], "2-3": [], "4+": []}
    keys = set(preds_by_entity) | set(truths_by_entity)
    for k in keys:
        n = len(truths_by_entity.get(k, ()))
        b = "0" if n == 0 else "1" if n == 1 else "2-3" if n <= 3 else "4+"
        buckets[b].append(f05_entity(preds_by_entity.get(k, ()),
                                     truths_by_entity.get(k, ())))
    return {b: (float(np.mean(v)) if v else float("nan")) for b, v in buckets.items()}


def demo():
    # hand-verified example: entity with true={a,b,c}, pred={a,b,d}
    # P=2/3, R=2/3 -> F05 = 1.25*(4/9)/(0.25*(2/3)+2/3) = (5/9)/(5/6) = 2/3
    assert abs(f05_entity(["a", "b", "d"], ["a", "b", "c"]) - 2 / 3) < 1e-12
    # both empty -> 1.0; pred non-empty, true empty -> 0.0
    assert f05_entity([], []) == 1.0
    assert f05_entity(["x"], []) == 0.0
    # true non-empty, pred empty -> P=0 (inter=0/0? pred empty -> P=0? spec: P=1.0 only
    # if BOTH empty; here pred empty & true non-empty: P undefined by count -> inter=0
    # gives 0 via inter/len(p) guard: len(p)=0 -> P=0.0), R=0 -> F=0
    assert f05_entity([], ["x"]) == 0.0
    # macro across two entities
    m = macro_f05({"e1": ["a", "b", "d"], "e2": []}, {"e1": ["a", "b", "c"], "e2": []})
    assert abs(m - (2 / 3 + 1.0) / 2) < 1e-12
    print("evaluate.py demo: all F_0.5 assertions passed")


if __name__ == "__main__":
    demo()
