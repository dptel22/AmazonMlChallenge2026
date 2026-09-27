import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from features import label_candidate_pairs, select_candidate_rows


def test_sampling_labels_first_and_retains_validation_and_every_positive():
    candidates = pd.DataFrame({
        "s1_idx": [0, 0, 0, 1, 1],
        "cand_idx": [0, 1, 2, 0, 1],
    })
    s1_ids = np.array(["S1-train", "S1-val"])
    pool_ids = np.array(["S2-positive", "S2-negative", "S3-negative"])
    truth = {"S1-train": {"S2-positive"}, "S1-val": set()}

    labels = label_candidate_pairs(candidates, s1_ids, pool_ids, truth)
    selected, selected_labels = select_candidate_rows(
        candidates, labels, s1_ids, {"S1-val"}, neg_ratio=0.0, seed=42)

    assert labels.tolist() == [1, 0, 0, 0, 0]
    assert selected.cand_idx.tolist() == [0, 0, 1]
    assert selected_labels.tolist() == [1, 0, 0]


def test_negative_sampling_is_seeded_and_never_drops_positives_or_validation():
    candidates = pd.DataFrame({
        "s1_idx": [0] * 101 + [1] * 20,
        "cand_idx": list(range(101)) + list(range(20)),
    })
    s1_ids = np.array(["S1-train", "S1-val"])
    pool_ids = np.array([f"S2-{i}" for i in range(101)])
    truth = {"S1-train": {"S2-0", "S2-100"}, "S1-val": set()}
    labels = label_candidate_pairs(candidates, s1_ids, pool_ids, truth)

    first, first_labels = select_candidate_rows(
        candidates, labels, s1_ids, {"S1-val"}, neg_ratio=0.05, seed=123)
    second, second_labels = select_candidate_rows(
        candidates, labels, s1_ids, {"S1-val"}, neg_ratio=0.05, seed=123)

    assert first.equals(second)
    assert np.array_equal(first_labels, second_labels)
    assert {0, 100}.issubset(set(first.loc[first.s1_idx == 0, "cand_idx"]))
    assert len(first.loc[first.s1_idx == 1]) == 20
    assert len(first.loc[first.s1_idx == 0]) < 101
