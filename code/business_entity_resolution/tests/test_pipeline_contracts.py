import os
import sys

import numpy as np
import pandas as pd

SRC = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, SRC)

from blocking import proxy_topk_indices
from train import make_model_config, sweep_thresholds


def test_model_config_matches_kaggle_return_contract():
    result = make_model_config(0.42, 0.81, {"0": 0.9, "1": 0.8, "2-3": 0.7, "4+": 0.6})
    assert result == {
        "threshold": 0.42,
        "val_macro_f05": 0.81,
        "per_bucket_f05": {"0": 0.9, "1": 0.8, "2-3": 0.7, "4+": 0.6},
    }


def test_proxy_topk_is_deterministic_and_uses_name_only_for_blank_address():
    s1 = pd.Series({"name_s": "acme retail", "addr_n": "10 main street"})
    pool = pd.DataFrame([
        {"name_s": "acme retail", "addr_n": "10 main street"},
        {"name_s": "acme retail", "addr_n": ""},
        {"name_s": "different", "addr_n": "10 main street"},
    ])
    selected = proxy_topk_indices(s1, pool, np.array([0, 1, 2]), 2)
    assert selected.tolist() == [0, 1]


def test_proxy_topk_empty_and_short_inputs_are_safe():
    s1 = pd.Series({"name_s": "", "addr_n": ""})
    pool = pd.DataFrame(columns=["name_s", "addr_n"])
    assert proxy_topk_indices(s1, pool, np.empty(0, dtype=np.int64), 3).size == 0


def test_proxy_topk_uses_name_only_when_s1_address_is_blank():
    s1 = pd.Series({"name_s": "acme retail", "addr_n": ""})
    pool = pd.DataFrame([
        {"name_s": "acme retail", "addr_n": "unrelated address"},
        {"name_s": "acme retal", "addr_n": ""},
    ])
    selected = proxy_topk_indices(s1, pool, np.array([0, 1]), 1)
    assert selected.tolist() == [0]


def test_threshold_sweep_ignores_truth_outside_validation_entities():
    result = sweep_thresholds(
        np.array([0.9, 0.1]),
        np.array(["val", "outside"]),
        np.array(["match", "wrong"]),
        np.array(["val"]),
        {"val": {"match"}, "outside": {"missing"}},
    )
    assert result.macro_f05.max() == 1.0
