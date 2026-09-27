import os
import subprocess
import sys

import numpy as np
import pandas as pd

SRC = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, SRC)

from blocking import proxy_topk_indices
from train import (choose_best_operating_point, make_model_config,
                   sweep_cutoffs_thresholds, sweep_thresholds, FEATURES,
                   model_runtime_config)
from gate3 import GATE_FEATURES
from prep import copy_ground_truth


def test_prep_copies_training_labels_to_shared_work_path(tmp_path):
    source = tmp_path / "train_ground_truth.tsv"
    source.write_text("source1_entity_id\tmatched_entity_ids\nS1-1\tS2-1\n", encoding="utf-8")
    work = tmp_path / "work"

    target = copy_ground_truth(str(source), str(work))

    assert target == str(work / "ground_truth.tsv")
    assert (work / "ground_truth.tsv").read_bytes() == source.read_bytes()


def test_feature_gate_uses_only_columns_emitted_by_current_cosine_pipeline():
    assert set(GATE_FEATURES).issubset(set(FEATURES))


def test_model_config_preserves_candidate_proxy_and_caps_for_inference():
    cfg = model_runtime_config({
        "topk": 50, "proxy": "mean", "token_cap": 1000,
        "addr_cap": 2000, "sn_cap": 2000, "fn_cap": 20000,
        "c3_cap": 2000, "max_candidates": 1500,
    }, selected_topk=30)
    assert cfg == {
        "topk": 30, "proxy": "mean",
        "caps": {"token_cap": 1000, "addr_cap": 2000, "sn_cap": 2000,
                 "fn_cap": 20000, "c3_cap": 2000},
        "max_candidates": 1500,
    }


def test_sample_training_keeps_model_artifacts_out_of_production_workdir(tmp_path):
    env = dict(os.environ, WORK_DIR=str(tmp_path))
    subprocess.run(
        [sys.executable, os.path.join(SRC, "train.py"), "--sample"],
        env=env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    assert (tmp_path / "synth" / "model.txt").is_file()
    assert (tmp_path / "synth" / "model_config.json").is_file()
    assert not (tmp_path / "model.txt").exists()
    assert not (tmp_path / "model_config.json").exists()


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


def test_cutoff_threshold_sweep_uses_validation_population_and_candidate_rank():
    result = sweep_cutoffs_thresholds(
        proba=np.array([0.8, 0.9, 0.1, 0.1]),
        entity_index=np.array([0, 0, 1, 1]),
        candidate_rank=np.array([0, 1, 0, 1]),
        labels=np.array([1, 0, 0, 0]),
        true_counts=np.array([1, 0]),
        cutoffs=[1, 2],
        thresholds=[0.5],
    )
    k1, k2 = result.sort_values("topk").itertuples(index=False)
    assert k1.macro_f05 == 1.0
    assert k1.avg_candidates_per_s1 == 1.0
    assert k2.macro_f05 < k1.macro_f05


def test_operating_point_prefers_smaller_candidate_lists_within_f05_tolerance():
    table = pd.DataFrame([
        {"topk": 100, "threshold": 0.5, "macro_f05": 0.900,
         "avg_candidates_per_s1": 100.0, "candidate_recall": 0.91},
        {"topk": 200, "threshold": 0.5, "macro_f05": 0.904,
         "avg_candidates_per_s1": 200.0, "candidate_recall": 0.93},
    ])
    chosen = choose_best_operating_point(table, tolerance=0.005)
    assert chosen.topk == 100
