# -*- coding: utf-8 -*-
"""Phase 4: LightGBM training + threshold sweep.

Run --full on prepared real feature parquet files, locally or on Kaggle. The
--sample option is only a synthetic plumbing check and is not a quality result.

Contract (Kaggle output, consumed by infer.py):
  work/model.txt           LightGBM model  (booster.save_model)
  work/model_config.json   threshold, validation metrics, and inference
                            candidate-generation settings
"""
import sys, os, glob, argparse, time, json
import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(__file__))
from evaluate import macro_f05, bucket_report

WORK = os.environ.get("WORK_DIR", "work")   # Kaggle: point at the mounted feature dataset
GT = os.path.join(WORK, "ground_truth.tsv")

FEATURES = ["name_jaccard_raw", "name_jaccard_normalized", "name_jaccard_suffix_stripped",
            "name_cos", "addr_cos",
            "addr_jaccard", "token_overlap_count",
            "street_number_match", "missing_address_flag",
            "len_delta_name", "len_delta_addr"]

# imbalance config: Phase 2's reduction ratio leaves roughly 1-2% positives among
# candidates. scale_pos_weight usually calibrates better for a precision-weighted
# F_0.5 threshold sweep; is_unbalance is the fallback. Try scale_pos_weight first.
PARAMS = {
    "objective": "binary",
    "metric": "average_precision",
    "learning_rate": 0.08,
    "num_leaves": 128,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "imbalance_mode": "scale_pos_weight",  # or "is_unbalance"
    "num_threads": -1,
}


def load_features(split="train"):
    files = sorted(glob.glob(os.path.join(WORK, f"feats_{split}", "part*.parquet")))
    if not files:
        raise FileNotFoundError(f"no feature files under {WORK}/feats_{split}/")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def sweep_thresholds(proba, s1_ids_arr, cand_ids_arr, val_ids, truth_by):
    """Return DataFrame[threshold, macro_f05] over val entities only."""
    vmask = np.isin(s1_ids_arr, val_ids)
    v_p, v_s1, v_c = proba[vmask], s1_ids_arr[vmask], cand_ids_arr[vmask]
    val_set = set(val_ids)
    truth_val = {e: truth_by.get(e, ()) for e in val_set}
    rows = []
    for th in np.arange(0.10, 0.95 + 1e-9, 0.02):
        pred = {e: set() for e in val_set}
        sel = v_p >= th
        for e, m in zip(v_s1[sel], v_c[sel]):
            pred[e].add(m)
        rows.append({"threshold": round(float(th), 3),
                     "macro_f05": macro_f05(pred, truth_val)})
    return pd.DataFrame(rows)


def sweep_cutoffs_thresholds(proba, entity_index, candidate_rank, labels,
                             true_counts, cutoffs, thresholds):
    """Vectorized macro F0.5 sweep over top-K prefixes and score thresholds."""
    proba = np.asarray(proba, dtype=np.float64)
    entity_index = np.asarray(entity_index, dtype=np.int64)
    candidate_rank = np.asarray(candidate_rank, dtype=np.int32)
    labels = np.asarray(labels, dtype=np.int8)
    true_counts = np.asarray(true_counts, dtype=np.int64)
    if not (len(proba) == len(entity_index) == len(candidate_rank) == len(labels)):
        raise ValueError("validation arrays must have the same number of pairs")
    n_entities = len(true_counts)
    if np.any(entity_index < 0) or np.any(entity_index >= n_entities):
        raise ValueError("entity_index must address the declared validation population")
    total_true = int(true_counts.sum())
    rows = []
    for k in cutoffs:
        in_topk = candidate_rank < k
        n_cand = np.bincount(entity_index[in_topk], minlength=n_entities)
        candidate_recall = (int(labels[in_topk].sum()) / total_true) if total_true else 1.0
        for threshold in thresholds:
            selected = in_topk & (proba >= threshold)
            predicted = np.bincount(entity_index[selected], minlength=n_entities)
            true_positive = np.bincount(
                entity_index[selected], weights=labels[selected], minlength=n_entities)
            precision = np.divide(true_positive, predicted,
                                  out=np.ones(n_entities, dtype=np.float64),
                                  where=predicted > 0)
            recall = np.divide(true_positive, true_counts,
                               out=np.ones(n_entities, dtype=np.float64),
                               where=true_counts > 0)
            denominator = 0.25 * precision + recall
            scores = np.divide(1.25 * precision * recall, denominator,
                               out=np.zeros(n_entities, dtype=np.float64),
                               where=denominator > 0)
            rows.append({"topk": int(k), "threshold": float(threshold),
                         "macro_f05": float(scores.mean()) if n_entities else float("nan"),
                         "avg_candidates_per_s1": float(n_cand.mean()) if n_entities else 0.0,
                         "candidate_recall": candidate_recall})
    return pd.DataFrame(rows)


def choose_best_operating_point(table, tolerance=0.005):
    """Maximize macro F0.5; near-ties prefer fewer candidates, then recall."""
    if table.empty or not np.isfinite(table.macro_f05).any():
        raise ValueError("operating-point table has no finite F0.5 scores")
    best_score = float(table.macro_f05.max())
    eligible = table[table.macro_f05 >= best_score - tolerance]
    chosen = eligible.sort_values(
        ["avg_candidates_per_s1", "candidate_recall", "macro_f05", "topk"],
        ascending=[True, False, False, True], kind="stable").iloc[0]
    return chosen


def make_model_config(threshold, val_macro_f05, per_bucket_f05):
    """The exact Kaggle-return contract consumed by infer.py."""
    return {"threshold": float(threshold),
            "val_macro_f05": float(val_macro_f05),
            "per_bucket_f05": {k: float(v) for k, v in per_bucket_f05.items()}}


def load_candidate_config(work_dir=WORK):
    """Read the exact blocking settings used to create the training candidates."""
    path = os.path.join(work_dir, "train_cands", "_stage_manifest.json")
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f).get("config", {})


def model_runtime_config(candidate_config, selected_topk):
    """Carry candidate generation settings into inference's model config."""
    cap_names = ("token_cap", "addr_cap", "sn_cap", "fn_cap", "c3_cap")
    return {
        "topk": int(selected_topk),
        "proxy": str(candidate_config.get("proxy", "mean")),
        "caps": {name: int(candidate_config[name])
                 for name in cap_names if name in candidate_config},
        "max_candidates": int(candidate_config.get("max_candidates", 2_000)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true", help="tiny synthetic smoke test (local)")
    ap.add_argument("--full", action="store_true", help="real run (Kaggle)")
    args = ap.parse_args()

    t0 = time.time()
    candidate_config = {} if args.sample else load_candidate_config()
    if args.sample:
        # ---- synthetic slice: verify plumbing end-to-end, no real fit ----
        rng = np.random.default_rng(0)
        n = 20000
        df = pd.DataFrame({
            "s1_idx": rng.integers(0, 500, n), "cand_idx": rng.integers(0, 2000, n),
            "cand_rank": rng.integers(0, 500, n),
            **{f: rng.random(n) for f in FEATURES}})
        y = (df.name_jaccard_normalized * 0.6 + df.addr_jaccard * 0.4 > 0.75).astype(int)
        df["is_match"] = y
        val_ids = np.array([f"S1-{i}" for i in range(100, 150)])
        s1_ids_arr = np.array([f"S1-{i % 500}" for i in df.s1_idx])
        cand_ids_arr = np.array([f"S2-{j}" for j in df.cand_idx])
        truth_by = {f"S1-{i}": set() for i in range(100, 150)}
    else:
        df = load_features("train")
        # slim id lookups are written by prep.py (and bundled in kaggle_upload.zip)
        s1_ids = pd.read_parquet(os.path.join(WORK, "train_s1_ids.parquet"))["entity_id"].values
        pool_ids = pd.read_parquet(os.path.join(WORK, "train_pool_ids.parquet"))["entity_id"].values
        s1_ids_arr = s1_ids[df.s1_idx.values]
        cand_ids_arr = pool_ids[df.cand_idx.values]
        val_ids = pd.read_csv(os.path.join(WORK, "val_s1_ids.txt"), header=None)[0].values
        gt = pd.read_csv(os.path.join(WORK, "ground_truth.tsv"),
                         sep="\t", dtype=str, keep_default_na=False)
        truth_by = {e: (set(t.split(",")) if t else set())
                    for e, t in zip(gt.source1_entity_id, gt.matched_entity_ids)}

    # early-stopping eval set carved from TRAIN-fold entities only (rng seed 1),
    # never from val_ids — val stays reserved for the F_0.5 threshold sweep.
    tr_mask = ~np.isin(s1_ids_arr, val_ids)
    train_only_ids = np.unique(s1_ids_arr[tr_mask])
    es_ids = np.random.default_rng(1).choice(train_only_ids,
                                             size=len(train_only_ids) // 10, replace=False)
    es_mask = np.isin(s1_ids_arr, es_ids) & tr_mask
    fit_mask = tr_mask & ~es_mask
    Xtr, ytr = df.loc[fit_mask, FEATURES], df.loc[fit_mask, "is_match"]
    print(f"fit pairs: {len(Xtr):,} (pos {float(ytr.mean()):.4%}), "
          f"es pairs: {int(es_mask.sum()):,}")

    params = dict(PARAMS)
    if params.pop("imbalance_mode") == "scale_pos_weight":
        params["scale_pos_weight"] = float((1 - ytr.mean()) / max(ytr.mean(), 1e-9))
    else:
        params["is_unbalance"] = True
    dtr = lgb.Dataset(Xtr, ytr)
    dval = lgb.Dataset(df.loc[es_mask, FEATURES], df.loc[es_mask, "is_match"], reference=dtr)
    model = lgb.train(params, dtr, valid_sets=[dval], num_boost_round=2000,
                      callbacks=[lgb.early_stopping(100)])

    vmask = np.isin(s1_ids_arr, val_ids)
    val_rows = df.loc[vmask]
    proba = model.predict(val_rows[FEATURES])
    tune_ids = list(dict.fromkeys(np.asarray(val_ids).tolist()))
    entity_index = {e: i for i, e in enumerate(tune_ids)}
    val_s1 = s1_ids_arr[vmask]
    entity_idx = np.fromiter((entity_index[e] for e in val_s1), dtype=np.int32,
                             count=len(val_s1))
    pool_id_set = set(pool_ids.tolist()) if not args.sample else set(cand_ids_arr.tolist())
    true_counts = np.asarray([len(truth_by.get(e, set()) & pool_id_set) for e in tune_ids],
                             dtype=np.int32)
    sweep_cutoffs = {10, 20, 30, 50, 100, 200}
    if int(candidate_config.get("topk", 0)) > 0:
        sweep_cutoffs.add(int(candidate_config["topk"]))
    sw = sweep_cutoffs_thresholds(
        proba, entity_idx, val_rows.cand_rank.to_numpy(dtype=np.int32),
        val_rows.is_match.to_numpy(dtype=np.int8), true_counts,
        cutoffs=sorted(sweep_cutoffs),
        thresholds=np.round(np.arange(0.10, 0.951, 0.02), 3))
    print(sw.to_string(index=False))
    best = choose_best_operating_point(sw, tolerance=0.005)
    print(f"best topk={best.topk} threshold={best.threshold} val_macro_f05={best.macro_f05:.4f}")

    pred = {e: set() for e in tune_ids}
    sel = (proba >= best.threshold) & (val_rows.cand_rank.to_numpy() < int(best.topk))
    for e, m in zip(val_s1[sel], cand_ids_arr[vmask][sel]):
        pred[e].add(m)
    truth_val = {e: truth_by.get(e, ()) for e in tune_ids}
    buckets = bucket_report(pred, truth_val)
    print("per-bucket F_0.5:", buckets)

    artifact_dir = os.path.join(WORK, "synth") if args.sample else WORK
    os.makedirs(artifact_dir, exist_ok=True)
    model.save_model(os.path.join(artifact_dir, "model.txt"))
    model_cfg = make_model_config(best.threshold, best.macro_f05, buckets)
    model_cfg.update(model_runtime_config(candidate_config, best.topk))
    model_cfg.update({"candidate_recall": float(best.candidate_recall),
                      "avg_candidates_per_s1": float(best.avg_candidates_per_s1)})
    pd.Series(model_cfg) \
        .to_json(os.path.join(artifact_dir, "model_config.json"))
    print(f"saved model.txt + model_config.json in {artifact_dir} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
