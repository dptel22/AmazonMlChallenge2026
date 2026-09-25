# -*- coding: utf-8 -*-
"""Phase 4: LightGBM training + threshold sweep.

*** THE REAL FULL-SCALE FIT RUNS ON KAGGLE, NOT LOCALLY. ***
This script is fully written and smoke-tested here on a tiny synthetic slice
(--sample); it runs standalone on Kaggle against the feature parquet files +
val_s1_ids.txt produced by phases 1-3 (see README "Training (Kaggle)").

Contract (Kaggle output, consumed by infer.py):
  work/model.txt           LightGBM model  (booster.save_model)
  work/model_config.json   {"threshold": float, "val_macro_f05": float,
                            "per_bucket_f05": {"0":f,"1":f,"2-3":f,"4+":f}}
"""
import sys, os, glob, argparse, time
import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(__file__))
from evaluate import macro_f05, bucket_report

WORK = os.environ.get("WORK_DIR", "work")   # Kaggle: point at the mounted feature dataset
GT = os.path.join("student_resource", "dataset", "train", "train_ground_truth.tsv")

FEATURES = ["name_jaccard_raw", "name_jaccard_normalized", "name_jaccard_suffix_stripped",
            "name_lev_raw", "name_lev_normalized", "name_lev_suffix_stripped",
            "addr_jaccard", "addr_lev", "token_overlap_count",
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
    rows = []
    for th in np.arange(0.10, 0.95 + 1e-9, 0.02):
        pred = {e: set() for e in val_set}
        sel = v_p >= th
        for e, m in zip(v_s1[sel], v_c[sel]):
            pred[e].add(m)
        rows.append({"threshold": round(float(th), 3),
                     "macro_f05": macro_f05(pred, truth_by)})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true", help="tiny synthetic smoke test (local)")
    ap.add_argument("--full", action="store_true", help="real run (Kaggle)")
    args = ap.parse_args()

    t0 = time.time()
    if args.sample:
        # ---- synthetic slice: verify plumbing end-to-end, no real fit ----
        rng = np.random.default_rng(0)
        n = 20000
        df = pd.DataFrame({
            "s1_idx": rng.integers(0, 500, n), "cand_idx": rng.integers(0, 2000, n),
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

    proba = model.predict(df[FEATURES])
    sw = sweep_thresholds(proba, s1_ids_arr, cand_ids_arr, val_ids, truth_by)
    print(sw.to_string(index=False))
    best = sw.loc[sw.macro_f05.idxmax()]
    print(f"best threshold {best.threshold} val_macro_f05={best.macro_f05:.4f}")

    vmask = np.isin(s1_ids_arr, val_ids)
    pred = {e: set() for e in set(val_ids)}
    sel = (proba >= best.threshold) & vmask
    for e, m in zip(s1_ids_arr[sel], cand_ids_arr[sel]):
        pred[e].add(m)
    buckets = bucket_report(pred, truth_by)
    print("per-bucket F_0.5:", buckets)

    os.makedirs(WORK, exist_ok=True)
    model.save_model(os.path.join(WORK, "model.txt"))
    pd.Series({"threshold": float(best.threshold),
               "val_macro_f05": float(best.macro_f05),
               **{f"bucket_{k}": float(v) for k, v in buckets.items()},
               }).to_json(os.path.join(WORK, "model_config.json"))
    print(f"saved model.txt + model_config.json in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
