# -*- coding: utf-8 -*-
"""Phase 5: inference on test + output writers.

Real mode:  python infer.py            (needs work/model.txt + work/model_config.json
                                        produced by Kaggle — see README "Training (Kaggle)")
Dummy mode: python infer.py --dummy    (no model; rule-based scorer, only to exercise
                                        the writers + validator before the model exists)

Writes output/matching_results.tsv and output/candidate_pairs.tsv:
  source1_entity_id \t comma-joined ids, no spaces, empty string if none.
Pre-write checks: matches ⊆ candidates, one row per test S1 entity, no dup ids in list.
"""
import sys, os, json, glob, time, argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from blocking import build_index, make_allowed, generate_pairs, load_pool, WORK
from features import compute_chunk, load_feats_data

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "..", "output")
DEFAULT_CAPS = dict(token_cap=1000, addr_cap=2000, sn_cap=2000, fn_cap=20000)


def load_model(work=WORK):
    import lightgbm as lgb
    model = lgb.Booster(model_file=os.path.join(work, "model.txt"))
    cfg = pd.read_json(os.path.join(work, "model_config.json"), typ="series").to_dict()
    return model, float(cfg["threshold"]), cfg


def score_pairs(model, df, features):
    return model.predict(df[features])


def write_outputs(s1_ids, matched_by, cand_by, out_dir):
    """matched_by/cand_by: dict s1_id -> iterable of ids. Checks then writes both files."""
    os.makedirs(out_dir, exist_ok=True)
    missing = 0
    bad_subset = 0
    dup_in_list = 0
    rows_m, rows_c = [], []
    for eid in s1_ids:  # exactly one row per test S1 entity, in file order
        m = list(dict.fromkeys(matched_by.get(eid, ())))   # dedupe, keep order
        c = list(dict.fromkeys(cand_by.get(eid, ())))
        if len(m) != len(matched_by.get(eid, ())):
            dup_in_list += 1
        extra = set(m) - set(c)
        if extra:
            bad_subset += 1
            c = c + sorted(extra)  # enforce matches ⊆ candidates
        rows_m.append((eid, ",".join(m)))
        rows_c.append((eid, ",".join(c)))
    if bad_subset or dup_in_list or missing:
        print(f"WARN: fixed {bad_subset} subset violations, {dup_in_list} dup lists")
    pd.DataFrame(rows_m, columns=["source1_entity_id", "matched_entity_ids"]) \
        .to_csv(os.path.join(out_dir, "matching_results.tsv"), sep="\t", index=False)
    pd.DataFrame(rows_c, columns=["source1_entity_id", "candidate_entity_ids"]) \
        .to_csv(os.path.join(out_dir, "candidate_pairs.tsv"), sep="\t", index=False)
    print(f"wrote {len(rows_m):,} rows to {out_dir} "
          f"({sum(1 for _, m in rows_m if m):,} non-empty match rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dummy", action="store_true", help="rule-based scorer, no model file")
    ap.add_argument("--s1-limit", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    split = "test"
    s1, pool = load_feats_data(split)
    if args.s1_limit:
        s1 = s1.iloc[:args.s1_limit]
    print(f"test: s1={len(s1):,} pool={len(pool):,}")

    index, dfs = build_index(pool)
    allowed = make_allowed(dfs, **DEFAULT_CAPS)
    del dfs

    # score all candidate pairs chunk-wise; keep (s1 -> cand, proba) only above threshold
    if args.dummy:
        threshold = 0.75
    else:
        model, threshold, cfg = load_model()
        print(f"model loaded: threshold={threshold} "
              f"val_macro_f05={cfg.get('val_macro_f05')}")
    features = ["name_jaccard_raw", "name_jaccard_normalized", "name_jaccard_suffix_stripped",
                "name_lev_raw", "name_lev_normalized", "name_lev_suffix_stripped",
                "addr_jaccard", "addr_lev", "token_overlap_count",
                "street_number_match", "missing_address_flag",
                "len_delta_name", "len_delta_addr"]
    matched_by, cand_by = {}, {}
    s1_ids = s1.entity_id.values
    pool_ids = pool.entity_id.values
    for part, (si, ci) in enumerate(generate_pairs(s1, index, allowed)):
        df = compute_chunk(s1, pool, si, ci)
        if args.dummy:
            proba = 0.5 * df.name_lev_normalized.fillna(0) + 0.5 * df.addr_lev.fillna(0)
        else:
            proba = score_pairs(model, df, features)
        keep = proba >= threshold
        for s, c in zip(si[keep], ci[keep]):
            matched_by.setdefault(s1_ids[s], []).append(pool_ids[c])
        for s, c in zip(si, ci):
            cand_by.setdefault(s1_ids[s], []).append(pool_ids[c])
        print(f"  part{part}: {len(si):,} pairs, {int(keep.sum()):,} kept "
              f"({time.time()-t0:.0f}s)", flush=True)
    # singleton rows for S1 entities that got no candidates at all
    for eid in s1_ids:
        cand_by.setdefault(eid, [])
        matched_by.setdefault(eid, [])

    write_outputs(s1_ids, matched_by, cand_by, OUT)


if __name__ == "__main__":
    main()
