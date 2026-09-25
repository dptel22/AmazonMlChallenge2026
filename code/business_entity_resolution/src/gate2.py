# -*- coding: utf-8 -*-
"""Phase 2 GATE: blocking recall + reduction ratio on the held-out validation split.

Split: 18% of S1 entity_ids via pandas sample(frac=0.18, random_state=42) — the
exact split is saved to work/val_s1_ids.txt; Kaggle's train.py MUST reuse it.
Recall = |true pairs (val S1) ∩ candidates| / |true pairs (val S1)|.
Reduction ratio = candidate pairs / naive cross join (|val S1| x |pool|).
Runs streaming: candidates per val row are counted/discarded, never accumulated.
"""
import sys, os, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from blocking import build_index, make_allowed, candidates_chunk, load_pool, WORK

GT = os.path.join(WORK, "ground_truth.tsv")


def make_val_split(save=True):
    s1 = pd.read_parquet(os.path.join(WORK, "train_s1.parquet"))
    val = s1.entity_id.sample(frac=0.18, random_state=42).sort_values()
    if save:
        val.to_frame().to_csv(os.path.join(WORK, "val_s1_ids.txt"), index=False, header=False)
    return set(val)


def main():
    val_ids = make_val_split()
    rng = np.random.default_rng(42)
    val_ids = set(rng.choice(sorted(val_ids), size=50000, replace=False))
    print(f"val S1 entities: {len(val_ids):,} (50k subsample of the saved 18% split)")
    pool = load_pool("train")
    s1 = pd.read_parquet(os.path.join(WORK, "train_s1.parquet"))
    val = s1[s1.entity_id.isin(val_ids)].reset_index(drop=True)
    del s1

    gt = pd.read_csv(GT, sep="\t", dtype=str, keep_default_na=False)
    tp = gt.assign(m=gt.matched_entity_ids.str.split(",")).explode("m")
    tp = tp[tp.m.str.strip() != ""]
    tp = tp[tp.source1_entity_id.isin(val_ids)]
    del gt
    print(f"true pairs in val: {len(tp):,}", flush=True)

    t0 = time.time()
    index, dfs = build_index(pool)
    print(f"index built in {time.time()-t0:.0f}s", flush=True)

    eid2i = {e: i for i, e in enumerate(val.entity_id.values)}
    pid2i = {e: i for i, e in enumerate(pool.entity_id.values)}
    tp_set = set((eid2i[s], pid2i[m]) for s, m in zip(tp.source1_entity_id.values, tp.m.values)
                 if s in eid2i and m in pid2i)
    del tp, pid2i, eid2i
    naive = len(val) * len(pool)

    for tok, addr_, sn, fn in [(1000, 2000, 2000, 20000),
                               (3000, 5000, 5000, 50000),
                               (10000, 10000, 10000, 100000)]:
        allowed = make_allowed(dfs, tok, addr_, sn, fn)
        t0 = time.time()
        n_pairs = 0
        hit = 0
        for base in range(0, len(val), 100_000):
            chunk = val.iloc[base:base + 100_000]
            cands = candidates_chunk(chunk, base, index, allowed)
            for j, c in enumerate(cands):
                n_pairs += len(c)
                s = base + j
                for m in c.tolist():
                    if (s, m) in tp_set:
                        hit += 1
            del cands
        print(f"caps=({tok},{addr_},{sn},{fn}): RECALL={hit/len(tp_set):.4%}  "
              f"pairs={n_pairs:,} ({n_pairs/len(val):.1f}/S1)  "
              f"reduction={n_pairs/naive:.6%} of naive ({naive:,})  [{time.time()-t0:.0f}s]",
              flush=True)


if __name__ == "__main__":
    main()
