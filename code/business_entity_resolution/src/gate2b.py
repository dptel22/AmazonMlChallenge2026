# -*- coding: utf-8 -*-
"""Phase 2 GATE v2: recall@K after sparse TF-IDF proxy + per-S1 top-K truncation.

Motivation (gate2 result): raw key union gives RECALL=97.70% but 14,339 candidates/S1
→ ~31.6B pairs at full scale — infeasible. Fix: rank each S1's candidates by a cheap
proxy (mean or max of word-level name/address TF-IDF cosine; name-only when
either address is blank) and keep top-K. This gate measures RECALL@K for
K in {10, 20, 30, 50, 100, 200, 300, 500, 750, 1000} at two cap settings so (caps, K) can be
measured jointly for the current configured operating range.
"""
import sys, os, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from blocking import PROXY_FNS, build_index, make_allowed, candidates_chunk, load_pool, proxy_scores, WORK

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
    val_ids = set(rng.choice(sorted(val_ids), size=20000, replace=False))
    print(f"val S1 entities: {len(val_ids):,} (20k subsample)", flush=True)
    pool = load_pool("train", columns=["entity_id", "country", "name_s", "name_t",
                                       "addr_n", "addr_t", "stno"])
    s1 = pd.read_parquet(os.path.join(WORK, "train_s1.parquet"))
    full_train_s1_count = len(s1)
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

    pid2i = {e: i for i, e in enumerate(pool.entity_id.values)}
    tp_by_s1 = {}
    for s, m in zip(tp.source1_entity_id.values, tp.m.values):
        if m in pid2i:
            tp_by_s1.setdefault(s, set()).add(pid2i[m])
    del tp, pid2i
    naive = len(val) * len(pool)
    ks = [10, 20, 30, 50, 100, 200, 300, 500, 750, 1000]
    # Only measure the configured broad-cap/mean operating path here. The
    # previous sweep held duplicate df maps, two large allowlists, both proxy
    # result tables, and the 115M-posting index at once; that exceeded Colab's
    # 12.7 GB worker after index construction. Reuse the index's key objects
    # for the allowlist instead of copying millions of df keys into new sets.
    proxy_names = ["mean"]
    settings = [(1000, 2000, 2000, 20000)]
    tok, addr_, sn, fn = settings[0]
    limits = {"c3": 2000, "t": tok, "a": addr_, "sn": sn, "fn": fn}
    allowed = {family: set() for family in limits}
    for key, postings in index.items():
        family = key.split("|", 1)[0]
        if len(postings) <= limits[family]:
            allowed[family].add(key)
    print(f"  allowed keys: c3={len(allowed['c3'])} t={len(allowed['t'])} "
          f"a={len(allowed['a'])} sn={len(allowed['sn'])} fn={len(allowed['fn'])}",
          flush=True)
    del dfs
    import gc
    gc.collect()

    for (tok, addr_, sn, fn) in settings:
        t0 = time.time()
        hits = {p: {k: 0 for k in ks} for p in proxy_names}
        n_kept = {k: 0 for k in ks}
        n_union = 0
        try:
            for base in range(0, len(val), 50_000):
                chunk = val.iloc[base:base + 50_000]
                cands = candidates_chunk(chunk, base, index, allowed)
                for j, c in enumerate(cands):
                    n_union += len(c)
                    for k in ks:
                        n_kept[k] += min(k, len(c))
                    truth = tp_by_s1.get(chunk.entity_id.values[j], ())
                    if not truth or c.size == 0:
                        continue
                    s1row = chunk.iloc[j]
                    p_name, p_addr = proxy_scores(s1row, pool, c)
                    for p in proxy_names:
                        proxy = PROXY_FNS[p](p_name, p_addr)
                        order = np.argsort(-proxy, kind="stable")
                        for k in ks:
                            top = c[order[:k]]
                            hits[p][k] += len(set(top.tolist()) & truth)
                del cands
                if (base // 50_000) % 2 == 1:
                    print(f"  ... {base + 50_000:,}/{len(val):,} val rows "
                          f"({time.time()-t0:.0f}s)", flush=True)
        except MemoryError:
            import traceback
            traceback.print_exc()
            print(f"MEMORYERROR at caps=({tok},{addr_},{sn},{fn}) — partial results above",
                  flush=True)
        r = sum(len(v) for v in tp_by_s1.values())
        print(f"caps=({tok},{addr_},{sn},{fn}): union={n_union:,} "
              f"({n_union/len(val):.0f}/S1) [{time.time()-t0:.0f}s]", flush=True)
        for p in proxy_names:
            line = f"  proxy={p}: "
            for k in ks:
                line += f"recall@{k}={hits[p][k]/r:.4%} "
            print(line, flush=True)
        for k in ks:
            avg_kept = n_kept[k] / max(len(val), 1)
            estimate = round(avg_kept * full_train_s1_count)
            print(f"  K={k}: {avg_kept:.1f} candidates/S1; "
                  f"estimated train candidate pairs={estimate:,}", flush=True)
        gc.collect()


if __name__ == "__main__":
    main()
