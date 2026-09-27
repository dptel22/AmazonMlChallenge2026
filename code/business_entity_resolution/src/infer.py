# -*- coding: utf-8 -*-
"""Phase 5: inference on test + output writers.

Real mode:  python infer.py            (needs work/model.txt + work/model_config.json
                                        produced by Kaggle — see README "Training (Kaggle)")
Dummy mode: python infer.py --dummy    (no model; rule-based scorer, only to exercise
                                        the writers + validator before the model exists)

Writes output/matching_results.tsv and output/candidate_pairs.tsv:
  source1_entity_id \t comma-joined ids, no spaces, empty string if none.
Each S1 entity's candidates are all produced by ONE chunk (each S1 row is processed
exactly once), so per-entity rows are joined and written per chunk — nothing is
accumulated across the whole run (OOM fix).
"""
import sys, os, time, argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from blocking import build_index, make_allowed, candidates_chunk, batch_topk_prune
from features import compute_chunk
from pipeline_state import ensure_local_path, prepare_stage, write_manifest

_SRC = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_SRC, os.pardir, os.pardir, os.pardir))
# OUTPUT_DIR override: on Kaggle src is copied shallow (/kaggle/working/src) so the
# 3-up default would land on /kaggle — the notebook pins it to /kaggle/working/output.
OUT = ensure_local_path(os.environ.get("OUTPUT_DIR", os.path.join(_ROOT, "output")))
WORK = ensure_local_path(os.environ.get("WORK_DIR", os.path.join(_ROOT, "work")))
DEFAULT_CAPS = dict(token_cap=1000, addr_cap=2000, sn_cap=2000, fn_cap=20000, c3_cap=2000)
PAIR_BUDGET = 100_000
MAX_CANDIDATES = 2_000

FEATURES = ["name_jaccard_raw", "name_jaccard_normalized", "name_jaccard_suffix_stripped",
            "name_cos", "addr_cos",
            "addr_jaccard", "token_overlap_count",
            "street_number_match", "missing_address_flag",
            "len_delta_name", "len_delta_addr"]


def load_model():
    import lightgbm as lgb
    model = lgb.Booster(model_file=os.path.join(WORK, "model.txt"))
    cfg = pd.read_json(os.path.join(WORK, "model_config.json"), typ="series").to_dict()
    return model, float(cfg["threshold"]), cfg


def group_rows(s1_idx_arr, cand_id_arr, s1_ids, keep_mask=None):
    """Per-entity id lists for one chunk. Returns (all_ids_by_entity, kept_ids_by_entity)."""
    all_by = {}
    kept_by = {}
    if keep_mask is not None:
        for s, c, k in zip(s1_idx_arr, cand_id_arr, keep_mask):
            e = s1_ids[s]
            all_by.setdefault(e, []).append(c)
            if k:
                kept_by.setdefault(e, []).append(c)
    else:
        for s, c in zip(s1_idx_arr, cand_id_arr):
            all_by.setdefault(s1_ids[s], []).append(c)
    return all_by, kept_by


def write_rows(f, rows_by_entity, chunk_ids):
    """Append per-entity rows for exactly the entities of this chunk (order-preserving)."""
    for eid in chunk_ids:
        ids = dict.fromkeys(rows_by_entity.get(eid, ()))  # dedupe, keep order
        f.write(f"{eid}\t{','.join(ids)}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dummy", action="store_true", help="rule-based scorer, no model file")
    ap.add_argument("--threshold", type=float, default=None,
                    help="override the match threshold (dummy mode/testing only; "
                         "real mode reads it from model_config.json)")
    ap.add_argument("--s1-limit", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    split = "test"
    s1 = pd.read_parquet(os.path.join(WORK, f"{split}_s1.parquet"))
    pool = pd.concat([
        pd.read_parquet(os.path.join(WORK, f"{split}_s2.parquet")),
        pd.read_parquet(os.path.join(WORK, f"{split}_s3.parquet")),
    ], ignore_index=True)
    if args.s1_limit:
        s1 = s1.iloc[:args.s1_limit]
    print(f"test: s1={len(s1):,} pool={len(pool):,}", flush=True)

    if args.dummy:
        threshold = args.threshold if args.threshold is not None else 0.75
        cfg = {"threshold": threshold, "topk": 500, "proxy": "max"}
    else:
        if args.threshold is not None:
            print("NOTE: --threshold ignored in real mode (model_config.json is authoritative)")
        model, threshold, cfg = load_model()
        print(f"model loaded: threshold={threshold} val_macro_f05={cfg.get('val_macro_f05')}")
        DEFAULT_CAPS.update(cfg.get("caps", {}))

    index, dfs = build_index(pool)
    allowed = make_allowed(dfs, **DEFAULT_CAPS)
    del dfs

    s1_ids = s1.entity_id.values
    pool_ids = pool.entity_id.values
    os.makedirs(OUT, exist_ok=True)
    m_path = os.path.join(OUT, "matching_results.tsv")
    c_path = os.path.join(OUT, "candidate_pairs.tsv")
    n_cand = n_kept = 0
    subset_violations = 0
    topk = int(cfg.get("topk", 500))
    proxy = str(cfg.get("proxy", "max"))
    max_candidates = int(cfg.get("max_candidates", MAX_CANDIDATES))
    stage_config = {"caps": DEFAULT_CAPS, "max_candidates": max_candidates,
                    "pair_budget": PAIR_BUDGET, "topk": topk, "proxy": proxy,
                    "threshold": threshold, "s1_limit": args.s1_limit}
    prepare_stage(OUT, "test_inference", stage_config)
    output_files = [m_path, c_path]
    W = 2_048
    for base in range(0, len(s1), W):
        chunk = s1.iloc[base:base + W]
        chunk_ids = chunk.entity_id.values
        cands = candidates_chunk(chunk, base, index, allowed, max_candidates)
        cands = batch_topk_prune(chunk, pool, cands, topk, proxy, PAIR_BUDGET)
        counts = [len(c) for c in cands]
        si = np.repeat(np.arange(base, base + len(chunk), dtype=np.int64), counts)
        ci = np.concatenate(cands) if cands else np.empty(0, dtype=np.int64)
        del cands
        keep = np.zeros(len(si), dtype=bool)
        cand_by, kept_by = {}, {}
        for pair_base in range(0, len(si), PAIR_BUDGET):
            pair_end = min(pair_base + PAIR_BUDGET, len(si))
            si_part, ci_part = si[pair_base:pair_end], ci[pair_base:pair_end]
            df = compute_chunk(s1, pool, si_part, ci_part)
            if args.dummy:
                proba = 0.5 * df.name_cos.fillna(0) + 0.5 * df.addr_cos.fillna(0)
            else:
                proba = model.predict(df[FEATURES])
            keep_part = np.asarray(proba >= threshold)
            keep[pair_base:pair_end] = keep_part
            cand_part, kept_part = group_rows(si_part, pool_ids[ci_part], s1_ids, keep_part)
            for e, ids in cand_part.items(): cand_by.setdefault(e, []).extend(ids)
            for e, ids in kept_part.items(): kept_by.setdefault(e, []).extend(ids)
            del df, si_part, ci_part, proba, keep_part, cand_part, kept_part
        for e, ms in kept_by.items():
            if not set(ms).issubset(set(cand_by.get(e, ()))):
                raise AssertionError(f"match outside candidate list for {e}")
        first = base == 0
        # one row per chunk entity, exactly once: to_csv append, header on first chunk
        pd.DataFrame([(e, ",".join(dict.fromkeys(kept_by.get(e, ())))) for e in chunk_ids],
                     columns=["source1_entity_id", "matched_entity_ids"]) \
            .to_csv(m_path, sep="\t", index=False, header=first,
                    mode="w" if first else "a", encoding="utf-8", lineterminator="\n")
        pd.DataFrame([(e, ",".join(dict.fromkeys(cand_by.get(e, ())))) for e in chunk_ids],
                     columns=["source1_entity_id", "candidate_entity_ids"]) \
            .to_csv(c_path, sep="\t", index=False, header=first,
                    mode="w" if first else "a", encoding="utf-8", lineterminator="\n")
        n_cand += len(si)
        n_kept += int(keep.sum()) if keep is not None else 0
        print(f"  {base + len(chunk):,}/{len(s1):,} rows: {len(si):,} pairs, "
              f"{n_kept:,} kept ({time.time()-t0:.0f}s)", flush=True)
        del si, ci, cand_by, kept_by
    print(f"wrote {len(s1):,} entity rows ({n_cand:,} candidate pairs, {n_kept:,} kept, "
          f"zero subset violations) in {time.time()-t0:.0f}s")
    write_manifest(OUT, "test_inference", stage_config, output_files, len(s1))


if __name__ == "__main__":
    main()
