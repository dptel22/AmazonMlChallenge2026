# -*- coding: utf-8 -*-
"""Phase 3: pairwise features over candidate pairs -> parquet feature table.

Columns (per pair):
  name_jaccard_raw / _normalized / _suffix_stripped   token-set Jaccard
  name_lev_raw / _lev_normalized / _lev_suffix        normalized Levenshtein ratio
  addr_jaccard, addr_lev
  token_overlap_count          |name_s tokens ∩|
  street_number_match          1/0; NaN if either side has no extractable number
  missing_address_flag         1 if S2/S3 side address blank
  len_delta_name, len_delta_addr (normalized char-length delta)
  is_match                     only when --labels (train/val); absent for test
"""
import sys, os, time, argparse, glob
import numpy as np
import pandas as pd
from rapidfuzz import process as rf_process
from rapidfuzz.distance import Levenshtein

WORK = os.environ.get("WORK_DIR",
                      os.path.join(os.path.dirname(__file__), "..", "..", "..", "work"))
GT = os.path.join(WORK, "ground_truth.tsv")


def jaccard(a_tokens, b_tokens):
    if not a_tokens and not b_tokens:
        return 1.0
    u = len(a_tokens | b_tokens)
    return len(a_tokens & b_tokens) / u if u else 1.0


def load_feats_data(split):
    s1 = pd.read_parquet(f"{WORK}/{split}_s1.parquet")
    pool = pd.concat([
        pd.read_parquet(f"{WORK}/{split}_s2.parquet"),
        pd.read_parquet(f"{WORK}/{split}_s3.parquet"),
    ], ignore_index=True)
    return s1, pool


def compute_chunk(s1, pool, s1i, ci):
    """All features for one chunk of (s1_idx, cand_idx) arrays."""
    a = s1.iloc[s1i]
    b = pool.iloc[ci]
    # precompute raw-lowercase variants (not cached in parquet)
    a_raw = a.business_name.str.lower().str.split().map(set).values
    b_raw = b.business_name.str.lower().str.split().map(set).values
    a_nt = a.name_t.str.split().map(set).values
    b_nt = b.name_t.str.split().map(set).values
    a_nn = a.name_nt.str.split().map(set).values
    b_nn = b.name_nt.str.split().map(set).values
    a_raws = a.business_name.str.lower().values
    b_raws = b.business_name.str.lower().values
    a_ns = a.name_s.values
    b_ns = b.name_s.values
    a_an = a.addr_n.values
    b_an = b.addr_n.values
    a_at = a.addr_t.str.split().map(set).values
    b_at = b.addr_t.str.split().map(set).values
    a_sn = a.stno.values
    b_sn = b.stno.values
    n = len(s1i)

    name_jac_raw = np.empty(n); name_jac_n = np.empty(n); name_jac_s = np.empty(n)
    overlap = np.empty(n, dtype=np.int32)
    addr_jac = np.full(n, np.nan)
    for i in range(n):
        name_jac_raw[i] = jaccard(a_raw[i], b_raw[i])
        name_jac_n[i] = jaccard(a_nn[i], b_nn[i])   # normalized, NOT suffix-stripped
        name_jac_s[i] = jaccard(a_nt[i], b_nt[i])   # suffix-stripped
        overlap[i] = len(a_nt[i] & b_nt[i])
        if a_an[i] and b_an[i]:
            addr_jac[i] = jaccard(a_at[i], b_at[i])

    lev_raw = np.array(rf_process.cpdist(a_raws, b_raws,
                    scorer=Levenshtein.normalized_similarity, workers=-1))
    lev_n = np.array(rf_process.cpdist(a.name_n.values, b.name_n.values,
                    scorer=Levenshtein.normalized_similarity, workers=-1))
    lev_s = np.array(rf_process.cpdist(a_ns, b_ns,
                    scorer=Levenshtein.normalized_similarity, workers=-1))
    addr_lev = np.full(n, np.nan)
    m = (a_an != "") & (b_an != "")
    addr_lev[m] = np.array(rf_process.cpdist(a_an[m], b_an[m],
                    scorer=Levenshtein.normalized_similarity, workers=-1))

    sn_match = np.full(n, np.nan)
    both = (a_sn != "") & (b_sn != "")
    sn_match[both] = (a_sn[both] == b_sn[both]).astype(float)
    missing_addr = (b_an == "").astype(np.int8)
    len_delta_name = np.abs(a.name_s.str.len().values - b.name_s.str.len().values)
    len_delta_addr = np.abs(a.addr_n.str.len().values - b.addr_n.str.len().values).astype(float)

    out = pd.DataFrame({
        "s1_idx": s1i, "cand_idx": ci,
        "name_jaccard_raw": name_jac_raw, "name_jaccard_normalized": name_jac_n,
        "name_jaccard_suffix_stripped": name_jac_s,
        "name_lev_raw": lev_raw, "name_lev_normalized": lev_n,
        "name_lev_suffix_stripped": lev_s,
        "addr_jaccard": addr_jac, "addr_lev": addr_lev,
        "token_overlap_count": overlap,
        "street_number_match": sn_match,
        "missing_address_flag": missing_addr,
        "len_delta_name": len_delta_name, "len_delta_addr": len_delta_addr,
    })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--labels", action="store_true", help="join is_match from ground truth (train only)")
    ap.add_argument("--neg-ratio", type=float, default=0.0,
                    help="train-side negative downsample: keep each negative with this "
                         "probability (0 = keep all). Val-entity rows and positives are "
                         "always kept so the F_0.5 sweep sees full candidate sets.")
    ap.add_argument("--chunk", type=int, default=2_000_000)
    args = ap.parse_args()

    s1, pool = load_feats_data(args.split)
    cands = pd.concat([pd.read_parquet(f) for f in
                       sorted(glob.glob(os.path.join(WORK, f"{args.split}_cands", "part*.parquet")))],
                      ignore_index=True)
    print(f"s1={len(s1):,} pool={len(pool):,} cand_pairs={len(cands):,}")

    gt_map = None
    val_ids = None
    if args.labels:
        gt = pd.read_csv(GT, sep="\t", dtype=str, keep_default_na=False)
        gt_map = {e: (set(m.split(",")) if m else set())
                  for e, m in zip(gt.source1_entity_id, gt.matched_entity_ids)}
        val_path = os.path.join(WORK, "val_s1_ids.txt")
        val_ids = set(pd.read_csv(val_path, header=None)[0].values) \
            if os.path.exists(val_path) else None

    os.makedirs(f"{WORK}/feats_{args.split}", exist_ok=True)
    s1_ids = s1.entity_id.values
    pool_ids = pool.entity_id.values
    t00 = time.time()
    for part, base in enumerate(range(0, len(cands), args.chunk)):
        t0 = time.time()
        blk = cands.iloc[base:base + args.chunk]
        df = compute_chunk(s1, pool, blk.s1_idx.values, blk.cand_idx.values)
        if gt_map is not None:
            df["is_match"] = [
                m in gt_map.get(s1_ids[si], ())
                for si, m in zip(df.s1_idx.values, pool_ids[df.cand_idx.values])
            ]
            df["is_match"] = df.is_match.astype(np.int8)
            if args.neg_ratio > 0 and val_ids:
                # keep ALL val-entity rows (sweep needs full candidate sets), all
                # positives; downsample train-fold negatives with a seeded rng
                is_val = np.fromiter((s1_ids[si] in val_ids for si in df.s1_idx.values),
                                     dtype=bool, count=len(df))
                rng = np.random.default_rng(42 + part)
                keep = is_val | (df.is_match.values == 1) | \
                       (rng.random(len(df)) < args.neg_ratio)
                df = df[keep].reset_index(drop=True)
                df["is_match"] = df["is_match"].astype(np.int8)
        df.to_parquet(f"{WORK}/feats_{args.split}/part{part:03d}.parquet", index=False)
        pos = df.is_match.mean() if gt_map is not None else float("nan")
        print(f"  part{part:03d}: {len(df):,} rows ({time.time()-t0:.0f}s, pos_rate={pos:.4f})")
    print(f"done in {time.time()-t00:.0f}s -> {WORK}/feats_{args.split}/")


if __name__ == "__main__":
    main()
