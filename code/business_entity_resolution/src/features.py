# -*- coding: utf-8 -*-
"""Phase 3: pairwise features over candidate pairs -> parquet feature table.

Columns (per pair):
  name_jaccard_raw / _normalized / _suffix_stripped   token-set Jaccard
  name_cos / addr_cos       sparse TF-IDF cosine similarities
  addr_jaccard
  token_overlap_count          |name_s tokens ∩|
  street_number_match          1/0; NaN if either side has no extractable number
  missing_address_flag         1 if S2/S3 side address blank
  len_delta_name, len_delta_addr (normalized char-length delta)
  is_match                     only when --labels (train/val); absent for test
"""
import sys, os, time, argparse, glob, hashlib
import numpy as np
import pandas as pd
from pipeline_state import prepare_stage, write_manifest
from normalize import attach_cosine, frame_cos_rows, cos_pair_scores

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


def _tok_set(v):
    return frozenset(v.split()) if isinstance(v, str) and v else frozenset()


def ensure_entity_feats(frame):
    """Per-entity token sets + scalars, computed once and cached on the frame.

    compute_chunk used to re-split the same entity strings for every 100k-pair
    chunk and take 100k random rows out of multi-million-row arrow-backed
    frames each time - that was the 47 s/part Stage D rate on Kaggle (and the
    memory death). With this cache, feature calculation is pure positional
    lookup + set math. The owner tag guards against attrs propagating through
    iloc (pandas 3.0), same as the cosine cache."""
    cached = frame.attrs.get("_ef")
    if cached is not None and cached.get("owner") == id(frame):
        return cached
    an = frame.addr_n.fillna("").to_numpy(dtype=object)
    cached = {
        "owner": id(frame),
        "raw": frame.business_name.str.lower().map(_tok_set).to_numpy(dtype=object),
        "nt": frame.name_t.map(_tok_set).to_numpy(dtype=object),
        "nn": frame.name_nt.map(_tok_set).to_numpy(dtype=object),
        "at": frame.addr_t.map(_tok_set).to_numpy(dtype=object),
        "an": an,
        "sn": frame.stno.fillna("").to_numpy(dtype=object),
        "len_s": frame.name_s.str.len().to_numpy(dtype=np.float64),
        "len_a": frame.addr_n.str.len().to_numpy(dtype=np.float64),
    }
    frame.attrs["_ef"] = cached
    return cached


def compute_chunk(s1, pool, s1i, ci):
    """All features for one chunk of (s1_idx, cand_idx) arrays."""
    ef1 = ensure_entity_feats(s1)
    efp = ensure_entity_feats(pool)
    e_raw, p_raw = ef1["raw"], efp["raw"]
    e_nt, p_nt = ef1["nt"], efp["nt"]
    e_nn, p_nn = ef1["nn"], efp["nn"]
    e_at, p_at = ef1["at"], efp["at"]
    e_an, p_an = ef1["an"], efp["an"]
    n = len(s1i)

    name_jac_raw = np.empty(n); name_jac_n = np.empty(n); name_jac_s = np.empty(n)
    overlap = np.empty(n, dtype=np.int32)
    addr_jac = np.full(n, np.nan)
    for i in range(n):
        s, c = s1i[i], ci[i]
        name_jac_raw[i] = jaccard(e_raw[s], p_raw[c])
        name_jac_n[i] = jaccard(e_nn[s], p_nn[c])   # normalized, NOT suffix-stripped
        name_jac_s[i] = jaccard(e_nt[s], p_nt[c])   # suffix-stripped
        overlap[i] = len(e_nt[s] & p_nt[c])
        if e_an[s] and p_an[c]:
            addr_jac[i] = jaccard(e_at[s], p_at[c])

    attach_cosine(pool)
    frame_cos_rows(s1, pool)
    name_cos, addr_cos_raw = cos_pair_scores(s1, pool, s1i, ci)
    both_addr = (e_an[s1i] != "") & (p_an[ci] != "")
    addr_cos = np.where(both_addr, addr_cos_raw, np.nan)

    e_sn, p_sn = ef1["sn"], efp["sn"]
    sn_match = np.full(n, np.nan)
    both = (e_sn[s1i] != "") & (p_sn[ci] != "")
    sn_match[both] = (e_sn[s1i][both] == p_sn[ci][both]).astype(float)
    missing_addr = (p_an[ci] == "").astype(np.int8)
    len_delta_name = np.abs(ef1["len_s"][s1i] - efp["len_s"][ci])
    len_delta_addr = np.abs(ef1["len_a"][s1i] - efp["len_a"][ci])

    out = pd.DataFrame({
        "s1_idx": s1i, "cand_idx": ci,
        "name_jaccard_raw": name_jac_raw, "name_jaccard_normalized": name_jac_n,
        "name_jaccard_suffix_stripped": name_jac_s,
        "name_cos": name_cos,
        "addr_jaccard": addr_jac, "addr_cos": addr_cos,
        "token_overlap_count": overlap,
        "street_number_match": sn_match,
        "missing_address_flag": missing_addr,
        "len_delta_name": len_delta_name, "len_delta_addr": len_delta_addr,
    })
    return out


def label_candidate_pairs(cands, s1_ids, pool_ids, gt_map):
    """Label a bounded candidate block before expensive feature calculation."""
    s1_idx = cands.s1_idx.to_numpy(dtype=np.int64, copy=False)
    cand_idx = cands.cand_idx.to_numpy(dtype=np.int64, copy=False)
    return np.fromiter(
        (pool_ids[ci] in gt_map.get(s1_ids[si], ())
         for si, ci in zip(s1_idx, cand_idx)),
        dtype=np.int8, count=len(cands))


def select_candidate_rows(cands, labels, s1_ids, val_ids, neg_ratio, seed):
    """Keep validation rows, all positives, and sampled train-fold negatives."""
    if not len(cands):
        return cands, labels
    s1_idx = cands.s1_idx.to_numpy(dtype=np.int64, copy=False)
    if isinstance(val_ids, (set, frozenset)):
        val_ids = sorted(val_ids)
    val_id_array = np.asarray(val_ids, dtype=object)
    is_val = np.isin(s1_ids[s1_idx], val_id_array)
    rng = np.random.default_rng(seed)
    keep = is_val | (labels == 1) | (rng.random(len(cands)) < neg_ratio)
    return cands.loc[keep].reset_index(drop=True), labels[keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--labels", action="store_true", help="join is_match from ground truth (train only)")
    ap.add_argument("--neg-ratio", type=float, default=0.0,
                    help="train-side negative downsample: keep each negative with this "
                         "probability (0 = keep all). Val-entity rows and positives are "
                         "always kept so the F_0.5 sweep sees full candidate sets.")
    ap.add_argument("--chunk", type=int, default=100_000,
                    help="maximum candidate pairs per feature calculation")
    args = ap.parse_args()

    s1, pool = load_feats_data(args.split)
    cand_files = sorted(glob.glob(os.path.join(WORK, f"{args.split}_cands", "part*.parquet")))
    if not cand_files:
        raise FileNotFoundError(f"no candidate files under {WORK}/{args.split}_cands/")
    print(f"s1={len(s1):,} pool={len(pool):,} candidate_parts={len(cand_files):,}")

    gt_map = None
    val_ids = None
    if args.labels:
        gt = pd.read_csv(GT, sep="\t", dtype=str, keep_default_na=False)
        gt_map = {e: (set(m.split(",")) if m else set())
                  for e, m in zip(gt.source1_entity_id, gt.matched_entity_ids)}
        val_path = os.path.join(WORK, "val_s1_ids.txt")
        if not os.path.exists(val_path):
            raise FileNotFoundError(f"validation IDs are required for sampled labels: {val_path}")
        val_ids = set(pd.read_csv(val_path, header=None)[0].values)
        val_id_array = np.asarray(sorted(val_ids), dtype=object)
        with open(GT, "rb") as f:
            gt_hash = hashlib.file_digest(f, "sha256").hexdigest()
    else:
        gt_hash = ""

    feat_dir = f"{WORK}/feats_{args.split}"
    stage_config = {"split": args.split, "labels": args.labels,
                    "neg_ratio": args.neg_ratio, "chunk_pairs": args.chunk,
                    "validation_ids": sorted(val_ids) if val_ids is not None else [],
                    "ground_truth_sha256": gt_hash,
                    "candidate_manifest_bytes": os.path.getsize(os.path.join(
                        WORK, f"{args.split}_cands", "_stage_manifest.json"))}
    if not prepare_stage(feat_dir, f"features_{args.split}", stage_config):
        print(f"feature stage complete and config-matched: {feat_dir}")
        return
    s1_ids = s1.entity_id.values
    pool_ids = pool.entity_id.values
    t00 = time.time()
    part = 0
    pair_total = 0
    feature_total = 0
    output_files = []
    for cand_file in cand_files:
        cand_part = pd.read_parquet(cand_file)
        pair_total += len(cand_part)
        for base in range(0, len(cand_part), args.chunk):
            t0 = time.time()
            blk = cand_part.iloc[base:base + args.chunk]
            labels = None
            if gt_map is not None:
                labels = label_candidate_pairs(blk, s1_ids, pool_ids, gt_map)
                if args.neg_ratio > 0:
                    blk, labels = select_candidate_rows(
                        blk, labels, s1_ids, val_id_array, args.neg_ratio, seed=42 + part)
            if len(blk):
                df = compute_chunk(s1, pool, blk.s1_idx.values, blk.cand_idx.values)
                if labels is not None:
                    df["is_match"] = labels
                if "cand_rank" in blk:
                    df["cand_rank"] = blk.cand_rank.to_numpy(dtype=np.int32, copy=False)
                output_path = os.path.join(feat_dir, f"part{part:06d}.parquet")
                df.to_parquet(output_path, index=False)
                output_files.append(output_path)
                feature_total += len(df)
                pos = float(df.is_match.mean()) if labels is not None else float("nan")
                print(f"  part{part:06d}: {len(df):,} rows ({time.time()-t0:.0f}s, "
                      f"pos_rate={pos:.4f})", flush=True)
            part += 1
        del cand_part
    print(f"done in {time.time()-t00:.0f}s -> {feat_dir}/; "
          f"read {pair_total:,} candidate pairs in bounded chunks")
    write_manifest(feat_dir, f"features_{args.split}", stage_config,
                   output_files, feature_total)


if __name__ == "__main__":
    main()
