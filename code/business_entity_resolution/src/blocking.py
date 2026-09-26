# -*- coding: utf-8 -*-
"""Phase 2: multi-key inverted-index blocking (hand-rolled, per PLAN.md — Splink deferred).

Country is a hard filter (0 cross-country true matches in the 7.64M-pair ground truth,
exhaustively verified). Keys are country-scoped; name fields are already
transliterated+normalized by prep.py, so every name key IS the transliterated index:

  c3 : first 3 chars of suffix-stripped normalized name      (prefix key)
  t  : each name token, df-capped                            (token key)
  a  : each address token (len>=3), df-capped                (address key)
  sn : street number                                          (address key)
  fn : full normalized suffix-stripped name, df-capped        (exact-name key)

df caps are applied at QUERY time (make_allowed) so one index serves all cap settings.
"""
import sys, os, time, argparse
from array import array
import numpy as np
import pandas as pd

WORK = os.environ.get("WORK_DIR",
                      os.path.join(os.path.dirname(__file__), "..", "..", "..", "work"))


# ---------- index construction ----------

def build_index(pool, chunk_rows=1_000_000):
    """Single-pass index: key -> array('i') of pool row idx. Memory-lean (no big temporaries).

    df caps are applied at QUERY time via make_allowed(); the index stores postings
    for every key so different caps can be tried without rebuilding.
    """
    from collections import defaultdict
    t0 = time.time()
    index = defaultdict(lambda: array("i"))
    dfs = {"t": {}, "a": {}, "sn": {}, "fn": {}}
    for base in range(0, len(pool), chunk_rows):
        chunk = pool.iloc[base:base + chunk_rows]
        # df counts (vectorized explode, chunk-local)
        t = pd.DataFrame({"toks": chunk.name_t.str.split().values,
                          "c": chunk.country.values}).explode("toks").dropna()
        for k, v in ("t|" + t.c + "|" + t.toks).value_counts().items():
            dfs["t"][k] = dfs["t"].get(k, 0) + int(v)
        a = pd.DataFrame({"toks": chunk.addr_t.str.split().values,
                          "c": chunk.country.values}).explode("toks").dropna()
        a = a[a.toks.str.len() >= 3]
        for k, v in ("a|" + a.c + "|" + a.toks).value_counts().items():
            dfs["a"][k] = dfs["a"].get(k, 0) + int(v)
        snm = chunk.stno != ""
        for k, v in ("sn|" + chunk.country[snm] + "|" + chunk.stno[snm]).value_counts().items():
            dfs["sn"][k] = dfs["sn"].get(k, 0) + int(v)
        for k, v in ("fn|" + chunk.country + "|" + chunk.name_s).value_counts().items():
            dfs["fn"][k] = dfs["fn"].get(k, 0) + int(v)
        # postings (python loop; C-speed array.append)
        cols = zip(chunk.country.values, chunk.name_s.values, chunk.name_t.values,
                   chunk.addr_t.values, chunk.stno.values)
        for j, (c, ns, nt, at, sn) in enumerate(cols):
            i = base + j
            if len(ns) >= 3:
                index[f"c3|{c}|{ns[:3]}"].append(i)
            for tok in nt.split():
                index[f"t|{c}|{tok}"].append(i)
            for tok in at.split():
                if len(tok) >= 3:
                    index[f"a|{c}|{tok}"].append(i)
            if sn:
                index[f"sn|{c}|{sn}"].append(i)
            if ns:
                index[f"fn|{c}|{ns}"].append(i)
        print(f"  chunk {base + len(chunk):,}/{len(pool):,} rows ({time.time()-t0:.0f}s)", flush=True)
    print(f"  index: {len(index):,} keys, {sum(len(v) for v in index.values()):,} postings "
          f"in {time.time()-t0:.0f}s", flush=True)
    return index, dfs


def make_allowed(dfs, token_cap, addr_cap, sn_cap, fn_cap):
    allowed = {
        "t": {k for k, v in dfs["t"].items() if v <= token_cap},
        "a": {k for k, v in dfs["a"].items() if v <= addr_cap},
        "sn": {k for k, v in dfs["sn"].items() if v <= sn_cap},
        "fn": {k for k, v in dfs["fn"].items() if v <= fn_cap},
    }
    print(f"  allowed keys: t={len(allowed['t'])} a={len(allowed['a'])} "
          f"sn={len(allowed['sn'])} fn={len(allowed['fn'])}")
    return allowed


# ---------- querying ----------

def record_keys(country, name_s, name_t, addr_t, stno, allowed):
    out = []
    if len(name_s) >= 3:
        out.append(f"c3|{country}|{name_s[:3]}")
    cf = f"{country}|"
    a_t = allowed["t"]; a_a = allowed["a"]; a_sn = allowed["sn"]; a_fn = allowed["fn"]
    for t in name_t.split():
        k = f"t|{cf}{t}"
        if k in a_t:
            out.append(k)
    for t in addr_t.split():
        if len(t) >= 3:
            k = f"a|{cf}{t}"
            if k in a_a:
                out.append(k)
    if stno:
        k = f"sn|{cf}{stno}"
        if k in a_sn:
            out.append(k)
    k = f"fn|{cf}{name_s}"
    if k in a_fn:
        out.append(k)
    return out


def candidates_chunk(s1_chunk, base, index, allowed):
    """Return list[np.ndarray] of candidate pool-idx arrays, one per S1 row."""
    out = []
    cols = zip(s1_chunk.country.values, s1_chunk.name_s.values, s1_chunk.name_t.values,
               s1_chunk.addr_t.values, s1_chunk.stno.values)
    for country, name_s, name_t, addr_t, stno in cols:
        cand = set()
        for k in record_keys(country, name_s, name_t, addr_t, stno, allowed):
            lst = index.get(k)
            # length-check, not truthiness: bool(np.array([0])) is False (element-
            # dependent) and bool(np.array([0,1])) raises — postings must be judged
            # by size regardless of container type (array('i') vs np arrays in tests)
            if lst is not None and len(lst):
                cand.update(lst)
        out.append(np.fromiter(cand, dtype=np.int64, count=len(cand)) if cand else
                   np.empty(0, dtype=np.int64))
    return out


def proxy_scores(s1, pool, cand):
    """Per-candidate (name_lev_suffix_stripped, addr_lev) arrays via rapidfuzz.
    Name-only for candidates whose pool address is blank (or when the S1 address
    is blank) — p_addr falls back to p_name there."""
    from rapidfuzz import process as rf_process
    from rapidfuzz.distance import Levenshtein
    ns = pool.name_s.values[cand]
    an = pool.addr_n.values[cand]
    q = [s1["name_s"]] * cand.size
    p_name = np.array(rf_process.cpdist(q, ns,
                      scorer=Levenshtein.normalized_similarity, workers=-1))
    both = (an != "") & bool(s1["addr_n"])
    p_addr = p_name.copy()          # blank addr -> name-only proxy
    if both.any():
        qa = [s1["addr_n"]] * int(both.sum())
        p_addr[both] = rf_process.cpdist(qa, an[both],
                       scorer=Levenshtein.normalized_similarity, workers=-1)
    return p_name, p_addr


PROXY_FNS = {
    "mean": lambda pn, pa: 0.5 * (pn + pa),
    "max": np.maximum,   # ranks a pair high if EITHER field nearly matches — the
                         # Jaccard-0 cohort (renames, script switches) needs this
}


def proxy_topk_indices(s1, pool, cand, k, mode="mean"):
    """Top-k candidate pool indices by the named proxy. Deterministic (stable
    sort). Safe on empty/short inputs. Same proxy gate2b measures recall@K with."""
    if cand.size == 0 or k <= 0:
        return cand[:max(k, 0)] if cand.size else cand
    if cand.size <= k:
        return cand
    p_name, p_addr = proxy_scores(s1, pool, cand)
    proxy = PROXY_FNS[mode](p_name, p_addr)
    order = np.argsort(-proxy, kind="stable")
    return cand[order[:k]]


def generate_pairs(s1, index, allowed, pool=None, topk=0, proxy="mean",
                   chunk_rows=200_000):
    """Yield (s1_idx_block, cand_idx_block) numpy arrays of equal length.

    topk > 0 (with pool passed) truncates each S1 row's candidates to the top-K by
    proxy_topk_indices. Rows at or under K are untouched.
    """
    for base in range(0, len(s1), chunk_rows):
        chunk = s1.iloc[base:base + chunk_rows]
        cands = candidates_chunk(chunk, base, index, allowed)
        if topk and pool is None:
            raise ValueError("pool is required when topk is enabled")
        if topk:
            for j, c in enumerate(cands):
                if c.size > topk:
                    cands[j] = proxy_topk_indices(chunk.iloc[j], pool, c, topk,
                                                  proxy)
        counts = [len(c) for c in cands]
        s1_side = np.repeat(np.arange(base, base + len(chunk), dtype=np.int64), counts)
        cand_side = np.concatenate(cands) if cands else np.empty(0, dtype=np.int64)
        yield s1_side, cand_side


def load_pool(split, columns=None):
    # default: ALL columns — compute_chunk needs raw name/address, name_n/name_nt
    # token bags, etc. Slim id-only lookups live in {split}_pool_ids.parquet.
    if columns is not None:
        return pd.concat([
            pd.read_parquet(f"{WORK}/{split}_s2.parquet", columns=columns),
            pd.read_parquet(f"{WORK}/{split}_s3.parquet", columns=columns),
        ], ignore_index=True)
    return pd.concat([
        pd.read_parquet(f"{WORK}/{split}_s2.parquet"),
        pd.read_parquet(f"{WORK}/{split}_s3.parquet"),
    ], ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--token-cap", type=int, default=1000)
    ap.add_argument("--addr-cap", type=int, default=2000)
    ap.add_argument("--sn-cap", type=int, default=2000)
    ap.add_argument("--fn-cap", type=int, default=20000)
    ap.add_argument("--topk", type=int, default=0,
                    help="per-S1 candidate cap via proxy ranking (0 = off); "
                         "volume knob chosen by gate2b recall@K")
    ap.add_argument("--proxy", default="mean", choices=sorted(PROXY_FNS),
                    help="proxy aggregation for top-K ranking")
    ap.add_argument("--s1-limit", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    pool = load_pool(args.split)
    s1 = pd.read_parquet(f"{WORK}/{args.split}_s1.parquet")
    if args.s1_limit:
        s1 = s1.iloc[:args.s1_limit]
    print(f"pool={len(pool):,} s1={len(s1):,}")

    index, dfs = build_index(pool)
    allowed = make_allowed(dfs, args.token_cap, args.addr_cap, args.sn_cap, args.fn_cap)
    del dfs
    t0 = time.time()
    out_dir = args.out or os.path.join(WORK, f"{args.split}_cands")
    os.makedirs(out_dir, exist_ok=True)
    total = 0
    for part, (a, b) in enumerate(generate_pairs(s1, index, allowed,
                                                 pool=pool, topk=args.topk,
                                                 proxy=args.proxy)):
        pd.DataFrame({"s1_idx": a, "cand_idx": b}).to_parquet(
            os.path.join(out_dir, f"part{part:03d}.parquet"), index=False)
        total += len(a)
        print(f"  part{part:03d}: {len(a):,} pairs (total {total:,}, {time.time()-t0:.0f}s)",
              flush=True)
    print(f"{total:,} candidate pairs ({total/len(s1):.1f}/S1 avg) -> {out_dir}")


if __name__ == "__main__":
    main()
