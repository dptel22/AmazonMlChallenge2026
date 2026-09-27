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
import sys, os, time, argparse, math
from array import array
import numpy as np
import pandas as pd
from pipeline_state import prepare_stage, write_manifest
from normalize import attach_cosine, frame_cos_rows, cos_pair_scores

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
    dfs = {"c3": {}, "t": {}, "a": {}, "sn": {}, "fn": {}}
    for base in range(0, len(pool), chunk_rows):
        chunk = pool.iloc[base:base + chunk_rows]
        c3_mask = chunk.name_s.str.len() >= 3
        for k, v in ("c3|" + chunk.country[c3_mask] + "|" +
                     chunk.name_s[c3_mask].str[:3]).value_counts().items():
            dfs["c3"][k] = dfs["c3"].get(k, 0) + int(v)
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


def make_allowed(dfs, token_cap, addr_cap, sn_cap, fn_cap, c3_cap=2_000):
    allowed = {
        "c3": {k for k, v in dfs.get("c3", {}).items() if v <= c3_cap},
        "t": {k for k, v in dfs["t"].items() if v <= token_cap},
        "a": {k for k, v in dfs["a"].items() if v <= addr_cap},
        "sn": {k for k, v in dfs["sn"].items() if v <= sn_cap},
        "fn": {k for k, v in dfs["fn"].items() if v <= fn_cap},
    }
    print(f"  allowed keys: c3={len(allowed['c3'])} t={len(allowed['t'])} a={len(allowed['a'])} "
          f"sn={len(allowed['sn'])} fn={len(allowed['fn'])}")
    return allowed


# ---------- querying ----------

def record_keys(country, name_s, name_t, addr_t, stno, allowed):
    out = []
    if len(name_s) >= 3:
        k = f"c3|{country}|{name_s[:3]}"
        if k in allowed.get("c3", ()):
            out.append(k)
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


_BLOCK_WEIGHT = {"c3": 0.25, "t": 1.0, "a": 1.25, "sn": 1.0, "fn": 1.5}


def _stable_topk(ids, scores, k):
    """Select descending scores, breaking all cutoff and output ties by pool ID."""
    ids = np.asarray(ids)
    scores = np.asarray(scores)
    if k <= 0 or not len(ids):
        return np.empty(0, dtype=np.int64)
    if len(ids) > k:
        cutoff = np.partition(scores, len(scores) - k)[len(scores) - k]
        above = np.flatnonzero(scores > cutoff)
        ties = np.flatnonzero(scores == cutoff)
        need = k - len(above)
        if need < len(ties):
            tie_order = np.argpartition(ids[ties], need - 1)[:need]
            ties = ties[tie_order]
        selected = np.concatenate((above, ties))
    else:
        selected = np.arange(len(ids))
    order = np.lexsort((ids[selected], -scores[selected]))
    return ids[selected[order]]


def _row_candidates(country, name_s, name_t, addr_t, stno, index, allowed,
                    max_candidates=2_000):
    """Rank the union cheaply from weighted block evidence before string scoring."""
    scores = {}
    for key in record_keys(country, name_s, name_t, addr_t, stno, allowed):
        postings = index.get(key)
        if postings is None or not len(postings):
            continue
        family = key.split("|", 1)[0]
        weight = _BLOCK_WEIGHT[family] / math.log2(2 + len(postings))
        for pool_idx in postings:
            scores[pool_idx] = scores.get(pool_idx, 0.0) + weight
    if not scores:
        return np.empty(0, dtype=np.int64)
    # Stable, repeatable ties; c3 receives less weight because it is a weak block.
    ids = np.fromiter(scores.keys(), dtype=np.int64, count=len(scores))
    if max_candidates > 0 and len(ids) > max_candidates:
        vals = np.fromiter(scores.values(), dtype=np.float64, count=len(scores))
        ids = _stable_topk(ids, vals, max_candidates)
    return np.sort(ids)


def candidates_chunk(s1_chunk, base, index, allowed, max_candidates=2_000):
    """Return deterministic, evidence-prefiltered pool-idx arrays, one per S1 row."""
    out = []
    cols = zip(s1_chunk.country.values, s1_chunk.name_s.values, s1_chunk.name_t.values,
               s1_chunk.addr_t.values, s1_chunk.stno.values)
    for country, name_s, name_t, addr_t, stno in cols:
        out.append(_row_candidates(country, name_s, name_t, addr_t, stno,
                                   index, allowed, max_candidates))
    return out


def _addr_name_only(p_name, p_addr, query_addrs, cand_addrs):
    """Blank address on either side -> p_addr := p_name (name-only proxy)."""
    both = (np.asarray(cand_addrs) != "") & (np.asarray(query_addrs) != "")
    return np.where(both, p_addr, p_name)


def proxy_scores(s1, pool, cand):
    """Cheap TF-IDF cosine proxy (name, addr) for one S1 row's candidates.
    Name-only where either address is blank — the same rule batch_topk_prune
    applies, so gate2b's recall@K measures the production proxy."""
    if isinstance(s1, pd.Series):
        s1 = s1.to_frame().T
    frame_cos_rows(s1, pool)
    p_name, p_addr = cos_pair_scores(s1, pool, np.arange(len(s1)), cand)
    p_addr = _addr_name_only(p_name, p_addr, s1.addr_n.values[0:1],
                             pool.addr_n.values[cand])
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
    cand = np.sort(cand)
    p_name, p_addr = proxy_scores(s1, pool, cand)
    proxy = PROXY_FNS[mode](p_name, p_addr)
    return _stable_topk(cand, proxy, k)


def batch_topk_prune(rows, pool, cands, topk, proxy_mode="mean", pair_budget=100_000,
                     rank_all=False):
    """Apply top-K with bounded batched scoring calls and deterministic tie-breaking.

    An oversized single row is scored in slices while retaining only its current
    best K, so no scoring call or temporary score array exceeds pair_budget.
    """
    if pair_budget <= 0:
        raise ValueError("pair_budget must be positive")
    out = [np.sort(c) for c in cands]
    need = [j for j, c in enumerate(out) if c.size > topk or (rank_all and c.size)]
    if not need or topk <= 0:
        return out

    row_names = rows.name_s.values
    row_addrs = rows.addr_n.values
    frame_cos_rows(rows, pool)

    def score(row_pos_rep, candidate_ids):
        p_name, p_addr = cos_pair_scores(rows, pool, row_pos_rep, candidate_ids)
        p_addr = _addr_name_only(p_name, p_addr, row_addrs[row_pos_rep],
                                 pool.addr_n.values[candidate_ids])
        return PROXY_FNS[proxy_mode](p_name, p_addr)

    def score_one(j):
        cand = out[j]
        best_ids = np.empty(0, dtype=np.int64)
        best_scores = np.empty(0, dtype=np.float64)
        for base in range(0, len(cand), pair_budget):
            part = cand[base:base + pair_budget]
            n = len(part)
            vals = score(np.full(n, j, dtype=np.int64), part)
            ids = np.concatenate((best_ids, part))
            all_scores = np.concatenate((best_scores, vals))
            best_ids = _stable_topk(ids, all_scores, topk)
            score_by_id = {int(i): float(v) for i, v in zip(ids, all_scores)}
            best_scores = np.fromiter((score_by_id[int(i)] for i in best_ids),
                                      dtype=np.float64, count=len(best_ids))
        out[j] = best_ids

    pending = []
    pending_pairs = 0

    def flush():
        nonlocal pending, pending_pairs
        if not pending:
            return
        counts = np.array([len(out[j]) for j in pending], dtype=np.int64)
        ids = np.concatenate([out[j] for j in pending])
        rep = np.repeat(np.asarray(pending, dtype=np.int64), counts)
        vals = score(rep, ids)
        offsets = np.concatenate(([0], np.cumsum(counts)))
        for i, j in enumerate(pending):
            start, end = offsets[i], offsets[i + 1]
            limit = len(out[j]) if rank_all and len(out[j]) <= topk else topk
            out[j] = _stable_topk(out[j], vals[start:end], limit)
        pending, pending_pairs = [], 0

    for j in need:
        size = len(out[j])
        if size > pair_budget:
            flush()
            score_one(j)
            continue
        if pending and pending_pairs + size > pair_budget:
            flush()
        pending.append(j)
        pending_pairs += size
    flush()
    return out


def generate_pairs(s1, index, allowed, pool=None, topk=0, proxy="mean",
                   chunk_rows=2_048, pair_budget=1_000_000,
                   max_candidates=2_000):
    """Yield aligned (s1_idx, cand_idx, rank) arrays within a pair budget.

    topk > 0 (with pool passed) truncates each S1 row's candidates to the top-K by
    the selected proxy. Candidate rows are scored in bounded batches.
    """
    for base in range(0, len(s1), chunk_rows):
        chunk = s1.iloc[base:base + chunk_rows]
        cands = candidates_chunk(chunk, base, index, allowed, max_candidates)
        if topk and pool is None:
            raise ValueError("pool is required when topk is enabled")
        if topk:
            cands = batch_topk_prune(chunk, pool, cands, topk, proxy,
                                     pair_budget=pair_budget, rank_all=True)
        buf_s1, buf_cand, buf_rank, buf_pairs = [], [], [], 0
        for j, c in enumerate(cands):
            if not c.size:
                continue
            if buf_pairs and buf_pairs + len(c) > pair_budget:
                yield (np.concatenate(buf_s1), np.concatenate(buf_cand),
                       np.concatenate(buf_rank))
                buf_s1, buf_cand, buf_rank, buf_pairs = [], [], [], 0
            buf_s1.append(np.full(len(c), base + j, dtype=np.int64))
            buf_cand.append(c)
            buf_rank.append(np.arange(len(c), dtype=np.int32))
            buf_pairs += len(c)
        if buf_s1:
            yield (np.concatenate(buf_s1), np.concatenate(buf_cand),
                   np.concatenate(buf_rank))


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
    ap.add_argument("--c3-cap", type=int, default=2000)
    ap.add_argument("--max-candidates", type=int, default=2000)
    ap.add_argument("--pair-budget", type=int, default=1_000_000)
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
    allowed = make_allowed(dfs, args.token_cap, args.addr_cap, args.sn_cap,
                           args.fn_cap, args.c3_cap)
    del dfs
    t0 = time.time()
    out_dir = args.out or os.path.join(WORK, f"{args.split}_cands")
    stage_config = {"split": args.split, "token_cap": args.token_cap,
                    "addr_cap": args.addr_cap, "sn_cap": args.sn_cap,
                    "fn_cap": args.fn_cap, "c3_cap": args.c3_cap,
                    "max_candidates": args.max_candidates, "topk": args.topk,
                    "proxy": args.proxy, "pair_budget": args.pair_budget}
    if not prepare_stage(out_dir, f"{args.split}_candidates", stage_config):
        print(f"candidate stage complete and config-matched: {out_dir}")
        return
    total = 0
    output_files = []
    for part, (a, b, rank) in enumerate(generate_pairs(s1, index, allowed,
                                                 pool=pool, topk=args.topk,
                                                 proxy=args.proxy,
                                                 pair_budget=args.pair_budget,
                                                 max_candidates=args.max_candidates)):
        part_path = os.path.join(out_dir, f"part{part:05d}.parquet")
        pd.DataFrame({"s1_idx": a, "cand_idx": b,
                      "cand_rank": rank}).to_parquet(part_path, index=False)
        output_files.append(part_path)
        total += len(a)
        print(f"  part{part:03d}: {len(a):,} pairs (total {total:,}, {time.time()-t0:.0f}s)",
              flush=True)
    print(f"{total:,} candidate pairs ({total/len(s1):.1f}/S1 avg) -> {out_dir}")
    write_manifest(out_dir, f"{args.split}_candidates", stage_config,
                   output_files, total)


if __name__ == "__main__":
    main()
