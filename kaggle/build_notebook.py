# -*- coding: utf-8 -*-
"""Build ONE self-contained notebook: every pipeline function inlined via AST
extraction from code/business_entity_resolution/src/*.py (no transcription drift).

Prints notebook JSON to stdout. Build:
    python kaggle/build_notebook.py > kaggle/kaggle_pipeline.ipynb
"""
import ast, json, io, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "code", "business_entity_resolution", "src")


def parse(name):
    return ast.parse(io.open(os.path.join(SRC, name), encoding="utf-8").read())


_PIPELINE_MODULES = {"normalize", "blocking", "features", "evaluate", "infer", "prep"}


def extract(tree, skip_main=True, only=None, skip_consts=()):
    """Unparse selected top-level nodes: functions (only=name filter) and
    module-level assigns/loops (skip_consts filters assigned names).
    Pipeline-module imports are dropped (their code is inlined in other cells)."""
    out = []
    for node in tree.body:
        if skip_main and isinstance(node, ast.If):
            continue  # the `if __name__ == "__main__"` gate
        if isinstance(node, ast.Import):
            names = [a.name.split(".")[0] for a in node.names]
            if any(n in _PIPELINE_MODULES for n in names):
                continue
            out.append(ast.unparse(node))
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] in _PIPELINE_MODULES:
                continue
            out.append(ast.unparse(node))
        elif isinstance(node, ast.FunctionDef):
            if only is None or node.name in only:
                out.append(ast.unparse(node))
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            out.append(ast.unparse(node))   # keep imports: idempotent, needed by inlined fns
        elif isinstance(node, (ast.Assign, ast.AugAssign, ast.For, ast.Try, ast.With)):
            names = []
            tgts = node.targets if isinstance(node, ast.Assign) else []
            for t in tgts:
                if isinstance(t, ast.Name):
                    names.append(t.id)
                elif isinstance(t, ast.Tuple):
                    names += [e.id for e in t.elts if isinstance(e, ast.Name)]
            if only is None:
                if any(n in skip_consts for n in names):
                    continue
                out.append(ast.unparse(node))
            elif any(n in only for n in names):
                out.append(ast.unparse(node))
    return "\n\n".join(out)


def cell(kind, source, cid=None):
    c = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        c["execution_count"] = None
        c["outputs"] = []
    return c


cells = []
md = lambda s: cells.append(cell("markdown", s))

md("""# BER full pipeline — ONE self-contained notebook (stages A-H)

Every function lives in this notebook (inlined from src via AST build — no external
.py, no subprocess). Only external reads: the competition dataset/ TSVs.

| Cell | Stage | Does | ~Time (full) |
|---|---|---|---|
| 4 | A prep | normalize 6 TSVs -> work/*.parquet + slim id lookups | 30-40 min |
| 5 | B sweep | recall@K table (K x proxy x caps) on val split -> pick TOPK/PROXY | 40-60 min |
| 6 | C cands | train candidates with TOPK/PROXY | 40-80 min |
| 7 | D feats | train features + labels + neg downsample | 1-1.5 h |
| 8 | E gate | adversarial identical-name check (PASS required) | 2 min |
| 9 | F train | LightGBM + ES + F_0.5 sweep -> model artifacts | 1-2 h |
| 10 | G infer | test inference + inline validator (PASS required) | ~1 h |
| 11 | H zip | kaggle_return.zip | 1 min |

CONFIG cell: SAMPLE_N>0 runs a small-slice end-to-end test; set SAMPLE_N=0 for the
real run. After Stage B, set TOPK/PROXY from the printed table and re-run C..H.""")

# ---- Section 0: setup + config ---------------------------------------------
cells.append(cell("code", """# S0a — setup: deps (install missing only, never pin over the image), paths, globals
import importlib, subprocess, sys
for mod in ("pandas", "numpy", "pyarrow", "lightgbm", "rapidfuzz"):
    try:
        importlib.import_module(mod)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", mod], check=True)
import glob, os, re, json, time, shutil, zipfile, hashlib, argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
from rapidfuzz import process as rf_process
from rapidfuzz.distance import Levenshtein

# mount detection: handles dataset/train/ + dataset/test/ layout AND flat layout
def find_data_dir():
    if os.environ.get("BER_DATA"):
        return os.path.abspath(os.environ["BER_DATA"])
    for m in glob.glob("/kaggle/input/*"):
        for base, dirs, files in os.walk(m):
            if "train_source1.tsv" in files:
                if "test_source1.tsv" in files:
                    return base                       # flat layout
                parent = os.path.dirname(base.rstrip("/"))
                if os.path.basename(base) == "train" and \\
                        os.path.exists(os.path.join(parent, "test", "test_source1.tsv")):
                    return parent                      # split train/ + test/ layout
    raise FileNotFoundError("train/test source TSVs not found under /kaggle/input — attach the payload dataset")

DATA_DIR = find_data_dir()
BASE = DATA_DIR                       # prep's name for it
if os.environ.get("BER_WORK"):
    WORK = os.path.abspath(os.environ["BER_WORK"])
elif os.path.isdir("/kaggle/working"):
    WORK = "/kaggle/working/work"
else:
    WORK = os.path.abspath("work_nb")
OUTPUT_DIR = os.path.join(os.path.dirname(WORK), "output") if os.environ.get("BER_WORK") \\
    else "/kaggle/working/output"
GT = os.path.join(WORK, "ground_truth.tsv")
# ground truth: copy from the dataset dir into WORK once
if not os.path.exists(GT):
    os.makedirs(WORK, exist_ok=True)
    shutil.copy(os.path.join(DATA_DIR, "train", "train_ground_truth.tsv"), GT)
os.makedirs(WORK, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
print("DATA_DIR:", DATA_DIR)
print("WORK:", WORK, "| OUTPUT_DIR:", OUTPUT_DIR)"""))

cells.append(cell("code", """# S0b — CONFIG (the only cell you edit between stages)
SAMPLE_N = 3000       # >0: small-slice end-to-end test (s1 rows; pool capped 60x). 0 = FULL
TOPK = 200            # per-S1 candidate cap (volume knob) — set from Stage B table
PROXY = "max"         # 'mean' or 'max' — from Stage B table
NEG_RATIO = 0.05      # train-side negative keep-rate; val rows + positives always kept
CAPS = {"token_cap": 1000, "addr_cap": 2000, "sn_cap": 2000, "fn_cap": 20000}
KS = [100, 200, 300, 500] if SAMPLE_N == 0 else [5, 10, 20, 50]
_SETTINGS = [(1000, 2000, 2000, 20000), (200, 300, 500, 5000)] if SAMPLE_N == 0 else \\
            [(50, 100, 100, 1000), (20, 30, 50, 300)]
FEATURES = ["name_jaccard_raw", "name_jaccard_normalized", "name_jaccard_suffix_stripped",
            "name_lev_raw", "name_lev_normalized", "name_lev_suffix_stripped",
            "addr_jaccard", "addr_lev", "token_overlap_count",
            "street_number_match", "missing_address_flag",
            "len_delta_name", "len_delta_addr"]
print("CONFIG: SAMPLE_N", SAMPLE_N, "| TOPK", TOPK, "| PROXY", PROXY, "| NEG_RATIO", NEG_RATIO)"""))

# ---- Section 1: normalize (inline whole module minus __main__ gate) --------
norm_tree = parse("normalize.py")
seg = extract(norm_tree)
cells.append(cell("code", "# S1a — normalization functions (inlined normalize.py)\n"
                        "import unicodedata\n" + seg))

cells.append(cell("code", """# S1b — normalization self-checks (synthetic; France is untestable against train)
assert norm_text("Société Générale") == "societe generale", norm_text("Société Générale")
assert norm_text("Café Crème SARL") == "cafe creme sarl", norm_text("Café Crème SARL")
a = norm_name("Raj Investments LLP")[1]; b = norm_name("ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி")[1]
assert "invest" in b and a == "raj investments", (a, b)
print("S1 self-checks PASS: diacritic fold + Indic transliteration + suffix strip")"""))

# ---- Section 2: prep (inline) ----------------------------------------------
prep_tree = parse("prep.py")
seg = extract(prep_tree, only={"prep_file", "save_slim_ids"})
cells.append(cell("code", "# S2 — Stage A: prep all 6 TSVs + slim id lookups (inlined prep.py)\n" + seg + """

for split in ("train", "test"):
    for n, label in (("s1", "S1"), ("s2", "S2"), ("s3", "S3")):
        src_tsv = os.path.join(DATA_DIR, split, f"{split}_source{n[1]}.tsv")
        out_pq = f"{WORK}/{split}_{n}.parquet"
        if not os.path.exists(out_pq):
            prep_file(src_tsv, out_pq, label)
        else:
            print(f"  {out_pq} exists - skipped")
if SAMPLE_N > 0:
    for split in ("train", "test"):
        for n in ("s1", "s2", "s3"):
            p = f"{WORK}/{split}_{n}.parquet"
            df = pd.read_parquet(p)
            cap = SAMPLE_N if n == "s1" else SAMPLE_N * 30
            if len(df) > cap:
                df.iloc[:cap].to_parquet(p, index=False)
    print(f"SAMPLE mode: s1 capped {SAMPLE_N}, pool capped {SAMPLE_N*30}")
for split in ("train", "test"):
    rows = {n: len(pd.read_parquet(f"{WORK}/{split}_{n}.parquet", columns=["entity_id"]))
            for n in ("s1", "s2", "s3")}
    print(split, rows)
save_slim_ids("train"); save_slim_ids("test")
print("Stage A DONE")
"""))

# ---- Section 3: blocking (inline) ------------------------------------------
blk_tree = parse("blocking.py")
seg = extract(blk_tree, skip_consts={"WORK"})
cells.append(cell("code", "# S3a — blocking functions (inlined blocking.py)\n" + seg))

cells.append(cell("code", """# S3b — Stage B: recall@K sweep on the val split (K x proxy x caps)
import gc

def make_val_split(save=True):
    s1 = pd.read_parquet(f"{WORK}/train_s1.parquet")
    val = s1.entity_id.sample(frac=0.18, random_state=42).sort_values()
    if save:
        val.to_frame().to_csv(f"{WORK}/val_s1_ids.txt", index=False, header=False)
    return set(val)

val_ids = make_val_split()
if SAMPLE_N > 0:
    val_ids = set(sorted(val_ids)[: max(SAMPLE_N // 6, 1)])
pool = load_pool("train")
s1 = pd.read_parquet(f"{WORK}/train_s1.parquet")
val = s1[s1.entity_id.isin(val_ids)].reset_index(drop=True)
del s1
gt = pd.read_csv(GT, sep="\\t", dtype=str, keep_default_na=False)
tp = gt.assign(m=gt.matched_entity_ids.str.split(",")).explode("m")
tp = tp[tp.m.str.strip() != ""]
tp = tp[tp.source1_entity_id.isin(val_ids)]
del gt
print(f"val S1: {len(val):,} | true pairs: {len(tp):,}", flush=True)

eid2i = {e: i for i, e in enumerate(val.entity_id.values)}          # val row index
pid2i = {e: i for i, e in enumerate(pool.entity_id.values)}
tp_set = set((eid2i[s], pid2i[m]) for s, m in zip(tp.source1_entity_id.values, tp.m.values)
             if s in eid2i and m in pid2i)
tp_by_s1 = {}
for s, m in zip(tp.source1_entity_id.values, tp.m.values):
    if s in eid2i and m in pid2i:
        tp_by_s1.setdefault(eid2i[s], set()).add(pid2i[m])
del tp, pid2i, eid2i
total_true = sum(len(v) for v in tp_by_s1.values())

index, dfs = build_index(pool)
results = []
for setting in _SETTINGS:
    allowed = make_allowed(dfs, *setting)
    hits = {p: {k: 0 for k in KS} for p in PROXY_FNS}
    sizes = {p: {k: 0 for k in KS} for p in PROXY_FNS}
    for base in range(0, len(val), 50_000):
        chunk = val.iloc[base:base + 50_000]
        cands = candidates_chunk(chunk, base, index, allowed)
        for j, c in enumerate(cands):
            truth = tp_by_s1.get(base + j, ())
            if c.size == 0:
                continue
            s1row = chunk.iloc[j]
            p_name, p_addr = proxy_scores(s1row, pool, c)
            for p, fn in PROXY_FNS.items():
                proxy = fn(p_name, p_addr)
                order = np.argsort(-proxy, kind="stable")
                for k in KS:
                    top = c[order[:k]]
                    sizes[p][k] += len(top)
                    if truth:
                        hits[p][k] += len(set(top.tolist()) & truth)
        del cands
    for p in PROXY_FNS:
        for k in KS:
            results.append({"caps": str(setting), "proxy": p, "topk": k,
                            "recall": hits[p][k] / total_true if total_true else 0.0,
                            "avg_cands_per_s1": sizes[p][k] / max(len(val), 1),
                            "reduction_vs_naive": sizes[p][k] / max(len(val) * len(pool), 1)})
    del allowed
    gc.collect()
res_df = pd.DataFrame(results)
print(res_df.to_string(index=False))
res_df.to_csv(f"{WORK}/stageB_recall_table.csv", index=False)
if total_true:
    assert res_df.recall.max() > 0.02, "recall ~0: eid2i/positional key bug has recurred"
else:
    print("Stage B WARNING: zero true pairs available in pool for this slice — table is informational only")
best = res_df.loc[res_df.recall.idxmax()]
print(f"Stage B DONE — best: caps={best.caps} proxy={best.proxy} topk={best.topk} "
      f"recall={best.recall:.4%}")
del index, pool, val
gc.collect()"""))

cells.append(cell("code", """# S3c — Stage C: train candidate generation with chosen TOPK/PROXY (test-side is
# generated in Section 7 by the SAME functions — no drift)
out_dir = f"{WORK}/train_cands"
if os.path.isdir(out_dir):
    print("Stage C already done")
else:
    os.makedirs(out_dir, exist_ok=True)
    pool = load_pool("train")
    s1 = pd.read_parquet(f"{WORK}/train_s1.parquet",
                         columns=["entity_id", "country", "name_s", "name_t",
                                  "addr_n", "addr_t", "stno"])
    index, dfs = build_index(pool)
    allowed = make_allowed(dfs, **CAPS)
    del dfs
    total = 0
    for part, (a, b) in enumerate(generate_pairs(s1, index, allowed, pool=pool,
                                                 topk=TOPK, proxy=PROXY)):
        pd.DataFrame({"s1_idx": a, "cand_idx": b}).to_parquet(
            os.path.join(out_dir, f"part{part:03d}.parquet"), index=False)
        total += len(a)
        print(f"  part{part:03d}: {len(a):,} (total {total:,})", flush=True)
    print(f"Stage C DONE: {total:,} train candidate pairs")
    del index, pool
    gc.collect()"""))

# ---- Section 4: features (inline) ------------------------------------------
feat_tree = parse("features.py")
seg = extract(feat_tree, skip_consts={"WORK", "GT"})
feat_tree = parse("features.py")
seg = extract(feat_tree, skip_consts={"WORK", "GT"})
cells.append(cell("code", "# S4 — Stage D: pairwise features (inlined features.py)\n" + seg + """
import glob as _glob
cands = pd.concat([pd.read_parquet(f) for f in
                   sorted(_glob.glob(f"{WORK}/train_cands/part*.parquet"))], ignore_index=True)
s1 = pd.read_parquet(f"{WORK}/train_s1.parquet")
pool = load_pool("train")
gt = pd.read_csv(GT, sep="\\t", dtype=str, keep_default_na=False)
gt_map = {e: (set(m.split(",")) if m else set())
          for e, m in zip(gt.source1_entity_id, gt.matched_entity_ids)}
val_ids = set(pd.read_csv(f"{WORK}/val_s1_ids.txt", header=None)[0].values)
os.makedirs(f"{WORK}/feats_train", exist_ok=True)
s1_ids = s1.entity_id.values
pool_ids = pool.entity_id.values
t00 = time.time()
for part, base in enumerate(range(0, len(cands), 2_000_000)):
    t0 = time.time()
    blk = cands.iloc[base:base + 2_000_000]
    df = compute_chunk(s1, pool, blk.s1_idx.values, blk.cand_idx.values)
    df["is_match"] = [m in gt_map.get(s1_ids[si], ())
                      for si, m in zip(df.s1_idx.values, pool_ids[df.cand_idx.values])]
    df["is_match"] = df.is_match.astype("int8")
    if NEG_RATIO > 0:
        is_val = np.fromiter((s1_ids[si] in val_ids for si in df.s1_idx.values),
                             dtype=bool, count=len(df))
        rng = np.random.default_rng(42 + part)
        keep = is_val | (df.is_match.values == 1) | (rng.random(len(df)) < NEG_RATIO)
        df = df[keep].reset_index(drop=True)
    df.to_parquet(f"{WORK}/feats_train/part{part:03d}.parquet", index=False)
    print(f"  part{part:03d}: {len(df):,} rows kept ({time.time()-t0:.0f}s, "
          f"pos_rate={df.is_match.mean():.4f})", flush=True)
print(f"Stage D DONE in {time.time()-t00:.0f}s")"""))

# ---- Section 5: adversarial gate (inline) ----------------------------------
cells.append(cell("code", """# S5 — Stage E: adversarial identical-name gate (must PASS)
files = sorted(glob.glob(f"{WORK}/feats_train/part*.parquet"))
df = pd.concat([pd.read_parquet(f, columns=[
    "name_lev_suffix_stripped", "addr_jaccard", "is_match"]) for f in files],
    ignore_index=True)
adv = df[(df.name_lev_suffix_stripped >= 0.99) & (df.is_match == 0)]
adv_pos = df[(df.name_lev_suffix_stripped >= 0.99) & (df.is_match == 1)]
if len(adv) == 0 or len(adv_pos) == 0:
    print("Stage E SKIP: no identical-name pairs in this slice (tiny sample mode)")
else:
    gap = adv_pos.addr_jaccard.median() - adv.addr_jaccard.median()
    print(f"adversarial subset: {len(adv):,} neg / {len(adv_pos):,} pos")
    print(f"addr_jaccard medians: pos={adv_pos.addr_jaccard.median():.4f} "
          f"adv_neg={adv.addr_jaccard.median():.4f} | addr_lev pos={adv_pos.addr_lev.median():.4f} "
          f"adv_neg={adv.addr_lev.median():.4f}")
    GATE_E_PASS = gap > 0.1
    assert GATE_E_PASS, 'address features do not separate identical-name negatives - halting pipeline'
    print(f"Stage E {'PASS' if GATE_E_PASS else 'FAIL'}: median addr_jaccard gap = {gap:.4f} "
          f"(threshold 0.1)")
print("Stage E complete")
# S5 gate complete
"""))

# ---- Section 6: evaluate + train (inline) ----------------------------------
ev_tree = parse("evaluate.py")
seg = extract(ev_tree, skip_consts={})
cells.append(cell("code", "# S6a — macro F_0.5 evaluator (inlined evaluate.py) + assertions\n" + seg + """

assert abs(f05_entity(["a", "b", "d"], ["a", "b", "c"]) - 2 / 3) < 1e-12
assert f05_entity([], []) == 1.0 and f05_entity(["x"], []) == 0.0 and f05_entity([], ["x"]) == 0.0
m = macro_f05({"e1": ["a", "b", "d"], "e2": []}, {"e1": ["a", "b", "c"], "e2": []})
assert abs(m - (2 / 3 + 1.0) / 2) < 1e-12
print("F_0.5 assertions passed")"""))

tr_tree = parse("train.py")
seg_tr = extract(tr_tree, only={"make_model_config"})
cells.append(cell("code", "# S6b-pre — model config contract (inlined train.py)\n" + seg_tr))
cells.append(cell("code", """# S6b — Stage F: LightGBM (ES on train-fold slice) + F_0.5 threshold sweep
if os.path.exists(f"{WORK}/model.txt"):
    print("Stage F already done")
else:
    feats = pd.concat([pd.read_parquet(f) for f in
                       sorted(glob.glob(f"{WORK}/feats_train/part*.parquet"))], ignore_index=True)
    s1_ids = pd.read_parquet(f"{WORK}/train_s1_ids.parquet")["entity_id"].values
    pool_ids = pd.read_parquet(f"{WORK}/train_pool_ids.parquet")["entity_id"].values
    s1_ids_arr = s1_ids[feats.s1_idx.values]
    pool_ids_arr = pool_ids[feats.cand_idx.values]
    val_ids_arr = pd.read_csv(f"{WORK}/val_s1_ids.txt", header=None)[0].values
    gt_all = pd.read_csv(GT, sep="\\t", dtype=str, keep_default_na=False)
    truth_by = {e: (set(m.split(",")) if m else set())
                for e, m in zip(gt_all.source1_entity_id, gt_all.matched_entity_ids)}
    tr_mask = ~np.isin(s1_ids_arr, val_ids_arr)
    train_only = np.unique(s1_ids_arr[tr_mask])
    es_ids = np.random.default_rng(1).choice(
        train_only, size=max(1, len(train_only) // 10), replace=False)
    es_mask = np.isin(s1_ids_arr, es_ids) & tr_mask
    fit_mask = tr_mask & ~es_mask
    Xtr, ytr = feats.loc[fit_mask, FEATURES], feats.loc[fit_mask, "is_match"]
    print(f"fit: {len(Xtr):,} (pos {float(ytr.mean()):.4%}) | es: {int(es_mask.sum()):,}")
    params = {"objective": "binary", "metric": "average_precision", "learning_rate": 0.08,
              "num_leaves": 128, "min_data_in_leaf": 200, "feature_fraction": 0.9,
              "bagging_fraction": 0.9, "bagging_freq": 1, "num_threads": -1,
              "scale_pos_weight": float((1 - ytr.mean()) / max(ytr.mean(), 1e-9))}
    dtr = lgb.Dataset(Xtr, ytr)
    dval = lgb.Dataset(feats.loc[es_mask, FEATURES], feats.loc[es_mask, "is_match"], reference=dtr)
    model = lgb.train(params, dtr, valid_sets=[dval], num_boost_round=2000,
                      callbacks=[lgb.early_stopping(100)])
    proba = model.predict(feats[FEATURES])
    vmask = np.isin(s1_ids_arr, val_ids_arr)
    v_p, v_s1, v_c = proba[vmask], s1_ids_arr[vmask], pool_ids_arr[vmask]
    rows = []
    for th in np.arange(0.10, 0.95 + 1e-9, 0.02):
        pred = {e: set() for e in val_ids_arr}
        sel = v_p >= th
        for e, m_ in zip(v_s1[sel], v_c[sel]):
            pred[e].add(m_)
        rows.append({"threshold": round(float(th), 3), "macro_f05": macro_f05(pred, truth_by)})
    sw = pd.DataFrame(rows)
    print(sw.to_string(index=False))
    best = sw.loc[sw.macro_f05.idxmax()]
    pred = {e: set() for e in val_ids_arr}
    sel = (proba >= best.threshold) & vmask
    for e, m_ in zip(s1_ids_arr[sel], pool_ids_arr[sel]):
        pred[e].add(m_)
    buckets = bucket_report(pred, truth_by)
    print(f"BEST threshold={best.threshold} val_macro_f05={best.macro_f05:.4f}")
    print("per-bucket F_0.5:", buckets)
    model.save_model(f"{WORK}/model.txt")
    pd.Series(make_model_config(best.threshold, best.macro_f05, buckets)) \\
        .to_json(f"{WORK}/model_config.json")
    print("Stage F DONE")"""))

# ---- Section 7: inference (inline) -----------------------------------------
inf_tree = parse("infer.py")
seg = extract(inf_tree, skip_consts={"OUT", "WORK", "DEFAULT_CAPS", "FEATURES", "_SRC", "_ROOT"},
              only={"group_rows"})   # Stage G is hand-written; only group_rows is shared
cells.append(cell("code", "# S7 — Stage G: test inference + inline validator (inlined infer.py)\n" + seg + """

test_pool = load_pool("test")
t1 = pd.read_parquet(f"{WORK}/test_s1.parquet")
index_t, dfs_t = build_index(test_pool)
allowed_t = make_allowed(dfs_t, **CAPS)
del dfs_t
model = lgb.Booster(model_file=f"{WORK}/model.txt")
cfg = pd.read_json(f"{WORK}/model_config.json", typ="series").to_dict()
threshold = float(cfg["threshold"])
s1_ids_t = t1.entity_id.values
pool_ids_t = test_pool.entity_id.values
m_path = os.path.join(OUTPUT_DIR, "matching_results.tsv")
c_path = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
first = True
n_cand = n_kept = 0
t0 = time.time()
W = 200_000
for base in range(0, len(t1), W):
    chunk = t1.iloc[base:base + W]
    chunk_ids = chunk.entity_id.values
    cands = candidates_chunk(chunk, base, index_t, allowed_t)
    if TOPK:
        for j, c in enumerate(cands):
            if c.size > TOPK:
                cands[j] = proxy_topk_indices(chunk.iloc[j], test_pool, c, TOPK, PROXY)
    counts = [len(c) for c in cands]
    si = np.repeat(np.arange(base, base + len(chunk), dtype=np.int64), counts)
    ci = np.concatenate(cands) if cands else np.empty(0, dtype=np.int64)
    del cands
    cand_by, kept_by = {}, {}
    if len(si):
        df = compute_chunk(t1, test_pool, si, ci)
        proba = model.predict(df[FEATURES])
        keep = proba >= threshold
        cand_by, kept_by = group_rows(si, pool_ids_t[ci], s1_ids_t, keep)
        del df
        for e, ms in kept_by.items():
            cs = set(cand_by.get(e, ()))
            extra = [m for m in dict.fromkeys(ms) if m not in cs]
            if extra:
                cand_by.setdefault(e, []).extend(extra)
    hdr = first
    pd.DataFrame([(e, ",".join(dict.fromkeys(kept_by.get(e, ())))) for e in chunk_ids],
                 columns=["source1_entity_id", "matched_entity_ids"]) \\
        .to_csv(m_path, sep="\\t", index=False, header=hdr, mode="w" if first else "a",
                encoding="utf-8", lineterminator="\\n")
    pd.DataFrame([(e, ",".join(dict.fromkeys(cand_by.get(e, ())))) for e in chunk_ids],
                 columns=["source1_entity_id", "candidate_entity_ids"]) \\
        .to_csv(c_path, sep="\\t", index=False, header=hdr, mode="w" if first else "a",
                encoding="utf-8", lineterminator="\\n")
    first = False
    n_cand += len(si); n_kept += int(keep.sum()) if len(si) else 0
    print(f"  {base + len(chunk):,}/{len(t1):,}: {len(si):,} pairs, {n_kept:,} kept "
          f"({time.time()-t0:.0f}s)", flush=True)
    del si, ci, cand_by, kept_by
del index_t, test_pool
gc.collect()
print(f"Stage G inference DONE: {len(t1):,} rows, {n_cand:,} pairs, {n_kept:,} kept")"""))

cells.append(cell("code", """# S7b — inline submission validator (reimplements utils/validate_submission.py checks)
def validate_submission(matching_path, test_dir):
    required = set()
    with open(os.path.join(test_dir, "test", "test_source1.tsv"), encoding="utf-8") as f:
        next(f)
        required = {l.split("\\t", 1)[0].strip() for l in f if l.strip()}
    seen, dup_rows, intra, self_m, wrong_pfx = set(), set(), set(), set(), set()
    n_rows = empties = 0
    with open(matching_path, encoding="utf-8") as f:
        header = f.readline().rstrip("\\n").split("\\t")
        assert header == ["source1_entity_id", "matched_entity_ids"], header
        for line in f:
            s1, tab, rest = line.partition("\\t")
            if not tab:
                continue
            n_rows += 1
            if s1 in seen:
                dup_rows.add(s1)
            seen.add(s1)
            ids = rest.rstrip("\\n").split(",") if rest.strip() else []
            if len(ids) != len(set(ids)):
                intra.add(s1)
            for mid in set(ids):
                if mid.startswith("S1-"):
                    self_m.add(mid)
                elif not mid.startswith(("S2-", "S3-")):
                    wrong_pfx.add(mid)
            if not ids:
                empties += 1
    errs = []
    if required - seen:
        errs.append(f"{len(required - seen)} required S1 entities missing")
    if seen - required:
        errs.append(f"{len(seen - required)} unknown S1 rows")
    if dup_rows:
        errs.append(f"{len(dup_rows)} duplicate rows")
    if intra:
        errs.append(f"{len(intra)} lists with duplicate ids")
    if self_m:
        errs.append(f"{len(self_m)} self-matches")
    if wrong_pfx:
        errs.append(f"{len(wrong_pfx)} wrong-prefix ids")
    print(f"  matching_results.tsv: {n_rows} rows ({empties} empty)")
    if errs:
        print("FAIL:", "; ".join(errs))
    else:
        print("PASS — no blocking issues found.")
    return not errs

ok = validate_submission(m_path, DATA_DIR)
assert ok, 'submission validation failed'
"""))

# ---- Section 8: package -----------------------------------------------------
cells.append(cell("code", """# S8 — Stage H: bundle kaggle_return.zip
out = "/kaggle/working/kaggle_return.zip" if os.path.isdir("/kaggle/working") \\
    else os.path.join(WORK, "kaggle_return.zip")
if os.path.exists(out):
    os.remove(out)
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
    for f in os.listdir(OUTPUT_DIR):
        zf.write(os.path.join(OUTPUT_DIR, f), "output/" + f)
    for f in ("model.txt", "model_config.json"):
        p = os.path.join(WORK, f)
        if os.path.exists(p):
            zf.write(p, f)
print(f"kaggle_return.zip: {os.path.getsize(out)/1e6:.1f} MB")
print("DONE — fill Documentation_template.md with the Stage B table + Stage F numbers")"""))

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11"}},
      "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, sys.stdout, indent=1)
