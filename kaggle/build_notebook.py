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


_PIPELINE_MODULES = {"normalize", "blocking", "features", "evaluate", "infer", "prep", "pipeline_state"}


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
real run. Stage B selects caps/proxy/candidate cutoff automatically; running the notebook
top-to-bottom completes stages A-H in one session. Same-session reuse requires a matching
completion manifest and intact outputs.""")

# ---- Section 0: setup + config ---------------------------------------------
cells.append(cell("code", """# S0a — setup: deps (install missing only, never pin over the image), paths, globals
import importlib, subprocess, sys
for mod, package in (("pandas", "pandas"), ("numpy", "numpy"),
                     ("pyarrow", "pyarrow"), ("lightgbm", "lightgbm"),
                     ("sklearn", "scikit-learn")):
    try:
        importlib.import_module(mod)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", package], check=True)
import glob, os, re, json, time, shutil, zipfile, hashlib, argparse, ast
import numpy as np
import pandas as pd
import lightgbm as lgb

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
# ground truth: support nested and flat layouts, and refresh if the attached data changes.
gt_src = os.path.join(DATA_DIR, "train", "train_ground_truth.tsv")
if not os.path.exists(gt_src):
    gt_src = os.path.join(DATA_DIR, "train_ground_truth.tsv")
if not os.path.isfile(gt_src):
    raise FileNotFoundError("train_ground_truth.tsv not found in flat or split dataset layout")
def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()
if not os.path.exists(GT) or file_sha256(GT) != file_sha256(gt_src):
    os.makedirs(WORK, exist_ok=True)
    shutil.copy2(gt_src, GT)
os.makedirs(WORK, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
print("DATA_DIR:", DATA_DIR)
print("WORK:", WORK, "| OUTPUT_DIR:", OUTPUT_DIR)"""))

cells.append(cell("code", """# S0b — CONFIG (the only cell you edit between stages)
SAMPLE_N = 0           # >0: small-slice end-to-end test. 0 = FULL
TOPK = 20              # emergency relaunch: largest K that fits the remaining clock
PROXY = "mean"         # 'mean' or 'max' — full-scale Stage B: mean > max at every K
NEG_RATIO = 0.05      # train-side negative keep-rate; val rows + positives always kept
CAPS = {"token_cap": 1000, "addr_cap": 2000, "sn_cap": 2000, "fn_cap": 20000, "c3_cap": 2000}
MAX_CANDIDATES = 2000
PAIR_BUDGET = 100_000
KS = [10, 20, 30, 50, 100, 200] if SAMPLE_N == 0 else [5, 10, 20, 50]
_SETTINGS = [(1000, 2000, 2000, 20000, 2000), (200, 300, 500, 5000, 1000)] if SAMPLE_N == 0 else \\
            [(50, 100, 100, 1000, 200), (20, 30, 50, 300, 100)]
FEATURES = ["name_jaccard_raw", "name_jaccard_normalized", "name_jaccard_suffix_stripped",
            "name_cos", "addr_cos", "addr_jaccard", "token_overlap_count",
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
state_for_prep = extract(parse("pipeline_state.py"))
cells.append(cell("code", "# S2 — Stage A: prep all 6 TSVs + slim id lookups (inlined prep.py)\n" + state_for_prep + "\n" + seg + """

prep_cached = {}
for split in ("train", "test"):
    inputs = [os.path.join(DATA_DIR, split, f"{split}_source{i}.tsv") for i in (1, 2, 3)]
    if not os.path.exists(inputs[0]):
        inputs = [os.path.join(DATA_DIR, f"{split}_source{i}.tsv") for i in (1, 2, 3)]
    prep_cfg = {"split": split, "sample_n": SAMPLE_N,
                "ground_truth_sha256": file_sha256(gt_src),
                "source_bytes": [os.path.getsize(p) for p in inputs]}
    prep_manifest = os.path.join(WORK, f"_{split}_prep_manifest.json")
    prep_outputs = [f"{WORK}/{split}_{n}.parquet" for n in ("s1", "s2", "s3")] + \
                   [f"{WORK}/{split}_s1_ids.parquet", f"{WORK}/{split}_pool_ids.parquet"]
    if files_complete(prep_manifest, f"{split}_prep", prep_cfg, prep_outputs):
        print(f"Stage A {split}: complete and config-matched")
        prep_cached[split] = True
        continue
    prep_cached[split] = False
    for n, label in (("s1", "S1"), ("s2", "S2"), ("s3", "S3")):
        src_tsv = os.path.join(DATA_DIR, split, f"{split}_source{n[1]}.tsv")
        if not os.path.exists(src_tsv):
            src_tsv = os.path.join(DATA_DIR, f"{split}_source{n[1]}.tsv")
        out_pq = f"{WORK}/{split}_{n}.parquet"
        prep_file(src_tsv, out_pq, label)
if SAMPLE_N > 0:
    for split in ("train", "test"):
        if prep_cached.get(split, False):
            continue
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
for split in ("train", "test"):
    if prep_cached.get(split, False):
        continue
    inputs = [os.path.join(DATA_DIR, split, f"{split}_source{i}.tsv") for i in (1, 2, 3)]
    if not os.path.exists(inputs[0]):
        inputs = [os.path.join(DATA_DIR, f"{split}_source{i}.tsv") for i in (1, 2, 3)]
    prep_cfg = {"split": split, "sample_n": SAMPLE_N,
                "ground_truth_sha256": file_sha256(gt_src),
                "source_bytes": [os.path.getsize(p) for p in inputs]}
    prep_outputs = [f"{WORK}/{split}_{n}.parquet" for n in ("s1", "s2", "s3")] + \
                   [f"{WORK}/{split}_s1_ids.parquet", f"{WORK}/{split}_pool_ids.parquet"]
    save_slim_ids(split)
    write_files_manifest(os.path.join(WORK, f"_{split}_prep_manifest.json"),
                         f"{split}_prep", prep_cfg, prep_outputs,
                         sum(len(pd.read_parquet(p, columns=["entity_id"])) for p in prep_outputs[:3]))
print("Stage A DONE")
"""))

# ---- Section 3: blocking (inline) ------------------------------------------
blk_tree = parse("blocking.py")
seg = extract(blk_tree, skip_consts={"WORK"})
state_seg = extract(parse("pipeline_state.py"))
cells.append(cell("code", "# S3a — blocking functions (inlined blocking.py)\nimport math\n" + state_seg + "\n" + seg))

cells.append(cell("code", """# S3b — Stage B: recall@K sweep on the val split (K x proxy x caps)
import gc

def make_val_split(save=True):
    s1 = pd.read_parquet(f"{WORK}/train_s1.parquet")
    target_n = min(20_000, len(s1))
    gt_local = pd.read_csv(GT, sep="\t", dtype=str, keep_default_na=False)
    degree = gt_local.set_index("source1_entity_id").matched_entity_ids.map(
        lambda x: min(4, len([m for m in x.split(",") if m])))
    labels = s1.entity_id.map(degree).fillna(0).astype(int)
    rng = np.random.default_rng(42)
    chosen = []
    for _, group in s1.assign(_stratum=labels).groupby("_stratum", sort=True):
        ids = group.entity_id.to_numpy()
        take = min(len(ids), int(round(target_n * len(ids) / max(len(s1), 1))))
        chosen.extend(rng.choice(ids, size=take, replace=False).tolist() if take else [])
    if len(chosen) < target_n:
        remaining = s1.loc[~s1.entity_id.isin(chosen), "entity_id"].to_numpy()
        chosen.extend(rng.choice(remaining, size=target_n - len(chosen),
                                 replace=False).tolist())
    chosen = np.asarray(chosen[:target_n], dtype=object)
    rng.shuffle(chosen)
    midpoint = len(chosen) // 2
    calibration_ids = set(chosen[:midpoint])
    tuning_ids = set(chosen[midpoint:])
    assert not calibration_ids.intersection(tuning_ids)
    assert calibration_ids.union(tuning_ids) == set(chosen.tolist())
    if save:
        pd.Series(sorted(calibration_ids)).to_csv(
            f"{WORK}/candidate_calibration_ids.txt", index=False, header=False)
        pd.Series(sorted(tuning_ids)).to_csv(f"{WORK}/val_s1_ids.txt", index=False, header=False)
    return calibration_ids

val_ids = make_val_split()
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
    for base in range(0, len(val), 512):
        chunk = val.iloc[base:base + 512]
        base_cands = candidates_chunk(chunk, base, index, allowed, MAX_CANDIDATES)
        for p in PROXY_FNS:
            ranked = batch_topk_prune(chunk, pool, base_cands, max(KS), p, PAIR_BUDGET,
                                       rank_all=True)
            for j, c in enumerate(ranked):
                truth = tp_by_s1.get(base + j, ())
                for k in KS:
                    top = c[:k]
                    sizes[p][k] += len(top)
                    if truth:
                        hits[p][k] += len(set(top.tolist()) & truth)
        del base_cands, ranked
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
best_recall = float(res_df.recall.max())
best_row = res_df.loc[res_df.recall.idxmax()]
cfg_row = res_df[(res_df.proxy == PROXY) & (res_df.topk == TOPK)]
if len(cfg_row):
    chosen = cfg_row.loc[cfg_row.recall.idxmax()]
    print(f"Stage B DONE — CONFIG honored: proxy={PROXY} topk={TOPK} "
          f"recall={chosen.recall:.4%} avg_cands={chosen.avg_cands_per_s1:.1f} "
          f"(table best recall {best_recall:.4%})")
else:
    print(f"Stage B WARNING: CONFIG proxy={PROXY} topk={TOPK} not in sweep (KS={KS}) — "
          f"kept as-is; table best: proxy={best_row.proxy} topk={int(best_row.topk)} "
          f"recall={best_recall:.4%}")
del index, pool, val
gc.collect()"""))

cells.append(cell("code", """# S3c — Stage C: train candidate generation with chosen TOPK/PROXY (test-side is
# generated in Section 7 by the SAME functions — no drift)
out_dir = f"{WORK}/train_cands"
stage_c_cfg = {"caps": CAPS, "max_candidates": MAX_CANDIDATES, "topk": TOPK,
               "proxy": PROXY, "pair_budget": PAIR_BUDGET}
if not prepare_stage(out_dir, "train_candidates", stage_c_cfg):
    print("Stage C complete and config-matched")
else:
    pool = load_pool("train")
    s1 = pd.read_parquet(f"{WORK}/train_s1.parquet",
                         columns=["entity_id", "country", "name_s", "name_t",
                                  "addr_n", "addr_t", "stno"])
    index, dfs = build_index(pool)
    allowed = make_allowed(dfs, **CAPS)
    del dfs
    total = 0
    candidate_files = []
    for part, (a, b, rank) in enumerate(generate_pairs(s1, index, allowed, pool=pool,
            topk=TOPK, proxy=PROXY, pair_budget=PAIR_BUDGET,
            max_candidates=MAX_CANDIDATES)):
        part_path = os.path.join(out_dir, f"part{part:05d}.parquet")
        pd.DataFrame({"s1_idx": a, "cand_idx": b, "cand_rank": rank}).to_parquet(part_path, index=False)
        candidate_files.append(part_path)
        total += len(a)
        print(f"  part{part:03d}: {len(a):,} (total {total:,})", flush=True)
    print(f"Stage C DONE: {total:,} train candidate pairs")
    write_manifest(out_dir, "train_candidates", stage_c_cfg, candidate_files, total)
    del index, pool
    gc.collect()"""))

# ---- Section 4: features (inline) ------------------------------------------
feat_tree = parse("features.py")
seg = extract(feat_tree, skip_consts={"WORK", "GT"})
feat_tree = parse("features.py")
seg = extract(feat_tree, skip_consts={"WORK", "GT"})
cells.append(cell("code", "# S4 — Stage D: pairwise features (inlined features.py)\n" + seg + """
import glob as _glob
cand_files = sorted(_glob.glob(f"{WORK}/train_cands/part*.parquet"))
s1 = pd.read_parquet(f"{WORK}/train_s1.parquet")
pool = load_pool("train")
gt = pd.read_csv(GT, sep="\\t", dtype=str, keep_default_na=False)
gt_map = {e: (set(m.split(",")) if m else set())
          for e, m in zip(gt.source1_entity_id, gt.matched_entity_ids)}
val_ids = set(pd.read_csv(f"{WORK}/val_s1_ids.txt", header=None)[0].values)
val_id_array = np.asarray(sorted(val_ids), dtype=object)
feat_dir = f"{WORK}/feats_train"
feat_cfg = {"neg_ratio": NEG_RATIO, "seed": 42, "chunk_pairs": 100_000,
            "candidate_config": stage_c_cfg,
            "candidate_manifest_bytes": os.path.getsize(f"{WORK}/train_cands/_stage_manifest.json"),
            "validation_ids": sorted(val_ids),
            "ground_truth_sha256": file_sha256(gt_src)}
run_stage_d = prepare_stage(feat_dir, "train_features", feat_cfg)
if not run_stage_d:
    print("Stage D complete and config-matched")
else:
    feat_files = []
    feature_total = 0
    s1_ids = s1.entity_id.values
    pool_ids = pool.entity_id.values
    t00 = time.time()
    part = 0
    for cand_file in cand_files:
        cand_part = pd.read_parquet(cand_file)
        for base in range(0, len(cand_part), 100_000):
            t0 = time.time()
            blk = cand_part.iloc[base:base + 100_000]
            labels = label_candidate_pairs(blk, s1_ids, pool_ids, gt_map)
            blk, labels = select_candidate_rows(blk, labels, s1_ids, val_id_array,
                                                NEG_RATIO, seed=42 + part)
            if len(blk):
                df = compute_chunk(s1, pool, blk.s1_idx.values, blk.cand_idx.values)
                df["is_match"] = labels
                if "cand_rank" in blk:
                    df["cand_rank"] = blk.cand_rank.to_numpy(dtype=np.int32)
                out_path = f"{feat_dir}/part{part:06d}.parquet"
                df.to_parquet(out_path, index=False)
                feat_files.append(out_path); feature_total += len(df)
                print(f"  part{part:06d}: {len(df):,} rows ({time.time()-t0:.0f}s)", flush=True)
            part += 1
        del cand_part
    write_manifest(feat_dir, "train_features", feat_cfg, feat_files, feature_total)
    print(f"Stage D DONE in {time.time()-t00:.0f}s; sampled before feature calculation")"""))

# ---- Section 5: adversarial gate (inline) ----------------------------------
cells.append(cell("code", """# S5 — Stage E: adversarial identical-name gate (must PASS)
files = sorted(glob.glob(f"{WORK}/feats_train/part*.parquet"))
adv_parts, pos_parts = [], []
for f in files:
    df_part = pd.read_parquet(f, columns=["name_jaccard_suffix_stripped", "addr_jaccard", "is_match"])
    adv_parts.append(df_part.loc[(df_part.name_jaccard_suffix_stripped >= 0.99) &
                                 (df_part.is_match == 0), "addr_jaccard"].dropna().to_numpy())
    pos_parts.append(df_part.loc[(df_part.name_jaccard_suffix_stripped >= 0.99) &
                                 (df_part.is_match == 1), "addr_jaccard"].dropna().to_numpy())
    del df_part
adv = np.concatenate(adv_parts) if adv_parts else np.empty(0)
adv_pos = np.concatenate(pos_parts) if pos_parts else np.empty(0)
if not len(adv) or not len(adv_pos):
    print("Stage E SKIP: no identical-name pairs in this slice (tiny sample mode)")
else:
    gap = float(np.median(adv_pos) - np.median(adv))
    print(f"adversarial subset: {len(adv):,} neg / {len(adv_pos):,} pos")
    print(f"addr_jaccard medians: pos={np.median(adv_pos):.4f} "
          f"adv_neg={np.median(adv):.4f}")
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
seg_tr = extract(tr_tree, only={"make_model_config", "sweep_cutoffs_thresholds",
                                "choose_best_operating_point"})
cells.append(cell("code", "# S6b-pre — model config contract (inlined train.py)\n" + seg_tr))
cells.append(cell("code", """# S6b — Stage F: LightGBM (ES on train-fold slice) + F_0.5 threshold sweep
train_cfg = {"candidate_config": stage_c_cfg, "topk": TOPK, "proxy": PROXY,
             "caps": CAPS, "neg_ratio": NEG_RATIO,
             "ground_truth_sha256": file_sha256(gt_src),
             "feature_manifest_bytes": os.path.getsize(f"{WORK}/feats_train/_stage_manifest.json"),
             "tuning_ids": sorted(val_ids)}
train_manifest = os.path.join(WORK, "_train_model_manifest.json")
model_outputs = [f"{WORK}/model.txt", f"{WORK}/model_config.json"]
if files_complete(train_manifest, "train_model", train_cfg, model_outputs):
    print("Stage F complete and config-matched")
else:
    for stale_model in model_outputs:
        if os.path.exists(stale_model): os.remove(stale_model)
    feature_files = sorted(glob.glob(f"{WORK}/feats_train/part*.parquet"))
    assert feature_files, "no feature files — Stage D did not produce training data"
    with open(f"{WORK}/feats_train/_stage_manifest.json", encoding="utf-8") as mf:
        n_features = int(json.load(mf)["total_rows"])
    assert n_features > 0, "feature manifest reports zero candidate rows"
    X = np.memmap(f"{WORK}/_train_X.float32", mode="w+", dtype=np.float32,
                  shape=(n_features, len(FEATURES)))
    y = np.memmap(f"{WORK}/_train_y.int8", mode="w+", dtype=np.int8, shape=(n_features,))
    s1_index = np.memmap(f"{WORK}/_train_s1.int32", mode="w+", dtype=np.int32, shape=(n_features,))
    candidate_index = np.memmap(f"{WORK}/_train_cand.int32", mode="w+", dtype=np.int32, shape=(n_features,))
    candidate_rank = np.memmap(f"{WORK}/_train_rank.int32", mode="w+", dtype=np.int32, shape=(n_features,))
    cursor = 0
    for feature_file in feature_files:
        part_df = pd.read_parquet(feature_file)
        end = cursor + len(part_df)
        X[cursor:end] = part_df[FEATURES].to_numpy(dtype=np.float32)
        y[cursor:end] = part_df.is_match.to_numpy(dtype=np.int8)
        s1_index[cursor:end] = part_df.s1_idx.to_numpy(dtype=np.int32)
        candidate_index[cursor:end] = part_df.cand_idx.to_numpy(dtype=np.int32)
        candidate_rank[cursor:end] = part_df.cand_rank.to_numpy(dtype=np.int32)
        cursor = end
        del part_df
    for mapped in (X, y, s1_index, candidate_index, candidate_rank): mapped.flush()
    s1_ids = pd.read_parquet(f"{WORK}/train_s1_ids.parquet")["entity_id"].values
    pool_ids = pd.read_parquet(f"{WORK}/train_pool_ids.parquet")["entity_id"].values
    val_ids_arr = pd.read_csv(f"{WORK}/val_s1_ids.txt", header=None)[0].values
    id_to_s1 = {e: i for i, e in enumerate(s1_ids)}
    val_s1_idx = np.asarray([id_to_s1[e] for e in val_ids_arr], dtype=np.int32)
    val_mask = np.isin(s1_index, val_s1_idx)
    gt_all = pd.read_csv(GT, sep="\\t", dtype=str, keep_default_na=False)
    truth_by = {e: (set(m.split(",")) if m else set())
                for e, m in zip(gt_all.source1_entity_id, gt_all.matched_entity_ids)}
    tr_mask = ~val_mask
    train_only = np.unique(s1_index[tr_mask])
    es_ids = np.random.default_rng(1).choice(
        train_only, size=max(1, len(train_only) // 10), replace=False)
    es_mask = np.isin(s1_index, es_ids) & tr_mask
    fit_mask = tr_mask & ~es_mask
    Xtr, ytr = X[fit_mask], y[fit_mask]
    print(f"fit: {len(Xtr):,} (pos {float(ytr.mean()):.4%}) | es: {int(es_mask.sum()):,}")
    params = {"objective": "binary", "metric": "average_precision", "learning_rate": 0.08,
              "num_leaves": 128, "min_data_in_leaf": 200, "feature_fraction": 0.9,
              "bagging_fraction": 0.9, "bagging_freq": 1, "num_threads": -1,
              "scale_pos_weight": float((1 - ytr.mean()) / max(ytr.mean(), 1e-9))}
    dtr = lgb.Dataset(Xtr, ytr)
    dval = lgb.Dataset(X[es_mask], y[es_mask], reference=dtr)
    model = lgb.train(params, dtr, valid_sets=[dval], num_boost_round=2000,
                      callbacks=[lgb.early_stopping(100)])
    v_p = model.predict(X[val_mask])
    v_s1 = s1_ids[s1_index[val_mask]]
    v_c = pool_ids[candidate_index[val_mask]]
    tune_ids = list(dict.fromkeys(val_ids_arr.tolist()))
    assert len(tune_ids) == len(val_ids_arr), "tuning entity IDs must be unique"
    eid = {e: i for i, e in enumerate(tune_ids)}
    entity_index = np.fromiter((eid[e] for e in v_s1), dtype=np.int32, count=len(v_s1))
    ranks = candidate_rank[val_mask]
    labels = y[val_mask]
    pool_id_set = set(pool_ids.tolist())
    true_counts = np.asarray([len(truth_by.get(e, set()) & pool_id_set) for e in tune_ids], dtype=np.int32)
    operating = sweep_cutoffs_thresholds(v_p, entity_index, ranks, labels, true_counts,
        cutoffs=sorted(set([100, 200, 300, 500, TOPK])),
        thresholds=np.round(np.arange(0.10, 0.951, 0.02), 3))
    sw = operating
    print(sw.to_string(index=False))
    best = choose_best_operating_point(sw, tolerance=0.005)
    pred = {e: set() for e in tune_ids}
    sel = (v_p >= best.threshold) & (ranks < int(best.topk))
    for e, m_ in zip(v_s1[sel], v_c[sel]): pred[e].add(m_)
    truth_tune = {e: truth_by.get(e, ()) for e in tune_ids}
    buckets = bucket_report(pred, truth_tune)
    print(f"BEST topk={best.topk} threshold={best.threshold} val_macro_f05={best.macro_f05:.4f} "
          f"avg_candidates={best.avg_candidates_per_s1:.1f} candidate_recall={best.candidate_recall:.4%}")
    print("per-bucket F_0.5:", buckets)
    model.save_model(f"{WORK}/model.txt")
    model_cfg = make_model_config(best.threshold, best.macro_f05, buckets)
    model_cfg["topk"] = int(best.topk)
    model_cfg["proxy"] = PROXY
    model_cfg["caps"] = CAPS
    model_cfg["max_candidates"] = MAX_CANDIDATES
    model_cfg["candidate_recall"] = float(best.candidate_recall)
    model_cfg["avg_candidates_per_s1"] = float(best.avg_candidates_per_s1)
    pd.Series(model_cfg) \\
        .to_json(f"{WORK}/model_config.json")
    write_files_manifest(train_manifest, "train_model", train_cfg, model_outputs, n_features)
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
TOPK = int(cfg.get("topk", TOPK))
PROXY = str(cfg.get("proxy", PROXY))
CAPS = dict(cfg.get("caps", CAPS))
MAX_CANDIDATES = int(cfg.get("max_candidates", MAX_CANDIDATES))
s1_ids_t = t1.entity_id.values
pool_ids_t = test_pool.entity_id.values
m_path = os.path.join(OUTPUT_DIR, "matching_results.tsv")
c_path = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
inference_cfg = {"caps": CAPS, "max_candidates": MAX_CANDIDATES, "topk": TOPK,
                 "proxy": PROXY, "pair_budget": PAIR_BUDGET,
                 "threshold": threshold, "model_bytes": os.path.getsize(f"{WORK}/model.txt")}
prepare_stage(OUTPUT_DIR, "test_inference", inference_cfg)
first = True
n_cand = n_kept = 0
t0 = time.time()
W = 2_048
for base in range(0, len(t1), W):
    chunk = t1.iloc[base:base + W]
    chunk_ids = chunk.entity_id.values
    cands = candidates_chunk(chunk, base, index_t, allowed_t, MAX_CANDIDATES)
    if TOPK:
        cands = batch_topk_prune(chunk, test_pool, cands, TOPK, PROXY, PAIR_BUDGET)
    counts = [len(c) for c in cands]
    si = np.repeat(np.arange(base, base + len(chunk), dtype=np.int64), counts)
    ci = np.concatenate(cands) if cands else np.empty(0, dtype=np.int64)
    del cands
    cand_by, kept_by = {}, {}
    if len(si):
        offsets = np.concatenate(([0], np.cumsum(counts)))
        for j, e in enumerate(chunk_ids):
            cand_by[e] = pool_ids_t[ci[offsets[j]:offsets[j + 1]]].tolist()
        for pair_base in range(0, len(si), PAIR_BUDGET):
            pair_end = min(pair_base + PAIR_BUDGET, len(si))
            si_part, ci_part = si[pair_base:pair_end], ci[pair_base:pair_end]
            df = compute_chunk(t1, test_pool, si_part, ci_part)
            keep = model.predict(df[FEATURES]) >= threshold
            _, kept_part = group_rows(si_part, pool_ids_t[ci_part], s1_ids_t, keep)
            for e, matches in kept_part.items():
                kept_by.setdefault(e, []).extend(matches)
            del df, si_part, ci_part, keep, kept_part
        for e, ms in kept_by.items():
            assert set(ms).issubset(set(cand_by.get(e, ()))), f"match outside candidates: {e}"
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
    n_cand += len(si); n_kept += sum(len(v) for v in kept_by.values())
    print(f"  {base + len(chunk):,}/{len(t1):,}: {len(si):,} pairs, {n_kept:,} kept "
          f"({time.time()-t0:.0f}s)", flush=True)
    del si, ci, cand_by, kept_by
del index_t, test_pool
gc.collect()
print(f"Stage G inference DONE: {len(t1):,} rows, {n_cand:,} pairs, {n_kept:,} kept")"""))

cells.append(cell("code", """# S7b — inline submission validator (reimplements utils/validate_submission.py checks)
def validate_submission(matching_path, candidate_path, test_dir):
    required = set()
    source1_path = os.path.join(test_dir, "test", "test_source1.tsv")
    if not os.path.exists(source1_path):
        source1_path = os.path.join(test_dir, "test_source1.tsv")
    with open(source1_path, encoding="utf-8") as f:
        next(f)
        required = {l.split("\\t", 1)[0].strip() for l in f if l.strip()}
    seen, dup_rows, intra, self_m, wrong_pfx = set(), set(), set(), set(), set()
    n_rows = empties = 0
    errs = []
    with open(matching_path, encoding="utf-8") as f, open(candidate_path, encoding="utf-8") as cf:
        header = f.readline().rstrip("\\n").split("\\t")
        cheader = cf.readline().rstrip("\\n").split("\\t")
        assert header == ["source1_entity_id", "matched_entity_ids"], header
        assert cheader == ["source1_entity_id", "candidate_entity_ids"], cheader
        for line in f:
            s1, tab, rest = line.partition("\\t")
            if not tab:
                continue
            n_rows += 1
            candidate_line = cf.readline()
            assert candidate_line, f"candidate file ended before matching row {s1}"
            cand_s1, ctab, cand_rest = candidate_line.rstrip("\\n").partition("\\t")
            assert ctab and cand_s1 == s1, f"candidate/matching row mismatch for {s1}"
            if s1 in seen:
                dup_rows.add(s1)
            seen.add(s1)
            ids = rest.rstrip("\\n").split(",") if rest.strip() else []
            cand_ids = cand_rest.rstrip("\\n").split(",") if cand_rest.strip() else []
            if not set(ids).issubset(set(cand_ids)):
                errs.append(f"matches outside candidate list for {s1}")
            if len(ids) != len(set(ids)):
                intra.add(s1)
            for mid in set(ids):
                if mid.startswith("S1-"):
                    self_m.add(mid)
                elif not mid.startswith(("S2-", "S3-")):
                    wrong_pfx.add(mid)
            if not ids:
                empties += 1
        if cf.readline():
            errs.append("candidate file has extra rows")
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

ok = validate_submission(m_path, c_path, DATA_DIR)
assert ok, 'submission validation failed'
write_manifest(OUTPUT_DIR, "test_inference", inference_cfg,
               [m_path, c_path], len(t1))
"""))

# ---- Section 8: package -----------------------------------------------------
cells.append(cell("code", """# S8 — Stage H: bundle kaggle_return.zip
out = "/kaggle/working/kaggle_return.zip" if os.path.isdir("/kaggle/working") \\
    else os.path.join(WORK, "kaggle_return.zip")
if os.path.exists(out):
    os.remove(out)
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
    for f in os.listdir(OUTPUT_DIR):
        if f == MANIFEST_NAME:
            continue
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
