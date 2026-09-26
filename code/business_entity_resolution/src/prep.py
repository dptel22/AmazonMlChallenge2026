# -*- coding: utf-8 -*-
"""Phase 2a: normalize all source files once, cache to parquet for blocking/features."""
import sys, os, time
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from normalize import norm_name, norm_text, addr_tokens, street_number

BASE = os.environ.get("DATA_DIR",
                      os.path.join(os.path.dirname(__file__), "..", "..", "..",
                                   "student_resource", "dataset"))
WORK = os.environ.get("WORK_DIR",
                      os.path.join(os.path.dirname(__file__), "..", "..", "..", "work"))
os.makedirs(WORK, exist_ok=True)


def prep_file(path, out, source_label=None):
    t0 = time.time()
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    nm = df.business_name.map(norm_name)
    df["name_n"] = nm.str[0]
    df["name_s"] = nm.str[1]
    df["name_t"] = df.name_s.map(lambda s: " ".join(sorted(set(s.split()))))  # suffix-stripped bag
    df["name_nt"] = df.name_n.map(lambda s: " ".join(sorted(set(s.split()))))  # unstripped bag
    df["addr_n"] = df.business_address.map(norm_text)
    df["addr_t"] = df.addr_n.map(lambda s: " ".join(sorted(set(s.split()))))
    df["stno"] = df.business_address.map(street_number)
    if source_label:
        df["src"] = source_label
    df.to_parquet(out, index=False)
    print(f"{os.path.basename(path)}: {len(df)} rows in {time.time()-t0:.0f}s -> {out}")


def save_slim_ids(split):
    """Slim entity_id lookups (train.py --full / Kaggle reads these, not full parquets)."""
    pd.read_parquet(f"{WORK}/{split}_s1.parquet", columns=["entity_id"]) \
        .to_parquet(f"{WORK}/{split}_s1_ids.parquet", index=False)
    pd.concat([pd.read_parquet(f"{WORK}/{split}_s2.parquet", columns=["entity_id"]),
               pd.read_parquet(f"{WORK}/{split}_s3.parquet", columns=["entity_id"])],
              ignore_index=True).to_parquet(f"{WORK}/{split}_pool_ids.parquet", index=False)
    print(f"slim id lookups written for {split}")


if __name__ == "__main__":
    for split in ("train", "test"):
        prep_file(f"{BASE}/{split}/{split}_source1.tsv", f"{WORK}/{split}_s1.parquet", "S1")
        prep_file(f"{BASE}/{split}/{split}_source2.tsv", f"{WORK}/{split}_s2.parquet", "S2")
        prep_file(f"{BASE}/{split}/{split}_source3.tsv", f"{WORK}/{split}_s3.parquet", "S3")
        save_slim_ids(split)
