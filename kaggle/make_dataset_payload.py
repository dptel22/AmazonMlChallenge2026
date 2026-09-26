# -*- coding: utf-8 -*-
"""Assemble kaggle_payload/ - the ONE folder to upload as a Kaggle private dataset.

LEAN payload: the notebook is fully self-contained, so it needs ONLY the dataset
TSVs + ground_truth.tsv + val_s1_ids.txt. DO NOT add source code, caches
or docs - extra files bloat the dataset for no benefit.

AUTH: must be the LEGACY credential (~/.kaggle/kaggle.json). KGAT_ tokens
(KAGGLE_API_TOKEN env) get 403 on CreateDatasetVersion - remove that env var."""
import os, shutil, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DST = os.path.join(ROOT, "kaggle_payload")


def copy(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)


def main():
    if os.path.isdir(DST):
        shutil.rmtree(DST)
    for split in ("train", "test"):
        for name in os.listdir(os.path.join(ROOT, "student_resource", "dataset", split)):
            copy(os.path.join(ROOT, "student_resource", "dataset", split, name),
                 os.path.join(DST, "dataset", split, name))
    copy(os.path.join(ROOT, "work", "ground_truth.tsv"), os.path.join(DST, "ground_truth.tsv"))
    copy(os.path.join(ROOT, "work", "val_s1_ids.txt"), os.path.join(DST, "val_s1_ids.txt"))
    # dataset-metadata.json — use KAGGLE_USERNAME when available; otherwise leave
    # an explicit placeholder so an accidental upload cannot target the wrong account.
    import json
    import pandas as pd
    username = os.environ.get("KAGGLE_USERNAME", "dptel22")
    pd.Series({"title": "BER Pipeline Data",
               "id": f"{username}/ber-pipeline",
               "licenses": [{"name": "CC0-1.0"}]}).to_json(
        os.path.join(DST, "dataset-metadata.json"))
    total = 0
    for dirpath, _, files in os.walk(DST):
        for f in files:
            total += os.path.getsize(os.path.join(dirpath, f))
    print(f"payload ready: {DST}  ({total/1e9:.2f} GB)")
    print("upload: kaggle datasets create -p kaggle_payload --dir-mode zip")


if __name__ == "__main__":
    main()
