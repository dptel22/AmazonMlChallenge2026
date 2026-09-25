# -*- coding: utf-8 -*-
"""Kaggle handoff packager: zips everything train.py needs into kaggle_upload.zip.

Contents (flat):
  feats_train/part*.parquet   feature table with is_match labels
  val_s1_ids.txt              exact Phase-2 validation split (reuse, don't regenerate)
  train_s1_ids.parquet        entity_id lookup for s1_idx (slim: entity_id only)
  train_pool_ids.parquet      entity_id lookup for cand_idx (slim, s2+s3 in order)
  ground_truth.tsv            train ground truth (for truth_by + threshold sweep)

Kaggle returns: model.txt + model_config.json (see README "Training (Kaggle)").
"""
import os, sys, zipfile
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")
WORK = os.path.join(ROOT, "work")


def main():
    out = os.path.join(ROOT, "kaggle_upload.zip")
    # slim id lookups (entity_id column only, row order = the *_idx space)
    pd.read_parquet(os.path.join(WORK, "train_s1.parquet"), columns=["entity_id"]) \
        .to_parquet(os.path.join(WORK, "train_s1_ids.parquet"), index=False)
    pd.concat([pd.read_parquet(os.path.join(WORK, n), columns=["entity_id"])
               for n in ("train_s2.parquet", "train_s3.parquet")], ignore_index=True) \
        .to_parquet(os.path.join(WORK, "train_pool_ids.parquet"), index=False)

    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(os.listdir(os.path.join(WORK, "feats_train"))):
            z.write(os.path.join(WORK, "feats_train", f), f"feats_train/{f}")
        z.write(os.path.join(WORK, "val_s1_ids.txt"), "val_s1_ids.txt")
        z.write(os.path.join(WORK, "train_s1_ids.parquet"), "train_s1_ids.parquet")
        z.write(os.path.join(WORK, "train_pool_ids.parquet"), "train_pool_ids.parquet")
        z.write(os.path.join(WORK, "ground_truth.tsv"), "ground_truth.tsv")
        z.write(os.path.join(os.path.dirname(__file__), "train.py"), "train.py")
        z.write(os.path.join(os.path.dirname(__file__), "evaluate.py"), "evaluate.py")
    print(f"wrote {out} ({os.path.getsize(out)/1e6:.0f} MB)")


if __name__ == "__main__":
    main()
