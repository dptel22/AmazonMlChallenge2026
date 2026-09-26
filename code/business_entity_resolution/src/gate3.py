# -*- coding: utf-8 -*-
"""Phase 3 GATE: feature class-separation check, focused on the adversarial subset.

Adversarial subset = candidate pairs whose normalized suffix-stripped names are
IDENTICAL but which are NOT true matches (the ~1.3M identical-name-negative pattern
from the data audit). If address features don't separate them from true pairs,
name features alone never will — flag loudly.
"""
import sys, os, glob
import numpy as np
import pandas as pd

WORK = os.environ.get("WORK_DIR", os.path.join(os.path.dirname(__file__), "..", "..", "..", "work"))


def main():
    files = sorted(glob.glob(os.path.join(WORK, "feats_train", "part*.parquet")))
    if not files:
        sys.exit("run features.py first")
    df = pd.concat([pd.read_parquet(f, columns=[
        "name_jaccard_normalized", "name_lev_suffix_stripped", "addr_jaccard", "addr_lev",
        "street_number_match", "is_match"]) for f in files], ignore_index=True)
    print(f"pairs: {len(df):,}  pos rate: {df.is_match.mean():.4%}")

    pos = df[df.is_match == 1]
    neg = df[df.is_match == 0]
    print("\n--- feature medians: true matches vs all negatives ---")
    for c in ["name_jaccard_normalized", "name_lev_suffix_stripped", "addr_jaccard",
              "addr_lev", "street_number_match"]:
        print(f"{c:32s} pos={pos[c].median():.4f}  neg={neg[c].median():.4f}")

    # adversarial subset: near-exact name (suffix-stripped lev >= 0.99) but not a match
    adv = df[(df.name_lev_suffix_stripped >= 0.99) & (df.is_match == 0)]
    adv_pos = df[(df.name_lev_suffix_stripped >= 0.99) & (df.is_match == 1)]
    print(f"\nadversarial identical-name pairs: {len(adv):,} negatives, {len(adv_pos):,} positives")
    if len(adv):
        print("--- address-feature separation on the adversarial subset ---")
        print(f"addr_jaccard: neg median={adv.addr_jaccard.median():.4f}  "
              f"p25={adv.addr_jaccard.quantile(.25):.4f}  pos median={adv_pos.addr_jaccard.median():.4f}")
        print(f"addr_lev:     neg median={adv.addr_lev.median():.4f}  "
              f"pos median={adv_pos.addr_lev.median():.4f}")
        print(f"street_number_match: neg mean={adv.street_number_match.mean():.4f}  "
              f"pos mean={adv_pos.street_number_match.mean():.4f}")
        sep = adv_pos.addr_jaccard.median() - adv.addr_jaccard.median()
        flag = "OK: address separates identical-name negatives" if sep > 0.1 else \
               "FLAG: address features weakly separate identical-name negatives"
        print(f"median addr_jaccard gap (pos - adversarial neg): {sep:.4f} -> {flag}")


if __name__ == "__main__":
    main()
