"""Cheap France-country spot-check; run only when the compute lock is free."""
import argparse
import random
from collections import Counter
from pathlib import Path

import pandas as pd


SIGNALS = {"usa", "united", "states", "india", "delhi", "mumbai"}


def scan(work_dir, output):
    work_dir, output = Path(work_dir), Path(output)
    report = []
    report.append("France sanity scan — test parquet spot-check")
    france_samples = []
    signal_counts = Counter()
    for split in ("s1", "s2", "s3"):
        path = work_dir / f"test_{split}.parquet"
        columns = ["entity_id", "country", "business_name", "name_s", "addr_n"]
        frame = pd.read_parquet(path, columns=columns)
        france = frame[frame.country == "France"]
        report.append(f"{path.name}: France={len(france):,} / total={len(frame):,}")
        if split == "s1":
            counts = frame.country.value_counts().to_dict()
            report.append("S1 country shares: " + ", ".join(
                f"{country}={counts.get(country, 0):,} ({counts.get(country, 0) / len(frame):.4%})"
                for country in sorted(counts)))
            france_samples = france.sample(n=min(10, len(france)), random_state=20260925).to_dict("records")
        names_non_ascii = sum(not str(value).isascii() for value in france.business_name)
        normalized_names_non_ascii = sum(not str(value).isascii() for value in france.name_s)
        addrs_non_ascii = sum(not str(value).isascii() for value in france.addr_n)
        report.append(
            f"France non-ASCII: raw_names={names_non_ascii / max(len(france), 1):.4%}, "
            f"normalized_names={normalized_names_non_ascii / max(len(france), 1):.4%}, "
            f"addresses={addrs_non_ascii / max(len(france), 1):.4%}"
        )
        for value in pd.concat([france.name_s, france.addr_n]).fillna(""):
            signal_counts.update(token for token in str(value).lower().split() if token in SIGNALS)
    report.append("Weak multi-country signal tokens: " + (repr(signal_counts.most_common(20)) if signal_counts else "none"))
    report.append("Random France S1 rows:")
    for row in france_samples:
        report.append(f"  {row['entity_id']}: {row['business_name']!r} | {row['addr_n']!r}")
    output.write_text("\n".join(report) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", default="work")
    parser.add_argument("--output", default="work/france_sanity.txt")
    args = parser.parse_args()
    print("\n".join(scan(args.work_dir, args.output)))
