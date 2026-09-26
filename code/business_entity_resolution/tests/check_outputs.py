"""Low-memory second-opinion checker for submission TSVs.

The candidate pass stores only required IDs plus per-entity count/sum/xor of a
64-bit stable hash. The subset comparison is therefore probabilistic when the
counts do not prove a violation; it is deliberately not a replacement for the
official validator. Duplicate IDs, wrong prefixes, extras, missing rows, and
match-count > candidate-count are definite violations.
"""
import argparse
import csv
import hashlib
from collections import Counter
from pathlib import Path


def _hash(value):
    return int.from_bytes(hashlib.blake2b(value.encode(), digest_size=8).digest(), "big")


def _ids(path, column):
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            yield row.get(column, "").strip(), row


def _validate_list(raw, expected_prefix):
    ids = [item for item in raw.split(",") if item]
    duplicate = len(ids) != len(set(ids))
    bad = [item for item in ids if not item.startswith(expected_prefix)]
    return ids, duplicate, bad


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--matching", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    required = set()
    with (args.test_dir / "test_source1.tsv").open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {row["entity_id"].strip() for row in reader}
    violations = Counter()
    candidates = {}
    candidate_rows = 0
    candidate_empty = 0
    seen_candidate_entities = set()
    for entity, row in _ids(args.candidate, "source1_entity_id"):
        candidate_rows += 1
        if entity in seen_candidate_entities:
            violations["duplicate candidate row"] += 1
        seen_candidate_entities.add(entity)
        if entity not in required:
            violations["candidate extra S1"] += 1
        values, duplicate, bad = _validate_list(row.get("candidate_entity_ids", ""), "S2-")
        # Source3 IDs are valid too; only reject S1/unknown-style prefixes.
        bad = [value for value in values if not (value.startswith("S2-") or value.startswith("S3-"))]
        if duplicate:
            violations["duplicate candidate id"] += 1
        if bad:
            violations["invalid candidate id"] += len(bad)
        if not values:
            candidate_empty += 1
        h = [_hash(value) for value in values]
        candidates[entity] = (len(values), sum(h) & ((1 << 64) - 1), _xor(h))
    if seen_candidate_entities != required:
        violations["missing candidate row"] += len(required - seen_candidate_entities)

    matching_rows = 0
    matching_empty = 0
    seen_matching_entities = set()
    for entity, row in _ids(args.matching, "source1_entity_id"):
        matching_rows += 1
        if entity in seen_matching_entities:
            violations["duplicate matching row"] += 1
        seen_matching_entities.add(entity)
        if entity not in required:
            violations["matching extra S1"] += 1
        values, duplicate, bad = _validate_list(row.get("matched_entity_ids", ""), "S2-")
        bad = [value for value in values if not (value.startswith("S2-") or value.startswith("S3-"))]
        if duplicate:
            violations["duplicate matching id"] += 1
        if bad:
            violations["invalid matching id"] += len(bad)
        if not values:
            matching_empty += 1
        h = [_hash(value) for value in values]
        c_count, c_sum, c_xor = candidates.get(entity, (0, 0, 0))
        if len(values) > c_count:
            violations["definite subset violation"] += 1
        # Aggregate comparison is meaningful only when both lists have the
        # same cardinality. For a proper subset with fewer items, the compact
        # candidate aggregate intentionally cannot prove membership; this is
        # the documented probabilistic limitation of this low-memory checker.
        elif len(values) == c_count and (sum(h) & ((1 << 64) - 1), _xor(h)) != (c_sum, c_xor):
            violations["probable subset violation"] += 1
    if seen_matching_entities != required:
        violations["missing matching row"] += len(required - seen_matching_entities)

    print(f"candidate rows={candidate_rows} empty={candidate_empty}")
    print(f"matching rows={matching_rows} empty={matching_empty}")
    print("violations=" + (", ".join(f"{k}:{v}" for k, v in violations.items()) or "none"))
    return 1 if violations else 0


def _xor(values):
    result = 0
    for value in values:
        result ^= value
    return result


if __name__ == "__main__":
    raise SystemExit(main())
