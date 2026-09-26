import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "code" / "business_entity_resolution" / "src"
WORK = ROOT / "work" / "synth"


def _normalized_row(entity_id, name, address, country, src):
    sys.path.insert(0, str(SRC))
    from normalize import addr_tokens, norm_name, norm_text, street_number

    name_n, name_s = norm_name(name)
    return {
        "entity_id": entity_id,
        "business_name": name,
        "business_address": address,
        "country": country,
        "name_n": name_n,
        "name_s": name_s,
        "name_t": " ".join(sorted(set(name_s.split()))),
        "name_nt": " ".join(sorted(set(name_n.split()))),
        "addr_n": norm_text(address),
        "addr_t": " ".join(addr_tokens(address)),
        "stno": street_number(address),
        "src": src,
    }


def _make_synthetic_inputs():
    if WORK.exists():
        import shutil
        shutil.rmtree(WORK)
    test_dir = WORK / "synth_test"
    test_dir.mkdir(parents=True)
    s1_rows = []
    for i in range(60):
        country = "France" if i >= 58 else ("India" if i % 2 else "US")
        s1_rows.append(_normalized_row(
            f"S1-{i:04d}", f"Acme Branch {i}", f"{1000 + i} Main Street, City", country, "s1"
        ))
    pools = {"source2": [], "source3": []}
    for j in range(300):
        src = "source2" if j < 150 else "source3"
        i = j % 60
        country = "France" if i >= 58 else ("India" if i % 2 else "US")
        name = f"Acme Branch {i}" if j % 5 else f"Acme Branch {i} Holdings"
        address = "" if j % 17 == 0 else f"{1000 + i} Main Street, City"
        pools[src].append(_normalized_row(f"S{2 if src == 'source2' else 3}-{j:04d}", name, address, country, "s2" if src == "source2" else "s3"))

    s1 = pd.DataFrame(s1_rows)
    s2 = pd.DataFrame(pools["source2"])
    s3 = pd.DataFrame(pools["source3"])
    s1.to_parquet(WORK / "test_s1.parquet", index=False)
    s2.to_parquet(WORK / "test_s2.parquet", index=False)
    s3.to_parquet(WORK / "test_s3.parquet", index=False)
    for src, frame in (("source1", s1), ("source2", s2), ("source3", s3)):
        frame[["entity_id", "business_name", "business_address", "country"]].to_csv(
            test_dir / f"test_{src}.tsv", sep="\t", index=False
        )


@pytest.mark.integration
def test_dummy_inference_writers_and_validator():
    pytest.importorskip("pandas")
    _make_synthetic_inputs()
    env = os.environ.copy()
    env["WORK_DIR"] = str(WORK)
    # list-form argv (no shell), all args are repo-internal constants — no injection surface
    subprocess.run([sys.executable, str(SRC / "infer.py"), "--dummy", "--threshold", "0.99"],
                   cwd=ROOT, env=env, check=True, shell=False)
    matching = ROOT / "output" / "matching_results.tsv"
    candidates = ROOT / "output" / "candidate_pairs.tsv"
    validator = ROOT / "student_resource" / "utils" / "validate_submission.py"
    result = subprocess.run([
        sys.executable, str(validator), "--matching", str(matching),
        "--candidate", str(candidates), "--test-dir", str(WORK / "synth_test"), "--check-ids",
    ], cwd=ROOT, text=True, capture_output=True, check=True, shell=False)
    assert "PASS" in result.stdout
    m = pd.read_csv(matching, sep="\t", keep_default_na=False)
    c = pd.read_csv(candidates, sep="\t", keep_default_na=False)
    assert len(m) == len(c) == 60
    assert any(not value for value in m.matched_entity_ids)
    candidate_map = dict(zip(c.source1_entity_id, c.candidate_entity_ids))
    for row in m.itertuples(index=False):
        matched = set(filter(None, row.matched_entity_ids.split(",")))
        candidate_ids = set(filter(None, candidate_map[row.source1_entity_id].split(",")))
        assert matched <= candidate_ids
        assert len(matched) == len(row.matched_entity_ids.split(",")) if row.matched_entity_ids else True


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
