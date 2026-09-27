import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pipeline_state import is_complete, prepare_stage, write_manifest


def test_stage_manifest_skips_only_complete_matching_outputs(tmp_path):
    out = tmp_path / "stage"
    config = {"topk": 200, "proxy": "mean"}
    assert prepare_stage(out, "candidate", config) is True
    part = out / "part000.parquet"
    part.write_bytes(b"part")
    write_manifest(out, "candidate", config, [part], total_rows=17)

    assert is_complete(out, "candidate", config)
    assert prepare_stage(out, "candidate", config) is False
    assert prepare_stage(out, "candidate", {"topk": 500, "proxy": "mean"}) is True
    assert list(out.iterdir()) == []


def test_stage_manifest_rejects_missing_or_truncated_part(tmp_path):
    out = tmp_path / "stage"
    out.mkdir()
    config = {"seed": 42}
    part = out / "part000.parquet"
    part.write_bytes(b"complete")
    write_manifest(out, "features", config, [part], total_rows=8)
    assert is_complete(out, "features", config)

    part.write_bytes(b"")
    assert not is_complete(out, "features", config)
    assert not is_complete(out, "features", {"seed": 7})
