import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def test_payload_builder_does_not_require_removed_stages_module():
    source = (ROOT / "kaggle" / "make_dataset_payload.py").read_text(encoding="utf-8")
    assert "stages.py" not in source


def test_notebook_builder_uses_boolean_header_flag_for_tsv_writers():
    source = (ROOT / "kaggle" / "build_notebook.py").read_text(encoding="utf-8")
    assert 'hdr = first' in source
    assert 'hdr = "source1_entity_id,matched_entity_ids"' not in source


def test_notebook_builder_prunes_test_candidates_and_measures_real_sizes():
    source = (ROOT / "kaggle" / "build_notebook.py").read_text(encoding="utf-8")
    assert "cands[j] = proxy_topk_indices(chunk.iloc[j], test_pool, c, TOPK, PROXY)" in source
    assert 'sizes[p][k] += len(top)' in source
    assert 'res_df["avg_cands_per_s1"] = res_df.topk' not in source
    assert 'size=max(1, len(train_only) // 10)' in source


def test_generated_notebook_is_current_and_has_expected_stage_count():
    import subprocess
    import sys
    rebuilt = ROOT / "kaggle" / "_test_rebuilt.ipynb"
    try:
        result = subprocess.run(
            [sys.executable, str(ROOT / "kaggle" / "build_notebook.py")],
            cwd=ROOT, capture_output=True, check=True,
        )
        generated = json.loads(result.stdout)
        checked_in = json.loads((ROOT / "kaggle" / "kaggle_pipeline.ipynb").read_text(encoding="utf-8"))
        assert generated == checked_in
        assert len(checked_in["cells"]) == 17
    finally:
        rebuilt.unlink(missing_ok=True)
