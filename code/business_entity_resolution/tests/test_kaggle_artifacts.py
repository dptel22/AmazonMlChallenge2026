import ast
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def test_payload_builder_does_not_require_removed_stages_module():
    source = (ROOT / "kaggle" / "make_dataset_payload.py").read_text(encoding="utf-8")
    assert "stages.py" not in source


def test_notebook_builder_uses_boolean_header_flag_for_tsv_writers():
    source = (ROOT / "kaggle" / "build_notebook.py").read_text(encoding="utf-8")
    assert 'hdr = first' in source
    assert 'hdr = "source1_entity_id,matched_entity_ids"' not in source


def test_notebook_installs_scikit_learn_distribution_for_sklearn_import():
    source = (ROOT / "kaggle" / "build_notebook.py").read_text(encoding="utf-8")
    assert '("sklearn", "scikit-learn")' in source


def test_notebook_builder_prunes_test_candidates_and_measures_real_sizes():
    source = (ROOT / "kaggle" / "build_notebook.py").read_text(encoding="utf-8")
    assert "batch_topk_prune(chunk, test_pool, cands, TOPK, PROXY, PAIR_BUDGET)" in source
    assert "proxy_topk_indices(chunk.iloc[j]" not in source
    assert "MAX_CANDIDATES" in source and "PAIR_BUDGET" in source
    assert 'sizes[p][k] += len(top)' in source
    assert 'res_df["avg_cands_per_s1"] = res_df.topk' not in source
    assert 'size=max(1, len(train_only) // 10)' in source


def test_generated_notebook_is_current_and_has_expected_stage_count():
    import contextlib
    import io
    import runpy
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        runpy.run_path(str(ROOT / "kaggle" / "build_notebook.py"), run_name="__main__")
    generated = json.loads(output.getvalue())
    checked_in = json.loads((ROOT / "kaggle" / "kaggle_pipeline.ipynb").read_text(encoding="utf-8-sig"))
    assert generated == checked_in
    assert len(checked_in["cells"]) == 17
    for i, c in enumerate(checked_in["cells"]):
        if c["cell_type"] == "code":
            compile("".join(c["source"]), f"generated-cell-{i}", "exec")
    norm_cell = next(c for c in checked_in["cells"]
                     if c["cell_type"] == "code" and "# S1a —" in "".join(c["source"]))
    cell_src = "".join(norm_cell["source"])
    # behavior: exercise the real normalize module (the notebook inlines it verbatim)
    sys_path_backup = list(sys.path)
    sys.path.insert(0, str(ROOT / "code" / "business_entity_resolution" / "src"))
    try:
        import normalize as normalize_mod
        assert "investments" in normalize_mod.norm_name("தமிழ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி")[0]
    finally:
        sys.path = sys_path_backup
    # the generated cell must carry the same norm_name source as src/normalize.py
    cell_fn = next(n for n in ast.parse(cell_src).body
                   if isinstance(n, ast.FunctionDef) and n.name == "norm_name")
    src_tree = ast.parse((ROOT / "code" / "business_entity_resolution" / "src" /
                          "normalize.py").read_text(encoding="utf-8"))
    src_fn = next(n for n in src_tree.body
                  if isinstance(n, ast.FunctionDef) and n.name == "norm_name")
    assert ast.dump(cell_fn) == ast.dump(src_fn)
