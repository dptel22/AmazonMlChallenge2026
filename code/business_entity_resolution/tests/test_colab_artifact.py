import contextlib
import io
import json
import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def test_colab_notebook_is_builder_output_and_has_guarded_real_run():
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        runpy.run_path(str(ROOT / "kaggle" / "build_colab_notebook.py"),
                       run_name="__main__")
    expected = json.loads(output.getvalue())
    notebook_path = ROOT / "kaggle" / "colab_pipeline.ipynb"
    actual = json.loads(notebook_path.read_text(encoding="utf-8"))
    assert actual == expected

    sources = ["".join(c["source"]) for c in actual["cells"] if c["cell_type"] == "code"]
    joined = "\n".join(sources)
    assert "drive.mount" in joined
    assert "recall@{TOPK}" in joined
    assert "assert configured_recall >= RECALL_FLOOR" in joined
    assert "AUTO_SELECT_TOPK" in joined
    assert "no tested K meets the recall floor" in joined
    assert "estimated train candidate pairs" in joined
    assert "MEMORYERROR" in joined
    assert "Stage E PASS" in joined
    assert "train.py" in joined and "--full" in joined
    assert "OUTPUT_DIR" in joined
    for i, source in enumerate(sources):
        compile(source, f"colab-cell-{i}", "exec")


def test_standalone_recall_gate_includes_larger_k_values_for_cosine_proxy():
    source = (ROOT / "code" / "business_entity_resolution" / "src" /
              "gate2b.py").read_text(encoding="utf-8")
    assert "ks = [10, 20, 30, 50, 100, 200, 300, 500, 750, 1000]" in source
    assert "estimated train candidate pairs" in source
    assert 'proxy_names = ["mean"]' in source
    assert "allowed_by_setting" not in source
    assert "limits = {\"c3\": 2000, \"t\": tok" in source
