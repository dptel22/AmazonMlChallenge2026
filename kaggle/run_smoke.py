"""Deterministic headless gate: execute the notebook on the tiny slice and
require every gate line with zero cell errors. Exit 0 only when all pass."""
import json, os, shutil, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "work_tiny", "dataset")
NB = os.path.join(ROOT, "kaggle", "kaggle_pipeline.ipynb")
SMOKE = os.path.join(ROOT, "work_nb", "_smoke.ipynb")
WORK = os.path.join(ROOT, "work_tiny", "work_wfgate")
GATES = ["S1 self-checks PASS", "Stage A DONE", "Stage B DONE", "Stage C DONE",
         "Stage D DONE", "F_0.5 assertions passed", "Stage F DONE",
         "Stage G inference DONE", "no blocking issues found",
         "kaggle_return.zip"]


def main():
    py = sys.executable
    os.makedirs(os.path.dirname(SMOKE), exist_ok=True)
    shutil.copy2(NB, SMOKE)
    shutil.rmtree(WORK, ignore_errors=True)
    env = dict(os.environ, BER_DATA=DATA, BER_WORK=WORK,
               PYTHONIOENCODING="utf-8")
    r = subprocess.run([py, "-m", "jupyter", "nbconvert", "--to", "notebook",
                        "--execute", "--inplace", SMOKE], env=env, cwd=ROOT,
                       capture_output=True, text=True)
    if r.returncode != 0:
        print("NBCONVERT FAILED")
        print((r.stdout or "")[-4000:])
        print((r.stderr or "")[-4000:])
        return 1
    with open(SMOKE, encoding="utf-8") as f:
        nb = json.load(f)
    text, errors = [], 0
    for c in nb["cells"]:
        for o in c.get("outputs", []):
            if o.get("output_type") == "error":
                errors += 1
                text.append("CELL ERROR: %s: %s" % (o.get("ename"), o.get("evalue")))
            if o.get("output_type") == "stream":
                text.append("".join(o.get("text", [])))
    blob = "\n".join(text)
    print(blob)
    missing = [g for g in GATES if g not in blob]
    if errors or missing:
        print("SMOKE FAIL: %d cell errors; missing gates: %s" % (errors, missing))
        return 1
    print("SMOKE PASS: all gates green, zero cell errors")
    return 0


if __name__ == "__main__":
    sys.exit(main())
