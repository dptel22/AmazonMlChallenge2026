"""Small explicit runner for the synthetic Phase-5 writer check."""
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
raise SystemExit(subprocess.call([sys.executable, "-m", "pytest", str(ROOT / "code/business_entity_resolution/tests/test_writers_e2e.py"), "-q"], cwd=ROOT))
