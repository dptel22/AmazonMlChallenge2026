# -*- coding: utf-8 -*-
"""Build a Google Colab runner for the staged local BER pipeline."""
import json


def cell(kind, source):
    result = {"cell_type": kind, "metadata": {},
              "source": source.splitlines(keepends=True)}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


cells = [
    cell("markdown", """# Business Entity Resolution — Google Colab runner

This notebook runs the repository pipeline on the challenge's supplied data.
Choose a **High-RAM CPU runtime**. Place the repository and `student_resource/dataset/`
under the Drive folder configured below. Checkpoints are copied to Drive after
each completed stage so a disconnected runtime can resume. The notebook selects
the smallest tested K meeting the recall floor, and stops before model fitting
if no tested K passes or the full-data feature separation gate fails.

The 20k-entity recall gate is a validation subsample, not a full-scale score.
The real LightGBM fit reports its own held-out macro F_0.5; no score is claimed
until that cell finishes. Package installation uses Colab's existing compatible
scientific stack and installs only missing packages.
"""),
    cell("code", """from google.colab import drive
drive.mount('/content/drive')

from pathlib import Path

# Change this to the Drive folder containing the repository.
PROJECT_ROOT = Path('/content/drive/MyDrive/AmazonMlChallenge2026')
DATA_DIR = PROJECT_ROOT / 'student_resource' / 'dataset'
SRC_DIR = PROJECT_ROOT / 'code' / 'business_entity_resolution' / 'src'
REQ_FILE = PROJECT_ROOT / 'code' / 'business_entity_resolution' / 'requirements.txt'
DRIVE_STATE = PROJECT_ROOT / 'colab_state'
DRIVE_OUTPUT = PROJECT_ROOT / 'colab_output'

required = [
    DATA_DIR / 'train' / 'train_source1.tsv',
    DATA_DIR / 'train' / 'train_source2.tsv',
    DATA_DIR / 'train' / 'train_source3.tsv',
    DATA_DIR / 'train' / 'train_ground_truth.tsv',
    DATA_DIR / 'test' / 'test_source1.tsv',
    DATA_DIR / 'test' / 'test_source2.tsv',
    DATA_DIR / 'test' / 'test_source3.tsv',
]
missing = [str(p) for p in required if not p.is_file()]
assert not missing, 'Missing challenge/repository files:\\n' + '\\n'.join(missing)
assert SRC_DIR.is_dir(), f'Missing source directory: {SRC_DIR}'
print('Project and train/test TSVs found:', PROJECT_ROOT)
"""),
    cell("code", """# Install only missing libraries; avoid replacing Colab's preinstalled NumPy/Pandas.
import importlib.util
import subprocess
import sys

packages = [('numpy', 'numpy'), ('pandas', 'pandas'), ('pyarrow', 'pyarrow'),
            ('lightgbm', 'lightgbm'), ('sklearn', 'scikit-learn')]
missing_packages = [package for module, package in packages
                    if importlib.util.find_spec(module) is None]
if missing_packages:
    subprocess.run([sys.executable, '-m', 'pip', 'install', *missing_packages], check=True)
else:
    print('Required scientific packages are already installed.')
"""),
    cell("code", """# Configure paths, restore checkpoints, and define stage helpers.
import os
import shutil
import subprocess
import sys
from pathlib import Path

LOCAL_ROOT = Path('/content/ber_pipeline')
WORK_DIR = LOCAL_ROOT / 'work'
OUTPUT_DIR = LOCAL_ROOT / 'output'
WORK_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ENV = dict(os.environ)
ENV.update(DATA_DIR=str(DATA_DIR), WORK_DIR=str(WORK_DIR), OUTPUT_DIR=str(OUTPUT_DIR))

def sync_tree(source, destination):
    # Incrementally copy completed artifacts; write each destination atomically.
    source, destination = Path(source), Path(destination)
    if not source.exists():
        return
    for item in source.rglob('*'):
        rel = item.relative_to(source)
        target = destination / rel
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_file() and target.stat().st_size == item.stat().st_size \\
                and target.stat().st_mtime_ns == item.stat().st_mtime_ns:
            continue
        temp = target.with_name(target.name + '.partial')
        shutil.copy2(item, temp)
        os.replace(temp, target)

def checkpoint():
    sync_tree(WORK_DIR, DRIVE_STATE / 'work')
    sync_tree(OUTPUT_DIR, DRIVE_STATE / 'output')
    print('Checkpoint saved to Drive:', DRIVE_STATE)

if DRIVE_STATE.is_dir():
    sync_tree(DRIVE_STATE / 'work', WORK_DIR)
    sync_tree(DRIVE_STATE / 'output', OUTPUT_DIR)
    print('Restored prior checkpoint from Drive.')

def run_script(script, *args):
    command = [sys.executable, str(SRC_DIR / script), *map(str, args)]
    print('\\n$', ' '.join(command), flush=True)
    process = subprocess.Popen(command, cwd=str(PROJECT_ROOT), env=ENV,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, bufsize=1)
    output = []
    for line in process.stdout:
        print(line, end='', flush=True)
        output.append(line)
    code = process.wait()
    if code:
        raise RuntimeError(f'{script} exited with code {code}')
    return ''.join(output)

def available_ram_gb():
    values = {}
    with open('/proc/meminfo', encoding='utf-8') as handle:
        for line in handle:
            if line.startswith(('MemAvailable:', 'MemTotal:')):
                key, value, _ = line.split()
                values[key.rstrip(':')] = int(value) / 1024 / 1024
    return values

ram = available_ram_gb()
print(f"Colab RAM: {ram['MemTotal']:.1f} GB total, {ram['MemAvailable']:.1f} GB available")
print(f"Colab local disk free: {shutil.disk_usage(LOCAL_ROOT).free / 1e9:.1f} GB")
"""),
    cell("code", """# CONFIG — keep caps aligned with the measured recall gate.
TOPK = 50
PROXY = 'mean'
RECALL_FLOOR = 0.95
AUTO_SELECT_TOPK = True
NEG_RATIO = 0.05
TOKEN_CAP, ADDR_CAP, SN_CAP, FN_CAP, C3_CAP = 1000, 2000, 2000, 20000, 2000
MAX_CANDIDATES = 2000

print(f'CONFIG: TOPK={TOPK} PROXY={PROXY} recall floor={RECALL_FLOOR:.0%} '
      f'NEG_RATIO={NEG_RATIO}')
"""),
    cell("code", """# Stage A — normalization. Reuse existing complete parquet artifacts on resume.
prepared = [WORK_DIR / f'{split}_{name}.parquet'
            for split in ('train', 'test') for name in ('s1', 's2', 's3')]
prepared += [WORK_DIR / f'{split}_{name}_ids.parquet'
             for split in ('train', 'test') for name in ('s1', 'pool')]
prepared += [WORK_DIR / 'ground_truth.tsv']
if not all(path.is_file() for path in prepared):
    run_script('prep.py')
    checkpoint()
else:
    print('Stage A outputs found; reusing prepared data.')
"""),
    cell("code", """# Stage B — measure cosine recall on the saved 20k validation subsample;
# if needed, select the smallest tested K that reaches the recall floor.
# The inverted index is large; use a High-RAM runtime and ensure at least 8 GB
# is available before building it.
import re

assert available_ram_gb()['MemAvailable'] >= 8.0, \\
    'Less than 8 GB RAM is available. Switch Colab to High-RAM, then rerun this cell.'
gate_log = run_script('gate2b.py')
checkpoint()  # includes the validation IDs and complete gate log artifacts, if any
assert 'MEMORYERROR' not in gate_log, 'Recall gate was incomplete due to memory pressure.'

in_default_caps = False
recalls = {}
projected_pairs = {}
for line in gate_log.splitlines():
    if line.startswith('caps='):
        in_default_caps = line.startswith('caps=(1000,2000,2000,20000)')
    if in_default_caps and line.strip().startswith(f'proxy={PROXY}:'):
        recalls.update({int(k): float(v) / 100.0
                        for k, v in re.findall(r'recall@(\\d+)=([0-9.]+)%', line)})
    if in_default_caps:
        volume = re.search(r'K=(\\d+): [0-9.]+ candidates/S1; estimated train candidate pairs=([0-9,]+)',
                           line)
        if volume:
            projected_pairs[int(volume.group(1))] = int(volume.group(2).replace(',', ''))
eligible_k = sorted(k for k, recall in recalls.items() if recall >= RECALL_FLOOR)
if AUTO_SELECT_TOPK:
    assert eligible_k, 'no tested K meets the recall floor for the configured proxy under default caps'
    TOPK = eligible_k[0]
configured_recall = recalls.get(TOPK)
assert configured_recall is not None, \\
    f'Could not find recall@{TOPK} for proxy={PROXY} under default caps in gate output.'
print(f'Configured recall@{TOPK} for {PROXY}: {configured_recall:.4%}')
if TOPK in projected_pairs:
    print(f'estimated train candidate pairs: {projected_pairs[TOPK]:,}')
assert configured_recall >= RECALL_FLOOR, (\\
    f'Recall gate failed: {configured_recall:.4%} < {RECALL_FLOOR:.0%}. '
    'Stop here; inspect the gate table and change TOPK/PROXY before proceeding.')
"""),
    cell("code", """# Stage C — create labeled-training candidate pairs with the measured config.
run_script('blocking.py', '--split', 'train', '--token-cap', TOKEN_CAP,
           '--addr-cap', ADDR_CAP, '--sn-cap', SN_CAP, '--fn-cap', FN_CAP,
           '--c3-cap', C3_CAP, '--max-candidates', MAX_CANDIDATES,
           '--topk', TOPK, '--proxy', PROXY)
checkpoint()
"""),
    cell("code", """# Stage D — pairwise features. Keep all positives and all validation pairs;
# sample only train-fold negatives at NEG_RATIO.
run_script('features.py', '--split', 'train', '--labels', '--neg-ratio', NEG_RATIO)
checkpoint()
"""),
    cell("code", """# Stage E — require the full-data adversarial feature gate to pass.
gate_log = run_script('gate3.py')
assert '-> OK:' in gate_log, \\
    'Stage E did not report PASS. Stop before fitting; inspect the separation output.'
print('Stage E PASS')
"""),
    cell("code", """# Stage F — the real LightGBM fit and entity-level macro-F0.5 threshold sweep.
# This is the first cell that trains the model. Its validation score is the
# quality result; the tiny smoke-test score is not used.
if not (WORK_DIR / 'model.txt').is_file() or not (WORK_DIR / 'model_config.json').is_file():
    run_script('train.py', '--full')
    checkpoint()
else:
    print('Real model artifacts already exist in the restored checkpoint; reusing them.')

import json
import math
with (WORK_DIR / 'model_config.json').open(encoding='utf-8') as handle:
    model_config = json.load(handle)
validation_f05 = float(model_config['val_macro_f05'])
assert math.isfinite(validation_f05) and validation_f05 > 0.0, \\
    f'Model validation F0.5 is not usable: {validation_f05}'
print(f"VALIDATION macro F0.5={validation_f05:.6f}; "
      f"threshold={model_config['threshold']}; topk={model_config.get('topk')}; "
      f"proxy={model_config.get('proxy')}")
"""),
    cell("code", """# Stage G — infer on the complete test set and write both required TSVs.
run_script('infer.py')
checkpoint()
"""),
    cell("code", """# Validate the exact output schemas and row coverage with the streaming checker.
# infer.py also enforces matches-subset-of-candidates per source-1 chunk and draws
# every output candidate ID directly from the test S2/S3 pool.
check_script = (PROJECT_ROOT / 'code' / 'business_entity_resolution' / 'tests' /
                'check_outputs.py')
run_script(str(check_script), '--matching', OUTPUT_DIR / 'matching_results.tsv',
           '--candidate', OUTPUT_DIR / 'candidate_pairs.tsv',
           '--test-dir', DATA_DIR / 'test')
sync_tree(OUTPUT_DIR, DRIVE_OUTPUT)
print('Output files saved to Drive:', DRIVE_OUTPUT)
"""),
    cell("markdown", """## After the run

The model artifacts are in `colab_state/work/` on Drive. Submission TSVs are in
`colab_output/`. The methodology document still needs the measured recall@K and
validation metrics filled in before making the final reproducibility archive.
Download `matching_results.tsv` for leaderboard scoring. Do not report the tiny
smoke run as a score.
"""),
]

notebook = {
    "cells": cells,
    "metadata": {
        "colab": {"name": "ber_colab_pipeline.ipynb", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
print(json.dumps(notebook, ensure_ascii=True, indent=1))
