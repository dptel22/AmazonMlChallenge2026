# Business Entity Resolution — Amazon ML Challenge 2026

Match deduplicated Source1 business entities to noisy Source2/Source3 records.
Hand-rolled multi-key blocking → pairwise string-similarity features → LightGBM →
F_0.5-tuned threshold. No external data lookups; transliteration is our own
deterministic Unicode character-mapping table (offline, stdlib only).

## Kaggle run (single notebook)  ← the real end-to-end execution happens HERE

- **`kaggle/kaggle_pipeline.ipynb`** — ONE fully self-contained notebook
  (stages A-H: prep, recall@K sweep, candidates, features, adversarial gate,
  LightGBM + F_0.5 sweep, test inference + inline validator, zip). Every
  pipeline function is INLINED into the cells — no subprocess calls, and no
  `src/` is needed on the dataset.
- **`kaggle/make_dataset_payload.py`** — assembles `kaggle_payload/` (dataset/
  TSVs + ground_truth.tsv + val_s1_ids.txt) for upload as one private Kaggle
  dataset.
- **`src/*.py` remain the readable reference implementation** the notebook is
  GENERATED from: `python kaggle/build_notebook.py > kaggle/kaggle_pipeline.ipynb`
  (AST extraction from src/). **Never hand-edit the `.ipynb`** — rebuilds must
  go through `build_notebook.py`.
- CONFIG cell knobs: `TOPK` / `PROXY` (set from the Stage B recall@K table,
  which measures mean and max proxies), `NEG_RATIO`, `SAMPLE_N` (>0 = small
  -slice end-to-end test). Stage G must print the validator **PASS**.
- Exact steps: [`kaggle/UPLOAD_INSTRUCTIONS.md`](../../kaggle/UPLOAD_INSTRUCTIONS.md).

## Pipeline (run from repo root, `student_resource/` must be a sibling of `code/`)

```bash
pip install -r code/business_entity_resolution/requirements.txt

# Phase 1+2a: normalize all source files -> work/*.parquet + slim id lookups
#   (~25 min, 16GB RAM)
python code/business_entity_resolution/src/prep.py

# Phase 2 gate: blocking recall + volume on the 18% val split (random_state=42)
#   saves work/val_s1_ids.txt — the EXACT split Kaggle must reuse
python code/business_entity_resolution/src/gate2.py

# Phase 2 gate b: recall@K after per-S1 top-K truncation (proxy pre-score)
#   fixes the volume knob K — recall must stay >=95% at the chosen K
python code/business_entity_resolution/src/gate2b.py

# Phase 2: candidate generation, train then test (~20-30 min each)
python code/business_entity_resolution/src/blocking.py --split train
python code/business_entity_resolution/src/blocking.py --split test

# Phase 3: pairwise features (train gets is_match labels, test does not)
python code/business_entity_resolution/src/features.py --split train --labels
python code/business_entity_resolution/src/features.py --split test

# Phase 3 gate: class-separation check incl. adversarial identical-name negatives
python code/business_entity_resolution/src/gate3.py
```

## Training + inference CLI (local reference path)  ← superseded as the execution path by the Kaggle notebook section above

The commands below document the readable `src/` reference implementation (the
Kaggle notebook inlines this same code). The old `make_kaggle_package.py`
zip-upload handoff is superseded by the single-notebook flow.

1. Build the upload bundle:

   ```bash
   python code/business_entity_resolution/src/make_kaggle_package.py
   # -> kaggle_upload.zip (feats_train/, val_s1_ids.txt, slim id lookups,
   #    ground_truth.tsv, train.py, evaluate.py)
   ```

2. Upload `kaggle_upload.zip` as a Kaggle Dataset. In a Kaggle notebook (GPU not
   required; LightGBM CPU is enough):

   ```python
   import os, zipfile
   zipfile.ZipFile("/kaggle/input/<dataset>/kaggle_upload.zip").extractall("/kaggle/working/work")
   os.environ["WORK_DIR"] = "/kaggle/working/work"
   !python /kaggle/working/work/train.py --full
   ```

3. Kaggle produces exactly two artifacts (download both):

   - `work/model.txt` — LightGBM booster (save_model format)
   - `work/model_config.json` — `{"threshold": float, "val_macro_f05": float,
     "per_bucket_f05": {"0": f, "1": f, "2-3": f, "4+": f}}`

4. Drop both back into local `work/`, then generate the submission:

   ```bash
   python code/business_entity_resolution/src/infer.py
   # writes output/matching_results.tsv + output/candidate_pairs.tsv
   python student_resource/utils/validate_submission.py \
       --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv \
       --test-dir student_resource/dataset/test   # must print PASS
   ```

`infer.py --dummy` runs the whole test-side path with a rule-based scorer and a
placeholder threshold — use it to verify writers + validator before the model exists.

## Design notes

- **Blocking keys** (country hard filter first — 0 cross-country true matches in the
  7.64M-pair ground truth, exhaustively verified; France is handled by the same
  code path, no country branching): `c3` name-prefix, `t` rare name tokens,
  `a` rare address tokens, `sn` street number, `fn` exact normalized name.
  df caps are applied at query time (one index serves all settings). Measured
  gate recall: 97.7038% at default caps, but un-truncated volume is infeasible
  (~31.6B pairs at train scale), so a rapidfuzz proxy pre-score keeps only each
  S1 entity's top-K candidates; gate2b measures recall@K (K ∈ {100, 200, 500})
  to pick K. Final numbers in Documentation_template.md.
- **Adversarial negatives**: ~1.3M pairs have byte-identical normalized name+country
  yet are NOT matches — name equality alone never decides; the classifier always
  sees address features, and the Phase-3 gate verifies they separate.
- **Model**: LightGBM binary classifier, 13 string-similarity features, ≤ small param
  count (we read the ≤8B constraint as a ceiling, not a target). MIT-licensed.
- **Threshold**: swept 0.10–0.95 on the saved val split against macro F_0.5
  (singletons included); per-bucket breakdown guards the singleton bucket.

## Files

```
code/business_entity_resolution/
├── src/normalize.py            Phase 1: normalization + transliteration (+ self-check gate)
├── src/prep.py                 Phase 2a: one-time normalization of all 6 source files
│                               (+ slim id lookups {split}_{s1,pool}_ids.parquet)
├── src/blocking.py             Phase 2: inverted-index candidate generation
├── src/gate2.py                Phase 2 gate: recall + volume on val split
├── src/gate2b.py               Phase 2 gate b: recall@K for top-K selection
├── src/features.py             Phase 3: pairwise feature table (parquet)
├── src/gate3.py                Phase 3 gate: separation + adversarial check
├── src/train.py                Phase 4: LightGBM training (RUNS ON KAGGLE)
├── src/evaluate.py             macro F_0.5 evaluator (standalone, importable)
├── src/infer.py                Phase 5: test inference + output writers
├── src/make_kaggle_package.py  Kaggle handoff bundler
├── requirements.txt            pinned versions
└── README.md                   this file
```
