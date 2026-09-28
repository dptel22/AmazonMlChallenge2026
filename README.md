# Business Entity Resolution — Amazon ML Challenge 2026

Match noisy business records across three independent data sources. Source 1 is
the deduplicated reference; for every Source 1 entity, find all matching
records in Sources 2 and 3. Records share no common identifiers; names and
addresses carry abbreviations, legal-suffix drift, transliterations across nine
Indic scripts, word-order transpositions, and partial addresses. Training data
covers US and India; the test set adds France (unseen country — the pipeline
treats country as an open label set). Scored with macro-averaged F_0.5
(precision-weighted; correctly empty predictions for singleton entities score
1.0). Candidate generation is part of the final evaluation: smaller candidate
sets per Source 1 entity rank higher.

## Architecture

Six stages, each skip-if-done and independently gated (stages live in
`code/business_entity_resolution/src/`, inlined into one self-contained Kaggle
notebook by `kaggle/build_notebook.py`):

1. **Prep** — TSV → parquet; normalization: NFKD diacritic folding, a
   hand-written transliteration table for all nine Indic scripts, legal-suffix
   stripping, address tokenization, street-number extraction.
2. **Blocking** — multi-key inverted index (115M postings at full scale) over
   name/address/street-number/country keys with per-key caps.
3. **Candidate pruning** — per-S1 top-K by a cheap ranking proxy, with the
   recall@K table measured on a stratified 10k-entity validation split before
   any volume commitment.
4. **Features** — 11 pairwise features: three Jaccards (raw / normalized /
   suffix-stripped), TF-IDF cosine on name and address, token overlap, street
   number match, missing-address flag, length deltas.
5. **Model** — LightGBM with early stopping; F_0.5-optimal threshold and
   per-entity cutoff chosen by sweep on a held-out split.
6. **Inference** — matches ⊆ candidates enforced at write time, inline
   submission validator, packaged output.

## Engineering under a hard wall

Kaggle gives one 12-hour CPU session. Two findings drove the design:

- **Pairwise Levenshtein does not fit.** Measured 17.55 s per 100k-pair
  feature chunk → ~25 h for a full run. Replaced with word-unigram TF-IDF
  cosine (sparse row dot products) at ~1000× lower per-pair cost.
- **The first cosine build was still slow (47 s/part) and OOM-killed the
  kernel.** Root cause: per-chunk re-splitting of entity strings and 100k-row
  random takes out of 12.5M-row arrow-backed frames. Fix: per-entity token
  sets and scalars precomputed once per split (`ensure_entity_feats`,
  owner-guarded against pandas-3.0 attrs propagation through `iloc`).
  Verified by cProfile: steady chunk **0.7 s** (0.03 GB peak), ~15× faster
  and 13× leaner.

## Results, honestly

Full-scale blocking measurements (10,000-entity stratified val split, 34,554
true pairs): recall@10 = 81.7%, @20 = 86.4%, @50 = 88.5%, @100 = 92.0%,
@200 = 92.6% at 48.3 candidates/entity at K=50 — a reduction ratio of ~4.8e-6
against the naive 1.73e13-pair space. The recall curve saturates with K: the
ceiling is the key set, not the cutoff. Reaching the leaderboard's top-100
cutoff (0.9885) requires ~99% candidate recall, i.e. a different blocking
architecture (unioned complementary key families or embedding-ANN retrieval).

The final Kaggle run crashed — kernel OOM at the start of Stage D — and no
submission was made before the deadline. The crash was root-caused and fixed
as described above (verified locally and by the full test suite), but the fix
landed after the submission window closed.

## Repository layout

```
code/business_entity_resolution/
  src/          pipeline source (normalize, prep, blocking, features,
                train, infer, evaluate, pipeline_state)
  tests/        43 tests incl. an executed-tiny-slice artifact contract
  RELEASE_CHECKLIST.md
kaggle/
  build_notebook.py       generates the self-contained notebook from src/
  kaggle_pipeline.ipynb   the generated notebook (17 cells, stages A–H)
  run_smoke.py            deterministic headless end-to-end gate
  make_submission_zip.py  one-command packaging + validator + structure asserts
  UPLOAD_INSTRUCTIONS.md
student_resource/          challenge-provided materials (dataset not tracked);
  Documentation_template.md  filled methodology write-up
  utils/validate_submission.py
work/                      local run artifacts (not tracked)
```

## Reproduce

```bash
# unit + artifact suite (43 tests)
.venv/Scripts/python.exe -m pytest code/business_entity_resolution/tests -q

# headless end-to-end on a tiny real-data slice (all stage gates asserted)
.venv/Scripts/python.exe kaggle/run_smoke.py

# full run: import kaggle/kaggle_pipeline.ipynb on Kaggle (CPU, internet off),
# attach the dataset, run all cells; CONFIG holds SAMPLE_N/TOPK/PROXY/NEG_RATIO

# package a finished run
.venv/Scripts/python.exe kaggle/make_submission_zip.py real --zip kaggle_return.zip --team <name>
```

## Security notes

All env-derived paths pass through an explicit traversal guard
(`pipeline_state.ensure_local_path`); the artifact contract test validates
generated code via AST inspection and module import rather than `exec`.
Static-analysis advisories from the Mimosa gate are tracked in-repo history.
