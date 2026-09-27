# Business Entity Resolution

Professional reference implementation for the Amazon ML Challenge 2026.
The pipeline resolves every Source 1 business record to zero or more matching
records in Source 2 and Source 3 using only the supplied data.

## Final submission structure

```text
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       ├── README.md
│       └── requirements.txt
└── Documentation_template.md
```

`matching_results.tsv` contains final predictions. `candidate_pairs.tsv` is the
last candidate set actually passed to the matching model; every predicted match
must be present in it.

## Method

1. **Normalization** — Unicode normalization, accent folding, deterministic
   transliteration for supported Indic scripts, legal-suffix variants, address
   tokenization, and street-number extraction.
2. **Blocking** — country-scoped inverted-index keys over name prefixes, name
   tokens, address tokens, street numbers, and full normalized names.
3. **Candidate control** — sparse TF-IDF cosine proxy ranking keeps the top `TOPK`
   candidates per Source 1 entity. Stage B measures recall and real candidate
   volume before selecting the operating point.
4. **Matching** — 11 name/address similarity features feed a LightGBM binary
   classifier. The threshold is selected using macro F_0.5 on an entity-level
   validation split.
5. **Validation and packaging** — schemas, row coverage, ID prefixes, duplicate
   IDs, and the matches-subset-of-candidates invariant are checked before release.

The system is fully offline: no web lookup, geocoding, external entity search,
or data augmentation is used.

## Submission constraints

- Both output files are tab-separated with the exact required column names.
- Every test Source 1 entity appears exactly once, including singletons.
- `matched_entity_ids` contains only existing Source 2/Source 3 IDs; no S1
  self-matches, unknown IDs, duplicate rows, or duplicate IDs are permitted.
- `matching_results.tsv` predictions are a subset of `candidate_pairs.tsv`.
- The model is MIT/Apache-2.0 compatible and well below the 8-billion-parameter
  maximum; this solution uses an offline LightGBM model.
- A valid submission must reach the challenge validator's `SCORED` status.

## Reproduce locally

Run from the repository root with challenge data under
`student_resource/dataset/{train,test}/`. Prep also copies train labels to
`work/ground_truth.tsv`, which the validation gates and labeled feature stage
read. The local pipeline uses the same cosine features and candidate proxy as
the notebook:

```powershell
py -3.14 -m venv .venv-laptop
.\.venv-laptop\Scripts\python.exe -m pip install -r code\business_entity_resolution\requirements.txt
.\.venv-laptop\Scripts\python.exe code\business_entity_resolution\src\prep.py
.\.venv-laptop\Scripts\python.exe code\business_entity_resolution\src\gate2.py
.\.venv-laptop\Scripts\python.exe code\business_entity_resolution\src\gate2b.py
$TOPK = 50  # replace with the smallest K meeting the gate's recall floor
$PROXY = "mean"
.\.venv-laptop\Scripts\python.exe code\business_entity_resolution\src\blocking.py --split train --topk $TOPK --proxy $PROXY
.\.venv-laptop\Scripts\python.exe code\business_entity_resolution\src\features.py --split train --labels --neg-ratio 0.05
.\.venv-laptop\Scripts\python.exe code\business_entity_resolution\src\gate3.py
.\.venv-laptop\Scripts\python.exe code\business_entity_resolution\src\train.py --full
.\.venv-laptop\Scripts\python.exe code\business_entity_resolution\src\infer.py
.\.venv-laptop\Scripts\python.exe student_resource\utils\validate_submission.py `
  --matching output\matching_results.tsv --candidate output\candidate_pairs.tsv `
  --test-dir student_resource\dataset\test
```

The full local run has not been timed on this machine. Stage B's cosine recall
table should be reviewed before generating all train features; `TOPK=50` is a
starting point, not a measured full-scale result for the cosine proxy.

The supported Kaggle path is the generated self-contained notebook:

```powershell
python kaggle/build_notebook.py > kaggle/kaggle_pipeline.ipynb
```

Import the notebook into a CPU Kaggle session, attach the payload dataset, run
once with `SAMPLE_N > 0`, then run the full pipeline with `SAMPLE_N = 0`.
Select `TOPK` and `PROXY` from Stage B before rerunning stages C through H.
Stage G must print `PASS`; Stage H produces `kaggle_return.zip`.

For Google Colab with Drive checkpoints and recall/feature gates before model
fitting, use [`kaggle/colab_pipeline.ipynb`](../../kaggle/colab_pipeline.ipynb)
and follow [`kaggle/COLAB_INSTRUCTIONS.md`](../../kaggle/COLAB_INSTRUCTIONS.md).

## Validate outputs

```powershell
python student_resource/utils/validate_submission.py `
  --matching output/matching_results.tsv `
  --candidate output/candidate_pairs.tsv `
  --test-dir student_resource/dataset/test

python code/business_entity_resolution/tests/check_outputs.py `
  --matching output/matching_results.tsv `
  --candidate output/candidate_pairs.tsv `
  --test-dir student_resource/dataset/test
```

Both checks must pass. The streaming checker is preferred for very large
candidate files.

## Release checklist

- Record the Stage B recall/volume table and selected `TOPK`/`PROXY`.
- Record validation macro F_0.5, threshold, and match-count bucket scores.
- Confirm Stage G prints `PASS` and download `kaggle_return.zip`.
- Place both TSVs under `output/`.
- Place `src/`, this README, and `requirements.txt` under
  `code/business_entity_resolution/`.
- Fill `Documentation_template.md`; remove all `PENDING` placeholders.
- Validate both TSVs, then create the exact archive structure shown above.

See [`kaggle/UPLOAD_INSTRUCTIONS.md`](../../kaggle/UPLOAD_INSTRUCTIONS.md) for
the Kaggle sequence and [`student_resource/Documentation_template.md`](../../student_resource/Documentation_template.md)
for the methodology report.
