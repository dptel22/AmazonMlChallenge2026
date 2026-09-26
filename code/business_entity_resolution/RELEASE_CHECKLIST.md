# Release checklist

| Deliverable | Verification | Status |
|---|---|---|
| Normalization tests | `.venv\Scripts\python.exe -m pytest code/business_entity_resolution/tests/test_normalize.py -q` | PASS (9/9 brief-1 suite) |
| F₀.₅ cross-check | `.venv\Scripts\python.exe -m pytest code/business_entity_resolution/tests/test_evaluate.py -q` | PASS (brief-1 suite) |
| Synthetic Phase-5 writer | `tests\run_synthetic_e2e.py` + `validate_submission.py --check-ids` | PASS; 60 rows |
| Blocking unit tests | `.venv\Scripts\python.exe -m pytest code/business_entity_resolution/tests/test_blocking.py -q` | PASS; 4 passed |
| Streaming checker | `python tests/check_outputs.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir student_resource/dataset/test` | PENDING real output |
| France sanity report | `python tests/france_sanity.py` | PASS; `work/france_sanity.txt` generated |
| Gate2b recall@K | `python src/gate2b.py` | PENDING compute agent |
| Full blocking/features | README phase commands | PENDING compute agent |
| Kaggle upload package | `python src/make_kaggle_package.py` | PENDING `feats_train/` |
| Kaggle return contract | `kaggle_train.ipynb` sanity cell | READY |
| Final validator on real output | `validate_submission.py --matching ... --candidate ... --check-ids` | PENDING real inference |
| Documentation template | Docs-agent checklist / review | PENDING remaining slots |

The synthetic output currently in `output/` is test data and must be overwritten by real inference before submission.
