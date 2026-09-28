# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Your Team Name]  
**Team Members:** [List all team members]  
**Submission Date:** [Date]

### Submission compliance statement

**Complete this statement after producing the final artifacts.** The final
archive follows the required structure: `output/` contains both
tab-separated output files; `code/business_entity_resolution/` contains the
complete `src/` implementation, this README, and `requirements.txt`; and this
methodology document is at the archive root. Every Source 1 test entity is
written exactly once. Predictions contain only existing Source 2/Source 3 IDs,
contain no duplicates or self-matches, and are a subset of the final candidate
set. The model is offline, MIT/Apache-2.0 compatible, and below the 8-billion
parameter limit. The official validator must report `PASS`, followed by a
`SCORED` challenge status.

---

## 1. Executive Summary

The solution is a fully offline, classical three-stage pipeline: (1) recall-first
multi-key blocking over a hand-rolled inverted index, built on script-aware
normalization that includes a hand-written deterministic Indic-to-ASCII
transliterator, so name and address keys stay comparable across Latin and nine
Indic scripts; (2) eleven pairwise similarity and address features, including
sparse word-level TF-IDF cosine similarities, scored by a LightGBM binary
classifier; (3) a global decision threshold swept specifically
for macro F_0.5 on an S1-entity-level validation split, with a per-match-count
bucket report guarding the singleton cohort. The competitive effort is placed
where it is durable at this scale — blocking recall (the naive comparison space
is 22.7 trillion pairs) and precision-first threshold calibration — while
candidate volume is controlled by a sparse TF-IDF cosine proxy pre-score with
per-entity top-K truncation. At the shipped setting (proxy=mean, K=50) the
full-scale run measures recall@K = 88.5223% at 48.3 candidates per S1 entity
(stratified 10,000-entity validation split) — below both the 97.70%
un-truncated gate and the 92.6289% sweep-table best; this is a deliberate
recall-for-volume trade that keeps the run inside the 12 h cap.
<!-- source: live full-scale run log, Stage B, pasted verbatim: "CONFIG honored: proxy=mean topk=50 recall=88.5223% avg_cands=48.3 (table best recall 92.6289%)"; 97.70% = Phase-2 un-truncated gate (Section 3) -->

---

## 2. Methodology

### 2.1 Problem Analysis

**Task.** Match each deduplicated Source1 business entity to 0..many noisy
records in Source2/Source3. The scored metric is macro-averaged F_0.5 over ALL
S1 entities, with per-entity precision/recall and singleton edge cases:
F_0.5 = 1.25·P·R / (0.25·P + R). Beta = 0.5 weights precision twice as heavily
as recall, so wrong merges cost more than missed matches; singletons (entities
with no match) score as entities in their own right and count fully in the
macro average.

**Data audit** (exhaustively verified on the full files, not samples):

| File | Rows | Blank address |
|---|---|---|
| train_source1 | 2,206,821 | 0 |
| train_source2 | 5,034,616 | 168,967 (3.36%) |
| train_source3 | 5,285,603 | 175,916 (3.33%) |
| test_source1 | 1,732,544 | 0 |
| test_source2 | 4,887,273 | 129,408 (2.65%) |
| test_source3 | 5,082,316 | 136,098 (2.68%) |

Key findings that shaped the design:

- **Integrity.** Zero malformed rows (tab count) across all 7 files; zero
  duplicate entity_ids; zero prefix mismatches; all ids match `^S[123]-\d+$`.
  Ground truth is a perfect bijection with source1: no orphans, no duplicate
  keys, clean comma-joined lists, every matched id exists in its source.
- **Ground-truth structure.** 7,638,365 total true pairs (48.36% to Source2,
  51.64% to Source3); match-list length median 3, max 11; 123,247 true
  singletons (5.59% of S1 entities).
- **Country.** Raw labels are exactly {US, India} in train and {US, India,
  France} in test (train ≈ 60/40 US/India). Cross-country true matches:
  **0 / 7,638,365 (exhaustive)** — country is a safe hard blocking filter.
  For France the assumption is extrapolated (no labeled French pairs exist)
  and is handled by the same generic code path with no country branching.
- **Name noise.** Length p5=12 / median=24 / p95=37 / max=105 chars; median 4
  tokens. True-pair token Jaccard median 0.60, but **15.7% have Jaccard = 0**
  and 25.6% < 0.4; Levenshtein ratio median 0.778 (13.8% < 0.4). Exact name
  equality after lowercase/strip/collapse covers only 15.77% of true pairs
  (32.89% after legal-suffix stripping; suffixes observed: Inc, LLC, Pvt, Ltd,
  Limited, Corp, Corporation, Company, Incorporated, Co, LLP, GmbH, PLC,
  Trust, plus bracketed forms). **13.9% of all true pairs are
  one-side-non-Latin** — nine Indic scripts (Devanagari, Telugu, Kannada,
  Tamil, Bengali, Gujarati, Malayalam, Oriya, Gurmukhi) plus Latin-1 accents
  and ZWNJ; both-non-Latin pairs: 0%. Script-switch pairs retain address
  signal (address Jaccard median 0.556; 53.7% > 0.5), so address-based keys
  rescue them. Random same-country negatives: name Jaccard median 0, only
  0.05% ≥ 0.5.
- **Address noise.** Unordered free-text components with state
  name/code/native-script swaps (OH/Ohio/महाराष्ट्र), city substitutions
  (Greenburgh/White Plains), typos, N/A placeholders, and unit numbers that
  appear/disappear. True-pair address Jaccard median 0.50 vs 0 for negatives
  (negative p95 = 0.083). **ZIP/PIN present on neither side in 91.4% of true
  pairs**, so PIN logic was demoted to non-primary. 4.41% of matched pairs
  have a blank address on the noisy side (S1 is never blank), so the model
  must degrade to name-only evidence without crashing.
- **Adversarial structure.** **1,307,881 S1×(S2/S3) pairs have byte-identical
  normalized name+country yet are NOT true matches** — name equality alone
  must never decide a match. S1 itself is not unique on name+country (844,718
  rows share name+country with another S1 row, in 177,643 groups), so
  name+country cannot serve as an implicit entity key anywhere.
- **Scale.** Naive train pair count: 2,206,821 × 10,320,219 =
  **22,774,876,013,799 (~22.7T)**. All candidate generation is therefore
  index-based, O(records). The best single blocking key (country + first-3
  normalized name characters) reaches only **75.50%** recall over all 7.64M
  true pairs — multi-key blocking is mandatory.

### 2.2 Solution Strategy

The pipeline is: **normalization → multi-key blocking → pairwise
string-similarity features → LightGBM binary classifier → global F_0.5-tuned
threshold → strict-schema outputs** (`matching_results.tsv` +
`candidate_pairs.tsv`, tab-separated, comma-joined ids, matches ⊆ candidates).

**Normalization as the foundation.** `norm_text` applies NFKC → Indic-to-ASCII
transliteration → lowercase → NFKD + stripping of combining marks (folds é→e)
→ non-alphanumeric-to-space → whitespace collapse. Names additionally get a
legal-suffix strip from both ends (both variants retained as separate
columns), and addresses yield a street-number feature (first leading numeric
token). The transliterator is a **hand-written deterministic character-mapping
table** derived from `unicodedata.name()` over code points 0x0900–0x0D7F: all
nine Indic scripts share Unicode letter names (e.g. "LETTER KA" = /k/
everywhere), so one name-to-romanization map covers them. It is stdlib-only,
offline, and deterministic — it is the team's own normalization code, not a
lookup service or a bundled external linguistic resource, and no network calls
exist anywhere in the pipeline. The normalization gate passed with **27/30
real true-match pairs improved-or-equal** (e.g. 'Raj Investments LLP' ↔
'ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி' moves from token Jaccard 0 to 0.25 with
matching tokens); the 3 unimproved pairs are genuine renames/DBA names,
irreducible by normalization. A synthetic self-check
(`norm_text("Société Générale") == "societe generale"`) guards the
accented-Latin path — this check caught and fixed a real bug during
development.

**Key design decisions.**

- *Classical GBM over embeddings:* interpretability, speed at this data scale,
  and trivial compliance with the model constraints. The ≤8B-parameter
  constraint is read as a **ceiling, not a target** — a LightGBM model's leaf
  values number in the millions at most — and this interpretation is stated
  explicitly rather than left implicit.
- *Validation split by S1 ENTITY, never by pair,* to avoid pair leakage: 18%
  of S1 ids (`pandas.sample(frac=0.18, random_state=42)`), saved to
  `work/val_s1_ids.txt`; the Kaggle training run reuses this exact file. A
  separate early-stopping set is carved from train-fold entities only
  (rng seed 1), so model capacity and the decision threshold are never tuned
  on the same data.
- *Country as a hard filter* (0/7,638,365 cross-country matches, exhaustively
  verified), required for the compute budget; the France extrapolation is
  documented as a stated limitation.
- *Precision-first calibration:* thresholds are swept against macro F_0.5
  (never F1 or accuracy), with a per-bucket report so the singleton cohort
  cannot be silently under-served.

**Approach Type:** Blocking + Classifier (multi-key inverted-index blocking →
hand-engineered pairwise features → gradient-boosted decision trees → globally
tuned threshold).

**Core Innovation:** Recall-first blocking under an explicit recall@K volume
gate. Script-aware normalization plus five complementary country-scoped key
families hold un-truncated blocking recall at 97.70% on a held-out validation
subsample. A sparse TF-IDF cosine proxy pre-score with per-S1 top-K truncation
then cuts candidate volume to 48.3 per entity (shipped setting proxy=mean,
K=50; measured on a stratified 10,000-entity validation split inside the
full-scale run) — a reduction ratio of ~4.8e-6 vs the ~1.73e13 naive test
pair space — at a measured recall@50 of 88.5223%: an explicit,
honestly-stated recall-for-volume trade under the 12 h cap (the sweep table's
best recall is 92.6289% at a setting not selected for runtime reasons).
<!-- source: Stage B paste "CONFIG honored: proxy=mean topk=50 recall=88.5223% avg_cands=48.3 (table best recall 92.6289%)"; naive test pairs 1,732,544 × 9,969,589 = 17,272,751,604,416 ≈ 1.73e13 (both factors from work/full_run_log_topk500_20260927.txt:37); reduction ratio 48.3/9,969,589 = 4.845e-6 (derived arithmetic, verified in-session) -->

---

## 3. Candidate Generation (Blocking)

The comparison space is reduced by a country hard filter followed by the union
of five country-scoped key families over a hand-rolled inverted index (Splink
was deliberately deferred as a stretch goal — zero install risk, full
debuggability, and custom script-aware keys were needed regardless).

- **Blocking keys used:**
  - `c3` — first 3 characters of the suffix-stripped normalized name (prefix
    key, no document-frequency cap);
  - `t` — each name token, df-capped (default 1000);
  - `a` — each address token of length ≥ 3, df-capped (default 2000);
  - `sn` — street number, df-capped (default 2000);
  - `fn` — full normalized suffix-stripped name, df-capped (default 20000).

  df caps are applied at **query time**, so one built index serves all cap
  settings without a rebuild. Index construction is a single chunked pass into
  dict-of-`array('i')` postings (memory-lean for a 16GB machine): the 10.32M
  pool records yield 8,533,382 keys / 115,094,877 postings in 407s.

- **Candidate pairs generated:** the top-K knob is fixed by the full-scale
  run at proxy=mean, K=50. Measured inside that run (Stage B, stratified
  10,000-entity validation split, cosine proxy): recall@50 = **88.5223%** at
  **48.3 candidates per S1 entity**. At test scale the naive space is
  1,732,544 × 9,969,589 = 17,272,751,604,416 ≈ **1.73e13** pairs, so the
  shipped setting is a reduction ratio of ~**4.8e-6** vs naive
  (48.3 / 9,969,589); extrapolating the measured per-entity mean over the
  1,732,544 test S1 entities gives ≈ **8.4e7** test candidate pairs
  (extrapolated arithmetic, not the logged total). Exact logged train
  total: the preserved full-scale log records **952,616,570 train candidate
  pairs** at its Stage C (`work/full_run_log_topk500_20260927.txt:9799`;
  that run was launched at TOPK=500 — a larger volume point than the shipped
  K=50, so it bounds rather than equals the shipped total). The shipped
  run's exact train/test candidate-pair totals: [PENDING — insert from the
  live run log when its Stage C / Stage G complete].
  <!-- source: Stage B paste "CONFIG honored: proxy=mean topk=50 recall=88.5223% avg_cands=48.3"; test S1 1,732,544 and pool 9,969,589 = 4,887,273 + 5,082,316 from work/full_run_log_topk500_20260927.txt:33-37; naive product, 48.3/9,969,589 = 4.845e-6, and 1,732,544 × 48.3 = 83,681,875 ≈ 8.4e7 verified by python in-session; "Stage C DONE: 952,616,570 train candidate pairs" verbatim from work/full_run_log_topk500_20260927.txt:9799 (read in-session) -->

- **How you ensured true matches were not lost:**
  - Multi-key union: no single key clears the recall bar (best single key
    75.50%), so five complementary families are unioned after the country
    filter — prefix keys (`c3`) survive name corruption, token keys (`t`)
    survive reordering/partial renames, `fn` catches full-name identity, and
    address keys (`a`, `sn`) rescue script-switched and renamed entities.
  - Script-aware normalization: the deterministic transliterator makes
    cross-script name keys comparable (13.9% of true pairs are
    one-side-non-Latin); script-switch pairs are additionally rescued by
    address-token keys (address Jaccard median 0.556 on that cohort).
  - A hard recall gate on a held-out S1-entity validation split: the
    un-truncated gate measures **97.7038%** (caps (1000, 2000, 2000, 20000),
    ≥95% target met); the top-K gate then re-measures recall@K for
    K ∈ {10, 20, 30, 50, 100, 200} with the cosine proxy. At the shipped
    K=50/mean setting the full-scale run measures **88.5223%** — below the
    ≥95% un-truncated target. This shortfall is accepted deliberately and
    stated as the recall ceiling of the shipped configuration: K=200 (the
    next volume point on the sweep) roughly doubles stages D+G (~6.5 h) and
    would overrun the 12 h cap, so candidate volume, not recall, set K=50.
  - df caps applied at query time let cap settings be re-measured against one
    index; the gate reports recall, pairs-per-S1, and reduction-vs-naive for
    each setting so the recall/volume trade-off is always explicit.

---

## 4. Matching Model

**Features used:** eleven pairwise features, computed in chunks:

- Name features:
  - `name_jaccard_raw` (lowercase raw tokens),
    `name_jaccard_normalized` (non-suffix-stripped normalized token bag),
    `name_jaccard_suffix_stripped` (suffix-stripped token bag) — a true
    3-way distinction;
  - `name_cos` (word-level TF-IDF cosine similarity over normalized names);
  - `token_overlap_count` (suffix-stripped token intersection);
  - `len_delta_name`.
- Address features:
  - `addr_jaccard`, `addr_cos` (word-level TF-IDF cosine) — NaN when either address is blank (blank is a
    real state, not a zero);
  - `street_number_match` (1/0, NaN when either side is unextractable);
  - `missing_address_flag` (S2/S3 side blank);
  - `len_delta_addr`.
- Other: the `is_match` training label is joined via precomputed per-entity
  sets (O(1) membership, no per-row string splitting).

**Feature-cost engineering (why TF-IDF cosine).** `name_cos` and `addr_cos`
are word-unigram TF-IDF cosine similarities, and the same sparse TF-IDF
cosine powers the Stage-B proxy pre-score. They replace an earlier per-pair
normalized Levenshtein feature: Levenshtein measured **17.55 s per 100k
pairs**, extrapolating to **~25 h** for one full run — the 12 h cap is
exceeded before model training even starts. Word-unigram TF-IDF cosine is
**~1000x cheaper per pair** and brings the full run to **~4 h**, which is
what makes the shipped K=50 setting (48.3 candidates per entity) executable
end to end — features, training, threshold sweep, and test inference — inside
the deadline. TOPK is the volume knob: K=200 would have doubled stages D+G
(~6.5 h), over the cap.
<!-- source: engineering measurements from the full-scale run: 17.55 s/100k pairs for per-pair Levenshtein (measured), ~1000x cheaper TF-IDF cosine, full run ~4 h, K=200 doubles D+G to ~6.5 h (measured/user-verified run facts) -->

**Model type:** LightGBM binary classifier (`objective=binary`). Parameters:
learning_rate 0.08, num_leaves 128, min_data_in_leaf 200, feature_fraction 0.9,
bagging_fraction 0.9, num_boost_round 2000 with early stopping (100 rounds).
Class imbalance is handled with `scale_pos_weight = (neg/pos)` computed at
runtime — preferred over `is_unbalance` (available behind a flag) for better
probability calibration under F_0.5 threshold sweeps. The early-stopping
evaluation set is 10% of train-fold S1 entities (rng seed 1), disjoint from
the 18% validation split used for threshold selection; because both splits are
by S1 entity, no candidate pair straddles split boundaries. The model is
orders of magnitude below the 8B-parameter ceiling, MIT-licensed, and fully
offline.

**Threshold selection method:** the probability cutoff is swept 0.10–0.95 in
steps of 0.02 on the held-out 18% S1-entity validation split, maximizing macro
F_0.5 (singletons included) as computed by a standalone evaluator implementing
the challenge metric: per-entity F_0.5 = 1.25·P·R/(0.25·P+R), macro-averaged
over ALL S1 entities. The evaluator is hand-verified by assertions: true =
{a,b,c} vs pred = {a,b,d} → F_0.5 = 2/3; both-empty → 1.0; pred-only → 0.0;
true-only → 0.0. Alongside the aggregate, F_0.5 is broken out by true
match-count bucket (0 / 1 / 2-3 / 4+): if the singleton bucket (5.59% of S1
entities) lags materially, the remedy is sought in blocking (spurious
candidates a global threshold cannot cleanly reject) rather than re-sweeping
alone. The selected threshold and scores are persisted to
`model_config.json` as `{threshold, val_macro_f05, per_bucket_f05}`.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** [PENDING — the full-scale run is still in
  progress; insert Stage F `val_macro_f05` (best validation macro F_0.5 at
  the swept threshold) and the selected threshold from the run log (also
  persisted as `model_config.json` fields `val_macro_f05` and `threshold`)
  once it completes.] The tiny smoke run checks plumbing only and is not a
  model-quality result.
- **Common false positives (wrong merges):** the designed-in suspect
  population is the 1,307,881 S1×(S2/S3) pairs with byte-identical
  normalized name+country that are NOT true matches; the adversarial gate
  (`gate3.py`, notebook Stage E) requires address features to separate this
  subset (required: median addr_jaccard gap > 0.1) before training, and the
  F_0.5 objective itself penalizes wrong merges 2:1 over misses. No
  post-hoc pair-level error mining was run under the time cap; the
  per-bucket F_0.5 report is the coarse error view. [PENDING — insert Stage
  F `per_bucket_f05` from the run log / `model_config.json` once the
  full-scale run completes.]
- **Common false negatives (missed matches):** two structural sources are
  known before any model output. (a) The blocking ceiling: at the shipped
  K=50/mean setting recall@K is 88.5223%, so ~11.48% of true pairs are
  absent from the candidate set entirely and no threshold can recover them
  — this is the accepted price of the 12 h cap, not a modeling defect. (b)
  Irreducible renames/DBA names: 3 of the 30 audited normalization-gate
  true-match pairs are unimprovable by normalization. Read jointly with the
  per-bucket F_0.5 report above.
  <!-- source: (a) 100 - 88.5223 = 11.4777 ≈ 11.48%, derived from the Stage B paste (verified in-session); (b) Phase-1 normalization gate (30 audited pairs) -->

---

## 6. Conclusion

The solution is a deliberately conservative, fully offline classical pipeline
whose edge is placed where it is durable at 22.7T naive pairs: blocking recall
(script-aware normalization with a hand-written deterministic transliterator,
five complementary key families, and a measured recall gate) and
precision-first calibration for macro F_0.5 (entity-level validation split,
per-bucket singleton guard), rather than in a heavier model. Engineering
lessons: everything must be index-based and streaming at this scale — two
out-of-memory failures reshaped the design toward single-pass chunked index
builds and per-chunk output emission — and validation must be split by entity,
never by pair, or the macro F_0.5 estimate leaks. The approach stays trivially
compliant with the ≤8B-parameter and license constraints, with both
interpretations documented explicitly for reviewers.

---

## Appendix

### A. Code Artefacts

Complete runnable code ships under `code/business_entity_resolution/`:

```
code/business_entity_resolution/
├── src/normalize.py            Phase 1: normalization + transliteration (+ self-check)
├── src/prep.py                 Phase 2a: one-time normalization of all 6 source
│                               files -> work/*.parquet + slim id lookups
├── src/blocking.py             Phase 2: inverted-index candidate generation
├── src/gate2.py                Phase 2 gate: recall + volume on the val split
├── src/gate2b.py               Phase 2 gate b: recall@K for top-K selection
├── src/features.py             Phase 3: pairwise feature table (parquet)
├── src/gate3.py                Phase 3 gate: separation + adversarial check
├── src/train.py                Phase 4: LightGBM training (RUNS ON KAGGLE)
├── src/evaluate.py             macro F_0.5 evaluator (standalone, importable)
├── src/infer.py                Phase 5: test inference + output writers
├── src/make_kaggle_package.py  Kaggle handoff bundler
├── requirements.txt            pinned versions
└── README.md                   reproduction commands
```

**Entry points, in run order** (from repo root, `student_resource/` a sibling
of `code/`):

1. `python code/business_entity_resolution/src/prep.py` — normalize all six
   source files once into `work/*.parquet` plus slim id lookups (~25 min,
   16GB RAM).
2. `python code/business_entity_resolution/src/gate2.py` — blocking gate on
   the 18% val split (writes `work/val_s1_ids.txt`, the exact split the
   Kaggle run must reuse); then `.../gate2b.py` — recall@K gate that fixes
   the top-K volume knob.
3. `python .../blocking.py --split train` and `--split test` — candidate
   generation (df caps configurable: `--token-cap`, `--addr-cap`, `--sn-cap`,
   `--fn-cap`).
4. `python .../features.py --split train --labels` then `--split test` —
   pairwise features.
5. `python .../gate3.py` — class-separation check including the adversarial
   identical-name negatives.
6. **Kaggle run (single self-contained notebook)** — all heavy stages execute in
   `kaggle/kaggle_pipeline.ipynb`, ONE notebook generated from `src/` by
   `kaggle/build_notebook.py` (every pipeline function is inlined into the cells
   via AST extraction — no subprocess calls, and no `src/` needed on the
   dataset). Flow: build the payload dataset with
   `python kaggle/make_dataset_payload.py` (dataset/ TSVs + ground_truth.tsv +
   val_s1_ids.txt), create/version the Kaggle dataset (Legacy API key auth),
   import the notebook (CPU, Internet off), and run in order: Cell 0 (deps —
   uses Kaggle's preinstalled stack, nothing pinned), Stage A prep, Stage B
   recall@K sweep (K × proxy × caps table), Stage C train candidates, Stage D
   features + labels (negative downsample), Stage E adversarial gate, Stage F
   LightGBM + F_0.5 sweep, Stage G test inference + inline validator (must
   print PASS), Stage H `kaggle_return.zip`. The CONFIG cell exposes
   TOPK / PROXY / NEG_RATIO / SAMPLE_N; after Stage B prints its table, TOPK
   and PROXY are set from it and C–H re-run. The run returns exactly
   `work/model.txt` (LightGBM save_model format) and `work/model_config.json`
   (`{threshold, val_macro_f05, per_bucket_f05}`) plus `output/*.tsv`, all
   inside `kaggle_return.zip`, which is dropped back into local `work/` /
   `output/`.
7. `python .../infer.py` — test inference in 200k-entity chunks; writes
   `output/matching_results.tsv` + `output/candidate_pairs.tsv`
   (tab-separated, comma-joined ids; matches ⊆ candidates enforced at write
   time; every test S1 entity gets exactly one row in both files, including
   zero-candidate entities as empty-list singletons; emission is per-chunk
   append so nothing accumulates in memory). This is the local reference path;
   on Kaggle the notebook's Stage G runs the same inlined code and validates
   inline. `infer.py --dummy` runs the whole test-side path with a rule-based
   scorer (0.5 × name cosine + 0.5 × address cosine, placeholder
   threshold 0.75) to prove the writers and validator before a trained model
   exists.
8. `python student_resource/utils/validate_submission.py --matching
   output/matching_results.tsv --candidate output/candidate_pairs.tsv
   --test-dir student_resource/dataset/test` — must print PASS.

**Reproducibility.** `requirements.txt` pins exact versions from the
development environment: pandas==3.0.6, numpy==2.5.3, pyarrow==25.0.1,
lightgbm==4.7.0, scikit-learn==1.9.1. No network calls exist anywhere in the
pipeline; all normalization/transliteration is stdlib-only and deterministic.
The Kaggle notebook (`kaggle/kaggle_pipeline.ipynb`) is GENERATED from `src/`
by `kaggle/build_notebook.py` — rebuilds go through the builder, never
hand-edits — and on Kaggle it deliberately uses the image's preinstalled
numpy/pandas/lightgbm, installing only what is missing and never pinning
versions (a pinned numpy once broke the image's scipy and killed the lightgbm
import); the pins above remain the local reproduction contract.
`train.py --sample` runs the training plumbing on synthetic data for a fast
sanity check; only `train.py --full` on the real feature files produces
meaningful validation metrics and a usable model. It can run locally or in the
Kaggle notebook, subject to available time and memory.

### B. Additional Results

- **Phase 1 normalization gate:** 27/30 real true-match pairs
  improved-or-equal after normalization (summed name+address Jaccard, raw vs
  normalized); the 3 unimproved pairs are genuine renames/DBA names. A
  synthetic self-check (`Société Générale`) permanently guards the
  accented-Latin fold.
- **Single-key blocking recall ceiling:** country + first-3 normalized name
  characters = 75.50% over all 7,638,365 train true pairs — the empirical
  motivation for multi-key blocking.
- **Phase-2 gate, un-truncated** (50k-entity validation subsample, caps
  (1000, 2000, 2000, 20000)): recall 97.7038% (target ≥95%), 716,959,975
  candidate pairs = 14,339.2 per S1 entity = 0.138943% of the naive pair
  count; extrapolated ~31.6B pairs at full train scale — the finding that
  motivated the proxy pre-score + per-S1 top-K design. **Top-K gate,
  full-scale run** (Stage B, stratified 10,000-entity validation split,
  cosine proxy): shipped setting proxy=mean, K=50 → recall@50 **88.5223%**
  at **48.3 candidates per S1 entity**, a reduction ratio of ~**4.8e-6** vs
  the ~1.73e13 naive test pair space; the sweep table's best recall is
  **92.6289%**, a setting not selected because K=200 roughly doubles stages
  D+G (~6.5 h) and overruns the 12 h cap.
  The Stage B sweep table below is transcribed verbatim from the preserved
  full-scale log (`work/full_run_log_topk500_20260927.txt:56-72`). That run
  was launched with TOPK=500, so its printed grid is K ∈ {100, 200, 300,
  500} × proxy ∈ {mean, max} over two cap tuples — not the {10, 20, 30, 50,
  100, 200} grid this template originally assumed; the shipped K=50 point
  (last row) is measured by the live run's Stage B paste, not by this table.

  | caps | proxy | topk | recall | avg_cands_per_s1 | reduction_vs_naive |
  |---|---|---|---|---|---|
  | (1000, 2000, 2000, 20000, 2000) | mean | 100 | 0.901661 | 95.0829 | 0.000009 |
  | (1000, 2000, 2000, 20000, 2000) | mean | 200 | 0.914597 | 185.1778 | 0.000018 |
  | (1000, 2000, 2000, 20000, 2000) | mean | 300 | 0.919083 | 271.0796 | 0.000026 |
  | (1000, 2000, 2000, 20000, 2000) | mean | 500 | 0.924263 | 431.5293 | 0.000042 |
  | (1000, 2000, 2000, 20000, 2000) | max | 100 | 0.857730 | 95.0829 | 0.000009 |
  | (1000, 2000, 2000, 20000, 2000) | max | 200 | 0.892140 | 185.1778 | 0.000018 |
  | (1000, 2000, 2000, 20000, 2000) | max | 300 | 0.902616 | 271.0796 | 0.000026 |
  | (1000, 2000, 2000, 20000, 2000) | max | 500 | 0.916623 | 431.5293 | 0.000042 |
  | (200, 300, 500, 5000, 1000) | mean | 100 | 0.811223 | 75.1169 | 0.000007 |
  | (200, 300, 500, 5000, 1000) | mean | 200 | 0.818400 | 128.5620 | 0.000012 |
  | (200, 300, 500, 5000, 1000) | mean | 300 | 0.820513 | 164.9402 | 0.000016 |
  | (200, 300, 500, 5000, 1000) | mean | 500 | 0.822770 | 205.4919 | 0.000020 |
  | (200, 300, 500, 5000, 1000) | max | 100 | 0.779939 | 75.1169 | 0.000007 |
  | (200, 300, 500, 5000, 1000) | max | 200 | 0.802628 | 128.5620 | 0.000012 |
  | (200, 300, 500, 5000, 1000) | max | 300 | 0.809487 | 164.9402 | 0.000016 |
  | (200, 300, 500, 5000, 1000) | max | 500 | 0.818169 | 205.4919 | 0.000020 |
  | (live run paste) | mean | 50 | 0.885223 | 48.3 | ~4.8e-6 (derived) |

  This table's best is recall = 0.924263 (caps (1000, 2000, 2000, 20000,
  2000), proxy=mean, topk=500; `work/full_run_log_topk500_20260927.txt:73`
  "Stage B DONE — best"). The two full-scale runs swept different grids, so
  their table bests differ: 0.924263 here vs the live paste's 92.6289%
  quoted in §2.2/§3 — both are stated rather than silently merged.
  <!-- source: 16 table rows verbatim from work/full_run_log_topk500_20260927.txt:57-72 (header :56), "Stage B DONE — best: caps=(1000, 2000, 2000, 20000, 2000) proxy=mean topk=500 recall=92.4263% avg_candidates=431.5" from :73, all read in-session; K=50 row from the user-pasted live-run Stage B line "CONFIG honored: proxy=mean topk=50 recall=88.5223% avg_cands=48.3 (table best recall 92.6289%)", its reduction cell 48.3/9,969,589 = 4.845e-6 derived (verified in-session, same as §2.2); naive test pairs 1,732,544 × 9,969,589 = 17,272,751,604,416 (factors from work/full_run_log_topk500_20260927.txt:37, product verified in-session) -->
- **Phase-4 results:** [PENDING — insert Stage F val_macro_f05, selected
  threshold, and per_bucket_f05 from the run log once the full-scale run
  completes (artifacts `work/model.txt` + `work/model_config.json`).]

---

**Note:** Teams can modify sections according to their approach while maintaining clarity and technical depth.
