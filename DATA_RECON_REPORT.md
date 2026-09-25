# Amazon ML Challenge 2026 — Data Reconnaissance Report (verified & extended)

All numbers below were recomputed from the raw TSVs. Sections marked **[FULL]** are exact computations over all rows/pairs (not samples); sections marked **[SAMPLE n=50,000]** are estimates from a uniform random sample (5–50× larger than the earlier n=1,000 draft). No matching logic or model was built.

Sources: `student_resource/dataset/train/{train_source1,train_source2,train_source3,train_ground_truth}.tsv` and `student_resource/dataset/test/{test_source1,test_source2,test_source3}.tsv`.

## 1. Basic shape — [FULL]

| File | Rows | Columns | Blank address | Other blanks |
|---|---|---|---|---|
| Train source1 | 2,206,821 | 4 | 0 | 0 |
| Train source2 | 5,034,616 | 4 | 168,967 (3.36%) | 0 |
| Train source3 | 5,285,603 | 4 | 175,916 (3.33%) | 0 |
| Test source1 | 1,732,544 | 4 | 0 | 0 |
| Test source2 | 4,887,273 | 4 | 129,408 (2.65%) | 0 |
| Test source3 | 5,082,316 | 4 | 136,098 (2.68%) | 0 |
| Train ground truth | 2,206,821 | 2 | — | 123,247 blank `matched_entity_ids` |

- Malformed rows (wrong tab count): **0 across all 7 files**. Empty lines: 0. UTF-8 decodes cleanly (Devanagari/Gujarati script present in India rows; that is content, not corruption).
- Blank fields are true empty strings; only S2/S3 `business_address` and empty GT lists are ever blank. **Source1 never has a blank address** (both splits).
- Entity IDs: every row matches `^S[123]-\d+$`; no duplicate IDs anywhere; no prefix mismatches (no S1- id inside S2/S3 files, both splits).

## 2. Ground truth structure — [FULL]

Match-list length histogram (per S1 entity):

| Length | Count | | Length | Count |
|---|---|---|---|---|
| 0 (singleton) | 123,247 | | 6 | 164,868 |
| 1 | 119,157 | | 7 | 63,968 |
| 2 | 375,212 | | 8 | 18,680 |
| 3 | 530,841 | | 9 | 4,205 |
| 4 | 484,115 | | 10 | 534 |
| 5 | 321,957 | | 11 | 37 |

- Total matched IDs: **7,638,365** — S2: 3,693,619 (48.36%), S3: 3,944,746 (51.64%).
- Malformed lists: **none** — no trailing/leading commas, no internal whitespace, no double commas, no duplicate IDs within a list, no non-S2/S3 prefixes, and every matched ID exists in its source file.
- Perfect bijection: every S1 entity appears exactly once in GT; no orphans in either direction; no duplicate `source1_entity_id` keys.
- Median match-list length is 3; 72% of S1 entities have ≥3 matches — a match is usually *one-to-many* (S2/S3 are noisy duplicates).

## 3. Country distribution — [FULL]

| Split | Source1 | Source2 | Source3 |
|---|---|---|---|
| Train | US 1,323,633; India 883,188 | US 3,016,817; India 2,017,799 | US 3,170,056; India 2,115,547 |
| Test | US 663,106; France 259,452; India 809,986 | US 1,871,330; France 703,378; India 2,312,565 | US 1,945,701; France 731,615; India 2,405,000 |

- Distinct raw labels are exactly `US`, `India` (train) and `US`, `India`, `France` (test) — no casing/whitespace/typo variants anywhere.
- **Cross-country true matches: 0 out of 7,638,365 pairs — verified exhaustively, not by sampling.** Country is a safe hard blocking key for train. For test, France rows can only match France rows (by the same logic, though strictly this is an assumption extended to unseen labels).

## 4. Name field noise

Length percentiles (min/p5/p25/median/p75/p95/max) — [FULL over all rows]:

| Source | Name chars | Name tokens |
|---|---|---|
| Train S1 | 3/12/18/24/30/37/105 | 1/2/3/4/4/5/19 |
| Train S2 | 2/12/19/25/31/40/104 | 1/2/3/4/5/14/30 |
| Train S3 | 2/11/18/25/31/42/123 | 1/2/3/4/5/8/29 |
| Test S1/S2/S3 | nearly identical | nearly identical |

### True-pair similarity — [SAMPLE n=50,000]

| Measure | Token Jaccard | Levenshtein ratio |
|---|---|---|
| True matches (name) | 0 / 0 / 0.333 / 0.600 / 0.750 / 1 / 1 | 0 / 0.088 / 0.556 / 0.778 / 0.931 / 1 / 1 |
| True matches (name, after legal-suffix strip) | 0 / 0 / 0.333 / 0.600 / 1 / 1 / 1 | 0 / 0.051 / 0.556 / 0.842 / 1 / 1 / 1 |
| Random same-country negatives | 0 / 0 / 0 / 0 / 0 / 0.167 / 0.600 | 0 / 0.059 / 0.143 / 0.192 / 0.250 / 0.407 / 0.926 |

- Exact name match (raw string): 4.64% of all 7.64M pairs. After lowercase+strip+whitespace-collapse: **15.77%**. After additionally stripping legal suffixes at both ends: **32.89%**.
- Suffixes actually observed at name endings: Inc, LLC, Pvt, Ltd, Limited, Corp, Corporation, Company, Incorporated, Co, LLP, GmbH, PLC, Trust, plus bracketed/punctuated variants (`[LLP]`, `(Inc.)`, `L.L.C.`).
- **Hard floor for blocking/classification: 15.7% of true pairs have name token Jaccard = 0, and 25.6% are below 0.4.** 13.8% have Levenshtein ratio < 0.4. Causes seen in the verbatim pairs below: transliteration to Devanagari/Gujarati (exact translation-level swap: `Jain Engineering` ↔ `जैन इंजीनियरिंग`), domain-style variants (`gyhomes.com`, `ZZLINSTITUTIONSGROUP.COM`), OCR-garble (`Rornlf`, `Icbna`, `Ianterdsatoiaal`), renamed entities (`Halonovi f/k/a …`), symbol pollution (`#historicalfederation`, `***`, `>>`).
- Negatives are extremely separable on the low end: only 0.05% of same-country random negatives reach Jaccard ≥ 0.5, 1.06% ≥ 0.3. The danger zone is the *upper tail of negatives* colliding with the *lower tail of true matches* (see §6/H).

## 5. Address field noise

Length percentiles — [FULL]: train S1 11/27/33/41/70/103/256 chars; S2/S3 have min 0 (blanks). Test mirrors train.

### True-pair similarity — [SAMPLE n=50,000]

| Measure | Token Jaccard | Levenshtein ratio |
|---|---|---|
| True matches (address) | 0 / 0.077 / 0.333 / 0.500 / 0.714 / 1 / 1 | 0 / 0.103 / 0.444 / 0.711 / 0.857 / 1 / 1 |
| Random same-country negatives | 0 / 0 / 0 / 0 / 0 / 0.083 / 0.400 | 0 / 0.128 / 0.180 / 0.214 / 0.250 / 0.318 / 0.672 |

- [FULL, all 7.64M pairs] Empty address on the matched side: **4.41%**; never on the S1 side; never both sides.
- [FULL] Neither side contains a 5- or 6-digit ZIP/PIN pattern: **91.4%** of true pairs. Either side contains "Near/Nr.": ~3% (S1 side). So PIN extraction is nearly useless as a feature here; street number + street name + city tokens carry the signal.
- Addresses are free text: comma-separated but with inconsistent component order (`Georgetown, TX, 500 Westinghouse Road` vs `Westinghouse Rd, GEORGETOWN, TX`), state name ↔ code ↔ native-script swaps (OH/Ohio/`महाराष्ट्र`), abbreviations (St/Street, Rd/Road), city substitutions (Greenburgh ↔ White Plains; Cortlandt ↔ Cortlandt Manor), typos (`Raod`, `Somerst`, `Vne`), added/blank unit numbers, and `N/A` placeholders.

## 6. Cross-field signal check — [SAMPLE n=50,000 + FULL subsets]

- Name near-zero but address strong (name Jaccard < 0.2, address Jaccard ≥ 0.8): ~1–4% of true pairs (e.g., `GY Homes Pvt Ltd` ↔ `gyhomes.com`, address Jaccard 0.94). **Both fields are needed** — neither alone covers the true-match distribution.
- The empty matched-side address (4.41%, FULL) makes address unusable for those pairs — name/other signals must carry them.

### Adversarial traps — [FULL]  *(new section, not in earlier draft)*

1. **Identical names that are NOT matches:** 1,307,881 S1×(S2/S3) row pairs with byte-identical normalized name+country that are *not* labeled matched (e.g., `One Consultancy Private Limited` (S1-260581545) vs `One Consultancy Private Limited` (S2-895132123), India). Name equality is heavily ambiguous — **name-only blocking produces a large hard-negative pool**, and name match alone cannot decide.
2. **S1 is not name-unique:** 844,718 S1 rows share an exact name+country with another S1 row (177,643 groups, e.g., `Wildlife Center`+US). This directly contradicts treating name+country as an entity key despite S1 being "deduplicated."
3. **Country is adversarially clean but test-shifted:** France exists only in test (~14% of each test file). Any country-specific feature learned on train must degrade gracefully.
4. **Language switching inside true matches** (Devanagari/Gujarati ↔ Latin) means pure ASCII normalization would zero out a chunk of true pairs; scripts must be handled (§4 examples).
5. **Near-duplicate-but-different entities** exist in the negative pool (neg name Jaccard max 0.60, Lev max 0.93 in 50k draws) — a similarity threshold below ~0.6 Jaccard admits negatives.

## 7. Scale and blocking — [FULL coverage on all 7,638,365 true pairs]

| Measure | Value |
|---|---|
| Train pairs if no blocking: 2,206,821 × (5,034,616+5,285,603) | 22,774,876,013,799 |
| Test pairs if no blocking | 17,272,751,604,416 |
| country + first-3 normalized name chars | 75.50% |
| country + first-4 chars | 74.34% |
| country + first token | 68.88% |
| country + first-2 tokens | 49.51% |
| no-country + first-3 | 75.50% |
| union(first-3, first-token) | 75.59% |

- No cheap name-prefix key reaches 95%. Roughly a quarter of true pairs are unreachable by any single name-prefix key — consistent with §4 (translation, renames, garble). Multi-key blocking (name prefix ∪ address tokens ∪ token-set signatures) is required; e.g., address street-number + first address token are natural secondary keys, and the 15.7% name-Jaccard-0 pairs are exactly where address blocking rescues recall.

## 8. Other findings — [FULL]

- Duplicate entity IDs in any source file: 0 (both splits). Source-prefix mismatches: 0.
- GT ↔ source1 relationship is exact and clean (§2). S2/S3 contain no labels — they are the pool to be linked.
- Nothing in the data contradicts the problem statement except the two sharpened points: (a) "source1 is deduplicated" is true only at the entity_id level — identical name+country rows recur; (b) the test set's France rows mean any learned embedding/threshold must extrapolate to an unseen country.

## Notes — verdict on the earlier draft (PDF)

- **Confirmed [FULL]:** all row counts, blank counts, GT histogram, S2/S3 split, ID integrity, country label purity, and — upgraded from a 1,000-pair sample to an exhaustive audit — **zero cross-country true matches**.
- **Corrections/precision:** exact-after-normalization is 15.77% on all pairs (draft said 16.6% ± sample noise); "neither side has ZIP/PIN" is 91.4% (draft: 90.3%); name-prefix blocking coverage is 75.50% (draft: 74.6%) — still far below 95%.
- **Added:** 50k-sample similarity percentiles, exhaustive empty-address and adversarial identical-name-negative counts, blocking-key comparison table, and the verbatim pair examples in §4/§5 of the working log (30 name pairs + 30 address pairs, e.g. `'Surya Investment Limited' ↔ 'SURYA INVESTMENT LIMITED'`, `'Gidc Mithi Rohar, Taluka, Gandhidham, Gujarat…' ↔ 'ગુજરાત, PLOT NO.C-362 , GIDC MITHI ROHAR…'`).
- Random negatives were drawn from same-country S1×(S2∪S3) pools with matched IDs excluded; they were not verified label-by-label against the full GT, so the negative distribution is a slight *underestimate* of similarity (excluded IDs are all true matches, never negatives).
