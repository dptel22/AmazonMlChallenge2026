# Business Entity Resolution — Architecture Decision Record

Status: locked pending your sign-off on open items (marked ⚠️ below)
Last updated: build-start

---

## 1. Problem recap (constraints that shape every decision below)

| Fact | Source | Implication |
|---|---|---|
| 22.7T possible train pairs (2.2M × 10.3M) | data audit | Blocking MUST be index-based. Any O(pairs) code is dead on arrival. |
| Scored metric: macro F_0.5, singletons included | problem statement | Precision >> recall. Wrong to optimize F1 or accuracy. |
| Country: US/India train, +France test (unseen) | problem statement | No per-country branching logic anywhere. |
| 0/7.64M true matches cross country (exhaustive) | data audit | Country is a safe **hard filter**, not a soft feature. |
| No name-prefix key clears 95% recall alone (best = 75.5%) | data audit | Single-key blocking is a loss. Multi-key required. |
| 15.7% of true pairs have name-token Jaccard = 0 | data audit | Name-only blocking silently drops ~1 in 6 matches. Root cause: script switching (Devanagari/Gujarati↔Latin), OCR garble, renames. |
| 1.3M+ pairs: identical normalized name+country, NOT a match | data audit | Name equality is not sufficient evidence. Classifier needs joint signal. |
| S1 not unique on name+country (844K dupes) | data audit | Can't use name+country as an implicit key anywhere, including for dedup sanity checks. |
| ZIP/PIN present in <9% of pairs | data audit | Don't build ZIP-dependent logic as anything but a minor bonus feature. |
| S2/S3 address blank ~4.4%, S1 never blank | data audit | Pipeline must degrade to name-only scoring without crashing. |
| Zero external lookups allowed | problem statement | Offline libraries only. No network calls anywhere in inference path. |
| Model ≤8B params, MIT/Apache-2.0 | problem statement | Rules out any hosted/proprietary LLM as the matcher. |
| candidate_pairs.tsv must be the *last*-stage candidate set, matches ⊆ candidates | problem statement | Can't generate a "rough" blocking file and a separately-filtered smaller candidate file — they must be the literal input to the classifier. |

---

## 2. Adversarial review of the original plan

Going through the plan we built up in conversation and attacking each piece as if reviewing a competitor's submission before it ships.

### 2.1 "Use Splink for blocking + Fellegi-Sunter scoring"
**Attack:** Splink is designed for person/household record linkage (names, DOBs, addresses with fairly standard Western structure). This dataset has Devanagari/Gujarati script-switching, OCR garble, and DBA/trade-name swaps — none of which Splink's built-in comparison functions (levenshtein_at_thresholds, jaro_winkler, etc.) handle out of the box. Using it well requires custom comparison functions anyway, which erodes the "just use the library" time savings.
**Attack 2:** Splink is a real, non-trivial dependency (DuckDB backend, its own API surface). If it breaks in the actual grading/reproduction environment (the methodology doc says "anyone should be able to reproduce from code/"), that's a submission-killing risk with zero upside if it fails silently or behaves unexpectedly on 10M+ row files under hackathon time pressure.
**Verdict:** ⚠️ Downgrade Splink from "primary blocking engine" to "optional bonus / stretch, only if hand-rolled blocking is done and validated first." Hand-rolled blocking is boring but it's fully understood, fully debuggable, and has zero install risk. **Decision: hand-rolled multi-key inverted-index blocking is the committed v1. Splink is not on the critical path.**

### 2.2 "GBM classifier on hand-engineered string-similarity features"
**Attack:** This is likely what most competent teams will build. If everyone converges on Jaccard/Levenshtein + LightGBM, this alone won't differentiate a top-3 finish from a top-30 finish. Where's the edge?
**Counter-attack:** The edge isn't the classifier — it's (a) blocking recall ceiling, which most teams under time pressure will under-invest in, and (b) precision calibration specifically for F_0.5, which most teams will get lazily wrong by defaulting to a 0.5 threshold or optimizing F1. Both of these are cheap to do right and expensive to get right under time pressure — that's exactly where an edge is durable.
**Verdict:** Keep GBM as the classifier. Do NOT treat "add a fancier model" as the lever for winning — treat blocking recall and threshold calibration as the two levers, and spend disproportionate time there.

### 2.3 "Transliteration library for script-crossing"
**Attack:** Is an offline transliteration library actually inside the fair-play rules, or does "any external data augmentation from internet sources" implicitly include static linguistic resource files bundled inside a pip package (which themselves came from "external" curated data)?
**Verdict:** ⚠️ Genuinely ambiguous, not resolvable by us alone — this reads as a competition-specific judgment call. **Decision: build two code paths.**
  - Path A (safe default): romanization via a **rule-based, deterministic character-mapping table we write ourselves** (Devanagari/Gujarati Unicode block → ASCII phonetic approx) — a few hundred lines, no external package, defensible as "our own normalization code," not a lookup service.
  - Path B (only if time permits and you're comfortable with the risk): a bundled offline library (e.g. `indic-transliteration`) as a higher-quality upgrade.
  - Document explicitly in Documentation_template.md that transliteration is deterministic character-mapping, not a database/API lookup, to preempt fair-play scrutiny.
  - **Your call, flagged, not resolved by me:** do you want to just skip script-crossing matches entirely (accept the ~4-6% recall hit within that 15.7% bucket) to remove all ambiguity? I'd lean against — building the character-mapping table ourselves is low-risk and clearly "our own code," not a lookup — but you should decide given this is a disqualification-risk axis, not just a scoring one.

### 2.4 "Country as hard blocking filter"
**Attack:** The audit found 0/7.64M cross-country matches in *train*. Is it actually safe to assume this holds for France in test, which we've never seen a single labeled example of?
**Verdict:** The assumption is reasonable (a business physically located in France is very unlikely to be the same real-world entity as a US-labeled record) but is *extrapolated*, not verified. Mitigate by not hard-deleting cross-country candidates in an unrecoverable way — implement it as a very strong down-weighting feature *if using Splink/EM*, or if using hard filtering (which we are, for speed), at minimum log the assumption clearly in the methodology doc as a stated limitation, and if time permits, spot check the France test rows for any obvious multi-country business chains (e.g. a name literally containing "France" + "USA" or similar franchise tells) that might contradict it.
**Decision: keep hard filter (required for the speed budget — soft-weighting country in a full EM model over 10M+ rows is not worth the complexity here), but document the extrapolation risk explicitly.**

### 2.5 "F_0.5 threshold tuned on held-out validation split"
**Attack:** How is the validation split actually constructed? If you split *pairs* randomly, candidate pairs sharing an S1 entity leak across train/val (the model implicitly learns entity-level patterns it shouldn't get credit for). If you split by S1 *entity* instead, is the split large enough to give a stable F_0.5 estimate, given macro-averaging is sensitive to small-N per-entity noise?
**Verdict:** Must split by **S1 entity ID**, never by pair — this was stated but needs to be enforced in code, not just intention. Use a reasonably large held-out slice (15-20% of S1 entities) given the dataset is large enough (2.2M) that this isn't a small-data problem — stability isn't actually a real risk here given volume, that concern is moot at this scale. Also worth doing a second train/val split (different random seed) as a sanity check that the chosen threshold isn't an artifact of one particular split.

### 2.6 "LightGBM ≤8B params" — is this even a real constraint check?
**Attack:** A LightGBM model with any reasonable number of trees/leaves is trivially far under 8B params (it's not a parametric neural net in the same sense — "params" for a GBM usually means total leaf values across trees, typically in the thousands-to-millions range even for large models). Is there a risk the grading rubric interprets this differently, or that they expect an actual ~1-8B-param *language* model as "the intended solution shape"?
**Verdict:** ⚠️ Genuinely can't resolve without more info. Two readings: (a) the constraint exists to *cap* teams tempted to use huge hosted LLMs, in which case GBM is obviously fine and actively the safer choice; (b) the constraint implies the intended solution involves a real (smaller) language/embedding model and a classical GBM might read as "not attempting the intended approach" even though it's compliant. **Flag this explicitly in your methodology doc as a stated interpretation** ("we interpret the 8B constraint as an upper bound, not a target; our approach uses classical ML for interpretability, speed, and reproducibility at this data scale") so graders see it was a conscious choice, not an oversight.

### 2.7 Singleton handling
**Attack:** 123,247/2,206,821 = ~5.6% of S1 entities in train are true singletons. If threshold-tuning is done purely by sweeping a global probability cutoff, is there a risk the optimal *global* threshold under-serves singletons specifically (since they're a minority class relative to matched entities, and F_0.5 macro-averages per-entity — meaning singleton correctness is worth exactly as much as a 10-match entity's correctness, disproportionate to its representation in a naive pair-level training objective)?
**Verdict:** Real risk. A pair-level classifier trained on positive/negative pairs doesn't "see" entities, it sees pairs — so it has no direct signal about calibrating for the entity-level singleton case beyond "this pair has low match probability." **Decision: after threshold selection, explicitly compute F_0.5 broken out by true match-count bucket (0, 1, 2-3, 4+) on validation, not just the overall macro-average** — if singleton-bucket F_0.5 is materially worse than others, that's the signal to tune blocking (are singletons getting spurious candidates through blocking that a global threshold can't cleanly reject?) rather than just re-sweeping the global threshold.

### 2.8 Reproducibility / grading risk
**Attack:** "Top teams' packages are reviewed in detail" — if the pipeline takes hours to run end-to-end on the full 10M+ row files, does that create risk at grading time (reviewer environment differences, timeouts, etc.)?
**Verdict:** Build with a `--sample` / `--limit` flag from day one so the reviewer (or you, iterating) can run the full pipeline on a small slice quickly to sanity-check reproducibility, with the full run being a separate documented (possibly long) step. Pin all dependency versions in requirements.txt exactly (not `>=`), since "anyone should be able to regenerate both output files" is an explicit requirement, and version drift is the single most common reason someone else's "reproduction" silently produces different numbers.

---

## 3. Locked decisions

| Decision | Choice | Why |
|---|---|---|
| Blocking engine | Hand-rolled multi-key inverted index (pure Python/pandas/dict-based), Splink deferred to stretch-goal only | Zero install risk, fully debuggable, meets the actual scale requirement without a heavy dependency |
| Blocking keys | (1) country hard filter, (2) normalized-name token/prefix index, (3) address-token index (street-number + first N tokens), (4) transliterated-name index | Multi-key needed — no single key clears 95% recall per data audit |
| Transliteration | Deterministic Unicode character-mapping table, hand-written (Path A). Library upgrade (Path B) only if time allows and risk is acceptable | Avoids fair-play ambiguity around bundled external linguistic resources |
| Matching model | LightGBM binary classifier on hand-engineered pairwise features | Fast, interpretable, trivially compliant with param/license constraint, appropriate for this data scale vs. deep models |
| Validation split | By S1 entity ID (never by pair), 15-20% held out, two seeds as sanity check | Prevents leakage; macro-F_0.5 is entity-level so split must respect that |
| Threshold selection | Sweep against macro F_0.5 on held-out split; break out by match-count bucket including singletons specifically | Global-only tuning risks under-serving the singleton bucket, which is scored equally per-entity |
| Country filter | Hard filter (not soft weight), documented as an extrapolated assumption for France | Speed requirement at this row count; audit only verified US/India |
| 8B param constraint | Interpreted as ceiling not target; documented explicitly in methodology doc | Avoids the constraint being misread as "not attempted" |
| Reproducibility | `--sample` flag for fast dev iteration + full pinned-version run for final; pinned requirements.txt | Grading explicitly involves reproduction; version drift is the #1 failure mode |

---

## 4. Workflow (phase-gated, in build order)

```
Phase 0 — Environment & scaffolding
  └─ repo structure per submission package spec, requirements.txt started, --sample flag wired early
  └─ GATE: pipeline skeleton runs end-to-end on a tiny synthetic slice (even with dummy logic) before real code goes in

Phase 1 — Normalization
  └─ name: lowercase, whitespace/punct collapse, legal-suffix strip (keep both stripped+unstripped)
  └─ address: loose tokenization, no order/component assumptions
  └─ transliteration: hand-written char-mapping table (Path A)
  └─ GATE: run against ~30 real true-match pairs pulled from train, confirm similarity visibly improves post-normalization

Phase 2 — Blocking / candidate generation
  └─ country hard filter → union(name-index, address-index, translit-index) candidates
  └─ GATE (hard stop, do not proceed without this number):
       measure recall (% of true matches present in candidate set) and reduction ratio
       on held-out validation split (S1-entity-level split, not pair-level)
       target ≥95% recall; do not proceed below ~90% without adding blocking keys

Phase 3 — Feature engineering
  └─ name Jaccard/Levenshtein (raw + normalized + suffix-stripped)
  └─ address Jaccard/Levenshtein, token overlap, street-number match flag
  └─ missing-address flag, length deltas
  └─ GATE: sanity-check feature distributions separate true-match vs. non-match pairs
       (spot check against the 1.3M identical-name-but-not-match adversarial set specifically —
       confirm address features actually separate these, since name features alone won't)

Phase 4 — Model training + threshold calibration
  └─ LightGBM on Phase 3 features, S1-entity-level train/val split
  └─ implement F_0.5 macro-average eval function (standalone, reusable) — not provided by organizers
  └─ sweep thresholds against macro F_0.5, not F1, not accuracy
  └─ break out F_0.5 by match-count bucket (0/1/2-3/4+), specifically check singleton bucket
  └─ GATE: report validation macro F_0.5, plus per-bucket breakdown, before generating final outputs

Phase 5 — Inference + output generation
  └─ run full pipeline on test data (S1 test entities, France included, no special-casing)
  └─ write candidate_pairs.tsv (blocking stage output) and matching_results.tsv (post-threshold)
  └─ programmatic check: every matched ID ⊆ that entity's candidate list
  └─ every test S1 entity has exactly one row in both files, no dupes
  └─ GATE: run utils/validate_submission.py — must print PASS before considering this done

Phase 6 — Packaging
  └─ code/business_entity_resolution/src/ organized by phase (normalize.py, blocking.py, features.py,
       train.py, infer.py, evaluate.py)
  └─ README.md with exact reproduction commands (including --sample fast-path and full-run path)
  └─ requirements.txt pinned exact versions
  └─ Documentation_template.md filled in with: methodology, blocking recall/reduction numbers from
       Phase 2, model architecture + features, threshold rationale, explicit notes on the two ⚠️ items
       above (transliteration approach, 8B-param interpretation)
  └─ zip per exact required structure
```

---

## 5. Open items requiring your decision (⚠️ flagged above, collected here)

1. **Transliteration approach** — hand-written char-mapping (safe, lower quality) vs. bundled library (higher quality, marginal fair-play ambiguity) vs. skip script-crossing entirely (safest, costs recall). Leaning toward hand-written char-mapping as the default; confirm or override.
2. **8B-param constraint interpretation** — proceeding with GBM under the "ceiling not target" reading, documenting the interpretation explicitly rather than guessing what graders intended. Flag if you have any signal (from the organizers, past years, etc.) that suggests otherwise.
3. **France cross-country assumption** — hard-filtering test candidates by country including the unseen France label, documented as extrapolated. No action needed unless you want a fallback (e.g. a small secondary no-country-filter blocking pass just for France rows, at extra compute cost, as a safety net) — flag if you want that built.

---

## 6. Research sources referenced in these decisions

- Blocking/filtering survey (ACM Computing Surveys) — confirms multi-key blocking as standard practice for trading comparison volume vs. recall at scale.
- BlockingPy / Statistics Poland case study — real-world precedent for the exact script-transliteration problem (Ukrainian/Russian↔Polish), solved via transliteration + ANN blocking, not hand rules alone — validates that this is a known, tractable problem class, not a novel one.
- Splink documentation — confirms viability at scale (millions of records in ~1 min via DuckDB) but also confirms it's a real dependency with its own comparison-function API, which is why it was downgraded to stretch-goal rather than critical path here.
- Deep-learning-for-entity-matching design-space literature — confirms embeddings/transformers are an active research area, not a settled default, supporting the decision to stay with classical ML given time constraints and the param/license clause.
