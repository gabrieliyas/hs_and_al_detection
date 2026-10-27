# Experiment Execution Protocol

Companion to `dataset_split_and_weighting.md`. Records execution-level
decisions for Experiments I, II, and III: seed counts, backbone fairness
and tuning, significance testing, compute budgeting, the pre-declared
backbone decision rule, and the C0-C4 ablation analysis design. Reference
for implementation and for the methodology chapter.

Revision note: Experiment I now compares two Transformer backbones
(IndoBERTweet and XLM-RoBERTa-base). The SVM+TF-IDF baseline is retired
(SVM studies are kept only as literature justification), so the former
Section 4 (SVM class weighting) is removed. XLM-RoBERTa-base is also
trained on C0-C4 and therefore belongs to **Experiment II** (backbone is a
factor in Experiment II-A); it is not a separate "Experiment I-B". All
functions named below live in `experiment_utils.py`. The active contract is
`schema_version: "thesis-run-config-v3"`, `run_id: "left_truncation_v2"`,
`CONTEXT_RULE_VERSION = "paired-oof-depth-tradeoff-v2"`, and
`BACKBONE_RULE_VERSION = "v3-paired-context-and-latency-gate"`. Artifacts
created with older schema or decision-rule versions are historical only and
are rejected by the active pipeline.

Statements are tagged where it matters: **Decision** = fixed by this
protocol; **Assumption** = a judgement call to be disclosed in the thesis,
not an established fact.

---

## 1. Experiment I: backbone comparison on the benchmark

- **Models:** `E1-IndoBERTweet-Target` and `E1-XLMR-Target`, target post
  only, trained and evaluated on the Ibrohim & Budi dataset (single fixed
  70/15/15 stratified split; weighted cross-entropy with the weights in
  `dataset_split_and_weighting.md` Section 1.4, computed from the training
  partition only).
- **Why XLM-R as the comparator:** a second Transformer pre-trained on a
  broader multilingual corpus, so the comparison asks whether an
  Indonesian-Twitter-specific backbone beats a general multilingual one on
  this task. Published comparisons are mixed, which is the reason to test it
  here rather than assume the outcome.
- **Seeds:** the same fixed Experiment I seeds `(42, 123, 2024, 7, 99)` for
  both backbones (varying classification-head initialisation and data shuffling).
  Experiment II-A and III use the shared CV seeds `(42, 123)`. 2 seeds was
  considered and rejected -- a standard deviation from 2 points carries
  almost no information; 3 is the practical floor, 5 is affordable.
- **Reporting (per backbone):** a per-seed table (accuracy, macro-F1,
  weighted-F1, and F1 for N, AL, HS) via `seed_metrics_table`, and the mean
  +/- std over seeds via `summarize_seed_metrics`. This is the training-
  stability statistic.
- **Head-to-head test:** seed-ensembled predictions
  (`ensemble_seed_probabilities`: average the softmax over the 5 seeds, then
  argmax) on the single test partition, then a paired bootstrap (10,000
  resamples, 95% CI) of the difference XLM-R minus IndoBERTweet in macro-F1
  and in each class's F1. `compare_backbones_exp1` produces all of this in
  one call; its macro-F1 interval is identical to `paired_bootstrap_ci`
  with the same `random_state` (verified in testing), and the per-class
  intervals come from the same resamples.
- **Reading the per-class intervals:** macro-F1 is the single pre-declared
  test. The per-class intervals are descriptive -- they show where a
  difference arises, and are not corrected for multiple comparisons.
- **Two statistics, two questions:** per-seed mean +/- std (stability) and
  the ensembled paired bootstrap (head-to-head) use different aggregations
  by design; report both.

---

## 2. Backbone parity and hyperparameter tuning

**Principle (Decision).** Only the model/tokenizer and the learning rate may
differ between backbones. The learning rate is chosen per backbone from the
same search space with the same tuning budget; it is neither copied from
IndoBERTweet's existing configuration (which would favour IndoBERTweet, since
XLM-R is reported to be more sensitive to the learning rate) nor tuned more
extensively for one backbone than the other.

### 2.1 What is identical and what may differ

| Identical for both backbones | May differ |
|---|---|
| Data, splits, `fold` column, labels | Pre-trained model and its tokenizer |
| Seeds (Exp I: 5 values; CV: 2 values) | Learning rate (from the shared grid) |
| ITFT procedure and checkpoint selection (validation macro-F1) | Per-device batch size (see below) |
| `max_length` (one value per stage), truncation side | Gradient-accumulation steps (see below) |
| Optimizer, weight decay, warm-up ratio, LR schedule | |
| Max epochs and early-stopping rule | |
| Effective batch size, loss, class weighting, mixed precision | |
| LR search grid, tuning budget, tuning seed | |

- **Effective batch size** is identical; if XLM-R needs a smaller per-device
  batch to fit in GPU memory, compensate with gradient accumulation.
- **Enforcement:** each run records its resolved configuration, and
  `check_config_parity` is called before Stage 5 and Stage 11 results are
  reported. It raises on any mismatch in the identical column, on a
  different effective batch size, or on a learning rate outside the grid.
- Differences that cannot be controlled (casing of the pre-trained vocabulary,
  tokenizer segmentation, pre-training corpus) are properties of the
  backbones themselves and are the object of the comparison.

### 2.2 Shared maximum sequence length (one value per stage)

- `max_length` is **one number per stage, shared by both backbones**, never
  chosen per tokenizer. `choose_shared_max_length` takes the token lengths
  under **both** tokenizers, takes the larger of the two 95th-percentile
  lengths, rounds up to a multiple of 8, caps at 512, and reports the
  truncation rate of each tokenizer at that value.
- **Benchmark stage** (`max_length_benchmark`): decided after Stage 1 by
  `1b_select_benchmark_max_length.py`, using the benchmark **training
  partition only** and both backbone tokenizers. Freeze the resulting value
  before benchmark learning-rate search, Experiment I, or ITFT. The helper
  uses `choose_shared_max_length(..., partition="benchmark_train")` and saves
  through `save_length_selection_config()` to
  `results/config/sequence_length.json`. The saved config must include
  `benchmark_length_source_partition: "benchmark_train"`,
  `benchmark_length_percentile: 95`, `benchmark_length_percentile_tokens`
  (one p95 value per backbone), and
  `benchmark_truncation_rate_at_max_length` (one rate per backbone).
  It must identify `schema_version: "thesis-run-config-v3"` and
  `run_id: "left_truncation_v2"`, declare
  `truncation_side: "left"` for contextual inputs and
  `target_only_truncation_side: "right"` for target-only benchmark inputs.
  Each resolved run config records the actual side used by that stage. Configs
  from an older schema are rejected rather than reused.
- **Contextual stage** (`max_length_context`): decided in Stage 8 from C4
  inputs using both backbone tokenizers, then used for every C0-C4 run, both
  backbones, all contextual inference, and final-model training. Stage 8 uses
  `choose_shared_max_length(..., partition="contextual_c4")` and merges its
  result through `save_length_selection_config()`. The final shared config must
  also include `context_length_source_partition: "contextual_c4"`,
  `context_length_percentile: 95`, `context_length_percentile_tokens` (one p95
  value per backbone), and `context_truncation_rate_at_max_length` (one rate
  per backbone). The shared-config loader verifies the two source-partition
  declarations, recomputes the rounded shared lengths from the saved p95s, and
  rejects missing, inconsistent, or legacy values.
- **Truncation side (Decision).** Context-first inputs put the target post
  last, so default right-side truncation would cut the target. Truncate from
  the left (the oldest parent first), identically for both backbones, and
  confirm this in the training script (`truncation_side` is a parity key).
- **Separator:** build inputs with `sep_from_tokenizer(tokenizer)` so each
  backbone uses its own separator token (`[SEP]` vs `</s>`).

### 2.3 Learning-rate search (equal budget)

- **Grid (Assumption):** `LR_SEARCH_GRID = (1e-5, 2e-5, 3e-5, 5e-5)`, the
  same for both backbones and both stages. It must contain the learning rate
  already used by the existing IndoBERTweet runs; if it does not, add that
  value to the grid for both backbones.
- **Stage A -- benchmark (before Stage 3/4):** for each backbone, one run
  per grid value on the benchmark training partition, with a fixed tuning
  seed that is not one of the Experiment I or contextual CV evaluation seeds.
  The configured `lr_tuning_seed` is checked against both frozen evaluation
  seed sets. The learning rate with
  the highest validation macro-F1 (`select_learning_rate`; ties go to the
  smaller rate) becomes `learning_rate_benchmark`, used for Experiment I
  **and** ITFT. Budget: 4 runs per backbone.
- **Stage B -- contextual (in the Stage 8 pilot):** for each backbone, one
  run per grid value on C4 starting from that backbone's ITFT checkpoint,
  trained on 80% of fold 0's **training portion** and validated on the
  remaining 20%. The fold-0 test portion is never touched. The selected
  rate becomes `learning_rate_context` and is used for **all** Experiment II
  **and** Experiment III runs of that backbone, so ITFT vs. direct
  fine-tuning differs only by the ITFT stage. Budget: 4 runs per backbone.
- **Equal budget is enforced in code:** `select_learning_rate` raises unless a
  validation score exists for every grid value. `results/config/learning_rates.json`
  must save, for each backbone, `benchmark_trial_macro_f1` and after Stage 8
  `context_trial_macro_f1` as mappings from every candidate LR in the frozen
  grid to its validation macro-F1. The config loader recalculates the selected
  rate from those scores (ties go to the smaller LR) and rejects a stored rate
  that does not match, a missing trial, or a tuning seed reused by the evaluation
  folds. Do not copy selected rates from historical runs into the v3 config.
- **Fixed rather than tuned:** all other hyperparameters (Section 2.1)
  stay at the existing IndoBERTweet configuration. **Assumption:** that
  configuration is reasonable for XLM-R once the learning rate is tuned; this
  is disclosed as a limitation (Section 8).
- **Existing IndoBERTweet runs.** The historical contextual runs also used
  implicit tokenizer defaults for truncation; see the audit note in Section
  8. Therefore, the old Experiment II-A and Experiment III results are not
  eligible for the finalized analysis even if their learning rates match.
  After the truncation and sequence-length settings are fixed, rerun them with
  the selected contextual rate. If the selected benchmark rate differs from
  the historical benchmark/ITFT rate, rerun the affected Experiment I and
  ITFT stages as well.

---

## 3. Experiments II and III

- **Backbone as a factor.** Experiment II-A runs the full C0-C4 grid for
  **both** backbones (`E2-IndoBERTweet-ITFT-C{n}` and `E2-XLMR-ITFT-C{n}`).
  Stage 11 resolves a context size for each backbone using paired out-of-fold
  bootstrap comparisons and applies the pre-declared backbone gate. From
  Stage 12 onward, retraining, II-B, Experiment III, error analysis, and final
  models use the selected backbone only.
- **ITFT per backbone:** one ITFT run per backbone (Stages 6-7) using
  `learning_rate_benchmark`. Optional saving: one Experiment I seed run
  may be designated as the ITFT checkpoint, since the procedure is identical
  (train on benchmark train, select by validation); this is legitimate
  because selection used the validation split only. Separate runs are
  cleaner to explain.
- **Timing pilot (Stage 8, per backbone):** run exactly one (condition,
  fold) pair -- C4, fold 0 -- and time (a) the one-time ITFT stage and
  (b) a single second-stage fine-tune on that fold's ~720-example training
  portion. Total budget estimate = ITFT time (paid once per backbone) +
  (single second-stage run time x planned runs). XLM-R-base is the larger
  model (its vocabulary is far larger), so its memory use and run time are
  measured, not assumed, and the Colab feasibility check below uses the
  measured values.
- **`max_length` for this stage** is decided in the same pilot (Section 2.2),
  and the learning-rate search of Section 2.3 (Stage B) runs here too.
- **Colab session limits:** check the total timing estimate against Colab's
  session runtime limits. If the full sweep exceeds one session, plan for
  checkpoint/resume logic across sessions rather than discovering this
  mid-sweep.
- **Fold assignment sharing:** the same `fold` column (from
  `ctx_matched_subset.csv`) is reused identically across Experiment II
  (ITFT+C), Experiment III (C-only), both backbones, and every C0-C4
  configuration. See `dataset_split_and_weighting.md` Section 2.3-2.4.
- **C0-C4 construction order:** fold assignment happens first, once, on
  the 900 base examples. The C0-C4 truncated views are generated afterward,
  at data-loading/tokenization time, from each example's already-assigned
  fold -- never the reverse.
- **ITFT checkpoint selection:** use the Ibrohim & Budi validation split
  (`load_best_model_at_end=True`, `metric_for_best_model` = validation
  macro-F1) -- do not fix ITFT training parameters in advance and skip
  validation. This is ordinary validation-based model selection, not a
  leakage risk: the leakage principle (Section 4.1) guards the test set and
  the shallow-pool generalization data, not routine use of a validation
  split. It carries outsized importance because ITFT runs once per backbone
  and its checkpoint is the shared starting point for all 50 Experiment II
  second-stage runs of that backbone -- the highest-leverage artifact in the
  pipeline.

---

## 4. C0-C4 ablation: main analysis vs. secondary analysis

Two candidate designs were considered:

- **Nested availability-aware cohorts:** compare C0 vs. Cn using the
  maximal subset with >= n parents available (C0-vs-C1 on 1,800; C0-vs-C2
  on 1,650; C0-vs-C3 on 1,350; C0-vs-C4 on 900).
- **Complete-case (matched-subset) analysis:** compare all of C0 through
  C4 on the identical 900 examples with exactly 4 parents available
  (the existing matched subset, shared fold assignments).

**Decision: complete-case analysis is the MAIN analysis.** The nested
design changes which population is evaluated at every comparison point,
which confounds "effect of context depth" with "which examples happen to
have that much depth available" -- e.g. the >=3 subset systematically
excludes shallow-thread posts that the >=2 subset includes. This is the
same confound the shared-fold-assignment principle already exists to
prevent. A genuine dose-response claim ("does performance plateau or
decline past a certain depth") requires holding the underlying examples
constant across all compared levels, which only the complete-case design
does.

**The nested availability-aware analysis is a SECONDARY analysis**, kept
for a different, complementary purpose: it reflects realistic deployment
conditions (most real conversations will not have 4 parents available,
which matters for the browser-extension demo's generalizability claim).
Each depth-specific comparison must be presented as an independent,
standalone finding on its own population -- NOT plotted as points on one
continuous curve, since that would silently reintroduce the population
confound.

**Practical consequence:** the headline F1-vs-context-window-size plot
(contribution: optimal context size / diminishing returns) is built from
the complete-case analysis only, one curve per backbone. Small gaps between
neighbouring context sizes (for example C3 vs. C4) must be checked with the
paired bootstrap on pooled out-of-fold predictions before a context size is
described as optimal.

### 4.1 Finalized design: availability-aware generalization analysis

(Experiment II-B; selected backbone only.)

- **Populations:** the "shallow pool" is the 900 posts with
  `parent_count_available` in {1, 2, 3} -- disjoint by construction from
  the 900-example matched subset (`parent_count_available == 4`). Per-model
  testable population:
  - C0 model -> all 900 shallow-pool posts
  - C1 model -> all 900 shallow-pool posts (identical to C0's population,
    since every post has >= 1 parent available)
  - C2 model -> shallow-pool posts with >= 2 parents available (750)
  - C3 model -> shallow-pool posts with >= 3 parents available (450)
  - C4 model -> no external population (the shallow pool is defined as
    everything outside the matched subset)
- **Model used (adopted):** after C_best is selected from 5-fold CV results
  alone (never from shallow-pool performance), retrain each of C0-C4 once
  on the FULL 900-example matched subset (Stage 12) and use these models --
  not fold-specific CV models -- for the shallow-pool evaluation (Stage 13).
  This cleanly separates model selection (CV) from final model production
  and avoids the ambiguity of choosing among 5 fold-specific models. Cost:
  5 additional training runs beyond the CV grids.
- **Model identity constraint:** the model used here must be trained on the
  900-example matched subset ONLY, never on the full 1,800-post dataset. A
  deployment model trained on all 1,800 posts (shallow pool included) is a
  DIFFERENT model and cannot be reused for this check.
- **Reporting:** descriptive only, per model, on its own population. Never
  plotted as an additional point on the main C0-C4 curve, and never used to
  override the CV-based choice of C_best.

---

## 5. Context-size and backbone decision rules (Stage 11)

**Purpose.** Stage 11 resolves a context size for each backbone and selects
one backbone to carry into Stages 12-17. These rules must be finalized in
`experiment_utils.py` and committed before the corrected Experiment II-A
runs start. `select_best_context_from_cv` must not use the highest mean alone
as the final definition of an optimal context size.

### 5.1 Resolve each backbone's context size

1. Build seed-ensembled pooled out-of-fold probabilities for each C0-C4
   condition from the shared 900-example matched subset.
2. Identify the condition with the highest mean pooled out-of-fold macro-F1
   as the provisional top-scoring condition.
3. Calculate paired bootstrap 95% confidence intervals (10,000 resamples)
   for the macro-F1 difference between that condition and each shallower
   condition, using the same post IDs and the same resampled indices. Pairwise
   confidence intervals must be based on the differences, not on overlap
   between independent confidence intervals.
4. **Parsimony/deployment trade-off rule (Decision):** for a provisional top
   context C>=2, check the two-level-shallower candidate C-2 first, then the
   immediately shallower candidate C-1. Choose C-2 if the paired 95% CI for
   `F1(top) - F1(C-2)` includes zero; otherwise choose C-1 if the CI for
   `F1(top) - F1(C-1)` includes zero; otherwise retain the top-scoring
   condition. For a top condition C1, check C0; for C0, retain C0. The C0/C1
   intervals are still reported when the top is C3/C4, but do not override
   this pre-declared practical comparison. This prioritizes a meaningful
   reduction in context depth and deployment cost rather than claiming that
   the minimum possible context is statistically equivalent. An interval
   containing zero does not establish equivalence.
5. For the current IndoBERTweet pattern where C4 has the highest observed
   mean, explicitly save the paired C4-C2 and C4-C3 intervals. If C4-C2
   includes zero, select C2 and describe gains as not resolved beyond C2;
   otherwise, if C4-C3 includes zero, select C3; otherwise retain C4. Save
   both intervals regardless of which candidate is selected. Apply the same
   rule to XLM-R and preserve all remaining pairwise comparisons for reporting.

The C0-C4 selection comparisons, selected context size, rule version, mean
scores, bootstrap intervals, and reason must be written into
`results/backbone_decision.json`. Apply the same rule to both backbones.

### 5.2 Select the backbone

**Inputs.** For each backbone, use the per-backbone selected context from
Section 5.1 and C0. Build seed-ensembled pooled out-of-fold predictions on
the same 900 matched examples. Use `pool_out_of_fold_probabilities` and
`ensemble_seed_probabilities` to preserve post-level pairing.

1. **Primary criterion:** paired bootstrap (10,000 resamples, 95% CI) of the
   macro-F1 difference XLM-R minus IndoBERTweet at each backbone's own
   selected context. If the interval excludes zero, the higher backbone is
   the leader.
2. **C0 guard:** the leader must not be credibly worse in target-only mode.
   If the C0 difference interval lies entirely on the losing side, treat the
   comparison as having no clear winner.
3. **Tie-break:** when there is no clear winner, select the backbone with the
   lower measured median single-post inference latency on the same intended
   backend machine, but only if the relative difference exceeds
   `LATENCY_TOLERANCE` (10%, an **Assumption**). Otherwise default to
   **IndoBERTweet**. Valid, positive latency measurements for both backbones
   are required; the script must not silently default if latency is missing.
4. **Experiment I is not a decision input.** Its result is target-only on a
   different dataset. Record whether it agrees with the final selection only
   for discussion (`concordant_with_experiment1`).

**Output.** `11_select_backbone.py` first validates exactly 50 Stage 9/10
run manifests per backbone (five contexts x five folds x two seeds), each
through `assert_current_run_artifact` with an explicit expected stage, script,
and run fingerprint. `audit_context_cv_grid` must verify current schema/run ID,
all expected context/fold/seed keys, dataset and fold hashes, shared contextual
sequence length, explicit left truncation, and unique configuration fingerprints.
Both backbones must use the same dataset hash, fold-assignment hash, contextual
length and truncation policy. Only then may it select contexts and the backbone.
The resulting `results/backbone_decision.json` records both per-backbone context
choices, every top-versus-shallower paired interval (including C4-C2 and C4-C3
when C4 is top), source-grid SHA-256 fingerprints and run counts, the backbone
and C0 guard intervals, latency values/machine, rule versions, and decision
reason. `select_backbone` also verifies that C_best and C0 predictions from
both backbones have the same ordered 900 post IDs as the context-selection
inputs. Every stage from 12 onward that depends on the selection must load this
JSON and fail clearly if it is absent, stale, inconsistent or invalid.

**Interpretation limits.** An interval containing zero means the data did not
resolve a difference at the chosen confidence level; it is not proof of equal
performance. The backbone rule chooses a model for this deployment setting,
not a universally superior architecture.

---

## 6. Final deployment models

- **Backbone:** the one selected at Stage 11. The non-selected backbone is
  not used for Stages 12-17.
- **Two models, one per mode, both trained on all 1,800 contextual posts:**
  (1) target-only C0 and (2) a context-aware model using the concrete context
  size selected at Stage 11. The final context-aware model uses up to that
  many available parents per example (fewer when a thread is shallower); target-only
  uses C0 (the target post only). The
  900-example complete-case dataset remains the basis for the main C0-C4
  comparison. These final models are not substitutes for the 900-example
  models used in Stage 13.
- **Training strategy, chosen per mode:** Stage 15 compares ITFT (Experiment
  II) with direct fine-tuning (Experiment III) on pooled out-of-fold,
  seed-ensembled predictions over the same folds, once for C0 and once for
  the selected context. ITFT is chosen only if it is credibly better; direct
  fine-tuning is chosen if it is credibly better; otherwise the parsimonious
  default is direct fine-tuning (`DEFAULT_STRATEGY_ON_TIE`, an **Assumption**).
  The two modes may select different strategies.
- **Stage 15 quantitative reporting outputs:** the source-of-truth report
  bundle and provenance manifest live under
  `results/runs/left_truncation_v2/stage_15/`. Thesis-ready CSV tables are
  exported to `results/tables/left_truncation_v2/`; figures (300 dpi PNG) to
  `results/figures/left_truncation_v2/`. Main Chapter IV tables are compact;
  per-seed, per-class, per-fold/seed, and long-form confusion-matrix data are
  separate supplementary tables. The complete confusion-matrix collection
  is machine-readable, while selected matrices are rendered as separate
  publication-ready figures. Micro precision/recall/F1, Hamming loss, and
  exact-match accuracy are omitted from main tables because they are redundant
  with accuracy for this single-label three-class task. Stage 15 reports but
  does not reselect the Stage 11 backbone or context sizes.
- Experiment I models are not reused as the final target-only model; they
  serve the benchmark comparison only.
- Save two distinct model directories (not a single `final.pt`):
  `models/deployment/<backbone>/target_only/` and
  `models/deployment/<backbone>/context_aware_C<n>/`, with a manifest recording
  selected backbone, strategy per mode, context size, tokenizer, truncation
  side, sequence length, and training configuration.

---

## 7. End-to-end pipeline order and script map

The numerical stages below are the authoritative order. The SVM file
`3_train_e1.svm` is retained only as an optional legacy comparison and is not
run in the main Transformer pipeline. `experiment_utils.py` is a shared
module, not a standalone stage; edit/test it before the experiment sweep but
do not run it by itself.

| Stage | Step | Script(s) | Environment | Main output / dependency |
|---|---|---|---|---|
| 1 | Preprocess Ibrohim & Budi benchmark | `1_preprocess_benchmark.py` | VS Code | Cleaned data, fixed stratified train/validation/test split, train-only class weights |
| 1b | Choose shared benchmark maximum length | `1b_select_benchmark_max_length.py` **(new)** | VS Code | `results/config/sequence_length.json`; uses training partition and both tokenizers only |
| 2 | Preprocess contextual data and assign shared folds | `2_preprocess_contextual.py` **(required)** | VS Code | Matched 900-row subset with fold column, disjoint shallow pool, stable post IDs |
| preflight | Benchmark learning-rate search (before training) | `preflight_tune_benchmark_lr.py` **(new)** | Colab | `learning_rate_benchmark` for each backbone; 4 trials/backbone |
| 3 | Experiment I: IndoBERTweet target-only, five seeds | `3_train_e1_indobertweet.py` (rename from `4_train_e1_indobertweet.py`) | Colab | Per-seed predictions, probabilities, and metrics; no per-seed checkpoints required |
| 4 | Experiment I: XLM-RoBERTa-base target-only, five seeds | `4_train_e1_xlmr.py` **(new)** | Colab | Per-seed predictions, probabilities, and metrics |
| 5 | Evaluate Experiment I | `5_evaluate_e1.py` (retain/complete) | VS Code | Per-seed mean ± SD, accuracy/macro-F1/weighted-F1/per-class F1, paired bootstrap intervals |
| 6-7 | ITFT, one validation-selected checkpoint per backbone | `6_train_itft.py` (rename from `6_itft.py`; invoke once per backbone) | Colab | `models/itft_checkpoint/indobertweet/` and `models/itft_checkpoint/xlmr/` |
| 8 | Timing pilot, contextual length selection, contextual LR search | `8_timing_pilot.py` (rename from `7_timing_pilot.py`) | Colab | Shared `max_length_context`, per-backbone contextual LR, latency/runtime, truncation rates |
| 9-10 | Experiment II-A: C0-C4, 5-fold CV × 2 seeds per backbone | `9_train_e2_ablation.py` (rename from `8_train_e2_ablation.py`; invoke once per backbone) | Colab | OOF predictions/probabilities and run manifest for 50 runs/backbone |
| 11 | Select C_best per backbone, then apply backbone decision gate | `11_select_backbone.py` **(new)** | VS Code | `results/backbone_decision.json`, including context-size and backbone CIs, latency, reason |
| 12 | Retrain selected-backbone C0-C4 on the full matched 900 | `12_train_e2_retrain_full.py` (rename from `9_train_e2_retrain_full.py`) | Colab | Five model directories, `e2_full_retrain_metrics.csv`; reads Stage 11 decision |
| 13 | Experiment II-B: availability-aware shallow-pool inference | `13_e2b_generalization.py` (rename from `10_e2b_generalization.py`) | Colab | Descriptive per-model/per-class results; reads Stage 11 decision and Stage 12 models |
| 14 | Experiment III: direct fine-tuning without ITFT, C0 and selected context | `14_train_e3_direct.py` (rename from `11_train_e3_context_only.py`) | Colab | 20 runs, OOF predictions/probabilities; reads Stage 11 context choice |
| 15 | Cross-experiment quantitative evaluation and strategy selection per deployment mode | `scripts/15_evaluate_all.py` | VS Code | Authoritative bundle under `results/runs/left_truncation_v2/stage_15/`; thesis-facing compact/detailed tables under `results/tables/left_truncation_v2/`; publication-ready figures under `results/figures/left_truncation_v2/`; canonical `results/training_strategy_decision.json` |
| 16 | Qualitative error analysis | `16_error_analysis.py` (rename from `13_error_analysis.py`) | VS Code | Annotated disagreements and context-type breakdown |
| 17 | Train final two comparison/deployment models on all 1,800 posts | `17_train_final_models.py` (renamed from `14_train_final_models.py`) | Colab | Separate target-only and context-aware model directories plus manifest; reads Stages 11 and 15 decisions |
| 18 | Extension backend | `extension/backend/` **(required)** | VS Code | Inference endpoints loading both final models |
| 19 | Extension frontend | `extension/frontend/` **(required)** | VS Code | DOM extraction, context preparation, backend calls, UI rendering |
| 20 | End-to-end integration test and demo capture | `20_test_extension_e2e.py` **(new)** | Local browser/backend | Reproducible test results, screenshots/recording, demo evidence |

### Required execution notes

- Run preprocessing first and verify its row counts, label distributions, post IDs,
  and fold assignments. The benchmark split uses `random_state=42`; contextual
  folds must be assigned once and reused, not regenerated independently by each
  experiment. Archive a hash or manifest of the prepared datasets. Because the
  current CV script's resume guard skips runs when old prediction/probability
  files exist, archive the old contextual outputs and write corrected runs to a
  clean versioned results root (for example `results/runs/left_truncation_v2/`);
  do not allow historical right-truncated results to satisfy the new run checks.
  The v3 run manifest fingerprints the run config, input-data hashes, utility code,
  stage script code, and available protocol/pipeline documents. A prior manifest/schema
  or a changed code/config fingerprint must fail the resume check.
- `1b_select_benchmark_max_length.py` is separate from benchmark cleaning. It
  must tokenize the benchmark **training split only** with both backbone
  tokenizers, choose the shared length using the documented percentile rule,
  and write the configuration. Do not use validation or test text lengths to
  choose the limit.
- `preflight_tune_benchmark_lr.py` and `8_timing_pilot.py` are tuning/preflight steps.
  Save all trial scores and the selected value; do not silently carry forward a
  historical learning rate just because it is in the grid.
- A single parameterized ITFT script is preferable to duplicated code; invoke
  `6_train_itft.py --backbone indobertweet` for Stage 6 and
  `6_train_itft.py --backbone xlmr` for Stage 7. Apply the same pattern to the
  Experiment II-A script for Stages 9 and 10. Keep output paths backbone-specific.
- Explicitly set `tokenizer.truncation_side = "left"` before tokenizing every
  contextual training/evaluation input. Keep benchmark target-only tokenization
  explicitly set to `right` and identical across backbones. In
  `results/config/sequence_length.json`, store `truncation_side: "left"` for the
  contextual policy and `target_only_truncation_side: "right"` for the benchmark
  policy; each stage's resolved run config must record its actual `truncation_side`.
  Load the same saved `max_length` and truncation configuration in training,
  retraining, evaluation, and deployment; avoid duplicated hard-coded values.
- Before the clean rerun, call `audit_pipeline_script_names()` and require a
  complete result; the registry deliberately does not accept old filenames as
  aliases. Each training/inference script must call `validate_resolved_run_config()`
  before training and create/resume output only through `make_run_manifest()` plus
  `prepare_run_directory()`. Downstream stages must call
  `assert_current_run_artifact(path, expected_stage=..., expected_script=...,
  expected_fingerprint=...)` on consumed artifacts; all three expected values
  are required, not optional. The fingerprint must come from the producer's
  recorded run manifest/receipt. `validate_resolved_run_config()` also requires
  a valid Stage 11 decision for stages 12-17 and a Stage 15 training-strategy
  decision for Stage 17. Scripts formerly using flat-path `run_already_done()` checks
  must replace that logic; the shared utility cannot protect a script that bypasses
  its manifest/config checks. `assert_no_legacy_outputs()` is available to fail closed
  if a legacy output directory is still configured as a resume source.
- Stage 11 runs only after both complete Experiment II-A grids have valid OOF
  predictions/probabilities. Any stage from 12 onward that depends on the
  selected backbone/context must fail if `results/backbone_decision.json` is
  missing or invalid.
- The benchmark LR search and Experiment I are separate: if only the context
  truncation rule changes, target-only Experiment I need not be retrained solely
  for that reason. For a clean, auditable final run, however, replay the pipeline
  in chronological order after freezing the source/configuration, while verifying
  that preprocessing reproduces the same split and fold assignments.
- Do not put SVM into the numbered main run order. Keep `3_train_e1.svm` out of
  the main sequence and label its output as optional/legacy; do not include it in
  Stage 5's declared Transformer-versus-Transformer decision unless the study
  design is formally changed.

### Logical flow

```text
Stage 1 + 1b: benchmark clean/split -> max_length_benchmark
Stage 2:     contextual data -> fixed 900-row folds + shallow pool
Preflight:   benchmark LR search
Stages 3-5:  Experiment I (IndoBERTweet vs XLM-R, target-only)
Stages 6-7:  ITFT per backbone, selected by benchmark validation
Stage 8:     contextual timing + shared max_length_context + contextual LR search
Stages 9-10: Experiment II-A (C0-C4, both backbones, same folds)
Stage 11:    paired-bootstrap context selection + backbone decision JSON
Stage 12:    retrain C0-C4 on matched 900 for selected backbone
Stage 13:    shallow-pool descriptive evaluation
Stage 14:    Experiment III direct fine-tuning on same folds
Stage 15:    cross-experiment evaluation + strategy choice per mode
Stage 16:    qualitative OOF error analysis
Stage 17:    train target-only and context-aware deployment models on 1,800
Stages 18-20: backend + frontend + end-to-end test/demo
```

**Approximate run counts.** Benchmark LR search: 8 trials. Experiment I: 10
final seed runs (plus 8 LR trials). ITFT: 2. Contextual LR search: 8 trials.
Experiment II-A: 100 runs across both backbones. Stage 12 retraining: 5 runs
for the selected backbone. Experiment III: 20 runs for the selected backbone.
Final model training: 2 runs. These counts exclude evaluation, inference, and
latency-only measurements. Historical contextual runs should be treated as
superseded after the left-truncation correction, even where the selected
learning rate remains `2e-5`.

---

## 8. Threats to validity

- **Prior exposure to results.** IndoBERTweet's Experiment II and III
  results were seen before XLM-R was added, and Experiment I was extended
  after supervisor review. The grid, the decision rule, and the parity
  requirements are fixed before any XLM-R Experiment II run, but cannot be
  declared blind with respect to IndoBERTweet's numbers. Disclose both facts.
- **Selection optimism.** Context size and backbone are selected from the
  same cross-validation predictions used to compare them. Bootstrap intervals
  quantify paired test-sample uncertainty conditional on these predictions;
  they do not fully correct optimism caused by selecting among contexts and
  backbones. Describe the results as evidence within this evaluated grid.
- **Learning-rate search leakage.** Stage B tunes on 20% of fold 0's
  training portion, which contains examples that are test examples in other
  folds. The effect is limited to a 4-way choice of one scalar and is the
  same for both backbones.
- **Limited tuning.** Only the learning rate is searched, over a coarse
  grid; other hyperparameters stay at IndoBERTweet's existing configuration,
  which could still favour IndoBERTweet. This is an assumption, not a
  verified property.
- **Seed variance vs. data variance.** Experiment I uses one fixed split;
  the 5 seeds capture initialisation and shuffling variance, not variance
  from the choice of split.
- **Small sample.** With 900 matched examples, bootstrap intervals are wide;
  the gate may end in a tie-break, and per-class intervals are descriptive
  only.
- **Truncation.** The uploaded historical contextual training and inference
  scripts did not explicitly set `tokenizer.truncation_side = "left"`.
  Since `build_context_text` puts parents before the target and Hugging Face
  tokenizers default to right truncation unless configured otherwise, those
  runs used right truncation unless the runtime tokenizer had been modified
  externally. Treat those results as superseded. In corrected runs, report the
  truncation rate for each backbone and preserve target content by truncating
  older context from the left.
- **Latency measurement.** Latency is hardware- and load-dependent; measure
  both backbones on the same machine under the same conditions.
- **Scope of Experiment III.** Run only for the selected backbone, so the
  ITFT-vs-direct comparison is not available for the other backbone.
