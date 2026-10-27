# Stage 16 — Prediction-driven OOF error analysis

## Purpose

`scripts/16_error_analysis.py` analyses the selected backbone's Experiment II-A / ITFT out-of-fold (OOF) predictions. Its primary contrast is target-only `C0` versus the Stage-11-selected `C_best`, using the same held-out posts, folds, seeds, and selected backbone. This avoids mixing the context comparison with the separate Stage-15 ITFT-versus-direct strategy decision.

The script performs no training. It validates the upstream run artifacts, data IDs, labels, probabilities, source fingerprints, and left-truncation configuration before emitting results.

## Prerequisites

Run Stage 16 in the actual project repository, from the repository root, after the upstream stages have completed:

1. Revised `experiment_utils.py`, `pipeline.md`, and `experiment_protocol.md` are in place.
2. Stage 11 created the verified backbone decision and its versioned decision bundle.
3. Stage 15 created the verified training-strategy decision and bundle.
4. Stage 9/10 completed the selected backbone's Experiment II-A OOF runs for `C0`–`C4`, all five folds and both seeds.
5. `data/processed/ctx_matched_subset.csv` and its fold assignment are present and match the run manifests.

Stage 16 checks run manifests and hashes; do not combine old, unversioned outputs with `results/runs/left_truncation_v2/` outputs. If code, protocol, utility, or data fingerprints changed after training, rerun the required upstream stages rather than bypassing provenance checks.

## Run commands

From the repository root:

```bash
# Audit prerequisites and compute in-memory summaries/sample; write nothing
python scripts/16_error_analysis.py --validate-only

# Generate the immutable Stage 16 bundle, Chapter IV tables, figures, and editable worksheet
python scripts/16_error_analysis.py

# Set a different review-sample target; the allocation remains adaptive
python scripts/16_error_analysis.py --target-review-sample-size 90 --review-seed 20261026

# After completing the editable worksheet, finalize manual-review summaries
python scripts/16_error_analysis.py --finalize-review
```

The default review size of 90 is a target, not a rigid quota. Allocation responds to the actual observed outcome and context-type strata, respects available cases, samples without replacement, and is reproducible for a fixed seed. Do not interpret proportions in the manually reviewed sample as prevalence among all 900 records.

## Main outputs

### Immutable Stage 16 run bundle

`results/runs/left_truncation_v2/stage_16/`

Includes source/provenance manifest, hashed machine-readable outputs, `stage16_summary.json`, completion marker, and artifact checksum index.

### Full machine-readable ledger

`results/tables/left_truncation_v2/stage16_all_oof_cases.csv`

Contains one row per matched post, gold label, the C0 and C_best predictions, correctness/outcome indicators, probability/confidence diagnostics, context-type metadata, parent turns, and target text. Its four core comparative outcomes are:

- `both_correct`
- `context_fixed_error` — C0 wrong, C_best correct
- `context_introduced_error` — C0 correct, C_best wrong
- `both_wrong`

If `C_best == C0`, the script marks the comparison unavailable and uses same-condition outcomes rather than implying a context effect.

### Thesis-facing context-type table

`results/tables/left_truncation_v2/stage16_context_type_crosstab.csv`

One row per observed `context_type`, with its denominator and share of all OOF cases, gold-label counts, C0/C_best correct/error counts and rates, and counts/within-type percentages for each outcome. `stage16_context_type_by_gold_label.csv` is a supplementary stratified table to help inspect class-composition effects.

### Other quantitative outputs

- `stage16_evaluation_summary.csv` — accuracy, macro precision/recall/F1 and weighted precision/recall/F1.
- `stage16_per_class_metrics.csv` — per-class precision, recall, F1 and support.
- `stage16_confusion_matrix.csv` — confusion counts plus percentages within gold label.
- `stage16_probability_diagnostics.csv` — predicted confidence, gold-label probability, top-two margin and entropy.
- `stage16_paired_bootstrap.csv` — paired OOF bootstrap intervals for C_best minus C0 macro-F1 and per-class F1. Interpret as conditional on the selected backbone/context; it does not correct selection optimism.
- `stage16_outcome_summary.csv`, `stage16_outcome_by_gold_label.csv`, `stage16_error_profile_by_gold_label.csv`, and `stage16_prediction_transitions.csv` — error direction and gold-to-prediction transitions.
- `stage16_codebook.csv` — context-type definitions and human-review candidate error mechanisms.
- `stage16_manual_review_sample_template.csv` — initial review sheet template.

Figures, when Matplotlib is available, are written to `results/figures/left_truncation_v2/`.

## Manual qualitative review

Stage 16 creates an editable copy at `results/manual_review/stage16_manual_review_sample.csv`, kept outside the immutable bundle. For each sampled case, review the gold label, C0/C_best predictions, target, context, and observed outcome; then record a primary mechanism, optional secondary mechanism, notes, and adjudication status. Candidate mechanism codes are hypotheses for human annotation, not automatically detected causes. Correctly classified controls can be coded `NOT_APPLICABLE_NO_ERROR`; they must not be counted as error mechanisms.

After reviewing, use `--finalize-review`. The resulting mechanism counts describe only the reviewed sample. The 900-case ledger and full-cohort quantitative tables remain the basis for cohort-level descriptive rates. The codebook belongs in the thesis appendix; Chapter III should explain the coding and review procedure, while Chapter IV reports the observed patterns and representative reviewed cases.

## Important interpretation safeguards

- Gold labels are carried through the ledger and used to define correctness, per-class metrics, the confusion matrix, outcomes, and prediction transitions.
- Primary C0-versus-C_best comparisons use the same backbone and Experiment II-A / ITFT prediction protocol. Do not substitute the final deployment models if Stage 15 selected different training strategies for the two deployment modes; that would confound context and training strategy.
- Context-type cross-tabs are descriptive. Small categories can have unstable rates, so show counts/denominators and avoid strong claims from small cells.
- Human-coded mechanism frequencies should not be extrapolated to all 900 records unless a separate probability-sampling design and its weighting assumptions justify it.
- The script will refuse to overwrite an existing Stage 16 output bundle or review worksheet. Archive old outputs before rerunning.

## Tests

```bash
python -m py_compile scripts/16_error_analysis.py tests/test_16_error_analysis.py
pytest -q tests/test_16_error_analysis.py
```

The Stage 16 implementation has unit tests for label/outcome handling, metrics and context cross-tabs, adaptive sample allocation, and review finalization validation. A full end-to-end run additionally requires the actual Stage 11/15 bundles, matched subset, and manifested OOF run artifacts.
