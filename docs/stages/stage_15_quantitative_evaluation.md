# Stage 15 — Cross-experiment quantitative evaluation

## Purpose

`scripts/15_evaluate_all.py` assembles the quantitative evidence from Experiments I, II-A, II-B, and III. It validates the upstream run manifests, artifact indexes, output hashes, input data fingerprints, post-ID alignment, and the Stage 11 decision. It reports the already-selected backbone/context choices and independently selects ITFT versus direct fine-tuning for the target-only (C0) and context-aware (Stage 11 C_best) deployment modes.

The script does not train models, select a new backbone, or reselect C_best based on shallow-pool results. Stage 13 remains descriptive.

## Outputs

The authoritative, versioned bundle is written to:

`results/runs/left_truncation_v2/stage_15/`

Thesis-facing CSV tables are exported to:

`results/tables/left_truncation_v2/`

Publication-ready 300-dpi PNG figures are exported to:

`results/figures/left_truncation_v2/`

The canonical strategy decision for Stage 17 is:

`results/training_strategy_decision.json`

Main Chapter IV tables are intentionally compact. Detailed per-seed, per-class, per-fold/seed, and long-form confusion-matrix data are separate supplementary CSVs. The complete confusion-matrix collection includes the nine cells for each reported group; selected comparisons also have separate figure files.

## Statistical rules

- Experiment I: five-seed mean/sample SD plus the Stage 5 paired-bootstrap comparison based on seed-ensembled test probabilities.
- Experiment II-A: the primary contextual point estimates use seed-ensembled pooled OOF predictions aligned to the same 900 `post_id` values. Fold/seed mean/SD are supplementary variability summaries.
- Stage 11 backbone and per-backbone C_best decisions are reported and verified, not recalculated by Stage 15.
- Stage 13 shallow-pool results are descriptive and do not change the selected backbone or C_best.
- ITFT versus direct fine-tuning is compared with a paired 10,000-resample, 95% bootstrap on aligned OOF predictions, separately for C0 and the selected context size.
- Micro precision/recall/F1, Hamming loss, and exact-match accuracy are omitted from main tables because they duplicate accuracy in this single-label, three-class task. Per-class precision/recall/F1/support and confusion matrices remain available in supplementary tables.

## Validate and run

Run from the repository root after the clean runs for Stages 1–14 have completed and their verified artifacts have been transferred into the repository:

```bash
python scripts/15_evaluate_all.py --validate-only
python scripts/15_evaluate_all.py
```

The first command loads and verifies all required upstream results and computes the report/strategy decisions in memory without publishing outputs. The second command additionally requires the complete script/directory audit to pass and publishes the Stage 15 bundle, report exports, and canonical strategy decision.

The script uses 10,000 paired-bootstrap resamples and rejects other `--n-bootstrap` values. Do not delete historical outputs indiscriminately. Archive any existing Stage 15 bundle, thesis-export folders, or canonical training-strategy decision before a deliberate fresh publication; the script refuses to overwrite them.

## Verification performed in this workspace

- Python compilation passed for `experiment_utils.py`, the Stage 15 script, and its tests.
- The full `tests/` suite passed: **145 passed, 5 subtests passed**.
- Tests use synthetic fixtures. This workspace did not contain a complete real set of Stage 1–14 run artifacts, so Stage 15 has **not** been run end to end on actual experiment outputs.
