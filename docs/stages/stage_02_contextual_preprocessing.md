# Stage 2 — Contextual dataset preprocessing

## Purpose

` scripts/2_preprocess_contextual.py` validates the custom annotated contextual corpus, preserves the modelling labels and analysis-only annotation fields, creates the full cleaned set, constructs the matched and shallow partitions, and assigns the shared five-fold mapping once. This is the source of the stable IDs/folds used by later experiments.

## Inputs

- `data/raw/contextual_dataset.csv`
- Current root-level `experiment_utils.py`, `pipeline.md`, and `experiment_protocol.md`
- The registered script `scripts/2_preprocess_contextual.py`

## Outputs

- `data/processed/ctx_cleaned.csv` — all 1,800 posts for final Stage 17 training
- `data/processed/ctx_matched_subset.csv` — 900 complete-case posts, each with four available parents and a fixed `fold`
- `data/processed/ctx_shallow_pool.csv` — the disjoint 900-post shallow pool with one to three available parents
- `results/tables/left_truncation_v2/ctx_distribution.csv`
- Complete provenance bundle under `results/runs/left_truncation_v2/stage_02/`

## Run

From the repository root, after placing the approved contextual source file at the expected path:

```bash
python scripts/2_preprocess_contextual.py
```

This script does not currently expose `--validate-only`; it performs the full preprocessing step. It validates the input and refuses to overwrite existing outputs. Preserve or archive an earlier run before regenerating outputs.

## Safeguards

- Check the 1,800-row count, three-class balance, unique `post_id` values, parent-availability distribution, and expected matched/shallow partition sizes.
- Confirm that the two partitions are disjoint and that fold assignments are generated once and remain stable.
- Analysis-only annotation columns must not enter model tokenization; use the shared utility's modelling view/input allowlist.
- Preserve the full Stage 2 provenance bundle alongside all processed CSVs. Keep the actual corpus in an access-controlled store rather than committing its post text to Git.
