# Stage 11 update: context/backbone decision gate

## Included files

- `experiment_utils.py`: current shared utility, including strict validation/provenance checks for the saved Stage 11 decision.
- `scripts/11_select_backbone.py`: audits both Experiment II-A 50-run grids, selects C_best per backbone, validates Stage 8 latency, applies the backbone decision rule, and publishes the decision.
- `tests/test_11_select_backbone.py`: standalone Stage 11 helper tests.
- `tests/test_experiment_utils.py`: authoritative utility regression test file. The duplicate root-level `test_experiment_utils.py` was removed.
- `experiment_protocol.md` and `pipeline.md`: current protocol and pipeline documents referenced by the script.

## What Stage 11 needs

Stage 11 must not be used to make a real decision until the clean current run has produced:

1. Stage 2's matched contextual subset, shared folds, and valid Stage 2 manifest.
2. Stage 8's published contextual sequence-length and learning-rate configuration, inference latency data for C0-C4 for both backbones, receipt, manifest, and hashes.
3. Stage 9's complete 50-run IndoBERTweet grid with current per-run manifests, metrics, ordered predictions, probabilities, and hashes.
4. Stage 10's complete 50-run XLM-RoBERTa-base grid with the corresponding verified artifacts.
5. The current source files `experiment_utils.py`, `experiment_protocol.md`, `pipeline.md`, and the registered Stage scripts, all frozen before the clean rerun.

Stage 5's verified Experiment I comparison is optional and recorded only for discussion. It does not affect the Stage 11 rule.

## Commands

Run from the repository root after the real upstream artifacts exist:

```bash
python -m pytest -q tests/test_11_select_backbone.py tests/test_experiment_utils.py
python -m pytest -q tests
python scripts/11_select_backbone.py --validate-only
python scripts/11_select_backbone.py
```

`--validate-only` audits all required upstream artifacts and computes the decision in memory. It does not write Stage 11 results. The publication run also requires the complete pipeline script/directory audit to pass.

## Decision outputs

- `results/backbone_decision.json` — public decision file consumed by Stages 12-17; it is accepted only when its versioned Stage 11 manifest, completion marker, artifact index, and copy all validate.
- `results/runs/left_truncation_v2/stage_11/decision/` — manifest, context-choice documents for each backbone, grid audit documents, decision copy, hash index, and completion marker.

A previous or partially written Stage 11 result must be archived before regenerating it. Do not edit the decision JSON manually.

## Validation status in this workspace

The Stage 11 script and utility compile successfully. Current tests under `tests/` pass. The actual decision has **not** been computed because the current workspace does not contain the real, complete Stage 8 latency bundle and both 50-run Stage 9/10 output grids. The pipeline filename/directory audit also remains incomplete; Stage 11 publication is therefore blocked until those scripts/directories are finalized.
