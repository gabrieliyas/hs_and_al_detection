# README and pipeline-file migration notes

Apply this update at the project root, preserving the relative paths shown in the package.

## Canonical files

Replace the current root files with the included versions of:

- `README.md`
- `experiment_utils.py`
- `pipeline.md`
- `experiment_protocol.md`
- `.gitignore`

Also replace the included adjusted source/test files under `scripts/` and `tests/`, and add `docs/COLAB_ARTIFACT_HANDOFF.md` and `docs/stages/`.

## Remove superseded active-name duplicates

Delete these old files after ensuring the canonical counterparts are present:

- `pipeline_revised.md` (superseded by `pipeline.md`; `pipeline_legacy.md` may be kept as history)
- `scripts/17_train_models.py` (canonical: `scripts/17_train_final_models.py`)
- `tests/test_17_train_models.py` (canonical: `tests/test_17_train_final_models.py`)
- root-level `test_experiment_utils.py` (canonical: `tests/test_experiment_utils.py`)

The utility keeps a rename-map entry for the old Stage 17 path as migration metadata; that does not mean the legacy script should remain in the repository.

## Validation status

After this update, the full test suite in `tests/` passes in the generated workspace (160 passed, 5 subtests passed). The full pipeline filename/directory audit is not yet complete because Stage 18 (`extension/backend/`), Stage 19 (`extension/frontend/`), and Stage 20 (`scripts/20_test_extension_e2e.py`) remain to be implemented. Do not start the clean, corrected experiment run until the complete project audit is satisfied.
