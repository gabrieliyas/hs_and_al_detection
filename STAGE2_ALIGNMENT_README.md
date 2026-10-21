# Stage 2 alignment update

This package aligns the current `experiment_utils.py` contract with Stage 2 contextual preprocessing.

## File placement

- `experiment_utils.py` — repository root
- `experiment_protocol.md` — repository root (current protocol used by manifests)
- `pipeline_revised.md` — repository root (current pipeline table used by manifests)
- `scripts/2_preprocess_contextual.py` — Stage 2 script
- `tests/test_experiment_utils.py` — shared-utility regression tests
- `tests/test_2_preprocess_contextual.py` — Stage 2 integration tests using generated synthetic data

## Test without using the private raw dataset

From the repository root:

```bash
python -m py_compile experiment_utils.py scripts/2_preprocess_contextual.py
python -m pytest -q tests/test_experiment_utils.py
python -m pytest -q tests/test_2_preprocess_contextual.py
```

The Stage 2 integration tests generate a synthetic dataset with the expected schema and counts. They do not need or alter `data/raw/contextual_dataset.csv`.

## Run the real Stage 2 preprocessing

After the full script set and shared protocol are frozen, archive all prior Stage 2 outputs and manifest, then run from the repository root:

```bash
python scripts/2_preprocess_contextual.py
```

The script fails rather than overwriting existing outputs or reusing an old Stage 2 manifest. It writes an immutable manifest under `results/runs/left_truncation_v2/stage_02/` and verifies hashes of the raw input and all outputs.
