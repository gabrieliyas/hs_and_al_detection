# Stage 14 — Experiment III direct fine-tuning

Files in this update:

- `scripts/14_train_e3_direct.py`: authoritative Stage 14 script.
- `tests/test_14_train_e3_direct.py`: independent model-free regression tests.

The script loads the selected backbone and that backbone's `C_best` from the verified Stage 11 decision. It trains directly from the pinned public pretrained checkpoint (without ITFT) for C0 and C_best, using five shared folds and seeds 42/123. If `C_best == C0`, it runs C0 only once rather than duplicating identical experiments.

Run tests from the repository root:

```bash
python -m pytest -q tests/test_14_train_e3_direct.py
python -m pytest -q tests
```

After the upstream stages finish and the complete pipeline audit passes, validate Stage 14 without training:

```bash
python scripts/14_train_e3_direct.py --validate-only
```

Actual training requires verified Stage 2, Stage 8, Stage 11, and selected-backbone Experiment II outputs, plus a GPU environment compatible with the Stage 8 mixed-precision setting.
