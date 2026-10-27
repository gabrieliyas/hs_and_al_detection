# Stage 9/10 Experiment II-A update

## Files
- `scripts/9_train_e2_ablation.py`: shared, backbone-parameterized training entry point.
- `tests/test_9_train_e2_ablation.py`: model-free regression tests.

## Invocations
Run once per backbone, from the repository root and in the same project environment as the current frozen configuration:

```bash
python scripts/9_train_e2_ablation.py --backbone indobertweet --validate-only
python scripts/9_train_e2_ablation.py --backbone xlmr --validate-only
```

Only after the complete pipeline audit passes, invoke each without `--validate-only` on a compatible GPU runtime.

Stage 9 (`indobertweet`) records 50 runs under `results/runs/left_truncation_v2/stage_09/e2_indobertweet/`. Stage 10 (`xlmr`) records 50 runs under `results/runs/left_truncation_v2/stage_10/e2_xlmr/`. Every run has a manifest, OOF predictions, probabilities, metrics and output hash inventory. Stage 11 must audit individual per-run manifests and outputs; the aggregate CSV is a convenience table, not a substitute for that audit.

The script expects completed, current Stage 2 and Stage 8 manifests, both valid ITFT checkpoints, and Stage 8's contextual max-length/LR configs. It intentionally blocks on legacy flat-path Experiment II outputs; archive old results before the clean rerun.

## Testing

```bash
python -m pytest -q tests/test_9_train_e2_ablation.py tests/test_experiment_utils.py
python -m pytest -q tests
```

Tests mock model execution and do not perform GPU fine-tuning.
