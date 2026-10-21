# Stage 8 update: contextual configuration, timing, and latency

## Files in this package

- `experiment_utils.py` — removes a duplicate `_CONTEXT_PARENT_COLUMNS` definition that accidentally removed C0 support from `build_context_text()`.
- `scripts/8_timing_pilot.py` — current Stage 8 script.
- `tests/test_experiment_utils.py` — includes a regression test confirming C0-C4 input construction.
- `tests/test_8_timing_pilot.py` — model-free tests for Stage 8 split/configuration rules, contextual LR selection, latency sampling/measurement shape, and summary outputs.
- `experiment_protocol.md` and `pipeline_revised.md` — current versions included so source fingerprints can be kept consistent with this workspace.

## Stage 8 run sequence

Run from the project root after Stages 1, 1b, 2, the benchmark LR preflight, and both ITFT runs are complete:

```bash
python scripts/8_timing_pilot.py --validate-only
python scripts/8_timing_pilot.py
```

The actual run requires PyTorch, Transformers, scikit-learn, pandas, the pinned Hugging Face tokenizer/model revisions, the benchmark-only shared configs, and both manifest-verified ITFT checkpoints. Use the intended GPU runtime. The full pipeline-name audit must pass before the training run is allowed.

Stage 8 measures single-post inference latency for C0-C4 using the same sampled post IDs and active device for both backbones (default: 30 observations/context, 5 warm-ups, batch size 1). The saved artifact reports the actual machine/device description, precision, max length, truncation policy, medians, p95s, and raw samples. These measurements should only be used as the Stage 11 latency tie-break when the measured machine/device is the intended inference backend; otherwise measure latency on the deployment target before making that decision.

## Tests

From the repository root:

```bash
python -m py_compile experiment_utils.py scripts/8_timing_pilot.py tests/test_experiment_utils.py tests/test_8_timing_pilot.py
python -m pytest -q tests/test_experiment_utils.py tests/test_8_timing_pilot.py
python -m pytest -q tests
```

The model-free test suite passes in this workspace. It does not perform actual model downloads, GPU fine-tuning, or physical inference benchmarking.
