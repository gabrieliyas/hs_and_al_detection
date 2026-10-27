# Stage 17 — Final deployment-model training

## Canonical filename

The authoritative utility registry currently defines Stage 17 as `scripts/17_train_final_models.py`. The shorter filename `17_train_models.py` is treated as a legacy name. Keep `scripts/17_train_final_models.py` as the canonical filename so `audit_pipeline_script_names()`, manifests, and run-config validation agree.

## Can the script be written before upstream results exist?

Yes. The code can be implemented and unit-tested before the model-selection results exist. **Executing it is not independent of previous stages.** The script deliberately requires current, provenance-verified outputs from:

- Stage 2: `data/processed/ctx_cleaned.csv` and its manifest, confirming all 1,800 posts.
- Stage 8: published shared sequence-length and learning-rate configuration, pinned model/tokenizer revisions, and training hyperparameters.
- Stage 11: `results/backbone_decision.json`, including the selected backbone and its selected context size `C_best`.
- Stage 15: `results/training_strategy_decision.json`, selecting `itft` or `direct` separately for `target_only` and `context_aware`.
- Stage 6 or 7: a verified ITFT checkpoint for the selected backbone **only if at least one mode selected `itft`**.

It does not require Stage 12–14 or Stage 16 results to train the final pair.

## Run

From the repository root, normally in a GPU-enabled Colab runtime:

```bash
python scripts/17_train_final_models.py --validate-only
python scripts/17_train_final_models.py
```

`--validate-only` checks data and upstream provenance without writing Stage 17 model artifacts. The actual run trains two separate models on all 1,800 contextual posts, one per deployment mode. It uses no validation split or checkpoint selection on the final training data; the frozen fixed-epoch schedule is used instead.

## Decisions and input construction

- Backbone and `C_best` come from Stage 11; Stage 17 does not reselect either.
- Strategy is selected independently for the two modes by Stage 15.
- `direct` initializes from the registered, revision-pinned pretrained backbone.
- `itft` initializes from the verified Stage 6/7 ITFT checkpoint for the selected backbone.
- C0 uses the target text only.
- The context-aware model uses up to `C_best` available parents per post, with fewer parents when the thread is shallow. Parent messages are ordered oldest-to-newest, followed by the target.
- Contextual inputs use the tokenizer's own separator token and explicit left truncation so the target at the end is preserved.
- Both models train on the same 1,800 posts, but may use different Stage 15-selected initialization strategies.

## Outputs

Model directories:

- `models/deployment/<backbone>/target_only/`
- `models/deployment/<backbone>/context_aware_C<n>/`

Versioned provenance and training summaries:

- `results/runs/left_truncation_v2/stage_17/final_<backbone>/`
- `stage17_summary.csv` and `stage17_summary.json`
- Per-mode manifests, metrics, training logs, SHA-256 inventories, and completion markers.

Training loss and training duration are optimization diagnostics, **not held-out generalization metrics**. Evaluate generalization using the cross-validation/OOF results from earlier stages, not the final 1,800-row training run.

## Safeguards

- Requires the `thesis-run-config-v3` schema and `left_truncation_v2` run ID.
- Validates Stage 2, 8, 11, and 15 provenance and hashes.
- Verifies ITFT files only if ITFT is selected for either mode.
- Rejects legacy outputs as resume signals and refuses to overwrite an existing unverified deployment model.
- Writes the selected backbone, `C_best`, per-mode strategies, source revisions, dataset hashes, sequence length, truncation policy, fixed training schedule, and checkpoint hashes into manifests.
