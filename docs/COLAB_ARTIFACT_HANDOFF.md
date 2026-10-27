# Colab Artifact Handoff Checklist

This guide follows the canonical `pipeline.md`, `experiment_protocol.md`, and `experiment_utils.py` conventions for `run_id = left_truncation_v2`. It is an operational transfer checklist, not an alternative experimental protocol.

## Recommended transfer pattern

Use a frozen Git commit for source code and Google Drive (or another persistent, access-controlled artifact store) for datasets, configuration outputs, checkpoints, raw predictions, and run bundles. Colab's ephemeral `/content` directory should be treated as temporary. After each run, copy the **entire listed directory tree**, preserving relative paths and filenames, to persistent storage and then into the local project tree where later VS Code stages need it. Do not flatten folders or copy only the headline CSV if a later stage verifies manifests or hashes.

For every fresh Colab runtime, make sure the project contains the same frozen versions of:

- `experiment_utils.py`
- `pipeline.md`
- `experiment_protocol.md`
- the registered `scripts/<stage>.py` file(s) and imported project modules
- the project's pinned dependency specification, if maintained (for example, `requirements-colab.txt`)
- required raw/processed inputs and all upstream run artifacts noted below

The canonical source data paths expected by Stage 1 and Stage 2 are `data/raw/re_dataset.csv` and `data/raw/contextual_dataset.csv`. Keep a stable byte-identical copy when provenance manifests refer to those paths. The contextual source file should not be moved into a different location without updating the preprocessing script and rerunning the affected provenance chain.

## Local VS Code stages that prepare artifacts for Colab

| Stage | Required local inputs | Outputs to retain locally and transfer to Colab |
|---|---|---|
| 1 — benchmark preprocessing | `data/raw/re_dataset.csv`; source tree and frozen protocol files | `data/processed/ib_cleaned.csv`, `ib_train.csv`, `ib_val.csv`, `ib_test.csv`; `results/tables/ib_class_weights.csv` |
| 1b — benchmark max length | `data/processed/ib_train.csv`; utility and both registered tokenizer definitions | `results/config/sequence_length.json` and `results/tables/left_truncation_v2/benchmark_length_audit.csv` |
| 2 — contextual preprocessing | `data/raw/contextual_dataset.csv`; utility, pipeline, protocol | `data/processed/ctx_cleaned.csv`, `ctx_matched_subset.csv`, `ctx_shallow_pool.csv`; `results/tables/left_truncation_v2/ctx_distribution.csv`; the entire `results/runs/left_truncation_v2/stage_02/` directory, including manifest/marker/index files |

Stage 1, 1b, 2, 5, 11, 15, and 16 are VS Code stages in the current pipeline. Keep their outputs in the project tree; the table below identifies artifacts produced by Colab that must be brought back before those stages run.

## Colab execution handoff, stage by stage

| Colab stage | Upload or make available before execution | Download/copy out after successful execution |
|---|---|---|
| Preflight benchmark learning-rate search | Source bundle; `data/raw/re_dataset.csv` if required by upstream manifest verification; `data/processed/ib_train.csv`, `ib_val.csv`; `results/tables/ib_class_weights.csv`; `results/config/sequence_length.json` | `results/config/learning_rates.json`; entire `results/runs/left_truncation_v2/stage_preflight/` tree (trial table, benchmark snapshot, manifest and completion/index files) |
| 3 — Experiment I, IndoBERTweet | Source bundle; `data/processed/ib_train.csv`, `ib_val.csv`, `ib_test.csv`; class-weight table; sequence-length and learning-rate configs | Entire `results/runs/left_truncation_v2/experiment_i/indobertweet/` tree, including each seed's `predictions.csv`, `probabilities.npy`, `metrics.json`, run manifest and artifact index. Checkpoints are not required for Stage 5. |
| 4 — Experiment I, XLM-R | Same benchmark files/configuration as Stage 3; use the XLM-R invocation of the registered script | Entire `results/runs/left_truncation_v2/experiment_i/xlmr/` tree with per-seed predictions, probability arrays, metrics and manifests. Checkpoints are not required for Stage 5. |
| 6 — ITFT, IndoBERTweet | Source bundle; `ib_train.csv`, `ib_val.csv`; class-weight table; sequence-length config and `learning_rates.json` | `models/itft_checkpoint/indobertweet/` (model/tokenizer/config files); the complete Stage 6 run bundle under `results/runs/left_truncation_v2/`; manifests, marker/index and checkpoint manifest |
| 7 — ITFT, XLM-R | Same as Stage 6; use `--backbone xlmr` | `models/itft_checkpoint/xlmr/` plus its complete Stage 7 run bundle and checkpoint provenance files |
| 8 — timing pilot and contextual LR tuning | Source bundle; raw files needed to verify the Stage 2 manifest; `ctx_matched_subset.csv`; `ib_train.csv`, `ib_val.csv`; class weights; both verified ITFT checkpoint directories and their run bundles; current `sequence_length.json` and `learning_rates.json`; full Stage 2 run bundle | Updated `results/config/sequence_length.json` and `results/config/learning_rates.json`; full `results/runs/left_truncation_v2/stage_08/` tree, especially `summary/` manifests/markers/index, token-length statistics, LR-trial tables, runtime summaries and `context_latency_measurements.json` |
| 9 — Experiment II-A, IndoBERTweet | Source bundle; `ctx_matched_subset.csv`; required benchmark split/weight inputs; Stage 2 and Stage 8 run bundles; Stage 8 configs; `models/itft_checkpoint/indobertweet/` with its verified provenance | Entire `results/runs/left_truncation_v2/stage_09/` tree for the IndoBERTweet grid: all C0–C4 × 5 folds × 2 seeds run folders, predictions, probabilities, metrics, manifests, completion markers and artifact indexes |
| 10 — Experiment II-A, XLM-R | Same as Stage 9 but use `models/itft_checkpoint/xlmr/` and `--backbone xlmr` | Entire `results/runs/left_truncation_v2/stage_10/` tree for the XLM-R grid, including every run's prediction/probability and provenance files |
| 12 — retrain C0–C4 on matched 900 | Source bundle; `ctx_matched_subset.csv`; Stage 2 and Stage 8 run bundles; Stage 8 configs; `results/backbone_decision.json` plus the complete Stage 11 decision bundle; the selected backbone's verified ITFT checkpoint | `models/e2_full_retrained/<selected_backbone>/C0/` through `C4/`; entire `results/runs/left_truncation_v2/stage_12/` tree, including metrics, summary, manifests, completion marker and artifact index |
| 13 — shallow-pool generalization | Source bundle; `ctx_shallow_pool.csv`, `ctx_matched_subset.csv`; Stage 8/11 verified bundles and configs; `results/backbone_decision.json`; Stage 12 model directories and Stage 12 run bundle | Entire `results/runs/left_truncation_v2/stage_13/` tree (tables, per-class/per-availability results, confusion matrices, summaries, manifests/markers/index). Copy it back to local for Stage 15. |
| 14 — Experiment III direct fine-tuning | Source bundle; `ctx_matched_subset.csv`; benchmark train/validation inputs and configs needed by provenance checks; Stage 2, Stage 8, and Stage 11 bundles; `results/backbone_decision.json`; selected-backbone Stage 9/10 grid outputs for the paired comparison | Entire `results/runs/left_truncation_v2/stage_14/` tree: all fold/seed predictions, probabilities, metrics, aggregate tables and provenance/complete markers |
| 17 — final deployment models | Source bundle; full `data/processed/ctx_cleaned.csv` (all 1,800 posts), `ctx_matched_subset.csv` for config/provenance checks; benchmark train/validation files and class-weight table required by the script; Stage 2, 8, 11 and 15 verified bundles; `results/config/sequence_length.json`, `learning_rates.json`; `results/backbone_decision.json`; `results/training_strategy_decision.json`; the selected backbone's verified ITFT checkpoint if either mode selects ITFT | Both final model directories `models/deployment/<backbone>/target_only/` and `models/deployment/<backbone>/context_aware_C<n>/`; full `results/runs/left_truncation_v2/stage_17/` bundle, including both model manifests, training logs, metrics, hashes, completion marker and index. Store persistent copies for Stages 18–19. |

### Downloads required for the intervening VS Code stages

| VS Code stage | Bring back from Colab before running | Keep after running |
|---|---|---|
| 5 — Experiment I evaluation | Both Stage 3 and Stage 4 `experiment_i/` trees | `results/tables/e1_results.csv`, all supplementary Experiment I tables; entire `results/runs/left_truncation_v2/experiment_i/evaluation/` tree including its manifest, marker and index |
| 11 — backbone/context decision | Full Stage 2 and Stage 8 bundles; all Stage 9 and Stage 10 run folders; Stage 5 evaluation bundle if available; both configs and `ctx_matched_subset.csv` | `results/backbone_decision.json`; entire `results/runs/left_truncation_v2/stage_11/decision/` directory including canonical decision copy, manifest, marker and index |
| 15 — quantitative evaluation and strategy selection | Benchmark train/val/test and class weights; `ctx_matched_subset.csv`, `ctx_shallow_pool.csv`; configs; full Stage 5, 8, 9, 10, 11, 12, 13 and 14 bundles and model/provenance artefacts as required by the evaluator | Entire `results/runs/left_truncation_v2/stage_15/`; `results/training_strategy_decision.json`; all `results/tables/left_truncation_v2/` tables and `results/figures/left_truncation_v2/` figures |
| 16 — qualitative error analysis | `ctx_matched_subset.csv`; Stage 11 decision bundle; Stage 15 bundle and `results/training_strategy_decision.json`; selected-backbone Stage 9/10 OOF run trees | Entire `results/runs/left_truncation_v2/stage_16/`; selected thesis-facing Stage 16 tables/figures; the manual-review worksheet and completed annotations kept in a restricted local/Drive location |

## What must be retained in the local project tree?

Always keep a byte-identical copy of every input/configuration and upstream run bundle needed by the next stage. Provenance validation may require `*_run_manifest.json`, `run_complete.json`/stage completion marker, `artifacts_sha256.json`, prediction CSVs, probability arrays, and summary files even when only a small CSV is ultimately used in Chapter IV. Do not keep only a rendered table and discard its source bundle.

Store large checkpoints and raw OOF prediction bundles in Google Drive or another persistent controlled store; keep them under their canonical paths when staging a run. At minimum, make the final Stage 17 model directories available on the machine running the FastAPI backend. If local disk permits, mirror the final models under the local project path as well. Do not commit model checkpoints or full per-run prediction dumps to Git.

## Git and data policy

The repository should contain the **scripts and directory placeholders**, not necessarily the processed dataset bytes. The supplied `.gitignore` already ignores `data/processed/*` except `.gitkeep`, and ignores model contents. Keep that policy by default for the custom X post corpus; keep the actual data in a private, access-controlled location and transfer it to Colab with checksums. Track `data/processed/.gitkeep` (and a README describing expected filenames/schema if desired) so the directory exists after cloning. The scripts regenerate the CSVs from the raw sources, but every run still needs the matching files available locally or in Colab.

Before publishing, also consider ignoring the raw source data and high-volume run bundles, because the current supplied `.gitignore` does **not** ignore `data/raw/*` or `results/runs/**`. A recommended rule set is:

```gitignore
data/raw/*
!data/raw/.gitkeep

data/processed/*
!data/processed/.gitkeep

models/**
!models/.gitkeep

results/runs/**
!results/runs/.gitkeep
results/raw/*
!results/raw/.gitkeep
results/manual_review/*
!results/manual_review/.gitkeep
```

This deliberately does not ignore `results/config/`, root decision JSONs, or compact thesis aggregate tables. Review any Stage 16 files that include actual post text before committing them. If your data governance/ethics approval and source terms explicitly permit distributing a derived dataset, you could choose a private/versioned data store; do not publish the full post text/IDs merely for convenience without checking those constraints.

## Freeze and provenance rule

Freeze `experiment_utils.py`, `pipeline.md`, `experiment_protocol.md`, configuration decisions, and the Stage 1–17 scripts **before the clean, corrected Stage 1–17 rerun begins**. Do not wait for Stage 20. Run bundles fingerprint the source utility/scripts and protocol/pipeline files; changing one of those files later can invalidate an upstream artifact check and require rerunning dependent stages. Once the extension is being implemented, put detailed UI/API choices and the Stage 20 manual browser procedure in separate files (for example, `docs/extension_architecture.md` and `docs/stage20_browser_test_procedure.md`) rather than editing the frozen experiment protocol. Freeze the extension separately after Stages 18–20 pass their tests.
