# Conversation-Aware Hate Speech and Abusive Language Detection on X Using IndoBERTweet

Undergraduate thesis project evaluating whether conversational context improves three-class hate-speech and abusive-language detection on Indonesian X (formerly Twitter), and demonstrating the selected models in a Manifest V3 browser extension.

## Source-of-truth files

These files must remain at the repository root because run manifests and the shared utility use their exact paths and hashes:

- [`experiment_utils.py`](experiment_utils.py) — model/tokenizer registry, experiment and stage identities, configuration validation, metrics, decisions, and provenance helpers.
- [`pipeline.md`](pipeline.md) — authoritative stage order, canonical script filenames, execution environments, and outputs.
- [`experiment_protocol.md`](experiment_protocol.md) — research design, data partitions, training protocol, selection rules, metrics, and interpretation limits.

Do not maintain a second active copy such as `pipeline_revised.md` or move the pipeline/protocol into `docs/` without updating the utility and every dependent manifest check. Keep stage-specific runbooks in [`docs/stages/`](docs/stages/README.md) and the Colab transfer checklist in [`docs/COLAB_ARTIFACT_HANDOFF.md`](docs/COLAB_ARTIFACT_HANDOFF.md).

## Project layout

```text
.
├── data/
│   ├── raw/                         # local/private inputs; do not commit unapproved post data
│   └── processed/                   # generated CSVs; gitignored
├── docs/
│   ├── stages/                      # focused stage runbooks, where useful
│   └── COLAB_ARTIFACT_HANDOFF.md    # exact Colab uploads/downloads and provenance retention
├── extension/
│   ├── backend/                     # Stage 18: FastAPI model-inference service
│   └── frontend/                    # Stage 19: TypeScript Manifest V3 extension
├── models/                           # checkpoints and deployment models; gitignored
├── results/
│   ├── config/                      # sequence-length and learning-rate configs
│   ├── runs/left_truncation_v2/      # source-of-truth versioned run bundles
│   ├── tables/left_truncation_v2/    # thesis-facing CSVs
│   └── figures/left_truncation_v2/   # generated thesis figures
├── scripts/                          # numbered pipeline scripts
├── tests/                            # unit/regression and synthetic integration tests
├── experiment_utils.py
├── experiment_protocol.md
├── pipeline.md
└── README.md
```

`pipeline.md` is the authoritative script map. The ordered Transformer pipeline excludes the optional/legacy SVM baseline. The registered Stage 17 entry point is `scripts/17_train_final_models.py`; do not use the former `14_train_final_models.py` or the shorter `17_train_models.py` as the canonical name. Stage 18 and 19 are source directories, not numbered Python scripts. Stage 20 is `scripts/20_test_extension_e2e.py` plus a written browser-test procedure; implement its real end-to-end checks after the extension interfaces are stable.

## Research and dataset overview

### Labels and data

The classification labels are `N` (Neither), `AL` (Abusive Language), and `HS` (Hate Speech), with the canonical ID mapping defined in `experiment_utils.py`.

The custom contextual corpus contains 1,800 annotated posts, balanced across the three classes (600 per class). Stage 2 creates:

- `data/processed/ctx_cleaned.csv` — all 1,800 posts; used to train the final Stage 17 models.
- `data/processed/ctx_matched_subset.csv` — 900 posts with four available parents and a fixed shared five-fold assignment; used for the primary C0–C4 context-depth comparisons.
- `data/processed/ctx_shallow_pool.csv` — a disjoint 900-post subset with one to three available parents; used for descriptive, availability-aware generalization analysis.

The benchmark dataset is the Ibrohim & Budi Indonesian hate-speech and abusive-language dataset. Stage 1 performs cleaning, canonical three-class mapping, stratified 70/15/15 splitting, and computes class weights from the training partition only. Stage 1b selects the shared benchmark maximum sequence length using the benchmark training partition only.

**Data handling:** `data/processed/` should not be committed merely to make the directory available. Keep `.gitkeep` and the processing scripts in Git, and store the approved raw/processed data in a persistent, access-controlled location such as a private Drive folder. Preserve the exact files and SHA-256 manifests required by subsequent stages. Do not publish the full contextual post text or annotated dataset unless its research permissions and distribution terms explicitly allow it. See the [Colab artifact handoff checklist](docs/COLAB_ARTIFACT_HANDOFF.md).

### Experiments

- **Experiment I (Stages 3–5):** compare IndoBERTweet and XLM-RoBERTa-base using target-only inputs on the same fixed benchmark split, with five seeds and paired evaluation.
- **Experiment II-A (Stages 6–12):** perform intermediate task fine-tuning (ITFT) on the benchmark, evaluate C0–C4 with five-fold cross-validation and two seeds per context configuration for each backbone, then select `C_best` and the deployment backbone using the predeclared Stage 11 decision rules. Stage 12 retrains C0–C4 on the complete matched 900-post subset.
- **Experiment II-B (Stage 13):** evaluate availability-aware inference on the shallow pool. These results are descriptive and do not select `C_best`.
- **Experiment III (Stage 14):** directly fine-tune the selected backbone without ITFT for C0 and `C_best`, using the shared folds and seeds.
- **Cross-experiment evaluation (Stage 15):** compare OOF evidence and choose the training strategy separately for target-only and context-aware deployment. The two modes may choose different strategies.
- **Qualitative error analysis (Stage 16):** inspect paired C0/C_best OOF predictions, gold labels, context-type patterns, and a reproducibly sampled manual-review subset.
- **Final model training (Stage 17):** train two separate models on all 1,800 posts, using the Stage 11 backbone/context choice and the Stage 15 strategy selected for each mode.
- **Extension and system evaluation (Stages 18–20):** integrate the models through the FastAPI backend, connect the TypeScript Manifest V3 frontend, then execute automated end-to-end and documented browser tests.

The selected model's held-out/OOF evaluation—not training loss or training duration from Stage 17—is the basis for generalization claims.

## Setup

Run project commands from the repository root. Use the project's pinned dependency specification when available; for example:

```bash
python -m pip install -r requirements.txt
```

For GPU training, use a Colab runtime compatible with the pinned PyTorch/Transformers versions and the saved Stage 8 precision/device configuration. Colab's ephemeral runtime is not persistent storage: after every successful stage, transfer the complete provenance bundle and required checkpoint files to persistent storage, preserving paths and filenames.

## Testing, stage validation, and execution are different

### 1. Python compilation check

```bash
python -m compileall -q experiment_utils.py scripts tests
```

This catches syntax/bytecode-compilation issues; it does not run tests or validate real data/artifacts.

### 2. Pytest regression tests

Run from the project root in the development environment. Tests exercise helper functions, safeguards, and selected workflows with synthetic or mocked inputs; they do not run the full GPU experiments.

```bash
# Shared utility tests only
python -m pytest -q tests/test_experiment_utils.py

# Focused utility + Stage 8 regression tests
python -m pytest -q tests/test_experiment_utils.py tests/test_8_timing_pilot.py

# Full available test suite — recommended after source changes and before freezing
python -m pytest -q tests
```

Use the targeted test commands while working on a component. Run the full suite before committing/finalizing a coordinated source change. The three commands are alternatives of increasing scope; they are not all required after every small edit. `python -m pytest -q tests` is the broadest regression command.

### 3. `--validate-only` preflight for a pipeline stage

Where supported, `python scripts/<stage_script>.py --validate-only` checks that stage's actual inputs, resolved configuration, manifests, hashes, and upstream artifacts without performing the stage's model training or publishing its outputs. It is **not** a pytest run and is not a substitute for the test suite. Run each validation just before the corresponding stage, after its prerequisites are available; validation for a later stage cannot pass before its upstream results exist.

Stage 1, Stage 1b, and Stage 2 currently do not expose `--validate-only`; running them executes preprocessing/length selection. The preflight LR command and Stages 3–17 support the validation flag. Stage-specific special arguments are shown below.

#### Validation commands (run at the appropriate point in the pipeline)

```bash
# Benchmark LR preflight
python scripts/preflight_tune_benchmark_lr.py --validate-only

# Experiment I
python scripts/3_train_e1_indobertweet.py --validate-only
python scripts/4_train_e1_xlmr.py --validate-only
python scripts/5_evaluate_e1.py --validate-only

# ITFT: the same script is invoked for Stages 6 and 7
python scripts/6_train_itft.py --backbone indobertweet --validate-only
python scripts/6_train_itft.py --backbone xlmr --validate-only

# Stage 8 tunes/measures both backbones; no --backbone argument is required
python scripts/8_timing_pilot.py --validate-only

# Experiment II-A: the same script is invoked for Stages 9 and 10
python scripts/9_train_e2_ablation.py --backbone indobertweet --validate-only
python scripts/9_train_e2_ablation.py --backbone xlmr --validate-only

# Decision gate, retraining, generalization, Experiment III, evaluation and error analysis
python scripts/11_select_backbone.py --validate-only
python scripts/12_train_e2_retrain_full.py --validate-only
python scripts/13_e2b_generalization.py --validate-only
python scripts/14_train_e3_direct.py --validate-only
python scripts/15_evaluate_all.py --validate-only
python scripts/16_error_analysis.py --validate-only

# Final two models (reads the verified Stage 11 and Stage 15 decisions)
python scripts/17_train_final_models.py --validate-only
```

A successful `--validate-only` run does not mean a training/evaluation stage was completed. Decision stages may compute a decision/report in memory but must not write it in validation mode. The full pipeline audit must also pass before actual run publication; it currently requires the registered scripts, `extension/backend/`, `extension/frontend/`, and `scripts/20_test_extension_e2e.py` to exist under their canonical paths.

### 4. Main pipeline execution commands

Execute in the order specified in [`pipeline.md`](pipeline.md), and do not skip an upstream stage simply because a later script exists. Commands below omit `--validate-only`; perform the matching validation first wherever supported.

```bash
# VS Code: preprocessing and shared inputs
python scripts/1_preprocess_benchmark.py
python scripts/1b_select_benchmark_max_length.py
python scripts/2_preprocess_contextual.py

# Colab: learning-rate preflight and Experiment I
python scripts/preflight_tune_benchmark_lr.py
python scripts/3_train_e1_indobertweet.py
python scripts/4_train_e1_xlmr.py

# VS Code: Experiment I evaluation
python scripts/5_evaluate_e1.py

# Colab: ITFT and timing/configuration pilot
python scripts/6_train_itft.py --backbone indobertweet
python scripts/6_train_itft.py --backbone xlmr
python scripts/8_timing_pilot.py

# Colab: Experiment II-A; run once per backbone
python scripts/9_train_e2_ablation.py --backbone indobertweet
python scripts/9_train_e2_ablation.py --backbone xlmr

# VS Code: select the backbone/context size
python scripts/11_select_backbone.py

# Colab: retrain on matched subset, shallow-pool evaluation, and Experiment III
python scripts/12_train_e2_retrain_full.py
python scripts/13_e2b_generalization.py
python scripts/14_train_e3_direct.py

# VS Code: cross-experiment evaluation/strategy selection and qualitative error analysis
python scripts/15_evaluate_all.py
python scripts/16_error_analysis.py

# Colab: final target-only and context-aware models on all 1,800 posts
python scripts/17_train_final_models.py
```

Stage 16's completed human-review worksheet is finalized separately with `python scripts/16_error_analysis.py --finalize-review`; see [`docs/stages/stage_16_error_analysis.md`](docs/stages/stage_16_error_analysis.md). Stage 20's invocation will be documented after the real FastAPI and extension interfaces are implemented.

## Output and reproducibility conventions

- Run namespace: `left_truncation_v2`.
- Versioned run bundles and provenance: `results/runs/left_truncation_v2/`.
- Shared configs: `results/config/sequence_length.json` and `results/config/learning_rates.json`.
- Stage 11 canonical selection: `results/backbone_decision.json`.
- Stage 15 canonical per-mode strategy decision: `results/training_strategy_decision.json`.
- Thesis-facing CSV tables: `results/tables/left_truncation_v2/`.
- Thesis figures: `results/figures/left_truncation_v2/`.
- Final deployment models: `models/deployment/<backbone>/target_only/` and `models/deployment/<backbone>/context_aware_C<n>/`.

Never flatten a run bundle or keep only its headline CSV. Downstream stages can validate manifests, completion markers, artifact indexes, prediction/probability files, and SHA-256 hashes. Archive old unversioned outputs instead of allowing them to satisfy a new run's resume check. Freeze `experiment_utils.py`, `pipeline.md`, `experiment_protocol.md`, configuration decisions, and stage source files before the clean corrected experiment rerun.

## Documentation map

- [`pipeline.md`](pipeline.md) — canonical stage sequence and files.
- [`experiment_protocol.md`](experiment_protocol.md) — experiment design and statistical/model-selection rules.
- [`docs/COLAB_ARTIFACT_HANDOFF.md`](docs/COLAB_ARTIFACT_HANDOFF.md) — exact Colab upload/download checklist and retained artifact paths.
- [`docs/stages/`](docs/stages/README.md) — focused runbooks for stages with special operational requirements.
- `docs/annotation_guidelines.md` — class/annotation definitions, if present in the project.
- `docs/dataset_split_and_weighting.md` — benchmark split and class-weighting decisions, if present in the project.

## Citation

The benchmark dataset is based on:

> Ibrohim, M. O., & Budi, I. (2019). Multi-label hate speech and abusive language detection in Indonesian Twitter. In *Proceedings of the Third Workshop on Abusive Language Online* (pp. 46–57). Association for Computational Linguistics.

Dataset source: <https://github.com/okkyibrohim/id-multi-label-hate-speech-and-abusive-language-detection>
