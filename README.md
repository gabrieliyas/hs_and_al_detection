# Conversation-Aware Hate Speech and Abusive Language Detection on X Using IndoBERTweet

Undergraduate thesis project: hate speech and abusive-language detection on Indonesian X (formerly Twitter), with a focus on conversational context.

## Repository structure

```text
data/                  raw and processed datasets (gitignored)
docs/                  methodology and experiment documentation
extension/             browser-extension demo (backend + frontend)
models/                research checkpoints and final comparison models (gitignored)
results/               generated predictions, tables, figures, and matrices
scripts/               finalized, reproducible pipeline scripts
experiment_utils.py    experiment-ID registry, label mapping, evaluation helpers
dataset_collection_targets.py
                       per-cell annotation collection targets and progress tracker
README.md
requirements.txt
```

### Model and result organization

- `models/final/` contains the final models produced after the experimental configuration and training strategy have been selected.
- `results/raw/` contains per-run predictions and probability outputs used for evaluation.
- `results/tables/` contains aggregated metrics and statistical results.
- `results/figures/` contains generated plots and visualizations.
- `results/confusion_matrices/` contains confusion-matrix outputs.

The exact checkpoint, seed, fold, and evaluation procedures are documented in `docs/experiment_protocol.md`.

## Experiments

### Experiment I -- Target-only benchmark

SVM+TF-IDF and IndoBERTweet are trained and evaluated on the Ibrohim & Budi (2019) benchmark dataset using the target post only, without conversational context.

The IndoBERTweet experiment uses multiple random seeds on the same fixed train/validation/test split to measure training stability. The SVM baseline is trained as a deterministic reference.

### Experiment II -- Context-aware with Intermediate Task Fine-Tuning

IndoBERTweet is first fine-tuned on the Ibrohim & Budi benchmark dataset and then further fine-tuned on the custom contextual dataset.

Conversational context is evaluated through a C0-C4 context-window ablation using stratified five-fold cross-validation and multiple random seeds. The main ablation analysis uses the same 900-example complete-context subset across all context-window sizes.

The optimal context configuration (`C_best`) is selected from the cross-validation results. A separate availability-aware generalization analysis is then performed on the shallow pool and does not determine `C_best`.

### Experiment III -- Context-aware without Intermediate Task Fine-Tuning

IndoBERTweet is fine-tuned directly on the custom contextual dataset without the intermediate task fine-tuning stage. The selected context configuration is compared with the corresponding ITFT approach using the same fold assignments.

## Experimental pipeline

The overall workflow is:

```text
Data preprocessing
        |
        v
Experiment I: target-only benchmark
        |
        v
Intermediate Task Fine-Tuning (ITFT)
        |
        v
Experiment II-A: C0-C4 context ablation
        |
        v
Select C_best from cross-validation results
        |
        +----> Experiment II-B: availability-aware generalization
        |
        v
Experiment III: ITFT vs. direct fine-tuning
        |
        v
Select final context/training strategy
        |
        v
Train final model(s) on all 1,800 contextual posts
        |
        v
Browser-extension integration
```

Detailed execution decisions, including seed counts, fold assignment, checkpoint selection, timing, significance testing, and generalization analysis, are documented in `docs/experiment_protocol.md`.

## Data

`data/processed/` contains regenerated processed datasets and is not versioned.

### Benchmark dataset

The benchmark dataset is the Ibrohim & Budi (2019) Indonesian hate-speech and abusive-language dataset. The preprocessing pipeline performs deduplication, cleaning, three-class mapping, stratified splitting, and class-weight calculation.

### Custom contextual dataset

- **Total posts:** 1,800
- **Class mapping:** `HS=1` -> Hate Speech (priority over Abusive); otherwise `Abusive=1` -> Abusive Language; otherwise -> Neither.
- **Class distribution:** Neither 600 / Abusive 600 / Hate Speech 600
- **Context size distribution (`parent_count_available`):**
  - 1 parent: 150
  - 2 parents: 300
  - 3 parents: 450
  - 4 parents: 900
- **Context needed:** 845
- **Context not needed:** 955
- **Target group scheme:** `individual` / `protected_group` / `non_protected_group` / `institution_policy_idea` / `none` / `unclear`
- **Context type taxonomy:** ambiguous reference / sarcasm-irony / coded language-euphemism / escalation / opinion-attack ambiguity / quotation-stance / other-mixed

See `docs/annotation_guidelines.md` for the complete annotation definitions and mapping rules.

## Scripts

The finalized pipeline scripts are located in `scripts/`:

```text
1_preprocess_benchmark.py
2_preprocess_contextual.py
3_train_e1_svm.py
4_train_e1_indobertweet.py
5_evaluate_e1.py
6_itft.py
7_timing_pilot.py
8_train_e2_ablation.py
9_train_e2_retrain_full.py
10_e2b_generalization.py
11_train_e3_context_only.py
12_evaluate_all.py
13_error_analysis.py
14_final_deployment_model.py
```

Scripts are designed to run from either a local environment or a Google Colab session after cloning the repository.

Example:

```bash
python scripts/1_preprocess_benchmark.py
```

or in Colab:

```python
!python scripts/1_preprocess_benchmark.py
```

## Documentation

- `docs/annotation_guidelines.md` -- annotation definitions and labelling rules
- `docs/dataset_split_and_weighting.md` -- dataset splitting and class-weighting decisions
- `docs/experiment_protocol.md` -- execution protocol for Experiments I, II, and III

## Citation

This project's primary benchmark dataset is from:

> Ibrohim, M. O., & Budi, I. (2019). Multi-label hate speech and abusive language detection in Indonesian Twitter. In *Proceedings of the Third Workshop on Abusive Language Online* (pp. 46-57). Association for Computational Linguistics.

Dataset source: https://github.com/okkyibrohim/id-multi-label-hate-speech-and-abusive-language-detection

## Setup

```bash
pip install -r requirements.txt
```