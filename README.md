# Conversation-Aware Hate Speech and Abusive Language Detection on X Using IndoBERTweet

Undergraduate thesis project: hate speech and abusive-language detection on
Indonesian X (formerly Twitter), with a focus on conversational context.

## Repository structure

```
data/           raw and processed datasets (gitignored -- see Data section)
extension/      browser-extension demo (backend + frontend, DOM-based, no live API calls)
models/         trained checkpoints (gitignored -- see Data section)
notebooks/      exploration, preprocessing drafts, dataset statistics, experiment notes
results/        generated tables, figures, confusion matrices (raw predictions gitignored)
scripts/        finalized, reproducible pipeline scripts (preprocessing, training, evaluation)
experiment_utils.py           experiment-ID registry, label mapping, evaluation helpers
dataset_collection_targets.py per-cell annotation collection targets and progress tracker
annotation_guidelines.md      label definitions, context-needed criteria, tie-break protocol
```

## Experiments

- **Experiment I -- Baseline:** SVM+TF-IDF and IndoBERTweet trained/evaluated on
  the Ibrohim & Budi (2019) dataset only (target post, no conversational context).
- **Experiment II -- Context-Aware with Intermediate Task Fine-Tuning (ITFT):**
  IndoBERTweet fine-tuned on Ibrohim & Budi first, then further fine-tuned on
  the custom contextual dataset, context window ablation C0-C4.
- **Experiment III -- Context-Aware without ITFT:** IndoBERTweet fine-tuned on
  the custom contextual dataset only, skipping the intermediate fine-tuning stage.

See `experiment_utils.py` for the canonical experiment-ID naming used across
all scripts, models, and results.

## Data

`data/processed/` are gitignored:
- Regenerable: processed files are produced by `scripts/1_preprocess.py` from
  the raw data and are not versioned to keep the repository lean.
- Access: the primary dataset (Ibrohim & Budi, 2019) is publicly available at
  the source below.

### Custom contextual dataset

- **Total posts:** 1,800
- **Class mapping:** `HS=1` -> Hate Speech (priority over Abusive); else
  `Abusive=1` -> Abusive Language; else -> Neither. See
  `annotation_guidelines.md` for full definitions and the priority-rule
  justification.
- **Class distribution:** Neither 600 / Abusive 600 / Hate Speech 600
- **Context size distribution (by `parent_count_available`):**
  1-parent 150 / 2-parent 300 / 3-parent 450 / 4-parent 900
- **Context-needed vs. not-needed:**
  - Context needed: 845 total (Neither 180 / Abusive 300 / Hate Speech 365)
  - Context not needed: 955 total (Neither 420 / Abusive 300 / Hate Speech 235)
- **Target group scheme:** `individual` / `protected_group` /
  `non_protected_group` / `institution_policy_idea` / `none` / `unclear`
  (single-annotator, descriptive field -- see `annotation_guidelines.md`)
- **Context type taxonomy:** ambiguous reference / sarcasm-irony / coded
  language-euphemism / escalation / opinion-attack ambiguity /
  quotation-stance / other-mixed, populated selectively for
  `context_needed=1` posts. Posts with `context_needed=0` are coded as
  *explicit* by definition -- see `annotation_guidelines.md`.


Full breakdown: see `results/tables/` for the final three-way
(parent_count x context_needed x label) distribution table once generated.

## Citation

This project's primary benchmark dataset is from:

> Ibrohim, M. O., & Budi, I. (2019). Multi-label hate speech and abusive
> language detection in Indonesian Twitter. In *Proceedings of the Third
> Workshop on Abusive Language Online* (pp. 46-57). Association for
> Computational Linguistics.

Dataset source: https://github.com/okkyibrohim/id-multi-label-hate-speech-and-abusive-language-detection

## Setup

```bash
pip install -r requirements.txt
```

Training scripts are designed to run identically from a local machine or a
Colab session (`git clone` the repo, then `!python scripts/<script>.py ...`).
