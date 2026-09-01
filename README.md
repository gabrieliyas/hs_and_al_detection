# Conversational Context-Aware Transformer for Hate Speech and Abusive Language Detection on Social Media

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

`data/raw/` and `data/processed/` are gitignored:
- Regenerable: processed files are produced by `scripts/1_preprocess.py` from
  the raw data and are not versioned to keep the repository lean.
- Access: the primary dataset (Ibrohim & Budi, 2019) is publicly available at
  the source below. The custom contextual dataset, collected for this thesis,
  is not published in this public repository due to platform Terms of Service
  considerations around bulk redistribution of scraped content; it is
  available to the examination committee through the thesis submission, and
  to bona fide researchers on request.

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
