"""
4_train_e1_indobertweet.py

Stage 4: train the Experiment I IndoBERTweet-Target baseline
(E1-IndoBERTweet-Target), 5 random seeds, on GPU.

*** This stage needs a GPU (Google Colab). See the chat response for the
    exact between-stage steps (getting data/processed/ib_*.csv onto Colab,
    and getting this script's outputs back onto your local machine before
    running Stage 5). ***

Design decisions carried over from docs/experiment_protocol.md Section 1:
  - 5 seeds, varying classification-head init and data shuffling, on the
    single fixed 70/15/15 split from Stage 1.
  - Class weights from experiment_utils.compute_class_weights(), applied
    via weighted CrossEntropyLoss (WeightedLossTrainer below).

Two things below are NOT in docs/experiment_protocol.md or decisions.md --
stated here as assumptions made for this baseline, not as prior decisions:
  - `max_length=128`: Experiment I tokenizes a single tweet (no
    conversational context), so the empirical max_length pilot specified
    for Experiments II/III (which concatenate up to 4 parent messages) is
    not needed here -- 128 subword tokens comfortably covers a single
    Indonesian tweet. If truncation turns out to be non-negligible, rerun
    with a higher value and note the change.
  - Validation-based checkpoint selection (`load_best_model_at_end`,
    `metric_for_best_model="macro_f1"`) is applied here for the same
    reason it's specified for the ITFT stage (Stage 6): it's ordinary,
    non-leaky use of a validation split, and keeps checkpoint-selection
    methodology consistent across every transformer training stage in
    this pipeline.

Per-seed IndoBERTweet checkpoints are deliberately NOT saved: per the
confirmed pipeline order (docs/experiment_protocol.md Section 5), ITFT
(Stage 6) fine-tunes IndoBERTweet from the base pretrained checkpoint
directly, not from this stage's output -- so keeping 5 full checkpoints
here (multi-GB) would cost real Colab disk/transfer for no downstream use.
Only predictions, probabilities and metrics are kept. If you want one
seed's checkpoint for the write-up, uncomment the `trainer.save_model(...)`
line marked below.

Resumable: if a seed's output files already exist, that seed is skipped --
safe to rerun this script after a Colab disconnect partway through the
5-seed sweep.

Usage (inside a cloned repo, on a GPU runtime):
    python scripts/4_train_e1_indobertweet.py

Input:
    data/processed/ib_train.csv
    data/processed/ib_val.csv
    data/processed/ib_test.csv

Outputs (per seed):
    results/raw/e1_transformer_seed{K}_predictions.csv   -- row_id, y_true, y_pred
                                                             (string labels "N"/"AL"/"HS",
                                                             same convention as Stage 3's output)
    results/raw/e1_transformer_seed{K}_probs.npy         -- (n_test, 3) softmax probabilities,
                                                             columns in LABEL2ID id order (0=N,1=AL,2=HS)
    results/tables/e1_transformer_seed_metrics.csv       -- one row per seed, appended incrementally
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiment_utils import (  # noqa: E402
    EXPERIMENT1_IDS,
    ID2LABEL,
    LABEL2ID,
    compute_class_weights,
)

TRAIN_PATH = Path("data/processed/ib_train.csv")
VAL_PATH = Path("data/processed/ib_val.csv")
TEST_PATH = Path("data/processed/ib_test.csv")
RAW_RESULTS_DIR = Path("results/raw")
TABLES_DIR = Path("results/tables")

# Verify this against the IndoBERTweet citation in your own references
# (Koto et al., 2021) if in doubt -- this is the public checkpoint that
# paper released.
MODEL_NAME = "indolem/indobertweet-base-uncased"

SEEDS = [42, 123, 2024, 7, 99]
MAX_LENGTH = 128
NUM_EPOCHS = 4
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
LEARNING_RATE = 2e-5


class TokenizedTextDataset(torch.utils.data.Dataset):
    """Pre-tokenized, fixed-length dataset. Simple and sufficient at this
    scale (<13k rows total) -- avoids adding the `datasets` library as a
    dependency for something a plain torch.utils.data.Dataset covers."""

    def __init__(self, encodings: dict, labels: np.ndarray):
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(int(self.labels[idx]))
        return item


class WeightedLossTrainer(Trainer):
    """Trainer subclass applying weighted CrossEntropyLoss, per
    docs/dataset_split_and_weighting.md Section 1.4."""

    def __init__(self, *args, class_weights: torch.Tensor, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred) -> dict:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
        "weighted_f1": f1_score(labels, preds, average="weighted"),
    }


def tokenize_split(tokenizer, texts: list[str]) -> dict:
    enc = tokenizer(
        texts, truncation=True, max_length=MAX_LENGTH, padding="max_length",
        return_tensors="pt",
    )
    return dict(enc)


def seed_already_done(seed: int) -> bool:
    pred_path = RAW_RESULTS_DIR / f"e1_transformer_seed{seed}_predictions.csv"
    probs_path = RAW_RESULTS_DIR / f"e1_transformer_seed{seed}_probs.npy"
    return pred_path.exists() and probs_path.exists()


def append_seed_metrics_row(row: dict) -> None:
    metrics_path = TABLES_DIR / "e1_transformer_seed_metrics.csv"
    row_df = pd.DataFrame([row])
    if metrics_path.exists():
        existing = pd.read_csv(metrics_path)
        existing = existing[existing["seed"] != row["seed"]]  # replace if rerun
        combined = pd.concat([existing, row_df], ignore_index=True)
    else:
        combined = row_df
    combined.to_csv(metrics_path, index=False)


def main() -> None:
    for p in (TRAIN_PATH, VAL_PATH, TEST_PATH):
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found. Run scripts/1_preprocess_benchmark.py first "
                "(see the chat response for how to get this data onto Colab)."
            )

    if not torch.cuda.is_available():
        print(
            "WARNING: no GPU detected (torch.cuda.is_available() is False). "
            "This script will run on CPU, which is not practical for 5 full "
            "IndoBERTweet fine-tuning runs. Make sure the Colab runtime type "
            "is set to a GPU (Runtime > Change runtime type > GPU)."
        )

    RAW_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)
    test_df = pd.read_csv(TEST_PATH)
    print(f"Loaded train ({len(train_df)}), val ({len(val_df)}), test ({len(test_df)}).")

    train_labels_id = train_df["label"].map(LABEL2ID).to_numpy()
    val_labels_id = val_df["label"].map(LABEL2ID).to_numpy()
    test_labels_id = test_df["label"].map(LABEL2ID).to_numpy()

    class_weights_by_id = compute_class_weights(train_df["label"])
    class_weights_tensor = torch.tensor(
        [class_weights_by_id[i] for i in range(len(LABEL2ID))], dtype=torch.float
    )
    print(f"Class weights (train partition): {dict(zip(ID2LABEL.values(), class_weights_tensor.tolist()))}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print("Tokenizing (fixed-length, done once, shared across all seeds)...")
    train_enc = tokenize_split(tokenizer, train_df["clean_text"].tolist())
    val_enc = tokenize_split(tokenizer, val_df["clean_text"].tolist())
    test_enc = tokenize_split(tokenizer, test_df["clean_text"].tolist())

    train_ds = TokenizedTextDataset(train_enc, train_labels_id)
    val_ds = TokenizedTextDataset(val_enc, val_labels_id)
    test_ds = TokenizedTextDataset(test_enc, test_labels_id)

    for seed in SEEDS:
        if seed_already_done(seed):
            print(f"\n[seed {seed}] Outputs already exist -- skipping (resumable run).")
            continue

        print(f"\n=== [seed {seed}] Training {EXPERIMENT1_IDS['transformer']} ===")
        set_seed(seed)
        t0 = time.time()

        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=len(LABEL2ID), id2label=ID2LABEL, label2id=LABEL2ID,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            args = TrainingArguments(
                output_dir=tmp_dir,
                num_train_epochs=NUM_EPOCHS,
                per_device_train_batch_size=TRAIN_BATCH_SIZE,
                per_device_eval_batch_size=EVAL_BATCH_SIZE,
                learning_rate=LEARNING_RATE,
                weight_decay=0.01,
                eval_strategy="epoch",
                save_strategy="epoch",
                save_total_limit=1,
                load_best_model_at_end=True,
                metric_for_best_model="macro_f1",
                greater_is_better=True,
                logging_steps=50,
                seed=seed,
                report_to="none",
                fp16=torch.cuda.is_available(),
            )

            trainer = WeightedLossTrainer(
                model=model,
                args=args,
                train_dataset=train_ds,
                eval_dataset=val_ds,
                compute_metrics=compute_metrics,
                class_weights=class_weights_tensor,
            )

            trainer.train()

            # Uncomment to keep one seed's checkpoint (e.g. for a demo or
            # write-up figure) -- see the module docstring for why this is
            # off by default:
            # trainer.save_model(f"models/E1_baseline/e1_indobertweet_seed{seed}")

            val_metrics = trainer.evaluate(val_ds)
            print(f"[seed {seed}] Best checkpoint validation macro_f1: "
                  f"{val_metrics['eval_macro_f1']:.4f}")

            test_output = trainer.predict(test_ds)

        elapsed = time.time() - t0

        logits = test_output.predictions
        probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
        y_pred = probs.argmax(axis=1)
        y_true = test_labels_id

        test_macro_f1 = f1_score(y_true, y_pred, average="macro")
        test_weighted_f1 = f1_score(y_true, y_pred, average="weighted")
        test_accuracy = accuracy_score(y_true, y_pred)
        print(f"[seed {seed}] Test macro_f1={test_macro_f1:.4f}  "
              f"weighted_f1={test_weighted_f1:.4f}  accuracy={test_accuracy:.4f}  "
              f"(elapsed {elapsed/60:.1f} min)")

        # Saved as canonical string labels ("N"/"AL"/"HS"), matching Stage 3's
        # e1_svm_test_predictions.csv format -- Stage 5 loads both under the
        # same convention. probs.npy keeps the raw float array separately,
        # with columns in LABEL2ID id order (0=N, 1=AL, 2=HS).
        preds_df = pd.DataFrame({
            "row_id": np.arange(len(test_df)),
            "y_true": pd.Series(y_true).map(ID2LABEL),
            "y_pred": pd.Series(y_pred).map(ID2LABEL),
        })
        preds_df.to_csv(
            RAW_RESULTS_DIR / f"e1_transformer_seed{seed}_predictions.csv", index=False
        )
        np.save(RAW_RESULTS_DIR / f"e1_transformer_seed{seed}_probs.npy", probs)

        append_seed_metrics_row({
            "seed": seed,
            "val_best_macro_f1": val_metrics["eval_macro_f1"],
            "test_accuracy": test_accuracy,
            "test_macro_f1": test_macro_f1,
            "test_weighted_f1": test_weighted_f1,
            "train_minutes": round(elapsed / 60, 2),
        })

        del model, trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\nAll seeds done (or already present). Outputs are in results/raw/ "
          "and results/tables/e1_transformer_seed_metrics.csv.")
    print(
        "\nNext: download results/raw/e1_transformer_seed*_predictions.csv, "
        "results/raw/e1_transformer_seed*_probs.npy, and "
        "results/tables/e1_transformer_seed_metrics.csv back to your local "
        "repo (same paths), then run scripts/5_evaluate_e1.py locally."
    )


if __name__ == "__main__":
    main()
