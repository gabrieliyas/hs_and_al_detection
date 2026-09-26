"""
6_itft.py

Stage 6: single Intermediate Task Fine-Tuning (ITFT) run on Ibrohim & Budi
(Phang et al., 2018 framing -- see docs/decisions.md). Produces the ONE
checkpoint that every Experiment II (C0-C4) run fine-tunes further, per
docs/experiment_protocol.md Section 2 / Section 5.

*** Needs a GPU (Colab). Same between-stage handling as Stage 4: get
    data/processed/ib_train.csv and ib_val.csv onto Colab first (see the
    chat response), and copy models/E2_itft_context/ back afterward. This
    checkpoint is a full model (several hundred MB) -- for a file this
    size, mounting Google Drive in the Colab session and copying it there
    is more practical than a browser download. ***

Unlike Stage 4 (5 seeds, no saved checkpoints -- see that script's
docstring for why), this is a SINGLE run and the checkpoint IS the
deliverable:
  - `docs/experiment_protocol.md` Section 2 calls this "the single
    highest-leverage artifact in the pipeline" -- a poorly-selected
    checkpoint here would quietly degrade every downstream Experiment II
    run from a shared root cause. Validation-based selection
    (`load_best_model_at_end`, `metric_for_best_model="macro_f1"`) is
    therefore used exactly as it is for Stage 4.
  - `SEED = 42` and `MAX_LENGTH = 128` are carried over from Stage 4's
    assumptions (single-tweet input, same text distribution as Ibrohim &
    Budi) -- stated here as the same assumption, not a new one.

The test-set macro F1 logged at the end is a DIAGNOSTIC ONLY, confirming
the ITFT checkpoint is a reasonable, non-degenerate model before it
becomes a shared starting point -- it is NOT an Experiment I result (that
is Stage 4/5's 5-seed IndoBERTweet-Target baseline) and must not be
reported as one.

Resumable in the coarse sense: if models/E2_itft_context/ already contains
a saved model, training is skipped entirely (safe to rerun this script
after a Colab disconnect that happened AFTER the save completed). A
disconnect DURING training just means rerunning from scratch -- at
~5-7 minutes (same dataset size as a single Stage 4 seed), that's cheap
enough not to need finer-grained resumability.

Usage (inside a cloned repo, on a GPU runtime):
    python scripts/6_itft.py

Input:
    data/processed/ib_train.csv
    data/processed/ib_val.csv
    data/processed/ib_test.csv   (diagnostic evaluation only)

Outputs:
    models/E2_itft_context/               -- saved model + tokenizer (the ITFT checkpoint)
    results/tables/itft_metrics.csv        -- val (selection) and test (diagnostic) metrics
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
from experiment_utils import ID2LABEL, LABEL2ID, compute_class_weights  # noqa: E402

TRAIN_PATH = Path("data/processed/ib_train.csv")
VAL_PATH = Path("data/processed/ib_val.csv")
TEST_PATH = Path("data/processed/ib_test.csv")
MODEL_OUT_DIR = Path("models/E2_itft_context")
TABLES_DIR = Path("results/tables")

MODEL_NAME = "indolem/indobertweet-base-uncased"
SEED = 42
MAX_LENGTH = 128
NUM_EPOCHS = 4
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
LEARNING_RATE = 2e-5


class TokenizedTextDataset(torch.utils.data.Dataset):
    """Same as in 4_train_e1_indobertweet.py -- see that script for why a
    plain torch.utils.data.Dataset is used instead of the `datasets` lib."""

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
    """Same weighted-CrossEntropyLoss Trainer subclass as Stage 4."""

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


def main() -> None:
    for p in (TRAIN_PATH, VAL_PATH, TEST_PATH):
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found. Run scripts/1_preprocess_benchmark.py first "
                "(see the chat response for how to get this data onto Colab)."
            )

    if (MODEL_OUT_DIR / "config.json").exists():
        print(f"[skip] {MODEL_OUT_DIR} already contains a saved model -- "
              "ITFT already completed. Delete it first if you want to rerun.")
        return

    if not torch.cuda.is_available():
        print(
            "WARNING: no GPU detected. Make sure the Colab runtime type is "
            "set to a GPU (Runtime > Change runtime type > GPU)."
        )

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)

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
    print(f"Class weights (train partition): "
          f"{dict(zip(ID2LABEL.values(), class_weights_tensor.tolist()))}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print("Tokenizing...")
    train_enc = tokenize_split(tokenizer, train_df["clean_text"].tolist())
    val_enc = tokenize_split(tokenizer, val_df["clean_text"].tolist())
    test_enc = tokenize_split(tokenizer, test_df["clean_text"].tolist())

    train_ds = TokenizedTextDataset(train_enc, train_labels_id)
    val_ds = TokenizedTextDataset(val_enc, val_labels_id)
    test_ds = TokenizedTextDataset(test_enc, test_labels_id)

    print(f"\n=== Training ITFT checkpoint (seed={SEED}) ===")
    set_seed(SEED)
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
            seed=SEED,
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

        val_metrics = trainer.evaluate(val_ds)
        print(f"\nBest checkpoint validation macro_f1: {val_metrics['eval_macro_f1']:.4f}")

        # --- Diagnostic only (see module docstring) -- NOT an Experiment I result ---
        test_output = trainer.predict(test_ds)
        logits = test_output.predictions
        y_pred = np.argmax(logits, axis=1)
        diag_test_accuracy = accuracy_score(test_labels_id, y_pred)
        diag_test_macro_f1 = f1_score(test_labels_id, y_pred, average="macro")
        print(f"[diagnostic only] Ibrohim & Budi test set: "
              f"accuracy={diag_test_accuracy:.4f}  macro_f1={diag_test_macro_f1:.4f} "
              "(sanity check that the checkpoint is non-degenerate -- "
              "not an Experiment I headline result)")

        # --- Save the checkpoint: this is the actual deliverable of this stage ---
        trainer.save_model(str(MODEL_OUT_DIR))
        tokenizer.save_pretrained(str(MODEL_OUT_DIR))

    elapsed = time.time() - t0
    print(f"\nITFT run complete in {elapsed/60:.1f} min. "
          f"Checkpoint saved to {MODEL_OUT_DIR}/")

    pd.DataFrame([{
        "seed": SEED,
        "val_best_macro_f1": val_metrics["eval_macro_f1"],
        "diagnostic_test_accuracy": diag_test_accuracy,
        "diagnostic_test_macro_f1": diag_test_macro_f1,
        "train_minutes": round(elapsed / 60, 2),
    }]).to_csv(TABLES_DIR / "itft_metrics.csv", index=False)

    print(f"Saved results/tables/itft_metrics.csv")
    print(
        f"\nNext: copy {MODEL_OUT_DIR}/ back to your local repo at the same "
        "path (via Google Drive is recommended for a checkpoint this size) "
        "-- Stage 8 (Experiment II C0-C4 ablation) loads its starting "
        "weights from here instead of the raw pretrained IndoBERTweet."
    )


if __name__ == "__main__":
    main()
