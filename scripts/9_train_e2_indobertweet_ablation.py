"""
8_train_e2_ablation.py

Stage 8: Experiment II -- the C0-C4 context-window ablation, 5-fold
stratified CV, 2 seeds, all starting from the ITFT checkpoint (Stage 6).
50 runs total (5 conditions x 5 folds x 2 seeds).

*** Needs a GPU (Colab). Run scripts/6_itft.py and scripts/7_timing_pilot.py
    first -- this script loads models/E2_itft_context/ (Stage 6) and uses
    max_length=152 (Stage 7's actual pilot result, not a guess). ***

Per docs/experiment_protocol.md Section 2's pilot result: 50 runs at
~0.41 min each + one ITFT run at 7.54 min totals ~28 min -- comfortably
inside any Colab session length, so (unlike what that section worried
about before the pilot ran) no cross-session checkpoint/resume logic is
needed. Per-run resumability is still included below, purely as cheap
insurance against an ordinary disconnect, not a session-limit workaround.

Design notes -- distinguishing what's an established decision from what's
an assumption made here:
  - Unweighted cross-entropy (plain Trainer, no custom loss): per
    docs/dataset_split_and_weighting.md Section 2.1, the custom contextual
    dataset needs no class weighting -- an established decision. (Stage 7's
    pilot applied per-fold weights anyway; that didn't affect its timing
    result, but Stage 8 follows the documented decision correctly.)
  - Text construction: experiment_utils.build_context_text(row,
    context_size), the same shared function Stage 7 used -- guarantees C0-C4
    are built identically wherever they're needed in this pipeline.
  - No per-run checkpoint selection (no `load_best_model_at_end`): with
    only a train/held-out split per fold (no third partition), selecting
    the epoch that scores best ON the held-out fold would leak into the
    very number this stage reports. Each run trains for a FIXED epoch
    count and is evaluated once, after the last epoch -- standard practice
    for a CV grid without a nested validation split. This is an assumption
    made here, not something decisions.md states explicitly.
  - `NUM_EPOCHS = 8` (double Stage 4/6/7's 4): the per-fold training set
    here is much smaller (~720 rows vs. ~8,900), so more gradient steps are
    likely needed to reach comparable convergence, and the pilot shows
    compute is not a constraint (50 runs at 8 epochs is still ~41 min).
    Stated as an assumption -- if training-loss curves in the console
    output look under- or over-fit once real runs come in, this is the
    constant to revisit.
  - `SEEDS_E2 = [42, 123]`: per docs/decisions.md ("5-fold CV, 2 seeds").
    Given how cheap each run is, more seeds costs very little extra Colab
    time -- see the chat response for that trade-off; this is the one
    constant most worth reconsidering before a long unattended run.

C_best selection from these results, and retraining C0-C4 once each on the
full 900-example matched subset, is Stage 9's job, NOT this one -- this
stage only produces the raw per-run CV predictions/metrics.

Usage (inside a cloned repo, on a GPU runtime):
    python scripts/8_train_e2_ablation.py

Input:
    data/processed/ctx_matched_subset.csv   (Stage 2 output; has the `fold` column)
    models/E2_itft_context/                 (Stage 6 output -- the ITFT checkpoint)

Outputs (per run):
    results/raw/e2_C{c}_fold{f}_seed{s}_predictions.csv   -- post_id, y_true, y_pred (string labels)
    results/raw/e2_C{c}_fold{f}_seed{s}_probs.npy         -- (n_held_out, 3) softmax probabilities,
                                                              columns in LABEL2ID id order
    results/tables/e2_grid_metrics.csv                    -- one row per run, appended incrementally
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
    CTX_ID_COLUMN,
    ID2LABEL,
    LABEL2ID,
    build_context_text,
)

MATCHED_SUBSET_PATH = Path("data/processed/ctx_matched_subset.csv")
ITFT_CHECKPOINT_DIR = Path("models/E2_itft_context")
RAW_RESULTS_DIR = Path("results/raw")
TABLES_DIR = Path("results/tables")
GRID_METRICS_PATH = TABLES_DIR / "e2_grid_metrics.csv"

CONDITIONS = [0, 1, 2, 3, 4]           # C0..C4
FOLDS = [0, 1, 2, 3, 4]
SEEDS_E2 = [42, 123]                   # see module docstring -- easy to extend

MAX_LENGTH = 152                       # Stage 7's actual pilot result (p95=146, rounded to a multiple of 8)
NUM_EPOCHS = 8                         # assumption -- see module docstring
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
LEARNING_RATE = 2e-5


class TokenizedTextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings: dict, labels: np.ndarray):
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(int(self.labels[idx]))
        return item


def compute_metrics(eval_pred) -> dict:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
        "weighted_f1": f1_score(labels, preds, average="weighted"),
    }


def tokenize_subset(tokenizer, sub_df: pd.DataFrame, condition: int):
    texts = sub_df.apply(
        lambda r: build_context_text(r, condition, sep=f" {tokenizer.sep_token} "), axis=1
    ).tolist()
    enc = tokenizer(
        texts, truncation=True, max_length=MAX_LENGTH, padding="max_length",
        return_tensors="pt",
    )
    labels_id = sub_df["label"].map(LABEL2ID).to_numpy()
    return TokenizedTextDataset(dict(enc), labels_id), labels_id


def run_files(condition: int, fold: int, seed: int) -> tuple[Path, Path]:
    stem = f"e2_C{condition}_fold{fold}_seed{seed}"
    return (RAW_RESULTS_DIR / f"{stem}_predictions.csv", RAW_RESULTS_DIR / f"{stem}_probs.npy")


def run_already_done(condition: int, fold: int, seed: int) -> bool:
    pred_path, probs_path = run_files(condition, fold, seed)
    return pred_path.exists() and probs_path.exists()


def append_grid_metrics_row(row: dict) -> None:
    row_df = pd.DataFrame([row])
    if GRID_METRICS_PATH.exists():
        existing = pd.read_csv(GRID_METRICS_PATH)
        keep = ~(
            (existing["condition"] == row["condition"])
            & (existing["fold"] == row["fold"])
            & (existing["seed"] == row["seed"])
        )
        combined = pd.concat([existing[keep], row_df], ignore_index=True)
    else:
        combined = row_df
    combined.to_csv(GRID_METRICS_PATH, index=False)


def main() -> None:
    if not MATCHED_SUBSET_PATH.exists():
        raise FileNotFoundError(
            f"{MATCHED_SUBSET_PATH} not found. Run scripts/2_preprocess_contextual.py first."
        )
    if not (ITFT_CHECKPOINT_DIR / "config.json").exists():
        raise FileNotFoundError(
            f"{ITFT_CHECKPOINT_DIR} not found or incomplete. Run scripts/6_itft.py first "
            "and copy models/E2_itft_context/ onto this machine."
        )
    if not torch.cuda.is_available():
        print("WARNING: no GPU detected. Make sure the Colab runtime type is "
              "set to a GPU (Runtime > Change runtime type > GPU).")

    RAW_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    matched = pd.read_csv(MATCHED_SUBSET_PATH)
    print(f"Loaded matched subset ({len(matched)} rows).")

    tokenizer = AutoTokenizer.from_pretrained(str(ITFT_CHECKPOINT_DIR))

    total_runs = len(CONDITIONS) * len(FOLDS) * len(SEEDS_E2)
    run_counter = 0
    grid_t0 = time.time()

    for condition in CONDITIONS:
        for fold in FOLDS:
            # Tokenized once per (condition, fold) -- reused across both
            # seeds, since the text/labels don't depend on the seed.
            train_sub = matched[matched["fold"] != fold]
            held_out_sub = matched[matched["fold"] == fold]
            train_ds, _ = tokenize_subset(tokenizer, train_sub, condition)
            held_out_ds, held_out_labels_id = tokenize_subset(tokenizer, held_out_sub, condition)
            held_out_post_ids = held_out_sub[CTX_ID_COLUMN].to_numpy()

            for seed in SEEDS_E2:
                run_counter += 1
                tag = f"[C{condition} fold{fold} seed{seed} | {run_counter}/{total_runs}]"

                if run_already_done(condition, fold, seed):
                    print(f"{tag} already done -- skipping (resumable run).")
                    continue

                print(f"\n=== {tag} ===")
                set_seed(seed)
                t0 = time.time()

                model = AutoModelForSequenceClassification.from_pretrained(
                    str(ITFT_CHECKPOINT_DIR), num_labels=len(LABEL2ID),
                    id2label=ID2LABEL, label2id=LABEL2ID,
                )

                with tempfile.TemporaryDirectory() as tmp_dir:
                    args = TrainingArguments(
                        output_dir=tmp_dir,
                        num_train_epochs=NUM_EPOCHS,
                        per_device_train_batch_size=TRAIN_BATCH_SIZE,
                        per_device_eval_batch_size=EVAL_BATCH_SIZE,
                        learning_rate=LEARNING_RATE,
                        weight_decay=0.01,
                        eval_strategy="epoch",     # logging only -- see module docstring
                        save_strategy="no",        # no per-run checkpoint kept (50 disposable runs)
                        load_best_model_at_end=False,
                        logging_steps=50,
                        seed=seed,
                        report_to="none",
                        fp16=torch.cuda.is_available(),
                    )
                    trainer = Trainer(
                        model=model,
                        args=args,
                        train_dataset=train_ds,
                        eval_dataset=held_out_ds,   # epoch-level logging only, not used for selection
                        compute_metrics=compute_metrics,
                    )
                    trainer.train()
                    held_out_output = trainer.predict(held_out_ds)  # final-epoch model, evaluated once

                elapsed = time.time() - t0
                logits = held_out_output.predictions
                probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
                y_pred = probs.argmax(axis=1)
                y_true = held_out_labels_id

                accuracy = accuracy_score(y_true, y_pred)
                macro_f1 = f1_score(y_true, y_pred, average="macro")
                weighted_f1 = f1_score(y_true, y_pred, average="weighted")
                print(f"{tag} held-out: accuracy={accuracy:.4f} macro_f1={macro_f1:.4f} "
                      f"weighted_f1={weighted_f1:.4f} ({elapsed/60:.2f} min)")

                pred_path, probs_path = run_files(condition, fold, seed)
                preds_df = pd.DataFrame({
                    CTX_ID_COLUMN: held_out_post_ids,
                    "y_true": pd.Series(y_true).map(ID2LABEL),
                    "y_pred": pd.Series(y_pred).map(ID2LABEL),
                })
                preds_df.to_csv(pred_path, index=False)
                np.save(probs_path, probs)

                append_grid_metrics_row({
                    "condition": condition,
                    "fold": fold,
                    "seed": seed,
                    "n_held_out": len(y_true),
                    "accuracy": accuracy,
                    "macro_f1": macro_f1,
                    "weighted_f1": weighted_f1,
                    "train_minutes": round(elapsed / 60, 2),
                })

                del model, trainer
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    total_elapsed = (time.time() - grid_t0) / 60
    print(f"\nGrid pass complete in {total_elapsed:.1f} min this run "
          "(runs already marked done above were skipped, not re-timed).")

    if GRID_METRICS_PATH.exists():
        results = pd.read_csv(GRID_METRICS_PATH)
        n_done = len(results)
        print(f"\n{n_done}/{total_runs} runs recorded in {GRID_METRICS_PATH}.")
        if n_done == total_runs:
            print("\nPer-condition mean macro_f1 (across folds and seeds) -- "
                  "informational only, NOT the formal C_best selection "
                  "(that's Stage 9, via experiment_utils.resolve_best_context_id):")
            summary = results.groupby("condition")["macro_f1"].agg(["mean", "std"])
            print(summary.to_string())
        else:
            print(f"Grid incomplete ({total_runs - n_done} runs remaining) -- "
                  "rerun this script to continue; already-completed runs will "
                  "be skipped.")

    print(
        "\nNext: once all 50 runs are done, copy results/raw/e2_*.csv, "
        "results/raw/e2_*.npy, and results/tables/e2_grid_metrics.csv back "
        "to your local repo -- Stage 9 selects C_best from these CV results "
        "and retrains C0-C4 once each on the full matched subset."
    )


if __name__ == "__main__":
    main()
