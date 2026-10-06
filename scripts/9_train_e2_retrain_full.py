"""
9_train_e2_retrain_full.py

Stage 9: Experiment II model-selection and full matched-subset retraining.

Purpose:
  1. Validate the completed Experiment II-A CV grid.
  2. Select the best context size from the 5-fold x 2-seed CV results
     using mean macro-F1 only.
  3. Retrain C0-C4 once each on the FULL 900-example matched subset,
     starting from the Stage 6 ITFT checkpoint.
  4. Save those five retrained models for Stage 10's availability-aware
     generalization analysis.

Important:
  - This stage does NOT select C_best from the shallow pool.
  - This stage does NOT produce the final deployment models.
  - The five saved models are experimental/retraining artefacts for Stage 10.
  - No held-out fold exists at this stage, so there is no validation-based
    epoch selection. All five runs use the fixed Experiment II training
    configuration established in Stage 8 (8 epochs, max_length=152).
  - A fixed seed of 42 is used for this single retraining run per C-level.
    This makes the five Stage 9 models deterministic and keeps the
    retraining budget at exactly five runs, as specified by the protocol.

Inputs:
  data/processed/ctx_matched_subset.csv
      The complete-case 900-example dataset with the pre-assigned `fold`
      column. All 900 examples are used for each C-level.
  models/E2_itft_context/
      Stage 6 ITFT checkpoint and tokenizer.
  results/tables/e2_grid_metrics.csv
      Complete 50-run Experiment II-A results table.

Outputs:
  results/tables/e2_best_context.csv
      Concrete C-level selected from Experiment II CV results.
  results/tables/e2_full_retrain_metrics.csv
      One row per C0-C4 retraining run with training configuration and
      elapsed time.
  models/E2_full_retrained/C0/
  models/E2_full_retrained/C1/
  models/E2_full_retrained/C2/
  models/E2_full_retrained/C3/
  models/E2_full_retrained/C4/
      Standalone Hugging Face model + tokenizer directories for Stage 10.

The model checkpoints are intentionally not committed to Git; they are
regenerable and should be transferred between local/Colab environments
through Google Drive or another persistent storage mechanism.
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
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
    select_best_context_from_cv,
)

MATCHED_SUBSET_PATH = Path("data/processed/ctx_matched_subset.csv")
ITFT_CHECKPOINT_DIR = Path("models/E2_itft_context")
RETRAIN_ROOT_DIR = Path("models/E2_full_retrained")
GRID_METRICS_PATH = Path("results/tables/e2_grid_metrics.csv")
TABLES_DIR = Path("results/tables")
SELECTION_PATH = TABLES_DIR / "e2_best_context.csv"
RETRAIN_METRICS_PATH = TABLES_DIR / "e2_full_retrain_metrics.csv"

CONDITIONS = [0, 1, 2, 3, 4]
EXPECTED_FOLDS = [0, 1, 2, 3, 4]
EXPECTED_SEEDS = [42, 123]

# Fixed once for Stage 9 because there is intentionally one retraining run
# per condition. Using seed 42 provides deterministic, reproducible models.
RETRAIN_SEED = 42

# Same configuration as Stage 8. Stage 9 has no held-out validation split,
# so the epoch count is fixed rather than selected from a held-out set.
MAX_LENGTH = 152
NUM_EPOCHS = 8
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
LEARNING_RATE = 2e-5


class TokenizedTextDataset(torch.utils.data.Dataset):
    """Simple pre-tokenized dataset compatible with Hugging Face Trainer."""

    def __init__(self, encodings: dict, labels: np.ndarray):
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(int(self.labels[idx]))
        return item


def validate_prerequisites() -> None:
    required_paths = (
        MATCHED_SUBSET_PATH,
        ITFT_CHECKPOINT_DIR / "config.json",
        GRID_METRICS_PATH,
    )
    missing = [str(p) for p in required_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Stage 9 prerequisites are missing:\n  - "
            + "\n  - ".join(missing)
        )

    # Model weights may be saved as safe tensors or legacy PyTorch weights.
    model_weights = (
        ITFT_CHECKPOINT_DIR / "model.safetensors",
        ITFT_CHECKPOINT_DIR / "pytorch_model.bin",
    )
    if not any(p.exists() for p in model_weights):
        raise FileNotFoundError(
            f"{ITFT_CHECKPOINT_DIR} contains config.json but no model weights "
            "(expected model.safetensors or pytorch_model.bin)."
        )

    if not torch.cuda.is_available():
        print(
            "WARNING: no GPU detected. This stage is intended for a Colab GPU "
            "runtime and will be slow on CPU."
        )


def load_and_validate_grid() -> pd.DataFrame:
    grid = pd.read_csv(GRID_METRICS_PATH)

    required_columns = {
        "condition",
        "fold",
        "seed",
        "n_held_out",
        "macro_f1",
    }
    missing = sorted(required_columns - set(grid.columns))
    if missing:
        raise KeyError(
            f"{GRID_METRICS_PATH} is missing required columns: {missing}"
        )

    grid = grid.copy()
    grid["condition"] = grid["condition"].astype(int)
    grid["fold"] = grid["fold"].astype(int)
    grid["seed"] = grid["seed"].astype(int)

    expected = {
        (c, f, s)
        for c in CONDITIONS
        for f in EXPECTED_FOLDS
        for s in EXPECTED_SEEDS
    }
    observed = set(
        zip(grid["condition"], grid["fold"], grid["seed"])
    )

    duplicates = grid.duplicated(
        subset=["condition", "fold", "seed"], keep=False
    )
    if duplicates.any():
        dup_rows = grid.loc[
            duplicates, ["condition", "fold", "seed"]
        ].drop_duplicates()
        raise ValueError(
            "Duplicate Experiment II run entries detected in "
            f"{GRID_METRICS_PATH}:\n{dup_rows.to_string(index=False)}"
        )

    missing_runs = sorted(expected - observed)
    unexpected_runs = sorted(observed - expected)

    if missing_runs or unexpected_runs or len(grid) != len(expected):
        message = [
            f"Expected exactly {len(expected)} unique runs "
            f"(5 conditions x 5 folds x 2 seeds), found {len(grid)}."
        ]
        if missing_runs:
            message.append(f"Missing runs: {missing_runs}")
        if unexpected_runs:
            message.append(f"Unexpected runs: {unexpected_runs}")
        raise ValueError("\n".join(message))

    if not np.isfinite(grid["macro_f1"]).all():
        raise ValueError("e2_grid_metrics.csv contains non-finite macro-F1 values.")

    # Each run should evaluate 180 held-out examples in the 900-example
    # matched subset under the current 5-fold design.
    if not (grid["n_held_out"] == 180).all():
        bad = grid.loc[grid["n_held_out"] != 180]
        raise ValueError(
            "Unexpected held-out sample counts found. Stage 8 should have "
            "180 held-out examples per fold:\n"
            f"{bad[['condition', 'fold', 'seed', 'n_held_out']].to_string(index=False)}"
        )

    return grid


def write_context_selection(
    best_n: int,
    e2_id: str,
    e3_id: str,
    summary: pd.DataFrame,
) -> None:
    selected = summary.copy()
    selected["experiment2_id"] = selected["condition"].map(
        lambda n: f"E2-IndoBERTweet-ITFT-C{int(n)}"
    )
    selected["selected"] = selected["condition"].eq(best_n)

    selected.to_csv(
        TABLES_DIR / "e2_context_ablation_summary.csv",
        index=False,
    )

    best_row = selected.loc[selected["condition"] == best_n].iloc[0]

    pd.DataFrame(
        [
            {
                "selected_context": int(best_n),
                "experiment2_id": e2_id,
                "corresponding_experiment3_id": e3_id,
                "selection_metric": "macro_f1",
                "selection_basis": "mean across 5 folds x 2 seeds",
                "selected_mean_macro_f1": float(best_row["mean"]),
                "selected_std_macro_f1": float(best_row["std"]),
                "tie_break": "smaller context window if exact mean tie",
                "n_cv_runs_per_condition": 10,
            }
        ]
    ).to_csv(SELECTION_PATH, index=False)


def tokenize_condition(
    tokenizer,
    dataframe: pd.DataFrame,
    condition: int,
) -> TokenizedTextDataset:
    texts = dataframe.apply(
        lambda row: build_context_text(
            row,
            condition,
            sep=f" {tokenizer.sep_token} ",
        ),
        axis=1,
    ).tolist()

    encodings = tokenizer(
        texts,
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
        return_tensors="pt",
    )
    labels_id = dataframe["label"].map(LABEL2ID).to_numpy()

    return TokenizedTextDataset(dict(encodings), labels_id)


def model_is_complete(model_dir: Path) -> bool:
    has_config = (model_dir / "config.json").exists()
    has_weights = (
        (model_dir / "model.safetensors").exists()
        or (model_dir / "pytorch_model.bin").exists()
    )
    has_tokenizer = (
        (model_dir / "tokenizer.json").exists()
        or (model_dir / "tokenizer_config.json").exists()
    )
    return has_config and has_weights and has_tokenizer


def write_retrain_metrics(rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    if RETRAIN_METRICS_PATH.exists():
        existing = pd.read_csv(RETRAIN_METRICS_PATH)
        existing = existing[
            ~existing["condition"].isin(df["condition"].tolist())
        ]
        df = pd.concat([existing, df], ignore_index=True)

    df.sort_values("condition").to_csv(RETRAIN_METRICS_PATH, index=False)


def train_one_condition(
    tokenizer,
    matched: pd.DataFrame,
    condition: int,
) -> dict:
    output_dir = RETRAIN_ROOT_DIR / f"C{condition}"

    if model_is_complete(output_dir):
        print(
            f"[C{condition}] model already exists at {output_dir}; skipping."
        )
        return {
            "condition": condition,
            "experiment_id": f"E2-IndoBERTweet-ITFT-C{condition}",
            "status": "skipped_existing",
            "n_train": len(matched),
            "seed": RETRAIN_SEED,
            "max_length": MAX_LENGTH,
            "num_epochs": NUM_EPOCHS,
            "train_batch_size": TRAIN_BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "train_minutes": np.nan,
        }

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Retraining C{condition} on full matched subset ===")
    print(f"Examples: {len(matched)} | seed={RETRAIN_SEED}")

    set_seed(RETRAIN_SEED)
    t0 = time.time()

    model = AutoModelForSequenceClassification.from_pretrained(
        str(ITFT_CHECKPOINT_DIR),
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    train_ds = tokenize_condition(tokenizer, matched, condition)

    with tempfile.TemporaryDirectory() as tmp_dir:
        args = TrainingArguments(
            output_dir=tmp_dir,
            num_train_epochs=NUM_EPOCHS,
            per_device_train_batch_size=TRAIN_BATCH_SIZE,
            per_device_eval_batch_size=EVAL_BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            weight_decay=0.01,
            eval_strategy="no",
            save_strategy="no",
            logging_steps=50,
            seed=RETRAIN_SEED,
            data_seed=RETRAIN_SEED,
            report_to="none",
            fp16=torch.cuda.is_available(),
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=None,
        )

        trainer.train()
        trainer.save_model(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))

    elapsed = time.time() - t0

    metadata = {
        "stage": 9,
        "experiment_id": f"E2-IndoBERTweet-ITFT-C{condition}",
        "condition": condition,
        "source_checkpoint": str(ITFT_CHECKPOINT_DIR),
        "training_dataset": str(MATCHED_SUBSET_PATH),
        "n_train": int(len(matched)),
        "seed": RETRAIN_SEED,
        "max_length": MAX_LENGTH,
        "num_epochs": NUM_EPOCHS,
        "train_batch_size": TRAIN_BATCH_SIZE,
        "eval_batch_size": EVAL_BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": 0.01,
        "validation_selection": False,
        "purpose": "Stage 10 availability-aware generalization",
        "train_minutes": round(elapsed / 60, 2),
    }
    (output_dir / "stage9_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    del model, trainer, train_ds
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(
        f"[C{condition}] saved to {output_dir} "
        f"({elapsed / 60:.2f} min)"
    )

    return {
        "condition": condition,
        "experiment_id": f"E2-IndoBERTweet-ITFT-C{condition}",
        "status": "trained",
        "n_train": len(matched),
        "seed": RETRAIN_SEED,
        "max_length": MAX_LENGTH,
        "num_epochs": NUM_EPOCHS,
        "train_batch_size": TRAIN_BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "train_minutes": round(elapsed / 60, 2),
    }


def main() -> None:
    validate_prerequisites()

    matched = pd.read_csv(MATCHED_SUBSET_PATH)

    if len(matched) != 900:
        raise ValueError(
            f"Expected exactly 900 rows in {MATCHED_SUBSET_PATH}; "
            f"found {len(matched)}."
        )

    if "fold" not in matched.columns:
        raise KeyError(
            f"{MATCHED_SUBSET_PATH} must contain the shared `fold` column."
        )

    if set(matched["fold"].astype(int).unique()) != set(EXPECTED_FOLDS):
        raise ValueError(
            "The matched subset does not contain the expected fold labels "
            f"{EXPECTED_FOLDS}."
        )

    grid = load_and_validate_grid()

    best_n, e2_id, e3_id, summary = select_best_context_from_cv(
        grid,
        metric="macro_f1",
    )
    write_context_selection(best_n, e2_id, e3_id, summary)

    print("\n=== Experiment II context selection ===")
    print(summary.to_string(index=False))
    print(
        f"\nSelected context: C{best_n} "
        f"({e2_id}) with mean macro-F1={summary.loc[summary['condition'] == best_n, 'mean'].iloc[0]:.4f}"
    )
    print(
        "The corresponding Experiment III condition is "
        f"{e3_id}. This stage does not use shallow-pool results."
    )

    tokenizer = AutoTokenizer.from_pretrained(str(ITFT_CHECKPOINT_DIR))
    RETRAIN_ROOT_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for condition in CONDITIONS:
        rows.append(
            train_one_condition(
                tokenizer=tokenizer,
                matched=matched,
                condition=condition,
            )
        )
        write_retrain_metrics(rows)

    retrained_dirs = [
        RETRAIN_ROOT_DIR / f"C{condition}"
        for condition in CONDITIONS
    ]
    missing_dirs = [
        str(path) for path in retrained_dirs
        if not model_is_complete(path)
    ]
    if missing_dirs:
        raise RuntimeError(
            "Stage 9 did not finish all five retrained models. "
            f"Missing/incomplete directories: {missing_dirs}"
        )

    print("\nStage 9 complete.")
    print(f"Selected context table: {SELECTION_PATH}")
    print(f"CV summary table: {TABLES_DIR / 'e2_context_ablation_summary.csv'}")
    print(f"Retraining metrics: {RETRAIN_METRICS_PATH}")
    print(f"Models: {RETRAIN_ROOT_DIR}/C0 ... C4")
    print(
        "\nNext: keep/copy models/E2_full_retrained/ and the Stage 9 result "
        "tables into the environment where Stage 10 will run."
    )


if __name__ == "__main__":
    main()
