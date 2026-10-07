"""
11_train_e3_context_only.py

Stage 11: Experiment III -- direct fine-tuning WITHOUT ITFT.

Design
------
Train IndoBERTweet directly from the public pretrained checkpoint on the custom
900-example matched subset, using the SAME fold assignments as Experiment II.

Only the target-only condition (C0) and the selected context condition
(C_best, selected from Stage 8/9) are evaluated here. With the current Stage 8/9
result, C_best is C4.

This stage is a methodological comparison:
    Experiment II: ITFT checkpoint -> contextual fine-tuning
    Experiment III: base IndoBERTweet -> contextual fine-tuning

The held-out fold is NEVER used for epoch/model selection. Each run uses a fixed
8-epoch training schedule and is evaluated once on its held-out fold after
training. This keeps the E2 vs E3 comparison free from test-fold checkpoint
selection.

Runs
----
    2 conditions (C0, C_best)
    x 5 shared folds
    x 2 seeds (42, 123)
    = 20 runs

Inputs
------
Required:
    data/processed/ctx_matched_subset.csv
    experiment_utils.py
    results/tables/e2_best_context.csv
    scripts/11_train_e3_context_only.py

For direct E2-vs-E3 summary:
    results/tables/e2_grid_metrics.csv

The base IndoBERTweet checkpoint/tokenizer
    indolem/indobertweet-base-uncased
is downloaded by Transformers when needed; no local model checkpoint is
required for Experiment III.

Outputs
-------
Per-run raw predictions:
    results/raw/e3_C{c}_fold{f}_seed{s}_predictions.csv
    results/raw/e3_C{c}_fold{f}_seed{s}_probs.npy

Summary tables:
    results/tables/e3_grid_metrics.csv
        One row per run (20 rows).
    results/tables/e3_context_summary.csv
        Mean/std across the 10 CV runs for C0 and C_best.
    results/tables/e3_grid_class_metrics.csv
        Per-run, per-class precision/recall/F1/support.
    results/tables/e3_confusion_matrices.csv
        Per-run confusion matrices in long format for C0 and C_best.
    results/tables/e3_vs_e2_run_comparison.csv
        Same condition/fold/seed matched metrics from E2 vs E3.
    results/tables/e3_vs_e2_summary.csv
        Mean/std performance and direct-minus-ITFT deltas by condition.
    results/tables/e3_manifest.csv
        Experiment configuration, selected C_best, run counts, and provenance.

C_best is never re-selected in Stage 11. The selected context comes from
Stage 8/9's CV result in e2_best_context.csv.
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiment_utils import (  # noqa: E402
    CTX_ID_COLUMN,
    CTX_FOLD_COLUMN,
    ID2LABEL,
    LABEL2ID,
    build_context_text,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MATCHED_SUBSET_PATH = ROOT / "data/processed/ctx_matched_subset.csv"
BEST_CONTEXT_PATH = ROOT / "results/tables/e2_best_context.csv"
E2_GRID_METRICS_PATH = ROOT / "results/tables/e2_grid_metrics.csv"

RAW_RESULTS_DIR = ROOT / "results/raw"
TABLES_DIR = ROOT / "results/tables"

# ---------------------------------------------------------------------------
# Experiment III configuration
# ---------------------------------------------------------------------------
MODEL_NAME = "indolem/indobertweet-base-uncased"

# Same as Experiment II so that E2 vs E3 differs in training strategy, not
# context construction, fold design, or core hyperparameters.
MAX_LENGTH = 152
NUM_EPOCHS = 8
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01

SEEDS_E3 = [42, 123]
FOLDS = [0, 1, 2, 3, 4]

BASE_CONDITION = 0


class TokenizedTextDataset(torch.utils.data.Dataset):
    """Minimal Dataset wrapper around pre-tokenized tensors."""

    def __init__(self, encodings: dict, labels: np.ndarray):
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(int(self.labels[idx]))
        return item


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def validate_best_context() -> int:
    require_file(BEST_CONTEXT_PATH, "Stage 9 best-context table")
    df = pd.read_csv(BEST_CONTEXT_PATH)

    required = {
        "selected_context",
        "experiment2_id",
        "selection_metric",
        "selection_basis",
        "selected_mean_macro_f1",
        "n_cv_runs_per_condition",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"e2_best_context.csv is missing column(s): {missing}")

    if len(df) != 1:
        raise ValueError(
            f"e2_best_context.csv must contain exactly one selected row; found {len(df)}."
        )

    best_context = int(df.iloc[0]["selected_context"])
    if best_context not in {0, 1, 2, 3, 4}:
        raise ValueError(f"selected_context must be in 0..4, got {best_context}")

    if str(df.iloc[0]["selection_metric"]) != "macro_f1":
        raise ValueError(
            "Stage 11 expects C_best to come from Stage 8/9 mean macro-F1 selection."
        )

    if int(df.iloc[0]["n_cv_runs_per_condition"]) != 10:
        raise ValueError(
            "Expected 10 E2 CV runs per condition (5 folds x 2 seeds), "
            f"got {df.iloc[0]['n_cv_runs_per_condition']}."
        )

    expected_id = f"E2-IndoBERTweet-ITFT-C{best_context}"
    if str(df.iloc[0]["experiment2_id"]) != expected_id:
        raise ValueError(
            f"Best-context experiment ID {df.iloc[0]['experiment2_id']!r} "
            f"does not match selected C{best_context}."
        )

    return best_context


def validate_matched_subset(df: pd.DataFrame) -> pd.DataFrame:
    required = {
        CTX_ID_COLUMN,
        "parent_4",
        "parent_3",
        "parent_2",
        "parent_1",
        "target_text",
        "context_size_available",
        "label",
        CTX_FOLD_COLUMN,
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(
            f"ctx_matched_subset.csv is missing required column(s): {missing}"
        )

    if len(df) != 900:
        raise AssertionError(
            f"Expected 900 matched-subset examples, found {len(df)}."
        )

    if not (df["context_size_available"].astype(int) == 4).all():
        raise AssertionError(
            "Matched subset must contain only context_size_available == 4."
        )

    if df[CTX_ID_COLUMN].duplicated().any():
        raise ValueError("Duplicate post_id values found in matched subset.")

    if sorted(df[CTX_FOLD_COLUMN].astype(int).unique().tolist()) != FOLDS:
        raise ValueError(
            f"Expected fold assignments {FOLDS}; found "
            f"{sorted(df[CTX_FOLD_COLUMN].astype(int).unique().tolist())}."
        )

    fold_counts = df[CTX_FOLD_COLUMN].astype(int).value_counts().sort_index().to_dict()
    if set(fold_counts.values()) != {180}:
        raise AssertionError(
            f"Each fold should contain 180 held-out examples; got {fold_counts}."
        )

    bad_labels = sorted(set(df["label"].dropna()) - set(LABEL2ID))
    if bad_labels:
        raise ValueError(
            f"Unexpected labels in matched subset: {bad_labels}"
        )

    return df.copy()


def build_texts(
    tokenizer,
    df: pd.DataFrame,
    condition: int,
) -> list[str]:
    return df.apply(
        lambda r: build_context_text(
            r,
            condition,
            sep=f" {tokenizer.sep_token} ",
        ),
        axis=1,
    ).tolist()


def tokenize_dataframe(
    tokenizer,
    df: pd.DataFrame,
    condition: int,
) -> tuple[TokenizedTextDataset, np.ndarray]:
    texts = build_texts(tokenizer, df, condition)

    encodings = tokenizer(
        texts,
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
        return_tensors="pt",
    )

    labels_id = df["label"].map(LABEL2ID).astype(int).to_numpy()

    return TokenizedTextDataset(dict(encodings), labels_id), labels_id


def compute_metrics(eval_pred) -> dict:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
        "weighted_f1": f1_score(labels, preds, average="weighted"),
    }


def run_paths(condition: int, fold: int, seed: int) -> tuple[Path, Path]:
    stem = f"e3_C{condition}_fold{fold}_seed{seed}"
    return (
        RAW_RESULTS_DIR / f"{stem}_predictions.csv",
        RAW_RESULTS_DIR / f"{stem}_probs.npy",
    )


def run_already_done(condition: int, fold: int, seed: int) -> bool:
    pred_path, probs_path = run_paths(condition, fold, seed)
    return pred_path.exists() and probs_path.exists()


def upsert_grid_row(row: dict) -> None:
    path = TABLES_DIR / "e3_grid_metrics.csv"
    new = pd.DataFrame([row])

    if path.exists():
        existing = pd.read_csv(path)
        keep = ~(
            (existing["condition"] == row["condition"])
            & (existing["fold"] == row["fold"])
            & (existing["seed"] == row["seed"])
        )
        combined = pd.concat([existing.loc[keep], new], ignore_index=True)
    else:
        combined = new

    combined = combined.sort_values(
        ["condition", "fold", "seed"]
    ).reset_index(drop=True)
    combined.to_csv(path, index=False)


def write_class_metrics(
    rows: list[dict],
    condition: int,
    fold: int,
    seed: int,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    labels = list(range(len(LABEL2ID)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )

    for class_id, p, r, f, s in zip(
        labels, precision, recall, f1, support
    ):
        rows.append(
            {
                "condition": condition,
                "fold": fold,
                "seed": seed,
                "class_id": int(class_id),
                "class_label": ID2LABEL[int(class_id)],
                "precision": float(p),
                "recall": float(r),
                "f1": float(f),
                "support": int(s),
            }
        )


def append_confusion_metrics(
    rows: list[dict],
    condition: int,
    fold: int,
    seed: int,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    labels = list(range(len(LABEL2ID)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    for true_id in labels:
        for pred_id in labels:
            rows.append(
                {
                    "condition": condition,
                    "fold": fold,
                    "seed": seed,
                    "true_class_id": int(true_id),
                    "true_class": ID2LABEL[int(true_id)],
                    "predicted_class_id": int(pred_id),
                    "predicted_class": ID2LABEL[int(pred_id)],
                    "count": int(cm[true_id, pred_id]),
                }
            )


def write_raw_predictions(
    condition: int,
    fold: int,
    seed: int,
    held_out_df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
) -> None:
    pred_path, probs_path = run_paths(condition, fold, seed)

    pred_df = pd.DataFrame(
        {
            CTX_ID_COLUMN: held_out_df[CTX_ID_COLUMN].to_numpy(),
            CTX_FOLD_COLUMN: held_out_df[CTX_FOLD_COLUMN].astype(int).to_numpy(),
            "condition": condition,
            "seed": seed,
            "y_true": [ID2LABEL[int(x)] for x in y_true],
            "y_pred": [ID2LABEL[int(x)] for x in y_pred],
            "confidence": probs.max(axis=1),
        }
    )

    # Keep useful analysis-only columns in raw output, but never use them as
    # model features.
    for col in (
        "context_needed",
        "target_group",
        "target_group_attribute",
        "context_type",
        "collection_method",
        "keyword_used",
        "post_date",
    ):
        if col in held_out_df.columns and col not in pred_df.columns:
            pred_df[col] = held_out_df[col].to_numpy()

    pred_df.to_csv(pred_path, index=False)
    np.save(probs_path, probs)


def summarise_e3_results(best_context: int, class_rows: list[dict], confusion_rows: list[dict]) -> None:
    grid_path = TABLES_DIR / "e3_grid_metrics.csv"
    grid = pd.read_csv(grid_path)

    expected_conditions = {BASE_CONDITION, best_context}
    if set(grid["condition"].astype(int)) != expected_conditions:
        raise AssertionError(
            f"E3 grid conditions must be {sorted(expected_conditions)}; "
            f"found {sorted(grid['condition'].astype(int).unique())}"
        )

    for condition in expected_conditions:
        sub = grid[grid["condition"] == condition]
        expected_runs = len(FOLDS) * len(SEEDS_E3)
        if len(sub) != expected_runs:
            raise AssertionError(
                f"C{condition} should have {expected_runs} E3 runs; found {len(sub)}."
            )

    summary_rows = []
    for condition in sorted(expected_conditions):
        sub = grid[grid["condition"] == condition]

        summary_rows.append(
            {
                "condition": int(condition),
                "experiment_id": f"E3-IndoBERTweet-C{condition}",
                "n_runs": int(len(sub)),
                "accuracy_mean": float(sub["accuracy"].mean()),
                "accuracy_std": float(sub["accuracy"].std(ddof=1)),
                "macro_f1_mean": float(sub["macro_f1"].mean()),
                "macro_f1_std": float(sub["macro_f1"].std(ddof=1)),
                "weighted_f1_mean": float(sub["weighted_f1"].mean()),
                "weighted_f1_std": float(sub["weighted_f1"].std(ddof=1)),
                "is_c_best": int(condition) == int(best_context),
            }
        )

    pd.DataFrame(summary_rows).sort_values("condition").to_csv(
        TABLES_DIR / "e3_context_summary.csv",
        index=False,
    )

    class_df = pd.DataFrame(class_rows)
    if not class_df.empty:
        class_df = class_df.drop_duplicates(
            subset=["condition", "fold", "seed", "class_id"],
            keep="last",
        ).sort_values(["condition", "fold", "seed", "class_id"])
        class_df.to_csv(
            TABLES_DIR / "e3_grid_class_metrics.csv",
            index=False,
        )

    confusion_df = pd.DataFrame(confusion_rows)
    confusion_df = confusion_df.drop_duplicates(
        subset=[
            "condition",
            "fold",
            "seed",
            "true_class_id",
            "predicted_class_id",
        ],
        keep="last",
    ).sort_values(
        ["condition", "fold", "seed", "true_class_id", "predicted_class_id"]
    )
    confusion_df.to_csv(
        TABLES_DIR / "e3_confusion_matrices.csv",
        index=False,
    )


def build_e2_comparison(best_context: int) -> None:
    """
    Join E2 and E3 on condition/fold/seed.

    This compares the training strategies on exactly the same held-out examples
    for each run. It does not perform a significance test; Stage 12 can perform
    prediction-level paired analyses after all selected outputs are available.
    """
    e2 = pd.read_csv(E2_GRID_METRICS_PATH)
    e3 = pd.read_csv(TABLES_DIR / "e3_grid_metrics.csv")

    conditions = [BASE_CONDITION, best_context]
    e2 = e2[e2["condition"].astype(int).isin(conditions)].copy()
    e3 = e3[e3["condition"].astype(int).isin(conditions)].copy()

    key = ["condition", "fold", "seed"]

    expected_e2 = len(conditions) * len(FOLDS) * len(SEEDS_E3)
    expected_e3 = expected_e2

    if len(e2) != expected_e2:
        raise AssertionError(
            f"Expected {expected_e2} E2 rows for C0/C_best; found {len(e2)}."
        )

    if len(e3) != expected_e3:
        raise AssertionError(
            f"Expected {expected_e3} E3 rows; found {len(e3)}."
        )

    merged = e2.merge(
        e3,
        on=key,
        how="inner",
        suffixes=("_e2_itft", "_e3_direct"),
    )

    if len(merged) != expected_e2:
        raise AssertionError(
            "E2/E3 comparison did not form a complete one-to-one matched grid "
            "by condition/fold/seed."
        )

    merged["accuracy_delta_e3_minus_e2"] = (
        merged["accuracy_e3_direct"] - merged["accuracy_e2_itft"]
    )
    merged["macro_f1_delta_e3_minus_e2"] = (
        merged["macro_f1_e3_direct"] - merged["macro_f1_e2_itft"]
    )
    merged["weighted_f1_delta_e3_minus_e2"] = (
        merged["weighted_f1_e3_direct"] - merged["weighted_f1_e2_itft"]
    )

    comparison_cols = [
        "condition",
        "fold",
        "seed",
        "accuracy_e2_itft",
        "accuracy_e3_direct",
        "accuracy_delta_e3_minus_e2",
        "macro_f1_e2_itft",
        "macro_f1_e3_direct",
        "macro_f1_delta_e3_minus_e2",
        "weighted_f1_e2_itft",
        "weighted_f1_e3_direct",
        "weighted_f1_delta_e3_minus_e2",
    ]
    comparison = merged[comparison_cols].sort_values(key)

    comparison.to_csv(
        TABLES_DIR / "e3_vs_e2_run_comparison.csv",
        index=False,
    )

    summary_rows = []
    for condition in conditions:
        sub = comparison[comparison["condition"] == condition]

        row = {
            "condition": int(condition),
            "e2_experiment_id": f"E2-IndoBERTweet-ITFT-C{condition}",
            "e3_experiment_id": f"E3-IndoBERTweet-C{condition}",
            "n_matched_runs": int(len(sub)),
        }

        for metric in ["accuracy", "macro_f1", "weighted_f1"]:
            e2_col = f"{metric}_e2_itft"
            e3_col = f"{metric}_e3_direct"
            delta_col = f"{metric}_delta_e3_minus_e2"

            row[f"e2_{metric}_mean"] = float(sub[e2_col].mean())
            row[f"e2_{metric}_std"] = float(sub[e2_col].std(ddof=1))
            row[f"e3_{metric}_mean"] = float(sub[e3_col].mean())
            row[f"e3_{metric}_std"] = float(sub[e3_col].std(ddof=1))
            row[f"delta_{metric}_e3_minus_e2_mean"] = float(sub[delta_col].mean())
            row[f"delta_{metric}_e3_minus_e2_std"] = float(
                sub[delta_col].std(ddof=1)
            )

        row["is_c_best"] = int(condition) == int(best_context)
        summary_rows.append(row)

    pd.DataFrame(summary_rows).sort_values("condition").to_csv(
        TABLES_DIR / "e3_vs_e2_summary.csv",
        index=False,
    )


def write_manifest(best_context: int) -> None:
    grid = pd.read_csv(TABLES_DIR / "e3_grid_metrics.csv")

    rows = []
    for condition in [BASE_CONDITION, best_context]:
        sub = grid[grid["condition"] == condition]
        rows.append(
            {
                "stage": "11-E3",
                "condition": int(condition),
                "experiment_id": f"E3-IndoBERTweet-C{condition}",
                "training_strategy": "direct_fine_tuning_without_ITFT",
                "base_model": MODEL_NAME,
                "matched_subset_examples": 900,
                "fold_count": len(FOLDS),
                "seeds": ",".join(map(str, SEEDS_E3)),
                "n_runs": int(len(sub)),
                "max_length": MAX_LENGTH,
                "num_epochs": NUM_EPOCHS,
                "train_batch_size": TRAIN_BATCH_SIZE,
                "eval_batch_size": EVAL_BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "held_out_examples_per_fold": 180,
                "best_context_selected_in_stage8_9": best_context,
                "is_c_best": int(condition) == int(best_context),
                "checkpoint_selection_on_held_out_fold": False,
                "stage11_role": (
                    "direct-vs-ITFT training-strategy comparison"
                ),
            }
        )

    pd.DataFrame(rows).sort_values("condition").to_csv(
        TABLES_DIR / "e3_manifest.csv",
        index=False,
    )


def main() -> None:
    print("=== Stage 11: Experiment III -- direct fine-tuning ===")

    require_file(MATCHED_SUBSET_PATH, "Matched contextual subset")
    require_file(BEST_CONTEXT_PATH, "Stage 9 best-context table")
    require_file(E2_GRID_METRICS_PATH, "Experiment II CV grid metrics")

    if not torch.cuda.is_available():
        print(
            "WARNING: no GPU detected. Stage 11 consists of 20 "
            "fine-tuning runs and should be run on a Colab GPU."
        )
    else:
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")

    RAW_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    best_context = validate_best_context()

    if best_context == BASE_CONDITION:
        raise AssertionError(
            "C_best cannot be C0 for the intended C0 vs C_best comparison."
        )

    matched = validate_matched_subset(
        pd.read_csv(MATCHED_SUBSET_PATH)
    )

    print(f"Loaded matched subset: {len(matched)} rows")
    print(f"C_best from Stage 8/9: C{best_context}")
    print(f"Conditions evaluated in E3: C0 and C{best_context}")
    print(f"Runs: {2 * len(FOLDS) * len(SEEDS_E3)}")

    class_rows: list[dict] = []

    for condition in [BASE_CONDITION, best_context]:
        for fold in FOLDS:
            train_df = matched[matched[CTX_FOLD_COLUMN].astype(int) != fold].reset_index(
                drop=True
            )
            held_out_df = matched[
                matched[CTX_FOLD_COLUMN].astype(int) == fold
            ].reset_index(drop=True)

            if len(train_df) != 720 or len(held_out_df) != 180:
                raise AssertionError(
                    f"C{condition}, fold {fold}: expected train=720/test=180; "
                    f"got {len(train_df)}/{len(held_out_df)}."
                )

            # Tokenizer is tied to the base pretrained model. The text differs
            # by condition, so tokenization is done per condition/fold.
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

            train_ds, train_labels = tokenize_dataframe(
                tokenizer, train_df, condition
            )
            held_out_ds, held_out_labels = tokenize_dataframe(
                tokenizer, held_out_df, condition
            )

            for seed in SEEDS_E3:
                if run_already_done(condition, fold, seed):
                    print(
                        f"[C{condition} fold={fold} seed={seed}] "
                        "raw outputs already exist -- skipping training."
                    )
                    continue

                run_t0 = time.time()
                print(
                    f"\n=== E3 C{condition} | fold {fold} | seed {seed} ==="
                )
                print(
                    "Training directly from the base IndoBERTweet checkpoint; "
                    "the held-out fold is not used for model selection."
                )

                set_seed(seed)

                model = AutoModelForSequenceClassification.from_pretrained(
                    MODEL_NAME,
                    num_labels=len(LABEL2ID),
                    id2label=ID2LABEL,
                    label2id=LABEL2ID,
                )

                with tempfile.TemporaryDirectory() as tmp_dir:
                    args = TrainingArguments(
                        output_dir=tmp_dir,
                        num_train_epochs=NUM_EPOCHS,
                        per_device_train_batch_size=TRAIN_BATCH_SIZE,
                        per_device_eval_batch_size=EVAL_BATCH_SIZE,
                        learning_rate=LEARNING_RATE,
                        weight_decay=WEIGHT_DECAY,
                        save_strategy="no",
                        eval_strategy="no",
                        logging_steps=50,
                        seed=seed,
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
                    output = trainer.predict(held_out_ds)

                elapsed = time.time() - run_t0

                logits = output.predictions
                probs = torch.softmax(
                    torch.tensor(logits), dim=1
                ).numpy()
                y_pred = probs.argmax(axis=1)
                y_true = held_out_labels

                accuracy = accuracy_score(y_true, y_pred)
                macro_f1 = f1_score(
                    y_true, y_pred, average="macro"
                )
                weighted_f1 = f1_score(
                    y_true, y_pred, average="weighted"
                )

                print(
                    f"C{condition} fold={fold} seed={seed}: "
                    f"accuracy={accuracy:.4f} "
                    f"macro-F1={macro_f1:.4f} "
                    f"weighted-F1={weighted_f1:.4f} "
                    f"(elapsed={elapsed/60:.2f} min)"
                )

                write_raw_predictions(
                    condition,
                    fold,
                    seed,
                    held_out_df,
                    y_true,
                    y_pred,
                    probs,
                )

                upsert_grid_row(
                    {
                        "experiment_id": f"E3-IndoBERTweet-C{condition}",
                        "condition": condition,
                        "fold": fold,
                        "seed": seed,
                        "n_train": len(train_df),
                        "n_held_out": len(held_out_df),
                        "accuracy": accuracy,
                        "macro_f1": macro_f1,
                        "weighted_f1": weighted_f1,
                        "train_minutes": round(elapsed / 60.0, 3),
                    }
                )

                write_class_metrics(
                    class_rows,
                    condition,
                    fold,
                    seed,
                    y_true,
                    y_pred,
                )

                del trainer
                del model

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            del held_out_ds
            del train_ds
            del tokenizer

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # -----------------------------------------------------------------------
    # Summaries
    # -----------------------------------------------------------------------
    grid = pd.read_csv(TABLES_DIR / "e3_grid_metrics.csv")

    # Always reconstruct per-run class metrics from the complete raw prediction
    # set. This is necessary on resumable reruns: some runs may be skipped
    # because their raw outputs already exist, so an in-memory list would
    # otherwise contain only the newly trained runs.
    reconstructed = []
    reconstructed_confusions = []
    for condition in [BASE_CONDITION, best_context]:
        for fold in FOLDS:
            for seed in SEEDS_E3:
                pred_path, _ = run_paths(condition, fold, seed)
                if not pred_path.exists():
                    raise FileNotFoundError(
                        "Cannot reconstruct class metrics because a required "
                        f"prediction file is missing: {pred_path}"
                    )
                pred_df = pd.read_csv(pred_path)
                if len(pred_df) != 180:
                    raise AssertionError(
                        f"{pred_path} should contain 180 held-out predictions; "
                        f"found {len(pred_df)}."
                    )
                y_true = pred_df["y_true"].map(LABEL2ID).to_numpy()
                y_pred = pred_df["y_pred"].map(LABEL2ID).to_numpy()
                write_class_metrics(
                    reconstructed,
                    condition,
                    fold,
                    seed,
                    y_true,
                    y_pred,
                )
                append_confusion_metrics(
                    reconstructed_confusions,
                    condition,
                    fold,
                    seed,
                    y_true,
                    y_pred,
                )

    summarise_e3_results(
        best_context,
        reconstructed,
        reconstructed_confusions,
    )
    build_e2_comparison(best_context)
    write_manifest(best_context)

    # Sanity-check the final grid.
    expected_total = 2 * len(FOLDS) * len(SEEDS_E3)
    if len(grid) != expected_total:
        raise AssertionError(
            f"Final E3 grid should have {expected_total} rows; found {len(grid)}."
        )

    required_outputs = [
        TABLES_DIR / "e3_grid_metrics.csv",
        TABLES_DIR / "e3_context_summary.csv",
        TABLES_DIR / "e3_grid_class_metrics.csv",
        TABLES_DIR / "e3_confusion_matrices.csv",
        TABLES_DIR / "e3_vs_e2_run_comparison.csv",
        TABLES_DIR / "e3_vs_e2_summary.csv",
        TABLES_DIR / "e3_manifest.csv",
    ]
    missing_outputs = [str(p) for p in required_outputs if not p.exists()]
    if missing_outputs:
        raise RuntimeError(
            "Stage 11 completed training but required summary output(s) are "
            f"missing: {missing_outputs}"
        )

    print("\n=== Stage 11 complete ===")
    print("\nE3 CV summary:")
    print(
        pd.read_csv(TABLES_DIR / "e3_context_summary.csv").to_string(
            index=False
        )
    )

    print("\nE3 vs E2 training-strategy summary:")
    print(
        pd.read_csv(TABLES_DIR / "e3_vs_e2_summary.csv").to_string(
            index=False
        )
    )

    print("\nOutputs:")
    for path in required_outputs:
        print(f"  {path}")
    print(
        "\nRaw predictions/probabilities are stored under "
        "results/raw/ for each C/fold/seed run."
    )


if __name__ == "__main__":
    main()
