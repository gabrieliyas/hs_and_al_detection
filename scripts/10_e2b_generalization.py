"""
10_e2b_generalization.py

Stage 10: Experiment II-B -- availability-aware generalization analysis.

Purpose
-------
Evaluate the five Stage 9 full-retrain models on the disjoint shallow pool
(context_size_available in {1, 2, 3}) under the finalized protocol.

This is a SECONDARY, DESCRIPTIVE analysis. It does NOT select C_best and does
not override the main C0-C4 complete-case ablation.

Evaluation populations:
    C0 -> all 900 shallow-pool posts
    C1 -> all 900 shallow-pool posts (all have >=1 parent)
    C2 -> shallow-pool posts with >=2 parents (750)
    C3 -> shallow-pool posts with >=3 parents (450)
    C4 -> not applicable: no external shallow population exists

Stage 9 models used:
    models/E2_full_retrained/C0/
    ...
    models/E2_full_retrained/C4/

Input
-----
    data/processed/ctx_shallow_pool.csv
    models/E2_full_retrained/C0..C4/
    results/tables/e2_best_context.csv
    results/tables/e2_context_ablation_summary.csv
    results/tables/e2_full_retrain_metrics.csv

Outputs
-------
Raw per-model evaluation:
    results/raw/e2b_C{c}_shallow_predictions.csv
    results/raw/e2b_C{c}_shallow_probs.npy

Summary tables:
    results/tables/e2b_generalization_metrics.csv
        One row per C0-C4. C4 is retained as a not_applicable row.
    results/tables/e2b_generalization_by_availability.csv
        Per-model metrics on each exact parent-count stratum that is eligible.
        This is a descriptive breakdown, not a continuous "performance curve".
    results/tables/e2b_generalization_by_class.csv
        One row per evaluated model and class on its own eligible population.
    results/tables/e2b_generalization_confusion_matrices.csv
        Long-form confusion matrices for C0-C3.
    results/tables/e2b_generalization_pairwise_vs_c0.csv
        Independent C0-vs-C1, C0-vs-C2, and C0-vs-C3 comparisons on the same
        eligible population for each depth. C4 is not compared because there
        is no shallow C4 population.
    results/tables/e2b_generalization_manifest.csv
        Provenance, population sizes, C_best cross-check, and model settings.

No significance test is performed here because the finalized protocol defines
this branch as descriptive only. C_best must come exclusively from the
cross-validation results in Stage 8 / Stage 9's selection table.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Make root-level experiment_utils.py importable when called as:
#     python scripts/10_e2b_generalization.py
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiment_utils import (  # noqa: E402
    CTX_ID_COLUMN,
    ID2LABEL,
    LABEL2ID,
    build_context_text,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SHALLOW_POOL_PATH = ROOT / "data/processed/ctx_shallow_pool.csv"
MODELS_ROOT = ROOT / "models/E2_full_retrained"
TABLES_DIR = ROOT / "results/tables"
RAW_RESULTS_DIR = ROOT / "results/raw"

BEST_CONTEXT_PATH = TABLES_DIR / "e2_best_context.csv"
ABLATION_SUMMARY_PATH = TABLES_DIR / "e2_context_ablation_summary.csv"
FULL_RETRAIN_PATH = TABLES_DIR / "e2_full_retrain_metrics.csv"

# ---------------------------------------------------------------------------
# Finalized Stage 7/8/9 inference configuration
# ---------------------------------------------------------------------------
CONDITIONS = [0, 1, 2, 3, 4]
EVALUATED_CONDITIONS = [0, 1, 2, 3]

EXPECTED_SHALLOW_TOTAL = 900
EXPECTED_SHALLOW_BY_PARENT_COUNT = {1: 150, 2: 300, 3: 450}

MAX_LENGTH = 152
EVAL_BATCH_SIZE = 32
SEED = 42

# Expected C_best selected by Stage 8/9 for this project's current result.
# The script reads the actual CSV value and never hard-codes the value for
# downstream selection. This constant is only used as a descriptive sanity
# check if desired.
MODEL_PREFIX = "E2-IndoBERTweet-ITFT"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def require_model_dir(condition: int) -> Path:
    model_dir = MODELS_ROOT / f"C{condition}"
    require_file(model_dir / "config.json", f"E2 full-retrained C{condition} model")
    # AutoModel can use model.safetensors or pytorch_model.bin depending on
    # Transformers version / save format.
    if not (model_dir / "model.safetensors").exists() and not (
        model_dir / "pytorch_model.bin"
    ).exists():
        raise FileNotFoundError(
            f"C{condition} model directory exists but has no model.safetensors "
            f"or pytorch_model.bin: {model_dir}"
        )
    return model_dir


def check_model_metadata(condition: int, full_retrain_df: pd.DataFrame) -> dict:
    """Return the corresponding Stage 9 metadata row and validate identity."""
    row = full_retrain_df[full_retrain_df["condition"] == condition]
    if len(row) != 1:
        raise ValueError(
            f"Expected exactly one Stage 9 retrain row for C{condition}; found {len(row)}."
        )
    rec = row.iloc[0].to_dict()

    if str(rec["status"]).lower() != "trained":
        raise ValueError(
            f"Stage 9 status for C{condition} is {rec['status']!r}, not 'trained'."
        )

    expected_id = f"{MODEL_PREFIX}-C{condition}"
    if str(rec["experiment_id"]) != expected_id:
        raise ValueError(
            f"C{condition} experiment_id mismatch: got {rec['experiment_id']!r}, "
            f"expected {expected_id!r}."
        )

    return rec


def validate_best_context(best_df: pd.DataFrame) -> int:
    required = {
        "selected_context",
        "experiment2_id",
        "selection_metric",
        "selection_basis",
        "selected_mean_macro_f1",
        "n_cv_runs_per_condition",
    }
    missing = sorted(required - set(best_df.columns))
    if missing:
        raise KeyError(f"e2_best_context.csv is missing column(s): {missing}")

    if len(best_df) != 1:
        raise ValueError(
            f"e2_best_context.csv must contain exactly one selected row; found {len(best_df)}."
        )

    best_n = int(best_df.iloc[0]["selected_context"])
    if best_n not in CONDITIONS:
        raise ValueError(f"selected_context must be 0-4, got {best_n}")

    selection_metric = str(best_df.iloc[0]["selection_metric"])
    if selection_metric != "macro_f1":
        raise ValueError(
            f"Stage 9 best-context table must select by macro_f1; got {selection_metric!r}."
        )

    if int(best_df.iloc[0]["n_cv_runs_per_condition"]) != 10:
        raise ValueError(
            "Expected 10 CV runs per condition (5 folds x 2 seeds); "
            f"got {best_df.iloc[0]['n_cv_runs_per_condition']}."
        )

    expected_id = f"E2-IndoBERTweet-ITFT-C{best_n}"
    if str(best_df.iloc[0]["experiment2_id"]) != expected_id:
        raise ValueError(
            "Best-context experiment ID does not match selected_context: "
            f"{best_df.iloc[0]['experiment2_id']!r} vs {expected_id!r}."
        )

    return best_n


def validate_shallow_pool(df: pd.DataFrame) -> pd.DataFrame:
    required = {
        CTX_ID_COLUMN,
        "parent_4",
        "parent_3",
        "parent_2",
        "parent_1",
        "target_text",
        "context_size_available",
        "label",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"ctx_shallow_pool.csv is missing required column(s): {missing}")

    if len(df) != EXPECTED_SHALLOW_TOTAL:
        raise AssertionError(
            f"Shallow pool has {len(df)} rows; expected {EXPECTED_SHALLOW_TOTAL}."
        )

    counts = (
        df["context_size_available"]
        .astype(int)
        .value_counts()
        .sort_index()
        .to_dict()
    )
    expected = EXPECTED_SHALLOW_BY_PARENT_COUNT
    if counts != expected:
        raise AssertionError(
            f"Unexpected shallow-pool parent-count distribution: {counts}; "
            f"expected {expected}."
        )

    unexpected_values = sorted(
        set(df["context_size_available"].astype(int).unique()) - {1, 2, 3}
    )
    if unexpected_values:
        raise ValueError(
            f"ctx_shallow_pool.csv contains unexpected context_size_available values: "
            f"{unexpected_values}"
        )

    if df[CTX_ID_COLUMN].duplicated().any():
        dupes = df.loc[df[CTX_ID_COLUMN].duplicated(), CTX_ID_COLUMN].tolist()
        raise ValueError(f"Duplicate post_id values found in shallow pool: {dupes[:20]}")

    bad_labels = sorted(set(df["label"].dropna()) - set(LABEL2ID))
    if bad_labels:
        raise ValueError(
            f"Unexpected label value(s) in shallow pool: {bad_labels}; "
            f"expected {sorted(LABEL2ID)}."
        )

    return df.copy()


def eligible_population(shallow_df: pd.DataFrame, condition: int) -> pd.DataFrame:
    """Return the protocol-defined external population for one C-level."""
    if condition == 0:
        mask = shallow_df["context_size_available"].astype(int) >= 1
    elif condition == 1:
        mask = shallow_df["context_size_available"].astype(int) >= 1
    elif condition == 2:
        mask = shallow_df["context_size_available"].astype(int) >= 2
    elif condition == 3:
        mask = shallow_df["context_size_available"].astype(int) >= 3
    elif condition == 4:
        return shallow_df.iloc[0:0].copy()
    else:
        raise ValueError(f"condition must be 0-4, got {condition}")

    return shallow_df.loc[mask].copy().reset_index(drop=True)


def label_ids(series: pd.Series) -> np.ndarray:
    mapped = series.map(LABEL2ID)
    if mapped.isna().any():
        bad = sorted(series[mapped.isna()].astype(str).unique())
        raise ValueError(f"Could not map label values to IDs: {bad}")
    return mapped.astype(int).to_numpy()


def predict_condition(
    condition: int,
    population_df: pd.DataFrame,
    model_dir: Path,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Run deterministic inference for one retrained model.

    Returns
    -------
    preds : ndarray[int]
    probs : ndarray[float] shape (n, 3)
    elapsed_seconds : float
    """
    if len(population_df) == 0:
        return np.array([], dtype=int), np.empty((0, len(LABEL2ID))), 0.0

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    texts = population_df.apply(
        lambda r: build_context_text(
            r,
            condition,
            sep=f" {tokenizer.sep_token} ",
        ),
        axis=1,
    ).tolist()

    encoded = tokenizer(
        texts,
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
        return_tensors="pt",
    )

    total = len(population_df)
    preds_parts: list[np.ndarray] = []
    probs_parts: list[np.ndarray] = []

    t0 = time.time()
    with torch.no_grad():
        for start in range(0, total, EVAL_BATCH_SIZE):
            end = min(start + EVAL_BATCH_SIZE, total)
            batch = {
                key: value[start:end].to(device)
                for key, value in encoded.items()
            }
            outputs = model(**batch)
            probs = torch.softmax(outputs.logits, dim=-1).detach().cpu().numpy()
            preds = probs.argmax(axis=1)

            probs_parts.append(probs)
            preds_parts.append(preds)

    elapsed = time.time() - t0

    preds_all = np.concatenate(preds_parts, axis=0)
    probs_all = np.concatenate(probs_parts, axis=0)

    # Release model/tokenizer memory between C-levels.
    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return preds_all, probs_all, elapsed


def overall_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "n_examples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
    }


def class_metrics_table(
    condition: int,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    labels = list(range(len(LABEL2ID)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )
    rows = []
    for class_id, p, r, f, s in zip(labels, precision, recall, f1, support):
        rows.append(
            {
                "condition": condition,
                "class_id": class_id,
                "class_label": ID2LABEL[class_id],
                "precision": float(p),
                "recall": float(r),
                "f1": float(f),
                "support": int(s),
            }
        )
    return pd.DataFrame(rows)


def confusion_table(
    condition: int,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    labels = list(range(len(LABEL2ID)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    rows = []
    for true_id, true_name in ID2LABEL.items():
        for pred_id, pred_name in ID2LABEL.items():
            rows.append(
                {
                    "condition": condition,
                    "true_class_id": int(true_id),
                    "true_class": true_name,
                    "predicted_class_id": int(pred_id),
                    "predicted_class": pred_name,
                    "count": int(cm[true_id, pred_id]),
                }
            )
    return pd.DataFrame(rows)


def availability_breakdown(
    condition: int,
    population_df: pd.DataFrame,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    """
    Evaluate exact parent-count strata that are eligible for this model.

    These rows are intentionally a standalone breakdown and are NOT intended
    to form a single depth-response curve.
    """
    rows = []
    parent_counts = sorted(population_df["context_size_available"].astype(int).unique())

    offset = 0
    # Use boolean masks against the original population so ordering remains
    # exactly aligned with y_pred.
    for parent_count in parent_counts:
        mask = population_df["context_size_available"].astype(int).to_numpy() == parent_count
        y_true_group = label_ids(population_df.loc[mask, "label"])
        y_pred_group = y_pred[mask]
        metrics = overall_metrics(y_true_group, y_pred_group)

        rows.append(
            {
                "condition": condition,
                "parent_count_available": int(parent_count),
                "n_examples": metrics["n_examples"],
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
            }
        )
        offset += int(mask.sum())

    return pd.DataFrame(rows)


def save_predictions(
    condition: int,
    population_df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
) -> None:
    stem = f"e2b_C{condition}_shallow"
    out_csv = RAW_RESULTS_DIR / f"{stem}_predictions.csv"
    out_probs = RAW_RESULTS_DIR / f"{stem}_probs.npy"

    out = pd.DataFrame(
        {
            CTX_ID_COLUMN: population_df[CTX_ID_COLUMN].to_numpy(),
            "context_size_available": population_df["context_size_available"].astype(int).to_numpy(),
            "y_true": [ID2LABEL[int(x)] for x in y_true],
            "y_pred": [ID2LABEL[int(x)] for x in y_pred],
            "confidence": probs.max(axis=1),
        }
    )

    # Keep selected analysis-only columns in the raw prediction file when
    # present. They are never passed to the model.
    for col in (
        "context_needed",
        "target_group",
        "target_group_attribute",
        "context_type",
        "collection_method",
        "keyword_used",
        "post_date",
    ):
        if col in population_df.columns and col not in out.columns:
            out[col] = population_df[col].to_numpy()

    out.to_csv(out_csv, index=False)
    np.save(out_probs, probs)


def pairwise_vs_c0(
    shallow_df: pd.DataFrame,
    predictions: dict[int, tuple[pd.DataFrame, np.ndarray]],
) -> pd.DataFrame:
    """
    Compare C0 with each deeper condition on the SAME eligible population.

    This directly implements the protocol's standalone nested comparison idea:
      C0 vs C1 -> 900 posts
      C0 vs C2 -> 750 posts
      C0 vs C3 -> 450 posts

    No hypothesis test is run; the table is descriptive only.
    """
    rows = []

    for deeper_condition in [1, 2, 3]:
        eligible = eligible_population(shallow_df, deeper_condition)
        post_ids = eligible[CTX_ID_COLUMN].astype(str).to_numpy()

        c0_df, c0_pred = predictions[0]
        deep_df, deep_pred = predictions[deeper_condition]

        c0_index = pd.Series(np.arange(len(c0_df)), index=c0_df[CTX_ID_COLUMN].astype(str))
        deep_index = pd.Series(np.arange(len(deep_df)), index=deep_df[CTX_ID_COLUMN].astype(str))

        if not set(post_ids).issubset(c0_index.index):
            raise AssertionError(f"C0 prediction set does not cover the C{deeper_condition} population.")
        if not set(post_ids).issubset(deep_index.index):
            raise AssertionError(
                f"C{deeper_condition} prediction set does not cover its own eligible population."
            )

        c0_order = c0_index.loc[post_ids].to_numpy()
        deep_order = deep_index.loc[post_ids].to_numpy()

        y_true = label_ids(eligible["label"])
        c0_preds_on_same = c0_pred[c0_order]
        deep_preds_on_same = deep_pred[deep_order]

        m_c0 = overall_metrics(y_true, c0_preds_on_same)
        m_deep = overall_metrics(y_true, deep_preds_on_same)

        rows.append(
            {
                "comparison": f"C0_vs_C{deeper_condition}",
                "context_depth": deeper_condition,
                "population_rule": (
                    "all shallow posts (>=1 parent)"
                    if deeper_condition == 1
                    else f"shallow posts with >= {deeper_condition} parents"
                ),
                "n_examples": m_c0["n_examples"],
                "c0_accuracy": m_c0["accuracy"],
                "deeper_accuracy": m_deep["accuracy"],
                "accuracy_delta_deeper_minus_c0": m_deep["accuracy"] - m_c0["accuracy"],
                "c0_macro_f1": m_c0["macro_f1"],
                "deeper_macro_f1": m_deep["macro_f1"],
                "macro_f1_delta_deeper_minus_c0": m_deep["macro_f1"] - m_c0["macro_f1"],
                "c0_weighted_f1": m_c0["weighted_f1"],
                "deeper_weighted_f1": m_deep["weighted_f1"],
                "weighted_f1_delta_deeper_minus_c0": (
                    m_deep["weighted_f1"] - m_c0["weighted_f1"]
                ),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=== Stage 10: Experiment II-B availability-aware generalization ===")

    # ---- prerequisites -----------------------------------------------------
    require_file(SHALLOW_POOL_PATH, "Shallow-pool dataset")
    require_file(BEST_CONTEXT_PATH, "Stage 9 best-context table")
    require_file(ABLATION_SUMMARY_PATH, "Stage 9 ablation summary")
    require_file(FULL_RETRAIN_PATH, "Stage 9 full-retrain metrics")

    for condition in CONDITIONS:
        require_model_dir(condition)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    RAW_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        print(
            "WARNING: no GPU detected. Stage 10 is inference-only, so it can run "
            "on CPU, but Colab GPU is recommended for faster model loading/inference."
        )
    else:
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")

    # ---- read Stage 9 provenance ------------------------------------------
    best_df = pd.read_csv(BEST_CONTEXT_PATH)
    best_context = validate_best_context(best_df)

    ablation_df = pd.read_csv(ABLATION_SUMMARY_PATH)
    retrain_df = pd.read_csv(FULL_RETRAIN_PATH)

    # Cross-check that the selected C is actually the maximum CV mean in the
    # ablation summary, with a smaller-C tie-break handled deterministically.
    if not {"condition", "mean", "std", "selected"}.issubset(ablation_df.columns):
        raise KeyError(
            "e2_context_ablation_summary.csv must contain condition, mean, std, selected."
        )

    recomputed_best = (
        ablation_df.sort_values(
            by=["mean", "condition"],
            ascending=[False, True],
            kind="stable",
        )
        .iloc[0]["condition"]
    )
    recomputed_best = int(recomputed_best)

    if recomputed_best != best_context:
        raise AssertionError(
            f"Best-context table selects C{best_context}, but recomputation from "
            f"the ablation summary gives C{recomputed_best}."
        )

    if not bool(
        ablation_df.loc[ablation_df["condition"] == best_context, "selected"].iloc[0]
    ):
        raise AssertionError("Best-context row is not marked selected=True in the ablation summary.")

    # All five Stage 9 rows must be present and trained.
    for condition in CONDITIONS:
        rec = check_model_metadata(condition, retrain_df)
        if int(rec["n_train"]) != 900:
            raise AssertionError(f"C{condition} Stage 9 model n_train={rec['n_train']}, expected 900.")
        if int(rec["seed"]) != SEED:
            raise AssertionError(f"C{condition} Stage 9 model seed={rec['seed']}, expected {SEED}.")
        if int(rec["max_length"]) != MAX_LENGTH:
            raise AssertionError(
                f"C{condition} Stage 9 max_length={rec['max_length']}, expected {MAX_LENGTH}."
            )
        if int(rec["num_epochs"]) != 8:
            raise AssertionError(
                f"C{condition} Stage 9 num_epochs={rec['num_epochs']}, expected 8."
            )
        if int(rec["train_batch_size"]) != 16:
            raise AssertionError(
                f"C{condition} Stage 9 train_batch_size={rec['train_batch_size']}, expected 16."
            )
        if not np.isclose(float(rec["learning_rate"]), 2e-5):
            raise AssertionError(
                f"C{condition} Stage 9 learning_rate={rec['learning_rate']}, expected 2e-5."
            )

    # ---- load and validate shallow pool -----------------------------------
    shallow = validate_shallow_pool(pd.read_csv(SHALLOW_POOL_PATH))
    print("\nShallow-pool population:")
    print(
        shallow["context_size_available"]
        .astype(int)
        .value_counts()
        .sort_index()
        .rename("n")
        .to_string()
    )

    # ---- C4 is explicitly N/A ---------------------------------------------
    print(
        f"\nC_best from Stage 8/9 CV = C{best_context}. "
        "Stage 10 will NOT use shallow-pool performance to change this choice."
    )

    predictions: dict[int, tuple[pd.DataFrame, np.ndarray]] = {}
    overall_rows = []
    class_tables = []
    cm_tables = []
    availability_tables = []

    for condition in EVALUATED_CONDITIONS:
        population = eligible_population(shallow, condition)

        expected_n = {
            0: 900,
            1: 900,
            2: 750,
            3: 450,
        }[condition]

        if len(population) != expected_n:
            raise AssertionError(
                f"C{condition} eligible population has {len(population)} rows; expected {expected_n}."
            )

        model_dir = MODELS_ROOT / f"C{condition}"
        print(
            f"\n--- C{condition}: evaluating {len(population)} shallow-pool posts "
            f"using {model_dir} ---"
        )

        y_true = label_ids(population["label"])
        y_pred, probs, elapsed = predict_condition(
            condition=condition,
            population_df=population,
            model_dir=model_dir,
        )

        if len(y_pred) != len(population) or len(probs) != len(population):
            raise AssertionError(
                f"C{condition}: prediction count does not match eligible population."
            )

        save_predictions(condition, population, y_true, y_pred, probs)

        predictions[condition] = (population.copy(), y_pred.copy())

        metrics = overall_metrics(y_true, y_pred)
        overall_rows.append(
            {
                "condition": condition,
                "experiment_id": f"E2-IndoBERTweet-ITFT-C{condition}",
                "status": "evaluated",
                "population": "shallow_pool",
                "population_rule": (
                    "all shallow posts (context_size_available in {1,2,3})"
                    if condition in [0, 1]
                    else f"shallow posts with context_size_available >= {condition}"
                ),
                "n_examples": metrics["n_examples"],
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "inference_seconds": elapsed,
                "inference_minutes": elapsed / 60.0,
                "is_c_best": condition == best_context,
                "selected_from_shallow_pool": False,
            }
        )

        class_tables.append(class_metrics_table(condition, y_true, y_pred))
        cm_tables.append(confusion_table(condition, y_true, y_pred))
        availability_tables.append(
            availability_breakdown(condition, population, y_pred)
        )

        print(
            f"C{condition}: accuracy={metrics['accuracy']:.4f}  "
            f"macro-F1={metrics['macro_f1']:.4f}  "
            f"weighted-F1={metrics['weighted_f1']:.4f}"
        )

    # C4 is not evaluated because there is no external C4 population.
    overall_rows.append(
        {
            "condition": 4,
            "experiment_id": "E2-IndoBERTweet-ITFT-C4",
            "status": "not_applicable",
            "population": "shallow_pool",
            "population_rule": "no external C4 population: shallow pool excludes context_size_available==4",
            "n_examples": 0,
            "accuracy": np.nan,
            "macro_f1": np.nan,
            "weighted_f1": np.nan,
            "inference_seconds": np.nan,
            "inference_minutes": np.nan,
            "is_c_best": best_context == 4,
            "selected_from_shallow_pool": False,
        }
    )

    # ---- pairwise independent comparisons vs C0 --------------------------
    pairwise_df = pairwise_vs_c0(shallow, predictions)

    # ---- write summaries ---------------------------------------------------
    overall_df = (
        pd.DataFrame(overall_rows)
        .sort_values("condition")
        .reset_index(drop=True)
    )
    by_class_df = pd.concat(class_tables, ignore_index=True).sort_values(
        ["condition", "class_id"]
    )
    by_cm_df = pd.concat(cm_tables, ignore_index=True).sort_values(
        ["condition", "true_class_id", "predicted_class_id"]
    )
    by_availability_df = pd.concat(
        availability_tables, ignore_index=True
    ).sort_values(["condition", "parent_count_available"])

    manifest_rows = []
    for condition in CONDITIONS:
        rec = check_model_metadata(condition, retrain_df)
        eligible_n = (
            int(overall_df.loc[overall_df["condition"] == condition, "n_examples"].iloc[0])
            if condition in overall_df["condition"].values
            else 0
        )
        manifest_rows.append(
            {
                "stage": "10-E2B",
                "condition": condition,
                "experiment_id": f"E2-IndoBERTweet-ITFT-C{condition}",
                "model_path": str(MODELS_ROOT / f"C{condition}"),
                "stage9_status": rec["status"],
                "stage9_seed": int(rec["seed"]),
                "stage9_n_train": int(rec["n_train"]),
                "stage9_max_length": int(rec["max_length"]),
                "stage9_num_epochs": int(rec["num_epochs"]),
                "stage9_train_batch_size": int(rec["train_batch_size"]),
                "stage9_learning_rate": float(rec["learning_rate"]),
                "shallow_pool_total": EXPECTED_SHALLOW_TOTAL,
                "stage10_eligible_examples": eligible_n,
                "stage10_role": (
                    "generalization evaluation"
                    if condition in EVALUATED_CONDITIONS
                    else "not applicable: no external C4 population"
                ),
                "selected_context_cv": best_context,
                "c_best_source": "Stage 8/9 CV mean macro-F1 only",
                "selected_from_shallow_pool": False,
            }
        )

    manifest_df = pd.DataFrame(manifest_rows)

    overall_df.to_csv(TABLES_DIR / "e2b_generalization_metrics.csv", index=False)
    by_availability_df.to_csv(
        TABLES_DIR / "e2b_generalization_by_availability.csv",
        index=False,
    )
    by_class_df.to_csv(
        TABLES_DIR / "e2b_generalization_by_class.csv",
        index=False,
    )
    by_cm_df.to_csv(
        TABLES_DIR / "e2b_generalization_confusion_matrices.csv",
        index=False,
    )
    pairwise_df.to_csv(
        TABLES_DIR / "e2b_generalization_pairwise_vs_c0.csv",
        index=False,
    )
    manifest_df.to_csv(
        TABLES_DIR / "e2b_generalization_manifest.csv",
        index=False,
    )

    # ---- console summary ---------------------------------------------------
    print("\n=== Stage 10 complete ===")
    print("\nOverall generalization metrics:")
    print(
        overall_df[
            [
                "condition",
                "status",
                "n_examples",
                "accuracy",
                "macro_f1",
                "weighted_f1",
                "is_c_best",
            ]
        ].to_string(index=False)
    )

    print("\nIndependent C0-vs-deeper comparisons:")
    print(pairwise_df.to_string(index=False))

    print("\nSaved summary tables:")
    for name in [
        "e2b_generalization_metrics.csv",
        "e2b_generalization_by_availability.csv",
        "e2b_generalization_by_class.csv",
        "e2b_generalization_confusion_matrices.csv",
        "e2b_generalization_pairwise_vs_c0.csv",
        "e2b_generalization_manifest.csv",
    ]:
        print(f"  {TABLES_DIR / name}")

    print("\nSaved raw outputs:")
    for condition in EVALUATED_CONDITIONS:
        print(f"  {RAW_RESULTS_DIR / f'e2b_C{condition}_shallow_predictions.csv'}")
        print(f"  {RAW_RESULTS_DIR / f'e2b_C{condition}_shallow_probs.npy'}")

    print(
        "\nMethodological note: Stage 10 is descriptive only. "
        "C_best remains the CV-selected C4 and is not re-selected from shallow-pool performance."
    )


if __name__ == "__main__":
    main()
