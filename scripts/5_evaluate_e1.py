"""
5_evaluate_e1.py

Stage 5: aggregate Experiment I results (CPU-only, no GPU needed) --
combines Stage 3's SVM output and Stage 4's 5-seed transformer output into
the final Experiment I comparison table:
  - E1-SVM-TFIDF: single deterministic run.
  - E1-IndoBERTweet-Target: per-seed mean +/- std (training stability,
    via experiment_utils.summarize_seed_runs), reported separately from
    the head-to-head significance test.
  - SVM vs. transformer significance test: seed-ensembled probabilities
    (experiment_utils.ensemble_seed_probabilities) vs. SVM's predictions,
    via experiment_utils.paired_bootstrap_ci -- per
    docs/experiment_protocol.md Section 1 ("a single arbitrary seed should
    not be fed into the paired bootstrap").

Both prediction CSVs store canonical string labels ("N"/"AL"/"HS"); this
script converts them to integer ids (via LABEL2ID) immediately after
loading and works in id space throughout, since
experiment_utils.ensemble_seed_probabilities returns integer class ids
(argmax over averaged softmax probabilities) rather than string labels --
mixing the two representations in one sklearn metric call raises a
"Mix of label input types" error.

Before combining anything, this script asserts that the SVM's y_true and
every seed's y_true are identical arrays. Both Stage 3 and Stage 4 load
data/processed/ib_test.csv without shuffling, so they should already be in
the same row order by construction -- this assertion is the safeguard that
actually catches it if that ever silently breaks (e.g. ib_test.csv gets
regenerated with a different split between the two runs), since a
mismatched pairing would silently corrupt the significance test rather
than raise an obvious error.

Usage:
    python scripts/5_evaluate_e1.py

Input:
    results/raw/e1_svm_test_predictions.csv
    results/raw/e1_transformer_seed{K}_predictions.csv  (one per seed)
    results/raw/e1_transformer_seed{K}_probs.npy        (one per seed)

Outputs:
    results/tables/e1_final_results.csv           -- SVM row + transformer mean+/-std row
    results/tables/e1_significance_test.csv        -- paired bootstrap result
    results/tables/e1_transformer_confusion_matrix.csv  -- ensembled predictions
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiment_utils import (  # noqa: E402
    EXPERIMENT1_IDS,
    ID2LABEL,
    LABEL2ID,
    ensemble_seed_probabilities,
    paired_bootstrap_ci,
    summarize_seed_runs,
)

RAW_RESULTS_DIR = Path("results/raw")
TABLES_DIR = Path("results/tables")
LABELS_IN_ID_ORDER = [ID2LABEL[i] for i in range(len(ID2LABEL))]  # ["N", "AL", "HS"]


def load_svm_predictions() -> pd.DataFrame:
    path = RAW_RESULTS_DIR / "e1_svm_test_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run scripts/3_train_e1_svm.py first.")
    return pd.read_csv(path)


def discover_seeds() -> list[int]:
    seeds = []
    for f in RAW_RESULTS_DIR.glob("e1_transformer_seed*_predictions.csv"):
        seed_str = f.stem.replace("e1_transformer_seed", "").replace("_predictions", "")
        seeds.append(int(seed_str))
    if not seeds:
        raise FileNotFoundError(
            f"No e1_transformer_seed*_predictions.csv files found in {RAW_RESULTS_DIR}. "
            "Run scripts/4_train_e1_indobertweet.py (on Colab) first, then copy its "
            "results/raw/ and results/tables/ outputs back here -- see the chat "
            "response for the exact between-stage steps."
        )
    return sorted(seeds)


def metric_row(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "model": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def main() -> None:
    if not RAW_RESULTS_DIR.exists():
        raise FileNotFoundError(f"{RAW_RESULTS_DIR} not found.")

    # --- Load SVM (Stage 3); convert to int ids immediately (see module docstring) ---
    svm_df = load_svm_predictions()
    svm_y_true = svm_df["y_true"].map(LABEL2ID).to_numpy()
    svm_y_pred = svm_df["y_pred"].map(LABEL2ID).to_numpy()
    print(f"[1/6] Loaded SVM predictions ({len(svm_df)} rows).")

    # --- Load transformer (Stage 4), all discovered seeds ---
    seeds = discover_seeds()
    print(f"[2/6] Discovered {len(seeds)} seed(s): {seeds}")

    seed_y_true_list = []
    seed_macro_f1 = []
    seed_weighted_f1 = []
    seed_accuracy = []
    prob_arrays = []

    for seed in seeds:
        pred_path = RAW_RESULTS_DIR / f"e1_transformer_seed{seed}_predictions.csv"
        probs_path = RAW_RESULTS_DIR / f"e1_transformer_seed{seed}_probs.npy"
        if not probs_path.exists():
            raise FileNotFoundError(
                f"{probs_path} not found (predictions CSV exists but probs .npy is "
                "missing) -- make sure both files were copied back from Colab."
            )
        seed_df = pd.read_csv(pred_path)
        probs = np.load(probs_path)

        seed_y_true = seed_df["y_true"].map(LABEL2ID).to_numpy()
        seed_y_pred = seed_df["y_pred"].map(LABEL2ID).to_numpy()

        seed_y_true_list.append(seed_y_true)
        prob_arrays.append(probs)

        seed_accuracy.append(accuracy_score(seed_y_true, seed_y_pred))
        seed_macro_f1.append(f1_score(seed_y_true, seed_y_pred, average="macro"))
        seed_weighted_f1.append(f1_score(seed_y_true, seed_y_pred, average="weighted"))

    # --- Alignment safeguard: every source must share the same y_true, same order ---
    for seed, seed_y_true in zip(seeds, seed_y_true_list):
        if len(seed_y_true) != len(svm_y_true) or not np.array_equal(seed_y_true, svm_y_true):
            raise AssertionError(
                f"y_true mismatch between SVM predictions and seed {seed}'s transformer "
                "predictions -- they must come from the exact same test set in the same "
                "row order (both load data/processed/ib_test.csv without shuffling by "
                "construction). Re-check that ib_test.csv wasn't regenerated between "
                "Stage 3 and Stage 4 runs."
            )
    print("[3/6] Alignment check passed -- SVM and all seeds share identical y_true.")

    y_true = svm_y_true  # now known-identical across all sources

    # --- Per-seed mean +/- std (training stability) ---
    seed_summary = {
        "accuracy": summarize_seed_runs(seed_accuracy),
        "macro_f1": summarize_seed_runs(seed_macro_f1),
        "weighted_f1": summarize_seed_runs(seed_weighted_f1),
    }
    print("\n[4/6] Per-seed mean +/- std (training stability):")
    for metric, s in seed_summary.items():
        print(f"  {metric}: {s['mean']:.4f} +/- {s['std']:.4f}  (n_seeds={s['n_seeds']})")

    # --- Ensembled prediction set for the head-to-head significance test ---
    ensembled_y_pred = ensemble_seed_probabilities(prob_arrays)
    sig_result = paired_bootstrap_ci(y_true, svm_y_pred, ensembled_y_pred)
    print(f"\n[5/6] SVM vs. {EXPERIMENT1_IDS['transformer']} (seed-ensembled) "
          f"paired bootstrap (macro F1, {sig_result['n_bootstrap']} resamples, "
          f"{sig_result['ci_level']}% CI):")
    print(f"  observed diff (transformer - svm): {sig_result['observed_diff']:.4f}")
    print(f"  {sig_result['ci_level']}% CI: [{sig_result['ci_lower']:.4f}, {sig_result['ci_upper']:.4f}]")
    print(f"  statistically credible at this level: {sig_result['significant']}")

    # --- Final results table ---
    svm_row = metric_row(EXPERIMENT1_IDS["svm"], y_true, svm_y_pred)
    transformer_row = {
        "model": EXPERIMENT1_IDS["transformer"],
        "accuracy": seed_summary["accuracy"]["mean"],
        "accuracy_std": seed_summary["accuracy"]["std"],
        "precision_macro": None,  # not tracked per-seed above; f1/accuracy are the headline metrics
        "recall_macro": None,
        "f1_macro": seed_summary["macro_f1"]["mean"],
        "f1_macro_std": seed_summary["macro_f1"]["std"],
        "f1_weighted": seed_summary["weighted_f1"]["mean"],
        "f1_weighted_std": seed_summary["weighted_f1"]["std"],
        "n_seeds": seed_summary["macro_f1"]["n_seeds"],
    }
    final_results_df = pd.DataFrame([svm_row, transformer_row])

    ensembled_row = metric_row(
        f"{EXPERIMENT1_IDS['transformer']} (seed-ensembled, for significance test)",
        y_true, ensembled_y_pred,
    )
    final_results_df = pd.concat([final_results_df, pd.DataFrame([ensembled_row])], ignore_index=True)

    sig_df = pd.DataFrame([{
        "comparison": f"{EXPERIMENT1_IDS['transformer']} (ensembled) - {EXPERIMENT1_IDS['svm']}",
        "metric": "macro_f1",
        **sig_result,
    }])

    cm = confusion_matrix(y_true, ensembled_y_pred, labels=list(range(len(ID2LABEL))))
    cm_df = pd.DataFrame(cm, index=LABELS_IN_ID_ORDER, columns=LABELS_IN_ID_ORDER)
    cm_df.index.name = "true \\ pred"

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    final_results_df.to_csv(TABLES_DIR / "e1_final_results.csv", index=False)
    sig_df.to_csv(TABLES_DIR / "e1_significance_test.csv", index=False)
    cm_df.to_csv(TABLES_DIR / "e1_transformer_confusion_matrix.csv")

    print("\n[6/6] Saved:")
    print("  results/tables/e1_final_results.csv")
    print("  results/tables/e1_significance_test.csv")
    print("  results/tables/e1_transformer_confusion_matrix.csv")
    print("\nExperiment I results:")
    print(final_results_df.to_string(index=False))


if __name__ == "__main__":
    main()
