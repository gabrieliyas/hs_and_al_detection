"""
3_train_e1_svm.py

Stage 3: train the Experiment I SVM+TF-IDF baseline (E1-SVM-TFIDF).

Per docs/experiment_protocol.md Section 1 and Section 4:
  - Single deterministic run (no hyperparameter search) -- the SVM is a
    baseline, not a tuning target, so `ib_val.csv` is not used here.
  - `class_weight="balanced"`, fit on the training partition only. This is
    the same weighting formula already verified for the transformer
    (n_samples / (n_classes * count_c)); after fitting, the SVM's own
    `class_weight_` attribute is checked against the documented values
    (N=0.7456, AL=2.5528, HS=0.7893) as a hard sanity check, not a soft one
    -- a mismatch would mean the two Experiment I baselines are no longer
    using comparable weighting, which is worth catching immediately.

Note on TF-IDF hyperparameters: `ngram_range=(1, 2)`, `min_df=2`, and
`max_features=20000` below are a reasonable first-pass configuration for
short Indonesian tweet text, not a tuned or previously-decided setting --
no hyperparameter search is in scope for this baseline (see above), so
these are stated here as an assumption, not an established decision.

Usage:
    python scripts/3_train_e1_svm.py

Input:
    data/processed/ib_train.csv
    data/processed/ib_test.csv

Outputs:
    models/E1_baseline/e1_svm_tfidf.joblib           -- fitted pipeline (vectorizer + SVM)
    results/raw/e1_svm_test_predictions.csv          -- row_id, y_true, y_pred (for Stage 5)
    results/tables/e1_svm_metrics.csv                -- accuracy/precision/recall/F1 (macro+weighted)
    results/tables/e1_svm_confusion_matrix.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.class_weight import compute_class_weight

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiment_utils import EXPERIMENT1_IDS, ID2LABEL  # noqa: E402

TRAIN_PATH = Path("data/processed/ib_train.csv")
TEST_PATH = Path("data/processed/ib_test.csv")
MODELS_DIR = Path("models/E1_baseline")
RAW_RESULTS_DIR = Path("results/raw")
TABLES_DIR = Path("results/tables")

RANDOM_STATE = 42
LABELS_IN_ID_ORDER = [ID2LABEL[i] for i in range(len(ID2LABEL))]  # ["N", "AL", "HS"]

# Documented, previously-verified class weights (see
# docs/dataset_split_and_weighting.md Section 1.4) -- checked against the
# SVM's own fitted class_weight_ as a hard sanity check below.
EXPECTED_WEIGHTS = {"N": 0.7456, "AL": 2.5528, "HS": 0.7893}


def main() -> None:
    for p in (TRAIN_PATH, TEST_PATH):
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found. Run scripts/1_preprocess_benchmark.py first."
            )

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    print(f"[1/5] Loaded train ({len(train_df)} rows), test ({len(test_df)} rows).")

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20_000)),
        ("svm", LinearSVC(class_weight="balanced", random_state=RANDOM_STATE)),
    ])
    pipeline.fit(train_df["clean_text"], train_df["label"])
    print(f"[2/5] Fitted {EXPERIMENT1_IDS['svm']} on the training partition.")

    # --- Sanity check: the 'balanced' weights scikit-learn computes for this
    # training partition match the documented values. LinearSVC (liblinear-
    # backed) does not expose a fitted class_weight_ attribute the way
    # SVC/NuSVC (libsvm-backed) do, so the weights are recomputed directly
    # via the same helper class_weight="balanced" uses internally --
    # equivalent to what the fitted estimator actually used, without
    # depending on an attribute LinearSVC doesn't provide.
    fitted_weights = dict(zip(
        LABELS_IN_ID_ORDER,
        compute_class_weight("balanced", classes=np.array(LABELS_IN_ID_ORDER), y=train_df["label"]),
    ))
    print("\nClass weights ('balanced', recomputed from the training partition):")
    for label in LABELS_IN_ID_ORDER:
        fitted = fitted_weights[label]
        expected = EXPECTED_WEIGHTS[label]
        status = "OK" if np.isclose(fitted, expected, atol=1e-3) else "MISMATCH"
        print(f"  {label}: fitted={fitted:.4f}  expected={expected:.4f}  [{status}]")
    mismatches = [
        label for label in LABELS_IN_ID_ORDER
        if not np.isclose(fitted_weights[label], EXPECTED_WEIGHTS[label], atol=1e-3)
    ]
    if mismatches:
        raise AssertionError(
            f"The 'balanced' weights for this training partition do not match "
            f"the documented values for: "
            f"{mismatches}. This means the SVM and transformer baselines are "
            "no longer using comparable class weighting -- check that "
            "ib_train.csv matches the training partition used to derive "
            "docs/dataset_split_and_weighting.md's documented weights."
        )
    print("[3/5] Class-weight sanity check passed -- matches "
          "docs/dataset_split_and_weighting.md Section 1.4.")

    # --- Predict on the held-out test set ---
    y_true = test_df["label"].to_numpy()
    y_pred = pipeline.predict(test_df["clean_text"])

    report = classification_report(
        y_true, y_pred, labels=LABELS_IN_ID_ORDER, output_dict=True, zero_division=0
    )
    metrics_rows = []
    for label in LABELS_IN_ID_ORDER:
        metrics_rows.append({
            "label": label,
            "precision": report[label]["precision"],
            "recall": report[label]["recall"],
            "f1": report[label]["f1-score"],
            "support": report[label]["support"],
        })
    metrics_rows.append({
        "label": "accuracy", "precision": None, "recall": None,
        "f1": report["accuracy"], "support": len(y_true),
    })
    metrics_rows.append({
        "label": "macro_avg",
        "precision": report["macro avg"]["precision"],
        "recall": report["macro avg"]["recall"],
        "f1": report["macro avg"]["f1-score"],
        "support": report["macro avg"]["support"],
    })
    metrics_rows.append({
        "label": "weighted_avg",
        "precision": report["weighted avg"]["precision"],
        "recall": report["weighted avg"]["recall"],
        "f1": report["weighted avg"]["f1-score"],
        "support": report["weighted avg"]["support"],
    })
    metrics_df = pd.DataFrame(metrics_rows)
    print("\n[4/5] Test-set performance:")
    print(metrics_df.to_string(index=False))

    cm = confusion_matrix(y_true, y_pred, labels=LABELS_IN_ID_ORDER)
    cm_df = pd.DataFrame(cm, index=LABELS_IN_ID_ORDER, columns=LABELS_IN_ID_ORDER)
    cm_df.index.name = "true \\ pred"

    # --- Save everything ---
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RAW_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, MODELS_DIR / "e1_svm_tfidf.joblib")

    preds_df = pd.DataFrame({
        "row_id": np.arange(len(test_df)),
        "y_true": y_true,
        "y_pred": y_pred,
    })
    preds_df.to_csv(RAW_RESULTS_DIR / "e1_svm_test_predictions.csv", index=False)

    metrics_df.to_csv(TABLES_DIR / "e1_svm_metrics.csv", index=False)
    cm_df.to_csv(TABLES_DIR / "e1_svm_confusion_matrix.csv")

    print("\n[5/5] Saved:")
    print("  models/E1_baseline/e1_svm_tfidf.joblib")
    print("  results/raw/e1_svm_test_predictions.csv")
    print("  results/tables/e1_svm_metrics.csv")
    print("  results/tables/e1_svm_confusion_matrix.csv")
    print(
        "\nNote: models/E1_baseline/e1_svm_tfidf.joblib is a large-ish binary "
        "artifact, regenerable by re-running this script. If your .gitignore "
        "only lists *.pt/*.pkl/*.bin/*.safetensors under models/, add "
        "*.joblib too (see the updated gitignore_template.txt)."
    )


if __name__ == "__main__":
    main()
