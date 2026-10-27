"""Independent tests for Stage 5 Experiment I evaluation helpers."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "5_evaluate_e1.py"
spec = importlib.util.spec_from_file_location("stage5_evaluate_e1", SCRIPT_PATH)
stage5 = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(stage5)


def test_probability_array_accepts_valid_three_class_probabilities():
    probs = np.array([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]], dtype=np.float32)
    validated = stage5.validate_probability_array(probs, expected_rows=2, name="probs")
    assert validated.shape == (2, 3)
    assert np.allclose(validated.sum(axis=1), 1.0)


@pytest.mark.parametrize(
    "probs, expected_rows, message",
    [
        (np.ones((2, 2)) / 2, 2, "shape"),
        (np.array([[0.2, 0.2, 0.2]]), 1, "sum to one"),
        (np.array([[np.nan, 0.0, 1.0]]), 1, "non-finite"),
        (np.array([[1.2, -0.1, -0.1]]), 1, r"outside \[0, 1\]"),
    ],
)
def test_probability_array_rejects_invalid_values(probs, expected_rows, message):
    with pytest.raises(ValueError, match=message):
        stage5.validate_probability_array(probs, expected_rows=expected_rows, name="bad_probs")


def test_labels_to_ids_uses_canonical_shared_label_mapping():
    result = stage5._labels_to_ids(pd.Series(["N", "AL", "HS"]), description="test")
    assert result.tolist() == [0, 1, 2]
    with pytest.raises(ValueError, match="non-canonical labels"):
        stage5._labels_to_ids(pd.Series(["hate_speech"]), description="test")


def test_prediction_metrics_include_accuracy_macro_weighted_and_each_class():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 1, 0, 2, 2])
    metrics = stage5._prediction_metrics(y_true, y_pred)
    assert set(metrics) == {"accuracy", "macro_f1", "weighted_f1", "f1_N", "f1_AL", "f1_HS"}
    assert metrics["accuracy"] == pytest.approx(4 / 6)


def test_result_tables_compare_xlmr_minus_indobertweet_and_report_all_classes(monkeypatch):
    monkeypatch.setattr(stage5, "N_BOOTSTRAP", 120)
    rng = np.random.default_rng(314)
    y_true = np.tile(np.array([0, 1, 2]), 30)

    def prob_run(error_rate: float) -> np.ndarray:
        preds = y_true.copy()
        count = int(round(len(y_true) * error_rate))
        indices = rng.choice(len(y_true), size=count, replace=False)
        preds[indices] = (preds[indices] + 1) % 3
        probs = np.full((len(y_true), 3), 0.025, dtype=np.float64)
        probs[np.arange(len(y_true)), preds] = 0.95
        probs /= probs.sum(axis=1, keepdims=True)
        return probs

    runs = {
        "indobertweet": [
            {"seed": seed, "probabilities": prob_run(0.22 + i * 0.005)}
            for i, seed in enumerate(stage5.eu.EVAL_SEEDS_EXP1)
        ],
        "xlmr": [
            {"seed": seed, "probabilities": prob_run(0.17 + i * 0.005)}
            for i, seed in enumerate(stage5.eu.EVAL_SEEDS_EXP1)
        ],
    }
    tables = stage5._build_result_tables(y_true, runs)
    assert len(tables["seed_metrics"]) == 10
    assert len(tables["results"]) == 2
    assert len(tables["ensemble_metrics"]) == 2
    assert set(tables["paired_bootstrap"]["metric"]) == {"macro_f1", "f1_N", "f1_AL", "f1_HS"}
    assert tables["paired_bootstrap"].iloc[0]["comparison"] == "E1-XLMR-Target minus E1-IndoBERTweet-Target"
    assert set(tables["confusion_matrices"]) == {
        "e1_confusion_matrix_indobertweet.csv",
        "e1_confusion_matrix_xlmr.csv",
    }


def test_expected_stage_registry_and_names_are_current():
    assert stage5.STAGE == 5
    assert stage5.SCRIPT_NAME == stage5.eu.PIPELINE_SCRIPTS[5]
    assert stage5.OUTPUT_NAMES[0] == "e1_seed_metrics.csv"
    assert "results/raw/e1_svm_test_predictions.csv" in stage5.LEGACY_OUTPUTS
