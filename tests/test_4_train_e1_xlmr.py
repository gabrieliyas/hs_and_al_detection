from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "4_train_e1_xlmr.py"
spec = importlib.util.spec_from_file_location("stage4_xlmr", SCRIPT_PATH)
stage4 = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(stage4)


def _labels(n: int) -> list[str]:
    return [("N", "AL", "HS")[i % 3] for i in range(n)]


def _write_frame(path: Path, n: int, prefix: str) -> pd.DataFrame:
    frame = pd.DataFrame({
        "text_raw": [f"raw {prefix} {i}" for i in range(n)],
        "clean_text": [f"clean {prefix} unique tweet {i}" for i in range(n)],
        "label": _labels(n),
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame


def _write_test_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    monkeypatch.setattr(stage4, "PROJECT_ROOT", tmp_path)
    train_path = tmp_path / "data/processed/ib_train.csv"
    val_path = tmp_path / "data/processed/ib_val.csv"
    test_path = tmp_path / "data/processed/ib_test.csv"
    weights_path = tmp_path / "results/tables/ib_class_weights.csv"
    seq_path = tmp_path / "results/config/sequence_length.json"
    lr_path = tmp_path / "results/config/learning_rates.json"
    train = _write_frame(train_path, 8922, "train")
    val = _write_frame(val_path, 1912, "validation")
    test = _write_frame(test_path, 1912, "test")
    (tmp_path / "results/tables").mkdir(parents=True, exist_ok=True)

    eu = stage4.eu
    weights = eu.compute_class_weights(train["label"])
    pd.DataFrame([
        {"label": eu.ID2LABEL[class_id], "weight": weights[class_id]}
        for class_id in range(len(eu.LABEL2ID))
    ]).to_csv(weights_path, index=False)

    train_sha = eu.sha256_file(train_path)
    val_sha = eu.sha256_file(val_path)
    test_sha = eu.sha256_file(test_path)
    weights_sha = eu.sha256_file(weights_path)
    revisions = {"indobertweet": "a" * 40, "xlmr": "b" * 40}
    model_ids = {b: eu.BACKBONES[b].hf_model_id for b in eu.BACKBONES}
    sequence_doc = {
        "schema_version": eu.CONFIG_SCHEMA_VERSION,
        "run_id": eu.RUN_ID,
        "truncation_side": "left",
        "target_only_truncation_side": "right",
        "benchmark_length_source_partition": "benchmark_train",
        "max_length_benchmark": 112,
        "benchmark_length_percentile": 95.0,
        "benchmark_length_percentile_tokens": {"indobertweet": 100.0, "xlmr": 108.0},
        "benchmark_truncation_rate_at_max_length": {"indobertweet": 0.01, "xlmr": 0.02},
        "benchmark_training_split_sha256": train_sha,
        "benchmark_training_rows": 8922,
        "benchmark_training_text_column": "clean_text",
        "tokenizer_model_ids": model_ids,
        "tokenizer_resolved_revisions": revisions,
    }
    seq_path.parent.mkdir(parents=True, exist_ok=True)
    seq_path.write_text(json.dumps(sequence_doc, indent=2), encoding="utf-8")
    seq_sha = eu.sha256_file(seq_path)
    dataset_sha = eu.sha256_json({"train_sha256": train_sha, "validation_sha256": val_sha})
    split_sha = eu.sha256_json({"train_sha256": train_sha, "validation_sha256": val_sha, "split_seed": 42})
    grid_scores_indobertweet = {f"{lr:.0e}": score for lr, score in zip(eu.LR_SEARCH_GRID, (0.61, 0.72, 0.69, 0.50))}
    grid_scores_xlmr = {f"{lr:.0e}": score for lr, score in zip(eu.LR_SEARCH_GRID, (0.60, 0.62, 0.67, 0.75))}
    lr_doc = {
        "schema_version": eu.CONFIG_SCHEMA_VERSION,
        "run_id": eu.RUN_ID,
        "lr_search_grid": list(eu.LR_SEARCH_GRID),
        "lr_tuning_seed": 314159,
        "learning_rate_benchmark": {"indobertweet": 2e-5, "xlmr": 5e-5},
        "benchmark_trial_macro_f1": {"indobertweet": grid_scores_indobertweet, "xlmr": grid_scores_xlmr},
        "benchmark_preflight": {
            "train_csv_sha256": train_sha,
            "validation_csv_sha256": val_sha,
            "class_weights_csv_sha256": weights_sha,
            "sequence_length_config_sha256": seq_sha,
            "benchmark_dataset_sha256": dataset_sha,
            "split_assignment_sha256": split_sha,
            "max_length_benchmark": 112,
            "truncation_side": "right",
            "model_ids": model_ids,
            "model_revisions": revisions,
        },
    }
    lr_path.write_text(json.dumps(lr_doc, indent=2), encoding="utf-8")

    monkeypatch.setattr(stage4, "TRAIN_PATH", Path("data/processed/ib_train.csv"))
    monkeypatch.setattr(stage4, "VAL_PATH", Path("data/processed/ib_val.csv"))
    monkeypatch.setattr(stage4, "TEST_PATH", Path("data/processed/ib_test.csv"))
    monkeypatch.setattr(stage4, "WEIGHTS_PATH", Path("results/tables/ib_class_weights.csv"))
    monkeypatch.setattr(stage4, "SEQUENCE_CONFIG_PATH", Path("results/config/sequence_length.json"))
    monkeypatch.setattr(stage4, "LEARNING_RATE_CONFIG_PATH", Path("results/config/learning_rates.json"))
    return {"train": train, "validation": val, "test": test, "sequence": sequence_doc, "learning_rates": lr_doc}


def test_validate_text_labels_accepts_canonical_split(tmp_path: Path) -> None:
    path = tmp_path / "split.csv"
    frame = pd.DataFrame({"clean_text": ["a unique text", "another unique text", "third unique text"], "label": ["N", "AL", "HS"]})
    frame.to_csv(path, index=False)
    assert stage4._validate_text_labels(frame, path, "test") == frame["clean_text"].tolist()


def test_validate_text_labels_rejects_blank_and_duplicate_text(tmp_path: Path) -> None:
    path = tmp_path / "split.csv"
    blank = pd.DataFrame({"clean_text": ["   ", "another text", "third text"], "label": ["N", "AL", "HS"]})
    with pytest.raises(ValueError, match="blank"):
        stage4._validate_text_labels(blank, path, "test")
    duplicate = pd.DataFrame({"clean_text": ["same text", "same text", "third text"], "label": ["N", "AL", "HS"]})
    with pytest.raises(ValueError, match="duplicate"):
        stage4._validate_text_labels(duplicate, path, "test")


def test_realistic_preflight_schema_and_stage4_config_validate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_test_project(tmp_path, monkeypatch)
    audit = stage4._load_and_validate_inputs()
    assert audit["max_length_benchmark"] == 112
    assert audit["learning_rate_benchmark"] == pytest.approx(5e-5)
    assert audit["model_revision"] == "b" * 40
    config = stage4._resolved_config(audit, seed=stage4.eu.EVAL_SEEDS_EXP1[0], mixed_precision=False)
    assert config["stage"] == 4
    assert config["backbone"] == "xlmr"
    assert config["hf_model_id"] == stage4.eu.BACKBONES["xlmr"].hf_model_id
    assert config["experiment_id"] == "E1-XLMR-Target"
    assert config["script_name"] == "scripts/4_train_e1_xlmr.py"
    assert config["truncation_side"] == "right"
    assert config["eval_seeds_exp1"] == list(stage4.eu.EVAL_SEEDS_EXP1)
    peer = dict(config, backbone="indobertweet", hf_model_id=stage4.eu.BACKBONES["indobertweet"].hf_model_id,
                model_revision="a" * 40, tokenizer_revision="a" * 40,
                learning_rate_benchmark=2e-5)
    stage4.eu.check_config_parity(config, peer, input_mode="target_only")


def test_hash_audit_rejects_changed_training_split(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_test_project(tmp_path, monkeypatch)
    train_path = tmp_path / "data/processed/ib_train.csv"
    frame = pd.read_csv(train_path)
    frame.loc[0, "clean_text"] = "changed after Stage 1b"
    frame.to_csv(train_path, index=False)
    with pytest.raises(ValueError, match="SHA-256 differs from Stage 1b"):
        stage4._load_and_validate_inputs()


def test_prediction_metrics_include_all_three_classes() -> None:
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 1, 0, 2, 2])
    metrics = stage4._metrics_for_ids(y_true, y_pred)
    assert set(metrics) == {"accuracy", "macro_f1", "weighted_f1", "f1_n", "f1_al", "f1_hs"}
    assert metrics["accuracy"] == pytest.approx(4 / 6)
    assert all(0.0 <= value <= 1.0 for value in metrics.values())


def test_legacy_output_names_are_detected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(stage4, "PROJECT_ROOT", tmp_path)
    legacy = tmp_path / "results/raw/e1_transformer_seed42_probs.npy"
    legacy.parent.mkdir(parents=True)
    legacy.touch()
    assert legacy in stage4._resolve_legacy_outputs()
    with pytest.raises(RuntimeError, match="Legacy outputs exist"):
        stage4.assert_no_legacy_stage4_outputs()
