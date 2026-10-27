"""Independent unit tests for the shared Stage 6/7 ITFT runner."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import experiment_utils as eu

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "6_train_itft.py"
spec = importlib.util.spec_from_file_location("itft_stage_script", SCRIPT_PATH)
assert spec is not None and spec.loader is not None
itft = importlib.util.module_from_spec(spec)
spec.loader.exec_module(itft)


def _audit_stub():
    return {
        "model_revisions": {"indobertweet": "a" * 40, "xlmr": "b" * 40},
        "tokenizer_revisions": {"indobertweet": "a" * 40, "xlmr": "b" * 40},
        "dataset_sha256": "c" * 64,
        "train_sha256": "d" * 64,
        "validation_sha256": "e" * 64,
        "class_weights_sha256": "f" * 64,
        "sequence_config_sha256": "1" * 64,
        "learning_rate_config_sha256": "2" * 64,
        "split_assignment_sha256": "3" * 64,
        "max_length_benchmark": 128,
        "learning_rates": {"indobertweet": 2e-5, "xlmr": 1e-5},
        "learning_rate_tuning_seed": 314159,
        "class_weights_by_id": {0: 1.0, 1: 1.0, 2: 1.0},
    }


def test_stage_number_and_script_identity_match_pipeline_registry():
    assert itft.SCRIPT_NAME == "scripts/6_train_itft.py"
    assert itft.STAGE_BY_BACKBONE == {"indobertweet": 6, "xlmr": 7}
    assert eu.PIPELINE_SCRIPTS[6] == itft.SCRIPT_NAME
    assert eu.PIPELINE_SCRIPTS[7] == itft.SCRIPT_NAME


def test_cli_requires_backbone_and_supports_both_logical_stages():
    a = itft.parse_args(["--backbone", "indobertweet", "--validate-only"])
    b = itft.parse_args(["--backbone", "xlmr", "--validate-only"])
    assert itft.STAGE_BY_BACKBONE[a.backbone] == 6
    assert itft.STAGE_BY_BACKBONE[b.backbone] == 7
    assert a.validate_only and b.validate_only
    with pytest.raises(SystemExit):
        itft.parse_args(["--validate-only"])


def test_resolved_configs_follow_shared_utility_and_differ_only_in_allowed_fields():
    audit = _audit_stub()
    cfg_ib = itft._resolved_config("indobertweet", audit, mixed_precision=False)
    cfg_xlmr = itft._resolved_config("xlmr", audit, mixed_precision=False)
    assert cfg_ib["stage"] == 6
    assert cfg_xlmr["stage"] == 7
    assert cfg_ib["script_name"] == cfg_xlmr["script_name"] == itft.SCRIPT_NAME
    assert cfg_ib["hf_model_id"] == eu.BACKBONES["indobertweet"].hf_model_id
    assert cfg_xlmr["hf_model_id"] == eu.BACKBONES["xlmr"].hf_model_id
    assert cfg_ib["learning_rate_benchmark"] == 2e-5
    assert cfg_xlmr["learning_rate_benchmark"] == 1e-5
    assert cfg_ib["max_length_benchmark"] == cfg_xlmr["max_length_benchmark"] == 128
    assert cfg_ib["truncation_side"] == cfg_xlmr["truncation_side"] == "right"
    eu.check_config_parity(cfg_ib, cfg_xlmr, input_mode="target_only")


def test_metrics_report_accuracy_macro_weighted_and_all_three_class_f1s():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 1, 0, 2, 2])
    metrics = itft._metrics_for_ids(y_true, y_pred)
    assert set(metrics) == {"accuracy", "macro_f1", "weighted_f1", "f1_n", "f1_al", "f1_hs"}
    assert metrics["accuracy"] == pytest.approx(4 / 6)
    assert all(0 <= value <= 1 for value in metrics.values())


def test_label_and_text_validator_rejects_missing_or_unknown_labels():
    valid = pd.DataFrame({"clean_text": ["post one", "post two", "post three"], "label": ["N", "AL", "HS"]})
    assert itft._validate_text_labels(valid, Path("train.csv"), "train") == ["post one", "post two", "post three"]
    invalid = valid.copy()
    invalid.loc[2, "label"] = "OTHER"
    with pytest.raises(ValueError, match="labels must be exactly"):
        itft._validate_text_labels(invalid, Path("train.csv"), "train")


def test_label_and_text_validator_rejects_duplicate_or_blank_text():
    duplicate = pd.DataFrame({"clean_text": ["same text", "same text", "third"], "label": ["N", "AL", "HS"]})
    with pytest.raises(ValueError, match="duplicate clean_text"):
        itft._validate_text_labels(duplicate, Path("train.csv"), "train")
    blank = pd.DataFrame({"clean_text": ["first", "   ", "third"], "label": ["N", "AL", "HS"]})
    with pytest.raises(ValueError, match="blank clean_text"):
        itft._validate_text_labels(blank, Path("train.csv"), "train")


def test_class_weight_audit_recomputes_from_training_partition(tmp_path):
    train = pd.DataFrame({"label": ["N", "N", "AL", "HS"]})
    expected = eu.compute_class_weights(train["label"])
    weights = pd.DataFrame({
        "label": [eu.ID2LABEL[i] for i in range(3)],
        "weight": [expected[i] for i in range(3)],
    })
    path = tmp_path / "weights.csv"
    weights.to_csv(path, index=False)
    observed, digest = itft._validate_weight_table(train, path)
    assert observed == pytest.approx(expected)
    assert len(digest) == 64
    weights.loc[0, "weight"] = 99.0
    weights.to_csv(path, index=False)
    with pytest.raises(ValueError, match="differs from training-only recomputation"):
        itft._validate_weight_table(train, path)


def test_legacy_output_guard_detects_old_shared_checkpoint(tmp_path):
    old_checkpoint = tmp_path / "models" / "E2_itft_context"
    old_checkpoint.mkdir(parents=True)
    with patch.object(itft, "PROJECT_ROOT", tmp_path):
        with pytest.raises(RuntimeError, match="Legacy outputs exist"):
            itft.assert_no_legacy_itft_outputs()


def test_checkpoint_hash_inventory_detects_files_and_ignores_its_own_manifest(tmp_path):
    model_dir = tmp_path / "checkpoint"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "model.safetensors").write_bytes(b"fake-weights")
    (model_dir / "itft_checkpoint_manifest.json").write_text("{}", encoding="utf-8")
    files = itft._sha_tree(model_dir)
    assert set(files) == {"config.json", "model.safetensors"}
    assert all(len(value) == 64 for value in files.values())
