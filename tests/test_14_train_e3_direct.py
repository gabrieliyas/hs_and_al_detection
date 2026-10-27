"""Unit tests for Stage 14 direct fine-tuning.

These tests exercise deterministic configuration and integrity helpers only;
they do not need Stage 8/11 artifacts, Hugging Face downloads, or a GPU.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import experiment_utils as eu

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "14_train_e3_direct.py"
SPEC = importlib.util.spec_from_file_location("stage14_e3_direct", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
stage14 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stage14)


def _audit(backbone: str = "xlmr", c_best: int = 4, root: Path | None = None):
    root = root or ROOT
    decision_path = root / "results" / "backbone_decision.json"
    decision_path.parent.mkdir(parents=True, exist_ok=True)
    decision_path.write_text('{"decision":"xlmr"}', encoding="utf-8")
    hyper = {
        "optimizer": "AdamW",
        "weight_decay": 0.01,
        "warmup_ratio": 0.0,
        "lr_scheduler": "linear",
        "max_epochs": 8,
        "early_stopping_patience": None,
        "per_device_train_batch_size": 16,
        "per_device_eval_batch_size": 32,
        "gradient_accumulation_steps": 1,
        "effective_batch_size": 16,
        "loss": "unweighted_cross_entropy",
        "checkpoint_metric": "macro_f1",
        "mixed_precision": False,
    }
    sequence = {
        "max_length_benchmark": 128,
        "max_length_context": 256,
        "target_only_truncation_side": "right",
        "context_tokenizer_resolved_revisions": {key: ("a" if key == "indobertweet" else "b") * 40 for key in eu.BACKBONES},
    }
    learning_rates = {
        "lr_tuning_seed": 314159,
        "learning_rate_benchmark": {"indobertweet": 2e-5, "xlmr": 1e-5},
        "learning_rate_context": {"indobertweet": 2e-5, "xlmr": 3e-5},
    }
    decision = {
        "decision": backbone,
        "c_best_by_backbone": {"indobertweet": 3, "xlmr": c_best},
        "source_grid_fingerprints": {"indobertweet": "1" * 64, "xlmr": "2" * 64},
    }
    return {
        "backbone": backbone,
        "c_best": c_best,
        "conditions": stage14._unique_conditions(c_best),
        "sequence": sequence,
        "learning_rates": learning_rates,
        "hyperparameters": hyper,
        "model_revisions": {"indobertweet": "a" * 40, "xlmr": "b" * 40},
        "tokenizer_revisions": {"indobertweet": "a" * 40, "xlmr": "b" * 40},
        "stage8": {"sequence": sequence, "learning_rates": learning_rates, "hyperparameters": hyper, "fingerprint": "3" * 64},
        "stage11": {
            "decision": decision,
            "manifest_fingerprint": "4" * 64,
            "root_path": decision_path,
        },
        "data": {
            "dataset_sha256": "5" * 64,
            "post_id_sha256": "6" * 64,
            "fold_assignment_sha256": "7" * 64,
            "stage2_manifest": {"fingerprint": "8" * 64},
            "matched": pd.DataFrame({"post_id": ["p1", "p2"], "fold": [0, 1]}),
        },
        "e2_grid": {"grid_audit": {"grid_fingerprint": "9" * 64}},
    }


def test_stage_registry_and_run_namespace_follow_current_pipeline():
    assert stage14.STAGE == 14
    assert stage14.SCRIPT_NAME == "scripts/14_train_e3_direct.py"
    assert eu.PIPELINE_SCRIPTS[14] == stage14.SCRIPT_NAME
    assert stage14.INPUT_MODE == "contextual"
    assert stage14.TRUNCATION_SIDE == "left"
    assert stage14.FOLDS == (0, 1, 2, 3, 4)
    assert stage14.SEEDS == (42, 123)


def test_selected_backbone_is_not_hardcoded_to_indobertweet():
    audit = _audit(backbone="xlmr", c_best=3)
    with patch.object(stage14.eu, "validate_resolved_run_config"):
        cfg = stage14._build_run_config(backbone="xlmr", condition=3, fold=0, seed=42, audit=audit)
    assert cfg["backbone"] == "xlmr"
    assert cfg["hf_model_id"] == eu.BACKBONES["xlmr"].hf_model_id
    assert cfg["experiment_id"] == eu.experiment3_id(3, "xlmr")
    assert cfg["selected_backbone"] == "xlmr"
    assert cfg["selected_context_size_c_best"] == 3
    assert cfg["model_initialization"] == "public_pretrained_base_without_ITFT"
    assert cfg["itft_used"] is False
    assert cfg["learning_rate_context"] == 3e-5
    assert cfg["truncation_side"] == "left"


def test_c0_and_c_best_conditions_are_unique_when_c_best_is_c0():
    assert stage14._unique_conditions(0) == (0,)
    assert stage14._unique_conditions(4) == (0, 4)
    with pytest.raises(ValueError, match="C_best must be in 0..4"):
        stage14._unique_conditions(5)


@pytest.mark.parametrize("backbone", ["indobertweet", "xlmr"])
def test_run_path_is_backbone_and_stage_specific(backbone):
    path = stage14._run_dir(backbone, 4, 2, 123, root=ROOT)
    assert path.as_posix().endswith(
        f"results/runs/{eu.RUN_ID}/stage_14/e3_{backbone}/C4/fold_02/seed_123"
    )


def test_probability_array_accepts_valid_probabilities_and_rejects_bad_values():
    valid = np.array([[0.2, 0.3, 0.5], [1.0, 0.0, 0.0]], dtype=float)
    stage14._validate_probability_array(valid, 2, label="valid")
    with pytest.raises(ValueError, match="probability shape"):
        stage14._validate_probability_array(np.ones((2, 2)), 2, label="wrong shape")
    with pytest.raises(ValueError, match="finite and non-negative"):
        stage14._validate_probability_array(np.array([[0.2, -0.1, 0.9]]), 1, label="negative")
    with pytest.raises(ValueError, match="sum to 1"):
        stage14._validate_probability_array(np.array([[0.2, 0.2, 0.2]]), 1, label="bad sum")


def test_metrics_include_accuracy_macro_weighted_and_each_class():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 1, 0, 2, 2])
    metrics = stage14._metrics(y_true, y_pred)
    assert set(metrics) == {"accuracy", "macro_f1", "weighted_f1", "f1_n", "f1_al", "f1_hs"}
    assert metrics["accuracy"] == pytest.approx(4 / 6)
    assert all(0.0 <= value <= 1.0 for value in metrics.values())


def test_unmanifested_partial_run_is_not_silently_reused(tmp_path):
    run_dir = tmp_path / "results" / "runs" / eu.RUN_ID / "stage_14" / "e3_xlmr" / "C4" / "fold_00" / "seed_42"
    run_dir.mkdir(parents=True)
    (run_dir / "predictions.csv").write_text("legacy", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Incomplete Stage 14 outputs"):
        stage14._verify_completed_run(
            run_dir,
            manifest={"fingerprint": "f" * 64},
            backbone="xlmr",
            condition=4,
            fold=0,
            seed=42,
            expected_post_ids=["p1"],
            expected_y_true=np.array([0]),
        )


def test_empty_run_directory_is_not_mistaken_for_completion(tmp_path):
    run_dir = tmp_path / "empty-run"
    run_dir.mkdir()
    assert stage14._verify_completed_run(
        run_dir,
        manifest={"fingerprint": "f" * 64},
        backbone="xlmr",
        condition=0,
        fold=0,
        seed=42,
        expected_post_ids=["p1"],
        expected_y_true=np.array([0]),
    ) is None


def test_contextual_tokenization_uses_left_truncation_and_shared_text_builder():
    class FakeTokenizer:
        sep_token = "</s>"
        truncation_side = "left"
        def __call__(self, texts, **kwargs):
            assert kwargs["truncation"] is True
            assert kwargs["max_length"] == 32
            assert kwargs["padding"] == "max_length"
            return {"input_ids": np.zeros((len(texts), 32), dtype=np.int64)}

    frame = pd.DataFrame([{
        "parent_4": "oldest parent", "parent_3": "parent three", "parent_2": "parent two",
        "parent_1": "newest parent", "target_text": "target", "label": "HS",
    }])
    enc, labels = stage14._tokenize_rows(FakeTokenizer(), frame, condition=4, max_length=32)
    assert enc["input_ids"].shape == (1, 32)
    np.testing.assert_array_equal(labels, [eu.LABEL2ID["HS"]])


def test_resolved_config_validator_is_called_for_selected_backbone_and_context():
    audit = _audit(backbone="xlmr", c_best=4)
    with patch.object(stage14.eu, "validate_resolved_run_config") as mock_validate:
        config = stage14._build_run_config(backbone="xlmr", condition=0, fold=2, seed=123, audit=audit)
    mock_validate.assert_called_once_with(
        config, stage=14, backbone="xlmr", input_mode="contextual", context_size=0, require_shared_config=True
    )


def test_training_args_inherit_frozen_stage8_hyperparameters():
    captured = {}
    def FakeTrainingArguments(**kwargs):
        captured.update(kwargs)
        return kwargs
    cfg = {
        "max_epochs": 8, "per_device_train_batch_size": 16, "per_device_eval_batch_size": 32,
        "gradient_accumulation_steps": 1, "learning_rate_context": 3e-5, "weight_decay": 0.01,
        "warmup_ratio": 0.0, "lr_scheduler": "linear", "seed": 123,
    }
    args = stage14._make_training_args(FakeTrainingArguments, output_dir=Path("tmp"), config=cfg,
                                       use_cuda=False, disable_tqdm=True)
    assert args["num_train_epochs"] == 8
    assert args["learning_rate"] == 3e-5
    assert args["per_device_train_batch_size"] == 16
    assert args["per_device_eval_batch_size"] == 32
    assert args["gradient_accumulation_steps"] == 1
    assert args["eval_strategy"] == "no"
    assert args["save_strategy"] == "no"
    assert args["fp16"] is False
