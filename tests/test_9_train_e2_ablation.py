"""Model-free tests for the Stage 9/10 Experiment II-A training runner."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import experiment_utils as eu

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "9_train_e2_ablation.py"
spec = importlib.util.spec_from_file_location("stage9_e2_ablation", SCRIPT_PATH)
assert spec is not None and spec.loader is not None
stage9 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stage9)


def _matched_df() -> pd.DataFrame:
    rows = []
    for fold in range(5):
        for label_idx, label in enumerate(("N", "AL", "HS")):
            for item in range(60):
                i = fold * 180 + label_idx * 60 + item
                rows.append({
                    "post_id": f"post-{i:04d}",
                    "parent_4": f"parent four {i}",
                    "parent_3": f"parent three {i}",
                    "parent_2": f"parent two {i}",
                    "parent_1": f"parent one {i}",
                    "target_text": f"target post {i}",
                    "context_size_available": 4,
                    "label": label,
                    "fold": fold,
                })
    return pd.DataFrame(rows)


def _audit_stub():
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
    return {
        "hyperparameters": hyper,
        "model_revisions": {"indobertweet": "a" * 40, "xlmr": "b" * 40},
        "tokenizer_revisions": {"indobertweet": "a" * 40, "xlmr": "b" * 40},
        "dataset_sha256": "c" * 64,
        "post_id_sha256": "d" * 64,
        "fold_assignment_sha256": "e" * 64,
        "max_length_benchmark": 128,
        "max_length_context": 256,
        "learning_rate_benchmark": {"indobertweet": 2e-5, "xlmr": 1e-5},
        "learning_rate_context": {"indobertweet": 2e-5, "xlmr": 3e-5},
        "lr_search_grid": [float(x) for x in eu.LR_SEARCH_GRID],
        "lr_tuning_seed": 314159,
        "itft_by_backbone": {
            "indobertweet": {"manifest_fingerprint": "1" * 64},
            "xlmr": {"manifest_fingerprint": "2" * 64},
        },
        "stage8_manifest_fingerprint": "3" * 64,
    }


def test_stage_registry_and_backbone_arguments_match_pipeline():
    assert stage9.SCRIPT_NAME == "scripts/9_train_e2_ablation.py"
    assert eu.PIPELINE_SCRIPTS[9] == stage9.SCRIPT_NAME
    assert eu.PIPELINE_SCRIPTS[10] == stage9.SCRIPT_NAME
    assert stage9.STAGE_BY_BACKBONE == {"indobertweet": 9, "xlmr": 10}
    assert stage9.parse_args(["--backbone", "indobertweet", "--validate-only"]).backbone == "indobertweet"
    assert stage9.parse_args(["--backbone", "xlmr", "--validate-only"]).backbone == "xlmr"
    with pytest.raises(SystemExit):
        stage9.parse_args(["--validate-only"])


def test_resolved_run_config_has_stage_specific_identity_and_current_config_fields():
    audit = _audit_stub()
    # Shared-config validation is tested separately in experiment_utils; here
    # inspect the payload this stage constructs without needing real configs.
    with patch.object(stage9.eu, "validate_resolved_run_config") as validation:
        ib = stage9._resolved_run_config("indobertweet", audit, condition=4, fold=0, seed=42, mixed_precision=False)
        xlmr = stage9._resolved_run_config("xlmr", audit, condition=4, fold=0, seed=42, mixed_precision=False)
    assert ib["stage"] == 9
    assert xlmr["stage"] == 10
    assert ib["script_name"] == xlmr["script_name"] == "scripts/9_train_e2_ablation.py"
    assert ib["hf_model_id"] == eu.BACKBONES["indobertweet"].hf_model_id
    assert xlmr["hf_model_id"] == eu.BACKBONES["xlmr"].hf_model_id
    assert ib["max_length_context"] == xlmr["max_length_context"] == 256
    assert ib["truncation_side"] == xlmr["truncation_side"] == "left"
    assert ib["learning_rate_context"] == 2e-5
    assert xlmr["learning_rate_context"] == 3e-5
    assert ib["cv_seeds"] == xlmr["cv_seeds"] == [42, 123]
    assert validation.call_count == 2


def test_config_validator_called_with_correct_stage_input_mode_and_shared_config():
    audit = _audit_stub()
    with patch.object(stage9.eu, "validate_resolved_run_config") as validation:
        stage9._resolved_run_config("xlmr", audit, condition=2, fold=3, seed=123, mixed_precision=False)
    kwargs = validation.call_args.kwargs
    assert kwargs["stage"] == 10
    assert kwargs["backbone"] == "xlmr"
    assert kwargs["input_mode"] == "contextual"
    assert kwargs["context_size"] == 2
    assert kwargs["require_shared_config"] is True


def test_fold_partition_has_720_train_and_180_heldout_and_is_disjoint():
    matched = _matched_df()
    train, heldout, y_train, y_hold = stage9._build_fold_data(matched, condition=3, fold=2)
    assert len(train) == 720
    assert len(heldout) == 180
    assert len(y_train) == 720 and len(y_hold) == 180
    assert set(train.post_id).isdisjoint(set(heldout.post_id))
    assert set(heldout.fold) == {2}
    assert all(train.label.value_counts()[x] == 240 for x in ("N", "AL", "HS"))


def test_fold_partition_rejects_nonstandard_sizes():
    matched = _matched_df().iloc[:-1].copy()
    with pytest.raises(ValueError, match="720 training and 180 held-out"):
        stage9._build_fold_data(matched, condition=1, fold=0)


def test_metrics_include_accuracy_macro_weighted_and_per_class_f1():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 1, 0, 2, 2])
    values = stage9._metrics(y_true, y_pred)
    assert set(values) == {"accuracy", "macro_f1", "weighted_f1", "f1_n", "f1_al", "f1_hs"}
    assert values["accuracy"] == pytest.approx(4 / 6)
    assert all(0 <= x <= 1 for x in values.values())


def test_probability_conversion_is_stable_and_normalized():
    logits = np.array([[1.0, 2.0, 3.0], [1000.0, 999.0, -1000.0]])
    probs = stage9._prediction_probabilities(logits)
    assert probs.shape == (2, 3)
    assert probs.dtype == np.float32
    assert np.all(np.isfinite(probs))
    assert np.allclose(probs.sum(axis=1), 1.0)
    with pytest.raises(ValueError, match="finite values"):
        stage9._prediction_probabilities(np.array([[np.nan, 1, 2]]))


def test_context_construction_uses_tokenizer_separator_and_target_last():
    matched = _matched_df().iloc[[0]].reset_index(drop=True)
    text = stage9._context_texts(matched, 2, " </s> ")[0]
    assert text == "parent two 0 </s> parent one 0 </s> target post 0"
    text0 = stage9._context_texts(matched, 0, " </s> ")[0]
    assert text0 == "target post 0"


def test_per_run_paths_are_backbone_stage_and_seed_specific():
    ib = stage9._run_dir("indobertweet", 4, 2, 42).as_posix()
    xlmr = stage9._run_dir("xlmr", 4, 2, 42).as_posix()
    assert "/stage_09/e2_indobertweet/C4/fold_02/seed_42" in ib
    assert "/stage_10/e2_xlmr/C4/fold_02/seed_42" in xlmr
    assert ib != xlmr


def test_training_arguments_do_not_enable_heldout_evaluation_or_checkpoint_selection():
    class FakeTrainingArguments:
        def __init__(self, output_dir, num_train_epochs, per_device_train_batch_size,
                     per_device_eval_batch_size, gradient_accumulation_steps, learning_rate,
                     weight_decay, warmup_ratio, lr_scheduler_type, eval_strategy,
                     save_strategy, logging_strategy, seed, data_seed, report_to,
                     fp16, disable_tqdm):
            self.kwargs = locals()

    audit = _audit_stub()
    audit["active_backbone"] = "indobertweet"
    args = stage9._make_training_arguments(
        FakeTrainingArguments,
        output_dir=Path("tmp-out"),
        audit=audit,
        seed=42,
        use_fp16=False,
        disable_tqdm=True,
    )
    assert args.kwargs["eval_strategy"] == "no"
    assert args.kwargs["save_strategy"] == "no"
    assert args.kwargs["num_train_epochs"] == 8
    assert args.kwargs["learning_rate"] == 2e-5


def test_legacy_output_guard_rejects_old_flat_result_paths(tmp_path):
    legacy = tmp_path / "results" / "raw" / "e2_C4_fold0_seed42_predictions.csv"
    legacy.parent.mkdir(parents=True)
    legacy.write_text("stale", encoding="utf-8")
    with patch.object(stage9, "PROJECT_ROOT", tmp_path):
        with pytest.raises(RuntimeError, match="Legacy outputs exist"):
            stage9._assert_no_legacy_outputs()
