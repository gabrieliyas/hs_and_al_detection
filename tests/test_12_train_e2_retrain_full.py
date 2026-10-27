"""Unit tests for Stage 12 full matched-subset retraining.

These tests do not run Transformer training or require Stage 11 artifacts. They
exercise stage naming, paths, hash inventories, manifest-output guards, and the
fixed training-argument configuration using lightweight test doubles.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "12_train_e2_retrain_full.py"
SPEC = importlib.util.spec_from_file_location("stage12_retrain", SCRIPT_PATH)
assert SPEC and SPEC.loader
stage12 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stage12)


def test_authoritative_stage_and_script_name():
    assert stage12.STAGE == 12
    assert stage12.SCRIPT_NAME == "scripts/12_train_e2_retrain_full.py"
    assert stage12.eu.PIPELINE_SCRIPTS[12] == stage12.SCRIPT_NAME


def test_context_run_and_model_dirs_are_backbone_specific(tmp_path):
    expected = stage12._expected_run_dir("xlmr", 3, tmp_path)
    model = stage12._model_dir("xlmr", 3, tmp_path)
    assert expected == tmp_path / "results/runs/left_truncation_v2/stage_12/e2_xlmr/C3"
    assert model == tmp_path / "models/e2_full_retrained/xlmr/C3"
    assert stage12._model_dir("indobertweet", 3, tmp_path) != model


@pytest.mark.parametrize("context", range(5))
def test_all_context_sizes_have_canonical_experiment_ids(context):
    for backbone in ("indobertweet", "xlmr"):
        value = stage12.eu.experiment2_id(context, backbone)
        assert value.startswith("E2-")
        assert value.endswith(f"-C{context}")


def test_sha_tree_excludes_its_own_checkpoint_manifest(tmp_path):
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "weights.bin").write_bytes(b"abc")
    (tmp_path / "retrain_checkpoint_manifest.json").write_text("{}", encoding="utf-8")
    hashes = stage12._sha_tree(tmp_path)
    assert set(hashes) == {"config.json", "weights.bin"}


def test_sha_tree_can_exclude_itft_manifest_name(tmp_path):
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "itft_checkpoint_manifest.json").write_text("{}", encoding="utf-8")
    hashes = stage12._sha_tree(tmp_path, excluded_names=("itft_checkpoint_manifest.json",))
    assert set(hashes) == {"config.json"}


def test_verify_hash_index_accepts_matching_identity_and_digest(tmp_path):
    payload = tmp_path / "metrics.json"
    payload.write_text('{"ok":true}', encoding="utf-8")
    import hashlib
    digest = hashlib.sha256(payload.read_bytes()).hexdigest()
    index = tmp_path / "artifacts_sha256.json"
    index.write_text(json.dumps({"stage": 12, "run_id": "left_truncation_v2", "sha256": {"metrics.json": digest}}), encoding="utf-8")
    loaded = stage12._verify_hash_index(
        tmp_path, index,
        expected_identity={"stage": 12, "run_id": "left_truncation_v2"},
        files=("metrics.json",),
    )
    assert loaded["sha256"]["metrics.json"] == digest


def test_verify_hash_index_rejects_modified_artifact(tmp_path):
    payload = tmp_path / "metrics.json"
    payload.write_text("first", encoding="utf-8")
    import hashlib
    digest = hashlib.sha256(payload.read_bytes()).hexdigest()
    index = tmp_path / "artifacts_sha256.json"
    index.write_text(json.dumps({"stage": 12, "sha256": {"metrics.json": digest}}), encoding="utf-8")
    payload.write_text("modified", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        stage12._verify_hash_index(tmp_path, index, expected_identity={"stage": 12}, files=("metrics.json",))


def test_legacy_output_discovery_finds_flat_previous_pipeline_paths(tmp_path):
    legacy = tmp_path / "results/tables/e2_full_retrain_metrics.csv"
    legacy.parent.mkdir(parents=True)
    legacy.write_text("old", encoding="utf-8")
    found = stage12._legacy_paths(tmp_path)
    assert legacy.resolve() in found


def test_legacy_output_guard_raises_if_old_outputs_exist(tmp_path):
    legacy = tmp_path / "models/E2_full_retrained"
    legacy.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="Legacy outputs exist"):
        stage12._assert_no_legacy_outputs(tmp_path)


def test_training_arguments_use_fixed_epochs_context_rate_and_no_evaluation():
    class FakeTrainingArguments:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    result = stage12._make_training_arguments(
        FakeTrainingArguments,
        output_dir=Path("/tmp/test-output"),
        learning_rate=3e-5,
        use_cuda=False,
        seed=42,
        disable_tqdm=True,
    )
    args = result.kwargs
    assert args["num_train_epochs"] == 8
    assert args["learning_rate"] == pytest.approx(3e-5)
    assert args["save_strategy"] == "no"
    assert args.get("eval_strategy", args.get("evaluation_strategy")) == "no"
    assert args["per_device_train_batch_size"] == 16
    assert args["fp16"] is False


def test_context_input_is_parent_first_target_last_and_left_truncation_is_required():
    row = {
        "parent_4": "oldest",
        "parent_3": "older",
        "parent_2": "recent-2",
        "parent_1": "recent-1",
        "target_text": "TARGET",
        "context_size_available": 4,
    }
    text = stage12.eu.build_context_text(row, 4, sep=" </s> ")
    assert text.index("oldest") < text.index("older") < text.index("recent-2") < text.index("recent-1") < text.index("TARGET")
    config = {"truncation_side": "left"}
    stage12.eu.validate_truncation_policy(config, input_mode="contextual")
    with pytest.raises(ValueError, match="requires truncation_side='left'"):
        stage12.eu.validate_truncation_policy({"truncation_side": "right"}, input_mode="contextual")


def test_stage11_decision_loader_fails_when_bundle_is_absent(tmp_path):
    with pytest.raises(FileNotFoundError, match="Stage 11 decision bundle is incomplete"):
        stage12._verify_stage11_decision(tmp_path)
