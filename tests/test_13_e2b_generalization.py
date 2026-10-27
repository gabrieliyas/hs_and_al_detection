"""Unit tests for Stage 13 descriptive shallow-pool evaluation.

These tests do not require Stage 11/12 outputs, model downloads, or a GPU.
They exercise deterministic population logic, metrics, alignment, hashing, and
legacy/stale-output safeguards.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "13_e2b_generalization.py"
SPEC = importlib.util.spec_from_file_location("stage13_e2b_generalization", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
stage13 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stage13)


def _small_shallow() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"post_id": "p1", "target_text": "target one", "label": "N", "context_size_available": 1,
             "parent_4": "", "parent_3": "", "parent_2": "", "parent_1": "parent one"},
            {"post_id": "p2", "target_text": "target two", "label": "AL", "context_size_available": 2,
             "parent_4": "", "parent_3": "", "parent_2": "parent two older", "parent_1": "parent two recent"},
            {"post_id": "p3", "target_text": "target three", "label": "HS", "context_size_available": 3,
             "parent_4": "", "parent_3": "parent three oldest", "parent_2": "parent three older", "parent_1": "parent three recent"},
        ]
    )


def test_authoritative_stage_script_and_namespace():
    assert stage13.STAGE == 13
    assert stage13.SCRIPT_NAME == "scripts/13_e2b_generalization.py"
    assert stage13.eu.PIPELINE_SCRIPTS[13] == stage13.SCRIPT_NAME
    assert stage13.TRUNCATION_SIDE == "left"
    assert stage13.STAGE_ROOT_PARTS.as_posix().endswith("stage_13")


@pytest.mark.parametrize(
    ("condition", "expected_ids"),
    [(0, ["p1", "p2", "p3"]), (1, ["p1", "p2", "p3"]), (2, ["p2", "p3"]), (3, ["p3"]), (4, [])],
)
def test_eligible_population_follows_protocol(condition, expected_ids):
    population = stage13.eligible_population(_small_shallow(), condition)
    assert population["post_id"].tolist() == expected_ids


def test_eligible_population_rejects_invalid_condition():
    with pytest.raises(ValueError, match="condition must be 0-4"):
        stage13.eligible_population(_small_shallow(), 5)


def test_label_ids_uses_canonical_mapping_and_rejects_unknown_label():
    actual = stage13.label_ids(pd.Series(["N", "AL", "HS"]))
    np.testing.assert_array_equal(actual, [0, 1, 2])
    with pytest.raises(ValueError, match="Cannot map labels"):
        stage13.label_ids(pd.Series(["Other"]))


def test_overall_metrics_and_empty_population():
    metrics = stage13.overall_metrics(np.array([0, 1, 2]), np.array([0, 2, 2]))
    assert metrics["n_examples"] == 3
    assert metrics["accuracy"] == pytest.approx(2 / 3)
    assert stage13.overall_metrics(np.array([], dtype=int), np.array([], dtype=int))["n_examples"] == 0
    assert stage13.overall_metrics(np.array([], dtype=int), np.array([], dtype=int))["macro_f1"] is None


def test_class_and_confusion_tables_cover_all_three_classes():
    y_true = np.array([0, 1, 2, 0])
    y_pred = np.array([0, 2, 2, 1])
    class_df = stage13.class_metrics_table(2, y_true, y_pred)
    cm_df = stage13.confusion_table(2, y_true, y_pred)
    assert class_df["class_label"].tolist() == ["N", "AL", "HS"]
    assert class_df["support"].tolist() == [2, 1, 1]
    assert len(cm_df) == 9
    assert set(cm_df["true_class"]) == {"N", "AL", "HS"}
    assert int(cm_df["count"].sum()) == len(y_true)


def test_availability_breakdown_only_reports_present_strata():
    shallow = _small_shallow()
    y_true = stage13.label_ids(shallow["label"])
    y_pred = y_true.copy()
    result = stage13.availability_breakdown(0, shallow, y_true, y_pred)
    assert result["parent_count_available"].tolist() == [1, 2, 3]
    assert result["n_examples"].tolist() == [1, 1, 1]
    assert (result["macro_f1"] == 1.0).all()


def test_pairwise_comparisons_use_identical_eligible_post_ids():
    shallow = _small_shallow()
    ids0 = ["p1", "p2", "p3"]
    ids1 = ["p3", "p1", "p2"]  # shuffled output order is aligned by post_id
    ids2 = ["p2", "p3"]
    ids3 = ["p3"]
    preds = {
        0: {"post_ids": ids0, "y_pred": np.array([0, 1, 2])},
        1: {"post_ids": ids1, "y_pred": np.array([2, 0, 1])},
        2: {"post_ids": ids2, "y_pred": np.array([1, 2])},
        3: {"post_ids": ids3, "y_pred": np.array([2])},
    }
    result = stage13.pairwise_vs_c0(shallow, preds)
    assert result["comparison"].tolist() == ["C0_vs_C1", "C0_vs_C2", "C0_vs_C3"]
    assert result["n_examples"].tolist() == [3, 2, 1]
    assert not result["significance_test_performed"].any()
    assert result.loc[0, "macro_f1_delta_deeper_minus_c0"] == pytest.approx(0.0)


def test_pairwise_comparison_rejects_missing_ids():
    shallow = _small_shallow()
    preds = {
        0: {"post_ids": ["p1", "p2"], "y_pred": np.array([0, 1])},
        1: {"post_ids": ["p1", "p2", "p3"], "y_pred": np.array([0, 1, 2])},
        2: {"post_ids": ["p2", "p3"], "y_pred": np.array([1, 2])},
        3: {"post_ids": ["p3"], "y_pred": np.array([2])},
    }
    with pytest.raises(ValueError, match="identical eligible population"):
        stage13.pairwise_vs_c0(shallow, preds)


def test_hash_index_verifies_identity_and_file_bytes(tmp_path):
    payload = tmp_path / "metrics.json"
    payload.write_text('{"ok": true}', encoding="utf-8")
    digest = stage13._sha(payload)
    index_path = tmp_path / "artifacts_sha256.json"
    index_path.write_text(json.dumps({"stage": 13, "run_id": stage13.eu.RUN_ID, "sha256": {"metrics.json": digest}}), encoding="utf-8")
    index = stage13._verify_index_entries(index_path, tmp_path, expected_identity={"stage": 13, "run_id": stage13.eu.RUN_ID})
    assert index["sha256"]["metrics.json"] == digest
    payload.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="Artifact hash mismatch"):
        stage13._verify_index_entries(index_path, tmp_path, expected_identity={"stage": 13, "run_id": stage13.eu.RUN_ID})


def test_manifest_compatibility_rejects_unmanifested_existing_outputs(tmp_path):
    run_dir = tmp_path / "results/runs/left_truncation_v2/stage_13/e2_xlmr/C0"
    run_dir.mkdir(parents=True)
    (run_dir / "old_predictions.csv").write_text("old", encoding="utf-8")
    with pytest.raises(RuntimeError, match="without a run manifest"):
        stage13._check_manifest_compatibility(run_dir, {"fingerprint": "x"})


def test_manifest_compatibility_rejects_a_different_fingerprint(tmp_path):
    run_dir = tmp_path / "results/runs/left_truncation_v2/stage_13/e2_xlmr/C0"
    run_dir.mkdir(parents=True)
    # An invalid or old manifest is rejected before its fingerprint could be reused.
    (run_dir / stage13.eu.RUN_MANIFEST_NAME).write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError):
        stage13._check_manifest_compatibility(run_dir, {"fingerprint": "new"})


def test_legacy_guard_detects_old_flat_outputs(tmp_path):
    legacy = tmp_path / "results/raw/e2b_C0_shallow_predictions.csv"
    legacy.parent.mkdir(parents=True)
    legacy.write_text("historical", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Legacy outputs exist"):
        stage13._assert_no_legacy_outputs(tmp_path)


def test_expected_population_counts_match_protocol():
    assert stage13.EXPECTED_PARENT_COUNTS == {1: 150, 2: 300, 3: 450}
    assert {"C0": 900, "C1": 900, "C2": 750, "C3": 450, "C4": 0} == {
        f"C{c}": len(stage13.eligible_population(
            pd.DataFrame([
                {"post_id": f"p{i}", "context_size_available": p}
                for i, p in enumerate([1] * 150 + [2] * 300 + [3] * 450)
            ]), c
        )) for c in range(5)
    }
