"""Independent Stage 11 helper tests; these require no model downloads or GPU."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "11_select_backbone.py"
spec = importlib.util.spec_from_file_location("stage11_select_backbone", SCRIPT_PATH)
assert spec is not None and spec.loader is not None
stage11 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stage11)


def test_stage_and_run_directory_naming_are_backbone_specific():
    ib = stage11._expected_run_dir("indobertweet", 4, 2, 123)
    xlmr = stage11._expected_run_dir("xlmr", 4, 2, 123)
    assert ib.as_posix().endswith("stage_09/e2_indobertweet/C4/fold_02/seed_123")
    assert xlmr.as_posix().endswith("stage_10/e2_xlmr/C4/fold_02/seed_123")
    assert stage11.SCRIPT_NAME == "scripts/11_select_backbone.py"


def test_pool_seed_probabilities_restores_canonical_row_order():
    # Fold 0 holds rows 0 and 2; fold 1 holds rows 1 and 3.
    indices = [np.array([0, 2]), np.array([1, 3]), np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int)]
    p0 = np.array([[0.90, 0.05, 0.05], [0.10, 0.80, 0.10]])
    p1 = np.array([[0.05, 0.90, 0.05], [0.10, 0.10, 0.80]])
    empty_probs = np.empty((0, 3), dtype=float)
    pooled = stage11._pool_seed_probabilities(indices, [p0, p1, empty_probs, empty_probs, empty_probs], expected_rows=4)
    np.testing.assert_allclose(pooled, np.array([p0[0], p1[0], p0[1], p1[1]]))


def test_pool_seed_probabilities_rejects_duplicate_or_wrong_fold_count():
    probs = np.tile(np.array([[0.7, 0.2, 0.1]]), (2, 1))
    with pytest.raises(ValueError, match="exactly one held-out array"):
        stage11._pool_seed_probabilities([np.array([0, 1])], [probs], expected_rows=2)
    with pytest.raises(ValueError, match="appears in more than one held-out fold"):
        stage11._pool_seed_probabilities(
            [np.array([0, 1]), np.array([1, 2]), np.array([3]), np.array([4]), np.array([5])],
            [np.tile([1/3, 1/3, 1/3], (2, 1)), np.tile([1/3, 1/3, 1/3], (2, 1)),
             np.tile([1/3, 1/3, 1/3], (1, 1)), np.tile([1/3, 1/3, 1/3], (1, 1)),
             np.tile([1/3, 1/3, 1/3], (1, 1))],
            expected_rows=6,
        )


def test_latency_selection_uses_each_backbones_own_c_best():
    medians = {
        f"C{i}": {"indobertweet": 10.0 + i, "xlmr": 20.0 + 2 * i}
        for i in range(5)
    }
    selected = stage11._latency_at_selected_context(
        medians, {"indobertweet": 2, "xlmr": 4}
    )
    assert selected == {"indobertweet": 12.0, "xlmr": 28.0}


def test_latency_selection_rejects_missing_or_nonpositive_medians():
    with pytest.raises(ValueError, match="exactly"):
        stage11._latency_at_selected_context({"C0": {}}, {"indobertweet": 0})
    with pytest.raises(ValueError, match="finite and positive"):
        stage11._latency_at_selected_context(
            {"C1": {"indobertweet": 0, "xlmr": 3}},
            {"indobertweet": 1, "xlmr": 1},
        )


def test_constants_match_the_frozen_stage11_grid():
    assert tuple(stage11.CONTEXTS) == (0, 1, 2, 3, 4)
    assert tuple(stage11.FOLDS) == (0, 1, 2, 3, 4)
    assert tuple(stage11.SEEDS) == (42, 123)
    assert stage11.EXPECTED_RUNS_PER_BACKBONE == 50
    assert stage11.BOOTSTRAP_REPLICATES == 10_000
    assert stage11.CONFIDENCE_LEVEL == 95.0
