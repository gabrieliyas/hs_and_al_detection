"""Synthetic, no-training tests for Stage 15 reporting and statistical helpers."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "15_evaluate_all.py"
spec = importlib.util.spec_from_file_location("stage15_evaluate_all_under_test", SCRIPT_PATH)
stage15 = importlib.util.module_from_spec(spec)
assert spec and spec.loader
import sys
sys.modules[spec.name] = stage15
spec.loader.exec_module(stage15)


def _make_probs(y: np.ndarray, *, flip_every: int = 0, flip_to: int = 1) -> np.ndarray:
    pred = np.asarray(y, dtype=int).copy()
    if flip_every > 0:
        indexes = np.arange(0, len(pred), flip_every)
        pred[indexes] = (pred[indexes] + flip_to) % 3
    probs = np.full((len(pred), 3), 0.05, dtype=float)
    probs[np.arange(len(pred)), pred] = 0.90
    probs /= probs.sum(axis=1, keepdims=True)
    return probs


def _metric_payload(y: np.ndarray, probs: np.ndarray, ids: list[str]) -> dict:
    pred = probs.argmax(axis=1)
    return {
        "post_ids": ids,
        "y_true": y.copy(),
        "probabilities_by_seed": {42: probs.copy(), 123: probs.copy()},
        "ensemble_probabilities": probs.copy(),
        "ensemble_predictions": pred,
        "metrics": stage15.classification_metrics(y, pred),
    }


def _synthetic_inputs():
    n = 900
    ids = [f"ctx-{i:04d}" for i in range(n)]
    y = np.tile(np.array([0, 1, 2], dtype=int), n // 3)
    matched = pd.DataFrame({
        "post_id": ids, "label": [stage15.eu.ID2LABEL[int(v)] for v in y],
        "context_size_available": 4, "context_needed": np.tile([0, 1, 0], n // 3),
        "fold": np.repeat(np.arange(5), 180),
        "target_text": [f"target text {i}" for i in range(n)],
        "parent_4": "parent 4", "parent_3": "parent 3", "parent_2": "parent 2", "parent_1": "parent 1",
    })
    shallow_n = 900
    shallow_ids = [f"shallow-{i:04d}" for i in range(shallow_n)]
    shallow_y = np.tile(np.array([0, 1, 2], dtype=int), shallow_n // 3)
    avail = np.repeat([1, 2, 3], [150, 300, 450])
    shallow = pd.DataFrame({
        "post_id": shallow_ids, "label": [stage15.eu.ID2LABEL[int(v)] for v in shallow_y],
        "context_size_available": avail, "context_needed": np.tile([0, 1, 0], shallow_n // 3),
        "target_text": [f"shallow target {i}" for i in range(shallow_n)],
        "parent_4": [None] * shallow_n, "parent_3": [None] * shallow_n,
        "parent_2": [None] * shallow_n, "parent_1": ["parent 1"] * shallow_n,
    })
    benchmark_parts = {
        name: pd.DataFrame({"clean_text": [f"text {i}" for i in range(90)],
                            "label": [stage15.eu.ID2LABEL[int(v)] for v in np.tile([0, 1, 2], 30)]})
        for name in ("benchmark_train", "benchmark_val", "benchmark_test")
    }
    ids = matched.post_id.astype(str).tolist()
    fold_post_ids = {fold: matched.loc[matched.fold == fold, "post_id"].astype(str).tolist() for fold in range(5)}
    fold_y = {fold: matched.loc[matched.fold == fold, "label"].map(stage15.eu.LABEL2ID).to_numpy() for fold in range(5)}

    e1_seed_rows = []
    e1_confusions = []
    e1_probabilities = {}
    e1_predictions = {}
    for backbone in stage15.BACKBONES:
        all_probs = []
        for seed in stage15.SEEDS_E1:
            probs = _make_probs(np.tile(np.array([0, 1, 2]), 90), flip_every=(101 if seed % 2 else 131), flip_to=(1 if backbone == "xlmr" else 2))
            all_probs.append(probs)
            pred = probs.argmax(axis=1)
            metrics = stage15.classification_metrics(np.tile(np.array([0, 1, 2]), 90), pred)
            e1_seed_rows.append({"backbone": backbone, "experiment_id": stage15.eu.experiment1_id(backbone), "seed": seed, **metrics})
            e1_confusions.extend(stage15.confusion_long_rows(analysis="E1_seed", backbone=backbone, context_size=None,
                                                              y_true=np.tile(np.array([0, 1, 2]), 90), y_pred=pred, seed=seed))
        mean_probs = np.mean(np.stack(all_probs), axis=0)
        pred = mean_probs.argmax(axis=1)
        e1_probabilities[backbone] = mean_probs
        e1_predictions[backbone] = pred
        e1_confusions.extend(stage15.confusion_long_rows(analysis="E1_seed_ensemble", backbone=backbone, context_size=None,
                                                          y_true=np.tile(np.array([0, 1, 2]), 90), y_pred=pred))
    seed_df = pd.DataFrame(e1_seed_rows)
    e1_summary_rows = []
    e1_ensemble_rows = []
    for backbone in stage15.BACKBONES:
        subset = seed_df[seed_df.backbone == backbone]
        row = {"backbone": backbone, "experiment_id": stage15.eu.experiment1_id(backbone), "n_seeds": len(subset), "n_test": 270}
        for metric in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_precision", "weighted_recall", "weighted_f1"):
            row[metric + "_mean"] = float(subset[metric].mean())
            row[metric + "_sd"] = float(subset[metric].std(ddof=1))
        for label in ("N", "AL", "HS"):
            row[f"f1_{label}_mean"] = float(subset[f"f1_{label}"].mean())
            row[f"f1_{label}_sd"] = float(subset[f"f1_{label}"].std(ddof=1))
        e1_summary_rows.append(row)
        e1_ensemble_rows.append({"backbone": backbone, "experiment_id": stage15.eu.experiment1_id(backbone),
                                 **stage15.classification_metrics(np.tile(np.array([0, 1, 2]), 90), e1_predictions[backbone])})
    e1 = {
        "seed_metrics": seed_df,
        "summary": pd.DataFrame(e1_summary_rows),
        "ensemble": pd.DataFrame(e1_ensemble_rows),
        "paired": pd.DataFrame([{
            "comparison": "E1-XLMR-Target minus E1-IndoBERTweet-Target", "metric": "macro_f1",
            "observed_difference": 0.01, "ci_lower": -0.01, "ci_upper": 0.03,
            "ci_level": 95.0, "n_bootstrap": 10000, "inferential_status": "primary",
        }]),
        "confusion_rows": e1_confusions,
        "ensemble_predictions": e1_predictions,
        "ensemble_probabilities": e1_probabilities,
        "y_true": np.tile(np.array([0, 1, 2]), 90),
    }

    e2_by_backbone = {}
    context_decisions = {}
    for backbone in stage15.BACKBONES:
        pooled = {}
        run_rows = []
        scores = {}
        for c in range(5):
            # Slightly different errors by C/backbone so all summary calculations execute.
            probs = _make_probs(y, flip_every=(97 + c * 7 + (3 if backbone == "xlmr" else 0)), flip_to=(c % 2 + 1))
            pooled[c] = _metric_payload(y, probs, ids)
            scores[f"C{c}"] = pooled[c]["metrics"]["macro_f1"]
            for fold in range(5):
                for seed in stage15.SEEDS_CV:
                    run_metrics = stage15.classification_metrics(y, pooled[c]["ensemble_predictions"])
                    run_rows.append({"backbone": backbone, "context_size": c, "fold": fold, "seed": seed,
                                     **run_metrics,
                                     "macro_f1": pooled[c]["metrics"]["macro_f1"] - 0.001 * fold + 0.0001 * (seed == 123)})
        best = 2 if backbone == "indobertweet" else 3
        ctx_doc = {
            "provisional_top_context": 4, "selected_context": best,
            "pooled_oof_macro_f1": scores, "n_oof_examples": 900,
            "selection_reason": f"test selection reason for C{best}",
            "paired_comparisons_top_minus_shallower": {
                "C4_minus_C2": {"observed_diff": 0.01, "ci_lower": -0.01, "ci_upper": 0.02, "ci_level": 95.0, "n_bootstrap": 10000, "includes_zero": True},
                "C4_minus_C3": {"observed_diff": 0.005, "ci_lower": -0.001, "ci_upper": 0.012, "ci_level": 95.0, "n_bootstrap": 10000, "includes_zero": True},
                "C4_minus_C1": {"observed_diff": 0.02, "ci_lower": 0.005, "ci_upper": 0.03, "ci_level": 95.0, "n_bootstrap": 10000, "includes_zero": False},
                "C4_minus_C0": {"observed_diff": 0.04, "ci_lower": 0.02, "ci_upper": 0.06, "ci_level": 95.0, "n_bootstrap": 10000, "includes_zero": False},
            },
        }
        context_decisions[backbone] = ctx_doc
        e2_by_backbone[backbone] = {
            "pooled": pooled, "run_metrics": pd.DataFrame(run_rows),
            "root_dir": stage15.PROJECT_ROOT / f"results/runs/left_truncation_v2/stage_{9 if backbone == 'indobertweet' else 10}/e2_{backbone}",
            "root_fingerprint": f"fp-{backbone}",
        }
    stage11_decision = {
        "decision": "xlmr", "c_best_by_backbone": {"indobertweet": 2, "xlmr": 3},
        "context_decisions": context_decisions, "latency_ms": {"indobertweet": 8.0, "xlmr": 7.0},
        "reason": "synthetic test decision", "rule_version": stage15.eu.BACKBONE_RULE_VERSION,
        "primary_c_best": {"observed_diff": 0.02, "ci_lower": 0.0, "ci_upper": 0.04},
        "guard_c0": {"observed_diff": 0.01, "ci_lower": -0.01, "ci_upper": 0.03},
    }
    stage11 = {"decision": stage11_decision, "fingerprint": "fp-stage11"}

    shallow_overall_rows = []
    shallow_class_rows = []
    shallow_availability_rows = []
    shallow_confusions = []
    for c in (0, 1, 2, 3):
        pop = shallow if c in (0, 1) else shallow[shallow.context_size_available >= c]
        yy = pop.label.map(stage15.eu.LABEL2ID).to_numpy()
        probs = _make_probs(yy, flip_every=(37 + c * 5), flip_to=(c % 2 + 1))
        pred = probs.argmax(axis=1)
        met = stage15.classification_metrics(yy, pred)
        shallow_overall_rows.append({"analysis": "E2B_shallow", "backbone": "xlmr", "experiment_id": stage15.eu.experiment2_id(c, "xlmr"),
                                     "context_size": c, **met, "status": "evaluated", "population_rule": "synthetic population", "is_c_best": c == 3})
        shallow_class_rows.extend(stage15._per_class_rows("E2B_shallow", "xlmr", c, met, population_rule="synthetic"))
        shallow_confusions.extend(stage15.confusion_long_rows(analysis="E2B_shallow", backbone="xlmr", context_size=c, y_true=yy, y_pred=pred))
        for pc in sorted(pop.context_size_available.unique()):
            mask = pop.context_size_available.to_numpy() == pc
            gm = stage15.classification_metrics(yy[mask], pred[mask])
            shallow_availability_rows.append({"backbone": "xlmr", "context_size": c, "parent_count_available": int(pc),
                                               "n_examples": int(mask.sum()), "accuracy": gm["accuracy"], "macro_precision": gm["macro_precision"],
                                               "macro_recall": gm["macro_recall"], "macro_f1": gm["macro_f1"], "weighted_precision": gm["weighted_precision"],
                                               "weighted_recall": gm["weighted_recall"], "weighted_f1": gm["weighted_f1"]})
    shallow_overall_rows.append({"analysis": "E2B_shallow", "backbone": "xlmr", "experiment_id": stage15.eu.experiment2_id(4, "xlmr"),
                                 "context_size": 4, "n_examples": 0, "accuracy": np.nan, "macro_precision": np.nan, "macro_recall": np.nan,
                                 "macro_f1": np.nan, "weighted_precision": np.nan, "weighted_recall": np.nan, "weighted_f1": np.nan,
                                 "status": "not_applicable", "population_rule": "N/A", "is_c_best": False})
    e2b = {
        "overall": pd.DataFrame(shallow_overall_rows), "by_class": pd.DataFrame(shallow_class_rows),
        "by_availability": pd.DataFrame(shallow_availability_rows), "confusion_rows": shallow_confusions,
        "stage12": {"dir": stage15.PROJECT_ROOT / "results/runs/left_truncation_v2/stage_12/e2_xlmr", "fingerprint": "fp-stage12"},
        "stage13": {"dir": stage15.PROJECT_ROOT / "results/runs/left_truncation_v2/stage_13/e2_xlmr", "fingerprint": "fp-stage13"},
    }
    e3_by_condition = {}
    e3_run_rows = []
    e3_confusions = []
    for c in (0, 3):
        probs = _make_probs(y, flip_every=83 + c * 5, flip_to=(c % 2 + 1))
        e3_by_condition[c] = _metric_payload(y, probs, ids)
        e3_confusions.extend(stage15.confusion_long_rows(analysis="E3_direct_seed_ensembled_OOF", backbone="xlmr", context_size=c, y_true=y, y_pred=probs.argmax(axis=1)))
        for fold in range(5):
            for seed in stage15.SEEDS_CV:
                met = stage15.classification_metrics(fold_y[fold], probs[matched.fold.to_numpy() == fold].argmax(axis=1))
                e3_run_rows.append({"backbone": "xlmr", "context_size": c, "fold": fold, "seed": seed, **met})
    e3 = {"backbone": "xlmr", "c_best": 3, "by_condition": e3_by_condition,
          "run_metrics": pd.DataFrame(e3_run_rows), "confusion_rows": e3_confusions,
          "bundle": {"dir": stage15.PROJECT_ROOT / "results/runs/left_truncation_v2/stage_14/e3_xlmr", "fingerprint": "fp-stage14"}}
    # Include stage data for provenance rows; paths do not need to exist for the pure table-builder test.
    upstream = {
        "stage1_manifest": {"fingerprint": "fp-stage1"}, "stage1b_manifest": {"fingerprint": "fp-stage1b"},
        "stage2_manifest": {"fingerprint": "fp-stage2"},
        "stage5": {"fingerprint": "fp-stage5"}, "stage8": {"fingerprint": "fp-stage8"},
    }
    return upstream, data if False else {"matched": matched, "shallow": shallow, "benchmark_parts": benchmark_parts,
                                         "matched_ids": ids, "matched_y": y}, e1, {"by_backbone": e2_by_backbone}, stage11, e2b, e3


def test_classification_metrics_contains_core_and_per_class_metrics():
    metrics = stage15.classification_metrics([0, 1, 2, 2], [0, 2, 2, 1])
    assert metrics["n_examples"] == 4
    assert metrics["accuracy"] == pytest.approx(0.5)
    assert "macro_f1" in metrics and "weighted_f1" in metrics
    for label in ("N", "AL", "HS"):
        assert f"precision_{label}" in metrics
        assert f"recall_{label}" in metrics
        assert f"f1_{label}" in metrics
        assert f"support_{label}" in metrics
    assert metrics["micro_f1"] == pytest.approx(metrics["accuracy"])


def test_probabilities_validate_and_reject_wrong_shape_or_sums():
    valid = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]])
    assert stage15.validate_probabilities(valid, expected_rows=2, name="test").shape == (2, 3)
    with pytest.raises(ValueError, match="shape"):
        stage15.validate_probabilities(np.ones((2, 2)), expected_rows=2, name="test")
    with pytest.raises(ValueError, match="sum to 1"):
        stage15.validate_probabilities(np.array([[0.8, 0.1, 0.1], [0.2, 0.2, 0.2]]), expected_rows=2, name="test")


def test_build_oof_arrays_reorders_folds_to_canonical_ids():
    ids = [f"id-{i}" for i in range(10)]
    y = np.array([i % 3 for i in range(10)])
    folds = {0: ids[5:], 1: ids[:5]}
    fold_probs = {}
    for seed in (42, 123):
        fold_probs[(0, seed)] = _make_probs(y[5:])
        fold_probs[(1, seed)] = _make_probs(y[:5])
    pooled = stage15.build_oof_arrays(canonical_post_ids=ids, canonical_y_true=y, fold_post_ids=folds,
                                      fold_probabilities=fold_probs, seeds=(42, 123), context_size=2)
    assert pooled["post_ids"] == ids
    np.testing.assert_array_equal(pooled["ensemble_predictions"], y)
    assert pooled["metrics"]["accuracy"] == pytest.approx(1.0)


def test_build_oof_arrays_rejects_duplicate_ids_across_folds():
    ids = [f"id-{i}" for i in range(6)]
    y = np.array([i % 3 for i in range(6)])
    fold_ids = {0: ids[:4], 1: [ids[3], ids[4], ids[5]]}
    probs = {(fold, seed): _make_probs(np.arange(len(fold_ids[fold])) % 3) for fold in fold_ids for seed in (42, 123)}
    with pytest.raises(ValueError, match="multiple folds"):
        stage15.build_oof_arrays(canonical_post_ids=ids, canonical_y_true=y, fold_post_ids=fold_ids,
                                 fold_probabilities=probs, seeds=(42, 123), context_size=1)


def test_confusion_long_rows_contains_all_nine_cells():
    rows = stage15.confusion_long_rows(analysis="test", backbone="xlmr", context_size=2,
                                       y_true=[0, 1, 2, 2], y_pred=[0, 2, 2, 1])
    assert len(rows) == 9
    assert sum(row["count"] for row in rows) == 4
    assert {row["true_label"] for row in rows} == {"N", "AL", "HS"}
    assert {row["predicted_label"] for row in rows} == {"N", "AL", "HS"}


def test_context_population_distribution_reports_support_and_percentage():
    matched = pd.DataFrame({"context_size_available": [4, 4, 4], "context_needed": [0, 1, 0], "label": ["N", "AL", "N"]})
    shallow = pd.DataFrame({"context_size_available": [1, 2, 2, 3], "context_needed": [0, 1, 0, 1], "label": ["N", "AL", "N", "HS"]})
    table = stage15._context_population_distribution(matched, shallow)
    assert {"support_n", "support_pct_within_available_context", "available_context_total_n"}.issubset(table.columns)
    row = table[(table.context_size_available == 1) & (table.class_code == "N")].iloc[0]
    assert row["support_n"] == 1
    assert row["support_pct_within_available_context"] == pytest.approx(100.0)


def test_report_table_and_figure_schemas_are_registered(monkeypatch, tmp_path):
    upstream, data, e1, e2, stage11, e2b, e3 = _synthetic_inputs()
    monkeypatch.setattr(stage15, "N_BOOTSTRAP", 40)
    tables, report = stage15._make_report_tables(upstream=upstream, data=data, e1=e1, e2=e2,
                                                stage11=stage11, e2b=e2b, e3=e3)
    assert set(tables) == set(stage15.TABLE_FILENAMES)
    assert report["selected_backbone"] == "xlmr"
    assert report["selected_c_best"] == 3
    assert set(report["strategy_decisions"]) == {"target_only", "context_aware"}
    assert len(tables["e1_seed_metrics.csv"].columns) <= 8
    assert len(tables["e1_model_summary.csv"].columns) <= 10
    assert len(tables["e2_context_summary.csv"].columns) <= 12
    assert len(tables["training_strategy_comparison.csv"].columns) <= 12
    assert len(tables["confusion_matrices.csv"]) >= 9 * (10 + 10 + 4 + 2)

    figures = stage15._create_figures(tables, e1=e1, e2=e2, e2b=e2b, e3=e3,
                                      report=report, figure_output_dir=tmp_path / "figures")
    assert set(stage15.FIGURE_FILENAMES).issubset(figures)
    assert all(path.is_file() and path.stat().st_size > 0 for path in figures.values())
    assert any(name.startswith("fig_cm_") for name in figures)


def test_main_tables_include_per_class_and_detailed_appendix_tables():
    assert "e1_seed_per_class_metrics.csv" in stage15.TABLE_FILENAMES
    assert "e1_model_summary_detailed.csv" in stage15.TABLE_FILENAMES
    assert "e2_context_metrics_detailed.csv" in stage15.TABLE_FILENAMES
    assert "e2b_generalization_metrics_detailed.csv" in stage15.TABLE_FILENAMES
    assert "e3_direct_per_class.csv" in stage15.TABLE_FILENAMES
    assert "confusion_matrices.csv" in stage15.TABLE_FILENAMES


def test_stage15_uses_versioned_run_bundle_and_thesis_export_folders():
    assert stage15.STAGE_ROOT.as_posix().endswith("results/runs/left_truncation_v2/stage_15")
    assert stage15.TABLES_EXPORT_DIR.as_posix().endswith("results/tables/left_truncation_v2")
    assert stage15.FIGURES_EXPORT_DIR.as_posix().endswith("results/figures/left_truncation_v2")


def test_n_bootstrap_is_frozen_at_10000():
    assert stage15.N_BOOTSTRAP == 10_000
    assert stage15.CI_LEVEL == 95.0
