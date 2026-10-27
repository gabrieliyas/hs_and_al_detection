"""Unit tests for Stage 16's deterministic case/outcome/sample helpers."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "16_error_analysis.py"
SPEC = importlib.util.spec_from_file_location("stage16_error_analysis", SCRIPT)
stage16 = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(stage16)


def _soft_probs(preds: list[int], correct_mass: float = 0.80) -> np.ndarray:
    rows = []
    for pred in preds:
        row = np.full(3, (1.0 - correct_mass) / 2)
        row[pred] = correct_mass
        rows.append(row)
    return np.asarray(rows, dtype=float)


def _fake_upstream(*, c_best: int = 2) -> dict:
    labels = [0, 1, 2] * 4
    # Outcome patterns include both correct, context fixes, context-introduced,
    # and both-wrong cases, with each gold class represented.
    pred0 = [0, 1, 2, 1, 1, 0, 0, 2, 2, 0, 0, 1]
    pred_best = [0, 2, 2, 1, 1, 1, 2, 2, 0, 0, 1, 1]
    fold = [i % 5 for i in range(len(labels))]
    matched = pd.DataFrame({
        "post_id": [f"p{i:03d}" for i in range(len(labels))],
        "label": [{0: "N", 1: "AL", 2: "HS"}[v] for v in labels],
        "fold": fold,
        "parent_4": [f"parent4-{i}" for i in range(len(labels))],
        "parent_3": [f"parent3-{i}" for i in range(len(labels))],
        "parent_2": [f"parent2-{i}" for i in range(len(labels))],
        "parent_1": [f"parent1-{i}" for i in range(len(labels))],
        "target_text": [f"target-{i}" for i in range(len(labels))],
        "context_type": ["reply", "quotation", "sarcasm"] * 4,
        "context_needed": [1, 0, 1] * 4,
        "context_size_available": [4] * len(labels),
        "target_group": ["group-a"] * len(labels),
        "target_group_attribute": ["attribute-x"] * len(labels),
        "collection_method": ["method"] * len(labels),
        "keyword_used": ["keyword"] * len(labels),
        "post_date": ["2025-01-01"] * len(labels),
    })
    probs0 = _soft_probs(pred0)
    probs_best = _soft_probs(pred_best)
    if c_best == 0:
        probs_best = probs0.copy()
    return {
        "matched": matched,
        "gold": np.asarray(labels, dtype=np.int64),
        "gold_ids": matched["post_id"].tolist(),
        "folds": np.asarray(fold, dtype=np.int64),
        "backbone": "indobertweet",
        "c_best": c_best,
        "same_condition": c_best == 0,
        "probabilities": {0: probs0, c_best: probs_best},
    }


def test_outcome_labels_use_gold_labels_and_distinguish_error_directions():
    gold = np.array([0, 1, 2, 1, 0])
    c0 = np.array([0, 0, 2, 1, 1])
    cb = np.array([0, 1, 2, 2, 2])
    outcomes = stage16._outcome_labels(gold, c0, cb, same_condition=False)
    assert outcomes == ["both_correct", "context_fixed_error", "both_correct", "context_introduced_error", "both_wrong"]


def test_same_context_outcome_does_not_claim_context_gain_or_loss():
    gold = np.array([0, 1, 2])
    pred = np.array([0, 2, 2])
    assert stage16._outcome_labels(gold, pred, pred, same_condition=True) == [
        "same_condition_correct", "same_condition_wrong", "same_condition_correct"
    ]


def test_ledger_includes_gold_predictions_probabilities_and_original_text():
    upstream = _fake_upstream()
    ledger = stage16._build_case_ledger(upstream)
    assert len(ledger) == len(upstream["matched"])
    assert set(ledger["gold_label"]) == {"N", "AL", "HS"}
    assert set(ledger["comparison_outcome"]) == {"both_correct", "context_fixed_error", "context_introduced_error", "both_wrong"}
    assert {"c0_prob_N", "c0_prob_AL", "c0_prob_HS", "c_best_prob_N", "c_best_prob_AL", "c_best_prob_HS"}.issubset(ledger.columns)
    assert ledger.loc[0, "target_text"] == "target-0"
    assert ledger.loc[0, "parent_1"] == "parent1-0"
    assert np.allclose(ledger["c0_confidence"], 0.8)
    assert np.array_equal(ledger["c0_correct"].to_numpy(dtype=bool), (upstream["probabilities"][0].argmax(axis=1) == upstream["gold"]))


def test_review_sample_is_reproducible_capped_and_covers_nonempty_outcomes():
    ledger = stage16._build_case_ledger(_fake_upstream())
    a, details_a = stage16._select_review_sample(ledger, target_n=8, seed=17)
    b, details_b = stage16._select_review_sample(ledger, target_n=8, seed=17)
    assert len(a) == 8
    assert a["post_id"].tolist() == b["post_id"].tolist()
    assert details_a == details_b
    assert set(a["comparison_outcome"]) == {"both_correct", "context_fixed_error", "context_introduced_error", "both_wrong"}
    assert a["post_id"].is_unique
    assert (a["adjudication_status"] == "not_reviewed").all()
    assert a["primary_error_mechanism"].eq("").all()
    capped, details = stage16._select_review_sample(ledger, target_n=999, seed=17)
    assert len(capped) == len(ledger)
    assert details["effective_n"] == len(ledger)


def test_metric_tables_and_context_cross_tabs_include_gold_label_denominators(monkeypatch):
    monkeypatch.setattr(stage16, "N_BOOTSTRAP", 75)
    upstream = _fake_upstream()
    ledger = stage16._build_case_ledger(upstream)
    metrics = stage16._metrics_tables(ledger, upstream)
    assert set(metrics["evaluation_summary"]["evaluation_model"]) == {"C0", "C_best"}
    assert set(metrics["per_class_metrics"]["class_code"]) == {"N", "AL", "HS"}
    assert len(metrics["confusion_matrix"]) == 18
    assert set(metrics["paired_bootstrap"]["metric"]) == {"macro_f1", "f1_N", "f1_AL", "f1_HS"}
    outcomes = stage16._outcome_tables(ledger, upstream)
    assert outcomes["outcome_summary"]["count"].sum() == len(ledger)
    assert set(outcomes["outcome_by_gold_label"]["gold_label"]) == {"N", "AL", "HS"}
    context = stage16._context_type_tables(ledger, upstream)
    assert context["context_type_crosstab"]["n_examples"].sum() == len(ledger)
    assert {"gold_N_n", "gold_AL_n", "gold_HS_n", "context_fixed_error_n", "context_introduced_error_n"}.issubset(context["context_type_crosstab"].columns)
    assert context["context_type_by_gold"]["n_examples"].sum() == len(ledger)


def test_codebook_preserves_dataset_context_types_and_candidate_mechanisms():
    ledger = stage16._build_case_ledger(_fake_upstream())
    codebook = stage16._make_codebook(ledger)
    families = set(codebook["code_family"])
    assert "dataset_context_type" in families
    assert "candidate_error_mechanism" in families
    assert "review_status" in families
    assert set(codebook.loc[codebook["code_family"] == "dataset_context_type", "code"]) == {"reply", "quotation", "sarcasm"}
    assert len(codebook[codebook["code_family"] == "candidate_error_mechanism"]) == 7


def test_c0_best_edge_case_marks_context_comparison_unavailable():
    upstream = _fake_upstream(c_best=0)
    ledger = stage16._build_case_ledger(upstream)
    assert ledger["comparison_available"].eq(False).all()
    assert set(ledger["comparison_outcome"]) == {"same_condition_correct", "same_condition_wrong"}
    context_tables = stage16._context_type_tables(ledger, upstream)
    assert context_tables["context_type_crosstab"]["comparison_available"].eq(False).all()


def test_finalize_review_excludes_correct_controls_from_error_mechanism_denominator(tmp_path, monkeypatch):
    monkeypatch.setattr(stage16, "REVIEW_DIR", tmp_path / "manual_review")
    stage_root = tmp_path / "stage_16"
    stage_root.mkdir()
    monkeypatch.setattr(stage16, "STAGE_ROOT", stage_root)
    monkeypatch.setattr(stage16, "TABLES_DIR", tmp_path / "tables")
    monkeypatch.setattr(stage16, "_verify_stage_bundle", lambda *args, **kwargs: {"fingerprint": "fake-fp"})
    path = tmp_path / "reviewed.csv"
    frame = pd.DataFrame([
        {"post_id": "control-1", "comparison_outcome": "both_correct", "context_type": "reply", "gold_label": "N",
         "primary_error_mechanism": "NOT_APPLICABLE_NO_ERROR", "secondary_error_mechanism": "", "reviewer_confidence": "high",
         "review_notes": "Control case", "adjudication_status": "reviewed"},
        {"post_id": "error-1", "comparison_outcome": "context_fixed_error", "context_type": "quotation", "gold_label": "AL",
         "primary_error_mechanism": "REFERENT_OR_TARGET_RESOLUTION", "secondary_error_mechanism": "", "reviewer_confidence": "medium",
         "review_notes": "Parent clarifies target", "adjudication_status": "reviewed"},
        {"post_id": "error-2", "comparison_outcome": "both_wrong", "context_type": "sarcasm", "gold_label": "HS",
         "primary_error_mechanism": "OTHER", "secondary_error_mechanism": "", "reviewer_confidence": "low",
         "review_notes": "Unlisted mechanism", "adjudication_status": "adjudicated"},
        {"post_id": "error-3", "comparison_outcome": "context_introduced_error", "context_type": "reply", "gold_label": "N",
         "primary_error_mechanism": "", "secondary_error_mechanism": "", "reviewer_confidence": "",
         "review_notes": "", "adjudication_status": "not_reviewed"},
    ])
    frame.to_csv(path, index=False)
    # The immutable template carries the same identity fields; only reviewer
    # annotation columns differ in the edited worksheet.
    template = frame[["post_id", "comparison_outcome", "context_type", "gold_label"]].copy()
    template["c0_predicted_label"] = ["N", "N", "AL", "N"]
    template["c_best_predicted_label"] = ["N", "AL", "HS", "N"]
    template.to_csv(stage_root / "stage16_manual_review_sample_template.csv", index=False)
    (stage_root / stage16.eu.RUN_MANIFEST_NAME).write_text('{"fingerprint":"fake-fp"}', encoding="utf-8")
    # The worksheet may annotate values but cannot rewrite these model/data fields.
    frame["c0_predicted_label"] = template["c0_predicted_label"]
    frame["c_best_predicted_label"] = template["c_best_predicted_label"]
    frame.to_csv(path, index=False)
    summary_doc = stage16._summarize_review_annotations(path)
    assert summary_doc["completed_reviewed_cases_n"] == 3
    assert summary_doc["mechanism_coded_error_cases_n"] == 2
    summary = pd.read_csv(tmp_path / "manual_review" / "stage16_error_mechanism_summary.csv")
    assert summary["mechanism_coded_error_cases_n"].sum() == 2
    coverage = pd.read_csv(tmp_path / "manual_review" / "stage16_review_coverage_summary.csv")
    row = coverage.loc[coverage["comparison_outcome"] == "both_correct"].iloc[0]
    assert row["sample_cases_n"] == 1
    assert row["mechanism_coded_error_cases_n"] == 0
    assert (tmp_path / "tables" / "stage16_error_mechanism_summary.csv").is_file()
