from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "17_train_final_models.py"
SPEC = importlib.util.spec_from_file_location("stage17_train_final_models", SCRIPT)
stage17 = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(stage17)


class DummyTokenizer:
    sep_token = "</s>"


def make_row(post_id: str, available: int) -> dict[str, object]:
    return {
        "post_id": post_id,
        "parent_4": "parent four" if available >= 4 else "",
        "parent_3": "parent three" if available >= 3 else "",
        "parent_2": "parent two" if available >= 2 else "",
        "parent_1": "parent one",
        "target_text": f"target {post_id}",
        "context_size_available": available,
        "label": "N",
    }


def test_effective_context_clamps_to_available_parent_count():
    assert stage17._effective_context_size(pd.Series(make_row("p1", 1)), 4) == 1
    assert stage17._effective_context_size(pd.Series(make_row("p2", 2)), 3) == 2
    assert stage17._effective_context_size(pd.Series(make_row("p3", 4)), 3) == 3
    assert stage17._effective_context_size(pd.Series(make_row("p4", 4)), 0) == 0


def test_build_input_texts_preserves_target_and_uses_tokenizer_separator_for_shallow_rows():
    rows = pd.DataFrame([make_row("shallow", 1), make_row("deep", 4)])
    texts, sizes = stage17.build_input_texts(rows, 3, DummyTokenizer())
    assert sizes == [1, 3]
    assert texts[0] == "parent one </s> target shallow"
    assert texts[1] == "parent three </s> parent two </s> parent one </s> target deep"
    assert all(text.endswith("target " + post_id) for text, post_id in zip(texts, ["shallow", "deep"]))


def test_target_only_input_does_not_require_parent_availability_for_composition():
    rows = pd.DataFrame([make_row("p1", 1), make_row("p2", 4)])
    texts, sizes = stage17.build_input_texts(rows, 0, DummyTokenizer())
    assert sizes == [0, 0]
    assert texts == ["target p1", "target p2"]


def test_invalid_context_size_is_rejected():
    rows = pd.DataFrame([make_row("p1", 1)])
    with pytest.raises(ValueError, match="0 through 4"):
        stage17.build_input_texts(rows, 5, DummyTokenizer())


def test_missing_parent_availability_fails_closed():
    row = pd.Series(make_row("p1", 1))
    row["context_size_available"] = None
    with pytest.raises(ValueError, match="Missing context_size_available"):
        stage17._effective_context_size(row, 3)


def test_stage17_registry_matches_requested_filename():
    import experiment_utils as eu

    assert stage17.SCRIPT_NAME == "scripts/17_train_final_models.py"
    assert eu.PIPELINE_SCRIPTS[17] == stage17.SCRIPT_NAME
    assert eu.SCRIPT_RENAMES["scripts/14_train_final_models.py"] == stage17.SCRIPT_NAME


def test_stage17_does_not_require_cv_fold_hash_for_full_dataset_config():
    import experiment_utils as eu

    # Only the contextual-fold requirement is waived for Stage 17; all ordinary
    # identity, model/tokenizer pin and shared length/LR requirements still apply.
    # Here use no shared config to isolate the full-data-vs-CV distinction.
    config = {
        "schema_version": eu.CONFIG_SCHEMA_VERSION,
        "run_id": eu.RUN_ID,
        "stage": 17,
        "script_name": stage17.SCRIPT_NAME,
        "backbone": "indobertweet",
        "hf_model_id": eu.get_backbone("indobertweet").hf_model_id,
        "dataset_sha256": "a" * 64,
        "tokenizer_revision": "b" * 40,
        "model_revision": "b" * 40,
        "truncation_side": "left",
    }
    # load_backbone_decision_document is normally required by Stage 17 validation;
    # this test isolates the fold-annotation rule by monkeypatching only that decision.
    original_loader = eu.load_backbone_decision_document
    original_strategy = eu.load_training_strategy_decision
    try:
        eu.load_backbone_decision_document = lambda: {"decision": "indobertweet"}
        eu.load_training_strategy_decision = lambda: {"by_mode": {"target_only": {"strategy": "direct"}, "context_aware": {"strategy": "direct"}}}
        eu.validate_resolved_run_config(config, stage=17, backbone="indobertweet", input_mode="contextual", context_size=0, require_shared_config=False)
    finally:
        eu.load_backbone_decision_document = original_loader
        eu.load_training_strategy_decision = original_strategy
