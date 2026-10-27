"""Model-free regression tests for Stage 8 contextual preflight logic.

These tests exercise deterministic split construction, context-length selection,
configuration parity, learning-rate selection/publication, and summary assembly.
They deliberately do not download models or execute GPU training.
"""
from __future__ import annotations

import contextlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "8_timing_pilot.py"
spec = importlib.util.spec_from_file_location("stage8_timing_pilot", SCRIPT_PATH)
assert spec is not None and spec.loader is not None
stage8 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stage8)
eu = stage8.eu


def _matched_df() -> pd.DataFrame:
    rows = []
    # Exactly 60 examples of each class in every fold: 900 rows, balanced
    # labels, five equally sized folds, and non-empty parent/target text.
    for fold in range(5):
        for label in ("N", "AL", "HS"):
            for within in range(60):
                i = fold * 180 + (0 if label == "N" else 60 if label == "AL" else 120) + within
                rows.append({
                    "post_id": f"post-{i:04d}",
                    "parent_4": f"oldest context {i}",
                    "parent_3": f"context three {i}",
                    "parent_2": f"context two {i}",
                    "parent_1": f"context one {i}",
                    "target_text": f"target message {i}",
                    "context_size_available": 4,
                    "label": label,
                    "fold": fold,
                })
    return pd.DataFrame(rows).sample(frac=1, random_state=77).reset_index(drop=True)


def _audit(matched: pd.DataFrame | None = None) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, str]:
    matched = matched if matched is not None else _matched_df()
    fold_audit = eu.validate_context_dataset(matched, expected_rows=900)
    modelling = eu.load_modelling_view(matched, include_fold=True)
    cv_train, tuning_train, tuning_val = stage8.make_context_tuning_split(modelling, 314159)
    audit = {
        "sequence_length_doc": {"max_length_benchmark": 128},
        "context_length_provenance": {},
        "model_revisions": {"indobertweet": "a" * 40, "xlmr": "b" * 40},
        "tokenizer_revisions": {"indobertweet": "a" * 40, "xlmr": "b" * 40},
        "benchmark_lr": {"indobertweet": 2e-5, "xlmr": 3e-5},
        "tuning_seed": 314159,
        "sequence_config_sha256_before_stage8": "c" * 64,
        "learning_rate_config_sha256_before_stage8": "d" * 64,
        "matched_sha256": eu.sha256_json({"matched": "synthetic"}),
    }
    return audit, modelling, cv_train, tuning_train, tuning_val, fold_audit


def _latency_document(repeats: int = 30) -> dict:
    medians = {
        f"C{context}": {
            "indobertweet": 2.0 + context * 0.5,
            "xlmr": 3.0 + context * 0.7,
        }
        for context in range(5)
    }
    summary_by_context = {
        context: {
            "backbones": {
                backbone: {"p95_ms": medians[context][backbone] * 1.2}
                for backbone in eu.BACKBONES
            }
        }
        for context in medians
    }
    measurements = {
        context: {
            backbone: [medians[context][backbone] * (0.9 + 0.2 * i / max(repeats - 1, 1)) for i in range(repeats)]
            for backbone in eu.BACKBONES
        }
        for context in medians
    }
    return {
        "measurement_machine": "Synthetic CPU test runner",
        "measurement_repeats_per_backbone_context": repeats,
        "median_ms_by_context": medians,
        "summary_by_context": summary_by_context,
        "measurements_ms_by_context": measurements,
        "sample_post_ids_sha256": eu.sha256_json([f"post-{i}" for i in range(repeats)]),
        "itft_manifest_fingerprints": {b: f"{i:064x}" for i, b in enumerate(eu.BACKBONES, start=1)},
    }


def test_pipeline_stage_and_namespace_match_shared_utility():
    assert stage8.STAGE == 8
    assert stage8.SCRIPT_NAME == "scripts/8_timing_pilot.py"
    assert eu.PIPELINE_SCRIPTS[8] == stage8.SCRIPT_NAME
    assert stage8.eu.RUN_ID == "left_truncation_v2"
    assert stage8.PLANNED_RUNS_PER_BACKBONE == 50


def test_context_tuning_split_excludes_fold_zero_and_is_deterministic():
    _, matched, cv_train, train_a, val_a, _ = _audit()
    _, train_b, val_b = stage8.make_context_tuning_split(matched, 314159)
    assert len(cv_train) == 720
    assert len(train_a) == 576
    assert len(val_a) == 144
    assert 0 not in set(cv_train["fold"])
    assert 0 not in set(train_a["fold"])
    assert 0 not in set(val_a["fold"])
    assert set(train_a.post_id).isdisjoint(set(val_a.post_id))
    assert train_a.post_id.tolist() == train_b.post_id.tolist()
    assert val_a.post_id.tolist() == val_b.post_id.tolist()
    assert train_a["label"].value_counts().to_dict() == {"N": 192, "AL": 192, "HS": 192}
    assert val_a["label"].value_counts().to_dict() == {"N": 48, "AL": 48, "HS": 48}


def test_context_tuning_split_fails_when_fold_zero_is_missing():
    matched = _matched_df().loc[lambda df: df["fold"] != 0].copy()
    with pytest.raises(ValueError, match="full 900-row matched subset"):
        stage8.make_context_tuning_split(matched, 314159)


def test_load_matched_context_data_validates_schema_and_returns_hash(tmp_path, monkeypatch):
    matched = _matched_df()
    path = tmp_path / "ctx_matched_subset.csv"
    matched.to_csv(path, index=False)
    modelling, audit, digest = stage8._load_matched_context_data(path)
    assert len(modelling) == 900
    assert set(modelling.columns) == {eu.CTX_ID_COLUMN, *eu.CTX_MODELLING_COLUMNS, eu.CTX_FOLD_COLUMN}
    assert audit["fold_counts"] == {str(i): 180 for i in range(5)}
    assert digest == eu.sha256_file(path)


def test_token_length_stats_are_computed_from_positive_lengths():
    stats = stage8._token_length_stats([8, 16, 24, 32, 40])
    assert stats["n"] == 5
    assert stats["min"] == 8
    assert stats["max"] == 40
    assert stats["p95"] == pytest.approx(38.4)
    with pytest.raises(ValueError, match="positive lengths"):
        stage8._token_length_stats([4, 0, 8])
    with pytest.raises(ValueError, match="non-empty vector"):
        stage8._token_length_stats([])


def test_shared_context_length_selection_uses_both_tokenizers():
    lengths = {
        "indobertweet": np.array([50] * 90 + [80] * 10),
        "xlmr": np.array([64] * 80 + [121] * 20),
    }
    result = eu.choose_shared_max_length(
        lengths,
        percentile=eu.LENGTH_PERCENTILE,
        multiple_of=eu.LENGTH_ROUNDING_MULTIPLE,
        hard_cap=eu.MAX_SEQUENCE_LENGTH,
        partition="contextual_c4",
    )
    assert result["partition"] == "contextual_c4"
    assert result["max_length"] == min(int(np.ceil(max(np.percentile(v, 95) for v in lengths.values()) / 8) * 8), 512)
    assert set(result["truncation_rate_at_max_length"]) == set(eu.BACKBONES)
    with pytest.raises(ValueError, match="every backbone"):
        eu.choose_shared_max_length({"indobertweet": [10, 20]}, partition="contextual_c4")


def test_resolved_stage8_configs_validate_and_have_backbone_parity():
    audit, matched, _, tuning_train, tuning_val, fold_audit = _audit()
    configs = {}
    for backbone in eu.BACKBONES:
        cfg = stage8._build_resolved_config(
            backbone, audit, audit["matched_sha256"], fold_audit, 160,
            tuning_train, tuning_val, 2e-5, mixed_precision=False,
        )
        eu.validate_resolved_run_config(
            cfg, stage=8, backbone=backbone, input_mode="contextual",
            context_size=4, require_shared_config=False,
        )
        configs[backbone] = cfg
    eu.check_config_parity(configs["indobertweet"], configs["xlmr"], input_mode="contextual")
    assert configs["indobertweet"]["truncation_side"] == "left"
    assert configs["xlmr"]["context_order"] == "parent_4,parent_3,parent_2,parent_1,target"


def test_context_learning_rate_document_requires_all_candidates_and_tie_breaks_low():
    audit, matched, _, tuning_train, tuning_val, fold_audit = _audit()
    scores = {}
    for backbone in eu.BACKBONES:
        scores[backbone] = {
            float(lr): {
                "validation_macro_f1": (0.90 if float(lr) in (1e-5, 2e-5) else 0.88),
                "elapsed_seconds": 100.0,
            }
            for lr in eu.LR_SEARCH_GRID
        }
    timing = {b: {"elapsed_seconds": 900.0, "mixed_precision": True} for b in eu.BACKBONES}
    itft = {b: {"manifest_fingerprint": f"{i:064x}"} for i, b in enumerate(eu.BACKBONES, start=1)}
    original_lr = {
        "schema_version": eu.CONFIG_SCHEMA_VERSION,
        "run_id": eu.RUN_ID,
        "lr_search_grid": list(eu.LR_SEARCH_GRID),
        "lr_tuning_seed": 314159,
        "learning_rate_benchmark": {"indobertweet": 2e-5, "xlmr": 3e-5},
        "benchmark_trial_macro_f1": {
            b: {format(lr, ".0e"): 0.8 for lr in eu.LR_SEARCH_GRID}
            for b in eu.BACKBONES
        },
    }
    doc = stage8._build_context_learning_rate_document(
        original_lr, scores, audit=audit, matched_sha=audit["matched_sha256"],
        fold_audit=fold_audit, context_length=160, tuning_train=tuning_train,
        tuning_val=tuning_val, itft_by_backbone=itft, timing_by_backbone=timing,
        latency_document=_latency_document(),
    )
    assert doc["learning_rate_context"] == {b: 1e-5 for b in eu.BACKBONES}
    assert all(set(doc["context_trial_macro_f1"][b]) == {format(lr, ".0e") for lr in eu.LR_SEARCH_GRID} for b in eu.BACKBONES)
    assert doc["context_preflight"]["uses_fold0_held_out"] is False
    assert doc["context_preflight"]["total_trial_count"] == 8
    incomplete = {b: dict(v) for b, v in scores.items()}
    incomplete["xlmr"].pop(float(eu.LR_SEARCH_GRID[-1]))
    with pytest.raises(ValueError, match="every candidate"):
        stage8._build_context_learning_rate_document(
            original_lr, incomplete, audit=audit, matched_sha=audit["matched_sha256"],
            fold_audit=fold_audit, context_length=160, tuning_train=tuning_train,
            tuning_val=tuning_val, itft_by_backbone=itft, timing_by_backbone=timing,
            latency_document=_latency_document(),
        )


def test_shared_configuration_publication_merges_length_and_lr_documents(tmp_path):
    sequence_path = tmp_path / "sequence_length.json"
    lr_path = tmp_path / "learning_rates.json"
    benchmark_selection = eu.choose_shared_max_length(
        {"indobertweet": [80] * 95 + [88] * 5, "xlmr": [80] * 90 + [96] * 10},
        percentile=eu.LENGTH_PERCENTILE, multiple_of=8, hard_cap=512,
        partition="benchmark_train",
    )
    eu.save_length_selection_config(benchmark_selection, sequence_path)
    benchmark_lrs = {"indobertweet": 2e-5, "xlmr": 3e-5}
    benchmark_scores = {b: {format(lr, ".0e"): (0.9 if np.isclose(lr, benchmark_lrs[b]) else 0.8) for lr in eu.LR_SEARCH_GRID} for b in eu.BACKBONES}
    lr_doc = {
        "schema_version": eu.CONFIG_SCHEMA_VERSION,
        "run_id": eu.RUN_ID,
        "lr_search_grid": list(eu.LR_SEARCH_GRID),
        "lr_tuning_seed": 314159,
        "learning_rate_benchmark": benchmark_lrs,
        "benchmark_trial_macro_f1": benchmark_scores,
    }
    lr_path.write_text(json.dumps(lr_doc), encoding="utf-8")
    context_selection = eu.choose_shared_max_length(
        {"indobertweet": [112] * 80 + [144] * 20, "xlmr": [120] * 80 + [152] * 20},
        percentile=eu.LENGTH_PERCENTILE, multiple_of=8, hard_cap=512,
        partition="contextual_c4",
    )
    context_scores = {b: {format(lr, ".0e"): (0.92 if np.isclose(lr, 2e-5) else 0.87) for lr in eu.LR_SEARCH_GRID} for b in eu.BACKBONES}
    lr_doc["learning_rate_context"] = {b: 2e-5 for b in eu.BACKBONES}
    lr_doc["context_trial_macro_f1"] = context_scores
    sequence_doc, lr_output = stage8._publish_shared_configs(
        selection=context_selection, lr_doc=lr_doc, sequence_path=sequence_path,
        lr_path=lr_path, sequence_metadata={"context_length_training_rows": 576},
    )
    assert sequence_doc["context_length_source_partition"] == "contextual_c4"
    assert sequence_doc["context_length_training_rows"] == 576
    assert lr_output["learning_rate_context"] == {b: 2e-5 for b in eu.BACKBONES}
    assert eu.load_shared_run_config(sequence_path, lr_path, require_context=True, require_context_lr=True)


def test_pilot_summary_declares_both_backbones_and_50_runs_each():
    selection = {
        "max_length": 160,
        "percentile_tokens": {"indobertweet": 151.2, "xlmr": 157.3},
        "truncation_rate_at_max_length": {"indobertweet": 0.02, "xlmr": 0.04},
    }
    token_stats = {b: {"n": 576, "min": 32, "mean": 100.0, "median": 98.0, "p90": 140.0, "p95": 150.0, "p99": 175.0, "max": 210} for b in eu.BACKBONES}
    selected_lrs = {b: 2e-5 for b in eu.BACKBONES}
    trial_results = {
        b: {
            float(lr): {
                "model_revision": "a" * 40 if b == "indobertweet" else "b" * 40,
                "tokenizer_revision": "a" * 40 if b == "indobertweet" else "b" * 40,
                "tuning_seed": 314159, "training_rows": 576, "validation_rows": 144,
                "validation_macro_f1": 0.8, "validation_accuracy": 0.8,
                "validation_weighted_f1": 0.8, "validation_f1_n": 0.8,
                "validation_f1_al": 0.8, "validation_f1_hs": 0.8,
                "elapsed_seconds": 100.0, "uses_fold0_held_out": False,
                "manifest_fingerprint": "c" * 64,
            } for lr in eu.LR_SEARCH_GRID
        } for b in eu.BACKBONES
    }
    timing = {b: {"elapsed_seconds": 900.0, "manifest_fingerprint": "d" * 64} for b in eu.BACKBONES}
    itft = {b: {"elapsed_seconds": 1200.0, "manifest_fingerprint": "e" * 64} for b in eu.BACKBONES}
    token_table, lr_table, timing_table, summary = stage8._build_summary_tables(
        token_stats=token_stats, selection=selection, trial_results=trial_results,
        timing_results=timing, itft_by_backbone=itft, selected_lr=selected_lrs,
        latency_document=_latency_document(),
    )
    assert set(token_table["backbone"]) == set(eu.BACKBONES)
    assert len(lr_table) == 8
    assert len(timing_table) == 2
    assert summary["cv_runs_per_backbone"] == 50
    assert summary["fold_0_held_out_used"] is False


def test_latency_sample_selection_is_stable_unique_and_requires_minimum_repeats():
    _, _, _, tuning_train, _, _ = _audit()
    sampled_a = stage8._select_latency_rows(tuning_train, 30)
    sampled_b = stage8._select_latency_rows(tuning_train.sample(frac=1, random_state=99), 30)
    assert len(sampled_a) == 30
    assert sampled_a[eu.CTX_ID_COLUMN].tolist() == sampled_b[eu.CTX_ID_COLUMN].tolist()
    assert sampled_a[eu.CTX_ID_COLUMN].is_unique
    with pytest.raises(ValueError, match="at least 20"):
        stage8._select_latency_rows(tuning_train, 19)


def test_latency_measurement_helper_uses_same_posts_and_emits_c0_to_c4_summaries(tmp_path):
    _, _, _, tuning_train, _, fold_audit = _audit()
    sample = stage8._select_latency_rows(tuning_train, 20)

    class FakeTensor:
        def to(self, device):
            return self

    class FakeTokenizer:
        sep_token = "[SEP]"
        truncation_side = "left"

        def __call__(self, text, **kwargs):
            assert kwargs.get("truncation") is True
            assert kwargs.get("max_length") == 160
            return {"input_ids": FakeTensor(), "attention_mask": FakeTensor()}

    class FakeModel:
        def to(self, device):
            return self

        def eval(self):
            return self

        def __call__(self, **kwargs):
            return None

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return FakeModel()

    fake_torch = SimpleNamespace(
        device=lambda value: value,
        inference_mode=contextlib.nullcontext,
        __version__="test",
    )
    tokenizers = {backbone: FakeTokenizer() for backbone in eu.BACKBONES}
    itft = {
        backbone: {
            "model_dir": tmp_path / backbone,
            "run_manifest": {"config": {"model_revision": "a" * 40, "tokenizer_revision": "a" * 40}},
            "manifest_fingerprint": f"{index:064x}",
        }
        for index, backbone in enumerate(eu.BACKBONES, start=1)
    }
    matched_path = tmp_path / "ctx_matched_subset.csv"
    _matched_df().to_csv(matched_path, index=False)
    result = stage8._measure_inference_latency_by_context(
        torch=fake_torch,
        AutoModel=FakeAutoModel,
        tokenizers=tokenizers,
        sample_rows=sample,
        context_length=160,
        itft_by_backbone=itft,
        use_cuda=False,
        warmup_runs=2,
        measurement_machine="Synthetic CPU",
        matched_subset_path=matched_path,
    )
    assert result["context_sizes"] == [0, 1, 2, 3, 4]
    assert result["sample_post_ids"] == sample[eu.CTX_ID_COLUMN].astype(str).tolist()
    assert result["measurement_repeats_per_backbone_context"] == 20
    assert set(result["median_ms_by_context"]) == {f"C{c}" for c in range(5)}
    assert result["matched_subset_sha256"] == eu.sha256_file(matched_path)
    assert result["fold_assignment_sha256"] == eu.validate_context_dataset(_matched_df(), expected_rows=900)["fold_assignment_sha256"]
    for context in range(5):
        assert set(result["measurements_ms_by_context"][f"C{context}"]) == set(eu.BACKBONES)
        assert all(len(result["measurements_ms_by_context"][f"C{context}"][b]) == 20 for b in eu.BACKBONES)


def test_latency_document_shape_contains_both_backbones_for_every_context():
    doc = _latency_document(30)
    assert set(doc["median_ms_by_context"]) == {f"C{c}" for c in range(5)}
    for context in range(5):
        assert set(doc["median_ms_by_context"][f"C{context}"]) == set(eu.BACKBONES)
        assert all(value > 0 for value in doc["median_ms_by_context"][f"C{context}"].values())
    assert doc["measurement_repeats_per_backbone_context"] >= stage8.LATENCY_MIN_REPEATS


def test_summary_includes_latency_medians_and_p95_by_context():
    selection = {
        "max_length": 160,
        "percentile_tokens": {"indobertweet": 151.2, "xlmr": 157.3},
        "truncation_rate_at_max_length": {"indobertweet": 0.02, "xlmr": 0.04},
    }
    token_stats = {b: {"n": 576, "min": 32, "mean": 100.0, "median": 98.0, "p90": 140.0, "p95": 150.0, "p99": 175.0, "max": 210} for b in eu.BACKBONES}
    selected_lrs = {b: 2e-5 for b in eu.BACKBONES}
    trial_results = {
        b: {
            float(lr): {
                "model_revision": "a" * 40 if b == "indobertweet" else "b" * 40,
                "tokenizer_revision": "a" * 40 if b == "indobertweet" else "b" * 40,
                "tuning_seed": 314159, "training_rows": 576, "validation_rows": 144,
                "validation_macro_f1": 0.8, "validation_accuracy": 0.8,
                "validation_weighted_f1": 0.8, "validation_f1_n": 0.8,
                "validation_f1_al": 0.8, "validation_f1_hs": 0.8,
                "elapsed_seconds": 100.0, "uses_fold0_held_out": False,
                "manifest_fingerprint": "c" * 64,
            } for lr in eu.LR_SEARCH_GRID
        } for b in eu.BACKBONES
    }
    timing = {b: {"elapsed_seconds": 900.0, "manifest_fingerprint": "d" * 64} for b in eu.BACKBONES}
    itft = {b: {"elapsed_seconds": 1200.0, "manifest_fingerprint": "e" * 64} for b in eu.BACKBONES}
    latency = _latency_document()
    _, lr_table, timing_table, summary = stage8._build_summary_tables(
        token_stats=token_stats, selection=selection, trial_results=trial_results,
        timing_results=timing, itft_by_backbone=itft, selected_lr=selected_lrs,
        latency_document=latency,
    )
    assert len(lr_table) == 8
    assert len(timing_table) == 2
    assert "median_inference_latency_ms_C4" in timing_table.columns
    assert "p95_inference_latency_ms_C0" in timing_table.columns
    assert summary["latency_measurement_machine"] == latency["measurement_machine"]
    assert summary["latency_median_ms_by_context"] == latency["median_ms_by_context"]


def test_stage8_parser_uses_authoritative_default_paths():
    args = stage8.parse_args(["--validate-only"])
    assert args.matched_csv == stage8.MATCHED_SUBSET_PATH
    assert args.sequence_config == stage8.SEQUENCE_CONFIG_PATH
    assert args.learning_rate_config == stage8.LEARNING_RATE_CONFIG_PATH
    assert args.validate_only is True
