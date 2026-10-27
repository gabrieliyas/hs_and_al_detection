from __future__ import annotations

import importlib.util
import json
import sys
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiment_utils as eu

SCRIPT_PATH = ROOT / "scripts" / "preflight_tune_benchmark_lr.py"
spec = importlib.util.spec_from_file_location("preflight_tune_benchmark_lr", SCRIPT_PATH)
preflight = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = preflight
spec.loader.exec_module(preflight)


def _write_inputs(tmp_path: Path):
    train_labels = ["N", "AL", "HS"] * 10
    val_labels = ["N", "AL", "HS"] * 2
    train = pd.DataFrame({
        "clean_text": [f"train unique tweet example {i}" for i in range(len(train_labels))],
        "label": train_labels,
    })
    val = pd.DataFrame({
        "clean_text": [f"validation unique tweet example {i}" for i in range(len(val_labels))],
        "label": val_labels,
    })
    train_path = tmp_path / "ib_train.csv"
    val_path = tmp_path / "ib_val.csv"
    weights_path = tmp_path / "ib_class_weights.csv"
    seq_path = tmp_path / "sequence_length.json"
    train.to_csv(train_path, index=False)
    val.to_csv(val_path, index=False)

    weights = eu.compute_class_weights(train["label"])
    id2label = eu.ID2LABEL
    pd.DataFrame([
        {"label": id2label[class_id], "weight": float(weight)}
        for class_id, weight in weights.items()
    ]).to_csv(weights_path, index=False)

    seq_doc = {
        "schema_version": eu.CONFIG_SCHEMA_VERSION,
        "run_id": eu.RUN_ID,
        "truncation_side": "left",
        "target_only_truncation_side": "right",
        "benchmark_length_source_partition": "benchmark_train",
        "max_length_benchmark": 112,
        "benchmark_length_percentile": 95.0,
        "benchmark_length_percentile_tokens": {"indobertweet": 101.0, "xlmr": 111.0},
        "benchmark_truncation_rate_at_max_length": {"indobertweet": 0.01, "xlmr": 0.02},
        "benchmark_hard_cap_hit": False,
        "benchmark_training_split_sha256": eu.sha256_file(train_path),
        "benchmark_training_rows": len(train),
        "benchmark_training_text_column": "clean_text",
        "tokenizer_model_ids": {b: eu.BACKBONES[b].hf_model_id for b in eu.BACKBONES},
        "tokenizer_resolved_revisions": {"indobertweet": "a" * 40, "xlmr": "b" * 40},
        "tokenizer_classes": {"indobertweet": "FakeTokenizer", "xlmr": "FakeTokenizer"},
        "token_count_includes_special_tokens": True,
    }
    seq_path.write_text(json.dumps(seq_doc), encoding="utf-8")
    return train_path, val_path, weights_path, seq_path, seq_doc


def test_load_and_validate_inputs_checks_train_only_weights_and_length_provenance(tmp_path):
    train, val, weights, seq, doc = _write_inputs(tmp_path)
    audit = preflight._load_and_validate_inputs(train, val, weights, seq)
    assert audit["train_rows"] == 30
    assert audit["validation_rows"] == 6
    assert audit["sequence_length_doc"]["max_length_benchmark"] == 112
    assert audit["class_weights_by_id"] == {0: 1.0, 1: 1.0, 2: 1.0}
    assert audit["sequence_config_sha256"] == eu.sha256_file(seq)


def test_load_and_validate_inputs_rejects_training_hash_mismatch(tmp_path):
    train, val, weights, seq, doc = _write_inputs(tmp_path)
    doc["benchmark_training_split_sha256"] = "0" * 64
    seq.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match the split used by Stage 1b"):
        preflight._load_and_validate_inputs(train, val, weights, seq)


def test_learning_rate_document_uses_shared_selector_and_loader_contract(tmp_path):
    train, val, weights, seq, _ = _write_inputs(tmp_path)
    audit = preflight._load_and_validate_inputs(train, val, weights, seq)
    scores = {
        "indobertweet": {
            1e-5: {"macro_f1": 0.70},
            2e-5: {"macro_f1": 0.80},
            3e-5: {"macro_f1": 0.80},
            5e-5: {"macro_f1": 0.75},
        },
        "xlmr": {
            1e-5: {"macro_f1": 0.60},
            2e-5: {"macro_f1": 0.70},
            3e-5: {"macro_f1": 0.76},
            5e-5: {"macro_f1": 0.73},
        },
    }
    doc = preflight._build_learning_rate_document(
        scores,
        tuning_seed=preflight.DEFAULT_TUNING_SEED,
        input_audit=audit,
        hyperparameters={"max_epochs": 4, "weight_decay": 0.01},
        resolved_model_revisions={"indobertweet": "a" * 40, "xlmr": "b" * 40},
    )
    assert doc["learning_rate_benchmark"] == {"indobertweet": 2e-5, "xlmr": 3e-5}
    assert doc["benchmark_preflight"]["uses_test_split"] is False

    rates_path = tmp_path / "learning_rates.json"
    rates_path.write_text(json.dumps(doc), encoding="utf-8")
    loaded = eu.load_shared_run_config(sequence_length_path=seq, learning_rate_path=rates_path)
    assert loaded["learning_rates"]["learning_rate_benchmark"] == doc["learning_rate_benchmark"]


def test_learning_rate_document_rejects_incomplete_grid(tmp_path):
    train, val, weights, seq, _ = _write_inputs(tmp_path)
    audit = preflight._load_and_validate_inputs(train, val, weights, seq)
    scores = {
        "indobertweet": {1e-5: {"macro_f1": 0.5}},
        "xlmr": {lr: {"macro_f1": 0.5} for lr in eu.LR_SEARCH_GRID},
    }
    with pytest.raises(ValueError, match="Equal-budget rule violated"):
        preflight._build_learning_rate_document(
            scores, tuning_seed=preflight.DEFAULT_TUNING_SEED, input_audit=audit,
            hyperparameters={}, resolved_model_revisions={"indobertweet": "a" * 40, "xlmr": "b" * 40},
        )


def test_tuning_seed_cannot_overlap_evaluation_seeds():
    with pytest.raises(ValueError, match="overlaps a frozen evaluation/CV seed"):
        preflight._validate_tuning_seed(42)
    preflight._validate_tuning_seed(preflight.DEFAULT_TUNING_SEED)


def test_run_trials_exercises_exact_eight_trial_matrix_without_transformers(tmp_path, monkeypatch):
    train, val, weights, seq, _ = _write_inputs(tmp_path)
    audit = preflight._load_and_validate_inputs(train, val, weights, seq)

    class FakeTokenizer:
        def __init__(self, revision):
            self._commit_hash = revision
            self.init_kwargs = {"_commit_hash": revision}
            self.truncation_side = "right"

        def __call__(self, texts, *, max_length, return_tensors, **kwargs):
            return {
                "input_ids": torch.ones((len(texts), max_length), dtype=torch.long),
                "attention_mask": torch.ones((len(texts), max_length), dtype=torch.long),
            }

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(model_id, revision, **kwargs):
            return FakeTokenizer(revision)

    class FakeModel:
        def __init__(self, model_id, revision):
            self.hf_model_id = model_id
            self.config = SimpleNamespace(_commit_hash=revision)

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(model_id, revision, **kwargs):
            return FakeModel(model_id, revision)

    class FakeTrainingArguments:
        def __init__(self, output_dir=None, num_train_epochs=4, per_device_train_batch_size=16,
                     per_device_eval_batch_size=32, gradient_accumulation_steps=1,
                     learning_rate=1e-5, weight_decay=0.01, warmup_ratio=0.0,
                     lr_scheduler_type="linear", eval_strategy="epoch", evaluation_strategy=None,
                     save_strategy="epoch", save_total_limit=1, load_best_model_at_end=True,
                     metric_for_best_model="macro_f1", greater_is_better=True,
                     logging_strategy="epoch", seed=1, data_seed=1, report_to="none",
                     fp16=False, disable_tqdm=False):
            self.output_dir = output_dir
            self.learning_rate = learning_rate

    class FakeTrainer:
        def __init__(self, model=None, args=None, **kwargs):
            self.model = model
            self.args = args
            self.state = SimpleNamespace(best_global_step=2, global_step=10, best_metric=None, best_model_checkpoint="checkpoint-2")

        def train(self):
            return SimpleNamespace()

        def evaluate(self, dataset=None):
            lr_scores = {1e-5: 0.60, 2e-5: 0.70, 3e-5: 0.80, 5e-5: 0.75}
            score = lr_scores[float(self.args.learning_rate)]
            self.state.best_metric = score
            return {
                "eval_macro_f1": score,
                "eval_accuracy": score,
                "eval_weighted_f1": score,
                "eval_f1_n": score,
                "eval_f1_al": score,
                "eval_f1_hs": score,
            }

    def fake_set_seed(seed):
        return None

    monkeypatch.setattr(
        preflight,
        "_training_dependencies",
        lambda: (torch, FakeAutoModel, FakeAutoTokenizer, FakeTrainer, FakeTrainingArguments, fake_set_seed),
    )

    scores, rows, revisions, use_cuda = preflight._run_trials(
        audit,
        tuning_seed=preflight.DEFAULT_TUNING_SEED,
        epochs=4,
        train_batch_size=16,
        eval_batch_size=32,
        gradient_accumulation_steps=1,
        allow_cpu=True,
        disable_tqdm=True,
        project_root=tmp_path,
    )
    assert len(rows) == 8
    assert len({(row["backbone"], row["learning_rate"]) for row in rows}) == 8
    assert set(scores) == set(eu.BACKBONES)
    assert all(set(scores[b]) == set(eu.LR_SEARCH_GRID) for b in eu.BACKBONES)
    assert revisions == {"indobertweet": "a" * 40, "xlmr": "b" * 40}
    assert use_cuda is False


def test_main_publishes_verified_manifest_and_shared_config_with_fake_training(tmp_path, monkeypatch):
    train, val, weights, seq, _ = _write_inputs(tmp_path)
    project_root = tmp_path / "project"
    (project_root / "scripts").mkdir(parents=True)
    (project_root / "data/processed").mkdir(parents=True)
    (project_root / "results/tables").mkdir(parents=True)
    (project_root / "results/config").mkdir(parents=True)
    (project_root / "experiment_protocol.md").write_text("test protocol\n", encoding="utf-8")
    (project_root / "pipeline.md").write_text("test pipeline\n", encoding="utf-8")
    shutil_path = project_root / eu.PIPELINE_SCRIPTS["preflight"]
    shutil_path.parent.mkdir(parents=True, exist_ok=True)
    shutil_path.write_bytes(SCRIPT_PATH.read_bytes())

    project_train = project_root / "data/processed/ib_train.csv"
    project_val = project_root / "data/processed/ib_val.csv"
    project_weights = project_root / "results/tables/ib_class_weights.csv"
    project_seq = project_root / "results/config/sequence_length.json"
    project_train.write_bytes(train.read_bytes())
    project_val.write_bytes(val.read_bytes())
    project_weights.write_bytes(weights.read_bytes())
    project_seq.write_bytes(seq.read_bytes())

    project_run_dir = eu.versioned_run_root(project_root) / "stage_preflight"
    monkeypatch.setattr(preflight, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(preflight, "STAGE_RUN_DIR", project_run_dir)
    monkeypatch.setattr(preflight, "TRIALS_CSV_PATH", project_run_dir / "benchmark_lr_trials.csv")
    monkeypatch.setattr(preflight, "BENCHMARK_SNAPSHOT_PATH", project_run_dir / "benchmark_learning_rates.json")
    monkeypatch.setattr(preflight, "LEARNING_RATE_CONFIG_PATH", Path("results/config/learning_rates.json"))

    scores = {
        "indobertweet": {
            1e-5: {"macro_f1": 0.60, "accuracy": 0.60, "weighted_f1": 0.60, "f1_n": 0.60, "f1_al": 0.60, "f1_hs": 0.60},
            2e-5: {"macro_f1": 0.72, "accuracy": 0.72, "weighted_f1": 0.72, "f1_n": 0.72, "f1_al": 0.72, "f1_hs": 0.72},
            3e-5: {"macro_f1": 0.70, "accuracy": 0.70, "weighted_f1": 0.70, "f1_n": 0.70, "f1_al": 0.70, "f1_hs": 0.70},
            5e-5: {"macro_f1": 0.65, "accuracy": 0.65, "weighted_f1": 0.65, "f1_n": 0.65, "f1_al": 0.65, "f1_hs": 0.65},
        },
        "xlmr": {
            1e-5: {"macro_f1": 0.61, "accuracy": 0.61, "weighted_f1": 0.61, "f1_n": 0.61, "f1_al": 0.61, "f1_hs": 0.61},
            2e-5: {"macro_f1": 0.71, "accuracy": 0.71, "weighted_f1": 0.71, "f1_n": 0.71, "f1_al": 0.71, "f1_hs": 0.71},
            3e-5: {"macro_f1": 0.77, "accuracy": 0.77, "weighted_f1": 0.77, "f1_n": 0.77, "f1_al": 0.77, "f1_hs": 0.77},
            5e-5: {"macro_f1": 0.68, "accuracy": 0.68, "weighted_f1": 0.68, "f1_n": 0.68, "f1_al": 0.68, "f1_hs": 0.68},
        },
    }
    rows = [
        {"backbone": b, "learning_rate": float(lr), "validation_macro_f1": float(scores[b][lr]["macro_f1"])}
        for b in eu.BACKBONES for lr in eu.LR_SEARCH_GRID
    ]
    monkeypatch.setattr(preflight, "_run_trials", lambda *args, **kwargs: (scores, rows, {"indobertweet": "a" * 40, "xlmr": "b" * 40}, False))
    monkeypatch.setattr(sys, "argv", [
        "preflight_tune_benchmark_lr.py",
        "--train-csv", str(project_train),
        "--val-csv", str(project_val),
        "--weights-csv", str(project_weights),
        "--sequence-config", str(project_seq),
        "--allow-cpu",
    ])
    monkeypatch.setattr(preflight, "_resolve_project_path", lambda path: Path(path).resolve() if Path(path).is_absolute() else (project_root / path).resolve())

    preflight.main()

    shared_config = project_root / "results/config/learning_rates.json"
    assert shared_config.is_file()
    assert project_run_dir.joinpath("benchmark_lr_trials.csv").is_file()
    assert project_run_dir.joinpath("benchmark_learning_rates.json").is_file()
    assert eu.assert_current_stage_manifest("preflight", project_root)["metadata"]["trial_count"] == 8
    loaded = eu.load_shared_run_config(sequence_length_path=project_seq, learning_rate_path=shared_config)
    assert loaded["learning_rates"]["learning_rate_benchmark"] == {"indobertweet": 2e-5, "xlmr": 3e-5}
