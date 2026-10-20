"""Benchmark learning-rate preflight for the Transformer thesis pipeline.

Run after Stage 1, Stage 1b, and Stage 2, before Experiment I or ITFT:
    python scripts/preflight_tune_benchmark_lr.py --validate-only
    python scripts/preflight_tune_benchmark_lr.py

The real search performs exactly eight runs: the four rates in
experiment_utils.LR_SEARCH_GRID for each registered backbone. Every trial:
- trains on data/processed/ib_train.csv only;
- selects its best epoch using data/processed/ib_val.csv macro-F1;
- applies balanced class weights calculated from the training split only;
- uses the same tuning seed and fixed non-LR hyperparameters;
- uses the benchmark max length chosen in Stage 1b and explicit right truncation;
- loads model weights and tokenizer from the pinned revisions recorded by Stage 1b.

The benchmark test split is deliberately never opened by this script.
Old shared configs and prior preflight runs are never overwritten or resumed.
The shared learning_rates.json is expected to be extended by Stage 8; an
immutable benchmark-only snapshot and the trial table are stored under the
versioned Stage-preflight directory and hashed by its manifest.
"""
from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import re
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


def _project_root() -> Path:
    here = Path(__file__).resolve().parent
    return here.parent if here.name.casefold() == "scripts" else here


PROJECT_ROOT = _project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import experiment_utils as eu  # noqa: E402

TRAIN_PATH = Path("data/processed/ib_train.csv")
VAL_PATH = Path("data/processed/ib_val.csv")
WEIGHTS_PATH = Path("results/tables/ib_class_weights.csv")
SEQUENCE_CONFIG_PATH = eu.SEQUENCE_LENGTH_CONFIG
LEARNING_RATE_CONFIG_PATH = eu.LEARNING_RATE_CONFIG
STAGE_RUN_DIR = eu.versioned_run_root(PROJECT_ROOT) / "stage_preflight"
TRIALS_CSV_PATH = STAGE_RUN_DIR / "benchmark_lr_trials.csv"
BENCHMARK_SNAPSHOT_PATH = STAGE_RUN_DIR / "benchmark_learning_rates.json"

# Frozen starting point from the existing Experiment I script. Change only if
# the protocol is formally revised; these values are written to each trial row.
DEFAULT_TUNING_SEED = 314159
NUM_EPOCHS = 4
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
GRADIENT_ACCUMULATION_STEPS = 1
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.0
LR_SCHEDULER = "linear"
OPTIMIZER = "AdamW"
TRUNCATION_SIDE = "right"
SPLIT_SEED = 42
CHECKPOINT_METRIC = "macro_f1"


def _resolve_project_path(path: str | Path, project_root: Path = PROJECT_ROOT) -> Path:
    p = Path(path)
    return p.resolve() if p.is_absolute() else (project_root / p).resolve()


def _safe_sha256(value: Any) -> bool:
    return isinstance(value, str) and bool(re.fullmatch(r"[0-9a-fA-F]{64}", value))


def _lr_key(value: float) -> str:
    """Stable JSON key accepted by load_shared_run_config (which casts to float)."""
    return format(float(value), ".0e")


def _validate_labels(df: pd.DataFrame, path: Path, split_name: str) -> list[str]:
    required = {"clean_text", "label"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"{path} is missing required columns {missing}; expected clean_text and label")
    if df.empty:
        raise ValueError(f"{split_name} split is empty: {path}")
    if df["label"].isna().any():
        raise ValueError(f"{split_name} split has missing labels")
    labels = set(df["label"].astype(str).unique())
    expected = set(eu.LABEL2ID)
    if labels != expected:
        raise ValueError(f"{split_name} labels must be exactly {sorted(expected)}; got {sorted(labels)}")
    texts = df["clean_text"].astype("string").fillna("").str.strip().tolist()
    blank_rows = [i for i, text in enumerate(texts) if not text]
    if blank_rows:
        raise ValueError(f"{split_name} contains {len(blank_rows)} blank clean_text value(s); first row positions: {blank_rows[:10]}")
    duplicate_count = int(pd.Series(texts).duplicated().sum())
    if duplicate_count:
        raise ValueError(f"{split_name} contains {duplicate_count} duplicate clean_text value(s); Stage 1 should remove exact duplicate tweets")
    return texts


def _load_and_validate_length_config(
    sequence_config_path: Path,
    train_csv_path: Path,
) -> dict[str, Any]:
    required = [
        "schema_version", "run_id", "benchmark_length_source_partition",
        "max_length_benchmark", "benchmark_length_percentile",
        "benchmark_length_percentile_tokens", "benchmark_truncation_rate_at_max_length",
        "truncation_side", "target_only_truncation_side",
        "benchmark_training_split_sha256", "tokenizer_model_ids",
        "tokenizer_resolved_revisions",
    ]
    doc = eu.load_json_config(sequence_config_path, required_keys=required)
    if doc["schema_version"] != eu.CONFIG_SCHEMA_VERSION or doc["run_id"] != eu.RUN_ID:
        raise ValueError("sequence_length.json has an incompatible schema/run_id; regenerate Stage 1b")
    if doc["benchmark_length_source_partition"] != "benchmark_train":
        raise ValueError("Benchmark maximum length must be selected from benchmark_train only")
    if float(doc["benchmark_length_percentile"]) != eu.LENGTH_PERCENTILE:
        raise ValueError(f"Benchmark length must use the protocol's p{eu.LENGTH_PERCENTILE:g}")
    if doc["truncation_side"] != "left" or doc["target_only_truncation_side"] != TRUNCATION_SIDE:
        raise ValueError("Expected contextual truncation_side='left' and target_only_truncation_side='right'")

    max_length = doc["max_length_benchmark"]
    if isinstance(max_length, bool) or not isinstance(max_length, int):
        raise ValueError("max_length_benchmark must be an integer")
    if not (8 <= max_length <= eu.MAX_SEQUENCE_LENGTH) or max_length % eu.LENGTH_ROUNDING_MULTIPLE:
        raise ValueError("max_length_benchmark must be 8..512 and divisible by 8")

    pct = doc["benchmark_length_percentile_tokens"]
    rates = doc["benchmark_truncation_rate_at_max_length"]
    if not isinstance(pct, Mapping) or set(pct) != set(eu.BACKBONES):
        raise ValueError("benchmark_length_percentile_tokens must contain both registered backbones")
    if not isinstance(rates, Mapping) or set(rates) != set(eu.BACKBONES):
        raise ValueError("benchmark_truncation_rate_at_max_length must contain both registered backbones")
    pct_values = {b: float(pct[b]) for b in eu.BACKBONES}
    rate_values = {b: float(rates[b]) for b in eu.BACKBONES}
    if any(not math.isfinite(v) or v <= 0 for v in pct_values.values()):
        raise ValueError("Benchmark p95 token lengths must be positive finite numbers")
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in rate_values.values()):
        raise ValueError("Benchmark truncation rates must be fractions in 0..1")
    expected_length = min(
        int(math.ceil(max(pct_values.values()) / eu.LENGTH_ROUNDING_MULTIPLE) * eu.LENGTH_ROUNDING_MULTIPLE),
        eu.MAX_SEQUENCE_LENGTH,
    )
    if max_length != expected_length:
        raise ValueError(f"max_length_benchmark={max_length} disagrees with p95 stats; expected {expected_length}")

    train_sha = eu.sha256_file(train_csv_path)
    if doc["benchmark_training_split_sha256"] != train_sha:
        raise ValueError(
            "ib_train.csv does not match the split used by Stage 1b. Do not tune with mismatched data; "
            "rerun Stage 1b after verifying Stage 1 outputs."
        )
    if doc.get("benchmark_training_text_column") != "clean_text":
        raise ValueError("Stage 1b must record benchmark_training_text_column='clean_text'")

    model_ids = doc["tokenizer_model_ids"]
    revisions = doc["tokenizer_resolved_revisions"]
    if not isinstance(model_ids, Mapping) or set(model_ids) != set(eu.BACKBONES):
        raise ValueError("tokenizer_model_ids must contain both backbones")
    if not isinstance(revisions, Mapping) or set(revisions) != set(eu.BACKBONES):
        raise ValueError("tokenizer_resolved_revisions must contain both pinned tokenizer revisions")
    for backbone, spec in eu.BACKBONES.items():
        if model_ids[backbone] != spec.hf_model_id:
            raise ValueError(f"Stage 1b tokenizer model for {backbone} is {model_ids[backbone]!r}; expected {spec.hf_model_id!r}")
        if not isinstance(revisions[backbone], str) or not re.fullmatch(r"[0-9a-fA-F]{40}", revisions[backbone]):
            raise ValueError(f"Stage 1b must record a full 40-character commit SHA for {backbone}")

    return doc


def _load_and_validate_inputs(
    train_path: Path,
    val_path: Path,
    weights_path: Path,
    sequence_config_path: Path,
) -> dict[str, Any]:
    for path in (train_path, val_path, weights_path, sequence_config_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required preflight input not found: {path}")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    train_texts = _validate_labels(train_df, train_path, "training")
    val_texts = _validate_labels(val_df, val_path, "validation")
    overlap = set(train_texts) & set(val_texts)
    if overlap:
        raise ValueError(f"Training and validation splits overlap by {len(overlap)} exact clean_text value(s)")

    train_weights = eu.compute_class_weights(train_df["label"].astype(str))
    weights_table = pd.read_csv(weights_path)
    if not {"label", "weight"}.issubset(weights_table.columns):
        raise KeyError(f"{weights_path} must contain label and weight columns")
    if weights_table["label"].astype(str).duplicated().any():
        raise ValueError("Class-weight table contains duplicate labels")
    observed_labels = set(weights_table["label"].astype(str))
    if observed_labels != set(eu.LABEL2ID):
        raise ValueError(f"Class-weight table labels must be {sorted(eu.LABEL2ID)}; got {sorted(observed_labels)}")
    observed_weights = dict(zip(weights_table["label"].astype(str), pd.to_numeric(weights_table["weight"], errors="coerce")))
    for label, idx in eu.LABEL2ID.items():
        actual = observed_weights.get(label)
        expected = float(train_weights[idx])
        if actual is None or not math.isfinite(float(actual)) or not np.isclose(float(actual), expected, rtol=1e-5, atol=1e-6):
            raise ValueError(
                f"Saved class weight for {label} does not match training-only recomputation: "
                f"saved={actual!r}, expected={expected:.10f}"
            )

    length_doc = _load_and_validate_length_config(sequence_config_path, train_path)
    train_sha = eu.sha256_file(train_path)
    val_sha = eu.sha256_file(val_path)
    weights_sha = eu.sha256_file(weights_path)
    split_assignment_sha = eu.sha256_json({"train_sha256": train_sha, "validation_sha256": val_sha, "split_seed": SPLIT_SEED})
    dataset_sha = eu.sha256_json({"train_sha256": train_sha, "validation_sha256": val_sha})

    return {
        "train_df": train_df,
        "val_df": val_df,
        "train_texts": train_texts,
        "val_texts": val_texts,
        "train_labels": train_df["label"].astype(str).tolist(),
        "val_labels": val_df["label"].astype(str).tolist(),
        "train_sha256": train_sha,
        "validation_sha256": val_sha,
        "class_weights_sha256": weights_sha,
        "dataset_sha256": dataset_sha,
        "split_assignment_sha256": split_assignment_sha,
        "sequence_length_doc": length_doc,
        "sequence_config_sha256": eu.sha256_file(sequence_config_path),
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(val_df)),
        "class_weights_by_id": train_weights,
    }


def _validate_tuning_seed(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("lr_tuning_seed must be an integer")
    if seed in eu.EVAL_SEEDS_EXP1 or seed in eu.CV_SEEDS:
        raise ValueError(
            f"Tuning seed {seed} overlaps a frozen evaluation/CV seed. Choose a dedicated seed not in "
            f"{sorted(set(eu.EVAL_SEEDS_EXP1) | set(eu.CV_SEEDS))}."
        )


def _metric_summary(y_true: np.ndarray, logits_or_predictions: np.ndarray) -> dict[str, float]:
    values = np.asarray(logits_or_predictions)
    predictions = values.argmax(axis=1) if values.ndim == 2 else values.astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "macro_f1": float(f1_score(y_true, predictions, labels=[0, 1, 2], average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, predictions, labels=[0, 1, 2], average="weighted", zero_division=0)),
        "f1_n": float(f1_score(y_true, predictions, labels=[0], average="macro", zero_division=0)),
        "f1_al": float(f1_score(y_true, predictions, labels=[1], average="macro", zero_division=0)),
        "f1_hs": float(f1_score(y_true, predictions, labels=[2], average="macro", zero_division=0)),
    }


def _build_learning_rate_document(
    scores_by_backbone: Mapping[str, Mapping[float, Mapping[str, float]]],
    *,
    tuning_seed: int,
    input_audit: Mapping[str, Any],
    hyperparameters: Mapping[str, Any],
    resolved_model_revisions: Mapping[str, str],
) -> dict[str, Any]:
    """Create the exact shared-config fields consumed by load_shared_run_config."""
    _validate_tuning_seed(tuning_seed)
    if set(scores_by_backbone) != set(eu.BACKBONES):
        raise ValueError(f"Trial results must contain exactly {sorted(eu.BACKBONES)}")
    selected: dict[str, float] = {}
    serialized_scores: dict[str, dict[str, float]] = {}
    for backbone in eu.BACKBONES:
        trial_scores = {float(lr): float(metrics["macro_f1"]) for lr, metrics in scores_by_backbone[backbone].items()}
        if any(not math.isfinite(score) or not 0.0 <= score <= 1.0 for score in trial_scores.values()):
            raise ValueError(f"Validation macro-F1 scores for {backbone} must be finite values in 0..1")
        selected[backbone] = eu.select_learning_rate(trial_scores, eu.LR_SEARCH_GRID)
        serialized_scores[backbone] = {_lr_key(lr): trial_scores[lr] for lr in sorted(trial_scores)}

    length_doc = input_audit["sequence_length_doc"]
    doc: dict[str, Any] = {
        "schema_version": eu.CONFIG_SCHEMA_VERSION,
        "run_id": eu.RUN_ID,
        "lr_search_grid": [float(x) for x in eu.LR_SEARCH_GRID],
        "lr_tuning_seed": int(tuning_seed),
        "learning_rate_benchmark": selected,
        "benchmark_trial_macro_f1": serialized_scores,
        "benchmark_preflight": {
            "selection_rule": "highest validation macro-F1; exact ties go to the smaller learning rate",
            "trial_count_per_backbone": len(eu.LR_SEARCH_GRID),
            "total_trial_count": len(eu.LR_SEARCH_GRID) * len(eu.BACKBONES),
            "validation_metric": "macro_f1",
            "uses_test_split": False,
            "train_csv_sha256": input_audit["train_sha256"],
            "validation_csv_sha256": input_audit["validation_sha256"],
            "class_weights_csv_sha256": input_audit["class_weights_sha256"],
            "sequence_length_config_sha256": input_audit["sequence_config_sha256"],
            "benchmark_dataset_sha256": input_audit["dataset_sha256"],
            "split_assignment_sha256": input_audit["split_assignment_sha256"],
            "max_length_benchmark": int(length_doc["max_length_benchmark"]),
            "truncation_side": TRUNCATION_SIDE,
            "model_ids": {b: eu.BACKBONES[b].hf_model_id for b in eu.BACKBONES},
            "model_revisions": dict(resolved_model_revisions),
            "hyperparameters": dict(hyperparameters),
        },
    }
    return doc


def _build_parity_config(backbone: str, learning_rate: float, audit: Mapping[str, Any], *, tuning_seed: int, fp16: bool, epochs: int, train_batch_size: int, eval_batch_size: int, gradient_accumulation_steps: int) -> dict[str, Any]:
    return {
        "schema_version": eu.CONFIG_SCHEMA_VERSION,
        "run_id": eu.RUN_ID,
        "stage": "preflight",
        "script_name": eu.PIPELINE_SCRIPTS["preflight"],
        "input_mode": "target_only",
        "dataset_sha256": audit["dataset_sha256"],
        "split_seed": SPLIT_SEED,
        "split_assignment_sha256": audit["split_assignment_sha256"],
        "eval_seeds_exp1": list(eu.EVAL_SEEDS_EXP1),
        "context_order": "target_only",
        "max_length_benchmark": int(audit["sequence_length_doc"]["max_length_benchmark"]),
        "truncation_side": TRUNCATION_SIDE,
        "optimizer": OPTIMIZER,
        "weight_decay": WEIGHT_DECAY,
        "warmup_ratio": WARMUP_RATIO,
        "lr_scheduler": LR_SCHEDULER,
        "max_epochs": int(epochs),
        "early_stopping_patience": None,
        "loss": "weighted_cross_entropy",
        "class_weighting_benchmark": "balanced_weights_from_training_split_only",
        "checkpoint_metric": CHECKPOINT_METRIC,
        "lr_search_grid": list(eu.LR_SEARCH_GRID),
        "lr_tuning_seed": int(tuning_seed),
        "mixed_precision": bool(fp16),
        "input_order": "target_only",
        "hf_model_id": eu.BACKBONES[backbone].hf_model_id,
        "learning_rate_benchmark": float(learning_rate),
        "per_device_batch_size": int(train_batch_size),
        "gradient_accumulation_steps": int(gradient_accumulation_steps),
        "eval_batch_size": int(eval_batch_size),
    }


def _training_dependencies():
    try:
        import torch
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
            set_seed,
        )
    except ImportError as exc:
        raise ImportError(
            "Benchmark LR tuning requires PyTorch and Hugging Face Transformers. "
            "Install the repository requirements and run on a Colab GPU runtime."
        ) from exc
    return torch, AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, set_seed


def _make_weighted_trainer_class(torch, Trainer):
    class WeightedLossTrainer(Trainer):
        def __init__(self, *args, class_weights, **kwargs):
            super().__init__(*args, **kwargs)
            self.class_weights = class_weights

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss_fn = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(outputs.logits.device))
            loss = loss_fn(outputs.logits, labels)
            return (loss, outputs) if return_outputs else loss

    return WeightedLossTrainer


def _make_torch_dataset(torch):
    class TokenizedTextDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = np.asarray(labels, dtype=np.int64)

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, index):
            item = {key: value[index] for key, value in self.encodings.items()}
            item["labels"] = torch.tensor(int(self.labels[index]), dtype=torch.long)
            return item

    return TokenizedTextDataset


def _make_training_args(TrainingArguments, *, output_dir: Path, learning_rate: float, tuning_seed: int, epochs: int, train_batch_size: int, eval_batch_size: int, gradient_accumulation_steps: int, use_fp16: bool, disable_tqdm: bool):
    signature = inspect.signature(TrainingArguments.__init__).parameters
    eval_key = "eval_strategy" if "eval_strategy" in signature else "evaluation_strategy"
    kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "num_train_epochs": int(epochs),
        "per_device_train_batch_size": int(train_batch_size),
        "per_device_eval_batch_size": int(eval_batch_size),
        "gradient_accumulation_steps": int(gradient_accumulation_steps),
        "learning_rate": float(learning_rate),
        "weight_decay": WEIGHT_DECAY,
        "warmup_ratio": WARMUP_RATIO,
        "lr_scheduler_type": LR_SCHEDULER,
        eval_key: "epoch",
        "save_strategy": "epoch",
        "save_total_limit": 1,
        "load_best_model_at_end": True,
        "metric_for_best_model": CHECKPOINT_METRIC,
        "greater_is_better": True,
        "logging_strategy": "epoch",
        "seed": int(tuning_seed),
        "data_seed": int(tuning_seed),
        "report_to": "none",
        "fp16": bool(use_fp16),
        "disable_tqdm": bool(disable_tqdm),
    }
    # Avoid incompatibility with minor Transformers versions that have not yet
    # exposed data_seed/lr_scheduler_type; omit only optional aliases if absent.
    for key in ("data_seed", "lr_scheduler_type", "warmup_ratio", "disable_tqdm"):
        if key not in signature:
            kwargs.pop(key, None)
    return TrainingArguments(**kwargs)


def _run_trials(
    audit: Mapping[str, Any],
    *,
    tuning_seed: int,
    epochs: int,
    train_batch_size: int,
    eval_batch_size: int,
    gradient_accumulation_steps: int,
    allow_cpu: bool,
    disable_tqdm: bool,
    project_root: Path,
) -> tuple[dict[str, dict[float, dict[str, float]]], list[dict[str, Any]], dict[str, str], bool]:
    torch, AutoModel, AutoTokenizer, Trainer, TrainingArguments, set_seed = _training_dependencies()
    use_cuda = bool(torch.cuda.is_available())
    if not use_cuda and not allow_cpu:
        raise RuntimeError(
            "No CUDA GPU is available. This preflight comprises eight Transformer fine-tuning runs; "
            "run it on a Colab GPU, or pass --allow-cpu only for a deliberately slow run."
        )

    WeightedLossTrainer = _make_weighted_trainer_class(torch, Trainer)
    TokenizedTextDataset = _make_torch_dataset(torch)
    train_labels = np.asarray([eu.LABEL2ID[label] for label in audit["train_labels"]], dtype=np.int64)
    val_labels = np.asarray([eu.LABEL2ID[label] for label in audit["val_labels"]], dtype=np.int64)
    class_weights = torch.tensor(
        [float(audit["class_weights_by_id"][class_id]) for class_id in range(len(eu.LABEL2ID))],
        dtype=torch.float32,
    )
    max_length = int(audit["sequence_length_doc"]["max_length_benchmark"])
    revisions = audit["sequence_length_doc"]["tokenizer_resolved_revisions"]

    tokenized: dict[str, dict[str, Any]] = {}
    resolved_revisions: dict[str, str] = {}
    for backbone, spec in eu.BACKBONES.items():
        revision = revisions[backbone]
        print(f"\nLoading tokenizer for {backbone}: {spec.hf_model_id} @ {revision}")
        tokenizer = AutoTokenizer.from_pretrained(spec.hf_model_id, revision=revision, use_fast=True)
        tokenizer.truncation_side = TRUNCATION_SIDE
        if tokenizer.truncation_side != TRUNCATION_SIDE:
            raise RuntimeError(f"Could not set target-only truncation_side='right' for {backbone}")
        train_enc = tokenizer(
            audit["train_texts"], truncation=True, max_length=max_length,
            padding="max_length", return_tensors="pt",
        )
        val_enc = tokenizer(
            audit["val_texts"], truncation=True, max_length=max_length,
            padding="max_length", return_tensors="pt",
        )
        tokenized[backbone] = {
            "tokenizer": tokenizer,
            "train_ds": TokenizedTextDataset(train_enc, train_labels),
            "val_ds": TokenizedTextDataset(val_enc, val_labels),
        }
        resolved = (
            getattr(tokenizer, "_commit_hash", None)
            or (getattr(tokenizer, "init_kwargs", {}) or {}).get("_commit_hash")
        )
        if resolved and str(resolved).lower() != str(revision).lower():
            raise RuntimeError(
                f"Tokenizer revision resolved to {resolved!r} for {backbone}, not the Stage 1b pinned revision {revision!r}"
            )
        resolved_revisions[backbone] = str(revision)

    all_scores: dict[str, dict[float, dict[str, float]]] = {b: {} for b in eu.BACKBONES}
    rows: list[dict[str, Any]] = []
    for backbone, spec in eu.BACKBONES.items():
        revision = resolved_revisions[backbone]
        for learning_rate in eu.LR_SEARCH_GRID:
            print(f"\n=== Benchmark LR trial {backbone}, lr={learning_rate:g} ===")
            set_seed(tuning_seed)
            trial_start = time.perf_counter()
            trial_temp = tempfile.TemporaryDirectory(prefix=f"lr-{backbone}-", dir=project_root)
            try:
                model = AutoModel.from_pretrained(
                    spec.hf_model_id,
                    revision=revision,
                    num_labels=len(eu.LABEL2ID),
                    id2label=eu.ID2LABEL,
                    label2id=eu.LABEL2ID,
                )
                train_args = _make_training_args(
                    TrainingArguments,
                    output_dir=Path(trial_temp.name) / "checkpoint",
                    learning_rate=float(learning_rate),
                    tuning_seed=tuning_seed,
                    epochs=epochs,
                    train_batch_size=train_batch_size,
                    eval_batch_size=eval_batch_size,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    use_fp16=use_cuda,
                    disable_tqdm=disable_tqdm,
                )
                trainer = WeightedLossTrainer(
                    model=model,
                    args=train_args,
                    train_dataset=tokenized[backbone]["train_ds"],
                    eval_dataset=tokenized[backbone]["val_ds"],
                    compute_metrics=lambda eval_pred: _metric_summary(eval_pred.label_ids, eval_pred.predictions),
                    class_weights=class_weights,
                )
                trainer.train()
                # load_best_model_at_end=True makes this the validation-selected checkpoint.
                metrics = trainer.evaluate(tokenized[backbone]["val_ds"])
                macro_f1 = float(metrics["eval_macro_f1"])
                if not math.isfinite(macro_f1) or not 0 <= macro_f1 <= 1:
                    raise RuntimeError(f"Invalid validation macro-F1 for {backbone}, lr={learning_rate}: {macro_f1}")
                all_scores[backbone][float(learning_rate)] = {
                    "macro_f1": macro_f1,
                    "accuracy": float(metrics["eval_accuracy"]),
                    "weighted_f1": float(metrics["eval_weighted_f1"]),
                    "f1_n": float(metrics["eval_f1_n"]),
                    "f1_al": float(metrics["eval_f1_al"]),
                    "f1_hs": float(metrics["eval_f1_hs"]),
                }
                rows.append({
                    "run_id": eu.RUN_ID,
                    "stage": "preflight",
                    "backbone": backbone,
                    "model_id": spec.hf_model_id,
                    "model_revision": revision,
                    "tokenizer_revision": revision,
                    "learning_rate": float(learning_rate),
                    "tuning_seed": int(tuning_seed),
                    "train_rows": int(len(audit["train_texts"])),
                    "validation_rows": int(len(audit["val_texts"])),
                    "train_csv_sha256": audit["train_sha256"],
                    "validation_csv_sha256": audit["validation_sha256"],
                    "sequence_length_config_sha256": audit["sequence_config_sha256"],
                    "dataset_sha256": audit["dataset_sha256"],
                    "max_length_benchmark": max_length,
                    "truncation_side": TRUNCATION_SIDE,
                    "epochs_configured": int(epochs),
                    "best_global_step": int(getattr(trainer.state, "best_global_step", 0) or 0),
                    "per_device_train_batch_size": int(train_batch_size),
                    "per_device_eval_batch_size": int(eval_batch_size),
                    "gradient_accumulation_steps": int(gradient_accumulation_steps),
                    "weight_decay": WEIGHT_DECAY,
                    "warmup_ratio": WARMUP_RATIO,
                    "lr_scheduler": LR_SCHEDULER,
                    "validation_macro_f1": macro_f1,
                    "validation_accuracy": all_scores[backbone][float(learning_rate)]["accuracy"],
                    "validation_weighted_f1": all_scores[backbone][float(learning_rate)]["weighted_f1"],
                    "validation_f1_n": all_scores[backbone][float(learning_rate)]["f1_n"],
                    "validation_f1_al": all_scores[backbone][float(learning_rate)]["f1_al"],
                    "validation_f1_hs": all_scores[backbone][float(learning_rate)]["f1_hs"],
                    "best_checkpoint_metric": float(getattr(trainer.state, "best_metric", macro_f1) or macro_f1),
                    "elapsed_seconds": round(time.perf_counter() - trial_start, 3),
                })
                del trainer, model
            finally:
                trial_temp.cleanup()
                if use_cuda:
                    torch.cuda.empty_cache()

    selected = {
        backbone: eu.select_learning_rate(
            {lr: metrics["macro_f1"] for lr, metrics in scores.items()},
            eu.LR_SEARCH_GRID,
        )
        for backbone, scores in all_scores.items()
    }
    parity_cfgs = [
        _build_parity_config(
            backbone, selected[backbone], audit,
            tuning_seed=tuning_seed, fp16=use_cuda, epochs=epochs,
            train_batch_size=train_batch_size, eval_batch_size=eval_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )
        for backbone in eu.BACKBONES
    ]
    eu.check_config_parity(parity_cfgs[0], parity_cfgs[1], input_mode="target_only")
    return all_scores, rows, resolved_revisions, use_cuda


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune benchmark learning rates fairly for both Transformer backbones.")
    parser.add_argument("--train-csv", type=Path, default=TRAIN_PATH)
    parser.add_argument("--val-csv", type=Path, default=VAL_PATH)
    parser.add_argument("--weights-csv", type=Path, default=WEIGHTS_PATH)
    parser.add_argument("--sequence-config", type=Path, default=SEQUENCE_CONFIG_PATH)
    parser.add_argument("--tuning-seed", type=int, default=DEFAULT_TUNING_SEED)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--train-batch-size", type=int, default=TRAIN_BATCH_SIZE)
    parser.add_argument("--eval-batch-size", type=int, default=EVAL_BATCH_SIZE)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=GRADIENT_ACCUMULATION_STEPS)
    parser.add_argument("--allow-cpu", action="store_true", help="Allow the full eight-trial search on CPU; usually very slow.")
    parser.add_argument("--disable-tqdm", action="store_true")
    parser.add_argument("--validate-only", action="store_true", help="Validate inputs/configuration and exit without loading models or training.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.train_batch_size < 1 or args.eval_batch_size < 1 or args.gradient_accumulation_steps < 1:
        raise ValueError("Epochs, batch sizes and gradient accumulation must be positive integers")
    _validate_tuning_seed(args.tuning_seed)

    paths = {
        "train": _resolve_project_path(args.train_csv),
        "validation": _resolve_project_path(args.val_csv),
        "weights": _resolve_project_path(args.weights_csv),
        "sequence_config": _resolve_project_path(args.sequence_config),
    }
    audit = _load_and_validate_inputs(paths["train"], paths["validation"], paths["weights"], paths["sequence_config"])
    print(f"Run namespace: {eu.RUN_ID}")
    print(f"Benchmark train/validation rows: {audit['train_rows']:,}/{audit['validation_rows']:,}")
    print(f"Benchmark sequence length: {audit['sequence_length_doc']['max_length_benchmark']} (source partition: benchmark_train)")
    print(f"Shared LR grid: {tuple(eu.LR_SEARCH_GRID)}")
    print(f"Dedicated tuning seed: {args.tuning_seed}")
    print(f"Training split SHA-256: {audit['train_sha256']}")
    print(f"Validation split SHA-256: {audit['validation_sha256']}")
    print("Benchmark test split: NOT READ")
    if args.validate_only:
        print("Preflight validation passed. No model was loaded and no output was written.")
        return

    # Do not train for several hours only to discover at save time that a prior
    # config/manifest would be overwritten or resumed.
    destinations = [LEARNING_RATE_CONFIG_PATH, STAGE_RUN_DIR, TRIALS_CSV_PATH, BENCHMARK_SNAPSHOT_PATH]
    existing = [str(p) for p in destinations if p.exists()]
    if existing:
        raise FileExistsError(
            "Benchmark LR preflight refuses to overwrite/reuse existing artifacts: "
            + ", ".join(existing)
            + ". Archive the historical run/config before starting the clean pipeline."
        )

    if not args.allow_cpu:
        try:
            import torch
        except ImportError as exc:
            raise ImportError("Install PyTorch in the Colab runtime before the LR preflight") from exc
        if not torch.cuda.is_available():
            raise RuntimeError("No CUDA GPU detected. Switch the Colab runtime to GPU or explicitly pass --allow-cpu.")

    started_at_utc = datetime.now(timezone.utc).isoformat()
    scores_by_backbone, trial_rows, revisions, use_cuda = _run_trials(
        audit,
        tuning_seed=args.tuning_seed,
        epochs=args.epochs,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        allow_cpu=args.allow_cpu,
        disable_tqdm=args.disable_tqdm,
        project_root=PROJECT_ROOT,
    )
    doc = _build_learning_rate_document(
        scores_by_backbone,
        tuning_seed=args.tuning_seed,
        input_audit=audit,
        hyperparameters={
            "optimizer": OPTIMIZER,
            "weight_decay": WEIGHT_DECAY,
            "warmup_ratio": WARMUP_RATIO,
            "lr_scheduler": LR_SCHEDULER,
            "max_epochs": args.epochs,
            "early_stopping_patience": None,
            "per_device_train_batch_size": args.train_batch_size,
            "per_device_eval_batch_size": args.eval_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "effective_batch_size": args.train_batch_size * args.gradient_accumulation_steps,
            "mixed_precision": bool(use_cuda),
            "loss": "weighted_cross_entropy",
            "checkpoint_metric": CHECKPOINT_METRIC,
        },
        resolved_model_revisions=revisions,
    )

    # The utility's shared config loader must accept the exact emitted config
    # before it is published. Validate it in-memory using a temporary file.
    with tempfile.TemporaryDirectory(prefix=".lr-config-validate-", dir=PROJECT_ROOT) as check_dir_name:
        check_dir = Path(check_dir_name)
        (check_dir / "sequence_length.json").write_text(
            json.dumps(audit["sequence_length_doc"], indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (check_dir / "learning_rates.json").write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        eu.load_shared_run_config(
            sequence_length_path=check_dir / "sequence_length.json",
            learning_rate_path=check_dir / "learning_rates.json",
            require_context=False,
            require_context_lr=False,
        )

    trial_df = pd.DataFrame(trial_rows).sort_values(["backbone", "learning_rate"]).reset_index(drop=True)
    expected_trial_count = len(eu.BACKBONES) * len(eu.LR_SEARCH_GRID)
    if len(trial_df) != expected_trial_count:
        raise RuntimeError(f"Expected exactly {expected_trial_count} LR trials, got {len(trial_df)}")
    expected_pairs = {(b, float(lr)) for b in eu.BACKBONES for lr in eu.LR_SEARCH_GRID}
    observed_pairs = set(zip(trial_df["backbone"], trial_df["learning_rate"].astype(float)))
    if observed_pairs != expected_pairs:
        raise RuntimeError(f"LR trial matrix incomplete or duplicated: missing={expected_pairs-observed_pairs}, unexpected={observed_pairs-expected_pairs}")

    completed_at_utc = datetime.now(timezone.utc).isoformat()
    hyperparams = doc["benchmark_preflight"]["hyperparameters"]
    stage_config = {
        "schema_version": eu.CONFIG_SCHEMA_VERSION,
        "run_id": eu.RUN_ID,
        "stage": "preflight",
        "script_name": eu.PIPELINE_SCRIPTS["preflight"],
        "lr_search_grid": [float(x) for x in eu.LR_SEARCH_GRID],
        "lr_tuning_seed": int(args.tuning_seed),
        "learning_rate_benchmark": doc["learning_rate_benchmark"],
        "benchmark_trial_macro_f1": doc["benchmark_trial_macro_f1"],
        "max_length_benchmark": int(audit["sequence_length_doc"]["max_length_benchmark"]),
        "benchmark_sequence_config_sha256": audit["sequence_config_sha256"],
        "benchmark_train_sha256": audit["train_sha256"],
        "benchmark_validation_sha256": audit["validation_sha256"],
        "class_weights_sha256": audit["class_weights_sha256"],
        "hyperparameters": hyperparams,
        "started_at_utc": started_at_utc,
        "completed_at_utc": completed_at_utc,
    }

    LEARNING_RATE_CONFIG_PATH_ROOT = PROJECT_ROOT / LEARNING_RATE_CONFIG_PATH
    if LEARNING_RATE_CONFIG_PATH_ROOT.exists() or STAGE_RUN_DIR.exists():
        raise FileExistsError("Learning-rate config or preflight run directory appeared during training; refusing to publish results")

    with tempfile.TemporaryDirectory(prefix=".preflight-lr-publish-", dir=PROJECT_ROOT) as temp_name:
        temp_dir = Path(temp_name)
        temp_cfg = temp_dir / "learning_rates.json"
        temp_snapshot = temp_dir / "benchmark_learning_rates.json"
        temp_trials = temp_dir / "benchmark_lr_trials.csv"
        text = json.dumps(doc, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        temp_cfg.write_text(text, encoding="utf-8")
        temp_snapshot.write_text(text, encoding="utf-8")
        trial_df.to_csv(temp_trials, index=False)

        snapshot_rel = (BENCHMARK_SNAPSHOT_PATH.relative_to(PROJECT_ROOT)).as_posix()
        trials_rel = TRIALS_CSV_PATH.relative_to(PROJECT_ROOT).as_posix()
        output_hashes = {
            snapshot_rel: eu.sha256_file(temp_snapshot),
            trials_rel: eu.sha256_file(temp_trials),
        }
        metadata = {
            "output_sha256": output_hashes,
            "trial_count": int(len(trial_df)),
            "trial_matrix_complete": True,
            "shared_config_path": LEARNING_RATE_CONFIG_PATH.as_posix(),
            "shared_config_sha256_at_creation": eu.sha256_file(temp_cfg),
            "shared_config_is_extended_by_stage_8": True,
            "selection_rule": "highest validation macro-F1; exact ties go to the smaller learning rate",
            "selected_learning_rates": doc["learning_rate_benchmark"],
        }
        manifest = eu.make_run_manifest(
            stage="preflight",
            script_name=eu.PIPELINE_SCRIPTS["preflight"],
            config=stage_config,
            input_paths=[
                paths["train"], paths["validation"], paths["weights"],
            ],
            metadata=metadata,
            project_root=PROJECT_ROOT,
        )
        run_dir = eu.prepare_run_directory(STAGE_RUN_DIR, manifest, allow_resume=False)
        LEARNING_RATE_CONFIG_PATH_ROOT.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temp_snapshot, run_dir / BENCHMARK_SNAPSHOT_PATH.name)
        os.replace(temp_trials, run_dir / TRIALS_CSV_PATH.name)
        os.replace(temp_cfg, LEARNING_RATE_CONFIG_PATH_ROOT)

    verified = eu.assert_current_stage_manifest("preflight", PROJECT_ROOT)
    print("\nBenchmark LR preflight completed and manifest verified.")
    for backbone in eu.BACKBONES:
        score_dict = {float(k): float(v) for k, v in doc["benchmark_trial_macro_f1"][backbone].items()}
        print(f"  {backbone}: selected LR={doc['learning_rate_benchmark'][backbone]:g}; "
              f"validation macro-F1 by LR={score_dict}")
    print(f"Shared config: {LEARNING_RATE_CONFIG_PATH_ROOT}")
    print(f"Trial metrics: {TRIALS_CSV_PATH}")
    print(f"Immutable benchmark snapshot: {BENCHMARK_SNAPSHOT_PATH}")
    print(f"Stage manifest fingerprint: {verified['fingerprint']}")
    print("Next: run Experiment I with the selected rates, then the ITFT stages. Do not read the benchmark test split during tuning.")


if __name__ == "__main__":
    main()
