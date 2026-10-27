"""Stage 6/7: benchmark Intermediate Task Fine-Tuning (ITFT), one backbone per run.

Authoritative pipeline script: ``scripts/6_train_itft.py``.

Invoke exactly once per backbone, on a GPU-enabled Colab runtime:
    python scripts/6_train_itft.py --backbone indobertweet --validate-only
    python scripts/6_train_itft.py --backbone xlmr --validate-only
    python scripts/6_train_itft.py --backbone indobertweet
    python scripts/6_train_itft.py --backbone xlmr

The requested backbone determines the logical stage:
    Stage 6: IndoBERTweet -> models/itft_checkpoint/indobertweet/
    Stage 7: XLM-RoBERTa-base -> models/itft_checkpoint/xlmr/

ITFT uses only the benchmark training and validation splits. The checkpoint is
selected using validation macro-F1, and the benchmark test split is deliberately
not loaded or evaluated here. The two backbone-specific checkpoints are the
starting points for their corresponding Experiment II contextual runs.

The script uses the shared benchmark max length and the backbone-specific
benchmark learning rate selected by the preflight. It pins model/tokenizer
revisions, applies training-only class weights, verifies split/configuration
hashes, and writes a manifest beneath the versioned run namespace. Historical
outputs from the old models/E2_itft_context/ and results/tables/itft_metrics.csv
paths are never accepted as current checkpoints.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


def _project_root() -> Path:
    """Resolve repository root for scripts stored in the scripts/ directory."""
    here = Path(__file__).resolve().parent
    return here.parent if here.name.casefold() == "scripts" else here


PROJECT_ROOT = _project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import experiment_utils as eu  # noqa: E402

SCRIPT_NAME = "scripts/6_train_itft.py"
STAGE_BY_BACKBONE = {"indobertweet": 6, "xlmr": 7}
INPUT_MODE = "target_only"
SPLIT_SEED = 42
ITFT_SEED = 42
NUM_EPOCHS = 4
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
GRADIENT_ACCUMULATION_STEPS = 1
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.0
LR_SCHEDULER = "linear"
CHECKPOINT_METRIC = "macro_f1"
OPTIMIZER_NAME = "AdamW"
CLASS_WEIGHTING_NAME = "balanced_weights_from_training_split_only"
LOSS_NAME = "weighted_cross_entropy"
TRUNCATION_SIDE = "right"

TRAIN_PATH = Path("data/processed/ib_train.csv")
VAL_PATH = Path("data/processed/ib_val.csv")
WEIGHTS_PATH = Path("results/tables/ib_class_weights.csv")
SEQUENCE_CONFIG_PATH = eu.SEQUENCE_LENGTH_CONFIG
LEARNING_RATE_CONFIG_PATH = eu.LEARNING_RATE_CONFIG

LEGACY_OUTPUT_PATTERNS = (
    "models/E2_itft_context",
    "results/tables/itft_metrics.csv",
)
EXPECTED_SPLIT_ROWS = {"train": 8922, "validation": 1912}


def _abs(path: str | Path) -> Path:
    path = Path(path)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _resolve_legacy_outputs() -> list[Path]:
    found: list[Path] = []
    for pattern in LEGACY_OUTPUT_PATTERNS:
        found.extend(path for path in PROJECT_ROOT.glob(pattern) if path.exists())
    return list(dict.fromkeys(found))


def assert_no_legacy_itft_outputs() -> None:
    """Fail closed on the old shared checkpoint and unversioned metrics table."""
    eu.assert_no_legacy_outputs(_resolve_legacy_outputs())


def _sha(path: Path) -> str:
    return eu.sha256_file(path)


def _validate_text_labels(df: pd.DataFrame, path: Path, split_name: str) -> list[str]:
    required = {"clean_text", "label"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"{path} is missing required column(s) {missing}; Stage 1 must emit clean_text and label")
    if df.empty:
        raise ValueError(f"{split_name} split is empty: {path}")
    if df["label"].isna().any():
        raise ValueError(f"{split_name} split contains missing labels")
    observed_labels = set(df["label"].astype(str).unique())
    expected_labels = set(eu.LABEL2ID)
    if observed_labels != expected_labels:
        raise ValueError(f"{split_name} labels must be exactly {sorted(expected_labels)}; got {sorted(observed_labels)}")
    texts = df["clean_text"].astype("string").fillna("").str.strip().tolist()
    if any(not text for text in texts):
        raise ValueError(f"{split_name} split contains blank clean_text values")
    if pd.Series(texts).duplicated().any():
        raise ValueError(f"{split_name} contains duplicate clean_text; Stage 1 should deduplicate first")
    return texts


def _validate_weight_table(train_df: pd.DataFrame, weights_path: Path) -> tuple[dict[int, float], str]:
    if not weights_path.is_file():
        raise FileNotFoundError(f"Missing training-only class-weight table: {weights_path}; rerun Stage 1")
    recomputed = eu.compute_class_weights(train_df["label"].astype(str))
    saved = pd.read_csv(weights_path)
    if not {"label", "weight"}.issubset(saved.columns):
        raise KeyError(f"{weights_path} must contain columns label and weight")
    if saved["label"].astype(str).duplicated().any():
        raise ValueError("Class-weight table contains duplicate labels")
    if set(saved["label"].astype(str)) != set(eu.LABEL2ID):
        raise ValueError(f"Class-weight labels must be exactly {sorted(eu.LABEL2ID)}")
    saved_by_label = dict(zip(saved["label"].astype(str), pd.to_numeric(saved["weight"], errors="coerce")))
    for label, class_id in eu.LABEL2ID.items():
        observed = saved_by_label.get(label)
        expected = float(recomputed[class_id])
        if observed is None or not math.isfinite(float(observed)) or not np.isclose(float(observed), expected, rtol=1e-5, atol=1e-6):
            raise ValueError(f"Saved class weight for {label} differs from training-only recomputation: {observed!r} vs {expected:.10f}")
    return {int(class_id): float(weight) for class_id, weight in recomputed.items()}, _sha(weights_path)


def _validate_shared_configs(
    train_path: Path,
    val_path: Path,
    weights_path: Path,
    sequence_config_path: Path,
    learning_rate_config_path: Path,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    weights_sha: str,
) -> dict[str, Any]:
    shared = eu.load_shared_run_config(
        sequence_length_path=sequence_config_path,
        learning_rate_path=learning_rate_config_path,
        require_context=False,
        require_context_lr=False,
    )
    length_doc = shared["sequence_length"]
    lr_doc = shared["learning_rates"]
    train_sha = _sha(train_path)
    val_sha = _sha(val_path)
    length_sha = _sha(sequence_config_path)
    preflight = lr_doc.get("benchmark_preflight")
    if not isinstance(preflight, Mapping):
        raise ValueError("learning_rates.json is missing benchmark_preflight provenance")
    checks = {
        "train_csv_sha256": train_sha,
        "validation_csv_sha256": val_sha,
        "class_weights_csv_sha256": weights_sha,
        "sequence_length_config_sha256": length_sha,
        "benchmark_dataset_sha256": eu.sha256_json({"train_sha256": train_sha, "validation_sha256": val_sha}),
        "split_assignment_sha256": eu.sha256_json({"train_sha256": train_sha, "validation_sha256": val_sha, "split_seed": SPLIT_SEED}),
        "max_length_benchmark": length_doc["max_length_benchmark"],
        "truncation_side": TRUNCATION_SIDE,
        "model_ids": {key: eu.get_backbone(key).hf_model_id for key in eu.BACKBONES},
    }
    for key, expected in checks.items():
        if preflight.get(key) != expected:
            raise ValueError(f"Benchmark LR preflight {key!r} does not match the current Stage 1/1b inputs: expected {expected!r}, got {preflight.get(key)!r}")

    if preflight.get("uses_test_split") is not False:
        raise ValueError("Benchmark LR preflight must explicitly record uses_test_split=false")
    expected_rows = EXPECTED_SPLIT_ROWS
    actual_rows = {"train": len(train_df), "validation": len(val_df)}
    if actual_rows != expected_rows:
        raise ValueError(f"Benchmark train/validation row counts differ from the frozen protocol: {actual_rows}; expected {expected_rows}")

    if length_doc.get("benchmark_training_split_sha256") != train_sha:
        raise ValueError("ib_train.csv SHA-256 differs from Stage 1b; rerun Stage 1b using the current Stage 1 output")
    if length_doc.get("benchmark_training_text_column") != "clean_text":
        raise ValueError("Stage 1b must record benchmark_training_text_column='clean_text'")
    expected_model_ids = {key: eu.get_backbone(key).hf_model_id for key in eu.BACKBONES}
    if length_doc.get("tokenizer_model_ids") != expected_model_ids:
        raise ValueError("Stage 1b tokenizer model IDs differ from experiment_utils.BACKBONES")

    tokenizer_revisions = length_doc.get("tokenizer_resolved_revisions")
    model_revisions = preflight.get("model_revisions")
    if not isinstance(tokenizer_revisions, Mapping) or set(tokenizer_revisions) != set(eu.BACKBONES):
        raise ValueError("Stage 1b must pin tokenizer_resolved_revisions for both backbones")
    if not isinstance(model_revisions, Mapping) or set(model_revisions) != set(eu.BACKBONES):
        raise ValueError("Benchmark LR preflight must pin model_revisions for both backbones")
    for key in eu.BACKBONES:
        tok_revision, model_revision = tokenizer_revisions[key], model_revisions[key]
        if not isinstance(tok_revision, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", tok_revision):
            raise ValueError(f"Tokenizer revision for {key} must be a full 40-character commit SHA")
        if not isinstance(model_revision, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", model_revision):
            raise ValueError(f"Model revision for {key} must be a full 40-character commit SHA")
        if tok_revision.lower() != model_revision.lower():
            raise ValueError(f"Model/tokenizer revisions differ for {key}; rerun Stage 1b and the benchmark LR preflight")

    # The benchmark tuning settings are also the fixed settings used by ITFT.
    hyper = preflight.get("hyperparameters")
    if not isinstance(hyper, Mapping):
        raise ValueError("Benchmark preflight must record its resolved hyperparameters")
    expected_hyper = {
        "optimizer": OPTIMIZER_NAME,
        "weight_decay": WEIGHT_DECAY,
        "warmup_ratio": WARMUP_RATIO,
        "lr_scheduler": LR_SCHEDULER,
        "max_epochs": NUM_EPOCHS,
        "early_stopping_patience": None,
        "per_device_train_batch_size": TRAIN_BATCH_SIZE,
        "per_device_eval_batch_size": EVAL_BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "effective_batch_size": TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS,
        "loss": LOSS_NAME,
        "checkpoint_metric": CHECKPOINT_METRIC,
    }
    mismatches = {key: (hyper.get(key), value) for key, value in expected_hyper.items() if hyper.get(key) != value}
    if mismatches:
        raise ValueError(f"ITFT fixed settings differ from benchmark preflight/protocol: {mismatches}")

    return {
        "sequence_length_doc": length_doc,
        "learning_rate_doc": lr_doc,
        "train_sha256": train_sha,
        "validation_sha256": val_sha,
        "class_weights_sha256": weights_sha,
        "sequence_config_sha256": length_sha,
        "learning_rate_config_sha256": _sha(learning_rate_config_path),
        "dataset_sha256": eu.sha256_json({"train_sha256": train_sha, "validation_sha256": val_sha}),
        "split_assignment_sha256": checks["split_assignment_sha256"],
        "model_revisions": dict(model_revisions),
        "tokenizer_revisions": dict(tokenizer_revisions),
        "max_length_benchmark": int(length_doc["max_length_benchmark"]),
        "class_weights_by_id": None,
        "learning_rates": {key: float(lr_doc["learning_rate_benchmark"][key]) for key in eu.BACKBONES},
        "learning_rate_tuning_seed": int(lr_doc["lr_tuning_seed"]),
        "lr_preflight_hyperparameters": dict(hyper),
    }


def _load_and_validate_inputs() -> dict[str, Any]:
    paths = {
        "train": _abs(TRAIN_PATH),
        "validation": _abs(VAL_PATH),
        "weights": _abs(WEIGHTS_PATH),
        "sequence_config": _abs(SEQUENCE_CONFIG_PATH),
        "learning_rate_config": _abs(LEARNING_RATE_CONFIG_PATH),
    }
    for name, path in paths.items():
        if not path.is_file():
            guidance = {
                "train": "Run Stage 1 and transfer ib_train.csv to this environment.",
                "validation": "Run Stage 1 and transfer ib_val.csv to this environment.",
                "weights": "Run Stage 1 and verify the training-only class-weight table.",
                "sequence_config": "Run Stage 1b; this script must not guess max_length_benchmark.",
                "learning_rate_config": "Run scripts/preflight_tune_benchmark_lr.py; do not reuse a historical hard-coded learning rate.",
            }[name]
            raise FileNotFoundError(f"Required ITFT input not found ({name}): {path}. {guidance}")

    train_df = pd.read_csv(paths["train"])
    val_df = pd.read_csv(paths["validation"])
    train_texts = _validate_text_labels(train_df, paths["train"], "train")
    val_texts = _validate_text_labels(val_df, paths["validation"], "validation")
    overlap = set(train_texts) & set(val_texts)
    if overlap:
        raise ValueError(f"Benchmark train/validation leakage: {len(overlap)} exact clean_text value(s) overlap")

    class_weights_by_id, weights_sha = _validate_weight_table(train_df, paths["weights"])
    audit = _validate_shared_configs(
        paths["train"], paths["validation"], paths["weights"], paths["sequence_config"],
        paths["learning_rate_config"], train_df, val_df, weights_sha,
    )
    audit.update({
        "paths": paths,
        "train_df": train_df,
        "val_df": val_df,
        "train_texts": train_texts,
        "val_texts": val_texts,
        "train_labels": train_df["label"].astype(str).tolist(),
        "val_labels": val_df["label"].astype(str).tolist(),
        "class_weights_by_id": class_weights_by_id,
    })
    return audit


def _resolved_config(backbone: str, audit: Mapping[str, Any], *, mixed_precision: bool) -> dict[str, Any]:
    """Return the stage-specific manifest config, validated against the utility."""
    if backbone not in STAGE_BY_BACKBONE:
        raise KeyError(f"Unsupported ITFT backbone {backbone!r}")
    stage = STAGE_BY_BACKBONE[backbone]
    spec = eu.get_backbone(backbone)
    cfg: dict[str, Any] = {
        "schema_version": eu.CONFIG_SCHEMA_VERSION,
        "run_id": eu.RUN_ID,
        "stage": stage,
        "script_name": SCRIPT_NAME,
        "experiment_id": f"ITFT-{spec.id_token}",
        "backbone": backbone,
        "hf_model_id": spec.hf_model_id,
        "model_revision": audit["model_revisions"][backbone],
        "tokenizer_revision": audit["tokenizer_revisions"][backbone],
        "seed": ITFT_SEED,
        "input_mode": INPUT_MODE,
        "input_order": "target_only",
        "context_order": "target_only",
        "dataset_sha256": audit["dataset_sha256"],
        "train_split_sha256": audit["train_sha256"],
        "validation_split_sha256": audit["validation_sha256"],
        "class_weights_sha256": audit["class_weights_sha256"],
        "sequence_length_config_sha256": audit["sequence_config_sha256"],
        "learning_rate_config_sha256": audit["learning_rate_config_sha256"],
        "split_assignment_sha256": audit["split_assignment_sha256"],
        "split_seed": SPLIT_SEED,
        "eval_seeds_exp1": list(eu.EVAL_SEEDS_EXP1),
        "cv_seeds": list(eu.CV_SEEDS),
        "max_length_benchmark": int(audit["max_length_benchmark"]),
        "truncation_side": TRUNCATION_SIDE,
        "optimizer": OPTIMIZER_NAME,
        "weight_decay": WEIGHT_DECAY,
        "warmup_ratio": WARMUP_RATIO,
        "lr_scheduler": LR_SCHEDULER,
        "max_epochs": NUM_EPOCHS,
        "early_stopping_patience": None,
        "loss": LOSS_NAME,
        "class_weighting_benchmark": CLASS_WEIGHTING_NAME,
        "checkpoint_metric": CHECKPOINT_METRIC,
        "lr_search_grid": [float(value) for value in eu.LR_SEARCH_GRID],
        "lr_tuning_seed": int(audit["learning_rate_tuning_seed"]),
        "mixed_precision": bool(mixed_precision),
        "per_device_batch_size": TRAIN_BATCH_SIZE,
        "eval_batch_size": EVAL_BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "effective_batch_size": TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS,
        "learning_rate_benchmark": float(audit["learning_rates"][backbone]),
        "class_weights_by_label": {
            eu.ID2LABEL[class_id]: float(weight)
            for class_id, weight in audit["class_weights_by_id"].items()
        },
    }
    eu.validate_resolved_run_config(
        cfg, stage=stage, backbone=backbone, input_mode=INPUT_MODE, require_shared_config=False,
    )
    return cfg


def _metrics_for_ids(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    labels = [0, 1, 2]
    return {
        "accuracy": float(np.mean(np.asarray(y_true) == np.asarray(y_pred))),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "f1_n": float(f1_score(y_true, y_pred, labels=[0], average="macro", zero_division=0)),
        "f1_al": float(f1_score(y_true, y_pred, labels=[1], average="macro", zero_division=0)),
        "f1_hs": float(f1_score(y_true, y_pred, labels=[2], average="macro", zero_division=0)),
    }


def _compute_metrics(eval_pred: Any) -> dict[str, float]:
    logits, labels = eval_pred
    y_true = np.asarray(labels, dtype=np.int64)
    y_pred = np.asarray(logits).argmax(axis=1)
    return _metrics_for_ids(y_true, y_pred)


def _write_json_atomic(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _sha_tree(model_dir: Path, *, exclude_manifest: bool = True) -> dict[str, str]:
    if not model_dir.is_dir():
        raise FileNotFoundError(f"ITFT checkpoint directory not found: {model_dir}")
    result: dict[str, str] = {}
    for path in sorted(model_dir.rglob("*")):
        if not path.is_file():
            continue
        if exclude_manifest and path.name == "itft_checkpoint_manifest.json":
            continue
        result[path.relative_to(model_dir).as_posix()] = _sha(path)
    if not result:
        raise RuntimeError(f"No model/tokenizer files were saved to {model_dir}")
    return result


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot read JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object in {path}")
    return value


def _completed_outputs_are_valid(run_dir: Path, model_dir: Path, manifest: Mapping[str, Any]) -> bool:
    marker_path = run_dir / "checkpoint_complete.json"
    artifact_index_path = run_dir / "artifacts_sha256.json"
    metrics_path = run_dir / "metrics.json"
    model_manifest_path = model_dir / "itft_checkpoint_manifest.json"

    if not marker_path.is_file():
        return False
    marker = _read_json(marker_path)
    fingerprint = str(manifest["fingerprint"])
    expected_model_path = str(model_dir.relative_to(PROJECT_ROOT))
    if (
        marker.get("manifest_fingerprint") != fingerprint
        or marker.get("stage") != manifest["stage"]
        or marker.get("backbone") != manifest["config"].get("backbone")
        or marker.get("checkpoint_directory") != expected_model_path
    ):
        raise RuntimeError(f"ITFT completion marker does not match the current run/checkpoint: {marker_path}")
    if not artifact_index_path.is_file() or not metrics_path.is_file() or not model_manifest_path.is_file():
        raise RuntimeError("ITFT run is marked complete but one or more required artifacts are missing")

    index = _read_json(artifact_index_path)
    if index.get("run_id") != eu.RUN_ID or index.get("stage") != manifest["stage"] or index.get("manifest_fingerprint") != fingerprint:
        raise RuntimeError(f"ITFT artifact hash index does not match the current manifest: {artifact_index_path}")
    checks = {
        "metrics.json": _sha(metrics_path),
        "checkpoint_manifest_sha256": _sha(model_manifest_path),
    }
    hashes = index.get("sha256", {})
    for key, actual in checks.items():
        if hashes.get(key) != actual:
            raise RuntimeError(f"ITFT artifact integrity failure for {key}; archive the run and retrain")
    if marker.get("checkpoint_manifest_sha256") != checks["checkpoint_manifest_sha256"]:
        raise RuntimeError("ITFT completion marker checkpoint-manifest hash is inconsistent")

    model_doc = _read_json(model_manifest_path)
    if model_doc.get("manifest_fingerprint") != fingerprint or model_doc.get("backbone") != marker.get("backbone"):
        raise RuntimeError("Saved ITFT checkpoint manifest belongs to a different run/backbone")
    expected_files = model_doc.get("checkpoint_files_sha256")
    if not isinstance(expected_files, Mapping):
        raise RuntimeError("ITFT checkpoint manifest lacks checkpoint_files_sha256")
    actual_files = _sha_tree(model_dir)
    if dict(expected_files) != actual_files:
        raise RuntimeError("ITFT checkpoint files were modified or differ from the manifest; do not reuse them")

    eu.assert_current_run_artifact(
        marker_path,
        expected_stage=manifest["stage"],
        expected_script=SCRIPT_NAME,
        expected_fingerprint=fingerprint,
    )
    return True


def _make_dataset_class(torch: Any):
    class TokenizedTextDataset(torch.utils.data.Dataset):
        def __init__(self, encodings: Mapping[str, Any], labels: Sequence[int]):
            self.encodings = encodings
            self.labels = np.asarray(labels, dtype=np.int64)

        def __len__(self) -> int:
            return len(self.labels)

        def __getitem__(self, index: int) -> dict[str, Any]:
            item = {key: value[index] for key, value in self.encodings.items()}
            item["labels"] = torch.tensor(int(self.labels[index]), dtype=torch.long)
            return item

    return TokenizedTextDataset


def _make_weighted_trainer_class(torch: Any, Trainer: Any):
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


def _make_training_arguments(
    TrainingArguments: Any,
    *,
    output_dir: Path,
    learning_rate: float,
    seed: int,
    use_fp16: bool,
    disable_tqdm: bool,
):
    import inspect

    signature = inspect.signature(TrainingArguments.__init__).parameters
    eval_key = "eval_strategy" if "eval_strategy" in signature else "evaluation_strategy"
    kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "num_train_epochs": NUM_EPOCHS,
        "per_device_train_batch_size": TRAIN_BATCH_SIZE,
        "per_device_eval_batch_size": EVAL_BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
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
        "seed": int(seed),
        "data_seed": int(seed),
        "report_to": "none",
        "fp16": bool(use_fp16),
        "disable_tqdm": bool(disable_tqdm),
    }
    for key in ("data_seed", "lr_scheduler_type", "warmup_ratio", "disable_tqdm"):
        if key not in signature:
            kwargs.pop(key, None)
    return TrainingArguments(**kwargs)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one backbone-specific benchmark ITFT checkpoint (Stage 6 or 7).")
    parser.add_argument("--backbone", choices=sorted(STAGE_BY_BACKBONE), required=True, help="indobertweet (Stage 6) or xlmr (Stage 7)")
    parser.add_argument("--validate-only", action="store_true", help="Validate data/configuration/path safety without loading models or training.")
    parser.add_argument("--allow-cpu", action="store_true", help="Allow training on CPU deliberately; not recommended.")
    parser.add_argument("--disable-tqdm", action="store_true", help="Disable Hugging Face Trainer progress bars.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    backbone = args.backbone
    stage = STAGE_BY_BACKBONE[backbone]
    assert_no_legacy_itft_outputs()
    audit = _load_and_validate_inputs()

    # Verify both configurations before either backbone is trained. The model,
    # tokenizer and tuned LR may differ; the rest of the effective protocol must
    # be shared (including effective batch size and the benchmark sequence length).
    parity_cfgs = {
        key: _resolved_config(key, audit, mixed_precision=False)
        for key in eu.BACKBONES
    }
    eu.check_config_parity(parity_cfgs["indobertweet"], parity_cfgs["xlmr"], input_mode=INPUT_MODE)

    migration_audit = eu.audit_pipeline_script_names(PROJECT_ROOT, require_all=False)
    if SCRIPT_NAME not in migration_audit["present"]:
        raise FileNotFoundError(
            f"The authoritative ITFT script is missing at {SCRIPT_NAME!r}; save this revised file in scripts/."
        )

    spec = eu.get_backbone(backbone)
    model_revision = audit["model_revisions"][backbone]
    tokenizer_revision = audit["tokenizer_revisions"][backbone]
    learning_rate = audit["learning_rates"][backbone]
    max_length = audit["max_length_benchmark"]
    print(f"Pipeline run ID: {eu.RUN_ID}")
    print(f"Stage {stage}: ITFT-{spec.id_token}")
    print(f"Backbone: {spec.display_name} ({spec.hf_model_id})")
    print(f"Model revision: {model_revision}")
    print(f"Tokenizer revision: {tokenizer_revision}")
    print(f"Benchmark sequence length: {max_length}; truncation_side={TRUNCATION_SIDE}")
    print(f"Selected benchmark learning rate: {learning_rate:g}")
    print(f"Training/validation rows: {len(audit['train_df'])}/{len(audit['val_df'])}")
    print("Checkpoint selection: validation macro-F1; benchmark test split is not loaded")
    print("Historical flat-path outputs: none found")

    if args.validate_only:
        cfg_indobertweet = _resolved_config("indobertweet", audit, mixed_precision=False)
        cfg_xlmr = _resolved_config("xlmr", audit, mixed_precision=False)
        eu.check_config_parity(cfg_indobertweet, cfg_xlmr, input_mode=INPUT_MODE)
        print(f"Stage {stage} data/configuration validation passed. No model was loaded and no outputs were written.")
        if migration_audit["complete"]:
            print("Full pipeline filename/directory audit: passed.")
        else:
            print("Full pipeline filename/directory audit is still incomplete; actual training remains blocked.")
            print(f"Missing scripts: {migration_audit['missing']}")
            print(f"Missing directories: {migration_audit['missing_directories']}")
            print(f"Old filenames still present: {migration_audit['rename_leftovers']}")
        return

    # The whole pipeline must be migrated before model training begins. The
    # validate-only mode remains available while other scripts are in revision.
    eu.audit_pipeline_script_names(PROJECT_ROOT, require_all=True)

    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, set_seed
    except ImportError as exc:
        raise ImportError("ITFT requires PyTorch and Hugging Face Transformers; use the project training environment/Colab.") from exc

    use_cuda = bool(torch.cuda.is_available())
    if not use_cuda and not args.allow_cpu:
        raise RuntimeError("No CUDA GPU detected. Use a Colab GPU runtime or pass --allow-cpu only for a deliberately slow run.")
    preflight_mixed_precision = bool(audit["lr_preflight_hyperparameters"].get("mixed_precision"))
    if preflight_mixed_precision != use_cuda:
        raise RuntimeError(
            "The current runtime's mixed-precision setting differs from the benchmark LR preflight. "
            f"Preflight mixed_precision={preflight_mixed_precision}, current CUDA availability={use_cuda}. "
            "Use the same GPU/precision mode as the preflight or rerun the preflight under the intended environment."
        )

    # Actual mixed precision is part of the resolved run config; check cross-
    # backbone parity again using the current runtime's effective setting.
    config = _resolved_config(backbone, audit, mixed_precision=use_cuda)
    peer_key = "xlmr" if backbone == "indobertweet" else "indobertweet"
    peer_config = _resolved_config(peer_key, audit, mixed_precision=use_cuda)
    eu.check_config_parity(config, peer_config, input_mode=INPUT_MODE)

    stage_run_dir = eu.versioned_run_root(PROJECT_ROOT) / f"stage_{stage:02d}" / f"itft_{backbone}"
    model_out_dir = _abs(Path("models") / "itft_checkpoint" / backbone)
    manifest = eu.make_run_manifest(
        stage=stage,
        script_name=SCRIPT_NAME,
        config=config,
        input_paths=[
            audit["paths"]["train"], audit["paths"]["validation"], audit["paths"]["weights"],
            audit["paths"]["sequence_config"], audit["paths"]["learning_rate_config"],
        ],
        metadata={
            "experiment_id": config["experiment_id"],
            "checkpoint_selection": "highest validation macro-F1",
            "benchmark_test_split_loaded": False,
            "canonical_checkpoint_directory": str(model_out_dir.relative_to(PROJECT_ROOT)),
        },
        project_root=PROJECT_ROOT,
    )
    eu.prepare_run_directory(stage_run_dir, manifest, allow_resume=True)
    if _completed_outputs_are_valid(stage_run_dir, model_out_dir, manifest):
        print(f"Stage {stage} ITFT checkpoint already exists and passed manifest/hash verification; skipping identical run.")
        return

    unindexed_run_files = [path for path in stage_run_dir.rglob("*") if path.is_file() and path.name != eu.RUN_MANIFEST_NAME]
    if unindexed_run_files:
        raise RuntimeError(f"Incomplete ITFT run artifacts exist in {stage_run_dir}; archive the run directory before retrying")
    if model_out_dir.exists():
        raise RuntimeError(
            f"Checkpoint directory already exists without a valid current completion record: {model_out_dir}. "
            "Archive it before rerunning; the script will not overwrite or reuse it."
        )

    tokenizer = AutoTokenizer.from_pretrained(spec.hf_model_id, revision=tokenizer_revision, use_fast=True)
    eu.configure_tokenizer_for_input(tokenizer, INPUT_MODE)
    actual_tok_revision = getattr(tokenizer, "_commit_hash", None) or (getattr(tokenizer, "init_kwargs", {}) or {}).get("_commit_hash")
    if not actual_tok_revision or str(actual_tok_revision).lower() != tokenizer_revision.lower():
        raise RuntimeError(f"Tokenizer resolved to {actual_tok_revision!r}; expected pinned revision {tokenizer_revision!r}")
    if tokenizer.truncation_side != TRUNCATION_SIDE:
        raise RuntimeError(f"Target-only benchmark ITFT requires truncation_side={TRUNCATION_SIDE!r}")

    train_labels = np.asarray([eu.LABEL2ID[label] for label in audit["train_labels"]], dtype=np.int64)
    val_labels = np.asarray([eu.LABEL2ID[label] for label in audit["val_labels"]], dtype=np.int64)
    print("Tokenizing benchmark train/validation partitions with the frozen shared max length...")
    train_enc = tokenizer(audit["train_texts"], truncation=True, max_length=max_length, padding="max_length", return_tensors="pt")
    val_enc = tokenizer(audit["val_texts"], truncation=True, max_length=max_length, padding="max_length", return_tensors="pt")
    Dataset = _make_dataset_class(torch)
    train_ds = Dataset(train_enc, train_labels)
    val_ds = Dataset(val_enc, val_labels)
    class_weights = torch.tensor([audit["class_weights_by_id"][i] for i in range(len(eu.LABEL2ID))], dtype=torch.float32)

    print(f"\n=== Stage {stage} | ITFT-{spec.id_token} | seed={ITFT_SEED} ===")
    set_seed(ITFT_SEED)
    start = time.perf_counter()
    model = AutoModelForSequenceClassification.from_pretrained(
        spec.hf_model_id,
        revision=model_revision,
        num_labels=len(eu.LABEL2ID),
        id2label=eu.ID2LABEL,
        label2id=eu.LABEL2ID,
    )
    actual_model_revision = getattr(model.config, "_commit_hash", None)
    if not actual_model_revision or str(actual_model_revision).lower() != model_revision.lower():
        raise RuntimeError(f"Model resolved to {actual_model_revision!r}; expected pinned revision {model_revision!r}")

    WeightedLossTrainer = _make_weighted_trainer_class(torch, Trainer)
    with tempfile.TemporaryDirectory(prefix=f"itft-{backbone}-", dir=PROJECT_ROOT) as tmp_name:
        temp_output = Path(tmp_name) / "trainer_checkpoints"
        training_args = _make_training_arguments(
            TrainingArguments,
            output_dir=temp_output,
            learning_rate=learning_rate,
            seed=ITFT_SEED,
            use_fp16=use_cuda,
            disable_tqdm=args.disable_tqdm,
        )
        trainer = WeightedLossTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            compute_metrics=_compute_metrics,
            class_weights=class_weights,
        )
        trainer.train()
        # load_best_model_at_end=True makes this the validation-selected model.
        val_metrics = trainer.evaluate(val_ds)
        elapsed_seconds = time.perf_counter() - start
        metrics = {
            "schema_version": eu.CONFIG_SCHEMA_VERSION,
            "run_id": eu.RUN_ID,
            "stage": stage,
            "script_name": SCRIPT_NAME,
            "backbone": backbone,
            "experiment_id": config["experiment_id"],
            "seed": ITFT_SEED,
            "checkpoint_selection": "highest validation macro-F1",
            "best_global_step": int(getattr(trainer.state, "best_global_step", 0) or 0),
            "best_checkpoint_metric": float(getattr(trainer.state, "best_metric", val_metrics["eval_macro_f1"]) or val_metrics["eval_macro_f1"]),
            "validation_accuracy": float(val_metrics["eval_accuracy"]),
            "validation_macro_f1": float(val_metrics["eval_macro_f1"]),
            "validation_weighted_f1": float(val_metrics["eval_weighted_f1"]),
            "validation_f1_n": float(val_metrics["eval_f1_n"]),
            "validation_f1_al": float(val_metrics["eval_f1_al"]),
            "validation_f1_hs": float(val_metrics["eval_f1_hs"]),
            "train_rows": int(len(train_ds)),
            "validation_rows": int(len(val_ds)),
            "train_csv_sha256": audit["train_sha256"],
            "validation_csv_sha256": audit["validation_sha256"],
            "class_weights_csv_sha256": audit["class_weights_sha256"],
            "sequence_length_config_sha256": audit["sequence_config_sha256"],
            "learning_rate_config_sha256": audit["learning_rate_config_sha256"],
            "max_length_benchmark": max_length,
            "learning_rate_benchmark": learning_rate,
            "truncation_side": TRUNCATION_SIDE,
            "model_revision": model_revision,
            "tokenizer_revision": tokenizer_revision,
            "elapsed_seconds": round(elapsed_seconds, 3),
            "effective_batch_size": TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS,
            "mixed_precision": use_cuda,
        }

        model_out_dir.parent.mkdir(parents=True, exist_ok=True)
        if model_out_dir.exists():
            raise FileExistsError(f"Refusing to overwrite ITFT output path {model_out_dir}")
        trainer.save_model(str(model_out_dir))
        tokenizer.save_pretrained(str(model_out_dir))

    checkpoint_files = _sha_tree(model_out_dir)
    model_manifest = {
        "schema_version": eu.CONFIG_SCHEMA_VERSION,
        "run_id": eu.RUN_ID,
        "stage": stage,
        "script_name": SCRIPT_NAME,
        "backbone": backbone,
        "hf_model_id": spec.hf_model_id,
        "model_revision": model_revision,
        "tokenizer_revision": tokenizer_revision,
        "manifest_fingerprint": manifest["fingerprint"],
        "config": config,
        "checkpoint_files_sha256": checkpoint_files,
        "validation_metrics": metrics,
    }
    _write_json_atomic(model_out_dir / "itft_checkpoint_manifest.json", model_manifest)
    _write_json_atomic(stage_run_dir / "metrics.json", metrics)
    _write_json_atomic(stage_run_dir / "artifacts_sha256.json", {
        "run_id": eu.RUN_ID,
        "stage": stage,
        "backbone": backbone,
        "manifest_fingerprint": manifest["fingerprint"],
        "sha256": {
            "metrics.json": _sha(stage_run_dir / "metrics.json"),
            "checkpoint_manifest_sha256": _sha(model_out_dir / "itft_checkpoint_manifest.json"),
        },
    })
    _write_json_atomic(stage_run_dir / "checkpoint_complete.json", {
        "schema_version": eu.CONFIG_SCHEMA_VERSION,
        "run_id": eu.RUN_ID,
        "stage": stage,
        "script_name": SCRIPT_NAME,
        "backbone": backbone,
        "manifest_fingerprint": manifest["fingerprint"],
        "checkpoint_directory": str(model_out_dir.relative_to(PROJECT_ROOT)),
        "checkpoint_manifest_sha256": _sha(model_out_dir / "itft_checkpoint_manifest.json"),
    })

    if not _completed_outputs_are_valid(stage_run_dir, model_out_dir, manifest):
        raise RuntimeError("ITFT checkpoint artifacts were written but failed post-save manifest/hash verification")

    print(f"\nStage {stage} completed in {elapsed_seconds / 60:.1f} minutes.")
    print(f"Validation macro-F1: {metrics['validation_macro_f1']:.4f}")
    print(f"Checkpoint: {model_out_dir}")
    print(f"Run manifest: {stage_run_dir / eu.RUN_MANIFEST_NAME}")
    print("The benchmark test split was not read; ITFT checkpoint selection used validation macro-F1 only.")


if __name__ == "__main__":
    main()
