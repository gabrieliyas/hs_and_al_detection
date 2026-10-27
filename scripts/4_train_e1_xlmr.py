"""Stage 4: Experiment I — XLM-RoBERTa-base target-only benchmark training.

Authoritative stage: 4 in pipeline.md / experiment_protocol.md.

Run on a GPU-enabled Colab runtime after completing Stage 1, Stage 1b,
Stage 2, the benchmark learning-rate preflight, and Stage 3 (IndoBERTweet):
    python scripts/4_train_e1_xlmr.py --validate-only
    python scripts/4_train_e1_xlmr.py

This script trains one run for each frozen Experiment I seed in
experiment_utils.EVAL_SEEDS_EXP1. It uses the pinned model/tokenizer revision,
the shared benchmark sequence length, and the selected XLM-RoBERTa-base benchmark
learning rate. It trains on ib_train.csv, selects checkpoints with validation
macro-F1, and evaluates the selected checkpoint on ib_test.csv only after
training is complete.

Outputs are isolated under:
    results/runs/left_truncation_v2/experiment_i/xlmr/seed_<seed>/
Each completed seed directory contains predictions.csv, probabilities.npy,
metrics.json, artifacts_sha256.json, and the immutable _run_manifest.json.
No legacy results/raw or results/tables files are used as resume signals.
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
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


def _project_root() -> Path:
    """Resolve repository root for scripts stored in scripts/."""
    here = Path(__file__).resolve().parent
    return here.parent if here.name.casefold() == "scripts" else here


PROJECT_ROOT = _project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import experiment_utils as eu  # noqa: E402

SCRIPT_NAME = "scripts/4_train_e1_xlmr.py"
STAGE = 4
BACKBONE = "xlmr"
INPUT_MODE = "target_only"
SPLIT_SEED = 42
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
LEGACY_OUTPUTS = (
    "results/raw/e1_transformer_seed*_predictions.csv",
    "results/raw/e1_transformer_seed*_probs.npy",
    "results/tables/e1_transformer_seed_metrics.csv",
    "models/E1_baseline/e1_xlmr_seed*",
)

TRAIN_PATH = Path("data/processed/ib_train.csv")
VAL_PATH = Path("data/processed/ib_val.csv")
TEST_PATH = Path("data/processed/ib_test.csv")
WEIGHTS_PATH = Path("results/tables/ib_class_weights.csv")
SEQUENCE_CONFIG_PATH = eu.SEQUENCE_LENGTH_CONFIG
LEARNING_RATE_CONFIG_PATH = eu.LEARNING_RATE_CONFIG


def _abs(path: str | Path) -> Path:
    path = Path(path)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _resolve_legacy_outputs() -> list[Path]:
    """Find historical flat-path outputs that must be archived, not reused."""
    found: list[Path] = []
    for pattern in LEGACY_OUTPUTS:
        found.extend(p for p in PROJECT_ROOT.glob(pattern) if p.exists())
    # De-duplicate while preserving order.
    return list(dict.fromkeys(found))


def assert_no_legacy_stage4_outputs() -> None:
    """Reject the original unversioned Experiment I result paths."""
    found = _resolve_legacy_outputs()
    eu.assert_no_legacy_outputs(found)


def _sha(path: Path) -> str:
    return eu.sha256_file(path)


def _validate_text_labels(df: pd.DataFrame, path: Path, split_name: str) -> list[str]:
    required = {"clean_text", "label"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"{path} lacks required column(s) {missing}; Stage 1 must emit clean_text and label")
    if df.empty:
        raise ValueError(f"{split_name} split is empty: {path}")
    if df["label"].isna().any():
        raise ValueError(f"{split_name} contains missing labels")
    labels = set(df["label"].astype(str).unique())
    expected = set(eu.LABEL2ID)
    if labels != expected:
        raise ValueError(f"{split_name} labels must be exactly {sorted(expected)}; got {sorted(labels)}")
    texts = df["clean_text"].astype("string").fillna("").str.strip().tolist()
    blank_rows = [i for i, text in enumerate(texts) if not text]
    if blank_rows:
        raise ValueError(f"{split_name} contains blank clean_text at row positions {blank_rows[:10]}")
    if pd.Series(texts).duplicated().any():
        raise ValueError(f"{split_name} contains duplicate clean_text rows; Stage 1 should deduplicate first")
    return texts


def _validate_weight_table(train_df: pd.DataFrame, weights_path: Path) -> tuple[dict[int, float], str]:
    if not weights_path.is_file():
        raise FileNotFoundError(f"Missing training-only class-weight table: {weights_path}; rerun Stage 1")
    by_id = eu.compute_class_weights(train_df["label"].astype(str))
    saved = pd.read_csv(weights_path)
    if not {"label", "weight"}.issubset(saved.columns):
        raise KeyError(f"{weights_path} must contain columns label and weight")
    if saved["label"].astype(str).duplicated().any():
        raise ValueError("Class-weight table contains duplicate labels")
    if set(saved["label"].astype(str)) != set(eu.LABEL2ID):
        raise ValueError(f"Class-weight table labels must be exactly {sorted(eu.LABEL2ID)}")
    saved_by_label = dict(zip(saved["label"].astype(str), pd.to_numeric(saved["weight"], errors="coerce")))
    for label, class_id in eu.LABEL2ID.items():
        observed = saved_by_label.get(label)
        expected = float(by_id[class_id])
        if observed is None or not math.isfinite(float(observed)) or not np.isclose(float(observed), expected, rtol=1e-5, atol=1e-6):
            raise ValueError(f"Saved class weight for {label} disagrees with training-only recomputation: saved={observed!r}, expected={expected:.10f}")
    return {int(k): float(v) for k, v in by_id.items()}, _sha(weights_path)


def _load_and_validate_inputs() -> dict[str, Any]:
    paths = {
        "train": _abs(TRAIN_PATH),
        "validation": _abs(VAL_PATH),
        "test": _abs(TEST_PATH),
        "weights": _abs(WEIGHTS_PATH),
        "sequence_config": _abs(SEQUENCE_CONFIG_PATH),
        "learning_rate_config": _abs(LEARNING_RATE_CONFIG_PATH),
    }
    for name, path in paths.items():
        if not path.is_file():
            if name == "sequence_config":
                detail = "Run Stage 1b after Stage 1; this script must not guess max_length_benchmark."
            elif name == "learning_rate_config":
                detail = "Run scripts/preflight_tune_benchmark_lr.py; do not reuse the former hard-coded 2e-5."
            elif name == "weights":
                detail = "Rerun Stage 1 and validate the training-only class weights."
            else:
                detail = "Run scripts/1_preprocess_benchmark.py and transfer the processed split to this environment."
            raise FileNotFoundError(f"Required Stage 4 input not found ({name}): {path}. {detail}")

    train_df = pd.read_csv(paths["train"])
    val_df = pd.read_csv(paths["validation"])
    test_df = pd.read_csv(paths["test"])
    train_texts = _validate_text_labels(train_df, paths["train"], "train")
    val_texts = _validate_text_labels(val_df, paths["validation"], "validation")
    test_texts = _validate_text_labels(test_df, paths["test"], "test")

    # Stage 1 should create disjoint splits from a deduplicated source dataset.
    split_sets = {"train": set(train_texts), "validation": set(val_texts), "test": set(test_texts)}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = split_sets[left] & split_sets[right]
        if overlap:
            raise ValueError(f"Benchmark split leakage: {len(overlap)} exact clean_text value(s) occur in {left} and {right}")

    # The protocol's documented post-cleaning count and fixed 70/15/15 split.
    expected_counts = {"train": 8922, "validation": 1912, "test": 1912}
    actual_counts = {"train": len(train_df), "validation": len(val_df), "test": len(test_df)}
    if actual_counts != expected_counts:
        raise ValueError(f"Benchmark split counts differ from the frozen Stage 1 output: actual={actual_counts}, expected={expected_counts}. Investigate Stage 1 rather than silently proceeding.")

    class_weights_by_id, weights_sha = _validate_weight_table(train_df, paths["weights"])
    # This shared loader verifies schema, run ID, length-selection statistics,
    # the two declared truncation policies, all LR trials, and selected rates.
    shared = eu.load_shared_run_config(
        sequence_length_path=paths["sequence_config"],
        learning_rate_path=paths["learning_rate_config"],
        require_context=False,
        require_context_lr=False,
    )
    length_doc = shared["sequence_length"]
    lr_doc = shared["learning_rates"]

    train_sha = _sha(paths["train"])
    val_sha = _sha(paths["validation"])
    test_sha = _sha(paths["test"])
    expected_train_sha = length_doc.get("benchmark_training_split_sha256")
    if expected_train_sha != train_sha:
        raise ValueError("ib_train.csv SHA-256 differs from Stage 1b's benchmark_training_split_sha256; rerun Stage 1b against this exact split")
    if length_doc.get("benchmark_training_text_column") != "clean_text":
        raise ValueError("Stage 1b must record benchmark_training_text_column='clean_text'")
    if length_doc.get("tokenizer_model_ids") != {b: eu.BACKBONES[b].hf_model_id for b in eu.BACKBONES}:
        raise ValueError("Stage 1b tokenizer_model_ids do not match the authoritative backbone registry")
    tokenizer_revisions = length_doc.get("tokenizer_resolved_revisions")
    if not isinstance(tokenizer_revisions, Mapping) or set(tokenizer_revisions) != set(eu.BACKBONES):
        raise ValueError("Stage 1b must pin tokenizer_resolved_revisions for both backbones")
    for backbone, revision in tokenizer_revisions.items():
        if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
            raise ValueError(f"Tokenizer revision for {backbone} must be a full 40-character commit SHA")

    preflight = lr_doc.get("benchmark_preflight")
    if not isinstance(preflight, Mapping):
        raise ValueError("learning_rates.json is missing benchmark_preflight provenance")
    if preflight.get("train_csv_sha256") != train_sha or preflight.get("validation_csv_sha256") != val_sha:
        raise ValueError("Learning-rate preflight used different benchmark train/validation CSVs")
    if preflight.get("class_weights_csv_sha256") != weights_sha:
        raise ValueError("Learning-rate preflight used a different class-weight table")
    if preflight.get("sequence_length_config_sha256") != _sha(paths["sequence_config"]):
        raise ValueError("Learning-rate preflight used a different sequence_length.json")
    if preflight.get("benchmark_dataset_sha256") != eu.sha256_json({"train_sha256": train_sha, "validation_sha256": val_sha}):
        raise ValueError("Learning-rate preflight benchmark dataset hash does not match current train/validation splits")
    expected_split_sha = eu.sha256_json({"train_sha256": train_sha, "validation_sha256": val_sha, "split_seed": SPLIT_SEED})
    if preflight.get("split_assignment_sha256") != expected_split_sha:
        raise ValueError("Learning-rate preflight split-assignment hash does not match current splits and seed")
    if preflight.get("max_length_benchmark") != length_doc["max_length_benchmark"]:
        raise ValueError("Learning-rate preflight max_length_benchmark differs from Stage 1b")
    if preflight.get("truncation_side") != "right":
        raise ValueError("Benchmark preflight must declare right truncation for target-only inputs")
    if preflight.get("model_ids") != {b: eu.BACKBONES[b].hf_model_id for b in eu.BACKBONES}:
        raise ValueError("Benchmark preflight model IDs differ from experiment_utils.BACKBONES")
    model_revisions = preflight.get("model_revisions")
    if not isinstance(model_revisions, Mapping) or set(model_revisions) != set(eu.BACKBONES):
        raise ValueError("Benchmark preflight must record a pinned model revision for both backbones")
    for backbone in eu.BACKBONES:
        if model_revisions[backbone] != tokenizer_revisions[backbone]:
            raise ValueError(f"Model/tokenizer revision mismatch for {backbone}; rerun Stage 1b and benchmark LR preflight using the same pinned commit")
        if not re.fullmatch(r"[0-9a-fA-F]{40}", str(model_revisions[backbone])):
            raise ValueError(f"Model revision for {backbone} must be a full 40-character commit SHA")

    # The preflight config is based on train+validation (matching the selected
    # LR decision); test is still fingerprinted separately in this run manifest.
    dataset_sha = eu.sha256_json({"train_sha256": train_sha, "validation_sha256": val_sha})
    split_sha = expected_split_sha
    return {
        "paths": paths,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "train_texts": train_texts,
        "val_texts": val_texts,
        "test_texts": test_texts,
        "train_sha256": train_sha,
        "validation_sha256": val_sha,
        "test_sha256": test_sha,
        "class_weights_sha256": weights_sha,
        "sequence_config_sha256": _sha(paths["sequence_config"]),
        "learning_rate_config_sha256": _sha(paths["learning_rate_config"]),
        "dataset_sha256": dataset_sha,
        "split_assignment_sha256": split_sha,
        "sequence_doc": length_doc,
        "learning_rate_doc": lr_doc,
        "class_weights_by_id": class_weights_by_id,
        "class_weights_by_label": {eu.ID2LABEL[i]: float(value) for i, value in class_weights_by_id.items()},
        "max_length_benchmark": int(length_doc["max_length_benchmark"]),
        "learning_rate_benchmark": float(lr_doc["learning_rate_benchmark"][BACKBONE]),
        "model_revision": str(model_revisions[BACKBONE]),
        "tokenizer_revision": str(tokenizer_revisions[BACKBONE]),
    }


def _resolved_config(audit: Mapping[str, Any], seed: int, mixed_precision: bool) -> dict[str, Any]:
    """Build the exact Stage 4 configuration recorded and hashed by utility."""
    cfg: dict[str, Any] = {
        "schema_version": eu.CONFIG_SCHEMA_VERSION,
        "run_id": eu.RUN_ID,
        "stage": STAGE,
        "script_name": SCRIPT_NAME,
        "experiment_id": eu.experiment1_id(BACKBONE),
        "backbone": BACKBONE,
        "hf_model_id": eu.get_backbone(BACKBONE).hf_model_id,
        "model_revision": audit["model_revision"],
        "tokenizer_revision": audit["tokenizer_revision"],
        "seed": int(seed),
        "input_mode": INPUT_MODE,
        "input_order": "target_only",
        "context_order": "target_only",
        "dataset_sha256": audit["dataset_sha256"],
        "train_split_sha256": audit["train_sha256"],
        "validation_split_sha256": audit["validation_sha256"],
        "test_split_sha256": audit["test_sha256"],
        "class_weights_sha256": audit["class_weights_sha256"],
        "sequence_length_config_sha256": audit["sequence_config_sha256"],
        "learning_rate_config_sha256": audit["learning_rate_config_sha256"],
        "split_assignment_sha256": audit["split_assignment_sha256"],
        "split_seed": SPLIT_SEED,
        "eval_seeds_exp1": list(eu.EVAL_SEEDS_EXP1),
        "cv_seeds": list(eu.CV_SEEDS),
        "max_length_benchmark": int(audit["max_length_benchmark"]),
        "truncation_side": "right",
        "optimizer": OPTIMIZER_NAME,
        "weight_decay": WEIGHT_DECAY,
        "warmup_ratio": WARMUP_RATIO,
        "lr_scheduler": LR_SCHEDULER,
        "max_epochs": NUM_EPOCHS,
        "early_stopping_patience": None,
        "loss": LOSS_NAME,
        "class_weighting_benchmark": CLASS_WEIGHTING_NAME,
        "checkpoint_metric": CHECKPOINT_METRIC,
        "lr_search_grid": [float(x) for x in eu.LR_SEARCH_GRID],
        "lr_tuning_seed": int(audit["learning_rate_doc"]["lr_tuning_seed"]),
        "mixed_precision": bool(mixed_precision),
        "per_device_batch_size": TRAIN_BATCH_SIZE,
        "eval_batch_size": EVAL_BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "effective_batch_size": TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS,
        "learning_rate_benchmark": float(audit["learning_rate_benchmark"]),
        "class_weights_by_label": audit["class_weights_by_label"],
    }
    eu.validate_resolved_run_config(
        cfg,
        stage=STAGE,
        backbone=BACKBONE,
        input_mode=INPUT_MODE,
        # _load_and_validate_inputs() has already called load_shared_run_config()
        # with absolute paths and verified the shared file hashes. Keep this
        # per-run validation cwd-independent rather than reloading defaults.
        require_shared_config=False,
    )
    return cfg


def _metrics_for_ids(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=[0, 1, 2], average="weighted", zero_division=0)),
        "f1_n": float(f1_score(y_true, y_pred, labels=[0], average="macro", zero_division=0)),
        "f1_al": float(f1_score(y_true, y_pred, labels=[1], average="macro", zero_division=0)),
        "f1_hs": float(f1_score(y_true, y_pred, labels=[2], average="macro", zero_division=0)),
    }


def _compute_metrics(eval_pred: Any) -> dict[str, float]:
    logits, labels = eval_pred
    y_pred = np.asarray(logits).argmax(axis=1)
    return _metrics_for_ids(np.asarray(labels, dtype=int), y_pred)


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


def _make_dataset_class(torch: Any):
    class TokenizedTextDataset(torch.utils.data.Dataset):
        def __init__(self, encodings: Mapping[str, Any], labels: np.ndarray):
            self.encodings = encodings
            self.labels = np.asarray(labels, dtype=np.int64)

        def __len__(self) -> int:
            return len(self.labels)

        def __getitem__(self, idx: int) -> dict:
            item = {key: value[idx] for key, value in self.encodings.items()}
            item["labels"] = torch.tensor(int(self.labels[idx]), dtype=torch.long)
            return item

    return TokenizedTextDataset


def _completed_artifacts_are_valid(run_dir: Path, manifest: Mapping[str, Any]) -> bool:
    """Return True only when all final outputs and their hashes are valid."""
    index_path = run_dir / "artifacts_sha256.json"
    final_paths = [run_dir / "predictions.csv", run_dir / "probabilities.npy", run_dir / "metrics.json"]
    if index_path.is_file():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Unreadable output hash index {index_path}; archive the run directory") from exc
        if index.get("run_id") != eu.RUN_ID or index.get("stage") != STAGE or index.get("manifest_fingerprint") != manifest["fingerprint"]:
            raise RuntimeError(f"Output hash index does not match the current run: {index_path}")
        hashes = index.get("sha256")
        if not isinstance(hashes, Mapping):
            raise RuntimeError(f"Output hash index has no sha256 mapping: {index_path}")
        for path in final_paths:
            expected = hashes.get(path.name)
            if not path.is_file() or not isinstance(expected, str) or _sha(path) != expected:
                raise RuntimeError(f"Missing or modified Stage 4 output {path}; do not silently reuse it")
        eu.assert_current_run_artifact(
            final_paths[0], expected_stage=STAGE, expected_script=SCRIPT_NAME,
            expected_fingerprint=str(manifest["fingerprint"]),
        )
        return True

    # A brand-new run directory contains only its manifest. If any partial
    # prediction/checkpoint/metric appears without the completion index, stop.
    unexpected = [p for p in run_dir.rglob("*") if p.is_file() and p.name != eu.RUN_MANIFEST_NAME]
    if unexpected:
        raise RuntimeError(
            f"Incomplete/unindexed outputs exist in {run_dir}: {[p.name for p in unexpected[:10]]}. "
            "Archive this directory before restarting the seed."
        )
    return False


def _write_json_atomic(path: Path, document: Mapping[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Experiment I XLM-RoBERTa-base on the fixed benchmark split.")
    parser.add_argument("--validate-only", action="store_true", help="Audit inputs/configuration and output namespace; do not download models or train.")
    parser.add_argument("--allow-cpu", action="store_true", help="Allow fine-tuning on CPU explicitly (not recommended for the five-seed run).")
    parser.add_argument("--disable-tqdm", action="store_true", help="Disable Trainer progress bars.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assert_no_legacy_stage4_outputs()
    audit = _load_and_validate_inputs()
    print(f"Pipeline run ID: {eu.RUN_ID}")
    print(f"Experiment ID: {eu.experiment1_id(BACKBONE)}")
    print(f"Model/tokenizer: {eu.get_backbone(BACKBONE).hf_model_id} @ {audit['model_revision']}")
    print(f"Seeds: {list(eu.EVAL_SEEDS_EXP1)}")
    print(f"Benchmark max length: {audit['max_length_benchmark']}")
    print(f"Benchmark learning rate: {audit['learning_rate_benchmark']:g}")
    print(f"Truncation: {INPUT_MODE} / right")
    print(f"Dataset SHA-256 (train+validation selection basis): {audit['dataset_sha256']}")
    print(f"Test split SHA-256 (manifest input): {audit['test_sha256']}")
    print("Historical flat-path outputs: none found")

    migration_audit = eu.audit_pipeline_script_names(PROJECT_ROOT, require_all=False)
    if SCRIPT_NAME not in migration_audit["present"]:
        raise FileNotFoundError(
            f"The active Stage 4 script is not at the authoritative path {SCRIPT_NAME!r}. "
            "Place this revised file in the repository's scripts/ directory."
        )

    if args.validate_only:
        # Validate against the shared contract without requiring Transformers,
        # fetching model weights, writing configs, or creating output dirs.
        cfg = _resolved_config(audit, seed=eu.EVAL_SEEDS_EXP1[0], mixed_precision=False)
        # Exercise the two-backbone fairness contract against Stage 3's
        # IndoBERTweet configuration using the same frozen dataset/settings.
        peer = dict(cfg)
        peer["backbone"] = "indobertweet"
        peer["hf_model_id"] = eu.get_backbone("indobertweet").hf_model_id
        peer["model_revision"] = audit["learning_rate_doc"]["benchmark_preflight"]["model_revisions"]["indobertweet"]
        peer["tokenizer_revision"] = audit["sequence_doc"]["tokenizer_resolved_revisions"]["indobertweet"]
        peer["learning_rate_benchmark"] = float(audit["learning_rate_doc"]["learning_rate_benchmark"]["indobertweet"])
        eu.check_config_parity(cfg, peer, input_mode="target_only")
        print("Stage 4 input/configuration validation passed. No model was loaded, no outputs were written.")
        if migration_audit["complete"]:
            print("Full pipeline filename/directory audit: passed.")
        else:
            print("Full pipeline filename/directory audit: not yet complete; training remains blocked until all registered scripts and extension directories are present.")
            print(f"Missing scripts: {migration_audit['missing']}")
            print(f"Missing directories: {migration_audit['missing_directories']}")
            print(f"Old filenames still present: {migration_audit['rename_leftovers']}")
        return

    # The protocol requires the entire repository audit to pass before any
    # expensive training starts. Independent unit tests remain runnable while
    # the other stage scripts are still being migrated.
    eu.audit_pipeline_script_names(PROJECT_ROOT, require_all=True)

    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, set_seed
    except ImportError as exc:
        raise ImportError("Stage 4 requires PyTorch and Hugging Face Transformers; use the project's training environment/Colab.") from exc

    use_cuda = bool(torch.cuda.is_available())
    if not use_cuda and not args.allow_cpu:
        raise RuntimeError("No CUDA GPU detected. Use a Colab GPU runtime or pass --allow-cpu only for a deliberately slow run.")

    # Resolve and verify the pinned revision before tokenization or training.
    revision = audit["tokenizer_revision"]
    spec = eu.get_backbone(BACKBONE)
    tokenizer = AutoTokenizer.from_pretrained(spec.hf_model_id, revision=revision, use_fast=True)
    eu.configure_tokenizer_for_input(tokenizer, INPUT_MODE)
    resolved_tok_rev = getattr(tokenizer, "_commit_hash", None) or (getattr(tokenizer, "init_kwargs", {}) or {}).get("_commit_hash")
    if not resolved_tok_rev or str(resolved_tok_rev).lower() != revision.lower():
        raise RuntimeError(f"Tokenizer resolved to {resolved_tok_rev!r}; expected pinned Stage 1b revision {revision!r}")

    max_length = int(audit["max_length_benchmark"])
    train_df = audit["train_df"]
    val_df = audit["val_df"]
    test_df = audit["test_df"]
    y_train = train_df["label"].astype(str).map(eu.LABEL2ID).to_numpy(dtype=np.int64)
    y_val = val_df["label"].astype(str).map(eu.LABEL2ID).to_numpy(dtype=np.int64)
    y_test = test_df["label"].astype(str).map(eu.LABEL2ID).to_numpy(dtype=np.int64)

    print("Tokenizing the fixed train/validation/test splits with the shared benchmark length...")
    train_enc = tokenizer(audit["train_texts"], truncation=True, max_length=max_length, padding="max_length", return_tensors="pt")
    val_enc = tokenizer(audit["val_texts"], truncation=True, max_length=max_length, padding="max_length", return_tensors="pt")
    # Test labels/texts are only used after validation-selected checkpoint training.
    test_enc = tokenizer(audit["test_texts"], truncation=True, max_length=max_length, padding="max_length", return_tensors="pt")
    Dataset = _make_dataset_class(torch)
    train_ds = Dataset(train_enc, y_train)
    val_ds = Dataset(val_enc, y_val)
    test_ds = Dataset(test_enc, y_test)

    weight_tensor = torch.tensor([audit["class_weights_by_id"][i] for i in range(len(eu.LABEL2ID))], dtype=torch.float32)
    WeightedLossTrainer = _make_weighted_trainer_class(torch, Trainer)
    run_root = eu.versioned_run_root(PROJECT_ROOT)
    stage_root = run_root / "experiment_i" / BACKBONE
    stage_root.mkdir(parents=True, exist_ok=True)

    for seed in eu.EVAL_SEEDS_EXP1:
        set_seed(int(seed))
        config = _resolved_config(audit, seed=int(seed), mixed_precision=use_cuda)
        manifest = eu.make_run_manifest(
            stage=STAGE,
            script_name=SCRIPT_NAME,
            config=config,
            input_paths=[
                audit["paths"]["train"], audit["paths"]["validation"], audit["paths"]["test"],
                audit["paths"]["weights"], audit["paths"]["sequence_config"], audit["paths"]["learning_rate_config"],
            ],
            metadata={
                "experiment_id": eu.experiment1_id(BACKBONE),
                "backbone": BACKBONE,
                "seed": int(seed),
                "prediction_row_order": "row position in data/processed/ib_test.csv",
            },
            project_root=PROJECT_ROOT,
        )
        run_dir = stage_root / f"seed_{seed}"
        eu.prepare_run_directory(run_dir, manifest, allow_resume=True)
        if _completed_artifacts_are_valid(run_dir, manifest):
            print(f"\n[seed {seed}] verified complete current-run outputs; skipping identical manifest.")
            continue

        print(f"\n=== Stage {STAGE} | {eu.experiment1_id(BACKBONE)} | seed {seed} ===")
        started = time.perf_counter()
        model = AutoModelForSequenceClassification.from_pretrained(
            spec.hf_model_id,
            revision=audit["model_revision"],
            num_labels=len(eu.LABEL2ID),
            id2label=eu.ID2LABEL,
            label2id=eu.LABEL2ID,
        )
        resolved_model_rev = getattr(getattr(model, "config", None), "_commit_hash", None)
        if resolved_model_rev and str(resolved_model_rev).lower() != audit["model_revision"].lower():
            raise RuntimeError(f"Model weights resolved to {resolved_model_rev!r}; expected pinned revision {audit['model_revision']!r}")

        with tempfile.TemporaryDirectory(prefix=f"e1-{BACKBONE}-seed{seed}-") as tmp_dir:
            try:
                arg_signature = inspect.signature(TrainingArguments.__init__).parameters
                eval_strategy_key = "eval_strategy" if "eval_strategy" in arg_signature else "evaluation_strategy"
                kwargs: dict[str, Any] = {
                    "output_dir": str(Path(tmp_dir) / "trainer_output"),
                    "num_train_epochs": NUM_EPOCHS,
                    "per_device_train_batch_size": TRAIN_BATCH_SIZE,
                    "per_device_eval_batch_size": EVAL_BATCH_SIZE,
                    "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
                    "learning_rate": float(audit["learning_rate_benchmark"]),
                    "weight_decay": WEIGHT_DECAY,
                    "warmup_ratio": WARMUP_RATIO,
                    "lr_scheduler_type": LR_SCHEDULER,
                    eval_strategy_key: "epoch",
                    "save_strategy": "epoch",
                    "save_total_limit": 1,
                    "load_best_model_at_end": True,
                    "metric_for_best_model": CHECKPOINT_METRIC,
                    "greater_is_better": True,
                    "logging_strategy": "epoch",
                    "seed": int(seed),
                    "data_seed": int(seed),
                    "report_to": "none",
                    "fp16": use_cuda,
                    "disable_tqdm": bool(args.disable_tqdm),
                }
                for optional in ("data_seed", "lr_scheduler_type", "warmup_ratio", "disable_tqdm"):
                    if optional not in arg_signature:
                        kwargs.pop(optional, None)
                train_args = TrainingArguments(**kwargs)
                trainer = WeightedLossTrainer(
                    model=model,
                    args=train_args,
                    train_dataset=train_ds,
                    eval_dataset=val_ds,
                    compute_metrics=_compute_metrics,
                    class_weights=weight_tensor,
                )
                trainer.train()
                # load_best_model_at_end=True: this is the validation-selected checkpoint.
                val_metrics_raw = trainer.evaluate(val_ds)
                test_output = trainer.predict(test_ds)
            except Exception:
                raise

        elapsed_seconds = time.perf_counter() - started
        logits = np.asarray(test_output.predictions)
        if logits.shape != (len(test_df), len(eu.LABEL2ID)):
            raise RuntimeError(f"Unexpected test logits shape {logits.shape}; expected {(len(test_df), len(eu.LABEL2ID))}")
        logits_tensor = torch.as_tensor(logits, dtype=torch.float32)
        probabilities = torch.softmax(logits_tensor, dim=1).cpu().numpy().astype(np.float32)
        y_pred = probabilities.argmax(axis=1).astype(np.int64)
        test_metrics = _metrics_for_ids(y_test, y_pred)
        val_metrics = {
            key.removeprefix("eval_"): float(value)
            for key, value in val_metrics_raw.items()
            if key.startswith("eval_") and isinstance(value, (int, float, np.integer, np.floating))
        }

        predictions = pd.DataFrame({
            "row_id": np.arange(len(test_df), dtype=np.int64),
            "y_true": pd.Series(y_test).map(eu.ID2LABEL),
            "y_pred": pd.Series(y_pred).map(eu.ID2LABEL),
        })
        pred_path = run_dir / "predictions.csv"
        probs_path = run_dir / "probabilities.npy"
        metrics_path = run_dir / "metrics.json"
        pred_tmp = run_dir / "predictions.csv.tmp"
        probs_tmp = run_dir / "probabilities.npy.tmp"
        predictions.to_csv(pred_tmp, index=False)
        with probs_tmp.open("wb") as file_obj:
            np.save(file_obj, probabilities, allow_pickle=False)
        os.replace(pred_tmp, pred_path)
        os.replace(probs_tmp, probs_path)
        metrics_doc = {
            "schema_version": eu.CONFIG_SCHEMA_VERSION,
            "run_id": eu.RUN_ID,
            "stage": STAGE,
            "script_name": SCRIPT_NAME,
            "experiment_id": eu.experiment1_id(BACKBONE),
            "backbone": BACKBONE,
            "seed": int(seed),
            "run_manifest_fingerprint": manifest["fingerprint"],
            "model_revision": audit["model_revision"],
            "tokenizer_revision": audit["tokenizer_revision"],
            "max_length_benchmark": max_length,
            "truncation_side": "right",
            "learning_rate_benchmark": audit["learning_rate_benchmark"],
            "validation_metrics": val_metrics,
            "test_metrics": test_metrics,
            "train_seconds": round(elapsed_seconds, 3),
            "train_minutes": round(elapsed_seconds / 60.0, 3),
            "n_train": int(len(train_df)),
            "n_validation": int(len(val_df)),
            "n_test": int(len(test_df)),
            "test_row_order_sha256": eu.sha256_json(np.arange(len(test_df), dtype=np.int64).tolist()),
        }
        _write_json_atomic(metrics_path, metrics_doc)
        artifact_hashes = {
            "schema_version": eu.CONFIG_SCHEMA_VERSION,
            "run_id": eu.RUN_ID,
            "stage": STAGE,
            "script_name": SCRIPT_NAME,
            "manifest_fingerprint": manifest["fingerprint"],
            "sha256": {p.name: _sha(p) for p in (pred_path, probs_path, metrics_path)},
        }
        _write_json_atomic(run_dir / "artifacts_sha256.json", artifact_hashes)
        # Verify the persisted result as a downstream consumer would.
        if not _completed_artifacts_are_valid(run_dir, manifest):
            raise RuntimeError(f"Newly written outputs failed completion verification in {run_dir}")

        print(
            f"[seed {seed}] validation macro-F1={val_metrics.get('macro_f1', float('nan')):.4f}; "
            f"test macro-F1={test_metrics['macro_f1']:.4f}; accuracy={test_metrics['accuracy']:.4f}; "
            f"elapsed={elapsed_seconds / 60:.1f} min"
        )
        del trainer, model
        if use_cuda:
            torch.cuda.empty_cache()

    print("\nStage 4 completed or verified for all frozen Experiment I seeds.")
    print(f"Outputs: {stage_root}")
    print("Next: after Stage 3 and Stage 4 both finish, run scripts/5_evaluate_e1.py locally with both backbones' predictions, probabilities, and manifests.")


if __name__ == "__main__":
    main()
