"""Stage 9/10: Experiment II-A contextual ablation for both registered backbones.

Authoritative pipeline script: ``scripts/9_train_e2_ablation.py``.

Invoke once per backbone on a GPU-enabled Colab runtime:
    python scripts/9_train_e2_ablation.py --backbone indobertweet --validate-only
    python scripts/9_train_e2_ablation.py --backbone xlmr --validate-only
    python scripts/9_train_e2_ablation.py --backbone indobertweet
    python scripts/9_train_e2_ablation.py --backbone xlmr

The selected backbone determines the logical pipeline stage and isolated output
namespace:
    Stage 9: IndoBERTweet -> results/runs/left_truncation_v2/stage_09/
    Stage 10: XLM-RoBERTa-base -> results/runs/left_truncation_v2/stage_10/

Each backbone runs all 5 context sizes (C0-C4), 5 shared folds, and 2 frozen
CV seeds (42, 123): 50 runs per backbone. Each fold's held-out partition is
predicted only after a fixed 8-epoch training run. The held-out fold is NEVER
used for checkpoint selection, early stopping, LR tuning, or training. The
selected contextual learning rate and maximum sequence length are read from
the Stage 8 shared configs; no old hard-coded length or learning rate is used.

Context text is built only through experiment_utils.build_context_text() and
the tokenizer's own separator token. Contextual tokenizers explicitly use
left truncation, so old context is truncated first and the target remains at
the right end of the sequence. No class weighting is used for the balanced
matched contextual dataset, as specified in the protocol.

Every individual run has its own source/data/configuration manifest and output
hash inventory. Fully completed runs are reusable only when the manifest and
all hashes match exactly; incomplete directories and old flat-path outputs are
rejected. Stage 11 consumes the 50 per-run manifests and the aligned
post_id/prediction/probability artifacts produced here.

This script deliberately does not select C_best. Stage 11 applies the paired
OOF bootstrap/parsimony rule after BOTH backbone grids have completed.
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

SCRIPT_NAME = "scripts/9_train_e2_ablation.py"
STAGE_BY_BACKBONE = {"indobertweet": 9, "xlmr": 10}
INPUT_MODE = "contextual"
CONDITIONS = tuple(range(5))
FOLDS = tuple(range(5))
SEEDS = tuple(eu.CV_SEEDS)
N_EPOCHS_EXPECTED = 8
WEIGHT_DECAY_EXPECTED = 0.01
WARMUP_RATIO_EXPECTED = 0.0
LR_SCHEDULER_EXPECTED = "linear"
OPTIMIZER_EXPECTED = "AdamW"
LOSS_EXPECTED = "unweighted_cross_entropy"
CLASS_WEIGHTING_EXPECTED = "none_balanced_matched_dataset"
CHECKPOINT_METRIC = "macro_f1"  # recorded for parity; no fold-based checkpoint selection
TRUNCATION_SIDE = "left"
SPLIT_SEED = 42

MATCHED_SUBSET_PATH = Path("data/processed/ctx_matched_subset.csv")
SEQUENCE_CONFIG_PATH = eu.SEQUENCE_LENGTH_CONFIG
LEARNING_RATE_CONFIG_PATH = eu.LEARNING_RATE_CONFIG
STAGE2_MANIFEST_PATH = eu.versioned_run_root(PROJECT_ROOT) / "stage_02" / eu.RUN_MANIFEST_NAME
STAGE8_SUMMARY_DIR = eu.versioned_run_root(PROJECT_ROOT) / "stage_08" / "summary"
STAGE8_COMPLETE_PATH = STAGE8_SUMMARY_DIR / "stage8_complete.json"
STAGE8_INDEX_PATH = STAGE8_SUMMARY_DIR / "artifacts_sha256.json"
STAGE8_RECEIPT_PATH = STAGE8_SUMMARY_DIR / "stage8_publish_receipt.json"
STAGE8_LATENCY_PATH = STAGE8_SUMMARY_DIR / "context_latency_measurements.json"
ITFT_CHECKPOINT_ROOT = Path("models/itft_checkpoint")
LEGACY_OUTPUT_GLOBS = (
    "results/raw/e2_C*_fold*_seed*_predictions.csv",
    "results/raw/e2_C*_fold*_seed*_probs.npy",
    "results/tables/e2_grid_metrics.csv",
    "models/E2_itft_context",
)


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _read_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read valid JSON from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object at {path}")
    return value


def _write_json_atomic(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _write_csv_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _sha(path: Path) -> str:
    return eu.sha256_file(path)


def _sha_tree(directory: Path) -> dict[str, str]:
    return {
        path.relative_to(directory).as_posix(): _sha(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.name != "itft_checkpoint_manifest.json"
    }


def _resolve_legacy_outputs() -> list[Path]:
    found: list[Path] = []
    for pattern in LEGACY_OUTPUT_GLOBS:
        found.extend(path for path in PROJECT_ROOT.glob(pattern) if path.exists())
    return list(dict.fromkeys(found))


def _assert_no_legacy_outputs() -> None:
    eu.assert_no_legacy_outputs(_resolve_legacy_outputs())


def _get_revision_map(sequence_doc: Mapping[str, Any], lr_doc: Mapping[str, Any]) -> tuple[dict[str, str], dict[str, str]]:
    context_tokenizers = sequence_doc.get("context_tokenizer_resolved_revisions")
    if not isinstance(context_tokenizers, Mapping) or set(context_tokenizers) != set(eu.BACKBONES):
        raise ValueError("Stage 8 sequence_length.json must pin context_tokenizer_resolved_revisions for both backbones")
    context_preflight = lr_doc.get("context_preflight")
    if not isinstance(context_preflight, Mapping):
        raise ValueError("Stage 8 learning_rates.json is missing context_preflight provenance")
    model_revisions = context_preflight.get("model_revisions")
    model_ids = context_preflight.get("model_ids")
    expected_ids = {key: eu.get_backbone(key).hf_model_id for key in eu.BACKBONES}
    if model_ids != expected_ids:
        raise ValueError("Stage 8 context_preflight model_ids differ from experiment_utils.BACKBONES")
    if not isinstance(model_revisions, Mapping) or set(model_revisions) != set(eu.BACKBONES):
        raise ValueError("Stage 8 context_preflight must pin model_revisions for both backbones")
    tokenizer_revisions: dict[str, str] = {}
    model_revisions_out: dict[str, str] = {}
    for backbone in eu.BACKBONES:
        tok_rev = str(context_tokenizers[backbone]).lower()
        model_rev = str(model_revisions[backbone]).lower()
        if not re.fullmatch(r"[0-9a-f]{40}", tok_rev):
            raise ValueError(f"Stage 8 tokenizer revision for {backbone} must be a full 40-character commit SHA")
        if not re.fullmatch(r"[0-9a-f]{40}", model_rev):
            raise ValueError(f"Stage 8 model revision for {backbone} must be a full 40-character commit SHA")
        if tok_rev != model_rev:
            raise ValueError(f"Stage 8 model/tokenizer revision differs for {backbone}; rerun the relevant preflight")
        tokenizer_revisions[backbone] = tok_rev
        model_revisions_out[backbone] = model_rev
    return model_revisions_out, tokenizer_revisions


def _validate_stage8_publication(
    *, matched_sha256: str, fold_audit: Mapping[str, Any], sequence_doc: Mapping[str, Any],
    learning_rate_doc: Mapping[str, Any], sequence_path: Path, learning_rate_path: Path,
) -> dict[str, Any]:
    """Verify the Stage 8 completion receipt, current configs, hashes and latency summary."""
    for path in (STAGE8_COMPLETE_PATH, STAGE8_INDEX_PATH, STAGE8_RECEIPT_PATH, STAGE8_LATENCY_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Stage 8 publication artifact is missing: {path}; complete Stage 8 first")

    marker = _read_json(STAGE8_COMPLETE_PATH)
    manifest_path = STAGE8_SUMMARY_DIR / eu.RUN_MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Stage 8 summary manifest is missing: {manifest_path}")
    manifest = _read_json(manifest_path)
    fingerprint = str(manifest.get("fingerprint", ""))
    eu.assert_current_run_artifact(
        STAGE8_COMPLETE_PATH,
        expected_stage=8,
        expected_script="scripts/8_timing_pilot.py",
        expected_fingerprint=fingerprint,
    )
    if marker.get("manifest_fingerprint") != fingerprint:
        raise ValueError("Stage 8 completion marker does not match its run manifest")
    index = _read_json(STAGE8_INDEX_PATH)
    if marker.get("artifact_index_sha256") != _sha(STAGE8_INDEX_PATH):
        raise ValueError("Stage 8 artifact index hash differs from the completion marker")
    if index.get("manifest_fingerprint") != fingerprint or index.get("stage") != 8 or index.get("run_id") != eu.RUN_ID:
        raise ValueError("Stage 8 artifact index identity is inconsistent")
    if marker.get("sequence_length_config_sha256") != _sha(sequence_path):
        raise ValueError("sequence_length.json changed after Stage 8 publication; regenerate the clean run")
    if marker.get("learning_rate_config_sha256") != _sha(learning_rate_path):
        raise ValueError("learning_rates.json changed after Stage 8 publication; regenerate the clean run")
    for name, expected in index.get("sha256", {}).items():
        path = STAGE8_SUMMARY_DIR / name
        if not path.is_file() or _sha(path) != expected:
            raise ValueError(f"Stage 8 output hash mismatch for {path}")

    receipt = _read_json(STAGE8_RECEIPT_PATH)
    if receipt.get("summary_manifest_fingerprint") != fingerprint:
        raise ValueError("Stage 8 publication receipt belongs to a different summary manifest")
    if receipt.get("sequence_length_config_sha256") != _sha(sequence_path):
        raise ValueError("Stage 8 publication receipt has stale sequence_length.json hash")
    if receipt.get("learning_rate_config_sha256") != _sha(learning_rate_path):
        raise ValueError("Stage 8 publication receipt has stale learning_rates.json hash")
    if receipt.get("latency_measurements_sha256") != _sha(STAGE8_LATENCY_PATH):
        raise ValueError("Stage 8 latency measurements do not match their publication receipt")

    preflight = learning_rate_doc.get("context_preflight")
    if not isinstance(preflight, Mapping):
        raise ValueError("learning_rates.json must contain context_preflight metadata")
    expected_context_checks = {
        "contextual_matched_csv_sha256": matched_sha256,
        "fold_assignment_sha256": fold_audit["fold_assignment_sha256"],
        "max_length_context": sequence_doc["max_length_context"],
        "context_length_source_partition": "contextual_c4",
        "truncation_side": TRUNCATION_SIDE,
        "lr_tuning_seed": learning_rate_doc["lr_tuning_seed"],
    }
    mismatches = {
        key: (preflight.get(key), expected)
        for key, expected in expected_context_checks.items()
        if preflight.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"Stage 8 contextual preflight provenance is inconsistent with current inputs: {mismatches}")
    seq_checks = {
        "context_matched_csv_sha256": matched_sha256,
        "context_fold_assignment_sha256": fold_audit["fold_assignment_sha256"],
        "context_pilot_script": "scripts/8_timing_pilot.py",
    }
    seq_mismatches = {key: (sequence_doc.get(key), val) for key, val in seq_checks.items() if sequence_doc.get(key) != val}
    if seq_mismatches:
        raise ValueError(f"sequence_length.json contextual provenance is stale or mismatched: {seq_mismatches}")

    latency = _read_json(STAGE8_LATENCY_PATH)
    if latency.get("run_id") != eu.RUN_ID or not isinstance(latency.get("median_ms_by_context"), Mapping):
        raise ValueError("Stage 8 latency measurements have an invalid schema/run ID")
    if preflight.get("latency_median_ms_by_context") != latency.get("median_ms_by_context"):
        raise ValueError("Stage 8 learning-rate config and latency artifact record different inference-latency medians")
    if preflight.get("latency_measurement_machine") != latency.get("measurement_machine"):
        raise ValueError("Stage 8 learning-rate config and latency artifact identify different measurement machines")
    median_by_context = latency["median_ms_by_context"]
    for context in CONDITIONS:
        by_backbone = median_by_context.get(f"C{context}")
        if not isinstance(by_backbone, Mapping) or set(by_backbone) != set(eu.BACKBONES):
            raise ValueError(f"Stage 8 latency results for C{context} must include both backbones")
        for backbone in eu.BACKBONES:
            value = float(by_backbone[backbone])
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"Invalid Stage 8 median latency for {backbone} C{context}: {value}")
    if not str(latency.get("measurement_machine", "")).strip():
        raise ValueError("Stage 8 latency document must identify the measurement machine")
    return {
        "manifest_fingerprint": fingerprint,
        "marker_path": STAGE8_COMPLETE_PATH,
        "manifest_path": manifest_path,
        "latency_path": STAGE8_LATENCY_PATH,
        "latency": latency,
        "preflight": dict(preflight),
        "latency_machine": str(latency["measurement_machine"]),
    }


def _validate_itft_checkpoint(backbone: str, *, audit: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the Stage 6/7 checkpoint, manifest, checkpoint hashes and provenance."""
    stage = 6 if backbone == "indobertweet" else 7
    spec = eu.get_backbone(backbone)
    run_dir = eu.versioned_run_root(PROJECT_ROOT) / f"stage_{stage:02d}" / f"itft_{backbone}"
    model_dir = _resolve(ITFT_CHECKPOINT_ROOT / backbone)
    manifest_path = run_dir / eu.RUN_MANIFEST_NAME
    marker_path = run_dir / "checkpoint_complete.json"
    index_path = run_dir / "artifacts_sha256.json"
    metrics_path = run_dir / "metrics.json"
    checkpoint_manifest_path = model_dir / "itft_checkpoint_manifest.json"
    required_paths = (manifest_path, marker_path, index_path, metrics_path, checkpoint_manifest_path, model_dir / "config.json")
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Stage {stage} ITFT checkpoint is incomplete: {missing}")

    run_manifest = _read_json(manifest_path)
    fingerprint = str(run_manifest.get("fingerprint", ""))
    eu.assert_current_run_artifact(
        marker_path,
        expected_stage=stage,
        expected_script="scripts/6_train_itft.py",
        expected_fingerprint=fingerprint,
    )
    marker = _read_json(marker_path)
    index = _read_json(index_path)
    metrics = _read_json(metrics_path)
    checkpoint_manifest = _read_json(checkpoint_manifest_path)
    if marker.get("manifest_fingerprint") != fingerprint or marker.get("backbone") != backbone or marker.get("stage") != stage:
        raise ValueError(f"Stage {stage} ITFT completion marker has the wrong identity")
    if index.get("manifest_fingerprint") != fingerprint or index.get("stage") != stage or index.get("backbone") != backbone:
        raise ValueError(f"Stage {stage} ITFT hash index has the wrong identity")
    if checkpoint_manifest.get("manifest_fingerprint") != fingerprint or checkpoint_manifest.get("backbone") != backbone:
        raise ValueError(f"Stage {stage} checkpoint manifest has the wrong identity")
    if metrics.get("manifest_fingerprint") not in (None, fingerprint) or metrics.get("backbone") != backbone:
        raise ValueError(f"Stage {stage} ITFT metrics have the wrong identity")
    if index.get("sha256", {}).get("metrics.json") != _sha(metrics_path):
        raise ValueError(f"Stage {stage} ITFT metrics hash does not match its index")
    if index.get("sha256", {}).get("checkpoint_manifest_sha256") != _sha(checkpoint_manifest_path):
        raise ValueError(f"Stage {stage} ITFT checkpoint-manifest hash does not match its index")
    if marker.get("checkpoint_manifest_sha256") != _sha(checkpoint_manifest_path):
        raise ValueError(f"Stage {stage} ITFT marker has a stale checkpoint-manifest hash")
    expected_files = checkpoint_manifest.get("checkpoint_files_sha256")
    if not isinstance(expected_files, Mapping) or dict(expected_files) != _sha_tree(model_dir):
        raise ValueError(f"Stage {stage} ITFT checkpoint file hashes do not match; archive and rerun the checkpoint")

    # The Stage 8 pilot appends contextual fields to the shared config JSONs.
    # Therefore, validate the frozen benchmark configuration values, but use
    # Stage 8's recorded pre-merge hashes for the two mutable shared config files.
    input_hashes = run_manifest.get("input_sha256", {})
    if not isinstance(input_hashes, Mapping):
        raise ValueError(f"Stage {stage} ITFT run manifest has no input hash map")
    frozen_paths = {
        "data/processed/ib_train.csv": _resolve("data/processed/ib_train.csv"),
        "data/processed/ib_val.csv": _resolve("data/processed/ib_val.csv"),
        "results/tables/ib_class_weights.csv": _resolve("results/tables/ib_class_weights.csv"),
    }
    for key, path in frozen_paths.items():
        if not path.is_file() or input_hashes.get(key) != _sha(path):
            raise ValueError(f"Stage {stage} ITFT input changed since the checkpoint was trained: {path}")
    preflight = audit["context_preflight"]
    expected_mutable_hashes = {
        "results/config/sequence_length.json": preflight.get("benchmark_sequence_config_sha256_before_context_merge"),
        "results/config/learning_rates.json": preflight.get("benchmark_learning_rate_config_sha256_before_context_merge"),
    }
    for key, expected in expected_mutable_hashes.items():
        if not isinstance(expected, str) or input_hashes.get(key) != expected:
            raise ValueError(f"Stage {stage} ITFT manifest does not match Stage 8's recorded pre-merge {key} hash")

    model_revision = audit["model_revisions"][backbone]
    tokenizer_revision = audit["tokenizer_revisions"][backbone]
    current_length_doc = audit["sequence_length_doc"]
    current_lr_doc = audit["learning_rate_doc"]
    for key, expected in {
        "hf_model_id": spec.hf_model_id,
        "model_revision": model_revision,
        "tokenizer_revision": tokenizer_revision,
    }.items():
        actual = checkpoint_manifest.get(key)
        if str(actual).lower() != str(expected).lower():
            raise ValueError(f"Stage {stage} ITFT checkpoint {key} mismatch: {actual!r} vs {expected!r}")
    expected_benchmark_lr = float(current_lr_doc["learning_rate_benchmark"][backbone])
    if int(metrics.get("max_length_benchmark", -1)) != int(current_length_doc["max_length_benchmark"]):
        raise ValueError(f"Stage {stage} ITFT checkpoint used a different max_length_benchmark")
    if not np.isclose(float(metrics.get("learning_rate_benchmark", -1)), expected_benchmark_lr):
        raise ValueError(f"Stage {stage} ITFT checkpoint used a different benchmark learning rate")
    if audit["context_preflight"].get("itft_manifest_fingerprints", {}).get(backbone) != fingerprint:
        raise ValueError(f"Stage 8 was tuned against a different Stage {stage} ITFT checkpoint for {backbone}")
    elapsed = float(metrics.get("elapsed_seconds", 0))
    if not math.isfinite(elapsed) or elapsed <= 0:
        raise ValueError(f"Stage {stage} ITFT metrics do not contain a positive elapsed_seconds")
    return {
        "stage": stage,
        "backbone": backbone,
        "model_dir": model_dir,
        "run_dir": run_dir,
        "marker_path": marker_path,
        "manifest_path": manifest_path,
        "checkpoint_manifest_path": checkpoint_manifest_path,
        "metrics_path": metrics_path,
        "manifest_fingerprint": fingerprint,
        "metrics": metrics,
        "model_revision": model_revision,
        "tokenizer_revision": tokenizer_revision,
        "elapsed_seconds": elapsed,
    }


def _validate_hyperparameters(preflight: Mapping[str, Any]) -> dict[str, Any]:
    hyper = preflight.get("hyperparameters")
    if not isinstance(hyper, Mapping):
        raise ValueError("Stage 8 context_preflight is missing its fixed hyperparameters")
    expected = {
        "optimizer": OPTIMIZER_EXPECTED,
        "weight_decay": WEIGHT_DECAY_EXPECTED,
        "warmup_ratio": WARMUP_RATIO_EXPECTED,
        "lr_scheduler": LR_SCHEDULER_EXPECTED,
        "max_epochs": N_EPOCHS_EXPECTED,
        "early_stopping_patience": None,
        "per_device_train_batch_size": 16,
        "per_device_eval_batch_size": 32,
        "gradient_accumulation_steps": 1,
        "effective_batch_size": 16,
        "loss": LOSS_EXPECTED,
        "checkpoint_metric": CHECKPOINT_METRIC,
    }
    mismatches = {
        key: (hyper.get(key), value)
        for key, value in expected.items()
        if hyper.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Stage 8 contextual tuning hyperparameters differ from the frozen Experiment II-A settings: {mismatches}")
    if not isinstance(hyper.get("mixed_precision"), bool):
        raise ValueError("Stage 8 context_preflight.hyperparameters must record mixed_precision as a boolean")
    return dict(hyper)


def _load_and_validate_inputs() -> dict[str, Any]:
    matched_path = _resolve(MATCHED_SUBSET_PATH)
    sequence_path = _resolve(SEQUENCE_CONFIG_PATH)
    learning_rate_path = _resolve(LEARNING_RATE_CONFIG_PATH)
    for path in (matched_path, sequence_path, learning_rate_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required Stage 9/10 input is missing: {path}")

    # Validate Stage 2's manifest and all declared source/output hashes.
    stage2_manifest = eu.assert_current_stage_manifest(2, PROJECT_ROOT)
    raw_matched = pd.read_csv(matched_path)
    data_audit = eu.validate_context_dataset(raw_matched, expected_rows=eu.CONTEXT_MATCHED_EXPECTED_ROWS)
    modelling = eu.load_modelling_view(raw_matched, include_fold=True)
    modelling[eu.CTX_FOLD_COLUMN] = pd.to_numeric(modelling[eu.CTX_FOLD_COLUMN], errors="raise").astype(int)
    modelling = modelling.sort_values(eu.CTX_ID_COLUMN, key=lambda col: col.astype(str)).reset_index(drop=True)
    matched_sha = _sha(matched_path)

    shared = eu.load_shared_run_config(
        sequence_length_path=sequence_path,
        learning_rate_path=learning_rate_path,
        require_context=True,
        require_context_lr=True,
    )
    sequence_doc = shared["sequence_length"]
    lr_doc = shared["learning_rates"]
    preflight = lr_doc.get("context_preflight")
    if not isinstance(preflight, Mapping):
        raise ValueError("learning_rates.json is missing Stage 8 context_preflight provenance")
    model_revisions, tokenizer_revisions = _get_revision_map(sequence_doc, lr_doc)

    # Confirm Stage 8 measured the exact contextual dataset/fold assignment now loaded.
    stage8 = _validate_stage8_publication(
        matched_sha256=matched_sha,
        fold_audit=data_audit,
        sequence_doc=sequence_doc,
        learning_rate_doc=lr_doc,
        sequence_path=sequence_path,
        learning_rate_path=learning_rate_path,
    )
    hyper = _validate_hyperparameters(preflight)
    if preflight.get("fold_assignment_sha256") != data_audit["fold_assignment_sha256"]:
        raise ValueError("Stage 8 contextual LR tuning used a different shared fold assignment")
    if sequence_doc.get("context_length_fold_assignment_sha256") != data_audit["fold_assignment_sha256"]:
        raise ValueError("Stage 8 contextual length selection used a different fold assignment")

    audit = {
        "paths": {
            "matched": matched_path,
            "sequence_config": sequence_path,
            "learning_rate_config": learning_rate_path,
            "stage2_manifest": STAGE2_MANIFEST_PATH,
            "stage8_marker": STAGE8_COMPLETE_PATH,
            "stage8_manifest": STAGE8_SUMMARY_DIR / eu.RUN_MANIFEST_NAME,
            "stage8_latency": STAGE8_LATENCY_PATH,
        },
        "matched": modelling,
        "dataset_sha256": matched_sha,
        "fold_audit": data_audit,
        "fold_assignment_sha256": data_audit["fold_assignment_sha256"],
        "post_id_sha256": data_audit["post_id_sha256"],
        "sequence_length_doc": sequence_doc,
        "learning_rate_doc": lr_doc,
        "context_preflight": dict(preflight),
        "model_revisions": model_revisions,
        "tokenizer_revisions": tokenizer_revisions,
        "max_length_benchmark": int(sequence_doc["max_length_benchmark"]),
        "max_length_context": int(sequence_doc["max_length_context"]),
        "learning_rate_benchmark": {key: float(value) for key, value in lr_doc["learning_rate_benchmark"].items()},
        "learning_rate_context": {key: float(value) for key, value in lr_doc["learning_rate_context"].items()},
        "lr_search_grid": [float(value) for value in eu.LR_SEARCH_GRID],
        "lr_tuning_seed": int(lr_doc["lr_tuning_seed"]),
        "hyperparameters": hyper,
        "mixed_precision": bool(hyper["mixed_precision"]),
        "stage2_manifest_fingerprint": stage2_manifest["fingerprint"],
        "stage8": stage8,
        "stage8_manifest_fingerprint": stage8["manifest_fingerprint"],
        "sequence_config_sha256": _sha(sequence_path),
        "learning_rate_config_sha256": _sha(learning_rate_path),
    }
    audit["itft_by_backbone"] = {
        backbone: _validate_itft_checkpoint(backbone, audit=audit)
        for backbone in eu.BACKBONES
    }

    # Check cross-backbone parity using the exact fields consumed by Stage 9/10.
    parity_configs = {
        backbone: _resolved_run_config(backbone, audit, condition=4, fold=0, seed=SEEDS[0], mixed_precision=audit["mixed_precision"])
        for backbone in eu.BACKBONES
    }
    eu.check_config_parity(parity_configs["indobertweet"], parity_configs["xlmr"], input_mode=INPUT_MODE)
    return audit


def _resolved_run_config(
    backbone: str, audit: Mapping[str, Any], *, condition: int, fold: int, seed: int, mixed_precision: bool,
) -> dict[str, Any]:
    stage = STAGE_BY_BACKBONE[backbone]
    spec = eu.get_backbone(backbone)
    hp = audit["hyperparameters"]
    config = {
        "schema_version": eu.CONFIG_SCHEMA_VERSION,
        "run_id": eu.RUN_ID,
        "stage": stage,
        "script_name": SCRIPT_NAME,
        "experiment_id": eu.experiment2_id(condition, backbone),
        "backbone": backbone,
        "hf_model_id": spec.hf_model_id,
        "model_revision": audit["model_revisions"][backbone],
        "tokenizer_revision": audit["tokenizer_revisions"][backbone],
        "input_mode": INPUT_MODE,
        "input_order": "parents_oldest_to_newest_then_target",
        "context_order": "parent_4,parent_3,parent_2,parent_1,target",
        "parent_columns_order": list(eu.CTX_MODELLING_COLUMNS[:5]),
        "separator_token_policy": "tokenizer_sep_token",
        "dataset_sha256": audit["dataset_sha256"],
        "post_id_sha256": audit["post_id_sha256"],
        "fold_assignment_sha256": audit["fold_assignment_sha256"],
        "fold_column": eu.CTX_FOLD_COLUMN,
        "n_folds": len(FOLDS),
        "cv_seeds": list(SEEDS),
        "eval_seeds_exp1": list(eu.EVAL_SEEDS_EXP1),
        "split_seed": SPLIT_SEED,
        "context_size": int(condition),
        "fold": int(fold),
        "seed": int(seed),
        "max_length_benchmark": int(audit["max_length_benchmark"]),
        "max_length_context": int(audit["max_length_context"]),
        "truncation_side": TRUNCATION_SIDE,
        "optimizer": str(hp["optimizer"]),
        "weight_decay": float(hp["weight_decay"]),
        "warmup_ratio": float(hp["warmup_ratio"]),
        "lr_scheduler": str(hp["lr_scheduler"]),
        "max_epochs": int(hp["max_epochs"]),
        "early_stopping_patience": None,
        "loss": str(hp["loss"]),
        "class_weighting_benchmark": CLASS_WEIGHTING_EXPECTED,
        "checkpoint_metric": CHECKPOINT_METRIC,
        "lr_search_grid": list(audit["lr_search_grid"]),
        "lr_tuning_seed": int(audit["lr_tuning_seed"]),
        "mixed_precision": bool(mixed_precision),
        "per_device_batch_size": int(hp["per_device_train_batch_size"]),
        "per_device_eval_batch_size": int(hp["per_device_eval_batch_size"]),
        "gradient_accumulation_steps": int(hp["gradient_accumulation_steps"]),
        "effective_batch_size": int(hp["effective_batch_size"]),
        "learning_rate_benchmark": float(audit["learning_rate_benchmark"][backbone]),
        "learning_rate_context": float(audit["learning_rate_context"][backbone]),
        "itft_manifest_fingerprint": audit["itft_by_backbone"][backbone]["manifest_fingerprint"],
        "stage8_manifest_fingerprint": audit["stage8_manifest_fingerprint"],
        "context_rule_version": eu.CONTEXT_RULE_VERSION,
    }
    eu.validate_resolved_run_config(
        config, stage=stage, backbone=backbone, input_mode=INPUT_MODE,
        context_size=condition, require_shared_config=True,
    )
    return config


def _run_dir(backbone: str, condition: int, fold: int, seed: int) -> Path:
    stage = STAGE_BY_BACKBONE[backbone]
    return (
        eu.versioned_run_root(PROJECT_ROOT)
        / f"stage_{stage:02d}"
        / f"e2_{backbone}"
        / f"C{condition}"
        / f"fold_{fold:02d}"
        / f"seed_{seed}"
    )


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    labels = [eu.LABEL2ID[label] for label in ("N", "AL", "HS")]
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "f1_n": float(f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)[0]),
        "f1_al": float(f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)[1]),
        "f1_hs": float(f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)[2]),
    }


def _verify_completed_run(
    run_dir: Path, *, manifest: Mapping[str, Any], backbone: str, condition: int, fold: int, seed: int,
    expected_post_ids: Sequence[Any], expected_y_true: np.ndarray,
) -> dict[str, Any] | None:
    marker_path = run_dir / "run_complete.json"
    expected_names = {"predictions.csv", "probabilities.npy", "metrics.json", "artifacts_sha256.json"}
    artifact_files = [p for p in run_dir.iterdir() if p.is_file() and p.name != eu.RUN_MANIFEST_NAME]
    if not marker_path.is_file():
        if artifact_files:
            raise RuntimeError(f"Incomplete artifacts exist under {run_dir}; archive this run directory instead of reusing partial outputs")
        return None
    unknown = {p.name for p in artifact_files} - expected_names - {marker_path.name}
    if unknown:
        raise RuntimeError(f"Unexpected files exist in completed run directory {run_dir}: {sorted(unknown)}")
    required_paths = [run_dir / name for name in expected_names] + [marker_path]
    missing = [str(p) for p in required_paths if not p.is_file()]
    if missing:
        raise RuntimeError(f"Run completion marker exists but outputs are missing: {missing}")

    marker = _read_json(marker_path)
    index_path = run_dir / "artifacts_sha256.json"
    index = _read_json(index_path)
    fingerprint = str(manifest["fingerprint"])
    identity = {"stage": STAGE_BY_BACKBONE[backbone], "backbone": backbone, "context_size": condition, "fold": fold, "seed": seed}
    for key, expected in identity.items():
        if marker.get(key) != expected or index.get(key) != expected:
            raise ValueError(f"Completed run identity mismatch for {key} under {run_dir}")
    if marker.get("manifest_fingerprint") != fingerprint or index.get("manifest_fingerprint") != fingerprint:
        raise ValueError(f"Completed run manifest fingerprint mismatch under {run_dir}")
    if marker.get("artifact_index_sha256") != _sha(index_path):
        raise ValueError(f"Run artifact index hash mismatch under {run_dir}")
    for name in ("predictions.csv", "probabilities.npy", "metrics.json"):
        path = run_dir / name
        if index.get("sha256", {}).get(name) != _sha(path):
            raise ValueError(f"Run output hash mismatch for {path}")
    eu.assert_current_run_artifact(
        marker_path,
        expected_stage=STAGE_BY_BACKBONE[backbone],
        expected_script=SCRIPT_NAME,
        expected_fingerprint=fingerprint,
    )

    predictions = pd.read_csv(run_dir / "predictions.csv")
    required_cols = {eu.CTX_ID_COLUMN, "y_true", "y_pred"}
    if not required_cols.issubset(predictions.columns):
        raise ValueError(f"Predictions file lacks columns {sorted(required_cols - set(predictions.columns))}")
    if predictions[eu.CTX_ID_COLUMN].astype(str).tolist() != [str(value) for value in expected_post_ids]:
        raise ValueError(f"Stored prediction post_id order differs from the current held-out fold under {run_dir}")
    mapped_truth = predictions["y_true"].map(eu.LABEL2ID)
    if mapped_truth.isna().any() or not np.array_equal(mapped_truth.to_numpy(dtype=np.int64), expected_y_true):
        raise ValueError(f"Stored y_true does not match the current held-out labels under {run_dir}")
    probs = np.load(run_dir / "probabilities.npy", allow_pickle=False)
    if probs.shape != (len(expected_post_ids), len(eu.LABEL2ID)):
        raise ValueError(f"Stored probability shape is invalid under {run_dir}: {probs.shape}")
    if not np.isfinite(probs).all() or (probs < 0).any() or not np.allclose(probs.sum(axis=1), 1.0, atol=1e-4):
        raise ValueError(f"Stored probabilities are invalid under {run_dir}")
    metrics = _read_json(run_dir / "metrics.json")
    return {"predictions": predictions, "probabilities": probs, "metrics": metrics, "manifest_fingerprint": fingerprint}


def _write_completed_run(
    *, run_dir: Path, manifest: Mapping[str, Any], backbone: str, condition: int, fold: int, seed: int,
    predictions: pd.DataFrame, probabilities: np.ndarray, metric_values: Mapping[str, Any],
) -> None:
    _write_csv_atomic(run_dir / "predictions.csv", predictions)
    probability_path = run_dir / "probabilities.npy"
    temporary = probability_path.with_suffix(".npy.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, probabilities, allow_pickle=False)
    os.replace(temporary, probability_path)

    metrics_doc = {
        "schema_version": eu.CONFIG_SCHEMA_VERSION,
        "run_id": eu.RUN_ID,
        "stage": STAGE_BY_BACKBONE[backbone],
        "script_name": SCRIPT_NAME,
        "backbone": backbone,
        "experiment_id": eu.experiment2_id(condition, backbone),
        "context_size": condition,
        "fold": fold,
        "seed": seed,
        "n_train": int(metric_values["n_train"]),
        "n_held_out": int(metric_values["n_held_out"]),
        "train_seconds": float(metric_values["train_seconds"]),
        "dataset_sha256": manifest["config"]["dataset_sha256"],
        "fold_assignment_sha256": manifest["config"]["fold_assignment_sha256"],
        "max_length_context": manifest["config"]["max_length_context"],
        "truncation_side": TRUNCATION_SIDE,
        "manifest_fingerprint": manifest["fingerprint"],
        **_metrics(np.asarray(metric_values["y_true"], dtype=np.int64), np.asarray(metric_values["y_pred"], dtype=np.int64)),
    }
    _write_json_atomic(run_dir / "metrics.json", metrics_doc)
    output_hashes = {
        name: _sha(run_dir / name)
        for name in ("predictions.csv", "probabilities.npy", "metrics.json")
    }
    _write_json_atomic(run_dir / "artifacts_sha256.json", {
        "schema_version": eu.CONFIG_SCHEMA_VERSION,
        "run_id": eu.RUN_ID,
        "stage": STAGE_BY_BACKBONE[backbone],
        "script_name": SCRIPT_NAME,
        "backbone": backbone,
        "context_size": condition,
        "fold": fold,
        "seed": seed,
        "manifest_fingerprint": manifest["fingerprint"],
        "sha256": output_hashes,
    })
    _write_json_atomic(run_dir / "run_complete.json", {
        "schema_version": eu.CONFIG_SCHEMA_VERSION,
        "run_id": eu.RUN_ID,
        "stage": STAGE_BY_BACKBONE[backbone],
        "script_name": SCRIPT_NAME,
        "backbone": backbone,
        "context_size": condition,
        "fold": fold,
        "seed": seed,
        "manifest_fingerprint": manifest["fingerprint"],
        "artifact_index_sha256": _sha(run_dir / "artifacts_sha256.json"),
        "post_id_sha256": eu.sha256_json(predictions[eu.CTX_ID_COLUMN].astype(str).tolist()),
    })
    eu.assert_current_run_artifact(
        run_dir / "run_complete.json",
        expected_stage=STAGE_BY_BACKBONE[backbone],
        expected_script=SCRIPT_NAME,
        expected_fingerprint=str(manifest["fingerprint"]),
    )


def _prepare_run(backbone: str, condition: int, fold: int, seed: int, audit: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    stage = STAGE_BY_BACKBONE[backbone]
    config = _resolved_run_config(
        backbone, audit, condition=condition, fold=fold, seed=seed,
        mixed_precision=bool(audit["mixed_precision"]),
    )
    run_dir = _run_dir(backbone, condition, fold, seed)
    input_paths = [
        audit["paths"]["matched"],
        audit["paths"]["sequence_config"],
        audit["paths"]["learning_rate_config"],
        audit["paths"]["stage2_manifest"],
        audit["paths"]["stage8_marker"],
        audit["paths"]["stage8_manifest"],
        audit["paths"]["stage8_latency"],
        audit["itft_by_backbone"][backbone]["marker_path"],
        audit["itft_by_backbone"][backbone]["checkpoint_manifest_path"],
        audit["itft_by_backbone"][backbone]["metrics_path"],
    ]
    manifest = eu.make_run_manifest(
        stage=stage,
        script_name=SCRIPT_NAME,
        config=config,
        input_paths=input_paths,
        metadata={
            "kind": "experiment_ii_a_cv_run",
            "held_out_fold_used_for_selection": False,
            "checkpoint_selection": "fixed final epoch; no held-out-fold checkpoint selection",
            "train_post_ids_sha256": audit["train_ids_sha256_by_fold"][fold],
            "held_out_post_ids_sha256": audit["heldout_ids_sha256_by_fold"][fold],
            "itft_manifest_fingerprint": audit["itft_by_backbone"][backbone]["manifest_fingerprint"],
            "stage8_manifest_fingerprint": audit["stage8_manifest_fingerprint"],
        },
        project_root=PROJECT_ROOT,
    )
    eu.prepare_run_directory(run_dir, manifest, allow_resume=True)
    return run_dir, manifest


def _load_training_dependencies():
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, set_seed
    except ImportError as exc:
        raise ImportError("Experiment II-A training requires PyTorch and Hugging Face Transformers; use the project GPU/Colab environment") from exc
    return torch, AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, set_seed


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


def _make_training_arguments(
    TrainingArguments: Any, *, output_dir: Path, audit: Mapping[str, Any], seed: int,
    use_fp16: bool, disable_tqdm: bool,
):
    signature = inspect.signature(TrainingArguments.__init__).parameters
    eval_key = "eval_strategy" if "eval_strategy" in signature else "evaluation_strategy"
    kwargs = {
        "output_dir": str(output_dir),
        "num_train_epochs": int(audit["hyperparameters"]["max_epochs"]),
        "per_device_train_batch_size": int(audit["hyperparameters"]["per_device_train_batch_size"]),
        "per_device_eval_batch_size": int(audit["hyperparameters"]["per_device_eval_batch_size"]),
        "gradient_accumulation_steps": int(audit["hyperparameters"]["gradient_accumulation_steps"]),
        "learning_rate": float(audit["learning_rate_context"][audit["active_backbone"]]),
        "weight_decay": float(audit["hyperparameters"]["weight_decay"]),
        "warmup_ratio": float(audit["hyperparameters"]["warmup_ratio"]),
        "lr_scheduler_type": str(audit["hyperparameters"]["lr_scheduler"]),
        eval_key: "no",
        "save_strategy": "no",
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


def _prediction_probabilities(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(eu.LABEL2ID) or not np.isfinite(values).all():
        raise ValueError(f"Model logits must have shape (n, 3) and finite values; got {values.shape}")
    values = values - values.max(axis=1, keepdims=True)
    probs = np.exp(values)
    probs /= probs.sum(axis=1, keepdims=True)
    return probs.astype(np.float32)


def _context_texts(df: pd.DataFrame, condition: int, sep: str) -> list[str]:
    return [eu.build_context_text(row, condition, sep=sep) for _, row in df.iterrows()]


def _encode_partition(tokenizer: Any, df: pd.DataFrame, condition: int, max_length: int, *, torch: Any) -> dict[str, Any]:
    texts = _context_texts(df, condition, eu.sep_from_tokenizer(tokenizer))
    encoded = tokenizer(
        texts,
        truncation=True,
        max_length=int(max_length),
        padding="max_length",
        return_tensors="pt",
    )
    return dict(encoded)


def _build_fold_data(matched: pd.DataFrame, condition: int, fold: int) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    train_df = matched.loc[matched[eu.CTX_FOLD_COLUMN] != fold].copy().reset_index(drop=True)
    heldout_df = matched.loc[matched[eu.CTX_FOLD_COLUMN] == fold].copy().reset_index(drop=True)
    if len(train_df) != 720 or len(heldout_df) != 180:
        raise ValueError(f"Fold {fold} must have 720 training and 180 held-out rows; got {len(train_df)} / {len(heldout_df)}")
    if train_df[eu.CTX_ID_COLUMN].astype(str).isin(heldout_df[eu.CTX_ID_COLUMN].astype(str)).any():
        raise AssertionError(f"Fold {fold} has overlapping training and held-out post IDs")
    if train_df[eu.CTX_ID_COLUMN].isna().any() or heldout_df[eu.CTX_ID_COLUMN].isna().any():
        raise ValueError("Contextual CV requires non-null post IDs")
    y_train = train_df["label"].map(eu.LABEL2ID)
    y_hold = heldout_df["label"].map(eu.LABEL2ID)
    if y_train.isna().any() or y_hold.isna().any():
        raise ValueError("Matched subset has labels outside the canonical LABEL2ID mapping")
    return train_df, heldout_df, y_train.to_numpy(dtype=np.int64), y_hold.to_numpy(dtype=np.int64)


def _train_one_run(
    *, torch: Any, AutoModel: Any, Trainer: Any, TrainingArguments: Any, set_seed: Any,
    Dataset: Any, backbone: str, condition: int, fold: int, seed: int,
    tokenizer: Any, itft: Mapping[str, Any], audit: Mapping[str, Any],
    train_df: pd.DataFrame, heldout_df: pd.DataFrame, y_train: np.ndarray, y_hold: np.ndarray,
    train_encodings: Mapping[str, Any], heldout_encodings: Mapping[str, Any],
    use_cuda: bool, disable_tqdm: bool,
) -> dict[str, Any]:
    run_dir, manifest = _prepare_run(backbone, condition, fold, seed, audit)
    post_ids = heldout_df[eu.CTX_ID_COLUMN].astype(str).tolist()
    completed = _verify_completed_run(
        run_dir, manifest=manifest, backbone=backbone, condition=condition, fold=fold, seed=seed,
        expected_post_ids=post_ids, expected_y_true=y_hold,
    )
    if completed is not None:
        print(f"[verified resume] Stage {STAGE_BY_BACKBONE[backbone]} {backbone} C{condition} fold {fold} seed {seed}")
        return completed["metrics"]

    # A manifest-only directory is safe to start. Any extra partial files are
    # rejected by _verify_completed_run rather than overwritten silently.
    set_seed(int(seed))
    start = time.perf_counter()
    model = AutoModel.from_pretrained(
        str(itft["model_dir"]),
        num_labels=len(eu.LABEL2ID),
        id2label=eu.ID2LABEL,
        label2id=eu.LABEL2ID,
    )
    if getattr(model.config, "num_labels", None) != len(eu.LABEL2ID):
        raise RuntimeError(f"ITFT checkpoint for {backbone} does not contain the expected three-class head")
    DatasetClass = Dataset
    train_ds = DatasetClass(train_encodings, y_train)
    heldout_ds = DatasetClass(heldout_encodings, y_hold)
    active_audit = dict(audit)
    active_audit["active_backbone"] = backbone
    with tempfile.TemporaryDirectory(prefix=f"e2-{backbone}-", dir=PROJECT_ROOT) as tmp_dir:
        training_args = _make_training_arguments(
            TrainingArguments,
            output_dir=Path(tmp_dir) / "trainer_output",
            audit=active_audit,
            seed=seed,
            use_fp16=use_cuda,
            disable_tqdm=disable_tqdm,
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            # Do not pass heldout_ds as eval_dataset: no fold-level selection.
        )
        trainer.train()
        prediction_output = trainer.predict(heldout_ds)
        elapsed = time.perf_counter() - start
        probabilities = _prediction_probabilities(prediction_output.predictions)
        y_pred = probabilities.argmax(axis=1).astype(np.int64)
        metric_values = {
            "n_train": len(train_ds),
            "n_held_out": len(heldout_ds),
            "train_seconds": round(float(elapsed), 3),
            "y_true": y_hold,
            "y_pred": y_pred,
        }
        predictions = pd.DataFrame({
            eu.CTX_ID_COLUMN: post_ids,
            "backbone": backbone,
            "context_size": condition,
            "fold": fold,
            "seed": seed,
            "y_true": [eu.ID2LABEL[int(value)] for value in y_hold],
            "y_pred": [eu.ID2LABEL[int(value)] for value in y_pred],
        })
        _write_completed_run(
            run_dir=run_dir,
            manifest=manifest,
            backbone=backbone,
            condition=condition,
            fold=fold,
            seed=seed,
            predictions=predictions,
            probabilities=probabilities,
            metric_values=metric_values,
        )
        del trainer, model
    # Reload the newly written run via the same checks used for resume.
    verified = _verify_completed_run(
        run_dir, manifest=manifest, backbone=backbone, condition=condition, fold=fold, seed=seed,
        expected_post_ids=post_ids, expected_y_true=y_hold,
    )
    if use_cuda:
        torch.cuda.empty_cache()
    return verified["metrics"]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one backbone's Experiment II-A C0-C4 grid (Stage 9 or Stage 10).")
    parser.add_argument("--backbone", choices=sorted(STAGE_BY_BACKBONE), required=True,
                        help="indobertweet (Stage 9) or xlmr (Stage 10)")
    parser.add_argument("--validate-only", action="store_true",
                        help="Validate data/configuration/manifests without loading tokenizers or training")
    parser.add_argument("--allow-cpu", action="store_true", help="Allow CPU training only when Stage 8 used mixed_precision=false")
    parser.add_argument("--disable-tqdm", action="store_true", help="Disable Hugging Face Trainer progress bars")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    backbone = args.backbone
    stage = STAGE_BY_BACKBONE[backbone]
    _assert_no_legacy_outputs()
    audit = _load_and_validate_inputs()
    spec = eu.get_backbone(backbone)

    print(f"Run ID: {eu.RUN_ID}")
    print(f"Logical stage: {stage}; authoritative script: {SCRIPT_NAME}")
    print(f"Backbone: {spec.display_name} ({spec.hf_model_id})")
    print(f"Model revision: {audit['model_revisions'][backbone]}")
    print(f"Tokenizer revision: {audit['tokenizer_revisions'][backbone]}")
    print(f"Contexts/folds/seeds: {list(CONDITIONS)} / {list(FOLDS)} / {list(SEEDS)} ({len(CONDITIONS) * len(FOLDS) * len(SEEDS)} runs)")
    print(f"Shared max_length_context={audit['max_length_context']}; contextual LR={audit['learning_rate_context'][backbone]:g}; truncation_side=left")
    print(f"Matched examples={len(audit['matched'])}; fold counts={audit['fold_audit']['fold_counts']}; dataset SHA-256={audit['dataset_sha256']}")
    print("Held-out fold policy: fixed 8 epochs, no validation/checkpoint selection on the held-out fold")

    migration = eu.audit_pipeline_script_names(PROJECT_ROOT, require_all=False)
    if SCRIPT_NAME not in migration["present"]:
        raise FileNotFoundError(f"Authoritative script {SCRIPT_NAME} is not present; save the revised file in scripts/.")
    if args.validate_only:
        print(f"Stage {stage} data, configuration, Stage 2/8 manifests and both ITFT checkpoints passed validation. No model weights were loaded.")
        if migration["complete"]:
            print("Pipeline filename/directory audit: passed.")
        else:
            print("Pipeline filename/directory audit is not yet complete; actual training remains blocked.")
            print(f"Missing scripts: {migration['missing']}")
            print(f"Missing directories: {migration['missing_directories']}")
            print(f"Old filenames still present: {migration['rename_leftovers']}")
        return

    # Training is deliberately blocked until every stage filename/path and
    # required extension directory matches the frozen pipeline table.
    eu.audit_pipeline_script_names(PROJECT_ROOT, require_all=True)
    torch, AutoModel, AutoTokenizer, Trainer, TrainingArguments, set_seed = _load_training_dependencies()
    use_cuda = bool(torch.cuda.is_available())
    if not use_cuda and not args.allow_cpu:
        raise RuntimeError("No CUDA GPU detected. Use the intended Colab GPU runtime or pass --allow-cpu deliberately.")
    if use_cuda != bool(audit["mixed_precision"]):
        raise RuntimeError(
            "The active precision environment differs from the Stage 8 contextual LR preflight: "
            f"Stage 8 mixed_precision={audit['mixed_precision']}, current CUDA availability={use_cuda}. "
            "Use the same precision mode or rerun Stage 8 under the intended environment."
        )

    # Recheck parity using the actual runtime's precision flag, not only the
    # preflight value, before any output directory is prepared.
    parity_cfgs = {
        key: _resolved_run_config(key, audit, condition=4, fold=0, seed=SEEDS[0], mixed_precision=use_cuda)
        for key in eu.BACKBONES
    }
    eu.check_config_parity(parity_cfgs["indobertweet"], parity_cfgs["xlmr"], input_mode=INPUT_MODE)

    spec = eu.get_backbone(backbone)
    tokenizer_revision = audit["tokenizer_revisions"][backbone]
    model_revision = audit["model_revisions"][backbone]
    tokenizer = AutoTokenizer.from_pretrained(spec.hf_model_id, revision=tokenizer_revision, use_fast=True)
    eu.configure_tokenizer_for_input(tokenizer, INPUT_MODE)
    resolved_tokenizer_revision = getattr(tokenizer, "_commit_hash", None) or (getattr(tokenizer, "init_kwargs", {}) or {}).get("_commit_hash")
    if not resolved_tokenizer_revision or str(resolved_tokenizer_revision).lower() != tokenizer_revision.lower():
        raise RuntimeError(f"Tokenizer for {backbone} resolved to {resolved_tokenizer_revision!r}, expected {tokenizer_revision!r}")
    if tokenizer.truncation_side != TRUNCATION_SIDE:
        raise RuntimeError(f"Contextual inputs require tokenizer.truncation_side={TRUNCATION_SIDE!r}")

    itft = audit["itft_by_backbone"][backbone]
    matched = audit["matched"]
    Dataset = _make_dataset_class(torch)
    total_runs = len(CONDITIONS) * len(FOLDS) * len(SEEDS)
    run_counter = 0
    grid_start = time.perf_counter()
    metric_rows: list[dict[str, Any]] = []

    for condition in CONDITIONS:
        for fold in FOLDS:
            train_df, heldout_df, y_train, y_hold = _build_fold_data(matched, condition, fold)
            train_encodings = _encode_partition(tokenizer, train_df, condition, audit["max_length_context"], torch=torch)
            heldout_encodings = _encode_partition(tokenizer, heldout_df, condition, audit["max_length_context"], torch=torch)
            for seed in SEEDS:
                run_counter += 1
                tag = f"[{backbone} C{condition} fold={fold} seed={seed} {run_counter}/{total_runs}]"
                print(f"\n=== {tag} ===")
                metrics = _train_one_run(
                    torch=torch,
                    AutoModel=AutoModel,
                    Trainer=Trainer,
                    TrainingArguments=TrainingArguments,
                    set_seed=set_seed,
                    Dataset=Dataset,
                    backbone=backbone,
                    condition=condition,
                    fold=fold,
                    seed=seed,
                    tokenizer=tokenizer,
                    itft=itft,
                    audit=audit,
                    train_df=train_df,
                    heldout_df=heldout_df,
                    y_train=y_train,
                    y_hold=y_hold,
                    train_encodings=train_encodings,
                    heldout_encodings=heldout_encodings,
                    use_cuda=use_cuda,
                    disable_tqdm=args.disable_tqdm,
                )
                metric_rows.append({
                    "run_id": eu.RUN_ID,
                    "stage": stage,
                    "backbone": backbone,
                    "experiment_id": eu.experiment2_id(condition, backbone),
                    "context_size": condition,
                    "fold": fold,
                    "seed": seed,
                    "n_train": metrics["n_train"],
                    "n_held_out": metrics["n_held_out"],
                    "accuracy": metrics["accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "weighted_f1": metrics["weighted_f1"],
                    "f1_n": metrics["f1_n"],
                    "f1_al": metrics["f1_al"],
                    "f1_hs": metrics["f1_hs"],
                    "train_seconds": metrics["train_seconds"],
                    "manifest_fingerprint": metrics["manifest_fingerprint"],
                })
            del train_encodings, heldout_encodings
            if use_cuda:
                torch.cuda.empty_cache()

    if len(metric_rows) != total_runs:
        raise RuntimeError(f"Expected {total_runs} run metric rows, got {len(metric_rows)}")
    metric_table = pd.DataFrame(metric_rows).sort_values(["context_size", "fold", "seed"]).reset_index(drop=True)
    summary_dir = eu.versioned_run_root(PROJECT_ROOT) / f"stage_{stage:02d}" / f"e2_{backbone}"
    summary_path = summary_dir / "e2_grid_metrics.csv"
    summary_meta_path = summary_dir / "e2_grid_summary.json"
    # These aggregate tables are deterministic summaries of the 50 individually
    # manifested runs. They are never used to resume or establish run validity;
    # Stage 11 audits each run manifest and its raw predictions/probabilities.
    expected_fingerprints = metric_table["manifest_fingerprint"].astype(str).tolist()
    summary_base = {
        "schema_version": eu.CONFIG_SCHEMA_VERSION,
        "run_id": eu.RUN_ID,
        "stage": stage,
        "script_name": SCRIPT_NAME,
        "backbone": backbone,
        "complete": True,
        "n_runs": len(metric_table),
        "expected_runs": total_runs,
        "contexts": list(CONDITIONS),
        "folds": list(FOLDS),
        "seeds": list(SEEDS),
        "dataset_sha256": audit["dataset_sha256"],
        "fold_assignment_sha256": audit["fold_assignment_sha256"],
        "max_length_context": audit["max_length_context"],
        "truncation_side": TRUNCATION_SIDE,
        "stage8_manifest_fingerprint": audit["stage8_manifest_fingerprint"],
        "itft_manifest_fingerprint": itft["manifest_fingerprint"],
        "run_manifest_fingerprints": expected_fingerprints,
    }
    if summary_path.exists() != summary_meta_path.exists():
        raise RuntimeError(
            f"Only one aggregate file exists under {summary_dir}; archive both and rerun the verified grid summary."
        )
    if summary_path.exists() and summary_meta_path.exists():
        # An exact completed rerun is idempotent only if both the existing table
        # and metadata still represent the 50 current manifest-verified runs.
        existing_summary = _read_json(summary_meta_path)
        for key, expected in summary_base.items():
            if existing_summary.get(key) != expected:
                raise RuntimeError(f"Existing grid summary is stale for {key}; archive aggregate files before proceeding.")
        if existing_summary.get("grid_metrics_sha256") != _sha(summary_path):
            raise RuntimeError("Existing e2_grid_metrics.csv does not match its summary hash; archive aggregate files.")
        existing_table = pd.read_csv(summary_path)
        try:
            pd.testing.assert_frame_equal(existing_table, metric_table, check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-12)
        except AssertionError as exc:
            raise RuntimeError("Existing e2_grid_metrics.csv differs from the currently verified per-run metrics; archive aggregate files.") from exc
        print(f"Verified existing aggregate summary for {len(metric_table)} current runs; no files overwritten.")
    else:
        _write_csv_atomic(summary_path, metric_table)
        summary = {
            **summary_base,
            "elapsed_grid_seconds_this_invocation": round(time.perf_counter() - grid_start, 3),
            "grid_metrics_sha256": _sha(summary_path),
        }
        _write_json_atomic(summary_meta_path, summary)
    print(f"\nStage {stage} finished: {total_runs}/{total_runs} run metrics available in {summary_path}.")
    print(f"Manifested OOF predictions/probabilities are under {summary_dir}.")
    print("No C_best or backbone decision was made; proceed to Stage 11 only after BOTH backbone grids pass their audit.")


if __name__ == "__main__":
    main()
