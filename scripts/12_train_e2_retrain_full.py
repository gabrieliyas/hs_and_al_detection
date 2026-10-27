"""Stage 12: retrain C0-C4 on the full matched subset for the Stage 11 winner.

Authoritative pipeline script: ``scripts/12_train_e2_retrain_full.py``.

This script can be authored and unit-tested before the upstream experiments
run, but execution requires current, verified outputs from Stages 2, 6/7, 8,
and 11. It does NOT reselect the backbone or C_best. It loads the Stage 11
backbone decision, retrains five models for the selected backbone only, and
uses the exact C_best chosen by the Stage 11 paired-OOF decision. The five
models use the complete 900-example matched subset; no shallow-pool example is
used here. C0-C4 are all retrained so Stage 13 can perform its availability-
aware descriptive evaluation without reusing fold-specific CV models.

Run in a GPU-enabled Colab runtime after the full script audit has passed:
    python scripts/12_train_e2_retrain_full.py --validate-only
    python scripts/12_train_e2_retrain_full.py

Outputs (selected backbone only):
    models/e2_full_retrained/<backbone>/C0/ ... C4/
    results/runs/left_truncation_v2/stage_12/e2_<backbone>/C0/ ... C4/
    results/runs/left_truncation_v2/stage_12/e2_<backbone>/e2_full_retrain_metrics.csv

Historical output folders from the previous single-backbone pipeline are never
accepted as a completion or resume signal. Every model checkpoint and run
artifact is verified against a current run manifest and SHA-256 inventory.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


def _project_root() -> Path:
    here = Path(__file__).resolve().parent
    return here.parent if here.name.casefold() == "scripts" else here


PROJECT_ROOT = _project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import experiment_utils as eu  # noqa: E402

STAGE = 12
SCRIPT_NAME = "scripts/12_train_e2_retrain_full.py"
CONDITIONS = tuple(range(5))
RETRAIN_SEED = 42
NUM_EPOCHS = 8
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
GRADIENT_ACCUMULATION_STEPS = 1
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.0
LR_SCHEDULER = "linear"
OPTIMIZER_NAME = "AdamW"
LOSS_NAME = "unweighted_cross_entropy"
INPUT_MODE = "contextual"
TRUNCATION_SIDE = "left"
EXPECTED_ROWS = eu.CONTEXT_MATCHED_EXPECTED_ROWS

MATCHED_SUBSET_PATH = Path("data/processed/ctx_matched_subset.csv")
SEQUENCE_CONFIG_PATH = eu.SEQUENCE_LENGTH_CONFIG
LEARNING_RATE_CONFIG_PATH = eu.LEARNING_RATE_CONFIG
DECISION_PATH = eu.BACKBONE_DECISION_PATH
STAGE8_RELATIVE_DIR = Path(*eu.RUN_ROOT_PARTS) / "stage_08" / "summary"
STAGE11_RELATIVE_DIR = Path(*eu.RUN_ROOT_PARTS) / "stage_11" / "decision"
LEGACY_OUTPUT_PATTERNS = (
    "models/E2_full_retrained",
    "models/E2_itft_context",
    "results/tables/e2_best_context.csv",
    "results/tables/e2_full_retrain_metrics.csv",
    "results/tables/e2_grid_metrics.csv",
    "results/raw/e2_C*_fold*_seed*_predictions.csv",
    "results/raw/e2_C*_fold*_seed*_probs.npy",
)


def _resolve(path: str | Path, root: Path = PROJECT_ROOT) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def _read_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read valid JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object at {path}")
    return value


def _write_json_atomic(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _write_csv_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _sha(path: Path) -> str:
    return eu.sha256_file(path)


def _sha_tree(directory: Path, *, excluded_names: Sequence[str] = ("retrain_checkpoint_manifest.json",)) -> dict[str, str]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {directory}")
    files = {
        path.relative_to(directory).as_posix(): _sha(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.name not in set(excluded_names)
    }
    if not files:
        raise RuntimeError(f"No model/tokenizer files found under {directory}")
    return files


def _legacy_paths(root: Path = PROJECT_ROOT) -> list[Path]:
    found: list[Path] = []
    for pattern in LEGACY_OUTPUT_PATTERNS:
        found.extend(path for path in root.glob(pattern) if path.exists())
    return list(dict.fromkeys(path.resolve() for path in found))


def _assert_no_legacy_outputs(root: Path = PROJECT_ROOT) -> None:
    eu.assert_no_legacy_outputs(_legacy_paths(root))


def _expected_run_dir(backbone: str, context: int, root: Path = PROJECT_ROOT) -> Path:
    return eu.versioned_run_root(root) / "stage_12" / f"e2_{backbone}" / f"C{context}"


def _model_dir(backbone: str, context: int, root: Path = PROJECT_ROOT) -> Path:
    return root / "models" / "e2_full_retrained" / backbone / f"C{context}"


def _verify_hash_index(directory: Path, index_path: Path, *, expected_identity: Mapping[str, Any], files: Sequence[str]) -> dict[str, Any]:
    index = _read_json(index_path)
    for key, expected in expected_identity.items():
        if str(index.get(key)) != str(expected):
            raise ValueError(f"Artifact index {index_path} has incorrect {key}: {index.get(key)!r} != {expected!r}")
    hashes = index.get("sha256")
    if not isinstance(hashes, Mapping):
        raise ValueError(f"Artifact index {index_path} has no sha256 mapping")
    for relative in files:
        path = directory / relative
        if not path.is_file():
            raise FileNotFoundError(f"Indexed artifact is missing: {path}")
        if hashes.get(relative) != _sha(path):
            raise ValueError(f"Artifact hash mismatch for {path}; archive and regenerate this run")
    return index


def _verify_stage11_decision(root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Verify the Stage 11 decision's run manifest, copy, marker and full index."""
    root_decision = _resolve(DECISION_PATH, root)
    stage_dir = (root / STAGE11_RELATIVE_DIR).resolve()
    manifest_path = stage_dir / eu.RUN_MANIFEST_NAME
    marker_path = stage_dir / "stage11_complete.json"
    index_path = stage_dir / "artifacts_sha256.json"
    copy_path = stage_dir / "backbone_decision.json"
    required = [root_decision, manifest_path, marker_path, index_path, copy_path]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Stage 11 decision bundle is incomplete. Complete scripts/11_select_backbone.py first. Missing: "
            + ", ".join(missing)
        )

    manifest = _read_json(manifest_path)
    fingerprint = str(manifest.get("fingerprint", ""))
    eu.assert_current_run_artifact(
        marker_path,
        expected_stage=11,
        expected_script="scripts/11_select_backbone.py",
        expected_fingerprint=fingerprint,
    )
    marker = _read_json(marker_path)
    if marker.get("manifest_fingerprint") != fingerprint:
        raise ValueError("Stage 11 marker does not match its run manifest")
    if marker.get("artifact_index_sha256") != _sha(index_path):
        raise ValueError("Stage 11 marker has a stale artifact-index hash")
    if marker.get("root_decision_sha256") != _sha(root_decision):
        raise ValueError("results/backbone_decision.json does not match the Stage 11 completion marker")
    if _sha(copy_path) != _sha(root_decision):
        raise ValueError("Stage 11 root decision and versioned decision copy differ")

    index = _read_json(index_path)
    expected_identity = {
        "schema_version": eu.CONFIG_SCHEMA_VERSION,
        "run_id": eu.RUN_ID,
        "stage": 11,
        "script_name": "scripts/11_select_backbone.py",
        "manifest_fingerprint": fingerprint,
    }
    for key, expected in expected_identity.items():
        if str(index.get(key)) != str(expected):
            raise ValueError(f"Stage 11 artifact index identity mismatch for {key}")
    index_hashes = index.get("sha256")
    if not isinstance(index_hashes, Mapping):
        raise ValueError("Stage 11 artifact index has no sha256 map")
    for filename, digest in index_hashes.items():
        path = stage_dir / filename
        if not path.is_file() or _sha(path) != digest:
            raise ValueError(f"Stage 11 indexed artifact is missing or stale: {path}")

    decision = eu.load_backbone_decision_document(root_decision)
    eu.validate_backbone_decision_document(decision)
    if decision.get("stage11_manifest_fingerprint") != fingerprint:
        raise ValueError("Stage 11 decision points to a different source manifest")
    expected_stage_dir = stage_dir.relative_to(root).as_posix()
    if decision.get("stage11_output_dir") != expected_stage_dir:
        raise ValueError("Stage 11 decision has an unexpected versioned output directory")
    if decision.get("decision") not in eu.BACKBONES:
        raise ValueError("Stage 11 decision contains an unregistered selected backbone")
    if marker.get("selected_backbone") != decision.get("decision"):
        raise ValueError("Stage 11 completion marker and decision JSON select different backbones")
    if marker.get("c_best_by_backbone") != decision.get("c_best_by_backbone"):
        raise ValueError("Stage 11 completion marker and decision JSON disagree on C_best")
    for backbone in eu.BACKBONES:
        context_path = stage_dir / f"context_selection_{backbone}.json"
        grid_path = stage_dir / f"grid_audit_{backbone}.json"
        context_doc = _read_json(context_path)
        grid_doc = _read_json(grid_path)
        if context_doc != decision["context_decisions"][backbone]:
            raise ValueError(f"Stage 11 embedded context decision differs from {context_path}")
        if grid_doc.get("grid_fingerprint") != decision["source_grid_fingerprints"][backbone]:
            raise ValueError(f"Stage 11 source-grid fingerprint differs from {grid_path}")
        if int(grid_doc.get("n_runs", -1)) != 50 or int(grid_doc.get("expected_runs", -1)) != 50:
            raise ValueError(f"Stage 11 grid audit does not certify 50 runs for {backbone}")
    return {
        "decision": decision,
        "manifest": manifest,
        "manifest_fingerprint": fingerprint,
        "root_path": root_decision,
        "copy_path": copy_path,
        "manifest_path": manifest_path,
        "marker_path": marker_path,
        "index_path": index_path,
        "stage_dir": stage_dir,
    }


def _verify_stage8_publication(root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Verify Stage 8's contextual configs, latency publication and manifests."""
    stage_dir = (root / STAGE8_RELATIVE_DIR).resolve()
    manifest_path = stage_dir / eu.RUN_MANIFEST_NAME
    marker_path = stage_dir / "stage8_complete.json"
    index_path = stage_dir / "artifacts_sha256.json"
    receipt_path = stage_dir / "stage8_publish_receipt.json"
    latency_path = stage_dir / "context_latency_measurements.json"
    sequence_path = _resolve(SEQUENCE_CONFIG_PATH, root)
    learning_rates_path = _resolve(LEARNING_RATE_CONFIG_PATH, root)
    required = [manifest_path, marker_path, index_path, receipt_path, latency_path, sequence_path, learning_rates_path]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Stage 8 has not published its complete configuration bundle. Missing: " + ", ".join(missing))

    manifest = _read_json(manifest_path)
    fingerprint = str(manifest.get("fingerprint", ""))
    eu.assert_current_run_artifact(
        marker_path,
        expected_stage=8,
        expected_script="scripts/8_timing_pilot.py",
        expected_fingerprint=fingerprint,
    )
    marker = _read_json(marker_path)
    if marker.get("manifest_fingerprint") != fingerprint:
        raise ValueError("Stage 8 completion marker does not match its manifest")
    if marker.get("artifact_index_sha256") != _sha(index_path):
        raise ValueError("Stage 8 completion marker has a stale artifact index")
    index = _read_json(index_path)
    for key, expected in {
        "schema_version": eu.CONFIG_SCHEMA_VERSION,
        "run_id": eu.RUN_ID,
        "stage": 8,
        "manifest_fingerprint": fingerprint,
    }.items():
        if str(index.get(key)) != str(expected):
            raise ValueError(f"Stage 8 artifact index identity mismatch for {key}")
    hashes = index.get("sha256")
    if not isinstance(hashes, Mapping):
        raise ValueError("Stage 8 artifact index has no sha256 map")
    for filename, digest in hashes.items():
        path = stage_dir / filename
        if not path.is_file() or _sha(path) != digest:
            raise ValueError(f"Stage 8 published artifact is missing or stale: {path}")
    if marker.get("sequence_length_config_sha256") != _sha(sequence_path):
        raise ValueError("sequence_length.json changed after the Stage 8 publication")
    if marker.get("learning_rate_config_sha256") != _sha(learning_rates_path):
        raise ValueError("learning_rates.json changed after the Stage 8 publication")

    shared = eu.load_shared_run_config(
        sequence_length_path=sequence_path,
        learning_rate_path=learning_rates_path,
        require_context=True,
        require_context_lr=True,
    )
    latency = _read_json(latency_path)
    if latency.get("schema_version") != eu.CONFIG_SCHEMA_VERSION or latency.get("run_id") != eu.RUN_ID:
        raise ValueError("Stage 8 latency artifact uses an incompatible schema or run ID")
    if latency.get("manifest_fingerprint") != fingerprint:
        raise ValueError("Stage 8 latency artifact belongs to another summary manifest")
    if latency.get("context_sizes") != list(CONDITIONS) or latency.get("truncation_side") != TRUNCATION_SIDE:
        raise ValueError("Stage 8 latency artifact must cover C0-C4 and declare left truncation")
    if not str(latency.get("measurement_machine", "")).strip():
        raise ValueError("Stage 8 latency artifact is missing measurement_machine")
    measurements = latency.get("measurements_ms_by_context")
    medians = latency.get("median_ms_by_context")
    if not isinstance(measurements, Mapping) or not isinstance(medians, Mapping):
        raise ValueError("Stage 8 latency artifact lacks raw samples or medians")
    for context in CONDITIONS:
        key = f"C{context}"
        if key not in measurements or key not in medians:
            raise ValueError(f"Stage 8 latency artifact is missing {key}")
        for backbone in eu.BACKBONES:
            values = np.asarray(measurements[key].get(backbone, []), dtype=float)
            recorded = medians[key].get(backbone)
            if values.ndim != 1 or len(values) < 20 or not np.isfinite(values).all() or (values <= 0).any():
                raise ValueError(f"Stage 8 requires at least 20 positive latency samples for {key}/{backbone}")
            if recorded is None or not math.isclose(float(recorded), float(np.median(values)), rel_tol=1e-8, abs_tol=1e-8):
                raise ValueError(f"Stage 8 latency median does not match raw samples for {key}/{backbone}")
    receipt = _read_json(receipt_path)
    if receipt.get("summary_manifest_fingerprint") != fingerprint or receipt.get("latency_measurements_sha256") != _sha(latency_path):
        raise ValueError("Stage 8 publish receipt does not match its manifest/latency file")
    expected_model_ids = {key: eu.get_backbone(key).hf_model_id for key in eu.BACKBONES}
    if latency.get("model_ids") != expected_model_ids:
        raise ValueError("Stage 8 latency model IDs differ from experiment_utils.BACKBONES")
    expected_tokenizer_revisions = shared["sequence_length"].get("context_tokenizer_resolved_revisions")
    context_preflight = shared["learning_rates"].get("context_preflight", {})
    expected_model_revisions = context_preflight.get("model_revisions") if isinstance(context_preflight, Mapping) else None
    if not isinstance(expected_tokenizer_revisions, Mapping) or not isinstance(expected_model_revisions, Mapping):
        raise ValueError("Stage 8 shared configurations must pin both model and tokenizer revisions")
    if latency.get("tokenizer_revisions") != expected_tokenizer_revisions:
        raise ValueError("Stage 8 latency tokenizer revisions differ from the sequence-length configuration")
    if latency.get("model_revisions") != expected_model_revisions:
        raise ValueError("Stage 8 latency model revisions differ from contextual LR preflight")

    return {
        "stage_dir": stage_dir,
        "manifest_path": manifest_path,
        "manifest": manifest,
        "manifest_fingerprint": fingerprint,
        "marker_path": marker_path,
        "index_path": index_path,
        "receipt_path": receipt_path,
        "latency_path": latency_path,
        "sequence_path": sequence_path,
        "learning_rates_path": learning_rates_path,
        "sequence_config": shared["sequence_length"],
        "learning_rate_config": shared["learning_rates"],
        "context_preflight": shared["learning_rates"].get("context_preflight", {}),
    }


def _validate_itft_checkpoint(backbone: str, stage8: Mapping[str, Any], root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Verify the selected backbone's Stage 6/7 ITFT checkpoint and all hashes."""
    stage = 6 if backbone == "indobertweet" else 7
    stage_dir = eu.versioned_run_root(root) / f"stage_{stage:02d}" / f"itft_{backbone}"
    model_dir = root / "models" / "itft_checkpoint" / backbone
    manifest_path = stage_dir / eu.RUN_MANIFEST_NAME
    marker_path = stage_dir / "checkpoint_complete.json"
    index_path = stage_dir / "artifacts_sha256.json"
    metrics_path = stage_dir / "metrics.json"
    checkpoint_manifest_path = model_dir / "itft_checkpoint_manifest.json"
    required = [manifest_path, marker_path, index_path, metrics_path, checkpoint_manifest_path, model_dir / "config.json"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Selected backbone's Stage {stage} ITFT checkpoint is incomplete: {missing}")

    run_manifest = _read_json(manifest_path)
    fingerprint = str(run_manifest.get("fingerprint", ""))
    eu.assert_current_run_artifact(marker_path, expected_stage=stage, expected_script="scripts/6_train_itft.py", expected_fingerprint=fingerprint)
    marker = _read_json(marker_path)
    index = _read_json(index_path)
    metrics = _read_json(metrics_path)
    checkpoint_doc = _read_json(checkpoint_manifest_path)
    expected_identity = {"stage": stage, "backbone": backbone, "manifest_fingerprint": fingerprint}
    for label, document in (("marker", marker), ("index", index), ("metrics", metrics), ("checkpoint manifest", checkpoint_doc)):
        for key, expected in expected_identity.items():
            if str(document.get(key)) != str(expected):
                raise ValueError(f"Stage {stage} ITFT {label} mismatch for {key}")
    if index.get("run_id") != eu.RUN_ID or marker.get("run_id") != eu.RUN_ID:
        raise ValueError(f"Stage {stage} ITFT artifacts use an incompatible run ID")
    if index.get("sha256", {}).get("metrics.json") != _sha(metrics_path):
        raise ValueError(f"Stage {stage} ITFT metrics hash mismatch")
    if index.get("sha256", {}).get("checkpoint_manifest_sha256") != _sha(checkpoint_manifest_path):
        raise ValueError(f"Stage {stage} ITFT checkpoint-manifest hash mismatch")
    if marker.get("checkpoint_manifest_sha256") != _sha(checkpoint_manifest_path):
        raise ValueError(f"Stage {stage} ITFT marker checkpoint-manifest hash mismatch")

    expected_files = checkpoint_doc.get("checkpoint_files_sha256")
    if not isinstance(expected_files, Mapping) or dict(expected_files) != _sha_tree(model_dir, excluded_names=("itft_checkpoint_manifest.json",)):
        raise ValueError(f"Stage {stage} ITFT model/tokenizer checkpoint files differ from their manifest")
    spec = eu.get_backbone(backbone)
    model_revision = str(checkpoint_doc.get("model_revision", "")).lower()
    tokenizer_revision = str(checkpoint_doc.get("tokenizer_revision", "")).lower()
    if checkpoint_doc.get("hf_model_id") != spec.hf_model_id:
        raise ValueError(f"Stage {stage} ITFT model ID differs from BACKBONES")
    expected_tokenizer_revisions = stage8["sequence_config"].get("context_tokenizer_resolved_revisions")
    expected_model_revisions = stage8["context_preflight"].get("model_revisions")
    if not isinstance(expected_tokenizer_revisions, Mapping) or not isinstance(expected_model_revisions, Mapping):
        raise ValueError("Stage 8 configuration does not pin model/tokenizer revisions")
    if model_revision != str(expected_model_revisions.get(backbone, "")).lower():
        raise ValueError(f"Stage {stage} ITFT model revision differs from Stage 8 configuration")
    if tokenizer_revision != str(expected_tokenizer_revisions.get(backbone, "")).lower():
        raise ValueError(f"Stage {stage} ITFT tokenizer revision differs from Stage 8 configuration")
    if checkpoint_doc.get("config", {}).get("max_length_benchmark") != stage8["sequence_config"].get("max_length_benchmark"):
        raise ValueError(f"Stage {stage} ITFT checkpoint used a different benchmark length")
    if not np.isclose(float(metrics.get("learning_rate_benchmark", -1)), float(stage8["learning_rate_config"]["learning_rate_benchmark"][backbone])):
        raise ValueError(f"Stage {stage} ITFT checkpoint used a different benchmark learning rate")
    if int(metrics.get("max_length_benchmark", -1)) != int(stage8["sequence_config"]["max_length_benchmark"]):
        raise ValueError(f"Stage {stage} ITFT metrics use a different benchmark length")
    if not math.isfinite(float(metrics.get("elapsed_seconds", 0))) or float(metrics.get("elapsed_seconds", 0)) <= 0:
        raise ValueError(f"Stage {stage} ITFT metrics lack a positive elapsed_seconds")

    # Stage 8 records the ITFT checkpoint fingerprints it used during contextual LR tuning.
    expected_itft_fp = stage8["context_preflight"].get("itft_manifest_fingerprints", {}).get(backbone)
    if expected_itft_fp != fingerprint:
        raise ValueError(f"Stage 8 contextual LR preflight used a different Stage {stage} ITFT checkpoint")

    # Stage 8 adds contextual selections to the shared config files. Validate those benchmark inputs
    # with the pre-merge hashes recorded by Stage 8 rather than expecting the original paths to
    # retain their pre-Stage-8 hashes.
    input_hashes = run_manifest.get("input_sha256", {})
    if not isinstance(input_hashes, Mapping):
        raise ValueError(f"Stage {stage} ITFT run manifest has no input hashes")
    fixed_inputs = (
        "data/processed/ib_train.csv",
        "data/processed/ib_val.csv",
        "results/tables/ib_class_weights.csv",
    )
    for name in fixed_inputs:
        path = _resolve(name, root)
        if not path.is_file() or input_hashes.get(name) != _sha(path):
            raise ValueError(f"Stage {stage} ITFT source input changed after training: {path}")
    for input_name, stage8_key in (
        ("results/config/sequence_length.json", "benchmark_sequence_config_sha256_before_context_merge"),
        ("results/config/learning_rates.json", "benchmark_learning_rate_config_sha256_before_context_merge"),
    ):
        expected_hash = stage8["context_preflight"].get(stage8_key)
        if not expected_hash or input_hashes.get(input_name) != expected_hash:
            raise ValueError(f"Stage {stage} ITFT checkpoint does not match Stage 8's recorded pre-merge hash for {input_name}")
    return {
        "stage": stage,
        "backbone": backbone,
        "model_dir": model_dir,
        "run_dir": stage_dir,
        "run_manifest": run_manifest,
        "manifest_fingerprint": fingerprint,
        "marker_path": marker_path,
        "index_path": index_path,
        "metrics_path": metrics_path,
        "checkpoint_manifest_path": checkpoint_manifest_path,
        "checkpoint_manifest": checkpoint_doc,
        "model_revision": model_revision,
        "tokenizer_revision": tokenizer_revision,
    }


def _load_and_validate_data(root: Path = PROJECT_ROOT) -> dict[str, Any]:
    matched_path = _resolve(MATCHED_SUBSET_PATH, root)
    if not matched_path.is_file():
        raise FileNotFoundError(f"Missing Stage 2 matched subset: {matched_path}")
    stage2_manifest = eu.assert_current_stage_manifest(2, root)
    raw = pd.read_csv(matched_path)
    data_audit = eu.validate_context_dataset(raw, expected_rows=EXPECTED_ROWS)
    modelling = eu.load_modelling_view(raw, include_fold=True)
    modelling[eu.CTX_FOLD_COLUMN] = pd.to_numeric(modelling[eu.CTX_FOLD_COLUMN], errors="raise").astype(int)
    modelling = modelling.sort_values(eu.CTX_ID_COLUMN, key=lambda col: col.astype(str)).reset_index(drop=True)
    if len(modelling) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} matched rows; found {len(modelling)}")
    fold_sizes = modelling[eu.CTX_FOLD_COLUMN].value_counts().to_dict()
    if fold_sizes != {0: 180, 1: 180, 2: 180, 3: 180, 4: 180}:
        raise ValueError(f"Shared matched-subset folds must contain 180 rows each; got {fold_sizes}")
    labels = modelling["label"].map(eu.LABEL2ID)
    if labels.isna().any():
        raise ValueError("Matched subset contains non-canonical label values")
    post_ids = modelling[eu.CTX_ID_COLUMN].astype(str).tolist()
    post_id_hash = eu.sha256_json(post_ids)
    if len(set(post_ids)) != EXPECTED_ROWS:
        raise ValueError("Matched subset post IDs must be unique")
    matched_sha = _sha(matched_path)
    return {
        "path": matched_path,
        "raw": raw,
        "matched": modelling,
        "stage2_manifest": stage2_manifest,
        "data_audit": data_audit,
        "dataset_sha256": matched_sha,
        "fold_assignment_sha256": data_audit["fold_assignment_sha256"],
        "post_ids": post_ids,
        "post_id_sha256": post_id_hash,
        "labels": labels.to_numpy(dtype=np.int64),
    }


def _build_resolved_config(
    *, backbone: str,
    condition: int,
    decision_bundle: Mapping[str, Any],
    stage8: Mapping[str, Any],
    itft: Mapping[str, Any],
    data: Mapping[str, Any],
) -> dict[str, Any]:
    spec = eu.get_backbone(backbone)
    sequence = stage8["sequence_config"]
    rates = stage8["learning_rate_config"]
    decision = decision_bundle["decision"]
    config = {
        "schema_version": eu.CONFIG_SCHEMA_VERSION,
        "run_id": eu.RUN_ID,
        "stage": STAGE,
        "script_name": SCRIPT_NAME,
        "backbone": backbone,
        "hf_model_id": spec.hf_model_id,
        "experiment_id": eu.experiment2_id(condition, backbone),
        "input_mode": INPUT_MODE,
        "input_order": "parent_messages_oldest_to_newest_then_target",
        "context_size": int(condition),
        "selected_context_size_c_best": int(decision["c_best_by_backbone"][backbone]),
        "selected_backbone": str(decision["decision"]),
        "stage11_manifest_fingerprint": decision_bundle["manifest_fingerprint"],
        "stage11_decision_sha256": _sha(decision_bundle["root_path"]),
        "stage8_manifest_fingerprint": stage8["manifest_fingerprint"],
        "itft_manifest_fingerprint": itft["manifest_fingerprint"],
        "dataset_sha256": data["dataset_sha256"],
        "fold_assignment_sha256": data["fold_assignment_sha256"],
        "post_id_sha256": data["post_id_sha256"],
        "max_length_benchmark": int(sequence["max_length_benchmark"]),
        "max_length_context": int(sequence["max_length_context"]),
        "truncation_side": TRUNCATION_SIDE,
        "target_only_truncation_side": sequence["target_only_truncation_side"],
        "model_revision": itft["model_revision"],
        "tokenizer_revision": itft["tokenizer_revision"],
        "learning_rate_benchmark": float(rates["learning_rate_benchmark"][backbone]),
        "learning_rate_context": float(rates["learning_rate_context"][backbone]),
        "seed": RETRAIN_SEED,
        "num_train_epochs": NUM_EPOCHS,
        "per_device_train_batch_size": TRAIN_BATCH_SIZE,
        "per_device_eval_batch_size": EVAL_BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "effective_batch_size": TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS,
        "optimizer": OPTIMIZER_NAME,
        "weight_decay": WEIGHT_DECAY,
        "warmup_ratio": WARMUP_RATIO,
        "lr_scheduler": LR_SCHEDULER,
        "loss": LOSS_NAME,
        "class_weighting": "none_balanced_matched_dataset",
        "checkpoint_selection": "fixed_8_epochs_full_matched_subset_no_heldout_fold",
    }
    eu.validate_resolved_run_config(
        config,
        stage=STAGE,
        backbone=backbone,
        input_mode=INPUT_MODE,
        context_size=condition,
        require_shared_config=True,
    )
    return config


def _manifest_input_paths(
    *, data: Mapping[str, Any], stage8: Mapping[str, Any], itft: Mapping[str, Any], decision_bundle: Mapping[str, Any], root: Path,
) -> list[Path]:
    paths: list[Path] = [
        data["path"],
        root / Path(*eu.RUN_ROOT_PARTS) / "stage_02" / eu.RUN_MANIFEST_NAME,
        stage8["sequence_path"],
        stage8["learning_rates_path"],
        stage8["manifest_path"] if "manifest_path" in stage8 else stage8["stage_dir"] / eu.RUN_MANIFEST_NAME,
        stage8["marker_path"], stage8["index_path"], stage8["receipt_path"], stage8["latency_path"],
        itft["run_dir"] / eu.RUN_MANIFEST_NAME, itft["marker_path"], itft["index_path"], itft["metrics_path"], itft["checkpoint_manifest_path"],
        decision_bundle["root_path"], decision_bundle["copy_path"], decision_bundle["manifest_path"], decision_bundle["marker_path"], decision_bundle["index_path"],
    ]
    # Include every file represented by the Stage 11 output inventory; this binds the retrain manifest
    # to the decision plus per-backbone context selections and both audited grid fingerprints.
    decision_index = _read_json(decision_bundle["index_path"])
    for filename in decision_index.get("sha256", {}):
        paths.append(decision_bundle["stage_dir"] / filename)
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = Path(path).resolve()
        if resolved not in seen:
            if not resolved.is_file():
                raise FileNotFoundError(f"Stage 12 manifest input is missing: {resolved}")
            unique.append(resolved)
            seen.add(resolved)
    return unique


def _prepare_root_manifest(
    *, backbone: str, decision_bundle: Mapping[str, Any], stage8: Mapping[str, Any], itft: Mapping[str, Any], data: Mapping[str, Any], root: Path,
) -> tuple[Path, dict[str, Any]]:
    stage_root = eu.versioned_run_root(root) / "stage_12" / f"e2_{backbone}"
    config = {
        "schema_version": eu.CONFIG_SCHEMA_VERSION,
        "run_id": eu.RUN_ID,
        "stage": STAGE,
        "script_name": SCRIPT_NAME,
        "kind": "stage12_retraining_summary",
        "backbone": backbone,
        "hf_model_id": eu.get_backbone(backbone).hf_model_id,
        "selected_context_size_c_best": int(decision_bundle["decision"]["c_best_by_backbone"][backbone]),
        "stage11_manifest_fingerprint": decision_bundle["manifest_fingerprint"],
        "stage11_decision_sha256": _sha(decision_bundle["root_path"]),
        "stage8_manifest_fingerprint": stage8["manifest_fingerprint"],
        "itft_manifest_fingerprint": itft["manifest_fingerprint"],
        "dataset_sha256": data["dataset_sha256"],
        "fold_assignment_sha256": data["fold_assignment_sha256"],
        "max_length_context": int(stage8["sequence_config"]["max_length_context"]),
        "truncation_side": TRUNCATION_SIDE,
    }
    manifest = eu.make_run_manifest(
        stage=STAGE,
        script_name=SCRIPT_NAME,
        config=config,
        input_paths=_manifest_input_paths(data=data, stage8=stage8, itft=itft, decision_bundle=decision_bundle, root=root),
        metadata={
            "kind": "stage12_summary_and_five_context_retrains",
            "contexts": list(CONDITIONS),
            "selected_backbone": backbone,
            "selected_c_best": config["selected_context_size_c_best"],
            "checkpoint_selection": "fixed eight epochs on all matched examples; no validation selection",
        },
        project_root=root,
    )
    # Validate an existing output root during --validate-only without creating a new one.
    if stage_root.exists():
        eu.prepare_run_directory(stage_root, manifest, allow_resume=True)
    return stage_root, manifest


def _condition_manifest(
    *, backbone: str, condition: int, stage_root: Path, root_manifest: Mapping[str, Any], decision_bundle: Mapping[str, Any], stage8: Mapping[str, Any], itft: Mapping[str, Any], data: Mapping[str, Any], root: Path,
) -> tuple[Path, Path, dict[str, Any]]:
    config = _build_resolved_config(
        backbone=backbone, condition=condition, decision_bundle=decision_bundle,
        stage8=stage8, itft=itft, data=data,
    )
    inputs = _manifest_input_paths(data=data, stage8=stage8, itft=itft, decision_bundle=decision_bundle, root=root)
    inputs.append(stage_root / eu.RUN_MANIFEST_NAME)
    manifest = eu.make_run_manifest(
        stage=STAGE,
        script_name=SCRIPT_NAME,
        config=config,
        input_paths=inputs,
        metadata={
            "kind": "stage12_full_matched_retraining",
            "model_directory": str(_model_dir(backbone, condition, root).relative_to(root)),
            "selected_backbone": backbone,
            "selected_c_best": int(decision_bundle["decision"]["c_best_by_backbone"][backbone]),
            "training_examples": EXPECTED_ROWS,
            "training_post_ids_sha256": data["post_id_sha256"],
            "checkpoint_selection": "none_fixed_epochs_no_heldout_partition",
        },
        project_root=root,
    )
    run_dir = _expected_run_dir(backbone, condition, root)
    eu.prepare_run_directory(run_dir, manifest, allow_resume=True)
    return run_dir, _model_dir(backbone, condition, root), manifest


def _verify_condition_complete(
    *, run_dir: Path, model_dir: Path, manifest: Mapping[str, Any], backbone: str, condition: int, root: Path = PROJECT_ROOT,
) -> dict[str, Any] | None:
    manifest_path = run_dir / eu.RUN_MANIFEST_NAME
    marker_path = run_dir / "run_complete.json"
    index_path = run_dir / "artifacts_sha256.json"
    metrics_path = run_dir / "metrics.json"
    checkpoint_manifest_path = model_dir / "retrain_checkpoint_manifest.json"
    if not marker_path.exists():
        if any(path.is_file() and path.name != eu.RUN_MANIFEST_NAME for path in run_dir.iterdir()):
            raise RuntimeError(f"Stage 12 C{condition} contains partial/unmanifested artifacts; archive it before retrying: {run_dir}")
        if model_dir.exists():
            raise RuntimeError(f"Stage 12 model directory exists without a verified completion marker: {model_dir}; archive it")
        return None
    required = (manifest_path, index_path, metrics_path, checkpoint_manifest_path, model_dir / "config.json")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Stage 12 C{condition} is marked complete but artifacts are missing: {missing}")
    fingerprint = str(manifest["fingerprint"])
    marker = _read_json(marker_path)
    eu.assert_current_run_artifact(marker_path, expected_stage=STAGE, expected_script=SCRIPT_NAME, expected_fingerprint=fingerprint)
    for key, expected in {
        "schema_version": eu.CONFIG_SCHEMA_VERSION,
        "run_id": eu.RUN_ID,
        "stage": STAGE,
        "script_name": SCRIPT_NAME,
        "backbone": backbone,
        "context_size": condition,
        "manifest_fingerprint": fingerprint,
    }.items():
        if str(marker.get(key)) != str(expected):
            raise ValueError(f"Stage 12 C{condition} completion marker mismatch for {key}")
    index = _verify_hash_index(
        run_dir,
        index_path,
        expected_identity={
            "schema_version": eu.CONFIG_SCHEMA_VERSION,
            "run_id": eu.RUN_ID,
            "stage": STAGE,
            "script_name": SCRIPT_NAME,
            "backbone": backbone,
            "context_size": condition,
            "manifest_fingerprint": fingerprint,
        },
        files=("metrics.json",),
    )
    checkpoint_doc = _read_json(checkpoint_manifest_path)
    if checkpoint_doc.get("manifest_fingerprint") != fingerprint or checkpoint_doc.get("backbone") != backbone or int(checkpoint_doc.get("context_size", -1)) != condition:
        raise ValueError(f"Stage 12 checkpoint manifest identity mismatch for {model_dir}")
    if checkpoint_doc.get("stage11_manifest_fingerprint") != manifest["config"].get("stage11_manifest_fingerprint"):
        raise ValueError("Stage 12 checkpoint was trained against a different Stage 11 decision")
    expected_files = checkpoint_doc.get("checkpoint_files_sha256")
    if not isinstance(expected_files, Mapping) or dict(expected_files) != _sha_tree(model_dir):
        raise ValueError(f"Stage 12 checkpoint files have changed: {model_dir}")
    checkpoint_sha = _sha(checkpoint_manifest_path)
    if index.get("sha256", {}).get("checkpoint_manifest_sha256") != checkpoint_sha:
        raise ValueError("Stage 12 artifact index checkpoint-manifest hash mismatch")
    if marker.get("checkpoint_manifest_sha256") != checkpoint_sha:
        raise ValueError("Stage 12 completion marker checkpoint-manifest hash mismatch")
    if marker.get("artifact_index_sha256") != _sha(index_path):
        raise ValueError("Stage 12 completion marker has a stale artifact-index hash")
    if marker.get("checkpoint_directory") != model_dir.relative_to(root).as_posix():
        raise ValueError("Stage 12 completion marker points to an unexpected model directory")
    metrics = _read_json(metrics_path)
    if metrics.get("backbone") != backbone or int(metrics.get("context_size", -1)) != condition or metrics.get("run_id") != eu.RUN_ID:
        raise ValueError("Stage 12 metrics identity mismatch")
    return metrics


def _write_condition_complete(
    *, run_dir: Path, model_dir: Path, manifest: Mapping[str, Any], config: Mapping[str, Any], metrics: Mapping[str, Any], root: Path,
) -> None:
    fingerprint = str(manifest["fingerprint"])
    checkpoint_manifest_path = model_dir / "retrain_checkpoint_manifest.json"
    checkpoint_sha = _sha(checkpoint_manifest_path)
    _write_json_atomic(run_dir / "metrics.json", dict(metrics))
    _write_json_atomic(run_dir / "artifacts_sha256.json", {
        "schema_version": eu.CONFIG_SCHEMA_VERSION,
        "run_id": eu.RUN_ID,
        "stage": STAGE,
        "script_name": SCRIPT_NAME,
        "backbone": config["backbone"],
        "context_size": config["context_size"],
        "manifest_fingerprint": fingerprint,
        "sha256": {
            "metrics.json": _sha(run_dir / "metrics.json"),
            "checkpoint_manifest_sha256": checkpoint_sha,
        },
    })
    _write_json_atomic(run_dir / "run_complete.json", {
        "schema_version": eu.CONFIG_SCHEMA_VERSION,
        "run_id": eu.RUN_ID,
        "stage": STAGE,
        "script_name": SCRIPT_NAME,
        "backbone": config["backbone"],
        "context_size": config["context_size"],
        "manifest_fingerprint": fingerprint,
        "artifact_index_sha256": _sha(run_dir / "artifacts_sha256.json"),
        "checkpoint_manifest_sha256": checkpoint_sha,
        "checkpoint_directory": model_dir.relative_to(root).as_posix(),
        "post_id_sha256": config["post_id_sha256"],
    })
    _verify_condition_complete(run_dir=run_dir, model_dir=model_dir, manifest=manifest,
                              backbone=str(config["backbone"]), condition=int(config["context_size"]), root=root)


def _train_condition(
    *, torch: Any, AutoModel: Any, AutoTokenizer: Any, Trainer: Any, TrainingArguments: Any, set_seed: Any,
    backbone: str, condition: int, data: Mapping[str, Any], stage8: Mapping[str, Any], itft: Mapping[str, Any],
    decision_bundle: Mapping[str, Any], stage_root: Path, root_manifest: Mapping[str, Any], use_cuda: bool,
    disable_tqdm: bool, root: Path,
) -> dict[str, Any]:
    run_dir, model_dir, manifest = _condition_manifest(
        backbone=backbone, condition=condition, stage_root=stage_root, root_manifest=root_manifest,
        decision_bundle=decision_bundle, stage8=stage8, itft=itft, data=data, root=root,
    )
    done = _verify_condition_complete(run_dir=run_dir, model_dir=model_dir, manifest=manifest,
                                      backbone=backbone, condition=condition, root=root)
    if done is not None:
        print(f"[verified resume] Stage 12 {backbone} C{condition}; checkpoint hashes match current manifest")
        return done

    # A directory with only its manifest is a fresh run; any previous partial files were rejected above.
    _assert_no_legacy_outputs(root)
    max_length = int(stage8["sequence_config"]["max_length_context"])
    learning_rate = float(stage8["learning_rate_config"]["learning_rate_context"][backbone])
    spec = eu.get_backbone(backbone)
    config = dict(manifest["config"])
    set_seed(RETRAIN_SEED)
    tokenizer = AutoTokenizer.from_pretrained(str(itft["model_dir"]), local_files_only=True, use_fast=True)
    eu.configure_tokenizer_for_input(tokenizer, INPUT_MODE)
    actual_tokenizer = getattr(tokenizer, "_commit_hash", None) or (getattr(tokenizer, "init_kwargs", {}) or {}).get("_commit_hash")
    if actual_tokenizer and str(actual_tokenizer).lower() != str(itft["tokenizer_revision"]).lower():
        raise RuntimeError(f"Saved ITFT tokenizer resolved to {actual_tokenizer!r}, expected {itft['tokenizer_revision']!r}")
    separator = eu.sep_from_tokenizer(tokenizer)
    texts = [eu.build_context_text(row, condition, sep=separator) for _, row in data["matched"].iterrows()]
    encoded = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )
    labels = data["labels"]

    Dataset = _make_dataset_class(torch)
    train_ds = Dataset(dict(encoded), labels)
    model = AutoModel.from_pretrained(
        str(itft["model_dir"]),
        local_files_only=True,
        num_labels=len(eu.LABEL2ID),
        id2label=eu.ID2LABEL,
        label2id=eu.LABEL2ID,
    )
    if getattr(model.config, "num_labels", None) != len(eu.LABEL2ID):
        raise RuntimeError(f"The selected {backbone} ITFT checkpoint does not have the expected three-class head")

    started = time.perf_counter()
    tmp_model_dir: Path | None = None
    model_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"stage12-{backbone}-C{condition}-", dir=root) as temp_root_name:
        temp_root = Path(temp_root_name)
        training_args = _make_training_arguments(
            TrainingArguments,
            output_dir=temp_root / "trainer_output",
            learning_rate=learning_rate,
            use_cuda=use_cuda,
            seed=RETRAIN_SEED,
            disable_tqdm=disable_tqdm,
        )
        trainer = Trainer(model=model, args=training_args, train_dataset=train_ds)
        trainer.train()
        tmp_model_dir = temp_root / "checkpoint"
        trainer.save_model(str(tmp_model_dir))
        tokenizer.save_pretrained(str(tmp_model_dir))
        elapsed_seconds = time.perf_counter() - started

        checkpoint_file_hashes = _sha_tree(tmp_model_dir)
        checkpoint_doc = {
            "schema_version": eu.CONFIG_SCHEMA_VERSION,
            "run_id": eu.RUN_ID,
            "stage": STAGE,
            "script_name": SCRIPT_NAME,
            "backbone": backbone,
            "hf_model_id": spec.hf_model_id,
            "model_revision": itft["model_revision"],
            "tokenizer_revision": itft["tokenizer_revision"],
            "context_size": condition,
            "selected_context_size_c_best": int(decision_bundle["decision"]["c_best_by_backbone"][backbone]),
            "stage11_manifest_fingerprint": decision_bundle["manifest_fingerprint"],
            "stage11_decision_sha256": _sha(decision_bundle["root_path"]),
            "stage8_manifest_fingerprint": stage8["manifest_fingerprint"],
            "itft_manifest_fingerprint": itft["manifest_fingerprint"],
            "manifest_fingerprint": manifest["fingerprint"],
            "dataset_sha256": data["dataset_sha256"],
            "fold_assignment_sha256": data["fold_assignment_sha256"],
            "post_id_sha256": data["post_id_sha256"],
            "config": config,
            "checkpoint_files_sha256": checkpoint_file_hashes,
        }
        _write_json_atomic(tmp_model_dir / "retrain_checkpoint_manifest.json", checkpoint_doc)
        if model_dir.exists():
            raise FileExistsError(f"Refusing to overwrite an existing Stage 12 checkpoint: {model_dir}; archive it first")
        shutil.move(str(tmp_model_dir), str(model_dir))

    elapsed = float(elapsed_seconds)
    metrics = {
        "schema_version": eu.CONFIG_SCHEMA_VERSION,
        "run_id": eu.RUN_ID,
        "stage": STAGE,
        "script_name": SCRIPT_NAME,
        "backbone": backbone,
        "experiment_id": eu.experiment2_id(condition, backbone),
        "context_size": condition,
        "selected_context_size_c_best": int(decision_bundle["decision"]["c_best_by_backbone"][backbone]),
        "stage11_manifest_fingerprint": decision_bundle["manifest_fingerprint"],
        "itft_manifest_fingerprint": itft["manifest_fingerprint"],
        "n_train": int(len(train_ds)),
        "seed": RETRAIN_SEED,
        "max_length_context": max_length,
        "truncation_side": TRUNCATION_SIDE,
        "num_train_epochs": NUM_EPOCHS,
        "per_device_train_batch_size": TRAIN_BATCH_SIZE,
        "learning_rate_context": learning_rate,
        "weight_decay": WEIGHT_DECAY,
        "loss": LOSS_NAME,
        "checkpoint_selection": "none_fixed_epochs_full_matched_subset",
        "train_seconds": round(elapsed, 3),
        "train_minutes": round(elapsed / 60, 3),
        "checkpoint_directory": model_dir.relative_to(root).as_posix(),
    }
    _write_condition_complete(run_dir=run_dir, model_dir=model_dir, manifest=manifest, config=config, metrics=metrics, root=root)
    print(f"[trained] {backbone} C{condition} | {elapsed / 60:.2f} min | checkpoint={model_dir.relative_to(root)}")
    del model, trainer, train_ds, encoded, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


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
    TrainingArguments: Any,
    *,
    output_dir: Path,
    learning_rate: float,
    use_cuda: bool,
    seed: int,
    disable_tqdm: bool,
):
    import inspect
    signature = inspect.signature(TrainingArguments.__init__).parameters
    eval_key = "eval_strategy" if "eval_strategy" in signature else "evaluation_strategy"
    kwargs = {
        "output_dir": str(output_dir),
        "num_train_epochs": NUM_EPOCHS,
        "per_device_train_batch_size": TRAIN_BATCH_SIZE,
        "per_device_eval_batch_size": EVAL_BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "learning_rate": float(learning_rate),
        "weight_decay": WEIGHT_DECAY,
        "warmup_ratio": WARMUP_RATIO,
        "lr_scheduler_type": LR_SCHEDULER,
        eval_key: "no",
        "save_strategy": "no",
        "logging_strategy": "epoch",
        "seed": int(seed),
        "data_seed": int(seed),
        "report_to": "none",
        "fp16": bool(use_cuda),
        "disable_tqdm": bool(disable_tqdm),
    }
    for key in ("data_seed", "lr_scheduler_type", "warmup_ratio", "disable_tqdm"):
        if key not in signature:
            kwargs.pop(key, None)
    return TrainingArguments(**kwargs)


def _write_stage_summary(
    *, stage_root: Path, root_manifest: Mapping[str, Any], backbone: str, selected_context: int,
    all_metrics: Sequence[Mapping[str, Any]], root: Path,
) -> None:
    csv_path = stage_root / "e2_full_retrain_metrics.csv"
    json_path = stage_root / "e2_full_retrain_summary.json"
    rows = sorted((dict(row) for row in all_metrics), key=lambda row: int(row["context_size"]))
    expected_contexts = [int(row["context_size"]) for row in rows]
    if expected_contexts != list(CONDITIONS):
        raise ValueError(f"Stage 12 summary must include exactly C0-C4; got {expected_contexts}")
    metrics_df = pd.DataFrame(rows)
    # Checkpoint locations and selected context are recorded rather than resolving a new winner from these retrains.
    _write_csv_atomic(csv_path, metrics_df)
    _write_json_atomic(json_path, {
        "schema_version": eu.CONFIG_SCHEMA_VERSION,
        "run_id": eu.RUN_ID,
        "stage": STAGE,
        "script_name": SCRIPT_NAME,
        "backbone": backbone,
        "selected_context_size_c_best": selected_context,
        "trained_contexts": list(CONDITIONS),
        "n_models": len(rows),
        "dataset_rows_per_model": EXPECTED_ROWS,
        "metrics_csv_sha256": _sha(csv_path),
        "stage12_manifest_fingerprint": root_manifest["fingerprint"],
        "stage11_manifest_fingerprint": root_manifest["config"]["stage11_manifest_fingerprint"],
        "interpretation": "Retraining outputs support Stage 13 descriptive inference; C_best remains the Stage 11 CV decision.",
    })
    index_hashes: dict[str, str] = {
        "e2_full_retrain_metrics.csv": _sha(csv_path),
        "e2_full_retrain_summary.json": _sha(json_path),
    }
    for condition in CONDITIONS:
        run_dir = _expected_run_dir(backbone, condition, root)
        for name in (eu.RUN_MANIFEST_NAME, "run_complete.json", "metrics.json", "artifacts_sha256.json"):
            artifact = run_dir / name
            if not artifact.is_file():
                raise FileNotFoundError(f"Cannot complete Stage 12 summary; missing {artifact}")
            index_hashes[f"C{condition}/{name}"] = _sha(artifact)
        model_manifest = _model_dir(backbone, condition, root) / "retrain_checkpoint_manifest.json"
        index_hashes[f"checkpoint_manifests/C{condition}"] = _sha(model_manifest)
    index_path = stage_root / "artifacts_sha256.json"
    _write_json_atomic(index_path, {
        "schema_version": eu.CONFIG_SCHEMA_VERSION,
        "run_id": eu.RUN_ID,
        "stage": STAGE,
        "script_name": SCRIPT_NAME,
        "backbone": backbone,
        "selected_context_size_c_best": selected_context,
        "manifest_fingerprint": root_manifest["fingerprint"],
        "sha256": index_hashes,
    })
    _write_json_atomic(stage_root / "stage12_complete.json", {
        "schema_version": eu.CONFIG_SCHEMA_VERSION,
        "run_id": eu.RUN_ID,
        "stage": STAGE,
        "script_name": SCRIPT_NAME,
        "backbone": backbone,
        "selected_context_size_c_best": selected_context,
        "n_models": len(rows),
        "manifest_fingerprint": root_manifest["fingerprint"],
        "artifact_index_sha256": _sha(index_path),
        "stage11_manifest_fingerprint": root_manifest["config"]["stage11_manifest_fingerprint"],
    })
    eu.assert_current_run_artifact(
        stage_root / "stage12_complete.json",
        expected_stage=STAGE,
        expected_script=SCRIPT_NAME,
        expected_fingerprint=str(root_manifest["fingerprint"]),
    )


def _verify_stage_summary(stage_root: Path, root_manifest: Mapping[str, Any], *, backbone: str, selected_context: int, root: Path) -> bool:
    marker_path = stage_root / "stage12_complete.json"
    index_path = stage_root / "artifacts_sha256.json"
    if not marker_path.exists():
        summary_outputs = [stage_root / name for name in ("e2_full_retrain_metrics.csv", "e2_full_retrain_summary.json", "artifacts_sha256.json") if (stage_root / name).exists()]
        if summary_outputs:
            raise RuntimeError(f"Stage 12 summary has partial outputs without a completion marker: {summary_outputs}; archive it")
        return False
    eu.assert_current_run_artifact(marker_path, expected_stage=STAGE, expected_script=SCRIPT_NAME, expected_fingerprint=str(root_manifest["fingerprint"]))
    marker = _read_json(marker_path)
    if marker.get("artifact_index_sha256") != _sha(index_path):
        raise ValueError("Stage 12 summary marker has a stale output index")
    index = _read_json(index_path)
    for key, expected in {
        "run_id": eu.RUN_ID,
        "stage": STAGE,
        "script_name": SCRIPT_NAME,
        "backbone": backbone,
        "selected_context_size_c_best": selected_context,
        "manifest_fingerprint": root_manifest["fingerprint"],
    }.items():
        if str(index.get(key)) != str(expected) or str(marker.get(key)) != str(expected):
            raise ValueError(f"Stage 12 summary bundle identity mismatch for {key}")
    hashes = index.get("sha256")
    if not isinstance(hashes, Mapping):
        raise ValueError("Stage 12 summary artifact index has no sha256 map")
    for relative, digest in hashes.items():
        if relative.startswith("checkpoint_manifests/"):
            model_context = relative.rsplit("C", 1)[-1]
            path = _model_dir(backbone, int(model_context), root) / "retrain_checkpoint_manifest.json"
        else:
            path = stage_root / relative
        if not path.is_file() or _sha(path) != digest:
            raise ValueError(f"Stage 12 summary artifact changed or is missing: {path}")
    return True


def validate_prerequisites(*, root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Audit all required upstream data, configs, selected decision and checkpoint provenance."""
    _assert_no_legacy_outputs(root)
    data = _load_and_validate_data(root)
    decision_bundle = _verify_stage11_decision(root)
    decision = decision_bundle["decision"]
    backbone = str(decision["decision"])
    selected_contexts = decision["c_best_by_backbone"]
    if set(selected_contexts) != set(eu.BACKBONES):
        raise ValueError("Stage 11 must provide C_best for both registered backbones")
    selected_context = int(selected_contexts[backbone])
    if selected_context not in CONDITIONS:
        raise ValueError(f"Stage 11 selected an invalid C_best value: C{selected_context}")
    if int(decision.get("n_oof_examples", -1)) != EXPECTED_ROWS:
        raise ValueError("Stage 11 decision does not describe the expected 900-example matched OOF set")
    if decision.get("post_id_sha256") != data["post_id_sha256"]:
        raise ValueError("Stage 11 decision post-ID hash differs from the current stable matched-subset order")

    stage8 = _verify_stage8_publication(root)
    itft = _validate_itft_checkpoint(backbone, stage8, root)
    # Ensure the selected backbone is exactly the one certified by Stage 11; no local override is accepted.
    shared_lr = stage8["learning_rate_config"]
    shared_seq = stage8["sequence_config"]
    if backbone not in eu.BACKBONES:
        raise ValueError(f"Stage 11 selected an unknown backbone {backbone!r}")
    if shared_lr.get("learning_rate_context", {}).get(backbone) is None:
        raise ValueError(f"Missing contextual LR for selected backbone {backbone}")
    if int(shared_seq.get("max_length_context", 0)) < 8:
        raise ValueError("Stage 8 contextual max length is missing or invalid")

    configs = {}
    for condition in CONDITIONS:
        configs[condition] = _build_resolved_config(
            backbone=backbone, condition=condition, decision_bundle=decision_bundle,
            stage8=stage8, itft=itft, data=data,
        )
    # Decision-dependent execution may proceed only if the Stage 11 choice and both upstream grids are current.
    root_dir, root_manifest = _prepare_root_manifest(
        backbone=backbone, decision_bundle=decision_bundle, stage8=stage8, itft=itft, data=data, root=root,
    )
    return {
        "backbone": backbone,
        "selected_context": selected_context,
        "decision_bundle": decision_bundle,
        "stage8": stage8,
        "itft": itft,
        "data": data,
        "resolved_configs": configs,
        "stage_root": root_dir,
        "root_manifest": root_manifest,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 12: retrain C0-C4 on the matched subset for the Stage 11 selected backbone.")
    parser.add_argument("--validate-only", action="store_true", help="Validate upstream artifacts/configurations; do not train or write model outputs.")
    parser.add_argument("--allow-cpu", action="store_true", help="Explicitly allow slow CPU training (not recommended for the clean sweep).")
    parser.add_argument("--disable-tqdm", action="store_true", help="Disable progress bars, useful in notebook logs.")
    args = parser.parse_args(argv)
    os.chdir(PROJECT_ROOT)

    audit = validate_prerequisites(root=PROJECT_ROOT)
    print("Stage 12 prerequisite audit passed.")
    print(f"  Selected backbone: {audit['backbone']}")
    print(f"  Selected C_best: C{audit['selected_context']}")
    print(f"  Matched examples: {len(audit['data']['matched'])}")
    print(f"  max_length_context: {audit['stage8']['sequence_config']['max_length_context']}")
    print(f"  learning_rate_context: {audit['stage8']['learning_rate_config']['learning_rate_context'][audit['backbone']]}")
    print(f"  Output root: {audit['stage_root']}")
    if args.validate_only:
        print("Validation-only complete; no models were trained and no Stage 12 completion marker was written.")
        return 0

    # The code for future stages can be prepared before those stages have run, but the full clean sweep
    # is not started until every authoritative script/directory has been migrated and audited.
    eu.audit_pipeline_script_names(PROJECT_ROOT, require_all=True)
    # The run manifest is prepared only for an execution attempt, not by --validate-only.
    eu.prepare_run_directory(audit["stage_root"], audit["root_manifest"], allow_resume=True)
    if _verify_stage_summary(
        audit["stage_root"], audit["root_manifest"], backbone=audit["backbone"],
        selected_context=audit["selected_context"], root=PROJECT_ROOT,
    ):
        # Still audit all five model checkpoint bundles before treating the stage as complete.
        for condition in CONDITIONS:
            run_dir = _expected_run_dir(audit["backbone"], condition, PROJECT_ROOT)
            model_dir = _model_dir(audit["backbone"], condition, PROJECT_ROOT)
            cond_manifest = _read_json(run_dir / eu.RUN_MANIFEST_NAME)
            _verify_condition_complete(run_dir=run_dir, model_dir=model_dir, manifest=cond_manifest,
                                       backbone=audit["backbone"], condition=condition, root=PROJECT_ROOT)
        print("Stage 12 already completed and all five model/checkpoint bundles passed hash verification; nothing to rerun.")
        return 0
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, set_seed
    except ImportError as exc:
        raise ImportError("Stage 12 training requires PyTorch and Hugging Face Transformers; use the project GPU/Colab environment") from exc
    use_cuda = bool(torch.cuda.is_available())
    if not use_cuda and not args.allow_cpu:
        raise RuntimeError("Stage 12 requires a GPU runtime. Set Colab to GPU, or pass --allow-cpu only for an intentional test run.")
    if not use_cuda:
        print("WARNING: CPU training explicitly allowed; runtime will be substantially slower than the protocol's GPU runs.")

    rows: list[dict[str, Any]] = []
    expected_fingerprints = []
    for condition in CONDITIONS:
        run_dir, model_dir, manifest = _condition_manifest(
            backbone=audit["backbone"], condition=condition, stage_root=audit["stage_root"],
            root_manifest=audit["root_manifest"], decision_bundle=audit["decision_bundle"],
            stage8=audit["stage8"], itft=audit["itft"], data=audit["data"], root=PROJECT_ROOT,
        )
        expected_fingerprints.append(manifest["fingerprint"])
        metrics = _train_condition(
            torch=torch,
            AutoModel=AutoModelForSequenceClassification,
            AutoTokenizer=AutoTokenizer,
            Trainer=Trainer,
            TrainingArguments=TrainingArguments,
            set_seed=set_seed,
            backbone=audit["backbone"],
            condition=condition,
            data=audit["data"],
            stage8=audit["stage8"],
            itft=audit["itft"],
            decision_bundle=audit["decision_bundle"],
            stage_root=audit["stage_root"],
            root_manifest=audit["root_manifest"],
            use_cuda=use_cuda,
            disable_tqdm=args.disable_tqdm,
            root=PROJECT_ROOT,
        )
        rows.append(metrics)

    # Verify every checkpoint again before publishing the aggregate summary.
    for condition, expected_fp in zip(CONDITIONS, expected_fingerprints):
        run_dir = _expected_run_dir(audit["backbone"], condition, PROJECT_ROOT)
        model_dir = _model_dir(audit["backbone"], condition, PROJECT_ROOT)
        run_manifest = _read_json(run_dir / eu.RUN_MANIFEST_NAME)
        if run_manifest.get("fingerprint") != expected_fp:
            raise ValueError(f"Stage 12 C{condition} manifest changed during retraining")
        result = _verify_condition_complete(
            run_dir=run_dir, model_dir=model_dir, manifest=run_manifest,
            backbone=audit["backbone"], condition=condition, root=PROJECT_ROOT,
        )
        if result is None:
            raise RuntimeError(f"Stage 12 C{condition} did not complete")
    _write_stage_summary(
        stage_root=audit["stage_root"], root_manifest=audit["root_manifest"],
        backbone=audit["backbone"], selected_context=audit["selected_context"],
        all_metrics=rows, root=PROJECT_ROOT,
    )
    _verify_stage_summary(
        audit["stage_root"], audit["root_manifest"], backbone=audit["backbone"],
        selected_context=audit["selected_context"], root=PROJECT_ROOT,
    )
    print("\nStage 12 complete. C0-C4 retrained on the full matched 900-example subset.")
    print(f"Selected C_best remains C{audit['selected_context']} from Stage 11; Stage 12 did not reselect it.")
    print(f"Models: {PROJECT_ROOT / 'models' / 'e2_full_retrained' / audit['backbone']} / C0-C4")
    print(f"Verified summary: {audit['stage_root'] / 'e2_full_retrain_metrics.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
