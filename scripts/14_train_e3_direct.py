"""Stage 14: Experiment III direct fine-tuning without ITFT.

Authoritative pipeline script: ``scripts/14_train_e3_direct.py``.

The selected backbone and C_best are loaded from the verified Stage 11
``results/backbone_decision.json`` bundle. This script never assumes that
IndoBERTweet wins and never reselects the context size. It trains only the
Stage 11-selected backbone, from its pinned public pretrained checkpoint
(without an ITFT checkpoint), for C0 and that backbone's selected C_best.

Each condition uses the shared five folds and two seeds (42, 123), with the
same contextual learning rate, maximum sequence length, epoch budget, batch
configuration, input construction, and left-truncation policy as Experiment
II-A. The held-out fold is never used for epoch selection or early stopping.
When C_best is C0, the unique condition is trained once (10 runs); the script
does not duplicate the same C0 runs under a second label.

Run after Stages 2, 8, 9/10, and 11 have completed and their artefacts passed
verification. The script can be authored/tested before those results exist,
but its validation/execution must fail closed until the real upstream bundles
are available.

Usage (repository root, normally in Colab with a GPU):
    python scripts/14_train_e3_direct.py --validate-only
    python scripts/14_train_e3_direct.py

Outputs are written beneath:
    results/runs/left_truncation_v2/stage_14/e3_<selected_backbone>/
        C{condition}/fold_{fold:02d}/seed_{seed}/

Each run includes predictions.csv, probabilities.npy, metrics.json, an
artifact-hash inventory, completion marker, and run manifest. Aggregate OOF
metrics and class/confusion summaries are written to the stage root. Old flat
``results/raw``/``results/tables`` Experiment III files are never accepted as
resume signals. A previous run is reusable only if its current manifest and
all indexed artefacts still match the frozen utility, script, protocol,
pipeline, dataset, Stage 8 configuration, Stage 11 decision, and paired E2
reference runs.
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


def _project_root() -> Path:
    here = Path(__file__).resolve().parent
    return here.parent if here.name.casefold() == "scripts" else here


PROJECT_ROOT = _project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import experiment_utils as eu  # noqa: E402

STAGE = 14
SCRIPT_NAME = "scripts/14_train_e3_direct.py"
INPUT_MODE = "contextual"
TRUNCATION_SIDE = "left"
FOLDS = tuple(range(5))
SEEDS = tuple(eu.CV_SEEDS)
BASE_CONDITION = 0
ALL_CONDITIONS = tuple(range(5))
EXPECTED_ROWS = eu.CONTEXT_MATCHED_EXPECTED_ROWS
EXPECTED_FOLD_COUNTS = {fold: EXPECTED_ROWS // len(FOLDS) for fold in FOLDS}

# Frozen contextual hyperparameters shared with Experiment II-A (Stage 9/10).
EXPECTED_HYPERPARAMETERS = {
    "optimizer": "AdamW",
    "weight_decay": 0.01,
    "warmup_ratio": 0.0,
    "lr_scheduler": "linear",
    "max_epochs": 8,
    "early_stopping_patience": None,
    "per_device_train_batch_size": 16,
    "per_device_eval_batch_size": 32,
    "gradient_accumulation_steps": 1,
    "effective_batch_size": 16,
    "loss": "unweighted_cross_entropy",
    "checkpoint_metric": "macro_f1",
}

MATCHED_SUBSET_PATH = Path("data/processed/ctx_matched_subset.csv")
SEQUENCE_CONFIG_PATH = eu.SEQUENCE_LENGTH_CONFIG
LEARNING_RATE_CONFIG_PATH = eu.LEARNING_RATE_CONFIG
DECISION_PATH = eu.BACKBONE_DECISION_PATH
STAGE8_DIR = Path(*eu.RUN_ROOT_PARTS) / "stage_08" / "summary"
STAGE11_DIR = Path(*eu.RUN_ROOT_PARTS) / "stage_11" / "decision"
STAGE14_DIR = Path(*eu.RUN_ROOT_PARTS) / "stage_14"
LEGACY_OUTPUT_PATTERNS = (
    "results/raw/e3_C*_fold*_seed*_predictions.csv",
    "results/raw/e3_C*_fold*_seed*_probs.npy",
    "results/tables/e3_grid_metrics.csv",
    "results/tables/e3_context_summary.csv",
    "results/tables/e3_grid_class_metrics.csv",
    "results/tables/e3_confusion_matrices.csv",
    "models/E3_direct*",
    "models/E3_context_only*",
)
SUMMARY_FILES = (
    "e3_grid_metrics.csv",
    "e3_context_summary.csv",
    "e3_grid_class_metrics.csv",
    "e3_confusion_matrices.csv",
    "e3_manifest.json",
)


def _resolve(path: str | Path, root: Path = PROJECT_ROOT) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def _read_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read valid JSON artefact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object at {path}")
    return value


def _write_json_atomic(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_csv_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _sha(path: Path) -> str:
    return eu.sha256_file(path)


def _sha_tree(directory: Path, *, excluded_names: Sequence[str] = ()) -> dict[str, str]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory not found for SHA-256 inventory: {directory}")
    excluded = set(excluded_names)
    return {
        path.relative_to(directory).as_posix(): _sha(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.name not in excluded
    }


def _write_numpy_atomic(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
    os.replace(temporary, path)


def _legacy_outputs(root: Path = PROJECT_ROOT) -> list[Path]:
    found: list[Path] = []
    for pattern in LEGACY_OUTPUT_PATTERNS:
        found.extend(path for path in root.glob(pattern) if path.exists())
    return list(dict.fromkeys(path.resolve() for path in found))


def _assert_no_legacy_outputs(root: Path = PROJECT_ROOT) -> None:
    eu.assert_no_legacy_outputs(_legacy_outputs(root))


def _verify_index_entries(
    index_path: Path,
    base_dir: Path,
    *,
    expected_identity: Mapping[str, Any],
    special_paths: Mapping[str, Path] | None = None,
) -> dict[str, Any]:
    index = _read_json(index_path)
    for key, expected in expected_identity.items():
        if str(index.get(key)) != str(expected):
            raise ValueError(f"Artifact index {index_path} has wrong {key}: {index.get(key)!r} != {expected!r}")
    hashes = index.get("sha256")
    if not isinstance(hashes, Mapping):
        raise ValueError(f"Artifact index {index_path} has no SHA-256 mapping")
    special_paths = dict(special_paths or {})
    for relative, digest in hashes.items():
        path = special_paths.get(str(relative), base_dir / str(relative))
        if not path.is_file() or _sha(path) != digest:
            raise ValueError(f"Missing or stale indexed artefact: {path}")
    return index


def _verify_stage8(root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Verify Stage 8's published sequence length, LR, latency, and provenance."""
    stage_dir = (root / STAGE8_DIR).resolve()
    manifest_path = stage_dir / eu.RUN_MANIFEST_NAME
    marker_path = stage_dir / "stage8_complete.json"
    index_path = stage_dir / "artifacts_sha256.json"
    receipt_path = stage_dir / "stage8_publish_receipt.json"
    latency_path = stage_dir / "context_latency_measurements.json"
    sequence_path = _resolve(SEQUENCE_CONFIG_PATH, root)
    rates_path = _resolve(LEARNING_RATE_CONFIG_PATH, root)
    required = (manifest_path, marker_path, index_path, receipt_path, latency_path, sequence_path, rates_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Stage 8 publication bundle is incomplete: " + ", ".join(missing))

    manifest = _read_json(manifest_path)
    fingerprint = str(manifest.get("fingerprint", ""))
    eu.assert_current_run_artifact(
        marker_path,
        expected_stage=8,
        expected_script="scripts/8_timing_pilot.py",
        expected_fingerprint=fingerprint,
    )
    marker = _read_json(marker_path)
    if marker.get("manifest_fingerprint") != fingerprint or marker.get("artifact_index_sha256") != _sha(index_path):
        raise ValueError("Stage 8 completion marker has stale manifest/index metadata")
    _verify_index_entries(
        index_path,
        stage_dir,
        expected_identity={
            "schema_version": eu.CONFIG_SCHEMA_VERSION,
            "run_id": eu.RUN_ID,
            "stage": 8,
            "manifest_fingerprint": fingerprint,
        },
    )
    if marker.get("sequence_length_config_sha256") != _sha(sequence_path):
        raise ValueError("Stage 8 sequence_length.json changed after publication")
    if marker.get("learning_rate_config_sha256") != _sha(rates_path):
        raise ValueError("Stage 8 learning_rates.json changed after publication")

    shared = eu.load_shared_run_config(
        sequence_length_path=sequence_path,
        learning_rate_path=rates_path,
        require_context=True,
        require_context_lr=True,
    )
    latency = _read_json(latency_path)
    receipt = _read_json(receipt_path)
    if latency.get("schema_version") != eu.CONFIG_SCHEMA_VERSION or latency.get("run_id") != eu.RUN_ID:
        raise ValueError("Stage 8 latency bundle has a stale schema/run ID")
    if latency.get("manifest_fingerprint") != fingerprint or receipt.get("summary_manifest_fingerprint") != fingerprint:
        raise ValueError("Stage 8 latency/receipt belongs to a different summary manifest")
    if receipt.get("latency_measurements_sha256") != _sha(latency_path):
        raise ValueError("Stage 8 latency file does not match its publish receipt")
    if receipt.get("sequence_length_config_sha256") != _sha(sequence_path) or receipt.get("learning_rate_config_sha256") != _sha(rates_path):
        raise ValueError("Stage 8 publication receipt has stale shared-config hashes")
    if latency.get("model_ids") != {key: eu.get_backbone(key).hf_model_id for key in eu.BACKBONES}:
        raise ValueError("Stage 8 latency artifact model IDs differ from BACKBONES")
    if latency.get("context_sizes") != list(ALL_CONDITIONS) or latency.get("truncation_side") != TRUNCATION_SIDE:
        raise ValueError("Stage 8 latency artifact must cover C0-C4 and declare left truncation")
    sequence = shared["sequence_length"]
    rates = shared["learning_rates"]
    token_revisions = sequence.get("context_tokenizer_resolved_revisions")
    preflight = rates.get("context_preflight")
    model_revisions = preflight.get("model_revisions") if isinstance(preflight, Mapping) else None
    if not isinstance(token_revisions, Mapping) or set(token_revisions) != set(eu.BACKBONES):
        raise ValueError("Stage 8 sequence config must pin tokenizer revisions for both backbones")
    if not isinstance(model_revisions, Mapping) or set(model_revisions) != set(eu.BACKBONES):
        raise ValueError("Stage 8 contextual preflight must pin model revisions for both backbones")
    if latency.get("tokenizer_revisions") != token_revisions or latency.get("model_revisions") != model_revisions:
        raise ValueError("Stage 8 latency revisions differ from published model/tokenizer revisions")

    measurements = latency.get("measurements_ms_by_context")
    medians = latency.get("median_ms_by_context")
    if not isinstance(measurements, Mapping) or not isinstance(medians, Mapping):
        raise ValueError("Stage 8 latency artefact is missing raw samples or medians")
    for condition in ALL_CONDITIONS:
        key = f"C{condition}"
        if key not in measurements or key not in medians:
            raise ValueError(f"Stage 8 latency artefact is missing {key}")
        for backbone in eu.BACKBONES:
            values = np.asarray(measurements[key].get(backbone, []), dtype=float)
            recorded = medians[key].get(backbone)
            if values.ndim != 1 or len(values) < 20 or not np.isfinite(values).all() or (values <= 0).any():
                raise ValueError(f"Stage 8 requires at least 20 finite positive latency samples for {key}/{backbone}")
            if recorded is None or not math.isclose(float(recorded), float(np.median(values)), rel_tol=1e-8, abs_tol=1e-8):
                raise ValueError(f"Stage 8 latency median is invalid for {key}/{backbone}")

    context_preflight = dict(preflight)
    hyper = context_preflight.get("hyperparameters")
    if not isinstance(hyper, Mapping):
        raise ValueError("Stage 8 contextual preflight is missing fixed hyperparameters")
    mismatches = {
        key: (hyper.get(key), expected)
        for key, expected in EXPECTED_HYPERPARAMETERS.items()
        if hyper.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"Stage 8 contextual hyperparameters differ from the frozen Experiment II/III setup: {mismatches}")
    if not isinstance(hyper.get("mixed_precision"), bool):
        raise ValueError("Stage 8 contextual preflight must record mixed_precision as a boolean")

    return {
        "stage_dir": stage_dir,
        "manifest_path": manifest_path,
        "manifest": manifest,
        "fingerprint": fingerprint,
        "marker_path": marker_path,
        "index_path": index_path,
        "receipt_path": receipt_path,
        "latency_path": latency_path,
        "sequence_path": sequence_path,
        "learning_rates_path": rates_path,
        "sequence": sequence,
        "learning_rates": rates,
        "latency": latency,
        "context_preflight": context_preflight,
        "hyperparameters": dict(hyper),
    }


def _verify_stage11_decision(root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Verify the Stage 11 decision bundle, its source hashes, and both C_best values."""
    root_decision = _resolve(DECISION_PATH, root)
    stage_dir = (root / STAGE11_DIR).resolve()
    manifest_path = stage_dir / eu.RUN_MANIFEST_NAME
    marker_path = stage_dir / "stage11_complete.json"
    index_path = stage_dir / "artifacts_sha256.json"
    copy_path = stage_dir / "backbone_decision.json"
    required = (root_decision, manifest_path, marker_path, index_path, copy_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Stage 11 decision bundle is incomplete: " + ", ".join(missing))

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
        raise ValueError("Stage 11 completion marker does not match its manifest")
    if marker.get("artifact_index_sha256") != _sha(index_path):
        raise ValueError("Stage 11 completion marker has a stale artifact-index hash")
    if marker.get("root_decision_sha256") != _sha(root_decision):
        raise ValueError("Root backbone_decision.json differs from the Stage 11 completion marker")
    if _sha(copy_path) != _sha(root_decision):
        raise ValueError("Root and versioned Stage 11 decision JSON files differ")

    index = _verify_index_entries(
        index_path,
        stage_dir,
        expected_identity={
            "schema_version": eu.CONFIG_SCHEMA_VERSION,
            "run_id": eu.RUN_ID,
            "stage": 11,
            "script_name": "scripts/11_select_backbone.py",
            "manifest_fingerprint": fingerprint,
        },
    )
    decision = eu.load_backbone_decision_document(root_decision)
    eu.validate_backbone_decision_document(decision)
    if decision.get("stage11_manifest_fingerprint") != fingerprint:
        raise ValueError("Stage 11 decision points to a different manifest")
    if decision.get("stage11_output_dir") != stage_dir.relative_to(root).as_posix():
        raise ValueError("Stage 11 decision has an unexpected versioned output directory")
    if decision.get("decision") not in eu.BACKBONES:
        raise ValueError("Stage 11 selected an unregistered backbone")
    if marker.get("selected_backbone") != decision.get("decision"):
        raise ValueError("Stage 11 marker and decision disagree on selected backbone")
    if marker.get("c_best_by_backbone") != decision.get("c_best_by_backbone"):
        raise ValueError("Stage 11 marker and decision disagree on C_best values")
    if decision.get("n_oof_examples") != EXPECTED_ROWS:
        raise ValueError("Stage 11 decision does not describe the expected 900-example OOF set")

    for backbone in eu.BACKBONES:
        context_path = stage_dir / f"context_selection_{backbone}.json"
        grid_path = stage_dir / f"grid_audit_{backbone}.json"
        context_doc = _read_json(context_path)
        grid_doc = _read_json(grid_path)
        if context_doc != decision["context_decisions"][backbone]:
            raise ValueError(f"Stage 11 decision and context document disagree for {backbone}")
        if grid_doc.get("grid_fingerprint") != decision["source_grid_fingerprints"][backbone]:
            raise ValueError(f"Stage 11 source-grid fingerprint mismatch for {backbone}")
        if int(grid_doc.get("n_runs", -1)) != 50 or int(grid_doc.get("expected_runs", -1)) != 50:
            raise ValueError(f"Stage 11 did not certify all 50 runs for {backbone}")
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
        "index": index,
    }


def _load_and_validate_data(root: Path = PROJECT_ROOT) -> dict[str, Any]:
    matched_path = _resolve(MATCHED_SUBSET_PATH, root)
    if not matched_path.is_file():
        raise FileNotFoundError(f"Stage 2 matched subset is missing: {matched_path}")
    stage2_manifest = eu.assert_current_stage_manifest(2, root)
    raw = pd.read_csv(matched_path)
    fold_audit = eu.validate_context_dataset(raw, expected_rows=EXPECTED_ROWS)
    modelling = eu.load_modelling_view(raw, include_fold=True)
    modelling[eu.CTX_FOLD_COLUMN] = pd.to_numeric(modelling[eu.CTX_FOLD_COLUMN], errors="raise").astype(int)
    modelling = modelling.sort_values(eu.CTX_ID_COLUMN, key=lambda col: col.astype(str)).reset_index(drop=True)
    fold_counts = modelling[eu.CTX_FOLD_COLUMN].value_counts().sort_index().to_dict()
    if fold_counts != EXPECTED_FOLD_COUNTS:
        raise ValueError(f"Shared folds must contain exactly 180 held-out posts each; found {fold_counts}")
    labels = modelling["label"].map(eu.LABEL2ID)
    if labels.isna().any():
        raise ValueError("Matched subset contains non-canonical labels")
    post_ids = modelling[eu.CTX_ID_COLUMN].astype(str).tolist()
    if len(post_ids) != EXPECTED_ROWS or len(set(post_ids)) != EXPECTED_ROWS:
        raise ValueError("Matched subset post IDs must be unique and cover all 900 rows")
    return {
        "path": matched_path,
        "raw": raw,
        "matched": modelling,
        "stage2_manifest": stage2_manifest,
        "fold_audit": fold_audit,
        "dataset_sha256": _sha(matched_path),
        "fold_assignment_sha256": fold_audit["fold_assignment_sha256"],
        "post_ids": post_ids,
        "post_id_sha256": eu.sha256_json(post_ids),
        "labels": labels.astype(int).to_numpy(dtype=np.int64),
        "stage2_manifest_path": root / Path(*eu.RUN_ROOT_PARTS) / "stage_02" / eu.RUN_MANIFEST_NAME,
    }


def _verify_e2_grid(
    *, backbone: str,
    decision_bundle: Mapping[str, Any],
    stage8: Mapping[str, Any],
    data: Mapping[str, Any],
    root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Reverify the selected backbone's complete E2 grid and output artefacts."""
    stage = 9 if backbone == "indobertweet" else 10
    stage_root = eu.versioned_run_root(root) / f"stage_{stage:02d}" / f"e2_{backbone}"
    metrics_path = stage_root / "e2_grid_metrics.csv"
    summary_path = stage_root / "e2_grid_summary.json"
    if not metrics_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError(f"Complete Stage {stage} aggregate grid is missing under {stage_root}")
    summary = _read_json(summary_path)
    if summary.get("grid_metrics_sha256") != _sha(metrics_path):
        raise ValueError("Selected-backbone E2 grid summary hash does not match e2_grid_metrics.csv")
    for key, expected in {
        "schema_version": eu.CONFIG_SCHEMA_VERSION,
        "run_id": eu.RUN_ID,
        "stage": stage,
        "script_name": eu.PIPELINE_SCRIPTS[stage],
        "backbone": backbone,
        "complete": True,
        "n_runs": 50,
        "expected_runs": 50,
        "contexts": list(ALL_CONDITIONS),
        "folds": list(FOLDS),
        "seeds": list(SEEDS),
        "dataset_sha256": data["dataset_sha256"],
        "fold_assignment_sha256": data["fold_assignment_sha256"],
        "max_length_context": int(stage8["sequence"]["max_length_context"]),
        "truncation_side": TRUNCATION_SIDE,
    }.items():
        if summary.get(key) != expected:
            raise ValueError(f"Stage {stage} E2 grid summary mismatch for {key}: {summary.get(key)!r} != {expected!r}")

    contexts_decision = decision_bundle["decision"]["c_best_by_backbone"]
    if int(contexts_decision[backbone]) != int(decision_bundle["decision"]["context_decisions"][backbone]["selected_context"]):
        raise ValueError(f"Stage 11 context decision is internally inconsistent for {backbone}")

    expected_id_order_by_fold = {
        fold: data["matched"].loc[data["matched"][eu.CTX_FOLD_COLUMN] == fold, eu.CTX_ID_COLUMN].astype(str).tolist()
        for fold in FOLDS
    }
    expected_y_by_fold = {
        fold: data["matched"].loc[data["matched"][eu.CTX_FOLD_COLUMN] == fold, "label"].map(eu.LABEL2ID).astype(int).to_numpy()
        for fold in FOLDS
    }
    run_records: list[dict[str, Any]] = []
    input_paths: list[Path] = [metrics_path, summary_path]
    run_metric_rows: list[dict[str, Any]] = []
    for condition in ALL_CONDITIONS:
        for fold in FOLDS:
            for seed in SEEDS:
                run_dir = stage_root / f"C{condition}" / f"fold_{fold:02d}" / f"seed_{seed}"
                manifest_path = run_dir / eu.RUN_MANIFEST_NAME
                marker_path = run_dir / "run_complete.json"
                index_path = run_dir / "artifacts_sha256.json"
                predictions_path = run_dir / "predictions.csv"
                probabilities_path = run_dir / "probabilities.npy"
                metrics_run_path = run_dir / "metrics.json"
                required = (manifest_path, marker_path, index_path, predictions_path, probabilities_path, metrics_run_path)
                missing = [str(path) for path in required if not path.is_file()]
                if missing:
                    raise FileNotFoundError(f"Stage {stage} E2 run C{condition}/fold {fold}/seed {seed} is incomplete: {missing}")
                manifest = _read_json(manifest_path)
                fingerprint = str(manifest.get("fingerprint", ""))
                eu.assert_current_run_artifact(
                    marker_path,
                    expected_stage=stage,
                    expected_script=eu.PIPELINE_SCRIPTS[stage],
                    expected_fingerprint=fingerprint,
                )
                identity = {
                    "schema_version": eu.CONFIG_SCHEMA_VERSION,
                    "run_id": eu.RUN_ID,
                    "stage": stage,
                    "script_name": eu.PIPELINE_SCRIPTS[stage],
                    "backbone": backbone,
                    "context_size": condition,
                    "fold": fold,
                    "seed": seed,
                    "manifest_fingerprint": fingerprint,
                }
                marker = _read_json(marker_path)
                for key, expected in identity.items():
                    if str(marker.get(key)) != str(expected):
                        raise ValueError(f"E2 run marker mismatch for {key}: {marker.get(key)!r} != {expected!r}")
                index = _verify_index_entries(index_path, run_dir, expected_identity=identity)
                if marker.get("artifact_index_sha256") != _sha(index_path):
                    raise ValueError(f"E2 run marker references a stale artefact index: {index_path}")
                predictions_df = pd.read_csv(predictions_path)
                required_pred_cols = {eu.CTX_ID_COLUMN, eu.CTX_FOLD_COLUMN, "context_size", "seed", "y_true", "y_pred"}
                if not required_pred_cols.issubset(predictions_df.columns):
                    raise ValueError(f"E2 prediction file {predictions_path} is missing required fields")
                if predictions_df[eu.CTX_ID_COLUMN].astype(str).tolist() != expected_id_order_by_fold[fold]:
                    raise ValueError(f"E2 held-out post-ID order is wrong in {predictions_path}")
                if not (predictions_df[eu.CTX_FOLD_COLUMN].astype(int) == fold).all():
                    raise ValueError(f"E2 prediction file fold values are wrong in {predictions_path}")
                y_true_stored = predictions_df["y_true"].map(eu.LABEL2ID)
                if y_true_stored.isna().any() or not np.array_equal(y_true_stored.astype(int).to_numpy(), expected_y_by_fold[fold]):
                    raise ValueError(f"E2 ground-truth labels do not align to the current matched subset in {predictions_path}")
                probabilities = np.load(probabilities_path, allow_pickle=False)
                if probabilities.shape != (len(expected_id_order_by_fold[fold]), len(eu.LABEL2ID)):
                    raise ValueError(f"E2 probability array has invalid shape in {probabilities_path}: {probabilities.shape}")
                if not np.isfinite(probabilities).all() or (probabilities < 0).any() or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-4):
                    raise ValueError(f"E2 probability array is invalid in {probabilities_path}")
                if "y_pred" in predictions_df.columns:
                    y_pred_stored = predictions_df["y_pred"].map(eu.LABEL2ID)
                    if y_pred_stored.isna().any() or not np.array_equal(probabilities.argmax(axis=1), y_pred_stored.astype(int).to_numpy()):
                        raise ValueError(f"E2 y_pred does not match probability argmax in {predictions_path}")
                metrics_doc = _read_json(metrics_run_path)
                for key, expected in identity.items():
                    if key in metrics_doc and str(metrics_doc.get(key)) != str(expected):
                        raise ValueError(f"E2 metrics identity mismatch for {key} at {metrics_run_path}")
                config = manifest.get("config")
                if not isinstance(config, Mapping):
                    raise ValueError(f"E2 run manifest has no config: {manifest_path}")
                for key, expected in {
                    "dataset_sha256": data["dataset_sha256"],
                    "fold_assignment_sha256": data["fold_assignment_sha256"],
                    "max_length_context": int(stage8["sequence"]["max_length_context"]),
                    "truncation_side": TRUNCATION_SIDE,
                    "learning_rate_context": float(stage8["learning_rates"]["learning_rate_context"][backbone]),
                    "model_revision": str(stage8["learning_rates"]["context_preflight"]["model_revisions"][backbone]).lower(),
                    "tokenizer_revision": str(stage8["sequence"]["context_tokenizer_resolved_revisions"][backbone]).lower(),
                }.items():
                    if key not in config or (isinstance(expected, float) and not math.isclose(float(config[key]), expected, rel_tol=1e-10, abs_tol=1e-12)) or (not isinstance(expected, float) and str(config[key]).lower() != str(expected).lower()):
                        raise ValueError(f"E2 run config mismatch for {key} at {manifest_path}")
                run_records.append({"artifact_path": marker_path, "manifest_fingerprint": fingerprint})
                run_metric_rows.append({"condition": condition, "fold": fold, "seed": seed, **metrics_doc})
                input_paths.extend((manifest_path, marker_path, index_path, predictions_path, probabilities_path, metrics_run_path))

    grid_audit = eu.audit_context_cv_grid(run_records, backbone=backbone)
    expected_grid_fingerprint = decision_bundle["decision"]["source_grid_fingerprints"][backbone]
    if grid_audit["grid_fingerprint"] != expected_grid_fingerprint:
        raise ValueError("Selected-backbone E2 grid differs from the exact grid fingerprint recorded by Stage 11")
    if summary.get("run_manifest_fingerprints") != grid_audit["manifest_fingerprints"]:
        raise ValueError("Stage E2 grid summary manifest fingerprints do not match verified per-run manifests")
    table = pd.read_csv(metrics_path)
    if len(table) != 50 or table.duplicated(["context_size", "fold", "seed"]).any():
        raise ValueError("Selected-backbone E2 aggregate table is not a unique complete 50-run grid")
    table_fps = table["manifest_fingerprint"].astype(str).tolist() if "manifest_fingerprint" in table else []
    if table_fps != grid_audit["manifest_fingerprints"]:
        raise ValueError("Selected-backbone E2 aggregate metric rows do not match verified run-manifest fingerprints")

    return {
        "stage": stage,
        "stage_root": stage_root,
        "metrics_path": metrics_path,
        "summary_path": summary_path,
        "summary": summary,
        "grid_audit": grid_audit,
        "run_records": run_records,
        "input_paths": input_paths,
        "metric_rows": run_metric_rows,
    }


def _unique_conditions(c_best: int) -> tuple[int, ...]:
    if c_best not in ALL_CONDITIONS:
        raise ValueError(f"C_best must be in 0..4, got {c_best}")
    return tuple(dict.fromkeys((BASE_CONDITION, int(c_best))))


def _build_run_config(
    *, backbone: str,
    condition: int,
    fold: int,
    seed: int,
    audit: Mapping[str, Any],
) -> dict[str, Any]:
    spec = eu.get_backbone(backbone)
    sequence = audit["stage8"]["sequence"]
    rates = audit["stage8"]["learning_rates"]
    hyper = audit["stage8"]["hyperparameters"]
    decision = audit["stage11"]["decision"]
    config = {
        "schema_version": eu.CONFIG_SCHEMA_VERSION,
        "run_id": eu.RUN_ID,
        "stage": STAGE,
        "script_name": SCRIPT_NAME,
        "backbone": backbone,
        "hf_model_id": spec.hf_model_id,
        "experiment_id": eu.experiment3_id(condition, backbone),
        "input_mode": INPUT_MODE,
        "input_order": "parents_oldest_to_newest_then_target",
        "context_order": "parent_4,parent_3,parent_2,parent_1,target",
        "parent_columns_order": ["parent_4", "parent_3", "parent_2", "parent_1", "target_text"],
        "separator_token_policy": "tokenizer_sep_token",
        "dataset_sha256": audit["data"]["dataset_sha256"],
        "post_id_sha256": audit["data"]["post_id_sha256"],
        "fold_assignment_sha256": audit["data"]["fold_assignment_sha256"],
        "fold_column": eu.CTX_FOLD_COLUMN,
        "n_folds": len(FOLDS),
        "cv_seeds": list(SEEDS),
        "eval_seeds_exp1": list(eu.EVAL_SEEDS_EXP1),
        "split_seed": 42,
        "context_size": int(condition),
        "fold": int(fold),
        "seed": int(seed),
        "selected_backbone": backbone,
        "selected_context_size_c_best": int(decision["c_best_by_backbone"][backbone]),
        "stage11_manifest_fingerprint": audit["stage11"]["manifest_fingerprint"],
        "stage11_decision_sha256": _sha(audit["stage11"]["root_path"]),
        "stage11_source_grid_fingerprint": audit["stage11"]["decision"]["source_grid_fingerprints"][backbone],
        "stage8_manifest_fingerprint": audit["stage8"]["fingerprint"],
        "stage2_manifest_fingerprint": audit["data"]["stage2_manifest"]["fingerprint"],
        "e2_reference_grid_fingerprint": audit["e2_grid"]["grid_audit"]["grid_fingerprint"],
        "model_initialization": "public_pretrained_base_without_ITFT",
        "itft_used": False,
        "model_revision": audit["model_revisions"][backbone],
        "tokenizer_revision": audit["tokenizer_revisions"][backbone],
        "max_length_benchmark": int(sequence["max_length_benchmark"]),
        "max_length_context": int(sequence["max_length_context"]),
        "truncation_side": TRUNCATION_SIDE,
        "target_only_truncation_side": sequence["target_only_truncation_side"],
        "optimizer": str(hyper["optimizer"]),
        "weight_decay": float(hyper["weight_decay"]),
        "warmup_ratio": float(hyper["warmup_ratio"]),
        "lr_scheduler": str(hyper["lr_scheduler"]),
        "max_epochs": int(hyper["max_epochs"]),
        "num_train_epochs": int(hyper["max_epochs"]),
        "early_stopping_patience": hyper["early_stopping_patience"],
        "loss": str(hyper["loss"]),
        "class_weighting_benchmark": "none_balanced_matched_dataset",
        "checkpoint_metric": str(hyper["checkpoint_metric"]),
        "lr_search_grid": list(eu.LR_SEARCH_GRID),
        "lr_tuning_seed": int(rates["lr_tuning_seed"]),
        "mixed_precision": bool(hyper["mixed_precision"]),
        "per_device_batch_size": int(hyper["per_device_train_batch_size"]),
        "per_device_train_batch_size": int(hyper["per_device_train_batch_size"]),
        "per_device_eval_batch_size": int(hyper["per_device_eval_batch_size"]),
        "gradient_accumulation_steps": int(hyper["gradient_accumulation_steps"]),
        "effective_batch_size": int(hyper["effective_batch_size"]),
        "learning_rate_benchmark": float(rates["learning_rate_benchmark"][backbone]),
        "learning_rate_context": float(rates["learning_rate_context"][backbone]),
        "checkpoint_selection": "fixed_final_epoch_no_heldout_fold_selection",
        "held_out_fold_used_for_checkpoint_selection": False,
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


def _build_root_config(audit: Mapping[str, Any]) -> dict[str, Any]:
    backbone = audit["backbone"]
    spec = eu.get_backbone(backbone)
    sequence = audit["stage8"]["sequence"]
    rates = audit["stage8"]["learning_rates"]
    hyper = audit["stage8"]["hyperparameters"]
    decision = audit["stage11"]["decision"]
    config = {
        "schema_version": eu.CONFIG_SCHEMA_VERSION,
        "run_id": eu.RUN_ID,
        "stage": STAGE,
        "script_name": SCRIPT_NAME,
        "kind": "stage14_direct_finetuning_summary",
        "backbone": backbone,
        "hf_model_id": spec.hf_model_id,
        "input_mode": INPUT_MODE,
        "input_order": "parents_oldest_to_newest_then_target",
        "context_order": "parent_4,parent_3,parent_2,parent_1,target",
        "dataset_sha256": audit["data"]["dataset_sha256"],
        "post_id_sha256": audit["data"]["post_id_sha256"],
        "fold_assignment_sha256": audit["data"]["fold_assignment_sha256"],
        "fold_column": eu.CTX_FOLD_COLUMN,
        "n_folds": len(FOLDS),
        "cv_seeds": list(SEEDS),
        "eval_seeds_exp1": list(eu.EVAL_SEEDS_EXP1),
        "split_seed": 42,
        "selected_context_size_c_best": int(decision["c_best_by_backbone"][backbone]),
        "stage11_manifest_fingerprint": audit["stage11"]["manifest_fingerprint"],
        "stage11_decision_sha256": _sha(audit["stage11"]["root_path"]),
        "stage8_manifest_fingerprint": audit["stage8"]["fingerprint"],
        "stage2_manifest_fingerprint": audit["data"]["stage2_manifest"]["fingerprint"],
        "e2_reference_grid_fingerprint": audit["e2_grid"]["grid_audit"]["grid_fingerprint"],
        "model_revision": audit["model_revisions"][backbone],
        "tokenizer_revision": audit["tokenizer_revisions"][backbone],
        "max_length_benchmark": int(sequence["max_length_benchmark"]),
        "max_length_context": int(sequence["max_length_context"]),
        "truncation_side": TRUNCATION_SIDE,
        "target_only_truncation_side": sequence["target_only_truncation_side"],
        "optimizer": str(hyper["optimizer"]),
        "weight_decay": float(hyper["weight_decay"]),
        "warmup_ratio": float(hyper["warmup_ratio"]),
        "lr_scheduler": str(hyper["lr_scheduler"]),
        "max_epochs": int(hyper["max_epochs"]),
        "early_stopping_patience": hyper["early_stopping_patience"],
        "loss": str(hyper["loss"]),
        "class_weighting_benchmark": "none_balanced_matched_dataset",
        "checkpoint_metric": str(hyper["checkpoint_metric"]),
        "lr_search_grid": list(eu.LR_SEARCH_GRID),
        "lr_tuning_seed": int(rates["lr_tuning_seed"]),
        "mixed_precision": bool(hyper["mixed_precision"]),
        "per_device_batch_size": int(hyper["per_device_train_batch_size"]),
        "gradient_accumulation_steps": int(hyper["gradient_accumulation_steps"]),
        "effective_batch_size": int(hyper["effective_batch_size"]),
        "learning_rate_benchmark": float(rates["learning_rate_benchmark"][backbone]),
        "learning_rate_context": float(rates["learning_rate_context"][backbone]),
        "model_initialization": "public_pretrained_base_without_ITFT",
        "itft_used": False,
    }
    eu.validate_resolved_run_config(
        config, stage=STAGE, backbone=backbone, input_mode=INPUT_MODE, require_shared_config=True,
    )
    return config


def _stage11_input_paths(bundle: Mapping[str, Any]) -> list[Path]:
    paths: list[Path] = [
        bundle["root_path"], bundle["copy_path"], bundle["manifest_path"], bundle["marker_path"], bundle["index_path"],
    ]
    index = bundle.get("index", {})
    for filename in (index.get("sha256") or {}):
        paths.append(bundle["stage_dir"] / str(filename))
    return paths


def _upstream_input_paths(audit: Mapping[str, Any], *, include_e2_runs: bool = True, root: Path = PROJECT_ROOT) -> list[Path]:
    data = audit["data"]
    stage8 = audit["stage8"]
    paths: list[Path] = [
        data["path"], data["stage2_manifest_path"],
        stage8["sequence_path"], stage8["learning_rates_path"], stage8["manifest_path"], stage8["marker_path"],
        stage8["index_path"], stage8["receipt_path"], stage8["latency_path"],
        *_stage11_input_paths(audit["stage11"]),
        audit["e2_grid"]["metrics_path"], audit["e2_grid"]["summary_path"],
    ]
    if include_e2_runs:
        # Bind every C0/C_best paired E2 reference run so Stage 15 can compare
        # direct and ITFT outputs on precisely the same folds/seeds/IDs.
        contexts = _unique_conditions(audit["c_best"])
        for condition in contexts:
            for fold in FOLDS:
                for seed in SEEDS:
                    run_dir = _e2_run_dir(audit["backbone"], condition, fold, seed, root=root)
                    paths.extend([
                        run_dir / eu.RUN_MANIFEST_NAME,
                        run_dir / "run_complete.json",
                        run_dir / "artifacts_sha256.json",
                        run_dir / "predictions.csv",
                        run_dir / "probabilities.npy",
                        run_dir / "metrics.json",
                    ])
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = Path(path).resolve()
        if resolved not in seen:
            if not resolved.is_file():
                raise FileNotFoundError(f"Stage 14 manifest input is missing: {resolved}")
            unique.append(resolved)
            seen.add(resolved)
    return unique


def _e2_run_dir(backbone: str, condition: int, fold: int, seed: int, *, root: Path = PROJECT_ROOT) -> Path:
    stage = 9 if backbone == "indobertweet" else 10
    return eu.versioned_run_root(root) / f"stage_{stage:02d}" / f"e2_{backbone}" / f"C{condition}" / f"fold_{fold:02d}" / f"seed_{seed}"


def _prepare_root_manifest(audit: Mapping[str, Any], root: Path = PROJECT_ROOT) -> tuple[Path, dict[str, Any]]:
    stage_root = eu.versioned_run_root(root) / "stage_14" / f"e3_{audit['backbone']}"
    config = _build_root_config(audit)
    manifest = eu.make_run_manifest(
        stage=STAGE,
        script_name=SCRIPT_NAME,
        config=config,
        input_paths=_upstream_input_paths(audit, root=root),
        metadata={
            "kind": "stage14_e3_direct_cv_summary",
            "selected_backbone": audit["backbone"],
            "selected_c_best": audit["c_best"],
            "contexts": list(_unique_conditions(audit["c_best"])),
            "folds": list(FOLDS),
            "seeds": list(SEEDS),
            "expected_runs": len(_unique_conditions(audit["c_best"])) * len(FOLDS) * len(SEEDS),
            "e2_reference_grid_fingerprint": audit["e2_grid"]["grid_audit"]["grid_fingerprint"],
            "checkpoint_selection": "fixed final epoch; held-out fold never used for selection",
        },
        project_root=root,
    )
    return stage_root, manifest


def _run_dir(backbone: str, condition: int, fold: int, seed: int, *, root: Path = PROJECT_ROOT) -> Path:
    return eu.versioned_run_root(root) / "stage_14" / f"e3_{backbone}" / f"C{condition}" / f"fold_{fold:02d}" / f"seed_{seed}"


def _prepare_run_manifest(
    *, backbone: str,
    condition: int,
    fold: int,
    seed: int,
    audit: Mapping[str, Any],
    stage_root: Path,
    root_manifest: Mapping[str, Any],
    root: Path = PROJECT_ROOT,
) -> tuple[Path, dict[str, Any]]:
    config = _build_run_config(backbone=backbone, condition=condition, fold=fold, seed=seed, audit=audit)
    input_paths = _upstream_input_paths(audit, root=root)
    config["stage14_root_manifest_fingerprint"] = str(root_manifest["fingerprint"])
    eu.validate_resolved_run_config(
        config, stage=STAGE, backbone=backbone, input_mode=INPUT_MODE,
        context_size=condition, require_shared_config=True,
    )
    manifest = eu.make_run_manifest(
        stage=STAGE,
        script_name=SCRIPT_NAME,
        config=config,
        input_paths=input_paths,
        metadata={
            "kind": "experiment_iii_direct_finetuning_oof_run",
            "condition": condition,
            "fold": fold,
            "seed": seed,
            "train_post_ids_sha256": eu.sha256_json(
                audit["data"]["matched"].loc[audit["data"]["matched"][eu.CTX_FOLD_COLUMN] != fold, eu.CTX_ID_COLUMN].astype(str).tolist()
            ),
            "held_out_post_ids_sha256": eu.sha256_json(
                audit["data"]["matched"].loc[audit["data"]["matched"][eu.CTX_FOLD_COLUMN] == fold, eu.CTX_ID_COLUMN].astype(str).tolist()
            ),
            "stage11_manifest_fingerprint": audit["stage11"]["manifest_fingerprint"],
            "stage8_manifest_fingerprint": audit["stage8"]["fingerprint"],
            "e2_reference_grid_fingerprint": audit["e2_grid"]["grid_audit"]["grid_fingerprint"],
            "initialization": "base pretrained checkpoint; no ITFT checkpoint loaded",
        },
        project_root=root,
    )
    return _run_dir(backbone, condition, fold, seed, root=root), manifest


def _validate_probability_array(probabilities: np.ndarray, n_rows: int, *, label: str) -> None:
    if probabilities.shape != (n_rows, len(eu.LABEL2ID)):
        raise ValueError(f"{label}: expected probability shape {(n_rows, len(eu.LABEL2ID))}, got {probabilities.shape}")
    if not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError(f"{label}: probabilities must be finite and non-negative")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-4):
        raise ValueError(f"{label}: each probability row must sum to 1")


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score

    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    class_f1 = f1_score(y_true, y_pred, labels=[0, 1, 2], average=None, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=[0, 1, 2], average="weighted", zero_division=0)),
        "f1_n": float(class_f1[0]),
        "f1_al": float(class_f1[1]),
        "f1_hs": float(class_f1[2]),
    }


def _verify_completed_run(
    run_dir: Path,
    *,
    manifest: Mapping[str, Any],
    backbone: str,
    condition: int,
    fold: int,
    seed: int,
    expected_post_ids: Sequence[Any],
    expected_y_true: np.ndarray,
) -> dict[str, Any] | None:
    marker_path = run_dir / "run_complete.json"
    output_names = {"predictions.csv", "probabilities.npy", "metrics.json", "artifacts_sha256.json"}
    files_except_manifest = [p for p in run_dir.iterdir() if p.is_file() and p.name != eu.RUN_MANIFEST_NAME] if run_dir.exists() else []
    if not marker_path.is_file():
        if files_except_manifest:
            raise RuntimeError(f"Incomplete Stage 14 outputs exist under {run_dir}; archive them rather than reusing them")
        return None
    unknown = {path.name for path in files_except_manifest} - output_names - {marker_path.name}
    if unknown:
        raise RuntimeError(f"Unexpected files exist in completed Stage 14 run {run_dir}: {sorted(unknown)}")
    required = [run_dir / name for name in output_names] + [marker_path, run_dir / eu.RUN_MANIFEST_NAME]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Stage 14 run is marked complete but lacks outputs: {missing}")

    fingerprint = str(manifest["fingerprint"])
    identity = {
        "schema_version": eu.CONFIG_SCHEMA_VERSION,
        "run_id": eu.RUN_ID,
        "stage": STAGE,
        "script_name": SCRIPT_NAME,
        "backbone": backbone,
        "context_size": condition,
        "fold": fold,
        "seed": seed,
        "manifest_fingerprint": fingerprint,
    }
    marker = _read_json(marker_path)
    for key, expected in identity.items():
        if str(marker.get(key)) != str(expected):
            raise ValueError(f"Stage 14 completion marker mismatch for {key} under {run_dir}")
    eu.assert_current_run_artifact(
        marker_path,
        expected_stage=STAGE,
        expected_script=SCRIPT_NAME,
        expected_fingerprint=fingerprint,
    )
    index_path = run_dir / "artifacts_sha256.json"
    index = _verify_index_entries(index_path, run_dir, expected_identity=identity)
    if marker.get("artifact_index_sha256") != _sha(index_path):
        raise ValueError(f"Stage 14 completion marker references a stale index: {index_path}")
    predictions_path = run_dir / "predictions.csv"
    probs_path = run_dir / "probabilities.npy"
    metrics_path = run_dir / "metrics.json"
    predictions_df = pd.read_csv(predictions_path)
    required_columns = {eu.CTX_ID_COLUMN, eu.CTX_FOLD_COLUMN, "context_size", "seed", "y_true", "y_pred", "confidence"}
    if not required_columns.issubset(predictions_df.columns):
        raise ValueError(f"Stage 14 predictions file is missing columns {sorted(required_columns - set(predictions_df.columns))}")
    ids = [str(value) for value in expected_post_ids]
    if predictions_df[eu.CTX_ID_COLUMN].astype(str).tolist() != ids:
        raise ValueError(f"Stage 14 post-ID order differs from the current held-out fold under {run_dir}")
    if not (pd.to_numeric(predictions_df[eu.CTX_FOLD_COLUMN], errors="raise").astype(int) == fold).all():
        raise ValueError(f"Stage 14 predictions contain incorrect fold values under {run_dir}")
    if not (pd.to_numeric(predictions_df["context_size"], errors="raise").astype(int) == condition).all():
        raise ValueError(f"Stage 14 predictions contain incorrect context size under {run_dir}")
    if not (pd.to_numeric(predictions_df["seed"], errors="raise").astype(int) == seed).all():
        raise ValueError(f"Stage 14 predictions contain incorrect seed under {run_dir}")
    y_stored = predictions_df["y_true"].map(eu.LABEL2ID)
    if y_stored.isna().any() or not np.array_equal(y_stored.astype(int).to_numpy(), np.asarray(expected_y_true, dtype=np.int64)):
        raise ValueError(f"Stage 14 ground-truth labels are not aligned with the current held-out fold under {run_dir}")
    y_pred_series = predictions_df["y_pred"].map(eu.LABEL2ID)
    if y_pred_series.isna().any():
        raise ValueError(f"Stage 14 predictions contain non-canonical labels under {run_dir}")
    probs = np.load(probs_path, allow_pickle=False)
    _validate_probability_array(probs, len(ids), label=str(probs_path))
    if not np.array_equal(probs.argmax(axis=1), y_pred_series.astype(int).to_numpy()):
        raise ValueError(f"Stage 14 y_pred differs from probabilities argmax under {run_dir}")
    metrics = _read_json(metrics_path)
    for key, expected in identity.items():
        if key in metrics and str(metrics.get(key)) != str(expected):
            raise ValueError(f"Stage 14 metrics identity mismatch for {key} under {run_dir}")
    if index.get("sha256", {}).get("predictions.csv") != _sha(predictions_path):
        raise ValueError(f"Stage 14 prediction hash mismatch under {run_dir}")
    if index.get("sha256", {}).get("probabilities.npy") != _sha(probs_path):
        raise ValueError(f"Stage 14 probability hash mismatch under {run_dir}")
    if index.get("sha256", {}).get("metrics.json") != _sha(metrics_path):
        raise ValueError(f"Stage 14 metrics hash mismatch under {run_dir}")
    return {"predictions": predictions_df, "probabilities": probs, "metrics": metrics, "manifest_fingerprint": fingerprint}


def _write_completed_run(
    *,
    run_dir: Path,
    manifest: Mapping[str, Any],
    backbone: str,
    condition: int,
    fold: int,
    seed: int,
    predictions: pd.DataFrame,
    probabilities: np.ndarray,
    metric_values: Mapping[str, Any],
) -> None:
    _write_csv_atomic(run_dir / "predictions.csv", predictions)
    _write_numpy_atomic(run_dir / "probabilities.npy", probabilities)
    metrics_doc = {
        "schema_version": eu.CONFIG_SCHEMA_VERSION,
        "run_id": eu.RUN_ID,
        "stage": STAGE,
        "script_name": SCRIPT_NAME,
        "backbone": backbone,
        "experiment_id": eu.experiment3_id(condition, backbone),
        "context_size": condition,
        "fold": fold,
        "seed": seed,
        "n_train": int(metric_values["n_train"]),
        "n_held_out": int(metric_values["n_held_out"]),
        "train_seconds": float(metric_values["train_seconds"]),
        "dataset_sha256": manifest["config"]["dataset_sha256"],
        "fold_assignment_sha256": manifest["config"]["fold_assignment_sha256"],
        "max_length_context": manifest["config"]["max_length_context"],
        "learning_rate_context": manifest["config"]["learning_rate_context"],
        "truncation_side": TRUNCATION_SIDE,
        "model_revision": manifest["config"]["model_revision"],
        "tokenizer_revision": manifest["config"]["tokenizer_revision"],
        "stage11_manifest_fingerprint": manifest["config"]["stage11_manifest_fingerprint"],
        "e2_reference_grid_fingerprint": manifest["config"]["e2_reference_grid_fingerprint"],
        "manifest_fingerprint": manifest["fingerprint"],
        "checkpoint_selection": "fixed final epoch; held-out fold not used for selection",
        **_metrics(np.asarray(metric_values["y_true"], dtype=np.int64), np.asarray(metric_values["y_pred"], dtype=np.int64)),
    }
    _write_json_atomic(run_dir / "metrics.json", metrics_doc)
    output_hashes = {name: _sha(run_dir / name) for name in ("predictions.csv", "probabilities.npy", "metrics.json")}
    _write_json_atomic(run_dir / "artifacts_sha256.json", {
        "schema_version": eu.CONFIG_SCHEMA_VERSION,
        "run_id": eu.RUN_ID,
        "stage": STAGE,
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
        "stage": STAGE,
        "script_name": SCRIPT_NAME,
        "backbone": backbone,
        "context_size": condition,
        "fold": fold,
        "seed": seed,
        "manifest_fingerprint": manifest["fingerprint"],
        "artifact_index_sha256": _sha(run_dir / "artifacts_sha256.json"),
        "post_id_sha256": eu.sha256_json(predictions[eu.CTX_ID_COLUMN].astype(str).tolist()),
    })


def _make_training_args(TrainingArguments, *, output_dir: Path, config: Mapping[str, Any], use_cuda: bool, disable_tqdm: bool):
    kwargs = {
        "output_dir": str(output_dir),
        "num_train_epochs": int(config["max_epochs"]),
        "per_device_train_batch_size": int(config["per_device_train_batch_size"]),
        "per_device_eval_batch_size": int(config["per_device_eval_batch_size"]),
        "gradient_accumulation_steps": int(config["gradient_accumulation_steps"]),
        "learning_rate": float(config["learning_rate_context"]),
        "weight_decay": float(config["weight_decay"]),
        "warmup_ratio": float(config["warmup_ratio"]),
        "lr_scheduler_type": str(config["lr_scheduler"]),
        "eval_strategy": "no",
        "save_strategy": "no",
        "logging_steps": 50,
        "seed": int(config["seed"]),
        "data_seed": int(config["seed"]),
        "report_to": "none",
        "fp16": bool(use_cuda),
        "disable_tqdm": bool(disable_tqdm),
        "dataloader_num_workers": 0,
        "remove_unused_columns": True,
    }
    return TrainingArguments(**kwargs)


def _load_training_dependencies():
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, set_seed
    except ImportError as exc:
        raise ImportError("Stage 14 training requires PyTorch and Hugging Face Transformers; use the project's GPU/Colab environment") from exc
    return torch, AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, set_seed


def _validate_prerequisites(root: Path = PROJECT_ROOT) -> dict[str, Any]:
    os.chdir(root)
    _assert_no_legacy_outputs(root)
    data = _load_and_validate_data(root)
    stage8 = _verify_stage8(root)
    stage11 = _verify_stage11_decision(root)
    decision = stage11["decision"]
    backbone = str(decision["decision"])
    c_best_map = decision.get("c_best_by_backbone")
    if not isinstance(c_best_map, Mapping) or set(c_best_map) != set(eu.BACKBONES):
        raise ValueError("Stage 11 decision must provide C_best for both registered backbones")
    c_best = int(c_best_map[backbone])
    if c_best not in ALL_CONDITIONS:
        raise ValueError(f"Stage 11 selected invalid C_best=C{c_best}")
    if decision.get("post_id_sha256") != data["post_id_sha256"]:
        raise ValueError("Stage 11 OOF post-ID hash differs from the current canonical matched-subset order")
    if int(decision.get("n_oof_examples", -1)) != EXPECTED_ROWS:
        raise ValueError("Stage 11 decision does not describe the 900-example matched OOF set")

    preflight = stage8["context_preflight"]
    hyper = stage8["hyperparameters"]
    sequence = stage8["sequence"]
    rates = stage8["learning_rates"]
    fold_hash = data["fold_assignment_sha256"]
    if preflight.get("fold_assignment_sha256") != fold_hash:
        raise ValueError("Stage 8 contextual LR search used a different shared fold assignment")
    if sequence.get("context_length_fold_assignment_sha256") != fold_hash:
        raise ValueError("Stage 8 contextual length selection used a different fold assignment")
    if preflight.get("context_dataset_sha256") not in (None, data["dataset_sha256"]):
        raise ValueError("Stage 8 contextual LR search used a different matched dataset hash")
    if preflight.get("fold_assignment_sha256") != fold_hash:
        raise ValueError("Stage 8 preflight fold-assignment hash differs from Stage 2")

    model_revisions = preflight.get("model_revisions")
    tokenizer_revisions = sequence.get("context_tokenizer_resolved_revisions")
    expected_model_ids = {key: eu.get_backbone(key).hf_model_id for key in eu.BACKBONES}
    if preflight.get("model_ids") != expected_model_ids:
        raise ValueError("Stage 8 contextual preflight model IDs differ from the utility registry")
    if not isinstance(model_revisions, Mapping) or set(model_revisions) != set(eu.BACKBONES):
        raise ValueError("Stage 8 preflight must pin model revisions for both backbones")
    if not isinstance(tokenizer_revisions, Mapping) or set(tokenizer_revisions) != set(eu.BACKBONES):
        raise ValueError("Stage 8 sequence config must pin tokenizer revisions for both backbones")
    for key in eu.BACKBONES:
        model_revision = str(model_revisions[key]).lower()
        tokenizer_revision = str(tokenizer_revisions[key]).lower()
        if not re.fullmatch(r"[0-9a-f]{40}", model_revision) or not re.fullmatch(r"[0-9a-f]{40}", tokenizer_revision):
            raise ValueError(f"Stage 8 must pin full 40-character model/tokenizer commit SHAs for {key}")
        if model_revision != tokenizer_revision:
            raise ValueError(f"Model/tokenizer revision mismatch for {key}: {model_revision} vs {tokenizer_revision}")
    learning_rates = rates.get("learning_rate_context")
    if not isinstance(learning_rates, Mapping) or set(learning_rates) != set(eu.BACKBONES):
        raise ValueError("Stage 8 must select a contextual learning rate for both backbones")

    e2_grid = _verify_e2_grid(backbone=backbone, decision_bundle=stage11, stage8=stage8, data=data, root=root)
    audit = {
        "backbone": backbone,
        "c_best": c_best,
        "conditions": _unique_conditions(c_best),
        "data": data,
        "stage8": stage8,
        "stage11": stage11,
        "e2_grid": e2_grid,
        "model_revisions": {key: str(value).lower() for key, value in model_revisions.items()},
        "tokenizer_revisions": {key: str(value).lower() for key, value in tokenizer_revisions.items()},
        "sequence": sequence,
        "learning_rates": rates,
        "hyperparameters": dict(hyper),
        "mixed_precision": bool(hyper["mixed_precision"]),
    }

    # Cross-check direct fine-tuning against the exact selected-backbone ITFT
    # config at matching C0/fold/seed. Only initialization strategy should differ.
    direct_cfg = _build_run_config(backbone=backbone, condition=0, fold=0, seed=SEEDS[0], audit=audit)
    e2_run_dir = _e2_run_dir(backbone, 0, 0, SEEDS[0], root=root)
    e2_manifest = _read_json(e2_run_dir / eu.RUN_MANIFEST_NAME)
    e2_cfg = e2_manifest.get("config")
    if not isinstance(e2_cfg, Mapping):
        raise ValueError("Selected E2 C0 reference run does not contain a resolved config")
    eu.check_config_parity(e2_cfg, direct_cfg, input_mode=INPUT_MODE)

    stage_root, root_manifest = _prepare_root_manifest(audit, root=root)
    audit["stage_root"] = stage_root
    audit["root_manifest"] = root_manifest
    audit["run_manifests"] = {}
    for condition in audit["conditions"]:
        for fold in FOLDS:
            for seed in SEEDS:
                run_dir, manifest = _prepare_run_manifest(
                    backbone=backbone, condition=condition, fold=fold, seed=seed,
                    audit=audit, stage_root=stage_root, root_manifest=root_manifest, root=root,
                )
                audit["run_manifests"][(condition, fold, seed)] = (run_dir, manifest)
    return audit


def _training_dataset(encodings: Mapping[str, Any], labels: np.ndarray):
    torch = __import__("torch")

    class TokenizedTextDataset(torch.utils.data.Dataset):
        def __init__(self, encoding, y):
            self.encoding = encoding
            self.labels = np.asarray(y, dtype=np.int64)

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            item = {key: value[idx] for key, value in self.encoding.items()}
            item["labels"] = torch.tensor(int(self.labels[idx]), dtype=torch.long)
            return item

    return TokenizedTextDataset(encodings, labels)


def _tokenize_rows(tokenizer, dataframe: pd.DataFrame, condition: int, max_length: int) -> tuple[dict[str, Any], np.ndarray]:
    texts = dataframe.apply(
        lambda row: eu.build_context_text(row, condition, sep=eu.sep_from_tokenizer(tokenizer)),
        axis=1,
    ).tolist()
    if tokenizer.truncation_side != TRUNCATION_SIDE:
        raise RuntimeError(f"Contextual tokenizer must use left truncation; got {tokenizer.truncation_side!r}")
    encodings = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )
    labels = dataframe["label"].map(eu.LABEL2ID)
    if labels.isna().any():
        raise ValueError("Non-canonical labels encountered while tokenizing Stage 14 data")
    return dict(encodings), labels.astype(int).to_numpy(dtype=np.int64)


def _train_run(
    *,
    torch,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    set_seed,
    tokenizer,
    audit: Mapping[str, Any],
    condition: int,
    fold: int,
    seed: int,
    run_dir: Path,
    manifest: Mapping[str, Any],
    use_cuda: bool,
    disable_tqdm: bool,
) -> dict[str, Any]:
    data = audit["data"]
    matched = data["matched"]
    train_df = matched.loc[matched[eu.CTX_FOLD_COLUMN] != fold].reset_index(drop=True)
    held_out_df = matched.loc[matched[eu.CTX_FOLD_COLUMN] == fold].reset_index(drop=True)
    if len(train_df) != EXPECTED_ROWS - EXPECTED_FOLD_COUNTS[fold] or len(held_out_df) != EXPECTED_FOLD_COUNTS[fold]:
        raise ValueError(f"C{condition}/fold {fold}: expected 720 train and 180 held-out examples")

    train_enc, y_train = _tokenize_rows(tokenizer, train_df, condition, int(audit["sequence"]["max_length_context"]))
    held_enc, y_hold = _tokenize_rows(tokenizer, held_out_df, condition, int(audit["sequence"]["max_length_context"]))
    train_dataset = _training_dataset(train_enc, y_train)
    held_out_dataset = _training_dataset(held_enc, y_hold)

    spec = eu.get_backbone(audit["backbone"])
    set_seed(seed)
    start = time.perf_counter()
    model = AutoModelForSequenceClassification.from_pretrained(
        spec.hf_model_id,
        revision=audit["model_revisions"][audit["backbone"]],
        num_labels=len(eu.LABEL2ID),
        id2label=eu.ID2LABEL,
        label2id=eu.LABEL2ID,
    )
    if getattr(model.config, "num_labels", None) != len(eu.LABEL2ID):
        raise RuntimeError("Loaded base checkpoint does not expose the expected three-class classification head")
    resolved_model_revision = getattr(model.config, "_commit_hash", None)
    if not resolved_model_revision or str(resolved_model_revision).lower() != audit["model_revisions"][audit["backbone"]]:
        raise RuntimeError(f"Base model resolved to revision {resolved_model_revision!r}; expected {audit['model_revisions'][audit['backbone']]!r}")
    # The selected model was loaded directly from the pinned public base model;
    # no Stage 6/7 ITFT checkpoint is passed to from_pretrained().
    with tempfile.TemporaryDirectory(prefix=f"e3-{audit['backbone']}-", dir=PROJECT_ROOT) as temp_dir:
        training_args = _make_training_args(
            TrainingArguments,
            output_dir=Path(temp_dir) / "trainer_output",
            config=manifest["config"],
            use_cuda=use_cuda,
            disable_tqdm=disable_tqdm,
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
        )
        trainer.train()
        output = trainer.predict(held_out_dataset)
        elapsed = time.perf_counter() - start
        logits = np.asarray(output.predictions)
        if logits.ndim != 2 or logits.shape != (len(held_out_df), len(eu.LABEL2ID)):
            raise RuntimeError(f"Unexpected logits shape for Stage 14 C{condition}/fold {fold}/seed {seed}: {logits.shape}")
        probs = torch.softmax(torch.as_tensor(logits), dim=1).detach().cpu().numpy()
        y_pred = probs.argmax(axis=1).astype(np.int64)
        y_true = y_hold
        values = {
            "n_train": len(train_df),
            "n_held_out": len(held_out_df),
            "train_seconds": float(elapsed),
            "y_true": y_true,
            "y_pred": y_pred,
        }
        predictions = pd.DataFrame({
            eu.CTX_ID_COLUMN: held_out_df[eu.CTX_ID_COLUMN].astype(str).to_numpy(),
            eu.CTX_FOLD_COLUMN: held_out_df[eu.CTX_FOLD_COLUMN].astype(int).to_numpy(),
            "backbone": audit["backbone"],
            "context_size": int(condition),
            "seed": int(seed),
            "y_true": [eu.ID2LABEL[int(value)] for value in y_true],
            "y_pred": [eu.ID2LABEL[int(value)] for value in y_pred],
            "confidence": probs.max(axis=1),
        })
        _write_completed_run(
            run_dir=run_dir,
            manifest=manifest,
            backbone=audit["backbone"],
            condition=condition,
            fold=fold,
            seed=seed,
            predictions=predictions,
            probabilities=probs,
            metric_values=values,
        )
        del trainer, model, train_dataset, held_out_dataset

    checked = _verify_completed_run(
        run_dir,
        manifest=manifest,
        backbone=audit["backbone"],
        condition=condition,
        fold=fold,
        seed=seed,
        expected_post_ids=held_out_df[eu.CTX_ID_COLUMN].astype(str).tolist(),
        expected_y_true=y_hold,
    )
    if checked is None:
        raise RuntimeError(f"Newly trained Stage 14 run failed completion verification: {run_dir}")
    if use_cuda:
        torch.cuda.empty_cache()
    return checked["metrics"]


def _reconstruct_summaries(audit: Mapping[str, Any]) -> dict[str, pd.DataFrame]:
    grid_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    backbone = audit["backbone"]
    for condition in audit["conditions"]:
        for fold in FOLDS:
            held_out = audit["data"]["matched"].loc[audit["data"]["matched"][eu.CTX_FOLD_COLUMN] == fold].reset_index(drop=True)
            y_true_expected = held_out["label"].map(eu.LABEL2ID).astype(int).to_numpy(dtype=np.int64)
            for seed in SEEDS:
                run_dir, manifest = audit["run_manifests"][(condition, fold, seed)]
                checked = _verify_completed_run(
                    run_dir,
                    manifest=manifest,
                    backbone=backbone,
                    condition=condition,
                    fold=fold,
                    seed=seed,
                    expected_post_ids=held_out[eu.CTX_ID_COLUMN].astype(str).tolist(),
                    expected_y_true=y_true_expected,
                )
                if checked is None:
                    raise FileNotFoundError(f"Missing complete Stage 14 run: {run_dir}")
                predictions = checked["predictions"]
                y_true = predictions["y_true"].map(eu.LABEL2ID).astype(int).to_numpy()
                y_pred = predictions["y_pred"].map(eu.LABEL2ID).astype(int).to_numpy()
                probs = checked["probabilities"]
                metrics = _metrics(y_true, y_pred)
                metric_doc = checked["metrics"]
                grid_rows.append({
                    "schema_version": eu.CONFIG_SCHEMA_VERSION,
                    "run_id": eu.RUN_ID,
                    "stage": STAGE,
                    "script_name": SCRIPT_NAME,
                    "backbone": backbone,
                    "experiment_id": eu.experiment3_id(condition, backbone),
                    "condition": int(condition),
                    "fold": int(fold),
                    "seed": int(seed),
                    "n_train": int(metric_doc["n_train"]),
                    "n_held_out": int(metric_doc["n_held_out"]),
                    **metrics,
                    "train_seconds": float(metric_doc["train_seconds"]),
                    "max_length_context": int(metric_doc["max_length_context"]),
                    "learning_rate_context": float(metric_doc["learning_rate_context"]),
                    "truncation_side": TRUNCATION_SIDE,
                    "manifest_fingerprint": manifest["fingerprint"],
                    "stage11_manifest_fingerprint": audit["stage11"]["manifest_fingerprint"],
                    "e2_reference_grid_fingerprint": audit["e2_grid"]["grid_audit"]["grid_fingerprint"],
                })
                from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
                precision, recall, f1, support = precision_recall_fscore_support(
                    y_true, y_pred, labels=[0, 1, 2], average=None, zero_division=0,
                )
                for class_id in range(len(eu.LABEL2ID)):
                    class_rows.append({
                        "backbone": backbone,
                        "condition": condition,
                        "fold": fold,
                        "seed": seed,
                        "class_id": class_id,
                        "class_label": eu.ID2LABEL[class_id],
                        "precision": float(precision[class_id]),
                        "recall": float(recall[class_id]),
                        "f1": float(f1[class_id]),
                        "support": int(support[class_id]),
                    })
                matrix = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
                for true_id in range(3):
                    for pred_id in range(3):
                        confusion_rows.append({
                            "backbone": backbone,
                            "condition": condition,
                            "fold": fold,
                            "seed": seed,
                            "true_class_id": true_id,
                            "true_class": eu.ID2LABEL[true_id],
                            "predicted_class_id": pred_id,
                            "predicted_class": eu.ID2LABEL[pred_id],
                            "count": int(matrix[true_id, pred_id]),
                        })

    grid = pd.DataFrame(grid_rows).sort_values(["condition", "fold", "seed"]).reset_index(drop=True)
    expected_n = len(audit["conditions"]) * len(FOLDS) * len(SEEDS)
    if len(grid) != expected_n or grid.duplicated(["condition", "fold", "seed"]).any():
        raise ValueError(f"Stage 14 grid expected {expected_n} unique condition/fold/seed rows; got {len(grid)}")
    summary_rows: list[dict[str, Any]] = []
    for condition, subset in grid.groupby("condition", sort=True):
        summary_rows.append({
            "backbone": backbone,
            "condition": int(condition),
            "experiment_id": eu.experiment3_id(int(condition), backbone),
            "n_runs": int(len(subset)),
            "accuracy_mean": float(subset["accuracy"].mean()),
            "accuracy_std": float(subset["accuracy"].std(ddof=1)) if len(subset) > 1 else 0.0,
            "macro_f1_mean": float(subset["macro_f1"].mean()),
            "macro_f1_std": float(subset["macro_f1"].std(ddof=1)) if len(subset) > 1 else 0.0,
            "weighted_f1_mean": float(subset["weighted_f1"].mean()),
            "weighted_f1_std": float(subset["weighted_f1"].std(ddof=1)) if len(subset) > 1 else 0.0,
            "is_c0": int(condition) == BASE_CONDITION,
            "is_c_best": int(condition) == int(audit["c_best"]),
            "selected_context_size_c_best": int(audit["c_best"]),
            "n_unique_conditions": len(audit["conditions"]),
        })
    return {
        "e3_grid_metrics.csv": grid,
        "e3_context_summary.csv": pd.DataFrame(summary_rows).sort_values("condition").reset_index(drop=True),
        "e3_grid_class_metrics.csv": pd.DataFrame(class_rows).sort_values(["condition", "fold", "seed", "class_id"]).reset_index(drop=True),
        "e3_confusion_matrices.csv": pd.DataFrame(confusion_rows).sort_values(["condition", "fold", "seed", "true_class_id", "predicted_class_id"]).reset_index(drop=True),
    }


def _verify_stage_complete(stage_root: Path, root_manifest: Mapping[str, Any], audit: Mapping[str, Any], root: Path = PROJECT_ROOT) -> bool:
    marker_path = stage_root / "stage14_complete.json"
    index_path = stage_root / "artifacts_sha256.json"
    if not marker_path.is_file():
        if index_path.exists():
            raise RuntimeError("Stage 14 artifact index exists without a completion marker; archive this incomplete stage output")
        return False
    if not index_path.is_file():
        raise FileNotFoundError("Stage 14 completion marker exists but artifacts_sha256.json is missing")
    fingerprint = str(root_manifest["fingerprint"])
    eu.assert_current_run_artifact(marker_path, expected_stage=STAGE, expected_script=SCRIPT_NAME, expected_fingerprint=fingerprint)
    marker = _read_json(marker_path)
    if marker.get("manifest_fingerprint") != fingerprint or marker.get("artifact_index_sha256") != _sha(index_path):
        raise ValueError("Stage 14 completion marker has stale manifest/index hashes")
    identity = {
        "schema_version": eu.CONFIG_SCHEMA_VERSION,
        "run_id": eu.RUN_ID,
        "stage": STAGE,
        "script_name": SCRIPT_NAME,
        "backbone": audit["backbone"],
        "selected_context_size_c_best": audit["c_best"],
        "manifest_fingerprint": fingerprint,
    }
    index = _verify_index_entries(index_path, stage_root, expected_identity=identity)
    for filename in SUMMARY_FILES:
        if not (stage_root / filename).is_file():
            raise FileNotFoundError(f"Stage 14 summary is missing: {stage_root / filename}")
    expected_n = len(audit["conditions"]) * len(FOLDS) * len(SEEDS)
    if int(marker.get("n_runs", -1)) != expected_n or int(index.get("n_runs", -1)) != expected_n:
        raise ValueError("Stage 14 completion marker/index have an incorrect run count")
    return True


def _publish_summaries(stage_root: Path, root_manifest: Mapping[str, Any], audit: Mapping[str, Any], tables: Mapping[str, pd.DataFrame], root: Path = PROJECT_ROOT) -> None:
    for name in SUMMARY_FILES:
        if name == "e3_manifest.json":
            continue
        _write_csv_atomic(stage_root / name, tables[name])
    metrics = tables["e3_grid_metrics.csv"]
    e3_manifest = {
        "schema_version": eu.CONFIG_SCHEMA_VERSION,
        "run_id": eu.RUN_ID,
        "stage": STAGE,
        "script_name": SCRIPT_NAME,
        "backbone": audit["backbone"],
        "hf_model_id": eu.get_backbone(audit["backbone"]).hf_model_id,
        "model_revision": audit["model_revisions"][audit["backbone"]],
        "tokenizer_revision": audit["tokenizer_revisions"][audit["backbone"]],
        "stage_manifest_fingerprint": root_manifest["fingerprint"],
        "stage11_manifest_fingerprint": audit["stage11"]["manifest_fingerprint"],
        "stage8_manifest_fingerprint": audit["stage8"]["fingerprint"],
        "stage2_manifest_fingerprint": audit["data"]["stage2_manifest"]["fingerprint"],
        "e2_reference_grid_fingerprint": audit["e2_grid"]["grid_audit"]["grid_fingerprint"],
        "dataset_sha256": audit["data"]["dataset_sha256"],
        "post_id_sha256": audit["data"]["post_id_sha256"],
        "fold_assignment_sha256": audit["data"]["fold_assignment_sha256"],
        "selected_context_size_c_best": audit["c_best"],
        "conditions": list(audit["conditions"]),
        "folds": list(FOLDS),
        "seeds": list(SEEDS),
        "n_runs": int(len(metrics)),
        "expected_runs": int(len(audit["conditions"]) * len(FOLDS) * len(SEEDS)),
        "max_length_context": int(audit["sequence"]["max_length_context"]),
        "learning_rate_context": float(audit["learning_rates"]["learning_rate_context"][audit["backbone"]]),
        "truncation_side": TRUNCATION_SIDE,
        "training_strategy": "direct_fine_tuning_without_ITFT",
        "checkpoint_selection": "fixed final epoch; held-out folds not used for selection",
        "c_best_equals_c0": bool(audit["c_best"] == 0),
    }
    _write_json_atomic(stage_root / "e3_manifest.json", e3_manifest)

    # The root index includes all summary files and every per-run artefact plus
    # its manifest/marker/index. Individual run indexes remain authoritative
    # for the three data files; this root index makes the complete stage easy to audit.
    hashes: dict[str, str] = {}
    for name in SUMMARY_FILES:
        path = stage_root / name
        hashes[name] = _sha(path)
    for condition in audit["conditions"]:
        for fold in FOLDS:
            for seed in SEEDS:
                run_dir = _run_dir(audit["backbone"], condition, fold, seed, root=root)
                for name in (eu.RUN_MANIFEST_NAME, "predictions.csv", "probabilities.npy", "metrics.json", "artifacts_sha256.json", "run_complete.json"):
                    path = run_dir / name
                    if not path.is_file():
                        raise FileNotFoundError(f"Cannot publish Stage 14; run artifact is missing: {path}")
                    hashes[path.relative_to(stage_root).as_posix()] = _sha(path)
    index_doc = {
        "schema_version": eu.CONFIG_SCHEMA_VERSION,
        "run_id": eu.RUN_ID,
        "stage": STAGE,
        "script_name": SCRIPT_NAME,
        "backbone": audit["backbone"],
        "selected_context_size_c_best": audit["c_best"],
        "manifest_fingerprint": root_manifest["fingerprint"],
        "n_runs": int(len(metrics)),
        "expected_runs": int(len(metrics)),
        "sha256": hashes,
    }
    index_path = stage_root / "artifacts_sha256.json"
    _write_json_atomic(index_path, index_doc)
    marker_doc = {
        "schema_version": eu.CONFIG_SCHEMA_VERSION,
        "run_id": eu.RUN_ID,
        "stage": STAGE,
        "script_name": SCRIPT_NAME,
        "backbone": audit["backbone"],
        "selected_backbone": audit["backbone"],
        "selected_context_size_c_best": audit["c_best"],
        "conditions": list(audit["conditions"]),
        "n_runs": int(len(metrics)),
        "expected_runs": int(len(metrics)),
        "manifest_fingerprint": root_manifest["fingerprint"],
        "artifact_index_sha256": _sha(index_path),
        "completed_utc": pd.Timestamp.utcnow().isoformat(),
    }
    marker_path = stage_root / "stage14_complete.json"
    _write_json_atomic(marker_path, marker_doc)
    eu.assert_current_run_artifact(marker_path, expected_stage=STAGE, expected_script=SCRIPT_NAME, expected_fingerprint=str(root_manifest["fingerprint"]))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 14: direct fine-tuning for the Stage 11-selected backbone, C0 and C_best.")
    parser.add_argument("--validate-only", action="store_true", help="Validate upstream manifests/configuration/artifacts without loading model weights, training, or writing outputs.")
    parser.add_argument("--allow-cpu", action="store_true", help="Allow CPU training only when Stage 8 also used mixed_precision=false.")
    parser.add_argument("--disable-tqdm", action="store_true", help="Disable Hugging Face Trainer progress bars.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    os.chdir(PROJECT_ROOT)
    migration = eu.audit_pipeline_script_names(PROJECT_ROOT, require_all=False)
    if SCRIPT_NAME not in migration["present"]:
        raise FileNotFoundError(f"Authoritative Stage 14 script is missing from {PROJECT_ROOT / SCRIPT_NAME}")

    audit = _validate_prerequisites(PROJECT_ROOT)
    stage_root: Path = audit["stage_root"]
    root_manifest: dict[str, Any] = audit["root_manifest"]
    print("Stage 14 prerequisite audit passed.")
    print(f"  Selected backbone: {audit['backbone']} ({eu.get_backbone(audit['backbone']).hf_model_id})")
    print(f"  Selected C_best: C{audit['c_best']}; direct conditions: {list(audit['conditions'])}")
    print(f"  Runs: {len(audit['conditions']) * len(FOLDS) * len(SEEDS)}")
    print(f"  max_length_context: {audit['sequence']['max_length_context']}")
    print(f"  learning_rate_context: {audit['learning_rates']['learning_rate_context'][audit['backbone']]:g}")
    print(f"  Model revision: {audit['model_revisions'][audit['backbone']]}")
    print(f"  Tokenizer revision: {audit['tokenizer_revisions'][audit['backbone']]}")
    print(f"  Output root: {stage_root}")
    if args.validate_only:
        print("Validation-only complete; no model weights were loaded and no Stage 14 outputs were written.")
        if migration["complete"]:
            print("Pipeline filename/directory audit: passed.")
        else:
            print("Pipeline audit is still incomplete; actual training remains blocked until all stage scripts/directories are migrated.")
        return 0

    # Never launch a GPU sweep while any authoritative downstream stage name
    # or required extension directory is missing from the frozen pipeline.
    eu.audit_pipeline_script_names(PROJECT_ROOT, require_all=True)
    torch, AutoModel, AutoTokenizer, Trainer, TrainingArguments, set_seed = _load_training_dependencies()
    use_cuda = bool(torch.cuda.is_available())
    if not use_cuda and not args.allow_cpu:
        raise RuntimeError("Stage 14 requires a CUDA GPU for the clean experiment run; use a GPU-enabled Colab runtime.")
    if use_cuda != bool(audit["mixed_precision"]):
        raise RuntimeError(
            "The active CUDA/mixed-precision environment differs from Stage 8's fixed contextual tuning environment: "
            f"Stage 8 mixed_precision={audit['mixed_precision']}, current CUDA={use_cuda}. Use the matching runtime or rerun Stage 8."
        )

    # Verify existing root manifest before it may be used for resume.
    eu.prepare_run_directory(stage_root, root_manifest, allow_resume=True)
    if _verify_stage_complete(stage_root, root_manifest, audit, root=PROJECT_ROOT):
        print("Stage 14 is already complete and its full output index matches the current upstream configuration.")
        return 0

    spec = eu.get_backbone(audit["backbone"])
    tokenizer_revision = audit["tokenizer_revisions"][audit["backbone"]]
    tokenizer = AutoTokenizer.from_pretrained(spec.hf_model_id, revision=tokenizer_revision, use_fast=True)
    eu.configure_tokenizer_for_input(tokenizer, INPUT_MODE)
    resolved_tokenizer_revision = getattr(tokenizer, "_commit_hash", None) or (getattr(tokenizer, "init_kwargs", {}) or {}).get("_commit_hash")
    if not resolved_tokenizer_revision or str(resolved_tokenizer_revision).lower() != tokenizer_revision:
        raise RuntimeError(f"Tokenizer resolved to {resolved_tokenizer_revision!r}; expected pinned revision {tokenizer_revision!r}")

    run_total = len(audit["conditions"]) * len(FOLDS) * len(SEEDS)
    run_count = 0
    for condition in audit["conditions"]:
        # Tokenize once for each (condition, fold), then reuse the encodings for
        # both seeds. Separate training/held-out splits preserve the shared fold.
        for fold in FOLDS:
            train_df = audit["data"]["matched"].loc[audit["data"]["matched"][eu.CTX_FOLD_COLUMN] != fold].reset_index(drop=True)
            held_df = audit["data"]["matched"].loc[audit["data"]["matched"][eu.CTX_FOLD_COLUMN] == fold].reset_index(drop=True)
            # Fail before training if the configured fold does not supply valid C-level inputs.
            for frame in (train_df.iloc[:1], held_df.iloc[:1]):
                if not frame.empty:
                    eu.build_context_text(frame.iloc[0], condition, sep=eu.sep_from_tokenizer(tokenizer))
            del train_df, held_df
            for seed in SEEDS:
                run_count += 1
                run_dir, manifest = audit["run_manifests"][(condition, fold, seed)]
                eu.prepare_run_directory(run_dir, manifest, allow_resume=True)
                held = audit["data"]["matched"].loc[audit["data"]["matched"][eu.CTX_FOLD_COLUMN] == fold].reset_index(drop=True)
                y_hold = held["label"].map(eu.LABEL2ID).astype(int).to_numpy(dtype=np.int64)
                existing = _verify_completed_run(
                    run_dir, manifest=manifest, backbone=audit["backbone"], condition=condition, fold=fold, seed=seed,
                    expected_post_ids=held[eu.CTX_ID_COLUMN].astype(str).tolist(), expected_y_true=y_hold,
                )
                if existing is not None:
                    print(f"[{run_count}/{run_total}] C{condition}/fold {fold}/seed {seed}: verified completed run, resume without retraining")
                    continue
                print(f"\n[{run_count}/{run_total}] Training E3 C{condition}/fold {fold}/seed {seed} ({audit['backbone']}; no ITFT)")
                metrics = _train_run(
                    torch=torch,
                    AutoModelForSequenceClassification=AutoModel,
                    Trainer=Trainer,
                    TrainingArguments=TrainingArguments,
                    set_seed=set_seed,
                    tokenizer=tokenizer,
                    audit=audit,
                    condition=condition,
                    fold=fold,
                    seed=seed,
                    run_dir=run_dir,
                    manifest=manifest,
                    use_cuda=use_cuda,
                    disable_tqdm=args.disable_tqdm,
                )
                print(f"  macro-F1={metrics['macro_f1']:.4f}; completed in {metrics['train_seconds']/60:.2f} min")

    tables = _reconstruct_summaries(audit)
    _publish_summaries(stage_root, root_manifest, audit, tables, root=PROJECT_ROOT)
    _verify_stage_complete(stage_root, root_manifest, audit, root=PROJECT_ROOT)
    print("\nStage 14 complete. Direct fine-tuning used the Stage 11-selected backbone and context size.")
    print(tables["e3_context_summary.csv"].to_string(index=False))
    print(f"Verified outputs: {stage_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
