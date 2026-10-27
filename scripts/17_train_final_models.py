"""Stage 17: train the final target-only and context-aware deployment models.

The two final models are trained on all 1,800 rows in
``data/processed/ctx_cleaned.csv``.  Their initialization strategy is read from
Stage 15's provenance-verified ``results/training_strategy_decision.json``:
``itft`` starts from the selected backbone's validated Stage 6/7 checkpoint,
whereas ``direct`` starts from the pinned public pretrained checkpoint.

Stage 11 decides the backbone and C_best; this script never reselects either.
Each deployment mode uses its own Stage 15 strategy decision, and no validation
split or checkpoint selection is performed on the final 1,800-row training set.
The context-aware model uses up to C_best available parents per example, using
fewer parents for shallower threads.  In all cases, parent messages precede the
target and contextual inputs use explicitly configured left truncation.

Run in the project repository, normally in a GPU-enabled Colab runtime:
    python scripts/17_train_final_models.py --validate-only
    python scripts/17_train_final_models.py

Outputs:
    models/deployment/<backbone>/target_only/
    models/deployment/<backbone>/context_aware_C<n>/
    results/runs/left_truncation_v2/stage_17/final_<backbone>/

The source-level script and its two study documents must be frozen before the
clean pipeline run. Old/unmanifested artifacts are never accepted as resume
signals; completed outputs are reusable only when all manifests and hashes
match the current inputs and code.
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

STAGE = 17
SCRIPT_NAME = "scripts/17_train_final_models.py"
INPUT_MODE = "contextual"
FINAL_TRAIN_SEED = 42
EXPECTED_ROWS = int(eu.CONTEXT_RAW_EXPECTED_ROWS)
CLEANED_DATA_PATH = Path("data/processed/ctx_cleaned.csv")
STAGE2_DIR = eu.versioned_run_root(PROJECT_ROOT) / "stage_02"
STAGE8_DIR = eu.versioned_run_root(PROJECT_ROOT) / "stage_08" / "summary"
STAGE11_DIR = eu.versioned_run_root(PROJECT_ROOT) / "stage_11" / "decision"
STAGE15_DIR = eu.versioned_run_root(PROJECT_ROOT) / "stage_15"
STRATEGY_DECISION_PATH = PROJECT_ROOT / eu.TRAINING_STRATEGY_DECISION_PATH
STAGE_ROOT = eu.versioned_run_root(PROJECT_ROOT) / "stage_17"
LEGACY_OUTPUT_PATTERNS = (
    "models/final_model.pt",
    "models/final_target_only",
    "models/final_context_aware",
    "models/E3_final*",
    "results/raw/final_*_predictions.csv",
    "results/raw/final_*_probabilities.npy",
)
STAGE8_EXPECTED_HYPERPARAMETERS = {
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


def _sha(path: str | Path) -> str:
    return eu.sha256_file(path)


def _sha_tree(directory: Path, *, excluded_names: Sequence[str] = ()) -> dict[str, str]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Checkpoint directory is missing: {directory}")
    excluded = set(excluded_names)
    hashes = {
        path.relative_to(directory).as_posix(): _sha(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.name not in excluded
    }
    if not hashes:
        raise RuntimeError(f"No checkpoint files were saved under {directory}")
    return hashes


def _resolve(path: str | Path, root: Path = PROJECT_ROOT) -> Path:
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()


def _unique_existing(paths: Sequence[Path]) -> list[Path]:
    output: list[Path] = []
    seen: set[Path] = set()
    for raw in paths:
        path = Path(raw).resolve()
        if path in seen:
            continue
        if not path.is_file():
            raise FileNotFoundError(f"Required provenance input is missing: {path}")
        output.append(path)
        seen.add(path)
    return output


def _assert_no_legacy_outputs(root: Path = PROJECT_ROOT) -> None:
    found: list[Path] = []
    for pattern in LEGACY_OUTPUT_PATTERNS:
        found.extend(path for path in root.glob(pattern) if path.exists())
    eu.assert_no_legacy_outputs(found)


def _verify_stage8(root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Verify Stage 8 configuration pins, manifest, publication marker and index."""
    stage_dir = _resolve(STAGE8_DIR, root)
    manifest_path = stage_dir / eu.RUN_MANIFEST_NAME
    marker_path = stage_dir / "stage8_complete.json"
    index_path = stage_dir / "artifacts_sha256.json"
    receipt_path = stage_dir / "stage8_publish_receipt.json"
    latency_path = stage_dir / "context_latency_measurements.json"
    sequence_path = _resolve(eu.SEQUENCE_LENGTH_CONFIG, root)
    rates_path = _resolve(eu.LEARNING_RATE_CONFIG, root)
    required = (manifest_path, marker_path, index_path, receipt_path, latency_path, sequence_path, rates_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Stage 8 publication bundle is incomplete: " + ", ".join(missing))

    manifest = _read_json(manifest_path)
    eu._validate_manifest_integrity(manifest)
    fingerprint = str(manifest.get("fingerprint", ""))
    if str(manifest.get("stage")) != "8" or manifest.get("script_name") != eu.PIPELINE_SCRIPTS[8]:
        raise ValueError("Stage 8 manifest has an invalid stage/script identity")
    eu.assert_current_run_artifact(
        marker_path, expected_stage=8, expected_script=eu.PIPELINE_SCRIPTS[8], expected_fingerprint=fingerprint
    )
    marker = _read_json(marker_path)
    index = _read_json(index_path)
    receipt = _read_json(receipt_path)
    if marker.get("manifest_fingerprint") != fingerprint or marker.get("artifact_index_sha256") != _sha(index_path):
        raise ValueError("Stage 8 completion marker has stale manifest/index hashes")
    expected_index_identity = {
        "schema_version": eu.CONFIG_SCHEMA_VERSION,
        "run_id": eu.RUN_ID,
        "stage": 8,
        "manifest_fingerprint": fingerprint,
    }
    for key, expected in expected_index_identity.items():
        if str(index.get(key)) != str(expected):
            raise ValueError(f"Stage 8 artifact index identity mismatch for {key}")
    indexed_hashes = index.get("sha256")
    if not isinstance(indexed_hashes, Mapping):
        raise ValueError("Stage 8 artifact index has no SHA-256 inventory")
    for relative_path, expected_hash in indexed_hashes.items():
        path = stage_dir / str(relative_path)
        if not path.is_file() or _sha(path) != expected_hash:
            raise ValueError(f"Missing or stale Stage 8 indexed artifact: {path}")
    if marker.get("sequence_length_config_sha256") != _sha(sequence_path):
        raise ValueError("Stage 8 sequence-length config changed after publication")
    if marker.get("learning_rate_config_sha256") != _sha(rates_path):
        raise ValueError("Stage 8 learning-rate config changed after publication")
    if receipt.get("summary_manifest_fingerprint") != fingerprint:
        raise ValueError("Stage 8 publication receipt references another manifest")
    if receipt.get("latency_measurements_sha256") != _sha(latency_path):
        raise ValueError("Stage 8 latency measurements do not match their publication receipt")
    if receipt.get("sequence_length_config_sha256") != _sha(sequence_path) or receipt.get("learning_rate_config_sha256") != _sha(rates_path):
        raise ValueError("Stage 8 publication receipt has stale shared-configuration hashes")

    shared = eu.load_shared_run_config(
        sequence_length_path=sequence_path,
        learning_rate_path=rates_path,
        require_context=True,
        require_context_lr=True,
    )
    sequence, rates = shared["sequence_length"], shared["learning_rates"]
    latency = _read_json(latency_path)
    if latency.get("schema_version") != eu.CONFIG_SCHEMA_VERSION or latency.get("run_id") != eu.RUN_ID:
        raise ValueError("Stage 8 latency measurements use a stale schema/run ID")
    if latency.get("manifest_fingerprint") != fingerprint or receipt.get("latency_measurements_sha256") != _sha(latency_path):
        raise ValueError("Stage 8 latency measurements do not match the published Stage 8 manifest")
    expected_ids = {key: eu.get_backbone(key).hf_model_id for key in eu.BACKBONES}
    if latency.get("model_ids") != expected_ids:
        raise ValueError("Stage 8 model IDs differ from experiment_utils.BACKBONES")
    if latency.get("context_sizes") != list(range(5)) or latency.get("truncation_side") != "left":
        raise ValueError("Stage 8 must record C0-C4 latency and explicit contextual left truncation")

    revisions = rates.get("context_preflight", {}).get("model_revisions")
    tokenizer_revisions = sequence.get("context_tokenizer_resolved_revisions")
    if not isinstance(revisions, Mapping) or set(revisions) != set(eu.BACKBONES):
        raise ValueError("Stage 8 does not pin model revisions for both registered backbones")
    if not isinstance(tokenizer_revisions, Mapping) or set(tokenizer_revisions) != set(eu.BACKBONES):
        raise ValueError("Stage 8 does not pin tokenizer revisions for both registered backbones")
    for backbone in eu.BACKBONES:
        model_rev = str(revisions[backbone]).lower()
        tok_rev = str(tokenizer_revisions[backbone]).lower()
        if not re.fullmatch(r"[0-9a-f]{40}", model_rev) or not re.fullmatch(r"[0-9a-f]{40}", tok_rev):
            raise ValueError(f"Stage 8 must pin full commit SHAs for {backbone}")
        if model_rev != tok_rev:
            raise ValueError(f"Stage 8 model/tokenizer revision mismatch for {backbone}")

    hyper = rates.get("context_preflight", {}).get("hyperparameters")
    if not isinstance(hyper, Mapping):
        raise ValueError("Stage 8 contextual preflight is missing fixed training hyperparameters")
    mismatches = {
        key: (hyper.get(key), expected)
        for key, expected in STAGE8_EXPECTED_HYPERPARAMETERS.items()
        if hyper.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"Stage 8 hyperparameters differ from the frozen Experiment II/III protocol: {mismatches}")
    if not isinstance(hyper.get("mixed_precision"), bool):
        raise ValueError("Stage 8 hyperparameters must record mixed_precision as a boolean")
    lr_map = rates.get("learning_rate_context")
    if not isinstance(lr_map, Mapping) or set(lr_map) != set(eu.BACKBONES):
        raise ValueError("Stage 8 is missing the selected contextual LR for both backbones")
    for backbone, rate in lr_map.items():
        if not math.isfinite(float(rate)) or float(rate) <= 0:
            raise ValueError(f"Invalid contextual learning rate for {backbone}: {rate}")

    return {
        "stage_dir": stage_dir,
        "manifest_path": manifest_path,
        "marker_path": marker_path,
        "index_path": index_path,
        "receipt_path": receipt_path,
        "latency_path": latency_path,
        "manifest": manifest,
        "manifest_fingerprint": fingerprint,
        "sequence_path": sequence_path,
        "learning_rates_path": rates_path,
        "sequence_config": sequence,
        "learning_rate_config": rates,
        "latency": latency,
        "hyperparameters": dict(hyper),
        "model_revisions": {key: str(value).lower() for key, value in revisions.items()},
        "tokenizer_revisions": {key: str(value).lower() for key, value in tokenizer_revisions.items()},
    }


def _verify_stage11(root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Verify Stage 11 decision files; Stage 15 loader also checks this chain."""
    root_decision = _resolve(eu.BACKBONE_DECISION_PATH, root)
    stage_dir = _resolve(STAGE11_DIR, root)
    manifest_path = stage_dir / eu.RUN_MANIFEST_NAME
    marker_path = stage_dir / "stage11_complete.json"
    index_path = stage_dir / "artifacts_sha256.json"
    copy_path = stage_dir / "backbone_decision.json"
    required = (root_decision, manifest_path, marker_path, index_path, copy_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Stage 11 decision bundle is incomplete: " + ", ".join(missing))
    manifest = _read_json(manifest_path)
    eu._validate_manifest_integrity(manifest)
    fingerprint = str(manifest.get("fingerprint", ""))
    if str(manifest.get("stage")) != "11" or manifest.get("script_name") != eu.PIPELINE_SCRIPTS[11]:
        raise ValueError("Stage 11 manifest has an invalid stage/script identity")
    eu.assert_current_run_artifact(
        marker_path, expected_stage=11, expected_script=eu.PIPELINE_SCRIPTS[11], expected_fingerprint=fingerprint
    )
    marker = _read_json(marker_path)
    index = _read_json(index_path)
    if marker.get("manifest_fingerprint") != fingerprint or marker.get("artifact_index_sha256") != _sha(index_path):
        raise ValueError("Stage 11 completion marker has stale manifest/index hashes")
    if marker.get("root_decision_sha256") != _sha(root_decision) or _sha(copy_path) != _sha(root_decision):
        raise ValueError("Stage 11 root and versioned decision files are not identical/current")
    expected = {
        "schema_version": eu.CONFIG_SCHEMA_VERSION,
        "run_id": eu.RUN_ID,
        "stage": 11,
        "script_name": eu.PIPELINE_SCRIPTS[11],
        "manifest_fingerprint": fingerprint,
    }
    for key, value in expected.items():
        if str(index.get(key)) != str(value):
            raise ValueError(f"Stage 11 artifact index identity mismatch for {key}")
    hashes = index.get("sha256")
    if not isinstance(hashes, Mapping):
        raise ValueError("Stage 11 artifact index has no SHA-256 map")
    for name, digest in hashes.items():
        path = stage_dir / str(name)
        if not path.is_file() or _sha(path) != digest:
            raise ValueError(f"Missing or stale Stage 11 indexed artifact: {path}")
    decision = eu.load_backbone_decision_document(root_decision)
    if decision.get("stage11_manifest_fingerprint") != fingerprint:
        raise ValueError("Stage 11 decision points to a different manifest")
    if decision.get("stage11_output_dir") != stage_dir.relative_to(root).as_posix():
        raise ValueError("Stage 11 decision points to an unexpected directory")
    if marker.get("selected_backbone") != decision.get("decision") or marker.get("c_best_by_backbone") != decision.get("c_best_by_backbone"):
        raise ValueError("Stage 11 completion marker and decision JSON disagree")
    return {
        "decision": decision,
        "manifest": manifest,
        "manifest_fingerprint": fingerprint,
        "root_path": root_decision,
        "manifest_path": manifest_path,
        "marker_path": marker_path,
        "index_path": index_path,
        "copy_path": copy_path,
        "stage_dir": stage_dir,
        "index": index,
    }


def _verify_itft_checkpoint(backbone: str, stage8: Mapping[str, Any], root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Verify the selected backbone's Stage 6/7 checkpoint only when ITFT is selected."""
    stage = 6 if backbone == "indobertweet" else 7
    stage_dir = eu.versioned_run_root(root) / f"stage_{stage:02d}" / f"itft_{backbone}"
    model_dir = root / "models" / "itft_checkpoint" / backbone
    manifest_path = stage_dir / eu.RUN_MANIFEST_NAME
    marker_path = stage_dir / "checkpoint_complete.json"
    index_path = stage_dir / "artifacts_sha256.json"
    metrics_path = stage_dir / "metrics.json"
    model_manifest_path = model_dir / "itft_checkpoint_manifest.json"
    required = (manifest_path, marker_path, index_path, metrics_path, model_manifest_path, model_dir / "config.json")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Selected strategy requires a complete Stage {stage} ITFT checkpoint: {missing}")
    run_manifest = _read_json(manifest_path)
    eu._validate_manifest_integrity(run_manifest)
    fingerprint = str(run_manifest.get("fingerprint", ""))
    eu.assert_current_run_artifact(
        marker_path, expected_stage=stage, expected_script=eu.PIPELINE_SCRIPTS[stage], expected_fingerprint=fingerprint
    )
    marker, index, metrics = _read_json(marker_path), _read_json(index_path), _read_json(metrics_path)
    checkpoint = _read_json(model_manifest_path)
    for document_name, document in (("marker", marker), ("index", index), ("metrics", metrics), ("checkpoint manifest", checkpoint)):
        for key, expected in (("stage", stage), ("backbone", backbone), ("manifest_fingerprint", fingerprint)):
            if str(document.get(key)) != str(expected):
                raise ValueError(f"Stage {stage} ITFT {document_name} mismatch for {key}")
    if marker.get("run_id") != eu.RUN_ID or index.get("run_id") != eu.RUN_ID:
        raise ValueError(f"Stage {stage} ITFT checkpoint uses an incompatible run ID")
    if index.get("sha256", {}).get("metrics.json") != _sha(metrics_path):
        raise ValueError(f"Stage {stage} ITFT metrics hash mismatch")
    if index.get("sha256", {}).get("checkpoint_manifest_sha256") != _sha(model_manifest_path):
        raise ValueError(f"Stage {stage} ITFT model-manifest hash mismatch")
    if marker.get("checkpoint_manifest_sha256") != _sha(model_manifest_path):
        raise ValueError(f"Stage {stage} ITFT marker model-manifest hash mismatch")
    expected_weights = checkpoint.get("checkpoint_files_sha256")
    if not isinstance(expected_weights, Mapping) or dict(expected_weights) != _sha_tree(model_dir, excluded_names=("itft_checkpoint_manifest.json",)):
        raise ValueError(f"Stage {stage} ITFT checkpoint files differ from their manifest")
    spec = eu.get_backbone(backbone)
    if checkpoint.get("hf_model_id") != spec.hf_model_id:
        raise ValueError(f"Stage {stage} ITFT model ID differs from the registered backbone")
    if str(checkpoint.get("model_revision", "")).lower() != stage8["model_revisions"][backbone]:
        raise ValueError(f"Stage {stage} ITFT model revision differs from Stage 8")
    if str(checkpoint.get("tokenizer_revision", "")).lower() != stage8["tokenizer_revisions"][backbone]:
        raise ValueError(f"Stage {stage} ITFT tokenizer revision differs from Stage 8")
    expected_itft_fp = stage8["learning_rate_config"].get("context_preflight", {}).get("itft_manifest_fingerprints", {}).get(backbone)
    if expected_itft_fp != fingerprint:
        raise ValueError(f"Stage 8 contextual preflight did not use this Stage {stage} ITFT checkpoint")
    # The benchmark training inputs must still match the producing ITFT manifest.
    source_hashes = run_manifest.get("input_sha256", {})
    for name in ("data/processed/ib_train.csv", "data/processed/ib_val.csv", "results/tables/ib_class_weights.csv"):
        path = _resolve(name, root)
        if not path.is_file() or source_hashes.get(name) != _sha(path):
            raise ValueError(f"Stage {stage} ITFT source input changed after training: {path}")
    return {
        "stage": stage,
        "backbone": backbone,
        "run_dir": stage_dir,
        "model_dir": model_dir,
        "manifest_path": manifest_path,
        "marker_path": marker_path,
        "index_path": index_path,
        "metrics_path": metrics_path,
        "model_manifest_path": model_manifest_path,
        "manifest": run_manifest,
        "manifest_fingerprint": fingerprint,
        "checkpoint_manifest": checkpoint,
        "model_revision": str(checkpoint["model_revision"]).lower(),
        "tokenizer_revision": str(checkpoint["tokenizer_revision"]).lower(),
        "index": index,
    }


def _load_and_validate_data(root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Verify Stage 2 outputs and select only the modelling allowlist for 1,800 rows."""
    data_path = _resolve(CLEANED_DATA_PATH, root)
    if not data_path.is_file():
        raise FileNotFoundError(f"Missing Stage 2 full contextual dataset: {data_path}")
    stage2_manifest = eu.assert_current_stage_manifest(2, root)
    raw = pd.read_csv(data_path)
    audit = eu.validate_context_source_dataset(raw.copy(), expected_rows=EXPECTED_ROWS)
    modelling = eu.load_modelling_view(raw, include_fold=False)
    modelling[eu.CTX_ID_COLUMN] = modelling[eu.CTX_ID_COLUMN].astype(str)
    modelling = modelling.sort_values(eu.CTX_ID_COLUMN, kind="mergesort").reset_index(drop=True)
    if len(modelling) != EXPECTED_ROWS:
        raise ValueError(f"Stage 17 requires exactly {EXPECTED_ROWS} contextual posts; found {len(modelling)}")
    if modelling[eu.CTX_ID_COLUMN].duplicated().any():
        raise ValueError("Stage 17 modelling view contains duplicate post_id values")
    labels = modelling["label"].map(eu.LABEL2ID)
    if labels.isna().any():
        raise ValueError("Stage 17 data contains non-canonical labels")
    ids = modelling[eu.CTX_ID_COLUMN].astype(str).tolist()
    ids_hash = eu.sha256_json(sorted(ids))
    # The Stage 2 source-level hash uses all source IDs, independent of CSV row order.
    expected_ids_hash = stage2_manifest.get("config", {}).get("post_id_sha256")
    if expected_ids_hash and ids_hash != expected_ids_hash:
        raise ValueError("Stage 17 full-dataset post-ID hash differs from Stage 2's source audit")
    return {
        "path": data_path,
        "raw": raw,
        "modelling": modelling,
        "stage2_manifest": stage2_manifest,
        "dataset_sha256": _sha(data_path),
        "post_ids": ids,
        "post_id_sha256": eu.sha256_json(ids),
        "post_id_set_sha256": ids_hash,
        "labels": labels.astype(int).to_numpy(dtype=np.int64),
        "label_counts": {eu.ID2LABEL[int(k)]: int(v) for k, v in pd.Series(labels.astype(int)).value_counts().sort_index().items()},
        "availability_counts": {str(int(k)): int(v) for k, v in pd.to_numeric(modelling["context_size_available"]).astype(int).value_counts().sort_index().items()},
        "audit": audit,
    }


def _upstream_input_paths(
    *, data: Mapping[str, Any], stage8: Mapping[str, Any], stage11: Mapping[str, Any],
    strategy_doc: Mapping[str, Any], itft: Mapping[str, Any] | None, root: Path = PROJECT_ROOT,
) -> list[Path]:
    paths: list[Path] = [
        data["path"],
        root / Path(*eu.RUN_ROOT_PARTS) / "stage_02" / eu.RUN_MANIFEST_NAME,
        stage8["manifest_path"], stage8["marker_path"], stage8["index_path"], stage8["receipt_path"],
        stage8["latency_path"], stage8["sequence_path"], stage8["learning_rates_path"],
        stage11["root_path"], stage11["copy_path"], stage11["manifest_path"], stage11["marker_path"], stage11["index_path"],
        STAGE15_DIR / eu.RUN_MANIFEST_NAME,
        STAGE15_DIR / "stage15_complete.json",
        STAGE15_DIR / "artifacts_sha256.json",
        STAGE15_DIR / "training_strategy_decision.json",
        STRATEGY_DECISION_PATH,
    ]
    # Include every indexed Stage 11 decision artifact; context-selection and grid-audit
    # files make the chosen C_best and selected backbone auditable downstream.
    for name in (stage11.get("index", {}).get("sha256") or {}):
        paths.append(stage11["stage_dir"] / str(name))
    if itft is not None:
        paths.extend([
            itft["manifest_path"], itft["marker_path"], itft["index_path"], itft["metrics_path"],
            itft["model_manifest_path"],
        ])
        for name in (itft.get("index", {}).get("sha256") or {}):
            # Stage 6/7's model file inventory is split between the stage run folder and checkpoint directory.
            candidate = itft["run_dir"] / str(name)
            if candidate.is_file():
                paths.append(candidate)
        for path in itft["model_dir"].rglob("*"):
            if path.is_file():
                paths.append(path)
    return _unique_existing(paths)


def _model_directory(backbone: str, mode: str, context_size: int, root: Path = PROJECT_ROOT) -> Path:
    if mode == "target_only":
        return root / "models" / "deployment" / backbone / "target_only"
    return root / "models" / "deployment" / backbone / f"context_aware_C{context_size}"


def _mode_run_directory(stage_root: Path, mode: str, context_size: int) -> Path:
    return stage_root / ("target_only" if mode == "target_only" else f"context_aware_C{context_size}")


def _effective_context_size(row: pd.Series, requested_context_size: int) -> int:
    """Use up to the selected number of parents, but never demand unavailable parents."""
    available_raw = row.get("context_size_available")
    if pd.isna(available_raw):
        raise ValueError(f"Missing context_size_available for post_id={row.get(eu.CTX_ID_COLUMN)!r}")
    available = int(available_raw)
    if available not in (1, 2, 3, 4):
        raise ValueError(f"Invalid parent availability {available!r} for post_id={row.get(eu.CTX_ID_COLUMN)!r}")
    return min(int(requested_context_size), available)


def build_input_texts(dataframe: pd.DataFrame, requested_context_size: int, tokenizer) -> tuple[list[str], list[int]]:
    """Build context-first text using the tokenizer's own separator and partial-context policy."""
    if requested_context_size not in range(5):
        raise ValueError("requested_context_size must be an integer from 0 through 4")
    separator = eu.sep_from_tokenizer(tokenizer)
    effective_sizes: list[int] = []
    texts: list[str] = []
    for _, row in dataframe.iterrows():
        used_context = 0 if requested_context_size == 0 else _effective_context_size(row, requested_context_size)
        # build_context_text is the shared canonical constructor; passing the row's
        # effective availability avoids a false error for shallow threads.
        texts.append(eu.build_context_text(row, used_context, sep=separator))
        effective_sizes.append(used_context)
    if len(texts) != len(dataframe) or any(not text.strip() for text in texts):
        raise ValueError("Tokenized input text count/contents do not match the full contextual dataset")
    return texts, effective_sizes


def _load_strategy_and_decisions(root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Resolve Stage 11 and Stage 15 decisions; never allow a local model/strategy override."""
    os.chdir(root)
    stage11 = _verify_stage11(root)
    strategy_doc = eu.load_training_strategy_decision(
        _resolve(eu.TRAINING_STRATEGY_DECISION_PATH, root),
        verify_provenance=True,
        project_root=root,
    )
    # Stage 15's verifier independently validates the same Stage 11 source manifest;
    # require byte-identical selected decision/context in both provenance chains.
    stage15_copy = _read_json(_resolve(STAGE15_DIR / "training_strategy_decision.json", root))
    if strategy_doc.get("by_mode") != stage15_copy.get("by_mode"):
        raise ValueError("Canonical and versioned Stage 15 strategy decisions disagree")
    if strategy_doc.get("provenance", {}).get("stage11_manifest_fingerprint") != stage11["manifest_fingerprint"]:
        raise ValueError("Stage 15 strategy decision and Stage 11 manifest fingerprints disagree")
    selected_backbone = str(stage11["decision"]["decision"])
    c_best = int(stage11["decision"]["c_best_by_backbone"][selected_backbone])
    by_mode = strategy_doc.get("by_mode")
    if not isinstance(by_mode, Mapping) or set(by_mode) != {"target_only", "context_aware"}:
        raise ValueError("Stage 15 must select one strategy for target_only and one for context_aware")
    strategies = {mode: str(by_mode[mode]["strategy"]) for mode in ("target_only", "context_aware")}
    if any(strategy not in {"itft", "direct"} for strategy in strategies.values()):
        raise ValueError(f"Invalid Stage 15 strategy selection: {strategies}")
    return {
        "stage11": stage11,
        "strategy_document": strategy_doc,
        "backbone": selected_backbone,
        "c_best": c_best,
        "strategies": strategies,
    }


def _validate_prerequisites(root: Path = PROJECT_ROOT) -> dict[str, Any]:
    os.chdir(root)
    _assert_no_legacy_outputs(root)
    data = _load_and_validate_data(root)
    stage8 = _verify_stage8(root)
    decisions = _load_strategy_and_decisions(root)
    backbone = decisions["backbone"]
    c_best = int(decisions["c_best"])
    if c_best not in range(5):
        raise ValueError(f"Stage 11 selected invalid C_best=C{c_best}")
    if stage8["sequence_config"].get("context_length_fold_assignment_sha256") != data["stage2_manifest"].get("config", {}).get("fold_assignment_sha256"):
        raise ValueError("Stage 8 context length and Stage 2 shared fold assignment provenance disagree")
    if stage8["learning_rate_config"].get("context_preflight", {}).get("fold_assignment_sha256") != data["stage2_manifest"].get("config", {}).get("fold_assignment_sha256"):
        raise ValueError("Stage 8 contextual LR tuning and Stage 2 shared fold assignment provenance disagree")
    if stage8["learning_rate_config"].get("context_preflight", {}).get("context_dataset_sha256") not in (None, _sha(_resolve(Path("data/processed/ctx_matched_subset.csv"), root))):
        raise ValueError("Stage 8 contextual LR tuning used a different matched contextual dataset")
    if stage8["learning_rate_config"].get("context_preflight", {}).get("model_ids") != {key: eu.get_backbone(key).hf_model_id for key in eu.BACKBONES}:
        raise ValueError("Stage 8 model IDs differ from the current backbone registry")

    # Stage 15 provenance makes the source selected backbone and Stage 11 C_best
    # authoritative. Do not infer a context size from summaries at this stage.
    selected_itft = None
    if "itft" in decisions["strategies"].values():
        selected_itft = _verify_itft_checkpoint(backbone, stage8, root)
    rates = stage8["learning_rate_config"]
    sequence = stage8["sequence_config"]
    hyper = stage8["hyperparameters"]
    if float(rates["learning_rate_context"][backbone]) <= 0:
        raise ValueError("Selected backbone contextual learning rate is invalid")
    if int(sequence["max_length_context"]) < 8:
        raise ValueError("Stage 8 contextual maximum sequence length is invalid")

    run_root = eu.versioned_run_root(root) / "stage_17" / f"final_{backbone}"
    return {
        **decisions,
        "data": data,
        "stage8": stage8,
        "itft": selected_itft,
        "hyperparameters": hyper,
        "sequence": sequence,
        "learning_rates": rates,
        "stage_root": run_root,
    }


def _stage_root_config(audit: Mapping[str, Any]) -> dict[str, Any]:
    backbone = str(audit["backbone"])
    spec = eu.get_backbone(backbone)
    return {
        "schema_version": eu.CONFIG_SCHEMA_VERSION,
        "run_id": eu.RUN_ID,
        "stage": STAGE,
        "script_name": SCRIPT_NAME,
        "kind": "stage17_final_deployment_models",
        "backbone": backbone,
        "hf_model_id": spec.hf_model_id,
        "selected_context_size_c_best": int(audit["c_best"]),
        "strategy_by_mode": dict(audit["strategies"]),
        "training_dataset": CLEANED_DATA_PATH.as_posix(),
        "dataset_sha256": audit["data"]["dataset_sha256"],
        "post_id_set_sha256": audit["data"]["post_id_set_sha256"],
        "n_training_examples": EXPECTED_ROWS,
        "label_counts": dict(audit["data"]["label_counts"]),
        "availability_counts": dict(audit["data"]["availability_counts"]),
        "stage2_manifest_fingerprint": audit["data"]["stage2_manifest"]["fingerprint"],
        "stage8_manifest_fingerprint": audit["stage8"]["manifest_fingerprint"],
        "stage11_manifest_fingerprint": audit["stage11"]["manifest_fingerprint"],
        "stage15_manifest_fingerprint": audit["strategy_document"]["provenance"]["stage15_manifest_fingerprint"],
        "max_length_benchmark": int(audit["sequence"]["max_length_benchmark"]),
        "max_length_context": int(audit["sequence"]["max_length_context"]),
        "learning_rate_benchmark": float(audit["learning_rates"]["learning_rate_benchmark"][backbone]),
        "truncation_side": "left",
        "input_mode": INPUT_MODE,
        "input_order": "available_parents_oldest_to_newest_then_target",
        "context_availability_policy": "use_min(C_best, context_size_available)_parents_per_row",
        "optimizer": str(audit["hyperparameters"]["optimizer"]),
        "weight_decay": float(audit["hyperparameters"]["weight_decay"]),
        "warmup_ratio": float(audit["hyperparameters"]["warmup_ratio"]),
        "lr_scheduler": str(audit["hyperparameters"]["lr_scheduler"]),
        "num_train_epochs": int(audit["hyperparameters"]["max_epochs"]),
        "per_device_train_batch_size": int(audit["hyperparameters"]["per_device_train_batch_size"]),
        "per_device_eval_batch_size": int(audit["hyperparameters"]["per_device_eval_batch_size"]),
        "gradient_accumulation_steps": int(audit["hyperparameters"]["gradient_accumulation_steps"]),
        "learning_rate_context": float(audit["learning_rates"]["learning_rate_context"][backbone]),
        "seed": FINAL_TRAIN_SEED,
        "validation_used": False,
        "checkpoint_selection": "none_fixed_epochs_full_1800_examples",
    }


def _stage_manifest_inputs(audit: Mapping[str, Any], root: Path = PROJECT_ROOT) -> list[Path]:
    itft = audit["itft"]
    return _upstream_input_paths(
        data=audit["data"], stage8=audit["stage8"], stage11=audit["stage11"],
        strategy_doc=audit["strategy_document"], itft=itft, root=root,
    )


def _prepare_stage_manifest(audit: Mapping[str, Any], root: Path = PROJECT_ROOT) -> tuple[Path, dict[str, Any]]:
    stage_root = audit["stage_root"]
    config = _stage_root_config(audit)
    manifest = eu.make_run_manifest(
        stage=STAGE,
        script_name=SCRIPT_NAME,
        config=config,
        input_paths=_stage_manifest_inputs(audit, root),
        metadata={
            "kind": "stage17_final_deployment_model_pair",
            "selected_backbone": audit["backbone"],
            "selected_context_size_c_best": int(audit["c_best"]),
            "strategy_by_mode": dict(audit["strategies"]),
            "n_training_examples": EXPECTED_ROWS,
            "checkpoint_selection": "none_fixed_epochs_full_1800_examples",
        },
        project_root=root,
    )
    return stage_root, manifest


def _mode_config(audit: Mapping[str, Any], mode: str, context_size: int, stage_root_manifest: Mapping[str, Any]) -> dict[str, Any]:
    backbone = str(audit["backbone"])
    spec = eu.get_backbone(backbone)
    strategy = str(audit["strategies"][mode])
    itft = audit["itft"] if strategy == "itft" else None
    source_model_revision = str(itft["model_revision"] if itft else audit["stage8"]["model_revisions"][backbone]).lower()
    source_tokenizer_revision = str(itft["tokenizer_revision"] if itft else audit["stage8"]["tokenizer_revisions"][backbone]).lower()
    config = {
        "schema_version": eu.CONFIG_SCHEMA_VERSION,
        "run_id": eu.RUN_ID,
        "stage": STAGE,
        "script_name": SCRIPT_NAME,
        "kind": "stage17_final_model",
        "backbone": backbone,
        "hf_model_id": spec.hf_model_id,
        "deployment_mode": mode,
        "context_size": int(context_size),
        "selected_context_size_c_best": int(audit["c_best"]),
        "selected_strategy": strategy,
        "model_initialization": "validated_stage6_7_itft_checkpoint" if strategy == "itft" else "pinned_public_pretrained_checkpoint",
        "itft_used": strategy == "itft",
        "itft_manifest_fingerprint": str(itft["manifest_fingerprint"]) if itft else None,
        "model_revision": source_model_revision,
        "tokenizer_revision": source_tokenizer_revision,
        "dataset_sha256": audit["data"]["dataset_sha256"],
        "post_id_sha256": audit["data"]["post_id_sha256"],
        "n_training_examples": EXPECTED_ROWS,
        "label_counts": dict(audit["data"]["label_counts"]),
        "input_mode": INPUT_MODE,
        "input_order": "available_parents_oldest_to_newest_then_target",
        "separator_token_policy": "tokenizer_sep_token",
        "context_availability_policy": "use_min(requested_context_size, context_size_available)_parents_per_row",
        "max_length_benchmark": int(audit["sequence"]["max_length_benchmark"]),
        "max_length_context": int(audit["sequence"]["max_length_context"]),
        "learning_rate_benchmark": float(audit["learning_rates"]["learning_rate_benchmark"][backbone]),
        "truncation_side": "left",
        "learning_rate_context": float(audit["learning_rates"]["learning_rate_context"][backbone]),
        "num_train_epochs": int(audit["hyperparameters"]["max_epochs"]),
        "per_device_train_batch_size": int(audit["hyperparameters"]["per_device_train_batch_size"]),
        "per_device_eval_batch_size": int(audit["hyperparameters"]["per_device_eval_batch_size"]),
        "gradient_accumulation_steps": int(audit["hyperparameters"]["gradient_accumulation_steps"]),
        "effective_batch_size": int(audit["hyperparameters"]["effective_batch_size"]),
        "optimizer": str(audit["hyperparameters"]["optimizer"]),
        "weight_decay": float(audit["hyperparameters"]["weight_decay"]),
        "warmup_ratio": float(audit["hyperparameters"]["warmup_ratio"]),
        "lr_scheduler": str(audit["hyperparameters"]["lr_scheduler"]),
        "loss": str(audit["hyperparameters"]["loss"]),
        "class_weighting": "none_balanced_1800_dataset",
        "seed": FINAL_TRAIN_SEED,
        "validation_used": False,
        "checkpoint_selection": "none_fixed_epochs_full_1800_examples",
        "stage17_root_manifest_fingerprint": str(stage_root_manifest["fingerprint"]),
        "stage2_manifest_fingerprint": audit["data"]["stage2_manifest"]["fingerprint"],
        "stage8_manifest_fingerprint": audit["stage8"]["manifest_fingerprint"],
        "stage11_manifest_fingerprint": audit["stage11"]["manifest_fingerprint"],
        "stage15_manifest_fingerprint": audit["strategy_document"]["provenance"]["stage15_manifest_fingerprint"],
        "matched_fold_assignment_reference_sha256": audit["data"]["stage2_manifest"].get("config", {}).get("fold_assignment_sha256"),
    }
    eu.validate_resolved_run_config(
        config, stage=STAGE, backbone=backbone, input_mode=INPUT_MODE,
        context_size=context_size, require_shared_config=True,
    )
    return config


def _mode_manifest(
    *, audit: Mapping[str, Any], mode: str, context_size: int,
    stage_root_manifest: Mapping[str, Any], upstream_paths: Sequence[Path], root: Path = PROJECT_ROOT,
) -> tuple[Path, dict[str, Any]]:
    run_dir = _mode_run_directory(audit["stage_root"], mode, context_size)
    config = _mode_config(audit, mode, context_size, stage_root_manifest)
    paths = list(upstream_paths) + [audit["stage_root"] / eu.RUN_MANIFEST_NAME]
    manifest = eu.make_run_manifest(
        stage=STAGE,
        script_name=SCRIPT_NAME,
        config=config,
        input_paths=_unique_existing(paths),
        metadata={
            "kind": "stage17_final_deployment_model",
            "deployment_mode": mode,
            "context_size": context_size,
            "selected_strategy": audit["strategies"][mode],
            "training_post_ids_sha256": audit["data"]["post_id_sha256"],
            "strategy_rule": audit["strategy_document"]["by_mode"][mode].get("reason"),
        },
        project_root=root,
    )
    return run_dir, manifest


def _verify_checkpoint(model_dir: Path, expected_config: Mapping[str, Any], *, expected_run_fingerprint: str | None = None, root: Path = PROJECT_ROOT) -> dict[str, Any]:
    manifest_path = model_dir / "final_model_manifest.json"
    if not model_dir.is_dir() or not manifest_path.is_file():
        raise FileNotFoundError(f"Final model checkpoint or manifest is missing: {model_dir}")
    doc = _read_json(manifest_path)
    if doc.get("schema_version") != eu.CONFIG_SCHEMA_VERSION or doc.get("run_id") != eu.RUN_ID:
        raise ValueError(f"Final model checkpoint has stale schema/run ID: {model_dir}")
    for key in ("stage", "script_name", "backbone", "deployment_mode", "context_size", "selected_strategy", "dataset_sha256", "stage17_root_manifest_fingerprint"):
        expected_key = "stage17_root_manifest_fingerprint" if key == "stage17_root_manifest_fingerprint" else key
        if str(doc.get(key)) != str(expected_config.get(expected_key)):
            raise ValueError(f"Final model manifest mismatch for {key} at {model_dir}")
    if expected_config.get("stage17_root_manifest_fingerprint") != doc.get("stage17_root_manifest_fingerprint"):
        raise ValueError(f"Final model checkpoint refers to a different Stage 17 root manifest: {model_dir}")
    if expected_run_fingerprint is not None and doc.get("mode_run_manifest_fingerprint") != expected_run_fingerprint:
        raise ValueError(f"Final model checkpoint refers to a different mode run manifest: {model_dir}")
    weights = doc.get("checkpoint_files_sha256")
    actual_weights = _sha_tree(model_dir, excluded_names=("final_model_manifest.json",))
    if not isinstance(weights, Mapping) or dict(weights) != actual_weights:
        raise ValueError(f"Final model checkpoint file hashes differ from manifest: {model_dir}")
    if not (model_dir / "config.json").is_file() or not any((model_dir / name).is_file() for name in ("model.safetensors", "pytorch_model.bin")):
        # Sharded Transformers checkpoints are also permitted.
        has_weight_shards = any(model_dir.glob("model-*.safetensors")) or any(model_dir.glob("pytorch_model-*.bin"))
        if not has_weight_shards:
            raise FileNotFoundError(f"Final model directory lacks a recognized Hugging Face model configuration/weights: {model_dir}")
    if not (model_dir / "tokenizer_config.json").is_file():
        raise FileNotFoundError(f"Final model directory lacks tokenizer_config.json: {model_dir}")
    return doc


def _verify_mode_complete(
    *, run_dir: Path, model_dir: Path, manifest: Mapping[str, Any], mode: str,
    context_size: int, backbone: str, root: Path = PROJECT_ROOT,
) -> dict[str, Any] | None:
    marker_path = run_dir / "run_complete.json"
    index_path = run_dir / "artifacts_sha256.json"
    metrics_path = run_dir / "metrics.json"
    training_log_path = run_dir / "training_log.csv"
    model_manifest_path = model_dir / "final_model_manifest.json"
    if not marker_path.is_file():
        run_files = [path for path in run_dir.rglob("*") if path.is_file() and path.name != eu.RUN_MANIFEST_NAME]
        if run_files:
            raise RuntimeError(f"Incomplete Stage 17 run outputs exist under {run_dir}; archive them rather than reusing them")
        if model_dir.exists():
            raise RuntimeError(f"Final model directory exists without a verified run completion marker: {model_dir}; archive it")
        return None
    required = (run_dir / eu.RUN_MANIFEST_NAME, index_path, metrics_path, training_log_path, model_manifest_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Stage 17 run is marked complete but artifacts are missing: {missing}")
    fingerprint = str(manifest["fingerprint"])
    marker = _read_json(marker_path)
    expected_identity = {
        "schema_version": eu.CONFIG_SCHEMA_VERSION, "run_id": eu.RUN_ID,
        "stage": STAGE, "script_name": SCRIPT_NAME, "backbone": backbone,
        "deployment_mode": mode, "context_size": context_size,
        "manifest_fingerprint": fingerprint,
    }
    for key, expected in expected_identity.items():
        if str(marker.get(key)) != str(expected):
            raise ValueError(f"Stage 17 completion marker identity mismatch for {key} under {run_dir}")
    eu.assert_current_run_artifact(marker_path, expected_stage=STAGE, expected_script=SCRIPT_NAME, expected_fingerprint=fingerprint)
    index = _read_json(index_path)
    for key, expected in expected_identity.items():
        if str(index.get(key)) != str(expected):
            raise ValueError(f"Stage 17 run artifact index identity mismatch for {key}")
    hashes = index.get("sha256")
    if not isinstance(hashes, Mapping):
        raise ValueError("Stage 17 run artifact index has no SHA-256 map")
    for name in ("metrics.json", "training_log.csv", "final_model_manifest.json"):
        actual_path = model_manifest_path if name == "final_model_manifest.json" else run_dir / name
        if not actual_path.is_file() or hashes.get(name) != _sha(actual_path):
            raise ValueError(f"Stage 17 indexed artifact missing or stale: {actual_path}")
    if marker.get("artifact_index_sha256") != _sha(index_path):
        raise ValueError("Stage 17 run completion marker has a stale artifact-index hash")
    if marker.get("checkpoint_manifest_sha256") != _sha(model_manifest_path):
        raise ValueError("Stage 17 run marker has a stale checkpoint-manifest hash")
    checkpoint_doc = _verify_checkpoint(
        model_dir, manifest["config"], expected_run_fingerprint=str(manifest["fingerprint"]), root=root
    )
    if marker.get("checkpoint_directory") != model_dir.relative_to(root).as_posix():
        raise ValueError("Stage 17 run marker points to an unexpected deployment model directory")
    metrics = _read_json(metrics_path)
    for key, expected in {
        "schema_version": eu.CONFIG_SCHEMA_VERSION,
        "run_id": eu.RUN_ID,
        "stage": STAGE,
        "script_name": SCRIPT_NAME,
        "backbone": backbone,
        "deployment_mode": mode,
        "context_size": context_size,
        "selected_strategy": manifest["config"]["selected_strategy"],
        "dataset_sha256": manifest["config"]["dataset_sha256"],
        "stage17_manifest_fingerprint": fingerprint,
    }.items():
        if str(metrics.get(key)) != str(expected):
            raise ValueError(f"Stage 17 metrics identity mismatch for {key} under {run_dir}")
    return {"metrics": metrics, "checkpoint_manifest": checkpoint_doc, "index": index}


def _training_dataset_class(torch: Any):
    class TokenizedTextDataset(torch.utils.data.Dataset):
        def __init__(self, encodings: Mapping[str, Any], labels: Sequence[int]):
            self.encodings = encodings
            self.labels = np.asarray(labels, dtype=np.int64)

        def __len__(self) -> int:
            return len(self.labels)

        def __getitem__(self, idx: int) -> dict[str, Any]:
            item = {key: torch.as_tensor(value[idx], dtype=torch.long) for key, value in self.encodings.items()}
            item["labels"] = torch.tensor(int(self.labels[idx]), dtype=torch.long)
            return item

    return TokenizedTextDataset


def _load_training_dependencies():
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, set_seed
    except ImportError as exc:
        raise ImportError(
            "Stage 17 training requires PyTorch and Hugging Face Transformers; run in the project's GPU-enabled Colab environment"
        ) from exc
    return torch, AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, set_seed


def _load_model_and_tokenizer(
    *, mode_config: Mapping[str, Any], audit: Mapping[str, Any],
    AutoModelForSequenceClassification: Any, AutoTokenizer: Any,
) -> tuple[Any, Any]:
    backbone = str(mode_config["backbone"])
    spec = eu.get_backbone(backbone)
    strategy = str(mode_config["selected_strategy"])
    if strategy == "itft":
        itft = audit["itft"]
        tokenizer = AutoTokenizer.from_pretrained(str(itft["model_dir"]), local_files_only=True, use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(str(itft["model_dir"]), local_files_only=True)
    elif strategy == "direct":
        tokenizer = AutoTokenizer.from_pretrained(
            spec.hf_model_id, revision=str(mode_config["tokenizer_revision"]), use_fast=True
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            spec.hf_model_id,
            revision=str(mode_config["model_revision"]),
            num_labels=len(eu.LABEL2ID),
            id2label=eu.ID2LABEL,
            label2id=eu.LABEL2ID,
        )
    else:
        raise ValueError(f"Unknown Stage 15 strategy {strategy!r} for {mode_config['deployment_mode']}")

    eu.configure_tokenizer_for_input(tokenizer, INPUT_MODE)
    if tokenizer.sep_token is None:
        raise ValueError(f"Selected tokenizer {backbone} does not define sep_token")
    if getattr(model.config, "num_labels", None) != len(eu.LABEL2ID):
        raise ValueError(f"Loaded checkpoint for {backbone} does not have the expected three-class head")
    model.config.id2label = eu.ID2LABEL
    model.config.label2id = eu.LABEL2ID
    model.config.problem_type = "single_label_classification"
    if strategy == "direct":
        resolved = str(getattr(model.config, "_commit_hash", "") or "").lower()
        expected = str(mode_config["model_revision"]).lower()
        if resolved != expected:
            raise RuntimeError(f"Base model resolved to revision {resolved!r}; expected pinned revision {expected!r}")
        token_resolved = getattr(tokenizer, "_commit_hash", None) or (getattr(tokenizer, "init_kwargs", {}) or {}).get("_commit_hash")
        if not token_resolved or str(token_resolved).lower() != str(mode_config["tokenizer_revision"]).lower():
            raise RuntimeError(f"Tokenizer revision {token_resolved!r} differs from the pinned revision")
    return model, tokenizer


def _make_training_arguments(TrainingArguments: Any, *, output_dir: Path, config: Mapping[str, Any], use_cuda: bool, disable_tqdm: bool):
    return TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=int(config["num_train_epochs"]),
        per_device_train_batch_size=int(config["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(config["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(config["gradient_accumulation_steps"]),
        learning_rate=float(config["learning_rate_context"]),
        weight_decay=float(config["weight_decay"]),
        warmup_ratio=float(config["warmup_ratio"]),
        lr_scheduler_type=str(config["lr_scheduler"]),
        eval_strategy="no",
        save_strategy="no",
        logging_steps=50,
        seed=int(config["seed"]),
        data_seed=int(config["seed"]),
        report_to="none",
        fp16=bool(use_cuda),
        disable_tqdm=bool(disable_tqdm),
        dataloader_num_workers=0,
        remove_unused_columns=True,
    )


def _write_mode_complete(
    *, run_dir: Path, model_dir: Path, manifest: Mapping[str, Any], metrics: Mapping[str, Any],
    training_log: pd.DataFrame, mode: str, context_size: int, backbone: str, root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    if run_dir.exists() and (run_dir / "run_complete.json").exists():
        raise FileExistsError(f"Refusing to overwrite completed Stage 17 run: {run_dir}")
    _write_json_atomic(run_dir / "metrics.json", metrics)
    _write_csv_atomic(run_dir / "training_log.csv", training_log)
    checkpoint_manifest_path = model_dir / "final_model_manifest.json"
    _verify_checkpoint(model_dir, manifest["config"], expected_run_fingerprint=str(manifest["fingerprint"]), root=root)
    index_doc = {
        "schema_version": eu.CONFIG_SCHEMA_VERSION,
        "run_id": eu.RUN_ID,
        "stage": STAGE,
        "script_name": SCRIPT_NAME,
        "backbone": backbone,
        "deployment_mode": mode,
        "context_size": context_size,
        "manifest_fingerprint": manifest["fingerprint"],
        "sha256": {
            "metrics.json": _sha(run_dir / "metrics.json"),
            "training_log.csv": _sha(run_dir / "training_log.csv"),
            "final_model_manifest.json": _sha(checkpoint_manifest_path),
        },
    }
    _write_json_atomic(run_dir / "artifacts_sha256.json", index_doc)
    marker = {
        "schema_version": eu.CONFIG_SCHEMA_VERSION,
        "run_id": eu.RUN_ID,
        "stage": STAGE,
        "script_name": SCRIPT_NAME,
        "backbone": backbone,
        "deployment_mode": mode,
        "context_size": context_size,
        "selected_strategy": manifest["config"]["selected_strategy"],
        "manifest_fingerprint": manifest["fingerprint"],
        "artifact_index_sha256": _sha(run_dir / "artifacts_sha256.json"),
        "checkpoint_directory": model_dir.relative_to(root).as_posix(),
        "checkpoint_manifest_sha256": _sha(checkpoint_manifest_path),
        "n_training_examples": EXPECTED_ROWS,
    }
    _write_json_atomic(run_dir / "run_complete.json", marker)
    eu.assert_current_run_artifact(
        run_dir / "run_complete.json", expected_stage=STAGE,
        expected_script=SCRIPT_NAME, expected_fingerprint=str(manifest["fingerprint"]),
    )
    checked = _verify_mode_complete(
        run_dir=run_dir, model_dir=model_dir, manifest=manifest,
        mode=mode, context_size=context_size, backbone=backbone, root=root,
    )
    if checked is None:
        raise RuntimeError(f"New Stage 17 model run failed its completion audit: {run_dir}")
    return checked


def _train_mode(
    *, audit: Mapping[str, Any], mode: str, context_size: int, run_dir: Path,
    model_dir: Path, manifest: Mapping[str, Any], root: Path = PROJECT_ROOT,
    allow_cpu: bool = False, disable_tqdm: bool = False,
) -> dict[str, Any]:
    torch, AutoModel, AutoTokenizer, Trainer, TrainingArguments, set_seed = _load_training_dependencies()
    use_cuda = bool(torch.cuda.is_available())
    if not use_cuda and not allow_cpu:
        raise RuntimeError("Stage 17 requires a GPU runtime. Use Colab GPU, or pass --allow-cpu only for an intentional slow run.")
    if not use_cuda:
        print("WARNING: CPU training explicitly allowed; this run may be substantially slower than the protocol's GPU runs.")

    config = manifest["config"]
    if model_dir.exists():
        raise FileExistsError(f"Deployment model path already exists without a verified current completion marker: {model_dir}; archive it first")
    model_dir.parent.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    eu.prepare_run_directory(run_dir, manifest, allow_resume=True)

    # Build tokenization only for the current deployment mode, freeing memory before the next model.
    model, tokenizer = _load_model_and_tokenizer(
        mode_config=config, audit=audit,
        AutoModelForSequenceClassification=AutoModel, AutoTokenizer=AutoTokenizer,
    )
    texts, effective_sizes = build_input_texts(audit["data"]["modelling"], int(context_size), tokenizer)
    if len(texts) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} tokenized inputs for {mode}, got {len(texts)}")
    tokenizer.truncation_side = "left"
    encodings = tokenizer(
        texts,
        truncation=True,
        max_length=int(config["max_length_context"]),
        padding="max_length",
    )
    labels = np.asarray(audit["data"]["labels"], dtype=np.int64)
    if len(labels) != EXPECTED_ROWS or len(encodings["input_ids"]) != EXPECTED_ROWS:
        raise RuntimeError("Encoded inputs or gold labels do not match the full 1,800-row dataset")
    Dataset = _training_dataset_class(torch)
    training_dataset = Dataset(encodings, labels)
    set_seed(int(config["seed"]))
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix=f"stage17-{audit['backbone']}-{mode}-", dir=model_dir.parent) as temp_root_str:
        temp_root = Path(temp_root_str)
        training_args = _make_training_arguments(
            TrainingArguments, output_dir=temp_root / "trainer_output", config=config,
            use_cuda=use_cuda, disable_tqdm=disable_tqdm,
        )
        trainer = Trainer(model=model, args=training_args, train_dataset=training_dataset)
        train_result = trainer.train()
        elapsed = time.perf_counter() - started
        training_log = pd.DataFrame(trainer.state.log_history)
        if training_log.empty:
            training_log = pd.DataFrame(columns=["step", "epoch", "loss", "learning_rate"])
        # Keep only scalar/serializable fields in the CSV; HF versions may add nested runtime fields.
        training_log = training_log.loc[:, [column for column in training_log.columns if not training_log[column].map(lambda value: isinstance(value, (dict, list, tuple))).any()]]
        if "train_loss" in train_result.metrics:
            final_train_loss = float(train_result.metrics["train_loss"])
        else:
            loss_values = pd.to_numeric(training_log.get("loss", pd.Series(dtype=float)), errors="coerce").dropna()
            final_train_loss = float(loss_values.iloc[-1]) if len(loss_values) else None
        if final_train_loss is not None and not math.isfinite(final_train_loss):
            raise RuntimeError("Training returned a non-finite training loss")

        temp_model_dir = temp_root / "checkpoint"
        temp_model_dir.mkdir(parents=True, exist_ok=False)
        model.config.id2label = eu.ID2LABEL
        model.config.label2id = eu.LABEL2ID
        model.config.num_labels = len(eu.LABEL2ID)
        model.config.problem_type = "single_label_classification"
        model.save_pretrained(str(temp_model_dir))
        tokenizer.save_pretrained(str(temp_model_dir))
        checkpoint_files = _sha_tree(temp_model_dir)
        checkpoint_doc = {
            "schema_version": eu.CONFIG_SCHEMA_VERSION,
            "run_id": eu.RUN_ID,
            "stage": STAGE,
            "script_name": SCRIPT_NAME,
            "backbone": audit["backbone"],
            "hf_model_id": eu.get_backbone(str(audit["backbone"])).hf_model_id,
            "deployment_mode": mode,
            "context_size": int(context_size),
            "selected_context_size_c_best": int(audit["c_best"]),
            "selected_strategy": config["selected_strategy"],
            "model_initialization": config["model_initialization"],
            "model_revision": config["model_revision"],
            "tokenizer_revision": config["tokenizer_revision"],
            "dataset_sha256": audit["data"]["dataset_sha256"],
            "post_id_sha256": audit["data"]["post_id_sha256"],
            "n_training_examples": EXPECTED_ROWS,
            "label_counts": dict(audit["data"]["label_counts"]),
            "stage2_manifest_fingerprint": audit["data"]["stage2_manifest"]["fingerprint"],
            "stage8_manifest_fingerprint": audit["stage8"]["manifest_fingerprint"],
            "stage11_manifest_fingerprint": audit["stage11"]["manifest_fingerprint"],
            "stage15_manifest_fingerprint": audit["strategy_document"]["provenance"]["stage15_manifest_fingerprint"],
            "stage17_root_manifest_fingerprint": config["stage17_root_manifest_fingerprint"],
            "mode_run_manifest_fingerprint": manifest["fingerprint"],
            "config": dict(config),
            "context_sizes_used": {str(k): int(v) for k, v in pd.Series(effective_sizes).value_counts().sort_index().items()},
            "checkpoint_files_sha256": checkpoint_files,
        }
        _write_json_atomic(temp_model_dir / "final_model_manifest.json", checkpoint_doc)
        if model_dir.exists():
            raise FileExistsError(f"Refusing to overwrite an existing deployment model: {model_dir}")
        shutil.move(str(temp_model_dir), str(model_dir))

    metrics = {
        "schema_version": eu.CONFIG_SCHEMA_VERSION,
        "run_id": eu.RUN_ID,
        "stage": STAGE,
        "script_name": SCRIPT_NAME,
        "backbone": audit["backbone"],
        "deployment_mode": mode,
        "context_size": int(context_size),
        "selected_context_size_c_best": int(audit["c_best"]),
        "selected_strategy": config["selected_strategy"],
        "strategy_reason": audit["strategy_document"]["by_mode"][mode].get("reason"),
        "model_initialization": config["model_initialization"],
        "n_training_examples": EXPECTED_ROWS,
        "label_counts": dict(audit["data"]["label_counts"]),
        "context_size_used_counts": {str(k): int(v) for k, v in pd.Series(effective_sizes).value_counts().sort_index().items()},
        "dataset_sha256": audit["data"]["dataset_sha256"],
        "post_id_sha256": audit["data"]["post_id_sha256"],
        "max_length_context": int(config["max_length_context"]),
        "truncation_side": "left",
        "learning_rate_context": float(config["learning_rate_context"]),
        "num_train_epochs": int(config["num_train_epochs"]),
        "per_device_train_batch_size": int(config["per_device_train_batch_size"]),
        "gradient_accumulation_steps": int(config["gradient_accumulation_steps"]),
        "effective_batch_size": int(config["effective_batch_size"]),
        "weight_decay": float(config["weight_decay"]),
        "warmup_ratio": float(config["warmup_ratio"]),
        "lr_scheduler": str(config["lr_scheduler"]),
        "loss": str(config["loss"]),
        "seed": int(config["seed"]),
        "validation_used": False,
        "checkpoint_selection": "none_fixed_epochs_full_1800_examples",
        "final_train_loss": final_train_loss,
        "train_seconds": round(float(elapsed), 3),
        "train_minutes": round(float(elapsed) / 60.0, 3),
        "checkpoint_directory": model_dir.relative_to(root).as_posix(),
        "stage17_manifest_fingerprint": manifest["fingerprint"],
    }
    checked = _write_mode_complete(
        run_dir=run_dir, model_dir=model_dir, manifest=manifest, metrics=metrics,
        training_log=training_log, mode=mode, context_size=context_size,
        backbone=str(audit["backbone"]), root=root,
    )
    del trainer, model, tokenizer, training_dataset, encodings, texts
    if use_cuda:
        torch.cuda.empty_cache()
    print(f"[trained] {audit['backbone']} | {mode} | strategy={config['selected_strategy']} | C{context_size} | {elapsed / 60:.2f} min")
    return checked["metrics"]


def _verify_stage_complete(
    *, stage_root: Path, root_manifest: Mapping[str, Any], audit: Mapping[str, Any], root: Path = PROJECT_ROOT,
) -> bool:
    marker_path = stage_root / "stage17_complete.json"
    index_path = stage_root / "artifacts_sha256.json"
    if not marker_path.is_file():
        if index_path.exists():
            raise RuntimeError("Stage 17 root artifact index exists without a completion marker; archive/investigate it")
        return False
    required = (stage_root / eu.RUN_MANIFEST_NAME, index_path, stage_root / "stage17_summary.csv", stage_root / "stage17_summary.json")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Stage 17 completion marker exists but stage artifacts are missing: {missing}")
    fingerprint = str(root_manifest["fingerprint"])
    eu.assert_current_run_artifact(marker_path, expected_stage=STAGE, expected_script=SCRIPT_NAME, expected_fingerprint=fingerprint)
    marker, index = _read_json(marker_path), _read_json(index_path)
    expected_identity = {
        "schema_version": eu.CONFIG_SCHEMA_VERSION, "run_id": eu.RUN_ID,
        "stage": STAGE, "script_name": SCRIPT_NAME,
        "backbone": audit["backbone"], "selected_context_size_c_best": int(audit["c_best"]),
        "manifest_fingerprint": fingerprint,
    }
    for document_name, document in (("marker", marker), ("index", index)):
        for key, expected in expected_identity.items():
            if str(document.get(key)) != str(expected):
                raise ValueError(f"Stage 17 {document_name} identity mismatch for {key}")
    if marker.get("artifact_index_sha256") != _sha(index_path):
        raise ValueError("Stage 17 root completion marker has a stale artifact-index hash")
    hashes = index.get("sha256")
    if not isinstance(hashes, Mapping):
        raise ValueError("Stage 17 root artifact index has no SHA-256 map")
    for relative, digest in hashes.items():
        path = stage_root / str(relative)
        if not path.is_file() or _sha(path) != digest:
            raise ValueError(f"Stage 17 root indexed artifact missing or stale: {path}")
    expected_modes = [("target_only", 0), ("context_aware", int(audit["c_best"]))]
    for mode, condition in expected_modes:
        run_dir = _mode_run_directory(stage_root, mode, condition)
        model_dir = _model_directory(str(audit["backbone"]), mode, condition, root)
        manifest = _read_json(run_dir / eu.RUN_MANIFEST_NAME)
        if manifest.get("fingerprint") != _read_json(run_dir / "run_complete.json").get("manifest_fingerprint"):
            raise ValueError(f"Stage 17 mode run manifest/marker mismatch: {run_dir}")
        checked = _verify_mode_complete(
            run_dir=run_dir, model_dir=model_dir, manifest=manifest, mode=mode,
            context_size=condition, backbone=str(audit["backbone"]), root=root,
        )
        if checked is None:
            raise FileNotFoundError(f"Stage 17 final deployment model is incomplete: {run_dir}")
    return True


def _publish_stage_summary(
    *, audit: Mapping[str, Any], stage_root: Path, root_manifest: Mapping[str, Any],
    mode_metrics: Mapping[str, Mapping[str, Any]], root: Path = PROJECT_ROOT,
) -> None:
    rows = []
    for mode, condition in (("target_only", 0), ("context_aware", int(audit["c_best"]))):
        metrics = mode_metrics[mode]
        model_dir = _model_directory(str(audit["backbone"]), mode, condition, root)
        checkpoint_manifest = model_dir / "final_model_manifest.json"
        rows.append({
            "backbone": audit["backbone"],
            "deployment_mode": mode,
            "context_size": condition,
            "selected_strategy": metrics["selected_strategy"],
            "strategy_reason": metrics["strategy_reason"],
            "n_training_examples": metrics["n_training_examples"],
            "dataset_sha256": metrics["dataset_sha256"],
            "label_counts": json.dumps(metrics["label_counts"], sort_keys=True),
            "context_size_used_counts": json.dumps(metrics["context_size_used_counts"], sort_keys=True),
            "max_length_context": metrics["max_length_context"],
            "truncation_side": metrics["truncation_side"],
            "learning_rate_context": metrics["learning_rate_context"],
            "num_train_epochs": metrics["num_train_epochs"],
            "per_device_train_batch_size": metrics["per_device_train_batch_size"],
            "seed": metrics["seed"],
            "validation_used": metrics["validation_used"],
            "final_train_loss": metrics["final_train_loss"],
            "train_seconds": metrics["train_seconds"],
            "checkpoint_directory": metrics["checkpoint_directory"],
            "checkpoint_manifest_sha256": _sha(checkpoint_manifest),
            "run_manifest_fingerprint": _read_json(_mode_run_directory(stage_root, mode, condition) / eu.RUN_MANIFEST_NAME)["fingerprint"],
        })
    summary = pd.DataFrame(rows)
    _write_csv_atomic(stage_root / "stage17_summary.csv", summary)
    _write_json_atomic(stage_root / "stage17_summary.json", {
        "schema_version": eu.CONFIG_SCHEMA_VERSION,
        "run_id": eu.RUN_ID,
        "stage": STAGE,
        "script_name": SCRIPT_NAME,
        "backbone": audit["backbone"],
        "selected_context_size_c_best": int(audit["c_best"]),
        "strategy_by_mode": dict(audit["strategies"]),
        "n_training_examples_per_model": EXPECTED_ROWS,
        "models": rows,
        "stage17_manifest_fingerprint": root_manifest["fingerprint"],
        "interpretation_note": "Training loss is descriptive optimization output, not a held-out generalization metric.",
    })
    # The root index includes the summary and every child run's public verification
    # files. Each child index in turn authenticates its checkpoint manifest; that
    # manifest authenticates every saved model/tokenizer file.
    hashes: dict[str, str] = {
        "stage17_summary.csv": _sha(stage_root / "stage17_summary.csv"),
        "stage17_summary.json": _sha(stage_root / "stage17_summary.json"),
    }
    for mode, condition in (("target_only", 0), ("context_aware", int(audit["c_best"]))):
        run_dir = _mode_run_directory(stage_root, mode, condition)
        for name in (eu.RUN_MANIFEST_NAME, "metrics.json", "training_log.csv", "artifacts_sha256.json", "run_complete.json"):
            path = run_dir / name
            if not path.is_file():
                raise FileNotFoundError(f"Stage 17 mode artifact is missing before publishing stage summary: {path}")
            hashes[path.relative_to(stage_root).as_posix()] = _sha(path)
    _write_json_atomic(stage_root / "artifacts_sha256.json", {
        "schema_version": eu.CONFIG_SCHEMA_VERSION,
        "run_id": eu.RUN_ID,
        "stage": STAGE,
        "script_name": SCRIPT_NAME,
        "backbone": audit["backbone"],
        "selected_context_size_c_best": int(audit["c_best"]),
        "manifest_fingerprint": root_manifest["fingerprint"],
        "sha256": hashes,
    })
    _write_json_atomic(stage_root / "stage17_complete.json", {
        "schema_version": eu.CONFIG_SCHEMA_VERSION,
        "run_id": eu.RUN_ID,
        "stage": STAGE,
        "script_name": SCRIPT_NAME,
        "backbone": audit["backbone"],
        "selected_context_size_c_best": int(audit["c_best"]),
        "manifest_fingerprint": root_manifest["fingerprint"],
        "artifact_index_sha256": _sha(stage_root / "artifacts_sha256.json"),
        "strategy_by_mode": dict(audit["strategies"]),
        "n_training_examples_per_model": EXPECTED_ROWS,
        "checkpoint_manifest_sha256_by_mode": {
            mode: _sha(_model_directory(str(audit["backbone"]), mode, condition, root) / "final_model_manifest.json")
            for mode, condition in (("target_only", 0), ("context_aware", int(audit["c_best"])))
        },
    })
    eu.assert_current_run_artifact(
        stage_root / "stage17_complete.json", expected_stage=STAGE,
        expected_script=SCRIPT_NAME, expected_fingerprint=str(root_manifest["fingerprint"]),
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 17: train the selected final deployment models on all 1,800 contextual posts.")
    parser.add_argument("--validate-only", action="store_true", help="Verify data, Stage 11/15 decisions, Stage 8 configuration, and required ITFT checkpoint without training.")
    parser.add_argument("--allow-cpu", action="store_true", help="Explicitly allow slow CPU training; GPU training is the protocol default.")
    parser.add_argument("--disable-tqdm", action="store_true", help="Disable Hugging Face progress bars in notebook logs.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    os.chdir(PROJECT_ROOT)
    audit = _validate_prerequisites(PROJECT_ROOT)
    stage_root, root_manifest = _prepare_stage_manifest(audit, PROJECT_ROOT)
    # --validate-only is deliberately read-only: do not create the Stage 17 output directory.
    print("Stage 17 prerequisite audit passed.")
    print(f"  Selected backbone: {audit['backbone']}")
    print(f"  Selected C_best: C{audit['c_best']}")
    print(f"  Strategy for target-only C0: {audit['strategies']['target_only']}")
    print(f"  Strategy for context-aware C{audit['c_best']}: {audit['strategies']['context_aware']}")
    print(f"  Full contextual training posts: {len(audit['data']['modelling'])}")
    print(f"  max_length_context: {audit['sequence']['max_length_context']}")
    print(f"  learning_rate_context: {audit['learning_rates']['learning_rate_context'][audit['backbone']]}")
    print(f"  Output root: {stage_root}")
    if args.validate_only:
        print("Validation-only completed; no model weights or run directories were written.")
        return 0

    # The frozen protocol requires the complete canonical stage tree to be present
    # before any final training run. The audit never deletes or overwrites history.
    eu.audit_pipeline_script_names(PROJECT_ROOT, require_all=True)
    eu.prepare_run_directory(stage_root, root_manifest, allow_resume=True)
    if _verify_stage_complete(stage_root=stage_root, root_manifest=root_manifest, audit=audit, root=PROJECT_ROOT):
        print("Stage 17 is already complete; both final model checkpoints and all provenance hashes verified. Nothing to rerun.")
        return 0

    upstream_paths = _stage_manifest_inputs(audit, PROJECT_ROOT)
    mode_metrics: dict[str, dict[str, Any]] = {}
    for mode, condition in (("target_only", 0), ("context_aware", int(audit["c_best"]))):
        run_dir, run_manifest = _mode_manifest(
            audit=audit, mode=mode, context_size=condition,
            stage_root_manifest=root_manifest, upstream_paths=upstream_paths, root=PROJECT_ROOT,
        )
        model_dir = _model_directory(str(audit["backbone"]), mode, condition, PROJECT_ROOT)
        eu.prepare_run_directory(run_dir, run_manifest, allow_resume=True)
        checked = _verify_mode_complete(
            run_dir=run_dir, model_dir=model_dir, manifest=run_manifest,
            mode=mode, context_size=condition, backbone=str(audit["backbone"]), root=PROJECT_ROOT,
        )
        if checked is not None:
            mode_metrics[mode] = checked["metrics"]
            print(f"[verified-existing] {audit['backbone']} | {mode} | {audit['strategies'][mode]} | {model_dir.relative_to(PROJECT_ROOT)}")
            continue
        result = _train_mode(
            audit=audit, mode=mode, context_size=condition,
            run_dir=run_dir, model_dir=model_dir, manifest=run_manifest,
            root=PROJECT_ROOT, allow_cpu=args.allow_cpu, disable_tqdm=args.disable_tqdm,
        )
        mode_metrics[mode] = result

    if set(mode_metrics) != {"target_only", "context_aware"}:
        raise RuntimeError("Stage 17 did not produce both required deployment-mode results")
    _publish_stage_summary(
        audit=audit, stage_root=stage_root, root_manifest=root_manifest,
        mode_metrics=mode_metrics, root=PROJECT_ROOT,
    )
    if not _verify_stage_complete(stage_root=stage_root, root_manifest=root_manifest, audit=audit, root=PROJECT_ROOT):
        raise RuntimeError("Stage 17 final publication did not pass its completion audit")
    print("\nStage 17 complete. Both final deployment models were trained on the full 1,800-post contextual dataset.")
    print(f"  Target-only model: {_model_directory(str(audit['backbone']), 'target_only', 0, PROJECT_ROOT)}")
    print(f"  Context-aware model: {_model_directory(str(audit['backbone']), 'context_aware', int(audit['c_best']), PROJECT_ROOT)}")
    print(f"  Summary: {stage_root / 'stage17_summary.csv'}")
    print("Note: these are training artifacts, not held-out generalization metrics.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
