"""Stage 13: Experiment II-B, availability-aware shallow-pool evaluation.

This is a descriptive generalization analysis for the single backbone selected
by Stage 11. It evaluates the five Stage 12 full-matched-subset checkpoints
(C0-C4) against the disjoint 900-example shallow pool. C0 and C1 use all 900
shallow examples; C2 uses examples with at least two available parents (750);
C3 uses examples with at least three parents (450); C4 is not applicable because
the shallow pool contains no four-parent examples.

Stage 13 never re-selects the backbone or C_best. It reads the verified Stage 11
decision and verifies Stage 12 checkpoint manifests, file hashes, and configs.
All results are descriptive; no significance test is performed here.

Usage (repository root, normally a Colab runtime):
    python scripts/13_e2b_generalization.py --validate-only
    python scripts/13_e2b_generalization.py

Outputs are written below:
    results/runs/left_truncation_v2/stage_13/e2_<selected_backbone>/

Old flat-path outputs are not accepted as current-run evidence. Archive them
before executing this stage. A completed output may be reused only if its run
manifest and all artifact hashes match the current utility, protocol, pipeline,
upstream data, Stage 11 decision, and Stage 12 checkpoints.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support


def _project_root() -> Path:
    here = Path(__file__).resolve().parent
    return here.parent if here.name.casefold() == "scripts" else here


PROJECT_ROOT = _project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import experiment_utils as eu  # noqa: E402

STAGE = 13
SCRIPT_NAME = "scripts/13_e2b_generalization.py"
CONDITIONS = tuple(range(5))
EVALUATED_CONDITIONS = (0, 1, 2, 3)
EXPECTED_SHALLOW_ROWS = eu.CONTEXT_SHALLOW_EXPECTED_ROWS
EXPECTED_MATCHED_ROWS = eu.CONTEXT_MATCHED_EXPECTED_ROWS
EXPECTED_PARENT_COUNTS = {1: 150, 2: 300, 3: 450}
EVAL_BATCH_SIZE = 32
INPUT_MODE = "contextual"
TRUNCATION_SIDE = "left"

SHALLOW_POOL_PATH = Path("data/processed/ctx_shallow_pool.csv")
MATCHED_SUBSET_PATH = Path("data/processed/ctx_matched_subset.csv")
DECISION_PATH = eu.BACKBONE_DECISION_PATH
STAGE8_DIR = Path(*eu.RUN_ROOT_PARTS) / "stage_08" / "summary"
STAGE11_DIR = Path(*eu.RUN_ROOT_PARTS) / "stage_11" / "decision"
STAGE_ROOT_PARTS = Path(*eu.RUN_ROOT_PARTS) / "stage_13"
LEGACY_OUTPUT_PATTERNS = (
    "models/E2_full_retrained",
    "models/E2_itft_context",
    "results/tables/e2_best_context.csv",
    "results/tables/e2_context_ablation_summary.csv",
    "results/tables/e2_full_retrain_metrics.csv",
    "results/tables/e2_grid_metrics.csv",
    "results/raw/e2b_C*_shallow_predictions.csv",
    "results/raw/e2b_C*_shallow_probs.npy",
    "results/tables/e2b_generalization_*.csv",
)
SUMMARY_TABLES = (
    "e2b_generalization_metrics.csv",
    "e2b_generalization_by_availability.csv",
    "e2b_generalization_by_class.csv",
    "e2b_generalization_confusion_matrices.csv",
    "e2b_generalization_pairwise_vs_c0.csv",
    "e2b_generalization_manifest.csv",
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
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
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
    excluded = set(excluded_names)
    files = {
        path.relative_to(directory).as_posix(): _sha(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.name not in excluded
    }
    if not files:
        raise RuntimeError(f"No checkpoint files found under {directory}")
    return files


def _assert_no_legacy_outputs(root: Path = PROJECT_ROOT) -> None:
    found: list[Path] = []
    for pattern in LEGACY_OUTPUT_PATTERNS:
        found.extend(path for path in root.glob(pattern) if path.exists())
    eu.assert_no_legacy_outputs(list(dict.fromkeys(path.resolve() for path in found)))


def _verify_json_manifest_inputs(manifest: Mapping[str, Any], root: Path, *, label: str) -> None:
    inputs = manifest.get("input_sha256")
    if not isinstance(inputs, Mapping):
        raise ValueError(f"{label} manifest has no input_sha256 mapping")
    for relative, expected_hash in inputs.items():
        path = Path(str(relative))
        if not path.is_absolute():
            path = root / path
        if not path.is_file() or _sha(path) != expected_hash:
            raise ValueError(f"{label} manifest input is missing or changed: {path}")


def _verify_index_entries(index_path: Path, base_dir: Path, *, expected_identity: Mapping[str, Any], special_paths: Mapping[str, Path] | None = None) -> dict[str, Any]:
    index = _read_json(index_path)
    for key, expected in expected_identity.items():
        if str(index.get(key)) != str(expected):
            raise ValueError(f"Artifact index {index_path} has incorrect {key}: {index.get(key)!r} != {expected!r}")
    hashes = index.get("sha256")
    if not isinstance(hashes, Mapping):
        raise ValueError(f"Artifact index {index_path} has no sha256 mapping")
    special_paths = dict(special_paths or {})
    for relative, expected_hash in hashes.items():
        path = special_paths.get(str(relative), base_dir / str(relative))
        if not path.is_file() or _sha(path) != expected_hash:
            raise ValueError(f"Artifact hash mismatch or missing file for {path}")
    return index


def _verify_stage11_decision(root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Validate Stage 11's decision bundle, index, marker and selected values."""
    root_decision = _resolve(DECISION_PATH, root)
    stage_dir = root / STAGE11_DIR
    manifest_path = stage_dir / eu.RUN_MANIFEST_NAME
    marker_path = stage_dir / "stage11_complete.json"
    index_path = stage_dir / "artifacts_sha256.json"
    copy_path = stage_dir / "backbone_decision.json"
    missing = [str(p) for p in (root_decision, manifest_path, marker_path, index_path, copy_path) if not p.is_file()]
    if missing:
        raise FileNotFoundError("Stage 11 decision bundle is incomplete; finish Stage 11 first. Missing: " + ", ".join(missing))

    manifest = _read_json(manifest_path)
    fingerprint = str(manifest.get("fingerprint", ""))
    marker = eu.assert_current_run_artifact(
        marker_path, expected_stage=11, expected_script=eu.PIPELINE_SCRIPTS[11], expected_fingerprint=fingerprint
    )
    marker = _read_json(marker_path)
    if marker.get("manifest_fingerprint") != fingerprint:
        raise ValueError("Stage 11 completion marker does not match its manifest")
    if marker.get("artifact_index_sha256") != _sha(index_path):
        raise ValueError("Stage 11 completion marker has a stale artifact-index hash")
    if marker.get("root_decision_sha256") != _sha(root_decision) or _sha(copy_path) != _sha(root_decision):
        raise ValueError("Stage 11 root decision, versioned copy, and completion marker differ")
    index = _verify_index_entries(
        index_path, stage_dir,
        expected_identity={
            "schema_version": eu.CONFIG_SCHEMA_VERSION,
            "run_id": eu.RUN_ID,
            "stage": 11,
            "script_name": eu.PIPELINE_SCRIPTS[11],
            "manifest_fingerprint": fingerprint,
        },
    )
    decision = eu.load_backbone_decision_document(root_decision)
    eu.validate_backbone_decision_document(decision)
    if decision.get("stage11_manifest_fingerprint") != fingerprint:
        raise ValueError("Stage 11 decision references a different run manifest")
    if decision.get("stage11_output_dir") != stage_dir.relative_to(root).as_posix():
        raise ValueError("Stage 11 decision has an unexpected output directory")
    if marker.get("selected_backbone") != decision.get("decision"):
        raise ValueError("Stage 11 marker and decision JSON select different backbones")
    if marker.get("c_best_by_backbone") != decision.get("c_best_by_backbone"):
        raise ValueError("Stage 11 marker and decision JSON disagree on C_best")
    for backbone in eu.BACKBONES:
        context_doc = _read_json(stage_dir / f"context_selection_{backbone}.json")
        grid_doc = _read_json(stage_dir / f"grid_audit_{backbone}.json")
        if context_doc != decision["context_decisions"][backbone]:
            raise ValueError(f"Stage 11 embedded context selection differs for {backbone}")
        if grid_doc.get("grid_fingerprint") != decision["source_grid_fingerprints"][backbone]:
            raise ValueError(f"Stage 11 source grid fingerprint differs for {backbone}")
        if int(grid_doc.get("n_runs", -1)) != 50 or int(grid_doc.get("expected_runs", -1)) != 50:
            raise ValueError(f"Stage 11 grid audit does not certify the complete 50-run {backbone} grid")
    _verify_json_manifest_inputs(manifest, root, label="Stage 11")
    return {
        "decision": decision,
        "backbone": str(decision["decision"]),
        "c_best": int(decision["c_best_by_backbone"][decision["decision"]]),
        "stage_dir": stage_dir,
        "manifest_path": manifest_path,
        "manifest": manifest,
        "manifest_fingerprint": fingerprint,
        "marker_path": marker_path,
        "index_path": index_path,
        "root_path": root_decision,
        "copy_path": copy_path,
    }


def _verify_stage8_config(root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Verify Stage 8's published configs and latency bundle used by Stages 11/12."""
    stage_dir = root / STAGE8_DIR
    manifest_path = stage_dir / eu.RUN_MANIFEST_NAME
    marker_path = stage_dir / "stage8_complete.json"
    index_path = stage_dir / "artifacts_sha256.json"
    receipt_path = stage_dir / "stage8_publish_receipt.json"
    latency_path = stage_dir / "context_latency_measurements.json"
    sequence_path = _resolve(eu.SEQUENCE_LENGTH_CONFIG, root)
    rates_path = _resolve(eu.LEARNING_RATE_CONFIG, root)
    required = (manifest_path, marker_path, index_path, receipt_path, latency_path, sequence_path, rates_path)
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError("Stage 8 configuration bundle is incomplete: " + ", ".join(missing))
    manifest = _read_json(manifest_path)
    fingerprint = str(manifest.get("fingerprint", ""))
    eu.assert_current_run_artifact(marker_path, expected_stage=8, expected_script=eu.PIPELINE_SCRIPTS[8], expected_fingerprint=fingerprint)
    marker = _read_json(marker_path)
    if marker.get("manifest_fingerprint") != fingerprint or marker.get("artifact_index_sha256") != _sha(index_path):
        raise ValueError("Stage 8 completion marker has stale manifest/index metadata")
    index = _verify_index_entries(
        index_path, stage_dir,
        expected_identity={"schema_version": eu.CONFIG_SCHEMA_VERSION, "run_id": eu.RUN_ID, "stage": 8, "manifest_fingerprint": fingerprint},
    )
    if marker.get("sequence_length_config_sha256") != _sha(sequence_path) or marker.get("learning_rate_config_sha256") != _sha(rates_path):
        raise ValueError("Stage 8 shared configuration files differ from their published hashes")
    shared = eu.load_shared_run_config(
        sequence_length_path=sequence_path, learning_rate_path=rates_path,
        require_context=True, require_context_lr=True,
    )
    latency = _read_json(latency_path)
    receipt = _read_json(receipt_path)
    if latency.get("schema_version") != eu.CONFIG_SCHEMA_VERSION or latency.get("run_id") != eu.RUN_ID:
        raise ValueError("Stage 8 latency measurements use an incompatible schema or run ID")
    if latency.get("manifest_fingerprint") != fingerprint or receipt.get("summary_manifest_fingerprint") != fingerprint:
        raise ValueError("Stage 8 latency/receipt belongs to a different manifest")
    if receipt.get("latency_measurements_sha256") != _sha(latency_path):
        raise ValueError("Stage 8 latency measurements do not match the publication receipt")
    if receipt.get("sequence_length_config_sha256") != _sha(sequence_path):
        raise ValueError("Stage 8 publication receipt has a stale sequence_length.json hash")
    if receipt.get("learning_rate_config_sha256") != _sha(rates_path):
        raise ValueError("Stage 8 publication receipt has a stale learning_rates.json hash")
    if latency.get("model_ids") != {b: eu.get_backbone(b).hf_model_id for b in eu.BACKBONES}:
        raise ValueError("Stage 8 latency model IDs differ from experiment_utils.BACKBONES")
    if latency.get("context_sizes") != list(CONDITIONS) or latency.get("truncation_side") != TRUNCATION_SIDE:
        raise ValueError("Stage 8 latency artifact must cover C0-C4 with left truncation")
    expected_tokenizers = shared["sequence_length"].get("context_tokenizer_resolved_revisions")
    context_preflight = shared["learning_rates"].get("context_preflight", {})
    expected_models = context_preflight.get("model_revisions") if isinstance(context_preflight, Mapping) else None
    if not isinstance(expected_tokenizers, Mapping) or not isinstance(expected_models, Mapping):
        raise ValueError("Stage 8 configuration does not pin both model and tokenizer revisions")
    if latency.get("tokenizer_revisions") != expected_tokenizers or latency.get("model_revisions") != expected_models:
        raise ValueError("Stage 8 latency revisions differ from the published shared configuration")
    medians = latency.get("median_ms_by_context")
    samples = latency.get("measurements_ms_by_context")
    if not isinstance(medians, Mapping) or not isinstance(samples, Mapping):
        raise ValueError("Stage 8 latency artifact lacks its raw samples or medians")
    for context in CONDITIONS:
        key = f"C{context}"
        if key not in medians or key not in samples:
            raise ValueError(f"Stage 8 latency artifact is missing {key}")
        for backbone in eu.BACKBONES:
            vals = np.asarray(samples[key].get(backbone, []), dtype=float)
            median = medians[key].get(backbone)
            if vals.ndim != 1 or len(vals) < 20 or not np.isfinite(vals).all() or (vals <= 0).any():
                raise ValueError(f"Stage 8 requires at least 20 finite positive latency samples for {key}/{backbone}")
            if median is None or not math.isclose(float(median), float(np.median(vals)), rel_tol=1e-8, abs_tol=1e-8):
                raise ValueError(f"Stage 8 recorded latency median is invalid for {key}/{backbone}")
    # Stage 8 publishes/merges the selected contextual values into the shared
    # sequence-length and learning-rate files. Its original manifest may also
    # contain pre-merge hashes, so validate the publication marker/index and
    # the explicit Stage 8 receipt instead of blindly comparing those inputs.
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
        "sequence": shared["sequence_length"],
        "learning_rates": shared["learning_rates"],
        "latency": latency,
    }


def _load_and_validate_data(root: Path = PROJECT_ROOT) -> dict[str, Any]:
    eu.assert_current_stage_manifest(2, root)
    shallow_path = _resolve(SHALLOW_POOL_PATH, root)
    matched_path = _resolve(MATCHED_SUBSET_PATH, root)
    for path in (shallow_path, matched_path):
        if not path.is_file():
            raise FileNotFoundError(f"Stage 2 output is missing: {path}")
    shallow_raw = pd.read_csv(shallow_path)
    matched_raw = pd.read_csv(matched_path)
    shallow_audit = eu.validate_context_shallow_pool(shallow_raw, expected_rows=EXPECTED_SHALLOW_ROWS)
    matched_audit = eu.validate_context_dataset(matched_raw, expected_rows=EXPECTED_MATCHED_ROWS)
    available_counts = pd.to_numeric(shallow_raw["context_size_available"], errors="raise").astype(int).value_counts().sort_index().to_dict()
    if available_counts != EXPECTED_PARENT_COUNTS:
        raise ValueError(f"Shallow-pool availability counts are {available_counts}; expected {EXPECTED_PARENT_COUNTS}")
    shallow_ids = set(shallow_raw[eu.CTX_ID_COLUMN].astype(str))
    matched_ids = set(matched_raw[eu.CTX_ID_COLUMN].astype(str))
    if shallow_ids & matched_ids:
        raise ValueError("The Stage 2 shallow pool and matched subset are not disjoint")
    if len(shallow_ids | matched_ids) != EXPECTED_SHALLOW_ROWS + EXPECTED_MATCHED_ROWS:
        raise ValueError("The Stage 2 shallow pool and matched subset do not cover exactly 1,800 unique post IDs")

    shallow = shallow_raw.copy()
    shallow["context_size_available"] = pd.to_numeric(shallow["context_size_available"], errors="raise").astype(int)
    shallow = shallow.sort_values(eu.CTX_ID_COLUMN, key=lambda col: col.astype(str)).reset_index(drop=True)
    matched = matched_raw.sort_values(eu.CTX_ID_COLUMN, key=lambda col: col.astype(str)).reset_index(drop=True)
    return {
        "shallow_path": shallow_path,
        "matched_path": matched_path,
        "shallow_raw": shallow_raw,
        "matched_raw": matched_raw,
        "shallow": shallow,
        "matched": matched,
        "shallow_audit": shallow_audit,
        "matched_audit": matched_audit,
        "shallow_sha256": _sha(shallow_path),
        "matched_sha256": _sha(matched_path),
        "shallow_post_id_sha256": eu.sha256_json(shallow[eu.CTX_ID_COLUMN].astype(str).tolist()),
        "matched_post_id_sha256": eu.sha256_json(matched[eu.CTX_ID_COLUMN].astype(str).tolist()),
        "fold_assignment_sha256": matched_audit["fold_assignment_sha256"],
        "stage2_manifest_path": root / Path(*eu.RUN_ROOT_PARTS) / "stage_02" / eu.RUN_MANIFEST_NAME,
    }


def _verify_stage12(backbone: str, decision_bundle: Mapping[str, Any], stage8: Mapping[str, Any], data: Mapping[str, Any], root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Verify the selected backbone's Stage 12 summary and all five checkpoints."""
    stage_root = root / STAGE_ROOT_PARTS.parent / "stage_12" / f"e2_{backbone}"
    stage_manifest_path = stage_root / eu.RUN_MANIFEST_NAME
    complete_path = stage_root / "stage12_complete.json"
    index_path = stage_root / "artifacts_sha256.json"
    summary_csv = stage_root / "e2_full_retrain_metrics.csv"
    summary_json = stage_root / "e2_full_retrain_summary.json"
    required = (stage_manifest_path, complete_path, index_path, summary_csv, summary_json)
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError("Stage 12 retraining bundle is incomplete for the selected backbone; missing: " + ", ".join(missing))

    stage_manifest = _read_json(stage_manifest_path)
    fingerprint = str(stage_manifest.get("fingerprint", ""))
    eu.assert_current_run_artifact(complete_path, expected_stage=12, expected_script=eu.PIPELINE_SCRIPTS[12], expected_fingerprint=fingerprint)
    complete = _read_json(complete_path)
    if complete.get("manifest_fingerprint") != fingerprint or complete.get("artifact_index_sha256") != _sha(index_path):
        raise ValueError("Stage 12 completion marker has stale manifest/index metadata")
    if complete.get("backbone") != backbone or int(complete.get("n_models", -1)) != 5:
        raise ValueError("Stage 12 completion marker does not certify all five selected-backbone models")

    selected_context = int(decision_bundle["decision"]["c_best_by_backbone"][backbone])
    root_config = stage_manifest.get("config", {})
    for key, expected in {
        "backbone": backbone,
        "selected_context_size_c_best": selected_context,
        "stage11_manifest_fingerprint": decision_bundle["manifest_fingerprint"],
        "stage8_manifest_fingerprint": stage8["fingerprint"],
        "dataset_sha256": data["matched_sha256"],
        "fold_assignment_sha256": data["fold_assignment_sha256"],
        "max_length_context": int(stage8["sequence"]["max_length_context"]),
        "truncation_side": TRUNCATION_SIDE,
    }.items():
        if str(root_config.get(key)) != str(expected):
            raise ValueError(f"Stage 12 root manifest is inconsistent for {key}: {root_config.get(key)!r} != {expected!r}")
    _verify_json_manifest_inputs(stage_manifest, root, label="Stage 12 root")

    specials: dict[str, Path] = {}
    for context in CONDITIONS:
        specials[f"checkpoint_manifests/C{context}"] = (
            root / "models" / "e2_full_retrained" / backbone / f"C{context}" / "retrain_checkpoint_manifest.json"
        )
    index = _verify_index_entries(
        index_path, stage_root,
        expected_identity={
            "schema_version": eu.CONFIG_SCHEMA_VERSION,
            "run_id": eu.RUN_ID,
            "stage": 12,
            "script_name": eu.PIPELINE_SCRIPTS[12],
            "backbone": backbone,
            "selected_context_size_c_best": selected_context,
            "manifest_fingerprint": fingerprint,
        },
        special_paths=specials,
    )
    summary = _read_json(summary_json)
    if summary.get("backbone") != backbone or int(summary.get("selected_context_size_c_best", -1)) != selected_context:
        raise ValueError("Stage 12 summary JSON disagrees with the Stage 11 selected backbone/context")
    metrics_df = pd.read_csv(summary_csv)
    if not {"context_size", "backbone", "experiment_id", "n_train", "learning_rate_context", "max_length_context"}.issubset(metrics_df.columns):
        raise ValueError("Stage 12 metrics summary is missing required columns")
    if sorted(pd.to_numeric(metrics_df["context_size"], errors="raise").astype(int).tolist()) != list(CONDITIONS):
        raise ValueError("Stage 12 metrics summary must contain exactly one row for C0-C4")
    if set(metrics_df["backbone"].astype(str)) != {backbone} or not (pd.to_numeric(metrics_df["n_train"], errors="raise").astype(int) == EXPECTED_MATCHED_ROWS).all():
        raise ValueError("Stage 12 metrics summary contains unexpected backbone or training sample count")

    checkpoint_infos: dict[int, dict[str, Any]] = {}
    model_revision_expected = stage8["learning_rates"].get("context_preflight", {}).get("model_revisions", {}).get(backbone)
    tokenizer_revision_expected = stage8["sequence"].get("context_tokenizer_resolved_revisions", {}).get(backbone)
    for context in CONDITIONS:
        run_dir = stage_root / f"C{context}"
        model_dir = root / "models" / "e2_full_retrained" / backbone / f"C{context}"
        run_manifest_path = run_dir / eu.RUN_MANIFEST_NAME
        run_marker_path = run_dir / "run_complete.json"
        run_index_path = run_dir / "artifacts_sha256.json"
        metrics_path = run_dir / "metrics.json"
        checkpoint_manifest_path = model_dir / "retrain_checkpoint_manifest.json"
        for path in (run_manifest_path, run_marker_path, run_index_path, metrics_path, checkpoint_manifest_path, model_dir / "config.json"):
            if not path.is_file():
                raise FileNotFoundError(f"Stage 12 C{context} artifact is missing: {path}")
        run_manifest = _read_json(run_manifest_path)
        run_fp = str(run_manifest.get("fingerprint", ""))
        eu.assert_current_run_artifact(run_marker_path, expected_stage=12, expected_script=eu.PIPELINE_SCRIPTS[12], expected_fingerprint=run_fp)
        run_marker = _read_json(run_marker_path)
        run_config = run_manifest.get("config", {})
        for key, expected in {
            "backbone": backbone,
            "context_size": context,
            "selected_context_size_c_best": selected_context,
            "stage11_manifest_fingerprint": decision_bundle["manifest_fingerprint"],
            "stage8_manifest_fingerprint": stage8["fingerprint"],
            "dataset_sha256": data["matched_sha256"],
            "fold_assignment_sha256": data["fold_assignment_sha256"],
            "max_length_context": int(stage8["sequence"]["max_length_context"]),
            "truncation_side": TRUNCATION_SIDE,
        }.items():
            if str(run_config.get(key)) != str(expected):
                raise ValueError(f"Stage 12 C{context} manifest mismatch for {key}")
        _verify_json_manifest_inputs(run_manifest, root, label=f"Stage 12 C{context}")
        for key, expected in {
            "schema_version": eu.CONFIG_SCHEMA_VERSION,
            "run_id": eu.RUN_ID,
            "stage": 12,
            "script_name": eu.PIPELINE_SCRIPTS[12],
            "backbone": backbone,
            "context_size": context,
            "manifest_fingerprint": run_fp,
        }.items():
            if str(run_marker.get(key)) != str(expected):
                raise ValueError(f"Stage 12 C{context} completion marker mismatch for {key}")
        run_index = _read_json(run_index_path)
        for key, expected in {
            "schema_version": eu.CONFIG_SCHEMA_VERSION,
            "run_id": eu.RUN_ID,
            "stage": 12,
            "script_name": eu.PIPELINE_SCRIPTS[12],
            "backbone": backbone,
            "context_size": context,
            "manifest_fingerprint": run_fp,
        }.items():
            if str(run_index.get(key)) != str(expected):
                raise ValueError(f"Stage 12 C{context} artifact-index mismatch for {key}")
        run_index_hashes = run_index.get("sha256", {})
        if not isinstance(run_index_hashes, Mapping):
            raise ValueError(f"Stage 12 C{context} artifact index has no sha256 map")
        if run_index_hashes.get("metrics.json") != _sha(metrics_path):
            raise ValueError(f"Stage 12 C{context} metrics hash mismatch")
        if run_index_hashes.get("checkpoint_manifest_sha256") != _sha(checkpoint_manifest_path):
            raise ValueError(f"Stage 12 C{context} checkpoint-manifest hash mismatch")
        if run_marker.get("artifact_index_sha256") != _sha(run_index_path) or run_marker.get("checkpoint_manifest_sha256") != _sha(checkpoint_manifest_path):
            raise ValueError(f"Stage 12 C{context} completion marker hashes are stale")
        if run_marker.get("checkpoint_directory") != model_dir.relative_to(root).as_posix():
            raise ValueError(f"Stage 12 C{context} marker points to an unexpected checkpoint directory")

        checkpoint_doc = _read_json(checkpoint_manifest_path)
        if checkpoint_doc.get("manifest_fingerprint") != run_fp or checkpoint_doc.get("backbone") != backbone or int(checkpoint_doc.get("context_size", -1)) != context:
            raise ValueError(f"Stage 12 C{context} checkpoint manifest identity mismatch")
        checkpoint_cfg = checkpoint_doc.get("config", {})
        if checkpoint_doc.get("hf_model_id") != eu.get_backbone(backbone).hf_model_id:
            raise ValueError(f"Stage 12 C{context} checkpoint model ID differs from the backbone registry")
        if str(checkpoint_doc.get("model_revision", "")).lower() != str(model_revision_expected).lower():
            raise ValueError(f"Stage 12 C{context} model revision differs from Stage 8")
        if str(checkpoint_doc.get("tokenizer_revision", "")).lower() != str(tokenizer_revision_expected).lower():
            raise ValueError(f"Stage 12 C{context} tokenizer revision differs from Stage 8")
        if checkpoint_doc.get("stage11_manifest_fingerprint") != decision_bundle["manifest_fingerprint"] or checkpoint_doc.get("stage8_manifest_fingerprint") != stage8["fingerprint"]:
            raise ValueError(f"Stage 12 C{context} checkpoint belongs to a different Stage 11/8 decision bundle")
        if checkpoint_doc.get("dataset_sha256") != data["matched_sha256"] or checkpoint_doc.get("fold_assignment_sha256") != data["fold_assignment_sha256"]:
            raise ValueError(f"Stage 12 C{context} checkpoint was trained on a different matched dataset/fold assignment")
        if int(checkpoint_cfg.get("max_length_context", -1)) != int(stage8["sequence"]["max_length_context"]):
            raise ValueError(f"Stage 12 C{context} checkpoint used a different contextual maximum length")
        if checkpoint_cfg.get("truncation_side") != TRUNCATION_SIDE:
            raise ValueError(f"Stage 12 C{context} checkpoint did not use left truncation")
        file_hashes = checkpoint_doc.get("checkpoint_files_sha256")
        if not isinstance(file_hashes, Mapping) or dict(file_hashes) != _sha_tree(model_dir):
            raise ValueError(f"Stage 12 C{context} model/tokenizer files differ from their checkpoint manifest")
        metrics = _read_json(metrics_path)
        if metrics.get("backbone") != backbone or int(metrics.get("context_size", -1)) != context or metrics.get("run_id") != eu.RUN_ID:
            raise ValueError(f"Stage 12 C{context} metrics identity mismatch")
        if metrics.get("stage11_manifest_fingerprint") != decision_bundle["manifest_fingerprint"] or metrics.get("itft_manifest_fingerprint") != checkpoint_doc.get("itft_manifest_fingerprint"):
            raise ValueError(f"Stage 12 C{context} metrics provenance mismatch")
        checkpoint_infos[context] = {
            "run_dir": run_dir, "run_manifest_path": run_manifest_path, "run_manifest": run_manifest,
            "run_fingerprint": run_fp, "marker_path": run_marker_path, "index_path": run_index_path,
            "metrics_path": metrics_path, "model_dir": model_dir, "checkpoint_manifest_path": checkpoint_manifest_path,
            "checkpoint_manifest": checkpoint_doc, "checkpoint_metrics": metrics,
        }

    # The Stage 12 root index itself includes hashes for the five run bundles and checkpoint manifests.
    return {
        "stage_dir": stage_root,
        "manifest_path": stage_manifest_path,
        "manifest": stage_manifest,
        "fingerprint": fingerprint,
        "complete_path": complete_path,
        "index_path": index_path,
        "metrics_path": summary_csv,
        "summary_path": summary_json,
        "metrics_df": metrics_df,
        "checkpoint_infos": checkpoint_infos,
        "selected_context": selected_context,
    }


def eligible_population(shallow: pd.DataFrame, condition: int) -> pd.DataFrame:
    """Return the protocol-defined shallow-pool population eligible for Cn."""
    if condition not in CONDITIONS:
        raise ValueError(f"condition must be 0-4, got {condition}")
    if condition == 4:
        return shallow.iloc[0:0].copy().reset_index(drop=True)
    available = pd.to_numeric(shallow["context_size_available"], errors="raise").astype(int)
    threshold = max(condition, 1)
    return shallow.loc[available >= threshold].copy().reset_index(drop=True)


def label_ids(labels: pd.Series) -> np.ndarray:
    mapped = labels.map(eu.LABEL2ID)
    if mapped.isna().any():
        bad = sorted(labels[mapped.isna()].astype(str).unique())
        raise ValueError(f"Cannot map labels to canonical IDs: {bad}")
    return mapped.astype(int).to_numpy()


def overall_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    if len(y_true) == 0:
        return {"n_examples": 0, "accuracy": None, "macro_f1": None, "weighted_f1": None}
    return {
        "n_examples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def class_metrics_table(condition: int, y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(eu.LABEL2ID))), average=None, zero_division=0
    )
    return pd.DataFrame([
        {
            "condition": condition, "class_id": class_id, "class_label": eu.ID2LABEL[class_id],
            "precision": float(precision[class_id]), "recall": float(recall[class_id]),
            "f1": float(f1[class_id]), "support": int(support[class_id]),
        }
        for class_id in range(len(eu.LABEL2ID))
    ])


def confusion_table(condition: int, y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    labels = list(range(len(eu.LABEL2ID)))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame([
        {
            "condition": condition, "true_class_id": int(true_id), "true_class": eu.ID2LABEL[true_id],
            "predicted_class_id": int(pred_id), "predicted_class": eu.ID2LABEL[pred_id],
            "count": int(matrix[true_id, pred_id]),
        }
        for true_id in labels for pred_id in labels
    ])


def availability_breakdown(condition: int, population: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    parent_counts = pd.to_numeric(population["context_size_available"], errors="raise").astype(int).to_numpy()
    for parent_count in sorted(set(parent_counts.tolist())):
        mask = parent_counts == parent_count
        metrics = overall_metrics(y_true[mask], y_pred[mask])
        rows.append({
            "condition": condition, "parent_count_available": int(parent_count),
            "n_examples": metrics["n_examples"], "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"], "weighted_f1": metrics["weighted_f1"],
        })
    return pd.DataFrame(rows)


def pairwise_vs_c0(shallow: pd.DataFrame, predictions: Mapping[int, Mapping[str, Any]]) -> pd.DataFrame:
    """Descriptive C0-vs-C1/C2/C3 comparisons on each deeper model's same IDs."""
    rows: list[dict[str, Any]] = []
    c0 = predictions[0]
    c0_lookup = {str(post_id): idx for idx, post_id in enumerate(c0["post_ids"])}
    for deeper in (1, 2, 3):
        deep = predictions[deeper]
        eligible = eligible_population(shallow, deeper)
        ids = eligible[eu.CTX_ID_COLUMN].astype(str).tolist()
        deep_lookup = {str(post_id): idx for idx, post_id in enumerate(deep["post_ids"])}
        missing_c0 = [post_id for post_id in ids if post_id not in c0_lookup]
        missing_deep = [post_id for post_id in ids if post_id not in deep_lookup]
        if missing_c0 or missing_deep:
            raise ValueError(f"C0/C{deeper} do not cover the identical eligible population")
        c0_indices = np.asarray([c0_lookup[post_id] for post_id in ids], dtype=int)
        deep_indices = np.asarray([deep_lookup[post_id] for post_id in ids], dtype=int)
        y_true = label_ids(eligible["label"])
        c0_pred = np.asarray(c0["y_pred"])[c0_indices]
        deep_pred = np.asarray(deep["y_pred"])[deep_indices]
        m0 = overall_metrics(y_true, c0_pred)
        md = overall_metrics(y_true, deep_pred)
        rows.append({
            "comparison": f"C0_vs_C{deeper}", "context_depth": deeper,
            "population_rule": "all shallow posts (>=1 parent)" if deeper == 1 else f"shallow posts with >= {deeper} parents",
            "n_examples": m0["n_examples"],
            "c0_accuracy": m0["accuracy"], "deeper_accuracy": md["accuracy"],
            "accuracy_delta_deeper_minus_c0": md["accuracy"] - m0["accuracy"],
            "c0_macro_f1": m0["macro_f1"], "deeper_macro_f1": md["macro_f1"],
            "macro_f1_delta_deeper_minus_c0": md["macro_f1"] - m0["macro_f1"],
            "c0_weighted_f1": m0["weighted_f1"], "deeper_weighted_f1": md["weighted_f1"],
            "weighted_f1_delta_deeper_minus_c0": md["weighted_f1"] - m0["weighted_f1"],
            "significance_test_performed": False,
        })
    return pd.DataFrame(rows)


def _root_manifest_inputs(root: Path, data: Mapping[str, Any], stage8: Mapping[str, Any], stage11: Mapping[str, Any], stage12: Mapping[str, Any]) -> list[Path]:
    paths: list[Path] = [
        data["shallow_path"], data["matched_path"], data["stage2_manifest_path"],
        stage8["sequence_path"], stage8["learning_rates_path"], stage8["manifest_path"], stage8["marker_path"],
        stage8["index_path"], stage8["receipt_path"], stage8["latency_path"],
        stage11["root_path"], stage11["copy_path"], stage11["manifest_path"], stage11["marker_path"], stage11["index_path"],
        stage12["manifest_path"], stage12["complete_path"], stage12["index_path"], stage12["metrics_path"], stage12["summary_path"],
    ]
    for condition, info in stage12["checkpoint_infos"].items():
        paths.extend([info["run_manifest_path"], info["marker_path"], info["index_path"], info["metrics_path"], info["checkpoint_manifest_path"], info["model_dir"] / "config.json"])
    for manifest_path, marker_path, index_path in ((stage8["manifest_path"], stage8["marker_path"], stage8["index_path"]), (stage11["manifest_path"], stage11["marker_path"], stage11["index_path"])):
        if not all(Path(p).is_file() for p in (manifest_path, marker_path, index_path)):
            raise FileNotFoundError("Incomplete upstream manifest bundle")
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = Path(path).resolve()
        if resolved not in seen:
            if not resolved.is_file():
                raise FileNotFoundError(f"Stage 13 manifest input is missing: {resolved}")
            unique.append(resolved)
            seen.add(resolved)
    return unique


def _make_root_manifest(*, backbone: str, c_best: int, data: Mapping[str, Any], stage8: Mapping[str, Any], stage11: Mapping[str, Any], stage12: Mapping[str, Any], root: Path = PROJECT_ROOT) -> tuple[Path, dict[str, Any]]:
    stage_root = root / STAGE_ROOT_PARTS / f"e2_{backbone}"
    config = {
        "schema_version": eu.CONFIG_SCHEMA_VERSION, "run_id": eu.RUN_ID, "stage": STAGE,
        "script_name": SCRIPT_NAME, "kind": "stage13_e2b_shallow_pool_generalization",
        "backbone": backbone, "hf_model_id": eu.get_backbone(backbone).hf_model_id,
        "selected_context_size_c_best": int(c_best), "stage11_manifest_fingerprint": stage11["manifest_fingerprint"],
        "stage12_manifest_fingerprint": stage12["fingerprint"], "stage8_manifest_fingerprint": stage8["fingerprint"],
        "matched_dataset_sha256": data["matched_sha256"], "shallow_pool_sha256": data["shallow_sha256"],
        "matched_fold_assignment_sha256": data["fold_assignment_sha256"],
        "shallow_post_id_sha256": data["shallow_post_id_sha256"],
        "max_length_context": int(stage8["sequence"]["max_length_context"]),
        "truncation_side": TRUNCATION_SIDE, "eval_batch_size": EVAL_BATCH_SIZE,
        "eligible_examples_by_context": {"C0": 900, "C1": 900, "C2": 750, "C3": 450, "C4": 0},
        "selection_role": "descriptive_only_does_not_select_or_override_c_best",
    }
    manifest = eu.make_run_manifest(
        stage=STAGE, script_name=SCRIPT_NAME, config=config,
        input_paths=_root_manifest_inputs(root, data, stage8, stage11, stage12),
        metadata={
            "kind": "stage13_shallow_pool_evaluation",
            "evaluated_contexts": list(EVALUATED_CONDITIONS), "not_applicable_contexts": [4],
            "no_significance_tests": True, "selected_backbone": backbone, "selected_c_best": int(c_best),
        }, project_root=root,
    )
    return stage_root, manifest


def _make_condition_manifest(*, condition: int, stage_root: Path, root_manifest: Mapping[str, Any], backbone: str, c_best: int, data: Mapping[str, Any], stage8: Mapping[str, Any], stage11: Mapping[str, Any], stage12: Mapping[str, Any], root: Path = PROJECT_ROOT) -> tuple[Path, dict[str, Any]]:
    population = eligible_population(data["shallow"], condition)
    config = {
        "schema_version": eu.CONFIG_SCHEMA_VERSION, "run_id": eu.RUN_ID, "stage": STAGE,
        "script_name": SCRIPT_NAME, "kind": "stage13_e2b_condition_inference", "backbone": backbone,
        "hf_model_id": eu.get_backbone(backbone).hf_model_id, "experiment_id": eu.experiment2_id(condition, backbone),
        "context_size": condition, "selected_context_size_c_best": c_best,
        "stage11_manifest_fingerprint": stage11["manifest_fingerprint"], "stage12_manifest_fingerprint": stage12["fingerprint"],
        "stage13_root_manifest_fingerprint": root_manifest["fingerprint"], "stage8_manifest_fingerprint": stage8["fingerprint"],
        "shallow_pool_sha256": data["shallow_sha256"], "eligible_post_id_sha256": eu.sha256_json(population[eu.CTX_ID_COLUMN].astype(str).tolist()),
        "n_examples": len(population), "max_length_context": int(stage8["sequence"]["max_length_context"]),
        "truncation_side": TRUNCATION_SIDE, "eval_batch_size": EVAL_BATCH_SIZE,
        "population_rule": "all_shallow_posts" if condition in (0, 1) else f"available_parent_count_gte_{condition}",
    }
    inputs = [
        data["shallow_path"], data["matched_path"], data["stage2_manifest_path"],
        stage8["sequence_path"], stage8["learning_rates_path"], stage8["manifest_path"], stage8["marker_path"], stage8["index_path"],
        stage11["root_path"], stage11["copy_path"], stage11["manifest_path"], stage11["marker_path"], stage11["index_path"],
        stage12["manifest_path"], stage12["complete_path"], stage12["index_path"], stage12["metrics_path"], stage12["summary_path"],
        stage12["checkpoint_infos"][condition]["run_manifest_path"], stage12["checkpoint_infos"][condition]["marker_path"],
        stage12["checkpoint_infos"][condition]["index_path"], stage12["checkpoint_infos"][condition]["metrics_path"],
        stage12["checkpoint_infos"][condition]["checkpoint_manifest_path"],
    ]
    manifest = eu.make_run_manifest(
        stage=STAGE, script_name=SCRIPT_NAME, config=config, input_paths=inputs,
        metadata={"kind": "stage13_e2b_condition", "no_significance_tests": True, "C_best_is_not_reselected": True}, project_root=root,
    )
    run_dir = stage_root / f"C{condition}"
    return run_dir, manifest


def _check_manifest_compatibility(run_dir: Path, manifest: Mapping[str, Any]) -> None:
    manifest_path = run_dir / eu.RUN_MANIFEST_NAME
    if not run_dir.exists():
        return
    files = [p for p in run_dir.rglob("*") if p.is_file() and p != manifest_path]
    if files and not manifest_path.is_file():
        raise RuntimeError(f"Refusing to reuse {run_dir}: it contains files without a run manifest; archive it")
    if manifest_path.is_file():
        existing = _read_json(manifest_path)
        eu._validate_manifest_integrity(existing)
        if existing.get("fingerprint") != manifest.get("fingerprint"):
            raise RuntimeError(f"Refusing to reuse {run_dir}: current manifest/config/input hashes differ; archive it")


def _verify_condition_output(run_dir: Path, manifest: Mapping[str, Any], *, backbone: str, condition: int, root: Path = PROJECT_ROOT) -> dict[str, Any] | None:
    marker_path = run_dir / "run_complete.json"
    index_path = run_dir / "artifacts_sha256.json"
    pred_path = run_dir / "predictions.csv"
    probs_path = run_dir / "probabilities.npy"
    metrics_path = run_dir / "metrics.json"
    manifest_path = run_dir / eu.RUN_MANIFEST_NAME
    if not marker_path.is_file():
        unexpected = [p for p in run_dir.iterdir() if p.name != eu.RUN_MANIFEST_NAME] if run_dir.exists() else []
        if unexpected:
            raise RuntimeError(f"Stage 13 C{condition} has partial/unverified output files; archive it before retrying: {run_dir}")
        return None
    required = (manifest_path, index_path, pred_path, probs_path, metrics_path)
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Stage 13 C{condition} is marked complete but artifacts are missing: {missing}")
    fingerprint = str(manifest["fingerprint"])
    eu.assert_current_run_artifact(marker_path, expected_stage=STAGE, expected_script=SCRIPT_NAME, expected_fingerprint=fingerprint)
    marker = _read_json(marker_path)
    index = _verify_index_entries(
        index_path, run_dir,
        expected_identity={"schema_version": eu.CONFIG_SCHEMA_VERSION, "run_id": eu.RUN_ID, "stage": STAGE,
                           "script_name": SCRIPT_NAME, "backbone": backbone, "context_size": condition,
                           "manifest_fingerprint": fingerprint},
    )
    for key, expected in {
        "schema_version": eu.CONFIG_SCHEMA_VERSION, "run_id": eu.RUN_ID, "stage": STAGE,
        "script_name": SCRIPT_NAME, "backbone": backbone, "context_size": condition,
        "manifest_fingerprint": fingerprint,
    }.items():
        if str(marker.get(key)) != str(expected):
            raise ValueError(f"Stage 13 C{condition} marker mismatch for {key}")
    if marker.get("artifact_index_sha256") != _sha(index_path):
        raise ValueError(f"Stage 13 C{condition} marker has a stale output-index hash")
    preds = pd.read_csv(pred_path)
    probs = np.load(probs_path, allow_pickle=False)
    if index.get("sha256", {}).get("predictions.csv") != _sha(pred_path) or index.get("sha256", {}).get("probabilities.npy") != _sha(probs_path):
        raise ValueError(f"Stage 13 C{condition} raw predictions/probabilities differ from the artifact index")
    if index.get("sha256", {}).get("metrics.json") != _sha(metrics_path):
        raise ValueError(f"Stage 13 C{condition} metrics hash mismatch")
    metrics = _read_json(metrics_path)
    ids = preds[eu.CTX_ID_COLUMN].astype(str).tolist()
    if metrics.get("post_id_sha256") != eu.sha256_json(ids):
        raise ValueError(f"Stage 13 C{condition} ordered post-ID hash mismatch")
    if probs.shape != (len(preds), len(eu.LABEL2ID)) or not np.isfinite(probs).all() or (probs < 0).any() or not np.allclose(probs.sum(axis=1), 1.0, atol=1e-4):
        raise ValueError(f"Stage 13 C{condition} probability array is invalid")
    if len(preds) != int(manifest["config"]["n_examples"]):
        raise ValueError(f"Stage 13 C{condition} prediction count differs from the run config")
    return {"predictions_df": preds, "probabilities": probs, "metrics": metrics, "run_dir": run_dir, "run_fingerprint": fingerprint}


def _predict_condition(*, condition: int, population: pd.DataFrame, model_dir: Path, max_length: int, disable_tqdm: bool = False) -> dict[str, Any]:
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise ImportError("Stage 13 inference requires PyTorch and Hugging Face Transformers") from exc
    if not torch.cuda.is_available():
        print("Warning: no GPU detected; Stage 13 inference will run on CPU and may be slower.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True, use_fast=True)
    eu.configure_tokenizer_for_input(tokenizer, INPUT_MODE)
    if tokenizer.truncation_side != TRUNCATION_SIDE:
        raise RuntimeError(f"Tokenizer truncation side is {tokenizer.truncation_side!r}, expected {TRUNCATION_SIDE!r}")
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir), local_files_only=True)
    model.to(device)
    model.eval()
    load_seconds = time.perf_counter() - load_started
    if int(getattr(model.config, "num_labels", -1)) != len(eu.LABEL2ID):
        raise ValueError(f"Stage 12 C{condition} checkpoint does not have a three-class classification head")

    texts = [eu.build_context_text(row, condition, sep=eu.sep_from_tokenizer(tokenizer)) for _, row in population.iterrows()]
    prob_parts: list[np.ndarray] = []
    tokenization_seconds = 0.0
    inference_seconds = 0.0
    for start in range(0, len(texts), EVAL_BATCH_SIZE):
        batch_texts = texts[start:start + EVAL_BATCH_SIZE]
        token_started = time.perf_counter()
        encodings = tokenizer(
            batch_texts, truncation=True, max_length=max_length, padding="max_length", return_tensors="pt"
        )
        tokenization_seconds += time.perf_counter() - token_started
        inputs = {key: value.to(device) for key, value in encodings.items()}
        inference_started = time.perf_counter()
        with torch.inference_mode():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        inference_seconds += time.perf_counter() - inference_started
        prob_parts.append(probs)
    probs = np.concatenate(prob_parts, axis=0) if prob_parts else np.empty((0, len(eu.LABEL2ID)), dtype=float)
    token_and_inference_seconds = tokenization_seconds + inference_seconds
    y_pred = probs.argmax(axis=1).astype(int) if len(probs) else np.array([], dtype=int)
    if len(probs) != len(population):
        raise RuntimeError(f"C{condition} produced {len(probs)} prediction rows for {len(population)} eligible examples")
    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "y_pred": y_pred, "probabilities": probs, "model_load_seconds": float(load_seconds),
        "tokenization_seconds": float(tokenization_seconds), "inference_seconds": float(inference_seconds),
        "tokenization_and_inference_seconds": float(token_and_inference_seconds),
    }


def _save_condition_output(*, run_dir: Path, manifest: Mapping[str, Any], backbone: str, condition: int, population: pd.DataFrame, y_true: np.ndarray, prediction: Mapping[str, Any], root: Path = PROJECT_ROOT) -> dict[str, Any]:
    eu.prepare_run_directory(run_dir, manifest, allow_resume=True)
    pred_path = run_dir / "predictions.csv"
    probs_path = run_dir / "probabilities.npy"
    metrics_path = run_dir / "metrics.json"
    index_path = run_dir / "artifacts_sha256.json"
    marker_path = run_dir / "run_complete.json"
    probs = np.asarray(prediction["probabilities"], dtype=float)
    y_pred = np.asarray(prediction["y_pred"], dtype=int)
    if probs.shape != (len(population), len(eu.LABEL2ID)) or y_pred.shape != (len(population),):
        raise ValueError(f"C{condition} inference returned unexpected prediction/probability shapes")
    if not np.isfinite(probs).all() or (probs < 0).any() or not np.allclose(probs.sum(axis=1), 1.0, atol=1e-4):
        raise ValueError(f"C{condition} inference returned invalid probability rows")
    output = pd.DataFrame({
        eu.CTX_ID_COLUMN: population[eu.CTX_ID_COLUMN].astype(str).to_numpy(),
        "context_size_available": population["context_size_available"].astype(int).to_numpy(),
        "y_true": [eu.ID2LABEL[int(v)] for v in y_true],
        "y_pred": [eu.ID2LABEL[int(v)] for v in y_pred],
        "confidence": probs.max(axis=1),
    })
    for column in eu.CTX_ANALYSIS_ONLY_COLUMNS:
        if column in population.columns and column not in output.columns:
            output[column] = population[column].to_numpy()
    _write_csv_atomic(pred_path, output)
    np.save(probs_path, probs, allow_pickle=False)
    metrics = {
        "schema_version": eu.CONFIG_SCHEMA_VERSION, "run_id": eu.RUN_ID, "stage": STAGE,
        "script_name": SCRIPT_NAME, "backbone": backbone, "experiment_id": eu.experiment2_id(condition, backbone),
        "context_size": condition, "selected_context_size_c_best": int(manifest["config"]["selected_context_size_c_best"]),
        "population_rule": manifest["config"]["population_rule"], "n_examples": int(len(population)),
        **overall_metrics(y_true, y_pred), "post_id_sha256": eu.sha256_json(output[eu.CTX_ID_COLUMN].astype(str).tolist()),
        "shallow_pool_sha256": manifest["config"]["shallow_pool_sha256"],
        "stage11_manifest_fingerprint": manifest["config"]["stage11_manifest_fingerprint"],
        "stage12_manifest_fingerprint": manifest["config"]["stage12_manifest_fingerprint"],
        "stage13_manifest_fingerprint": manifest["fingerprint"],
        "model_load_seconds": float(prediction["model_load_seconds"]),
        "tokenization_seconds": float(prediction["tokenization_seconds"]),
        "inference_seconds": float(prediction["inference_seconds"]),
        "tokenization_and_inference_seconds": float(prediction["tokenization_and_inference_seconds"]),
    }
    _write_json_atomic(metrics_path, metrics)
    identity = {
        "schema_version": eu.CONFIG_SCHEMA_VERSION, "run_id": eu.RUN_ID, "stage": STAGE,
        "script_name": SCRIPT_NAME, "backbone": backbone, "context_size": condition,
        "manifest_fingerprint": manifest["fingerprint"],
    }
    _write_json_atomic(index_path, {**identity, "sha256": {
        "predictions.csv": _sha(pred_path), "probabilities.npy": _sha(probs_path), "metrics.json": _sha(metrics_path),
    }})
    _write_json_atomic(marker_path, {**identity, "artifact_index_sha256": _sha(index_path), "post_id_sha256": metrics["post_id_sha256"], "n_examples": len(population)})
    eu.assert_current_run_artifact(marker_path, expected_stage=STAGE, expected_script=SCRIPT_NAME, expected_fingerprint=str(manifest["fingerprint"]))
    verified = _verify_condition_output(run_dir, manifest, backbone=backbone, condition=condition, root=root)
    assert verified is not None
    return verified


def _make_summary_tables(*, backbone: str, c_best: int, predictions: Mapping[int, Mapping[str, Any]], data: Mapping[str, Any], stage11: Mapping[str, Any], stage12: Mapping[str, Any]) -> dict[str, pd.DataFrame]:
    overall_rows: list[dict[str, Any]] = []
    availability_tables: list[pd.DataFrame] = []
    class_tables: list[pd.DataFrame] = []
    cm_tables: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        if condition == 4:
            overall_rows.append({
                "condition": 4, "experiment_id": eu.experiment2_id(4, backbone), "backbone": backbone,
                "status": "not_applicable", "population": "shallow_pool",
                "population_rule": "no external C4 population; shallow pool contains only 1-3 available parents",
                "n_examples": 0, "accuracy": np.nan, "macro_f1": np.nan, "weighted_f1": np.nan,
                "model_load_seconds": np.nan, "tokenization_seconds": np.nan, "inference_seconds": np.nan, "tokenization_and_inference_seconds": np.nan,
                "is_c_best": condition == c_best, "selected_from_shallow_pool": False,
            })
        else:
            item = predictions[condition]
            pred_df = item["predictions_df"]
            y_true = label_ids(pred_df["y_true"])
            y_pred = np.asarray(item["y_pred"], dtype=int)
            metrics = overall_metrics(y_true, y_pred)
            overall_rows.append({
                "condition": condition, "experiment_id": eu.experiment2_id(condition, backbone), "backbone": backbone,
                "status": "evaluated", "population": "shallow_pool",
                "population_rule": "all shallow posts (>=1 available parent)" if condition in (0, 1) else f"shallow posts with >= {condition} available parents",
                **metrics, "model_load_seconds": float(item["metrics"]["model_load_seconds"]),
                "inference_seconds": float(item["metrics"]["inference_seconds"]),
                "tokenization_and_inference_seconds": float(item["metrics"]["tokenization_and_inference_seconds"]),
                "is_c_best": condition == c_best, "selected_from_shallow_pool": False,
            })
            availability_tables.append(availability_breakdown(condition, item["population"], y_true, y_pred))
            class_tables.append(class_metrics_table(condition, y_true, y_pred))
            cm_tables.append(confusion_table(condition, y_true, y_pred))
        manifest_rows.append({
            "stage": "13-E2B", "backbone": backbone, "condition": condition,
            "experiment_id": eu.experiment2_id(condition, backbone),
            "model_path": str((Path("models") / "e2_full_retrained" / backbone / f"C{condition}").as_posix()),
            "stage12_manifest_fingerprint": stage12["fingerprint"],
            "stage11_manifest_fingerprint": stage11["manifest_fingerprint"],
            "selected_context_cv": c_best, "c_best_source": "Stage 11 paired OOF bootstrap decision",
            "selected_from_shallow_pool": False,
            "eligible_examples": {0: 900, 1: 900, 2: 750, 3: 450, 4: 0}[condition],
            "role": "generalization evaluation" if condition in EVALUATED_CONDITIONS else "not applicable: no shallow C4 population",
        })
    pairwise_df = pairwise_vs_c0(data["shallow"], predictions)
    return {
        "e2b_generalization_metrics.csv": pd.DataFrame(overall_rows).sort_values("condition").reset_index(drop=True),
        "e2b_generalization_by_availability.csv": pd.concat(availability_tables, ignore_index=True).sort_values(["condition", "parent_count_available"]).reset_index(drop=True),
        "e2b_generalization_by_class.csv": pd.concat(class_tables, ignore_index=True).sort_values(["condition", "class_id"]).reset_index(drop=True),
        "e2b_generalization_confusion_matrices.csv": pd.concat(cm_tables, ignore_index=True).sort_values(["condition", "true_class_id", "predicted_class_id"]).reset_index(drop=True),
        "e2b_generalization_pairwise_vs_c0.csv": pairwise_df,
        "e2b_generalization_manifest.csv": pd.DataFrame(manifest_rows),
    }


def _verify_complete_stage(*, stage_root: Path, root_manifest: Mapping[str, Any], backbone: str, c_best: int, root: Path) -> bool:
    marker_path = stage_root / "stage13_complete.json"
    index_path = stage_root / "artifacts_sha256.json"
    if not marker_path.is_file():
        existing_tables = [stage_root / name for name in (*SUMMARY_TABLES, "e2b_generalization_summary.json", "artifacts_sha256.json") if (stage_root / name).exists()]
        if existing_tables:
            raise RuntimeError(f"Stage 13 has partial summary outputs without stage13_complete.json; archive them before retrying: {existing_tables}")
        return False
    eu.assert_current_run_artifact(marker_path, expected_stage=STAGE, expected_script=SCRIPT_NAME, expected_fingerprint=str(root_manifest["fingerprint"]))
    marker = _read_json(marker_path)
    if marker.get("artifact_index_sha256") != _sha(index_path):
        raise ValueError("Stage 13 completion marker has a stale artifact-index hash")
    index = _verify_index_entries(
        index_path, stage_root,
        expected_identity={"schema_version": eu.CONFIG_SCHEMA_VERSION, "run_id": eu.RUN_ID, "stage": STAGE,
                           "script_name": SCRIPT_NAME, "backbone": backbone, "selected_context_size_c_best": c_best,
                           "manifest_fingerprint": root_manifest["fingerprint"]},
    )
    for key, expected in {
        "backbone": backbone, "selected_context_size_c_best": c_best,
        "stage11_manifest_fingerprint": root_manifest["config"]["stage11_manifest_fingerprint"],
        "stage12_manifest_fingerprint": root_manifest["config"]["stage12_manifest_fingerprint"],
        "manifest_fingerprint": root_manifest["fingerprint"],
    }.items():
        if str(marker.get(key)) != str(expected):
            raise ValueError(f"Stage 13 completion marker mismatch for {key}")
    return True


def _publish_summary(*, stage_root: Path, root_manifest: Mapping[str, Any], backbone: str, c_best: int, tables: Mapping[str, pd.DataFrame], predictions: Mapping[int, Mapping[str, Any]], stage11: Mapping[str, Any], stage12: Mapping[str, Any], root: Path) -> None:
    for name, frame in tables.items():
        _write_csv_atomic(stage_root / name, frame)
    summary_path = stage_root / "e2b_generalization_summary.json"
    _write_json_atomic(summary_path, {
        "schema_version": eu.CONFIG_SCHEMA_VERSION, "run_id": eu.RUN_ID, "stage": STAGE,
        "script_name": SCRIPT_NAME, "backbone": backbone, "hf_model_id": eu.get_backbone(backbone).hf_model_id,
        "selected_context_size_c_best": c_best, "stage11_manifest_fingerprint": stage11["manifest_fingerprint"],
        "stage12_manifest_fingerprint": stage12["fingerprint"], "stage13_manifest_fingerprint": root_manifest["fingerprint"],
        "evaluated_contexts": list(EVALUATED_CONDITIONS), "not_applicable_contexts": [4],
        "population_sizes": {"C0": 900, "C1": 900, "C2": 750, "C3": 450, "C4": 0},
        "number_of_significance_tests": 0,
        "interpretation": "Descriptive availability-aware generalization only; this stage does not select or override C_best.",
        "prediction_post_ids_sha256": {f"C{c}": eu.sha256_json(item["predictions_df"][eu.CTX_ID_COLUMN].astype(str).tolist()) for c, item in predictions.items()},
    })
    index_hashes: dict[str, str] = {}
    for name in (*SUMMARY_TABLES, "e2b_generalization_summary.json"):
        index_hashes[name] = _sha(stage_root / name)
    for condition in EVALUATED_CONDITIONS:
        run_dir = stage_root / f"C{condition}"
        for name in (eu.RUN_MANIFEST_NAME, "run_complete.json", "artifacts_sha256.json", "predictions.csv", "probabilities.npy", "metrics.json"):
            path = run_dir / name
            if not path.is_file():
                raise FileNotFoundError(f"Cannot publish Stage 13 summary; missing {path}")
            index_hashes[f"C{condition}/{name}"] = _sha(path)
    index_path = stage_root / "artifacts_sha256.json"
    identity = {
        "schema_version": eu.CONFIG_SCHEMA_VERSION, "run_id": eu.RUN_ID, "stage": STAGE,
        "script_name": SCRIPT_NAME, "backbone": backbone, "selected_context_size_c_best": c_best,
        "manifest_fingerprint": root_manifest["fingerprint"],
    }
    _write_json_atomic(index_path, {**identity, "sha256": index_hashes})
    marker_path = stage_root / "stage13_complete.json"
    _write_json_atomic(marker_path, {
        **identity, "stage11_manifest_fingerprint": stage11["manifest_fingerprint"],
        "stage12_manifest_fingerprint": stage12["fingerprint"],
        "artifact_index_sha256": _sha(index_path), "evaluated_conditions": list(EVALUATED_CONDITIONS),
        "not_applicable_conditions": [4],
    })
    eu.assert_current_run_artifact(marker_path, expected_stage=STAGE, expected_script=SCRIPT_NAME, expected_fingerprint=str(root_manifest["fingerprint"]))
    if not _verify_complete_stage(stage_root=stage_root, root_manifest=root_manifest, backbone=backbone, c_best=c_best, root=root):
        raise RuntimeError("Stage 13 completion marker could not be verified after publication")


def validate_prerequisites(*, root: Path = PROJECT_ROOT) -> dict[str, Any]:
    _assert_no_legacy_outputs(root)
    data = _load_and_validate_data(root)
    stage11 = _verify_stage11_decision(root)
    stage8 = _verify_stage8_config(root)
    backbone = stage11["backbone"]
    c_best = stage11["c_best"]
    if backbone not in eu.BACKBONES:
        raise ValueError(f"Stage 11 selected an unregistered backbone: {backbone}")
    decision = stage11["decision"]
    if decision.get("post_id_sha256") != decision["context_decisions"][backbone].get("post_id_sha256"):
        raise ValueError("Stage 11 selected-backbone context decision is not aligned to the common OOF post IDs")
    if int(decision.get("n_oof_examples", -1)) != EXPECTED_MATCHED_ROWS:
        raise ValueError("Stage 11 decision does not certify the expected 900-example OOF set")
    if int(stage8["sequence"].get("max_length_context", 0)) < 8:
        raise ValueError("Stage 8 max_length_context is missing or invalid")
    selected_lr = stage8["learning_rates"].get("learning_rate_context", {}).get(backbone)
    if selected_lr is None or not math.isfinite(float(selected_lr)) or float(selected_lr) <= 0:
        raise ValueError(f"Stage 8 has no valid contextual learning rate for selected backbone {backbone}")
    tokenizer_revision = stage8["sequence"].get("context_tokenizer_resolved_revisions", {}).get(backbone)
    model_revision = stage8["learning_rates"].get("context_preflight", {}).get("model_revisions", {}).get(backbone)
    if not tokenizer_revision or not model_revision:
        raise ValueError(f"Stage 8 does not pin model/tokenizer revisions for selected backbone {backbone}")
    stage12 = _verify_stage12(backbone, stage11, stage8, data, root)
    stage_root, root_manifest = _make_root_manifest(backbone=backbone, c_best=c_best, data=data, stage8=stage8, stage11=stage11, stage12=stage12, root=root)
    _check_manifest_compatibility(stage_root, root_manifest)
    condition_manifests = {}
    for condition in EVALUATED_CONDITIONS:
        run_dir, run_manifest = _make_condition_manifest(
            condition=condition, stage_root=stage_root, root_manifest=root_manifest, backbone=backbone, c_best=c_best,
            data=data, stage8=stage8, stage11=stage11, stage12=stage12, root=root,
        )
        _check_manifest_compatibility(run_dir, run_manifest)
        condition_manifests[condition] = (run_dir, run_manifest)
    return {"backbone": backbone, "c_best": c_best, "data": data, "stage8": stage8, "stage11": stage11,
            "stage12": stage12, "stage_root": stage_root, "root_manifest": root_manifest,
            "condition_manifests": condition_manifests}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 13: Experiment II-B availability-aware shallow-pool generalization.")
    parser.add_argument("--validate-only", action="store_true", help="Verify all upstream data, decision, model and config artifacts without inference or output writes.")
    parser.add_argument("--disable-progress", action="store_true", help="Reserved for consistent notebook/runner interfaces; inference is batch-loop based.")
    args = parser.parse_args(argv)
    os.chdir(PROJECT_ROOT)
    audit = validate_prerequisites(root=PROJECT_ROOT)
    print("Stage 13 prerequisite audit passed.")
    print(f"  Selected backbone: {audit['backbone']} ({eu.get_backbone(audit['backbone']).hf_model_id})")
    print(f"  Selected C_best: C{audit['c_best']}")
    print(f"  Shallow-pool rows: {len(audit['data']['shallow'])}")
    print(f"  max_length_context: {audit['stage8']['sequence']['max_length_context']}")
    print(f"  Output root: {audit['stage_root']}")
    if args.validate_only:
        print("Validation-only complete; no inference was run and no Stage 13 output marker was written.")
        return 0

    eu.audit_pipeline_script_names(PROJECT_ROOT, require_all=True)
    stage_root = audit["stage_root"]
    root_manifest = audit["root_manifest"]
    eu.prepare_run_directory(stage_root, root_manifest, allow_resume=True)
    if _verify_complete_stage(stage_root=stage_root, root_manifest=root_manifest, backbone=audit["backbone"], c_best=audit["c_best"], root=PROJECT_ROOT):
        print("Stage 13 already completed and all summary hashes match the current upstream configuration; no inference rerun needed.")
        return 0

    predictions: dict[int, dict[str, Any]] = {}
    for condition in EVALUATED_CONDITIONS:
        run_dir, manifest = audit["condition_manifests"][condition]
        eu.prepare_run_directory(run_dir, manifest, allow_resume=True)
        saved = _verify_condition_output(run_dir, manifest, backbone=audit["backbone"], condition=condition, root=PROJECT_ROOT)
        population = eligible_population(audit["data"]["shallow"], condition)
        if saved is not None:
            pred_df = saved["predictions_df"]
            expected_ids = population[eu.CTX_ID_COLUMN].astype(str).tolist()
            if pred_df[eu.CTX_ID_COLUMN].astype(str).tolist() != expected_ids:
                raise ValueError(f"Verified C{condition} outputs do not match the current eligible population order")
            predictions[condition] = {
                "predictions_df": pred_df, "probabilities": saved["probabilities"],
                "y_pred": label_ids(pred_df["y_pred"]), "metrics": saved["metrics"], "population": population,
                "post_ids": expected_ids,
            }
            print(f"[verified resume] Stage 13 {audit['backbone']} C{condition}: hashes and post IDs match")
            continue

        print(f"Evaluating {audit['backbone']} C{condition} on {len(population)} eligible shallow-pool examples...")
        model_dir = audit["stage12"]["checkpoint_infos"][condition]["model_dir"]
        predicted = _predict_condition(
            condition=condition, population=population, model_dir=model_dir,
            max_length=int(audit["stage8"]["sequence"]["max_length_context"]),
            disable_tqdm=args.disable_progress,
        )
        y_true = label_ids(population["label"])
        saved = _save_condition_output(
            run_dir=run_dir, manifest=manifest, backbone=audit["backbone"], condition=condition,
            population=population, y_true=y_true, prediction=predicted, root=PROJECT_ROOT,
        )
        predictions[condition] = {
            "predictions_df": saved["predictions_df"], "probabilities": saved["probabilities"],
            "y_pred": label_ids(saved["predictions_df"]["y_pred"]), "metrics": saved["metrics"],
            "population": population, "post_ids": saved["predictions_df"][eu.CTX_ID_COLUMN].astype(str).tolist(),
        }

    tables = _make_summary_tables(
        backbone=audit["backbone"], c_best=audit["c_best"], predictions=predictions, data=audit["data"],
        stage11=audit["stage11"], stage12=audit["stage12"],
    )
    _publish_summary(
        stage_root=stage_root, root_manifest=root_manifest, backbone=audit["backbone"], c_best=audit["c_best"],
        tables=tables, predictions=predictions, stage11=audit["stage11"], stage12=audit["stage12"], root=PROJECT_ROOT,
    )
    print("\nStage 13 complete. Shallow-pool metrics are descriptive only; Stage 11 C_best is unchanged.")
    print(tables["e2b_generalization_metrics.csv"].to_string(index=False))
    print("\nSummary outputs:")
    for name in SUMMARY_TABLES:
        print(f"  {stage_root / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
