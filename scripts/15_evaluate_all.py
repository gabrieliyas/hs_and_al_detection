"""Stage 15: cross-experiment quantitative evaluation and strategy selection.

This CPU-only stage validates and consolidates results from Experiments I, II-A,
II-B, and III. It reports the Stage 11 backbone/context decisions without
reselecting either, and selects ITFT versus direct fine-tuning separately for
the target-only (C0) and context-aware (Stage 11 C_best) deployment modes.

Authoritative result bundle:
    results/runs/left_truncation_v2/stage_15/
Thesis-ready tables:
    results/tables/left_truncation_v2/
Thesis-ready figures:
    results/figures/left_truncation_v2/
Canonical downstream strategy decision:
    results/training_strategy_decision.json

The report uses macro-F1 as the primary inferential metric. OOF point estimates
for contextual experiments are computed from seed-ensembled probabilities
aligned to the same 900 post IDs. Fold/seed means and standard deviations are
supplementary variability summaries, not the primary point estimates. The
Stage 13 shallow-pool evaluation remains descriptive and cannot change C_best.

Run after upstream stages 1-14 have completed and the repository's complete
pipeline audit passes:
    python scripts/15_evaluate_all.py --validate-only
    python scripts/15_evaluate_all.py
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score, f1_score


def _project_root() -> Path:
    here = Path(__file__).resolve().parent
    return here.parent if here.name.casefold() == "scripts" else here


PROJECT_ROOT = _project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import experiment_utils as eu  # noqa: E402

STAGE = 15
SCRIPT_NAME = "scripts/15_evaluate_all.py"
N_BOOTSTRAP = 10_000
CI_LEVEL = 95.0
BOOTSTRAP_SEED = 42
BACKBONES = tuple(eu.BACKBONES)
CONDITIONS = tuple(range(5))
FOLDS = tuple(range(5))
SEEDS_E1 = tuple(eu.EVAL_SEEDS_EXP1)
SEEDS_CV = tuple(eu.CV_SEEDS)
EXPECTED_MATCHED_ROWS = int(eu.CONTEXT_MATCHED_EXPECTED_ROWS)
EXPECTED_SHALLOW_ROWS = int(eu.CONTEXT_SHALLOW_EXPECTED_ROWS)

RUN_ROOT = eu.versioned_run_root(PROJECT_ROOT)
STAGE_ROOT = RUN_ROOT / "stage_15"
TABLES_EXPORT_DIR = PROJECT_ROOT / "results/tables/left_truncation_v2"
FIGURES_EXPORT_DIR = PROJECT_ROOT / "results/figures/left_truncation_v2"
STRATEGY_DECISION_PATH = PROJECT_ROOT / eu.TRAINING_STRATEGY_DECISION_PATH

BENCHMARK_TRAIN = PROJECT_ROOT / "data/processed/ib_train.csv"
BENCHMARK_VAL = PROJECT_ROOT / "data/processed/ib_val.csv"
BENCHMARK_TEST = PROJECT_ROOT / "data/processed/ib_test.csv"
MATCHED_PATH = PROJECT_ROOT / "data/processed/ctx_matched_subset.csv"
SHALLOW_PATH = PROJECT_ROOT / "data/processed/ctx_shallow_pool.csv"
SEQUENCE_CONFIG_PATH = PROJECT_ROOT / eu.SEQUENCE_LENGTH_CONFIG
LR_CONFIG_PATH = PROJECT_ROOT / eu.LEARNING_RATE_CONFIG
STAGE5_DIR = RUN_ROOT / "experiment_i/evaluation"
STAGE8_DIR = RUN_ROOT / "stage_08/summary"
STAGE11_DIR = RUN_ROOT / "stage_11/decision"

STAGE5_OUTPUTS = (
    "e1_seed_metrics.csv", "e1_results.csv", "e1_ensemble_metrics.csv",
    "e1_paired_bootstrap.csv", "e1_confusion_matrix_indobertweet.csv",
    "e1_confusion_matrix_xlmr.csv", "e1_summary.json",
)
STAGE13_TABLES = (
    "e2b_generalization_metrics.csv", "e2b_generalization_by_availability.csv",
    "e2b_generalization_by_class.csv", "e2b_generalization_confusion_matrices.csv",
    "e2b_generalization_pairwise_vs_c0.csv", "e2b_generalization_manifest.csv",
)
STAGE14_TABLES = (
    "e3_grid_metrics.csv", "e3_context_summary.csv", "e3_grid_class_metrics.csv",
    "e3_confusion_matrices.csv", "e3_manifest.json",
)

TABLE_FILENAMES = (
    # Compact thesis-facing tables.
    "dataset_class_distribution.csv",
    "context_population_distribution.csv",
    "e1_model_summary.csv",
    "e1_seed_metrics.csv",
    "e1_ensemble_metrics.csv",
    "e1_paired_comparison.csv",
    "e2_context_summary.csv",
    "e2_context_pairwise.csv",
    "backbone_decision_summary.csv",
    "e2b_generalization_summary.csv",
    "e2b_generalization_by_availability.csv",
    "e3_direct_summary.csv",
    "training_strategy_comparison.csv",
    "e2_e3_matched_run_comparison.csv",
    "e2_e3_matched_run_summary.csv",
    # Detailed/appendix-friendly metrics kept in long or wider-form tables.
    "e1_model_summary_detailed.csv",
    "e1_seed_per_class_metrics.csv",
    "e1_ensemble_per_class_metrics.csv",
    "e2_context_metrics_detailed.csv",
    "e2_context_per_class.csv",
    "e2_cv_run_metrics.csv",
    "backbone_decision_detailed.csv",
    "e2b_generalization_metrics_detailed.csv",
    "e2b_generalization_by_class.csv",
    "e3_direct_metrics_detailed.csv",
    "e3_direct_per_class.csv",
    "e3_cv_run_metrics.csv",
    "training_strategy_details.csv",
    "confusion_matrices.csv",
    "run_provenance.csv",
)
FIGURE_FILENAMES = (
    "fig_e1_backbone_macro_f1.png",
    "fig_e2_context_depth_macro_f1.png",
    "fig_e2_per_class_f1.png",
    "fig_e2b_shallow_generalization.png",
    "fig_training_strategy_comparison.png",
)


# ---------------------------------------------------------------------------
# Small, pure helpers; these are also tested independently using synthetic data.
# ---------------------------------------------------------------------------
def _sha(path: Path) -> str:
    return eu.sha256_file(path)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read valid JSON artifact {path}: {exc}") from exc
    if not isinstance(doc, dict):
        raise ValueError(f"Expected a JSON object at {path}")
    return doc


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


def _label_ids(values: Sequence[Any] | pd.Series, *, where: str = "labels") -> np.ndarray:
    series = pd.Series(values)
    mapped = series.astype(str).map(eu.LABEL2ID)
    if mapped.isna().any():
        bad = sorted(series[mapped.isna()].astype(str).unique().tolist())
        raise ValueError(f"{where} contains non-canonical labels: {bad}")
    return mapped.to_numpy(dtype=np.int64)


def validate_probabilities(probabilities: Any, *, expected_rows: int, name: str) -> np.ndarray:
    values = np.asarray(probabilities)
    expected_shape = (expected_rows, len(eu.LABEL2ID))
    if values.shape != expected_shape:
        raise ValueError(f"{name} shape {values.shape}; expected {expected_shape}")
    if not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all():
        raise ValueError(f"{name} contains non-numeric or non-finite values")
    values = values.astype(np.float64, copy=False)
    if (values < -1e-6).any() or (values > 1.0 + 1e-6).any():
        raise ValueError(f"{name} contains values outside [0, 1]")
    if not np.allclose(values.sum(axis=1), 1.0, atol=1e-4, rtol=1e-4):
        raise ValueError(f"{name} rows do not sum to 1")
    return values


def classification_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> dict[str, Any]:
    """One consistent metric contract for the three-class single-label task."""
    truth = np.asarray(y_true, dtype=np.int64).reshape(-1)
    pred = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    if len(truth) == 0 or len(truth) != len(pred):
        raise ValueError("y_true/y_pred must have equal, non-zero length")
    expected_ids = set(eu.ID2LABEL)
    if not set(np.unique(truth)).issubset(expected_ids) or not set(np.unique(pred)).issubset(expected_ids):
        raise ValueError(f"Labels must be IDs in {sorted(expected_ids)}")
    labels = [eu.LABEL2ID[label] for label in ("N", "AL", "HS")]
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
        truth, pred, labels=labels, average="macro", zero_division=0
    )
    p_weighted, r_weighted, f_weighted, _ = precision_recall_fscore_support(
        truth, pred, labels=labels, average="weighted", zero_division=0
    )
    p_micro, r_micro, f_micro, _ = precision_recall_fscore_support(
        truth, pred, labels=labels, average="micro", zero_division=0
    )
    p_cls, r_cls, f_cls, support = precision_recall_fscore_support(
        truth, pred, labels=labels, average=None, zero_division=0
    )
    out: dict[str, Any] = {
        "n_examples": int(len(truth)),
        "accuracy": float(accuracy_score(truth, pred)),
        "macro_precision": float(p_macro),
        "macro_recall": float(r_macro),
        "macro_f1": float(f_macro),
        "weighted_precision": float(p_weighted),
        "weighted_recall": float(r_weighted),
        "weighted_f1": float(f_weighted),
        # Retained in the detailed output only; in single-label multiclass
        # classification, micro P/R/F1 equal accuracy and are redundant.
        "micro_precision": float(p_micro),
        "micro_recall": float(r_micro),
        "micro_f1": float(f_micro),
    }
    for idx, label in enumerate(("N", "AL", "HS")):
        out[f"precision_{label}"] = float(p_cls[idx])
        out[f"recall_{label}"] = float(r_cls[idx])
        out[f"f1_{label}"] = float(f_cls[idx])
        out[f"support_{label}"] = int(support[idx])
    return out


def confusion_long_rows(
    *, analysis: str, backbone: str, context_size: int | None,
    y_true: Sequence[int], y_pred: Sequence[int], seed: int | str = "ensemble",
    fold: int | None = None,
) -> list[dict[str, Any]]:
    """Nine-cell long-form confusion matrix; machine-readable and pivotable."""
    labels = [eu.LABEL2ID[label] for label in ("N", "AL", "HS")]
    cm = confusion_matrix(np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int), labels=labels)
    rows = []
    for true_id, true_label in enumerate(("N", "AL", "HS")):
        for pred_id, pred_label in enumerate(("N", "AL", "HS")):
            rows.append({
                "analysis": analysis, "backbone": backbone,
                "context_size": context_size, "fold": fold, "seed": seed,
                "true_label": true_label, "predicted_label": pred_label,
                "count": int(cm[true_id, pred_id]),
            })
    return rows


def build_oof_arrays(
    *, canonical_post_ids: Sequence[Any], canonical_y_true: Sequence[int],
    fold_post_ids: Mapping[int, Sequence[Any]],
    fold_probabilities: Mapping[tuple[int, int], np.ndarray],
    seeds: Sequence[int], context_size: int,
) -> dict[str, Any]:
    """Pool fold outputs per seed into the canonical ordered OOF post sequence.

    `fold_probabilities[(fold, seed)]` has exactly the row order represented by
    `fold_post_ids[fold]`. Each post must appear once per seed, never in more
    than one held-out fold. The returned seed arrays are aligned to
    `canonical_post_ids` and can be averaged safely.
    """
    ids = [str(value) for value in canonical_post_ids]
    if len(ids) == 0 or len(set(ids)) != len(ids):
        raise ValueError("Canonical post IDs must be non-empty and unique")
    truth = np.asarray(canonical_y_true, dtype=np.int64)
    if len(truth) != len(ids):
        raise ValueError("Canonical labels and IDs differ in length")
    id_to_index = {post_id: idx for idx, post_id in enumerate(ids)}
    arrays_by_seed: dict[int, np.ndarray] = {}
    for seed in seeds:
        pooled = np.full((len(ids), len(eu.LABEL2ID)), np.nan, dtype=np.float64)
        seen: set[str] = set()
        for fold in sorted(fold_post_ids):
            fold_ids = [str(value) for value in fold_post_ids[fold]]
            key = (fold, seed)
            if key not in fold_probabilities:
                raise KeyError(f"Missing fold probabilities for context C{context_size}, fold {fold}, seed {seed}")
            probs = validate_probabilities(
                fold_probabilities[key], expected_rows=len(fold_ids),
                name=f"C{context_size}/fold{fold}/seed{seed}",
            )
            if len(set(fold_ids)) != len(fold_ids):
                raise ValueError(f"Duplicate post IDs in fold {fold}")
            unknown = sorted(set(fold_ids) - set(id_to_index))
            if unknown:
                raise ValueError(f"Fold {fold} contains unknown post IDs: {unknown[:5]}")
            duplicate_across_folds = seen.intersection(fold_ids)
            if duplicate_across_folds:
                raise ValueError(f"OOF post IDs appear in multiple folds: {sorted(duplicate_across_folds)[:5]}")
            seen.update(fold_ids)
            indices = np.asarray([id_to_index[post_id] for post_id in fold_ids], dtype=int)
            if np.isfinite(pooled[indices]).any():
                raise ValueError("A post received more than one held-out probability row")
            pooled[indices] = probs
        if seen != set(ids):
            raise ValueError(
                f"C{context_size}/seed{seed}: pooled OOF post IDs do not cover the canonical set "
                f"(missing={len(set(ids)-seen)}, extra={len(seen-set(ids))})"
            )
        if not np.isfinite(pooled).all():
            raise ValueError(f"C{context_size}/seed{seed}: incomplete OOF probability matrix")
        arrays_by_seed[int(seed)] = pooled
    ensemble_probs = np.mean(np.stack(list(arrays_by_seed.values()), axis=0), axis=0)
    ensemble_pred = ensemble_probs.argmax(axis=1).astype(np.int64)
    return {
        "post_ids": ids,
        "y_true": truth,
        "probabilities_by_seed": arrays_by_seed,
        "ensemble_probabilities": ensemble_probs,
        "ensemble_predictions": ensemble_pred,
        "metrics": classification_metrics(truth, ensemble_pred),
    }


def _verify_manifest_inputs(manifest: Mapping[str, Any], *, root: Path) -> None:
    input_hashes = manifest.get("input_sha256")
    if not isinstance(input_hashes, Mapping):
        raise ValueError("Run manifest has no input_sha256 map")
    for relative, expected in input_hashes.items():
        path = Path(str(relative))
        path = path if path.is_absolute() else root / path
        if not path.is_file():
            raise FileNotFoundError(f"Manifest input is missing: {path}")
        if _sha(path) != expected:
            raise ValueError(f"Input hash mismatch for {path}; upstream artifact changed")


def _verify_artifact_index(
    *, index_path: Path, artifact_root: Path, expected_identity: Mapping[str, Any],
    check_paths: bool = True,
) -> dict[str, Any]:
    index = _read_json(index_path)
    for key, expected in expected_identity.items():
        if str(index.get(key)) != str(expected):
            raise ValueError(f"Artifact index {index_path} has invalid {key}: {index.get(key)!r}; expected {expected!r}")
    hashes = index.get("sha256")
    if not isinstance(hashes, Mapping):
        raise ValueError(f"Artifact index {index_path} has no sha256 mapping")
    if check_paths:
        for name, expected_hash in hashes.items():
            path = artifact_root / str(name)
            if not path.is_file():
                raise FileNotFoundError(f"Indexed artifact is missing: {path}")
            if _sha(path) != expected_hash:
                raise ValueError(f"Artifact hash mismatch: {path}")
    return index


def _verify_stage_root(
    *, stage_dir: Path, stage: int, script_name: str,
    required_files: Sequence[str], marker_name: str | None = None,
    index_name: str = "artifacts_sha256.json", check_input_hashes: bool = True, check_index_paths: bool = True,
    check_index_script_name: bool = True,
) -> dict[str, Any]:
    manifest_path = stage_dir / eu.RUN_MANIFEST_NAME
    index_path = stage_dir / index_name
    missing = [str(path) for path in [manifest_path, index_path] if not path.is_file()]
    missing.extend(str(stage_dir / name) for name in required_files if not (stage_dir / name).is_file())
    if marker_name and not (stage_dir / marker_name).is_file():
        missing.append(str(stage_dir / marker_name))
    if missing:
        raise FileNotFoundError(f"Stage {stage} bundle is incomplete: {missing}")
    manifest = _read_json(manifest_path)
    eu._validate_manifest_integrity(manifest)
    fingerprint = str(manifest["fingerprint"])
    if str(manifest.get("stage")) != str(stage) or manifest.get("script_name") != script_name:
        raise ValueError(f"Manifest identity mismatch in {stage_dir}: stage/script are not Stage {stage}/{script_name}")
    # Validate current source fingerprints and pipeline registration.
    eu.assert_current_run_artifact(
        stage_dir / required_files[0], expected_stage=stage,
        expected_script=script_name, expected_fingerprint=fingerprint,
    )
    if check_input_hashes:
        _verify_manifest_inputs(manifest, root=PROJECT_ROOT)
    expected_identity = {
        "schema_version": eu.CONFIG_SCHEMA_VERSION, "run_id": eu.RUN_ID,
        "stage": stage, "script_name": script_name, "manifest_fingerprint": fingerprint,
    }
    if not check_index_script_name:
        # Stage 8 artifact index schema intentionally omits script_name; its
        # manifest and completion marker still bind it to the registered script.
        expected_identity.pop("script_name", None)
    index = _verify_artifact_index(index_path=index_path, artifact_root=stage_dir, expected_identity=expected_identity, check_paths=check_index_paths)
    marker = None
    if marker_name:
        marker = _read_json(stage_dir / marker_name)
        if marker.get("manifest_fingerprint") != fingerprint:
            raise ValueError(f"Completion marker does not match manifest in {stage_dir}")
        if marker.get("artifact_index_sha256") != _sha(index_path):
            raise ValueError(f"Completion marker's artifact-index hash is stale in {stage_dir}")
    return {"dir": stage_dir, "manifest": manifest, "fingerprint": fingerprint, "index": index, "marker": marker}


def _verify_leaf_run(
    *, run_dir: Path, stage: int, script_name: str,
    expected_identity: Mapping[str, Any], expected_artifacts: Sequence[str],
    marker_name: str | None = "run_complete.json", require_marker: bool = True,
) -> dict[str, Any]:
    manifest_path = run_dir / eu.RUN_MANIFEST_NAME
    index_path = run_dir / "artifacts_sha256.json"
    missing = [str(path) for path in [manifest_path, index_path] if not path.is_file()]
    missing.extend(str(run_dir / name) for name in expected_artifacts if not (run_dir / name).is_file())
    if require_marker and marker_name and not (run_dir / marker_name).is_file():
        missing.append(str(run_dir / marker_name))
    if missing:
        raise FileNotFoundError(f"Run bundle is incomplete: {missing}")
    manifest = _read_json(manifest_path)
    eu._validate_manifest_integrity(manifest)
    fingerprint = str(manifest["fingerprint"])
    if str(manifest.get("stage")) != str(stage) or manifest.get("script_name") != script_name:
        raise ValueError(f"Run manifest identity mismatch in {run_dir}")
    eu.assert_current_run_artifact(run_dir / expected_artifacts[0], expected_stage=stage, expected_script=script_name, expected_fingerprint=fingerprint)
    _verify_manifest_inputs(manifest, root=PROJECT_ROOT)
    identity = {
        "schema_version": eu.CONFIG_SCHEMA_VERSION, "run_id": eu.RUN_ID,
        "stage": stage, "script_name": script_name,
        "manifest_fingerprint": fingerprint, **dict(expected_identity),
    }
    index = _verify_artifact_index(index_path=index_path, artifact_root=run_dir, expected_identity=identity)
    marker = None
    if marker_name and (run_dir / marker_name).exists():
        marker = _read_json(run_dir / marker_name)
        for key, expected in identity.items():
            if key in marker and str(marker.get(key)) != str(expected):
                raise ValueError(f"Run completion marker mismatch for {key} in {run_dir}")
        if marker.get("artifact_index_sha256") and marker.get("artifact_index_sha256") != _sha(index_path):
            raise ValueError(f"Run completion marker has stale index hash: {run_dir}")
    elif require_marker:
        raise FileNotFoundError(f"Missing {marker_name} in {run_dir}")
    return {"dir": run_dir, "manifest": manifest, "fingerprint": fingerprint, "index": index, "marker": marker}


def _read_prediction_frame(path: Path, *, ids_column: str, expected_ids: Sequence[Any], expected_y: Sequence[int], name: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {ids_column, "y_true", "y_pred"}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"{path} lacks columns {sorted(missing)}")
    ids = frame[ids_column].astype(str).tolist()
    expected = [str(value) for value in expected_ids]
    if ids != expected:
        raise ValueError(f"{name} post-ID order differs from the canonical held-out rows")
    y_true = _label_ids(frame["y_true"], where=f"{name}.y_true")
    expected_true = np.asarray(expected_y, dtype=np.int64)
    if not np.array_equal(y_true, expected_true):
        raise ValueError(f"{name} y_true does not match the canonical data labels")
    y_pred = _label_ids(frame["y_pred"], where=f"{name}.y_pred")
    frame = frame.copy()
    frame["_y_true_id"] = y_true
    frame["_y_pred_id"] = y_pred
    return frame


def _class_distribution_tables(benchmark_parts: Mapping[str, pd.DataFrame], matched: pd.DataFrame, shallow: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name, frame in [*benchmark_parts.items(), ("contextual_all", pd.concat([matched, shallow], ignore_index=True)), ("contextual_matched", matched), ("contextual_shallow", shallow)]:
        counts = frame["label"].astype(str).value_counts().to_dict()
        total = len(frame)
        for label in ("N", "AL", "HS"):
            support = int(counts.get(label, 0))
            rows.append({"dataset_partition": name, "class_code": label, "class_name": eu.ID2LABEL[eu.LABEL2ID[label]], "support_n": support, "support_pct": (100.0 * support / total if total else 0.0), "n_examples": total})
    return pd.DataFrame(rows)


def _context_population_distribution(matched: pd.DataFrame, shallow: pd.DataFrame) -> pd.DataFrame:
    """Long-form support table for available parents x context-needed status x class."""
    frame = pd.concat([matched, shallow], ignore_index=True)
    required = {"context_size_available", "context_needed", "label"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"Contextual dataset is missing distribution columns: {missing}")
    frame["context_size_available"] = pd.to_numeric(frame["context_size_available"], errors="raise").astype(int)
    frame["context_needed"] = pd.to_numeric(frame["context_needed"], errors="raise").astype(int)
    frame["label"] = frame["label"].astype(str)
    context_totals = frame.groupby("context_size_available", sort=True).size().to_dict()
    rows: list[dict[str, Any]] = []
    grouped = frame.groupby(["context_size_available", "context_needed", "label"], dropna=False, sort=True).size()
    for (available, needed, label), support in grouped.items():
        rows.append({
            "context_size_available": int(available), "context_needed": int(needed),
            "class_code": str(label), "class_name": eu.ID2LABEL[eu.LABEL2ID[str(label)]],
            "support_n": int(support),
            "support_pct_within_available_context": 100.0 * int(support) / int(context_totals[int(available)]),
            "available_context_total_n": int(context_totals[int(available)]),
        })
    return pd.DataFrame(rows)


def _metric_row(analysis: str, backbone: str, context_size: int | None, metrics: Mapping[str, Any], **metadata: Any) -> dict[str, Any]:
    # Compact table schema. Per-class details are written to a separate table.
    row = {
        "analysis": analysis, "backbone": backbone, "experiment_id": metadata.pop("experiment_id", ""),
        "context_size": context_size, "n_examples": int(metrics["n_examples"]),
        "accuracy": float(metrics["accuracy"]),
        "macro_precision": float(metrics["macro_precision"]), "macro_recall": float(metrics["macro_recall"]),
        "macro_f1": float(metrics["macro_f1"]),
        "weighted_precision": float(metrics["weighted_precision"]), "weighted_recall": float(metrics["weighted_recall"]),
        "weighted_f1": float(metrics["weighted_f1"]),
    }
    row.update(metadata)
    return row


def _per_class_rows(analysis: str, backbone: str, context_size: int | None, metrics: Mapping[str, Any], **metadata: Any) -> list[dict[str, Any]]:
    rows = []
    for label in ("N", "AL", "HS"):
        rows.append({
            "analysis": analysis, "backbone": backbone, "context_size": context_size,
            "class_code": label, "precision": metrics[f"precision_{label}"],
            "recall": metrics[f"recall_{label}"], "f1": metrics[f"f1_{label}"],
            "support": metrics[f"support_{label}"], **metadata,
        })
    return rows


def _verify_stage5_bundle() -> dict[str, Any]:
    bundle = _verify_stage_root(
        stage_dir=STAGE5_DIR, stage=5, script_name=eu.PIPELINE_SCRIPTS[5],
        required_files=STAGE5_OUTPUTS, marker_name=None,
    )
    # Stage 5 has no separate completion marker; the output inventory itself
    # and manifest fingerprint are authoritative.
    return bundle


def _verify_stage11_bundle() -> dict[str, Any]:
    root_decision = PROJECT_ROOT / eu.BACKBONE_DECISION_PATH
    bundle = _verify_stage_root(
        stage_dir=STAGE11_DIR, stage=11, script_name=eu.PIPELINE_SCRIPTS[11],
        required_files=("backbone_decision.json", "context_selection_indobertweet.json", "context_selection_xlmr.json"),
        marker_name="stage11_complete.json",
    )
    versioned_decision = STAGE11_DIR / "backbone_decision.json"
    if not root_decision.is_file():
        raise FileNotFoundError(f"Canonical Stage 11 decision missing: {root_decision}")
    if _sha(root_decision) != _sha(versioned_decision):
        raise ValueError("Canonical and versioned Stage 11 decision documents differ")
    decision = eu.load_backbone_decision_document(root_decision)
    if decision.get("stage11_manifest_fingerprint") != bundle["fingerprint"]:
        raise ValueError("Stage 11 decision references a different Stage 11 manifest")
    bundle["decision"] = decision
    bundle["canonical_decision_path"] = root_decision
    return bundle


def _load_e1_results(test_df: pd.DataFrame, stage5: Mapping[str, Any]) -> dict[str, Any]:
    y_true = _label_ids(test_df["label"], where="ib_test.csv")
    row_ids = np.arange(len(test_df), dtype=np.int64)
    seed_metrics_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    per_seed_probs: dict[str, list[np.ndarray]] = {backbone: [] for backbone in BACKBONES}
    per_seed_predictions: dict[str, list[np.ndarray]] = {backbone: [] for backbone in BACKBONES}
    input_paths: list[Path] = [BENCHMARK_TEST]

    for backbone in BACKBONES:
        stage = 3 if backbone == "indobertweet" else 4
        script_name = eu.PIPELINE_SCRIPTS[stage]
        base_dir = RUN_ROOT / "experiment_i" / backbone
        for seed in SEEDS_E1:
            run_dir = base_dir / f"seed_{seed}"
            leaf = _verify_leaf_run(
                run_dir=run_dir, stage=stage, script_name=script_name,
                expected_identity={},
                expected_artifacts=("predictions.csv", "probabilities.npy", "metrics.json", "artifacts_sha256.json"),
                marker_name=None, require_marker=False,
            )
            run_config = leaf["manifest"].get("config", {})
            for key, expected in {"backbone": backbone, "seed": int(seed), "input_mode": "target_only"}.items():
                if str(run_config.get(key)) != str(expected):
                    raise ValueError(f"Experiment I manifest config mismatch for {key} in {run_dir}")
            pred_path = run_dir / "predictions.csv"
            probs_path = run_dir / "probabilities.npy"
            metrics_path = run_dir / "metrics.json"
            preds = pd.read_csv(pred_path)
            if "row_id" not in preds.columns or preds["row_id"].astype(int).tolist() != row_ids.tolist():
                raise ValueError(f"{pred_path} row_id does not match row order of current ib_test.csv")
            mapped_true = _label_ids(preds["y_true"], where=f"{pred_path}.y_true")
            mapped_pred = _label_ids(preds["y_pred"], where=f"{pred_path}.y_pred")
            if not np.array_equal(mapped_true, y_true):
                raise ValueError(f"{pred_path} y_true does not match current ib_test.csv")
            probabilities = validate_probabilities(np.load(probs_path, allow_pickle=False), expected_rows=len(test_df), name=str(probs_path))
            if not np.array_equal(probabilities.argmax(axis=1), mapped_pred):
                raise ValueError(f"{pred_path} predictions disagree with probabilities.npy argmax")
            metrics = classification_metrics(y_true, mapped_pred)
            metrics_doc = _read_json(metrics_path)
            if metrics_doc.get("run_id") != eu.RUN_ID or metrics_doc.get("backbone") != backbone or int(metrics_doc.get("seed", -1)) != seed:
                raise ValueError(f"Metrics identity mismatch in {metrics_path}")
            per_seed_probs[backbone].append(probabilities)
            per_seed_predictions[backbone].append(mapped_pred)
            row = {
                "backbone": backbone, "experiment_id": eu.experiment1_id(backbone), "seed": int(seed),
                **{k: v for k, v in metrics.items() if not k.startswith("precision_") and not k.startswith("recall_") and not k.startswith("f1_") and not k.startswith("support_")},
                **{k: v for k, v in metrics.items() if k.startswith(("precision_", "recall_", "f1_", "support_"))},
            }
            seed_metrics_rows.append(row)
            confusion_rows.extend(confusion_long_rows(analysis="E1_seed", backbone=backbone, context_size=None, y_true=y_true, y_pred=mapped_pred, seed=seed))
            input_paths.extend([run_dir / name for name in (eu.RUN_MANIFEST_NAME, "predictions.csv", "probabilities.npy", "metrics.json", "artifacts_sha256.json")])

    seed_metrics_df = pd.DataFrame(seed_metrics_rows)
    compact_summary_rows: list[dict[str, Any]] = []
    ensemble_rows: list[dict[str, Any]] = []
    ensemble_predictions: dict[str, np.ndarray] = {}
    ensemble_probs: dict[str, np.ndarray] = {}
    for backbone in BACKBONES:
        sub = seed_metrics_df[seed_metrics_df["backbone"] == backbone]
        summary = {"backbone": backbone, "experiment_id": eu.experiment1_id(backbone), "n_seeds": len(sub), "n_test": len(y_true)}
        for metric in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_precision", "weighted_recall", "weighted_f1"):
            values = sub[metric].astype(float).to_numpy()
            summary[f"{metric}_mean"] = float(values.mean())
            summary[f"{metric}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        for label in ("N", "AL", "HS"):
            values = sub[f"f1_{label}"].astype(float).to_numpy()
            summary[f"f1_{label}_mean"] = float(values.mean())
            summary[f"f1_{label}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        compact_summary_rows.append(summary)
        mean_probs = np.mean(np.stack(per_seed_probs[backbone], axis=0), axis=0)
        pred = mean_probs.argmax(axis=1)
        ensemble_predictions[backbone] = pred
        ensemble_probs[backbone] = mean_probs
        ens_metrics = classification_metrics(y_true, pred)
        ensemble_rows.append({"backbone": backbone, "experiment_id": eu.experiment1_id(backbone), **ens_metrics})
        confusion_rows.extend(confusion_long_rows(analysis="E1_seed_ensemble", backbone=backbone, context_size=None, y_true=y_true, y_pred=pred, seed="ensemble"))

    stage5_seed = pd.read_csv(STAGE5_DIR / "e1_seed_metrics.csv")
    # Cross-check shared primary results from Stage 5 rather than silently
    # changing the previously reported Experiment I result.
    if "macro_f1" in stage5_seed.columns:
        for backbone in BACKBONES:
            prior = stage5_seed[stage5_seed.get("backbone", pd.Series(index=stage5_seed.index, dtype=object)) == backbone]
            if not prior.empty and "seed" in prior.columns:
                for row in seed_metrics_rows:
                    if row["backbone"] == backbone:
                        old = prior[prior["seed"].astype(int) == row["seed"]]
                        if len(old) and not math.isclose(float(old.iloc[0]["macro_f1"]), float(row["macro_f1"]), rel_tol=1e-7, abs_tol=1e-7):
                            raise ValueError(f"Stage 15 recomputed E1 macro-F1 differs from Stage 5 for {backbone}, seed {row['seed']}")
    e1_bootstrap = pd.read_csv(STAGE5_DIR / "e1_paired_bootstrap.csv")
    macro_rows = e1_bootstrap[e1_bootstrap["metric"].astype(str).eq("macro_f1")]
    if len(macro_rows) != 1:
        raise ValueError("Stage 5 must contain exactly one primary macro-F1 paired-bootstrap row")
    macro_bootstrap = macro_rows.iloc[0].to_dict()
    direction = str(macro_bootstrap.get("comparison", ""))
    if "XLMR" not in direction and "xlmr" not in direction.lower():
        raise ValueError("Stage 5 macro-F1 bootstrap comparison direction is not XLM-R minus IndoBERTweet")
    # Verify Stage 5 ensembled point estimates equal the probability-based re-computation.
    stage5_ensemble = pd.read_csv(STAGE5_DIR / "e1_ensemble_metrics.csv")
    for backbone in BACKBONES:
        match = stage5_ensemble[stage5_ensemble["backbone"] == backbone]
        if len(match) != 1 or not math.isclose(float(match.iloc[0]["macro_f1"]), float(ensemble_rows[BACKBONES.index(backbone)]["macro_f1"]), rel_tol=1e-7, abs_tol=1e-7):
            raise ValueError(f"Stage 5 ensemble macro-F1 mismatch for {backbone}")

    paired_rows = []
    for _, old_row in e1_bootstrap.iterrows():
        metric = str(old_row["metric"])
        paired_rows.append({
            "comparison": "XLM-R minus IndoBERTweet", "metric": metric,
            "observed_difference": float(old_row["observed_diff"]),
            "ci_lower": float(old_row["ci_lower"]), "ci_upper": float(old_row["ci_upper"]),
            "ci_level": float(old_row["ci_level"]), "n_bootstrap": int(old_row["n_bootstrap"]),
            "inferential_status": "primary" if metric == "macro_f1" else "descriptive_unadjusted",
        })
    return {
        "seed_metrics": seed_metrics_df,
        "summary": pd.DataFrame(compact_summary_rows),
        "ensemble": pd.DataFrame(ensemble_rows),
        "paired": pd.DataFrame(paired_rows),
        "ensemble_predictions": ensemble_predictions,
        "ensemble_probabilities": ensemble_probs,
        "y_true": y_true,
        "confusion_rows": confusion_rows,
        "input_paths": input_paths,
    }


def _load_context_data() -> dict[str, Any]:
    for path in (MATCHED_PATH, SHALLOW_PATH, BENCHMARK_TRAIN, BENCHMARK_VAL, BENCHMARK_TEST, SEQUENCE_CONFIG_PATH, LR_CONFIG_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Required Stage 15 dataset/config is missing: {path}")
    matched_raw = pd.read_csv(MATCHED_PATH)
    shallow_raw = pd.read_csv(SHALLOW_PATH)
    matched_audit = eu.validate_context_dataset(matched_raw, expected_rows=EXPECTED_MATCHED_ROWS)
    shallow_audit = eu.validate_context_shallow_pool(shallow_raw, expected_rows=EXPECTED_SHALLOW_ROWS)
    # Stage 9/10 and Stage 14 canonicalize by post_id before creating each
    # fold prediction file. Stage 15 must use precisely that same row order.
    matched = matched_raw.sort_values(eu.CTX_ID_COLUMN, key=lambda col: col.astype(str), kind="mergesort").reset_index(drop=True)
    shallow = shallow_raw.sort_values(eu.CTX_ID_COLUMN, key=lambda col: col.astype(str), kind="mergesort").reset_index(drop=True)
    if len(matched) != EXPECTED_MATCHED_ROWS or len(shallow) != EXPECTED_SHALLOW_ROWS:
        raise ValueError(f"Expected matched/shallow rows {EXPECTED_MATCHED_ROWS}/{EXPECTED_SHALLOW_ROWS}; got {len(matched)}/{len(shallow)}")
    needed = {eu.CTX_ID_COLUMN, eu.CTX_FOLD_COLUMN, "label", "context_size_available", "target_text", "parent_4", "parent_3", "parent_2", "parent_1"}
    if not needed.issubset(matched.columns):
        raise KeyError(f"Matched contextual data lacks {sorted(needed-set(matched.columns))}")
    if eu.CTX_ID_COLUMN not in shallow or "label" not in shallow or "context_size_available" not in shallow:
        raise KeyError("Shallow pool must contain post_id, label and context_size_available")
    if matched[eu.CTX_ID_COLUMN].astype(str).duplicated().any() or shallow[eu.CTX_ID_COLUMN].astype(str).duplicated().any():
        raise ValueError("Matched subset and shallow pool must each have unique post IDs")
    if set(matched[eu.CTX_ID_COLUMN].astype(str)) & set(shallow[eu.CTX_ID_COLUMN].astype(str)):
        raise ValueError("Matched subset and shallow pool are not disjoint")
    if set(matched["label"].astype(str).unique()) != set(eu.LABEL2ID) or set(shallow["label"].astype(str).unique()) != set(eu.LABEL2ID):
        raise ValueError("Contextual populations must contain all three canonical labels")
    if sorted(matched[eu.CTX_FOLD_COLUMN].astype(int).value_counts().to_list()) != [180]*5:
        raise ValueError("Matched subset must have five shared folds of 180 examples")
    if not (matched["context_size_available"].astype(int) == 4).all():
        raise ValueError("Matched subset must contain only complete four-parent examples")
    if shallow["context_size_available"].astype(int).value_counts().sort_index().to_dict() != {1: 150, 2: 300, 3: 450}:
        raise ValueError("Shallow pool parent-availability distribution differs from the frozen protocol")
    parts = {
        "benchmark_train": pd.read_csv(BENCHMARK_TRAIN),
        "benchmark_val": pd.read_csv(BENCHMARK_VAL),
        "benchmark_test": pd.read_csv(BENCHMARK_TEST),
    }
    for name, part in parts.items():
        if "label" not in part or set(part["label"].astype(str).unique()) != set(eu.LABEL2ID):
            raise ValueError(f"{name} is missing canonical labels or one of the three classes")
    return {
        "matched": matched, "shallow": shallow, "matched_audit": matched_audit, "shallow_audit": shallow_audit, "benchmark_parts": parts,
        "matched_ids": matched[eu.CTX_ID_COLUMN].astype(str).tolist(),
        "matched_y": _label_ids(matched["label"], where="ctx_matched_subset.csv"),
        "folds": {fold: matched.loc[matched[eu.CTX_FOLD_COLUMN].astype(int) == fold, eu.CTX_ID_COLUMN].astype(str).tolist() for fold in FOLDS},
        "fold_y": {fold: _label_ids(matched.loc[matched[eu.CTX_FOLD_COLUMN].astype(int) == fold, "label"], where=f"matched fold {fold}") for fold in FOLDS},
    }


def _verify_cv_run(*, backbone: str, condition: int, fold: int, seed: int, matched_data: Mapping[str, Any], expected_stage8_fingerprint: str | None = None) -> dict[str, Any]:
    stage = 9 if backbone == "indobertweet" else 10
    run_dir = RUN_ROOT / f"stage_{stage:02d}" / f"e2_{backbone}" / f"C{condition}" / f"fold_{fold:02d}" / f"seed_{seed}"
    run = _verify_leaf_run(
        run_dir=run_dir, stage=stage, script_name=eu.PIPELINE_SCRIPTS[stage],
        expected_identity={"backbone": backbone, "context_size": condition, "fold": fold, "seed": seed},
        expected_artifacts=("predictions.csv", "probabilities.npy", "metrics.json", "artifacts_sha256.json"),
        marker_name="run_complete.json", require_marker=True,
    )
    config = run["manifest"].get("config", {})
    expected_dataset_sha = _sha(MATCHED_PATH)
    fold_assignment_sha = matched_data["matched_audit"]["fold_assignment_sha256"]
    checks = {
        "dataset_sha256": expected_dataset_sha,
        "fold_assignment_sha256": fold_assignment_sha,
        "max_length_context": int(eu.load_json_config(SEQUENCE_CONFIG_PATH)["max_length_context"]),
        "truncation_side": "left",
        "context_size": condition,
        "fold": fold,
        "seed": seed,
        "backbone": backbone,
    }
    if expected_stage8_fingerprint is not None:
        checks["stage8_manifest_fingerprint"] = expected_stage8_fingerprint
    for field, expected in checks.items():
        if str(config.get(field)) != str(expected):
            raise ValueError(f"Experiment II-A run {run_dir} manifest config {field}={config.get(field)!r}; expected {expected!r}")

    expected_ids = matched_data["folds"][fold]
    expected_y = matched_data["fold_y"][fold]
    pred_df = _read_prediction_frame(run_dir / "predictions.csv", ids_column=eu.CTX_ID_COLUMN, expected_ids=expected_ids, expected_y=expected_y, name=f"{backbone}/C{condition}/fold{fold}/seed{seed}")
    probs = validate_probabilities(np.load(run_dir / "probabilities.npy", allow_pickle=False), expected_rows=len(expected_ids), name=str(run_dir / "probabilities.npy"))
    if not np.array_equal(probs.argmax(axis=1), pred_df["_y_pred_id"].to_numpy(dtype=np.int64)):
        raise ValueError(f"Predictions and probabilities disagree in {run_dir}")
    metric_doc = _read_json(run_dir / "metrics.json")
    if metric_doc.get("backbone") != backbone or int(metric_doc.get("context_size", -1)) != condition or int(metric_doc.get("fold", -1)) != fold or int(metric_doc.get("seed", -1)) != seed:
        raise ValueError(f"Metrics identity mismatch in {run_dir}")
    if int(metric_doc.get("n_held_out", -1)) != len(expected_ids):
        raise ValueError(f"Metrics held-out count mismatch in {run_dir}")
    for name, actual in (("dataset_sha256", expected_dataset_sha), ("fold_assignment_sha256", fold_assignment_sha), ("max_length_context", checks["max_length_context"]), ("truncation_side", "left")):
        if str(metric_doc.get(name)) != str(actual):
            raise ValueError(f"Metrics {name} mismatch in {run_dir}")
    return {"run_dir": run_dir, "predictions": pred_df, "probabilities": probs, "metrics": classification_metrics(expected_y, pred_df["_y_pred_id"].to_numpy()), "manifest_fingerprint": run["fingerprint"], "manifest": run["manifest"]}


def _load_e2_oof(matched_data: Mapping[str, Any], *, stage11_decision: Mapping[str, Any]) -> dict[str, Any]:
    """Rebuild pooled OOF arrays from the two fully manifested Stage 9/10 grids.

    The current Stage 9/10 scripts deliberately do not create a separate root
    manifest for the aggregate CSV. The aggregate JSON binds the CSV hash and
    ordered run-manifest fingerprints; each of the 50 leaf runs remains the
    authoritative manifested unit. Stage 15 verifies all leaf outputs, then
    recomputes the utility grid fingerprint and compares it with Stage 11.
    """
    result: dict[str, Any] = {"by_backbone": {}, "input_paths": [], "confusion_rows": []}
    expected_dataset_sha = _sha(MATCHED_PATH)
    expected_fold_sha = matched_data["matched_audit"]["fold_assignment_sha256"]
    stage8_manifest_path = STAGE8_DIR / eu.RUN_MANIFEST_NAME
    stage8_manifest = _read_json(stage8_manifest_path)
    stage8_fingerprint = str(stage8_manifest["fingerprint"])
    sequence_doc = eu.load_json_config(SEQUENCE_CONFIG_PATH)

    for backbone in BACKBONES:
        stage = 9 if backbone == "indobertweet" else 10
        stage_root = RUN_ROOT / f"stage_{stage:02d}" / f"e2_{backbone}"
        summary_path = stage_root / "e2_grid_metrics.csv"
        summary_json_path = stage_root / "e2_grid_summary.json"
        if not summary_path.is_file() or not summary_json_path.is_file():
            raise FileNotFoundError(f"Stage {stage} aggregate grid summary is incomplete under {stage_root}")
        summary_doc = _read_json(summary_json_path)
        expected_summary = {
            "schema_version": eu.CONFIG_SCHEMA_VERSION,
            "run_id": eu.RUN_ID,
            "backbone": backbone,
            "stage": stage,
            "script_name": eu.PIPELINE_SCRIPTS[stage],
            "complete": True,
            "n_runs": 50,
            "expected_runs": 50,
            "contexts": list(CONDITIONS),
            "folds": list(FOLDS),
            "seeds": list(SEEDS_CV),
            "dataset_sha256": expected_dataset_sha,
            "fold_assignment_sha256": expected_fold_sha,
            "max_length_context": int(sequence_doc["max_length_context"]),
            "truncation_side": "left",
            "stage8_manifest_fingerprint": stage8_fingerprint,
        }
        for key, expected in expected_summary.items():
            if summary_doc.get(key) != expected:
                raise ValueError(f"Stage {stage} grid summary metadata mismatch for {key}: {summary_doc.get(key)!r} != {expected!r}")
        if summary_doc.get("grid_metrics_sha256") != _sha(summary_path):
            raise ValueError(f"Stage {stage} grid summary CSV hash mismatch")

        pooled_for_backbone: dict[int, dict[str, Any]] = {}
        per_run_metrics: list[dict[str, Any]] = []
        run_records_for_audit: list[dict[str, Any]] = []
        run_fingerprints: list[str] = []
        input_paths = [summary_path, summary_json_path]
        for condition in CONDITIONS:
            fold_ids = {fold: matched_data["folds"][fold] for fold in FOLDS}
            probs_by_fold_seed: dict[tuple[int, int], np.ndarray] = {}
            for fold in FOLDS:
                for seed in SEEDS_CV:
                    run = _verify_cv_run(
                        backbone=backbone, condition=condition, fold=fold, seed=seed,
                        matched_data=matched_data, expected_stage8_fingerprint=stage8_fingerprint,
                    )
                    probs_by_fold_seed[(fold, seed)] = run["probabilities"]
                    per_run_metrics.append({"backbone": backbone, "context_size": condition, "fold": fold, "seed": seed, **run["metrics"]})
                    result["confusion_rows"].extend(confusion_long_rows(
                        analysis="E2A_fold_seed_OOF", backbone=backbone, context_size=condition,
                        y_true=run["predictions"]["_y_true_id"].to_numpy(dtype=np.int64),
                        y_pred=run["predictions"]["_y_pred_id"].to_numpy(dtype=np.int64), seed=seed, fold=fold,
                    ))
                    run_fingerprints.append(run["manifest_fingerprint"])
                    run_records_for_audit.append({"artifact_path": run["run_dir"] / "run_complete.json", "manifest_fingerprint": run["manifest_fingerprint"]})
                    input_paths.extend([run["run_dir"] / name for name in (eu.RUN_MANIFEST_NAME, "run_complete.json", "artifacts_sha256.json", "predictions.csv", "probabilities.npy", "metrics.json")])
            pooled = build_oof_arrays(
                canonical_post_ids=matched_data["matched_ids"], canonical_y_true=matched_data["matched_y"],
                fold_post_ids=fold_ids, fold_probabilities=probs_by_fold_seed,
                seeds=SEEDS_CV, context_size=condition,
            )
            pooled_for_backbone[condition] = pooled

        # Aggregate row columns follow the current Stage 9/10 schema: context_size,
        # not the legacy column name `condition`.
        table = pd.read_csv(summary_path)
        required = {"context_size", "fold", "seed", "macro_f1", "n_held_out", "manifest_fingerprint"}
        if not required.issubset(table.columns):
            raise KeyError(f"{summary_path} lacks {sorted(required-set(table.columns))}")
        if len(table) != 50 or table.duplicated(["context_size", "fold", "seed"]).any():
            raise ValueError(f"{summary_path} must contain exactly the 50 unique context/fold/seed runs")
        observed_keys = {(int(row.context_size), int(row.fold), int(row.seed)) for row in table.itertuples()}
        expected_keys = {(c, f, seed) for c in CONDITIONS for f in FOLDS for seed in SEEDS_CV}
        if observed_keys != expected_keys:
            raise ValueError(f"{summary_path} has missing or unexpected run keys")
        for record in per_run_metrics:
            key = (record["context_size"], record["fold"], record["seed"])
            row = table[(table["context_size"].astype(int) == key[0]) & (table["fold"].astype(int) == key[1]) & (table["seed"].astype(int) == key[2])]
            if len(row) != 1 or not math.isclose(float(row.iloc[0]["macro_f1"]), record["macro_f1"], rel_tol=1e-6, abs_tol=1e-6):
                raise ValueError(f"Stage {stage} aggregate summary disagrees with verified run metrics for {key}")
        if table["n_held_out"].astype(int).ne(180).any():
            raise ValueError(f"{summary_path} must report 180 held-out examples for every fold run")
        if table["manifest_fingerprint"].astype(str).tolist() != run_fingerprints:
            raise ValueError(f"Stage {stage} summary run-manifest fingerprints do not align with the verified run grid")
        grid_audit = eu.audit_context_cv_grid(
            run_records_for_audit, backbone=backbone, required_folds=FOLDS,
            required_seeds=SEEDS_CV, required_contexts=CONDITIONS,
        )
        if grid_audit["dataset_sha256"] != expected_dataset_sha or grid_audit["fold_assignment_sha256"] != expected_fold_sha:
            raise ValueError(f"Stage {stage} independently audited grid uses a different dataset/fold assignment")
        expected_stage11_grid_fp = stage11_decision.get("source_grid_fingerprints", {}).get(backbone)
        if not expected_stage11_grid_fp or grid_audit["grid_fingerprint"] != expected_stage11_grid_fp:
            raise ValueError(f"Stage {stage} grid fingerprint differs from the grid selected by Stage 11 for {backbone}")
        aggregate_fingerprint = eu.sha256_json({
            "stage": stage, "backbone": backbone, "grid_metrics_sha256": _sha(summary_path),
            "summary_json_sha256": _sha(summary_json_path), "grid_fingerprint": grid_audit["grid_fingerprint"],
        })
        result["by_backbone"][backbone] = {
            "pooled": pooled_for_backbone, "run_metrics": pd.DataFrame(per_run_metrics),
            "root_dir": stage_root, "root_fingerprint": aggregate_fingerprint,
            "grid_fingerprint": grid_audit["grid_fingerprint"], "grid_audit": grid_audit,
        }
        result["input_paths"].extend(input_paths)
    return result

def _verify_stage12_13(stage11: Mapping[str, Any], data: Mapping[str, Any]) -> dict[str, Any]:
    decision = stage11["decision"]
    backbone = str(decision["decision"])
    c_best = int(decision["c_best_by_backbone"][backbone])
    stage12_dir = RUN_ROOT / "stage_12" / f"e2_{backbone}"
    stage12 = _verify_stage_root(
        stage_dir=stage12_dir, stage=12, script_name=eu.PIPELINE_SCRIPTS[12],
        required_files=("e2_full_retrain_metrics.csv", "e2_full_retrain_summary.json"), marker_name="stage12_complete.json", check_index_paths=False,
    )
    s12_summary = _read_json(stage12_dir / "e2_full_retrain_summary.json")
    if s12_summary.get("backbone") != backbone or int(s12_summary.get("selected_context_size_c_best", -1)) != c_best:
        raise ValueError("Stage 12 summary does not match Stage 11 selected backbone/C_best")
    retrain = pd.read_csv(stage12_dir / "e2_full_retrain_metrics.csv")
    if len(retrain) != 5 or set(retrain["context_size"].astype(int)) != set(CONDITIONS):
        raise ValueError("Stage 12 must contain five complete C0-C4 retraining records")
    if not (retrain["n_train"].astype(int) == EXPECTED_MATCHED_ROWS).all():
        raise ValueError("Stage 12 retraining metrics must show all 900 matched examples for every condition")
    # Stage 12's index also records checkpoint-manifest hashes held under
    # models/e2_full_retrained/<backbone>/C<n>, outside the stage bundle.
    stage12_index = stage12["index"].get("sha256", {})
    for rel_name, expected_hash in stage12_index.items():
        if rel_name.startswith("checkpoint_manifests/"):
            condition = int(rel_name.rsplit("C", 1)[1])
            actual_path = PROJECT_ROOT / "models" / "e2_full_retrained" / backbone / f"C{condition}" / "retrain_checkpoint_manifest.json"
        else:
            actual_path = stage12_dir / rel_name
        if not actual_path.is_file() or _sha(actual_path) != expected_hash:
            raise ValueError(f"Stage 12 indexed artifact is missing/stale: {actual_path}")

    stage13_dir = RUN_ROOT / "stage_13" / f"e2_{backbone}"
    stage13 = _verify_stage_root(
        stage_dir=stage13_dir, stage=13, script_name=eu.PIPELINE_SCRIPTS[13],
        required_files=(*STAGE13_TABLES, "e2b_generalization_summary.json"), marker_name="stage13_complete.json",
    )
    s13_summary = _read_json(stage13_dir / "e2b_generalization_summary.json")
    if s13_summary.get("backbone") != backbone or int(s13_summary.get("selected_context_size_c_best", -1)) != c_best:
        raise ValueError("Stage 13 summary does not match the Stage 11 backbone/C_best")
    if s13_summary.get("stage12_manifest_fingerprint") != stage12["fingerprint"]:
        raise ValueError("Stage 13 summary refers to a different Stage 12 manifest")
    if stage13["marker"].get("stage12_manifest_fingerprint") != stage12["fingerprint"]:
        raise ValueError("Stage 13 completion marker refers to a different Stage 12 manifest")

    shallow = data["shallow"].copy()
    overall_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    availability_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    condition_predictions: dict[int, tuple[pd.DataFrame, np.ndarray]] = {}
    population_sizes = {0: 900, 1: 900, 2: 750, 3: 450}
    for condition in (0, 1, 2, 3):
        if condition <= 1:
            population = shallow.copy().reset_index(drop=True)
        else:
            population = shallow[shallow["context_size_available"].astype(int) >= condition].copy().reset_index(drop=True)
        run_dir = stage13_dir / f"C{condition}"
        # The Stage 13 root artifact index covers all files in each C0-C3 leaf.
        pred_path = run_dir / "predictions.csv"
        probs_path = run_dir / "probabilities.npy"
        metrics_path = run_dir / "metrics.json"
        leaf_manifest = _read_json(run_dir / eu.RUN_MANIFEST_NAME)
        eu.assert_current_run_artifact(pred_path, expected_stage=13, expected_script=eu.PIPELINE_SCRIPTS[13], expected_fingerprint=str(leaf_manifest["fingerprint"]))
        leaf_index = _verify_artifact_index(
            index_path=run_dir / "artifacts_sha256.json", artifact_root=run_dir,
            expected_identity={"schema_version": eu.CONFIG_SCHEMA_VERSION, "run_id": eu.RUN_ID, "stage": 13,
                             "script_name": eu.PIPELINE_SCRIPTS[13], "backbone": backbone, "context_size": condition,
                             "manifest_fingerprint": leaf_manifest["fingerprint"]},
        )
        for filename in ("predictions.csv", "probabilities.npy", "metrics.json"):
            if leaf_index.get("sha256", {}).get(filename) != _sha(run_dir / filename):
                raise ValueError(f"Stage 13 C{condition} hash mismatch for {filename}")
        pred_df = pd.read_csv(pred_path)
        if pred_df[eu.CTX_ID_COLUMN].astype(str).tolist() != population[eu.CTX_ID_COLUMN].astype(str).tolist():
            raise ValueError(f"Stage 13 C{condition} predictions are not aligned to the expected shallow population")
        y_true = _label_ids(population["label"], where=f"Stage 13 C{condition} population")
        y_pred = _label_ids(pred_df["y_pred"], where=f"Stage 13 C{condition} predictions")
        if not np.array_equal(_label_ids(pred_df["y_true"], where=f"Stage 13 C{condition} truth"), y_true):
            raise ValueError(f"Stage 13 C{condition} y_true differs from shallow dataset")
        probs = validate_probabilities(np.load(probs_path, allow_pickle=False), expected_rows=len(population), name=str(probs_path))
        if not np.array_equal(probs.argmax(axis=1), y_pred):
            raise ValueError(f"Stage 13 C{condition} probabilities and predictions disagree")
        metrics = classification_metrics(y_true, y_pred)
        condition_predictions[condition] = (population, y_pred)
        overall_rows.append(_metric_row("E2B_shallow", backbone, condition, metrics,
                                        experiment_id=eu.experiment2_id(condition, backbone),
                                        population_rule=("all shallow posts" if condition in (0, 1) else f"available parents >= {condition}"),
                                        status="evaluated", is_c_best=(condition == c_best)))
        class_rows.extend(_per_class_rows("E2B_shallow", backbone, condition, metrics, population_rule=("all shallow posts" if condition in (0, 1) else f"available parents >= {condition}")))
        confusion_rows.extend(confusion_long_rows(analysis="E2B_shallow", backbone=backbone, context_size=condition, y_true=y_true, y_pred=y_pred))
        for parent_count in sorted(population["context_size_available"].astype(int).unique()):
            mask = population["context_size_available"].astype(int).to_numpy() == parent_count
            group_metrics = classification_metrics(y_true[mask], y_pred[mask])
            availability_rows.append({"backbone": backbone, "context_size": condition, "parent_count_available": int(parent_count),
                                      "n_examples": int(mask.sum()), "accuracy": group_metrics["accuracy"],
                                      "macro_precision": group_metrics["macro_precision"], "macro_recall": group_metrics["macro_recall"],
                                      "macro_f1": group_metrics["macro_f1"], "weighted_precision": group_metrics["weighted_precision"],
                                      "weighted_recall": group_metrics["weighted_recall"], "weighted_f1": group_metrics["weighted_f1"]})
    overall_rows.append({"analysis": "E2B_shallow", "backbone": backbone, "experiment_id": eu.experiment2_id(4, backbone),
                         "context_size": 4, "n_examples": 0, "accuracy": np.nan, "macro_precision": np.nan, "macro_recall": np.nan,
                         "macro_f1": np.nan, "weighted_precision": np.nan, "weighted_recall": np.nan, "weighted_f1": np.nan,
                         "population_rule": "not applicable: shallow pool has no four-parent examples", "status": "not_applicable", "is_c_best": c_best == 4})
    # Cross-check Stage 13's published summary table against the recomputation.
    published = pd.read_csv(stage13_dir / "e2b_generalization_metrics.csv")
    if set(published["condition"].astype(int)) != set(CONDITIONS):
        raise ValueError("Stage 13 generalization summary must include C0-C4, with C4 marked not applicable")
    for condition in (0, 1, 2, 3):
        old = published[published["condition"].astype(int) == condition]
        if len(old) != 1 or not math.isclose(float(old.iloc[0]["macro_f1"]), float(overall_rows[condition]["macro_f1"]), rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(f"Stage 13 published macro-F1 differs from verified predictions for C{condition}")
    return {
        "backbone": backbone, "c_best": c_best, "stage12": stage12, "stage13": stage13,
        "overall": pd.DataFrame(overall_rows), "by_class": pd.DataFrame(class_rows),
        "by_availability": pd.DataFrame(availability_rows), "confusion_rows": confusion_rows,
        "predictions": condition_predictions,
        "input_paths": [stage12_dir / eu.RUN_MANIFEST_NAME, stage12_dir / "stage12_complete.json", stage12_dir / "artifacts_sha256.json", stage12_dir / "e2_full_retrain_metrics.csv", stage12_dir / "e2_full_retrain_summary.json",
                        stage13_dir / eu.RUN_MANIFEST_NAME, stage13_dir / "stage13_complete.json", stage13_dir / "artifacts_sha256.json", stage13_dir / "e2b_generalization_summary.json", *[stage13_dir / name for name in STAGE13_TABLES]],
    }


def _load_e3_oof(*, matched_data: Mapping[str, Any], backbone: str, c_best: int) -> dict[str, Any]:
    stage_dir = RUN_ROOT / "stage_14" / f"e3_{backbone}"
    bundle = _verify_stage_root(
        stage_dir=stage_dir, stage=14, script_name=eu.PIPELINE_SCRIPTS[14],
        required_files=STAGE14_TABLES, marker_name="stage14_complete.json",
    )
    manifest_doc = _read_json(stage_dir / "e3_manifest.json")
    if manifest_doc.get("backbone") != backbone or int(manifest_doc.get("selected_context_size_c_best", -1)) != c_best:
        raise ValueError("Stage 14 manifest does not match Stage 11 backbone/C_best")
    conditions = (0,) if c_best == 0 else (0, c_best)
    if set(int(value) for value in manifest_doc.get("conditions", [])) != set(conditions):
        raise ValueError("Stage 14 conditions do not match C0 and the Stage 11 selected context")
    by_condition: dict[int, dict[str, Any]] = {}
    run_metrics_rows = []
    confusion_rows = []
    input_paths: list[Path] = [stage_dir / name for name in (*STAGE14_TABLES, eu.RUN_MANIFEST_NAME, "stage14_complete.json", "artifacts_sha256.json")]
    for condition in conditions:
        prob_by_fold_seed: dict[tuple[int, int], np.ndarray] = {}
        for fold in FOLDS:
            expected_ids = matched_data["folds"][fold]
            expected_y = matched_data["fold_y"][fold]
            for seed in SEEDS_CV:
                run_dir = stage_dir / f"C{condition}" / f"fold_{fold:02d}" / f"seed_{seed}"
                leaf = _verify_leaf_run(
                    run_dir=run_dir, stage=14, script_name=eu.PIPELINE_SCRIPTS[14],
                    expected_identity={"backbone": backbone, "context_size": condition, "fold": fold, "seed": seed},
                    expected_artifacts=("predictions.csv", "probabilities.npy", "metrics.json", "artifacts_sha256.json"),
                    marker_name="run_complete.json", require_marker=True,
                )
                pred_df = _read_prediction_frame(run_dir / "predictions.csv", ids_column=eu.CTX_ID_COLUMN, expected_ids=expected_ids, expected_y=expected_y, name=f"E3/{backbone}/C{condition}/fold{fold}/seed{seed}")
                probs = validate_probabilities(np.load(run_dir / "probabilities.npy", allow_pickle=False), expected_rows=len(expected_ids), name=str(run_dir / "probabilities.npy"))
                y_pred = pred_df["_y_pred_id"].to_numpy(dtype=np.int64)
                confusion_rows.extend(confusion_long_rows(
                    analysis="E3_direct_fold_seed_OOF", backbone=backbone, context_size=condition,
                    y_true=expected_y, y_pred=y_pred, seed=seed, fold=fold,
                ))
                if not np.array_equal(probs.argmax(axis=1), y_pred):
                    raise ValueError(f"E3 predictions and probabilities disagree in {run_dir}")
                run_metrics_rows.append({"backbone": backbone, "context_size": condition, "fold": fold, "seed": seed,
                                         **classification_metrics(expected_y, y_pred)})
                prob_by_fold_seed[(fold, seed)] = probs
                input_paths.extend([run_dir / name for name in (eu.RUN_MANIFEST_NAME, "run_complete.json", "artifacts_sha256.json", "predictions.csv", "probabilities.npy", "metrics.json")])
        pooled = build_oof_arrays(
            canonical_post_ids=matched_data["matched_ids"], canonical_y_true=matched_data["matched_y"],
            fold_post_ids=matched_data["folds"], fold_probabilities=prob_by_fold_seed, seeds=SEEDS_CV, context_size=condition,
        )
        by_condition[condition] = pooled
        confusion_rows.extend(confusion_long_rows(analysis="E3_direct_OOF", backbone=backbone, context_size=condition,
                                                   y_true=matched_data["matched_y"], y_pred=pooled["ensemble_predictions"]))
    # Reconcile expected run count and point estimates against Stage 14's own metrics table.
    grid = pd.read_csv(stage_dir / "e3_grid_metrics.csv")
    expected_count = len(conditions) * len(FOLDS) * len(SEEDS_CV)
    if len(grid) != expected_count or set(grid["condition"].astype(int)) != set(conditions):
        raise ValueError(f"Stage 14 metrics grid expected {expected_count} runs for {conditions}; found {len(grid)}")
    for condition in conditions:
        for record in run_metrics_rows:
            if record["context_size"] != condition:
                continue
            row = grid[(grid["condition"].astype(int) == condition) & (grid["fold"].astype(int) == record["fold"]) & (grid["seed"].astype(int) == record["seed"])]
            if len(row) != 1 or not math.isclose(float(row.iloc[0]["macro_f1"]), float(record["macro_f1"]), rel_tol=1e-6, abs_tol=1e-6):
                raise ValueError(f"Stage 14 aggregate grid disagrees with verified run metrics for C{condition}/fold{record['fold']}/seed{record['seed']}")
    return {"backbone": backbone, "c_best": c_best, "by_condition": by_condition,
            "run_metrics": pd.DataFrame(run_metrics_rows), "confusion_rows": confusion_rows, "bundle": bundle,
            "input_paths": input_paths}


def _make_report_tables(*, upstream: Mapping[str, Any], data: Mapping[str, Any], e1: Mapping[str, Any], e2: Mapping[str, Any], stage11: Mapping[str, Any], e2b: Mapping[str, Any], e3: Mapping[str, Any]) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    decision = stage11["decision"]
    selected_backbone = str(decision["decision"])
    c_best_map = {key: int(value) for key, value in decision["c_best_by_backbone"].items()}
    selected_c_best = c_best_map[selected_backbone]

    dist_table = _class_distribution_tables(data["benchmark_parts"], data["matched"], data["shallow"])
    context_distribution_table = _context_population_distribution(data["matched"], data["shallow"])

    e2_summary_rows: list[dict[str, Any]] = []
    e2_class_rows: list[dict[str, Any]] = []
    e2_confusions: list[dict[str, Any]] = list(e2.get("confusion_rows", []))
    e2_context_pairs: list[dict[str, Any]] = []
    for backbone in BACKBONES:
        context_docs = decision["context_decisions"][backbone]
        provisional = int(context_docs["provisional_top_context"])
        selected = int(context_docs["selected_context"])
        run_metrics = e2["by_backbone"][backbone]["run_metrics"]
        for condition in CONDITIONS:
            pooled = e2["by_backbone"][backbone]["pooled"][condition]
            run_sub = run_metrics[run_metrics["context_size"] == condition]
            row = _metric_row("E2A_OOF", backbone, condition, pooled["metrics"], experiment_id=eu.experiment2_id(condition, backbone),
                              is_c_best=(condition == selected), provisional_top=(condition == provisional),
                              run_macro_f1_mean=float(run_sub["macro_f1"].mean()),
                              run_macro_f1_sd=float(run_sub["macro_f1"].std(ddof=1)),
                              n_folds=len(FOLDS), n_seeds=len(SEEDS_CV))
            e2_summary_rows.append(row)
            e2_class_rows.extend(_per_class_rows("E2A_OOF", backbone, condition, pooled["metrics"], is_c_best=(condition == selected)))
            e2_confusions.extend(confusion_long_rows(analysis="E2A_seed_ensembled_OOF", backbone=backbone, context_size=condition,
                                                       y_true=pooled["y_true"], y_pred=pooled["ensemble_predictions"]))
        pairwise = context_docs.get("paired_comparisons_top_minus_shallower", {})
        for comparison, result in pairwise.items():
            candidate = int(comparison.split("_minus_")[1].replace("C", ""))
            top = int(comparison.split("_minus_")[0].replace("C", ""))
            e2_context_pairs.append({
                "backbone": backbone, "provisional_top_context": top, "candidate_context": candidate,
                "comparison": comparison, "observed_f1_difference_top_minus_candidate": float(result["observed_diff"]),
                "ci_lower": float(result["ci_lower"]), "ci_upper": float(result["ci_upper"]),
                "ci_level": float(result["ci_level"]), "n_bootstrap": int(result["n_bootstrap"]),
                "ci_includes_zero": bool(result.get("includes_zero", result["ci_lower"] <= 0 <= result["ci_upper"])),
                "selected_context": selected, "selection_reason": context_docs["selection_reason"],
                "role": "predeclared_context_selection_comparison",
            })

    context_rows = []
    for backbone in BACKBONES:
        ctx = decision["context_decisions"][backbone]
        for condition in CONDITIONS:
            context_rows.append({"backbone": backbone, "provisional_top_context": int(ctx["provisional_top_context"]),
                                 "selected_c_best": int(ctx["selected_context"]), "context_size": condition,
                                 "pooled_oof_macro_f1": float(ctx["pooled_oof_macro_f1"][f"C{condition}"]),
                                 "is_selected": condition == int(ctx["selected_context"]),
                                 "selection_reason": ctx["selection_reason"], "n_oof_examples": int(ctx["n_oof_examples"])})
    backbone_summary = pd.DataFrame([{
        "backbone": backbone, "c_best": c_best_map[backbone],
        "selected": backbone == selected_backbone,
        "c0_macro_f1": float(decision["context_decisions"][backbone]["pooled_oof_macro_f1"]["C0"]),
        "c_best_macro_f1": float(decision["context_decisions"][backbone]["pooled_oof_macro_f1"][f"C{c_best_map[backbone]}"]),
        "median_latency_ms_at_c_best": float(decision["latency_ms"][backbone]),
        "decision": selected_backbone, "decision_reason": decision["reason"],
        "primary_difference_xlmr_minus_indobertweet": float(decision["primary_c_best"]["observed_diff"]),
        "primary_ci_lower": float(decision["primary_c_best"]["ci_lower"]),
        "primary_ci_upper": float(decision["primary_c_best"]["ci_upper"]),
        "c0_guard_difference": float(decision["guard_c0"]["observed_diff"]),
        "c0_guard_ci_lower": float(decision["guard_c0"]["ci_lower"]),
        "c0_guard_ci_upper": float(decision["guard_c0"]["ci_upper"]),
        "rule_version": decision["rule_version"],
    } for backbone in BACKBONES])

    # Stage 13 is descriptive. Recomputed above from the verified raw predictions.
    e2b_overall = e2b["overall"].copy()
    e2b_by_availability = e2b["by_availability"].copy()
    e2b_by_class = e2b["by_class"].copy()

    e3_summary_rows: list[dict[str, Any]] = []
    e3_class_rows: list[dict[str, Any]] = []
    e3_confusions: list[dict[str, Any]] = []
    e3_run = e3["run_metrics"]
    for condition, pooled in e3["by_condition"].items():
        run_sub = e3_run[e3_run["context_size"] == condition]
        e3_summary_rows.append(_metric_row("E3_direct_OOF", selected_backbone, condition, pooled["metrics"],
                                           experiment_id=eu.experiment3_id(condition, selected_backbone),
                                           is_c_best=(condition == selected_c_best), n_folds=len(FOLDS), n_seeds=len(SEEDS_CV),
                                           run_macro_f1_mean=float(run_sub["macro_f1"].mean()),
                                           run_macro_f1_sd=float(run_sub["macro_f1"].std(ddof=1))))
        e3_class_rows.extend(_per_class_rows("E3_direct_OOF", selected_backbone, condition, pooled["metrics"], is_c_best=(condition == selected_c_best)))
        e3_confusions.extend(confusion_long_rows(analysis="E3_direct_seed_ensembled_OOF", backbone=selected_backbone, context_size=condition,
                                                  y_true=pooled["y_true"], y_pred=pooled["ensemble_predictions"]))

    e1 = dict(e1)
    e2 = dict(e2)
    # The strategy selector requires strict common post-ID order and a common y_true.
    canonical_ids = data["matched_ids"]
    canonical_y = data["matched_y"]
    strategy_decisions: dict[str, dict[str, Any]] = {}
    strategy_rows: list[dict[str, Any]] = []
    for mode, condition in (("target_only", 0), ("context_aware", selected_c_best)):
        itft = e2["by_backbone"][selected_backbone]["pooled"][condition]
        direct = e3["by_condition"].get(condition)
        if direct is None:
            raise ValueError(f"Stage 14 lacks the required direct-fine-tuning condition C{condition}")
        if itft["post_ids"] != canonical_ids or direct["post_ids"] != canonical_ids:
            raise ValueError(f"ITFT/direct OOF post IDs differ for deployment mode {mode}")
        if not np.array_equal(itft["y_true"], canonical_y) or not np.array_equal(direct["y_true"], canonical_y):
            raise ValueError(f"ITFT/direct y_true arrays differ for deployment mode {mode}")
        decision_doc = eu.select_training_strategy(
            canonical_y, itft["ensemble_predictions"], direct["ensemble_predictions"],
            n_bootstrap=N_BOOTSTRAP, ci=CI_LEVEL, random_state=BOOTSTRAP_SEED,
            post_ids=canonical_ids, post_ids_itft=itft["post_ids"], post_ids_direct=direct["post_ids"],
            expected_n=EXPECTED_MATCHED_ROWS,
        )
        strategy_decisions[mode] = decision_doc
        bootstrap = decision_doc["bootstrap"]
        strategy_rows.append({
            "deployment_mode": mode, "backbone": selected_backbone, "context_size": condition,
            "n_oof_examples": len(canonical_ids), "itft_macro_f1": itft["metrics"]["macro_f1"],
            "direct_macro_f1": direct["metrics"]["macro_f1"],
            "difference_itft_minus_direct": bootstrap["observed_diff"],
            "ci_lower": bootstrap["ci_lower"], "ci_upper": bootstrap["ci_upper"],
            "ci_level": bootstrap["ci_level"], "n_bootstrap": bootstrap["n_bootstrap"],
            "selected_strategy": decision_doc["strategy"], "reason": decision_doc["reason"],
            "rule_version": decision_doc["rule_version"],
            "same_condition_as_target_only": condition == 0,
        })

    confusion_rows = list(e1["confusion_rows"]) + e2_confusions + e2b["confusion_rows"] + e3_confusions

    # Thesis-facing tables stay compact. Detailed per-seed/per-class and run-level
    # data are kept in separate appendix/reproducibility tables rather than one
    # very wide CSV that is difficult to format in Chapter IV.
    e1_seed_full = e1["seed_metrics"].copy()
    e1_seed_main_cols = [
        "backbone", "experiment_id", "seed", "n_examples", "accuracy", "macro_f1", "weighted_f1",
    ]
    e1_seed_main = e1_seed_full[[c for c in e1_seed_main_cols if c in e1_seed_full.columns]].copy()
    e1_seed_class_rows: list[dict[str, Any]] = []
    for row in e1_seed_full.to_dict(orient="records"):
        for label in ("N", "AL", "HS"):
            e1_seed_class_rows.append({
                "backbone": row["backbone"], "experiment_id": row["experiment_id"], "seed": int(row["seed"]),
                "class_code": label, "precision": row.get(f"precision_{label}"),
                "recall": row.get(f"recall_{label}"), "f1": row.get(f"f1_{label}"),
                "support": row.get(f"support_{label}"),
            })

    e1_summary_full = e1["summary"].copy()
    e1_summary_main_cols = [
        "backbone", "experiment_id", "n_seeds", "n_test", "accuracy_mean", "accuracy_sd",
        "macro_f1_mean", "macro_f1_sd", "weighted_f1_mean", "weighted_f1_sd",
    ]
    e1_summary_main = e1_summary_full[[c for c in e1_summary_main_cols if c in e1_summary_full.columns]].copy()

    e1_ensemble_full = e1["ensemble"].copy()
    e1_ensemble_main_cols = ["backbone", "experiment_id", "n_examples", "accuracy", "macro_f1", "weighted_f1"]
    e1_ensemble_main = e1_ensemble_full[[c for c in e1_ensemble_main_cols if c in e1_ensemble_full.columns]].copy()
    e1_ensemble_class_rows: list[dict[str, Any]] = []
    for row in e1_ensemble_full.to_dict(orient="records"):
        for label in ("N", "AL", "HS"):
            e1_ensemble_class_rows.append({
                "backbone": row["backbone"], "experiment_id": row["experiment_id"],
                "class_code": label, "precision": row.get(f"precision_{label}"),
                "recall": row.get(f"recall_{label}"), "f1": row.get(f"f1_{label}"),
                "support": row.get(f"support_{label}"),
            })

    e2_summary_full = pd.DataFrame(e2_summary_rows)
    e2_summary_main_cols = [
        "backbone", "experiment_id", "context_size", "n_examples", "accuracy", "macro_f1", "weighted_f1",
        "is_c_best", "run_macro_f1_mean", "run_macro_f1_sd", "n_folds", "n_seeds",
    ]
    e2_summary_main = e2_summary_full[[c for c in e2_summary_main_cols if c in e2_summary_full.columns]].copy()

    backbone_summary_full = backbone_summary.copy()
    backbone_summary_main_cols = [
        "backbone", "c_best", "c0_macro_f1", "c_best_macro_f1", "median_latency_ms_at_c_best",
        "selected", "decision_reason",
    ]
    backbone_summary_main = backbone_summary_full[[c for c in backbone_summary_main_cols if c in backbone_summary_full.columns]].copy()

    e2b_overall_full = e2b_overall.copy()
    e2b_summary_main_cols = [
        "backbone", "context_size", "status", "population_rule", "n_examples", "accuracy", "macro_f1", "weighted_f1", "is_c_best",
    ]
    e2b_summary_main = e2b_overall_full[[c for c in e2b_summary_main_cols if c in e2b_overall_full.columns]].copy()

    e3_summary_full = pd.DataFrame(e3_summary_rows)
    e3_summary_main_cols = [
        "backbone", "experiment_id", "context_size", "n_examples", "accuracy", "macro_f1", "weighted_f1",
        "is_c_best", "run_macro_f1_mean", "run_macro_f1_sd",
    ]
    e3_summary_main = e3_summary_full[[c for c in e3_summary_main_cols if c in e3_summary_full.columns]].copy()

    strategy_full = pd.DataFrame(strategy_rows)
    strategy_main_cols = [
        "deployment_mode", "backbone", "context_size", "n_oof_examples", "itft_macro_f1", "direct_macro_f1",
        "difference_itft_minus_direct", "ci_lower", "ci_upper", "selected_strategy", "reason",
    ]
    strategy_main = strategy_full[[c for c in strategy_main_cols if c in strategy_full.columns]].copy()

    e2_run_frames = []
    for backbone in BACKBONES:
        frame = e2["by_backbone"][backbone]["run_metrics"].copy()
        frame.insert(0, "experiment_id", [eu.experiment2_id(int(c), backbone) for c in frame["context_size"]])
        e2_run_frames.append(frame)
    e2_run_metrics = pd.concat(e2_run_frames, ignore_index=True).sort_values(
        ["backbone", "context_size", "fold", "seed"]
    ).reset_index(drop=True)
    # Direct/ITFT run-level comparisons are descriptive and matched by the same
    # context, fold, and seed. Formal strategy selection still uses seed-ensembled
    # pooled OOF predictions and the paired bootstrap above.
    strategy_conditions = sorted(set([0, selected_c_best]))
    e2_selected_runs = e2_run_metrics[
        (e2_run_metrics["backbone"] == selected_backbone)
        & (e2_run_metrics["context_size"].astype(int).isin(strategy_conditions))
    ].copy()
    e3_run_metrics = e3["run_metrics"].copy()
    if not e3_run_metrics.empty:
        e3_run_metrics.insert(0, "experiment_id", [eu.experiment3_id(int(c), selected_backbone) for c in e3_run_metrics["context_size"]])
        e3_run_metrics = e3_run_metrics.sort_values(["context_size", "fold", "seed"]).reset_index(drop=True)
    matched_run_comparison = e2_selected_runs.merge(
        e3_run_metrics, on=["context_size", "fold", "seed"], how="inner",
        suffixes=("_itft", "_direct"), validate="one_to_one",
    )
    expected_matched_run_rows = len(strategy_conditions) * len(FOLDS) * len(SEEDS_CV)
    if len(matched_run_comparison) != expected_matched_run_rows:
        raise ValueError(
            f"Expected {expected_matched_run_rows} matched E2/E3 run rows, found {len(matched_run_comparison)}"
        )
    for metric in ("accuracy", "macro_f1", "weighted_f1"):
        matched_run_comparison[f"{metric}_delta_direct_minus_itft"] = (
            matched_run_comparison[f"{metric}_direct"] - matched_run_comparison[f"{metric}_itft"]
        )
    matched_run_comparison["backbone"] = selected_backbone
    matched_run_comparison["itft_experiment_id"] = [eu.experiment2_id(int(c), selected_backbone) for c in matched_run_comparison["context_size"]]
    matched_run_comparison["direct_experiment_id"] = [eu.experiment3_id(int(c), selected_backbone) for c in matched_run_comparison["context_size"]]
    matched_run_comparison = matched_run_comparison.sort_values(["context_size", "fold", "seed"]).reset_index(drop=True)
    run_summary_rows = []
    for condition, sub in matched_run_comparison.groupby("context_size", sort=True):
        row = {"backbone": selected_backbone, "context_size": int(condition), "n_matched_runs": int(len(sub))}
        for metric in ("accuracy", "macro_f1", "weighted_f1"):
            for strategy in ("itft", "direct"):
                values = sub[f"{metric}_{strategy}"].astype(float)
                row[f"{strategy}_{metric}_mean"] = float(values.mean())
                row[f"{strategy}_{metric}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            delta = sub[f"{metric}_delta_direct_minus_itft"].astype(float)
            row[f"delta_{metric}_direct_minus_itft_mean"] = float(delta.mean())
            row[f"delta_{metric}_direct_minus_itft_sd"] = float(delta.std(ddof=1)) if len(delta) > 1 else 0.0
        run_summary_rows.append(row)
    matched_run_summary = pd.DataFrame(run_summary_rows)

    tables = {
        "dataset_class_distribution.csv": dist_table,
        "context_population_distribution.csv": context_distribution_table,
        "e1_model_summary.csv": e1_summary_main,
        "e1_seed_metrics.csv": e1_seed_main,
        "e1_ensemble_metrics.csv": e1_ensemble_main,
        "e1_paired_comparison.csv": e1["paired"],
        "e2_context_summary.csv": e2_summary_main,
        "e2_context_pairwise.csv": pd.DataFrame(e2_context_pairs),
        "backbone_decision_summary.csv": backbone_summary_main,
        "e2b_generalization_summary.csv": e2b_summary_main,
        "e2b_generalization_by_availability.csv": e2b_by_availability,
        "e3_direct_summary.csv": e3_summary_main,
        "training_strategy_comparison.csv": strategy_main,
        "e2_e3_matched_run_comparison.csv": matched_run_comparison,
        "e2_e3_matched_run_summary.csv": matched_run_summary,
        "e1_model_summary_detailed.csv": e1_summary_full,
        "e1_seed_per_class_metrics.csv": pd.DataFrame(e1_seed_class_rows),
        "e1_ensemble_per_class_metrics.csv": pd.DataFrame(e1_ensemble_class_rows),
        "e2_context_metrics_detailed.csv": e2_summary_full,
        "e2_context_per_class.csv": pd.DataFrame(e2_class_rows),
        "e2_cv_run_metrics.csv": e2_run_metrics,
        "backbone_decision_detailed.csv": backbone_summary_full,
        "e2b_generalization_metrics_detailed.csv": e2b_overall_full,
        "e2b_generalization_by_class.csv": e2b_by_class,
        "e3_direct_metrics_detailed.csv": e3_summary_full,
        "e3_direct_per_class.csv": pd.DataFrame(e3_class_rows),
        "e3_cv_run_metrics.csv": e3_run_metrics,
        "training_strategy_details.csv": strategy_full,
        "confusion_matrices.csv": pd.DataFrame(confusion_rows),
    }

    stage2_fp = str(upstream["stage2_manifest"]["fingerprint"])
    stage1_fp = str(upstream["stage1_manifest"]["fingerprint"])
    stage1b_fp = str(upstream["stage1b_manifest"]["fingerprint"])
    stage5_fp = str(upstream["stage5"]["fingerprint"])
    stage8_fp = str(upstream["stage8"]["fingerprint"])
    provenance_rows = [
        {"stage": 1, "artifact_group": "benchmark preprocessing", "path": "results/runs/left_truncation_v2/stage_01", "manifest_fingerprint": stage1_fp},
        {"stage": "1b", "artifact_group": "benchmark sequence-length selection", "path": "results/runs/left_truncation_v2/stage_1b", "manifest_fingerprint": stage1b_fp},
        {"stage": 2, "artifact_group": "contextual preprocessing and folds", "path": "results/runs/left_truncation_v2/stage_02", "manifest_fingerprint": stage2_fp},
        {"stage": 5, "artifact_group": "Experiment I evaluation", "path": STAGE5_DIR.relative_to(PROJECT_ROOT).as_posix(), "manifest_fingerprint": stage5_fp},
        {"stage": 8, "artifact_group": "contextual config/latency", "path": STAGE8_DIR.relative_to(PROJECT_ROOT).as_posix(), "manifest_fingerprint": stage8_fp},
        {"stage": 11, "artifact_group": "backbone/context decision", "path": STAGE11_DIR.relative_to(PROJECT_ROOT).as_posix(), "manifest_fingerprint": stage11["fingerprint"]},
        {"stage": 12, "artifact_group": "full-subset retraining", "path": e2b["stage12"]["dir"].relative_to(PROJECT_ROOT).as_posix(), "manifest_fingerprint": e2b["stage12"]["fingerprint"]},
        {"stage": 13, "artifact_group": "shallow-pool generalization", "path": e2b["stage13"]["dir"].relative_to(PROJECT_ROOT).as_posix(), "manifest_fingerprint": e2b["stage13"]["fingerprint"]},
        {"stage": 14, "artifact_group": "direct fine-tuning OOF", "path": e3["bundle"]["dir"].relative_to(PROJECT_ROOT).as_posix(), "manifest_fingerprint": e3["bundle"]["fingerprint"]},
    ]
    for backbone in BACKBONES:
        stage = 9 if backbone == "indobertweet" else 10
        provenance_rows.append({"stage": stage, "artifact_group": f"Experiment II-A {backbone}",
                                "path": e2["by_backbone"][backbone]["root_dir"].relative_to(PROJECT_ROOT).as_posix(),
                                "manifest_fingerprint": e2["by_backbone"][backbone]["root_fingerprint"]})
    tables["run_provenance.csv"] = pd.DataFrame(provenance_rows)

    # Compact figure data is saved in tables; supplementary metrics stay out of
    # the main tables to keep Chapter IV readable.
    summary_doc = {
        "schema_version": eu.CONFIG_SCHEMA_VERSION, "run_id": eu.RUN_ID,
        "stage": STAGE, "script_name": SCRIPT_NAME,
        "selected_backbone": selected_backbone, "c_best_by_backbone": c_best_map,
        "selected_c_best": selected_c_best,
        "n_bootstrap": N_BOOTSTRAP, "ci_level": CI_LEVEL,
        "primary_metric": "macro_f1",
        "interpretation_notes": [
            "Stage 11 backbone/context choices were reported, not reselected.",
            "Contextual point estimates use the mean of the two seed-level pooled OOF probability arrays, aligned by post_id.",
            "Stage 13 is descriptive and does not affect backbone or context selection.",
            "Micro P/R/F1 are omitted from thesis-facing main tables because they equal accuracy in single-label multiclass classification.",
            "Hamming loss and exact-match accuracy are omitted because they are redundant with 1-accuracy and ordinary accuracy here.",
        ],
        "strategy_decisions": strategy_decisions,
        "stage11_manifest_fingerprint": stage11["fingerprint"],
        "input_stage_manifest_fingerprints": {row["stage"]: row["manifest_fingerprint"] for row in provenance_rows if row["manifest_fingerprint"]},
    }
    return tables, {"summary": summary_doc, "strategy_decisions": strategy_decisions,
                    "selected_backbone": selected_backbone, "selected_c_best": selected_c_best,
                    "c_best_by_backbone": c_best_map,
                    "stage11_manifest_fingerprint": stage11["fingerprint"], "provenance_rows": provenance_rows}


def _create_figures(tables: Mapping[str, pd.DataFrame], *, e1: Mapping[str, Any], e2: Mapping[str, Any], e2b: Mapping[str, Any], e3: Mapping[str, Any], report: Mapping[str, Any], figure_output_dir: Path) -> dict[str, Any]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Do not set fixed colours: let matplotlib's default cycle remain accessible.
    created: dict[str, Any] = {}
    figure_output_dir.mkdir(parents=True, exist_ok=True)
    # Figure 1: Experiment I mean +/- SD across the five seeds.
    summary = tables["e1_model_summary.csv"]
    fig, ax = plt.subplots(figsize=(6.8, 4.4), constrained_layout=True)
    x = np.arange(len(summary))
    means = summary["macro_f1_mean"].to_numpy(dtype=float)
    sds = summary["macro_f1_sd"].to_numpy(dtype=float)
    ax.bar(x, means, yerr=sds, capsize=5)
    ax.set_xticks(x, [eu.get_backbone(str(b)).display_name for b in summary["backbone"]])
    ax.set_ylabel("Macro-F1 (mean ± SD across seeds)")
    ax.set_title("Experiment I: benchmark backbone comparison")
    ax.set_ylim(0, min(1.0, max(0.1, float(np.max(means + sds)) + 0.08)))
    ax.grid(axis="y", alpha=0.25)
    path = figure_output_dir / "fig_e1_backbone_macro_f1.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    created[path.name] = path

    # Figure 2: pooled OOF macro-F1 by context. Uncertainty for the predeclared
    # paired contrasts is reported in e2_context_pairwise.csv, not as error bars.
    e2_table = tables["e2_context_summary.csv"]
    fig, ax = plt.subplots(figsize=(7.6, 4.8), constrained_layout=True)
    for backbone in BACKBONES:
        sub = e2_table[e2_table["backbone"] == backbone].sort_values("context_size")
        ax.plot(sub["context_size"], sub["macro_f1"], marker="o",
                label=eu.get_backbone(backbone).display_name)
        c_best = int(report["c_best_by_backbone"][backbone])
        selected_row = sub[sub["context_size"].astype(int) == c_best]
        if len(selected_row):
            ax.scatter([c_best], selected_row["macro_f1"], marker="*", s=130, zorder=5)
    ax.set_xticks(list(CONDITIONS), [f"C{c}" for c in CONDITIONS])
    ax.set_xlabel("Number of parent messages included")
    ax.set_ylabel("Pooled OOF macro-F1")
    ax.set_title("Experiment II-A: context-depth ablation")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    path = figure_output_dir / "fig_e2_context_depth_macro_f1.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    created[path.name] = path

    # Figure 3: per-class F1 by depth, with small multiples to avoid a crowded legend.
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.8), sharey=True, constrained_layout=True)
    per_cls = tables["e2_context_per_class.csv"]
    for ax, label in zip(axes, ("N", "AL", "HS")):
        for backbone in BACKBONES:
            sub = per_cls[(per_cls["class_code"] == label) & (per_cls["analysis"] == "E2A_OOF") & (per_cls["backbone"] == backbone)].sort_values("context_size")
            ax.plot(sub["context_size"], sub["f1"], marker="o", label=eu.get_backbone(backbone).display_name)
        ax.set_title(label)
        ax.set_xticks(list(CONDITIONS), [f"C{c}" for c in CONDITIONS])
        ax.set_xlabel("Context size")
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Per-class F1")
    axes[-1].legend(frameon=False, fontsize=8)
    fig.suptitle("Experiment II-A: per-class F1 by context size")
    path = figure_output_dir / "fig_e2_per_class_f1.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    created[path.name] = path

    # Figure 4: Stage 13 is descriptive; indicate different eligible sample sizes.
    e2b_table = tables["e2b_generalization_summary.csv"]
    plot = e2b_table[e2b_table["status"] == "evaluated"].sort_values("context_size")
    fig, ax = plt.subplots(figsize=(7.0, 4.5), constrained_layout=True)
    x = np.arange(len(plot))
    ax.bar(x, plot["macro_f1"].to_numpy(dtype=float))
    ax.set_xticks(x, [f"C{int(row.context_size)}\n(n={int(row.n_examples)})" for row in plot.itertuples()])
    ax.set_ylabel("Macro-F1")
    ax.set_title("Experiment II-B: shallow-pool generalization (descriptive)")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.25)
    path = figure_output_dir / "fig_e2b_shallow_generalization.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    created[path.name] = path

    # Figure 5: paired CI of ITFT-minus-direct difference; dot marks estimate.
    strategy = tables["training_strategy_comparison.csv"]
    fig, ax = plt.subplots(figsize=(7.0, 4.4), constrained_layout=True)
    y_positions = np.arange(len(strategy))
    estimates = strategy["difference_itft_minus_direct"].to_numpy(dtype=float)
    lower = strategy["ci_lower"].to_numpy(dtype=float)
    upper = strategy["ci_upper"].to_numpy(dtype=float)
    ax.hlines(y_positions, lower, upper, linewidth=1.5)
    ax.plot(estimates, y_positions, "o")
    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_yticks(y_positions, [f"{row.deployment_mode.replace('_', ' ')} (C{int(row.context_size)})" for row in strategy.itertuples()])
    ax.set_xlabel("Macro-F1 difference (ITFT − direct); 95% paired bootstrap CI")
    ax.set_title("Experiment III: training strategy comparison")
    ax.grid(axis="x", alpha=0.25)
    path = figure_output_dir / "fig_training_strategy_comparison.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    created[path.name] = path

    # Separate publication-quality confusion matrices for key comparisons.
    selected_backbone = str(report["selected_backbone"])
    selected_c = int(report["selected_c_best"])
    cms = tables["confusion_matrices.csv"]
    figure_requests = [
        ("E1_seed_ensemble", "indobertweet", None, "fig_cm_e1_indobertweet.png", "Experiment I — IndoBERTweet"),
        ("E1_seed_ensemble", "xlmr", None, "fig_cm_e1_xlmr.png", "Experiment I — XLM-RoBERTa-base"),
        ("E2A_seed_ensembled_OOF", selected_backbone, 0, "fig_cm_e2_selected_c0.png", f"Experiment II-A — {selected_backbone}, C0"),
        ("E2A_seed_ensembled_OOF", selected_backbone, selected_c, "fig_cm_e2_selected_c_best.png", f"Experiment II-A — {selected_backbone}, C{selected_c}"),
        ("E2B_shallow", selected_backbone, 0, "fig_cm_e2b_shallow_c0.png", "Experiment II-B — shallow C0"),
        ("E2B_shallow", selected_backbone, min(selected_c, 3), "fig_cm_e2b_shallow_cbest_eligible.png", f"Experiment II-B — shallow C{min(selected_c, 3)} (eligible)"),
        ("E3_direct_seed_ensembled_OOF", selected_backbone, 0, "fig_cm_e3_direct_c0.png", "Experiment III — direct C0"),
        ("E3_direct_seed_ensembled_OOF", selected_backbone, selected_c, "fig_cm_e3_direct_c_best.png", f"Experiment III — direct C{selected_c}"),
    ]
    seen_names: set[tuple[str, str, int | None]] = set()
    for analysis, backbone, context, filename, title in figure_requests:
        key = (analysis, backbone, context)
        if key in seen_names:
            continue
        seen_names.add(key)
        subset = cms[(cms["analysis"] == analysis) & (cms["backbone"] == backbone)]
        if context is None:
            subset = subset[subset["context_size"].isna()]
        else:
            subset = subset[subset["context_size"].astype("Int64") == int(context)]
        if subset.empty:
            continue
        matrix = subset.pivot(index="true_label", columns="predicted_label", values="count").reindex(index=["N", "AL", "HS"], columns=["N", "AL", "HS"], fill_value=0)
        fig, ax = plt.subplots(figsize=(4.7, 4.3), constrained_layout=True)
        image = ax.imshow(matrix.to_numpy(), interpolation="nearest")
        ax.set_xticks(range(3), ["N", "AL", "HS"])
        ax.set_yticks(range(3), ["N", "AL", "HS"])
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_title(title)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Count")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{int(matrix.iloc[i, j])}", ha="center", va="center")
        path = figure_output_dir / filename
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        created[path.name] = path
    return created


def _copy_exports(tables_dir: Path, figures_dir: Path, stage_root: Path, stage_manifest_fingerprint: str) -> tuple[Path, Path]:
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    expected_table_paths = [tables_dir / name for name in (*TABLE_FILENAMES, "stage15_summary.json")]
    expected_figure_paths = [figures_dir / name for name in FIGURE_FILENAMES]
    expected_figure_paths += [figures_dir / path.name for path in sorted((stage_root / "figures").glob("fig_cm_*.png"))]
    collisions = [str(path) for path in expected_table_paths + expected_figure_paths if path.exists()]
    if collisions:
        raise FileExistsError("Stage 15 report export paths already exist. Archive historical/export outputs before publishing: " + ", ".join(collisions[:10]))
    for filename in TABLE_FILENAMES:
        shutil.copy2(stage_root / filename, tables_dir / filename)
    shutil.copy2(stage_root / "stage15_summary.json", tables_dir / "stage15_summary.json")
    figure_paths = sorted((stage_root / "figures").glob("*.png"))
    for path in figure_paths:
        shutil.copy2(path, figures_dir / path.name)
    table_manifest = {
        "schema_version": eu.CONFIG_SCHEMA_VERSION, "run_id": eu.RUN_ID, "stage": STAGE,
        "script_name": SCRIPT_NAME, "stage15_manifest_fingerprint": stage_manifest_fingerprint,
        "files_sha256": {name: _sha(tables_dir / name) for name in [*TABLE_FILENAMES, "stage15_summary.json"]},
    }
    figure_manifest = {
        "schema_version": eu.CONFIG_SCHEMA_VERSION, "run_id": eu.RUN_ID, "stage": STAGE,
        "script_name": SCRIPT_NAME, "stage15_manifest_fingerprint": stage_manifest_fingerprint,
        "files_sha256": {path.name: _sha(figures_dir / path.name) for path in figure_paths},
    }
    _write_json_atomic(tables_dir / "stage15_export_manifest.json", table_manifest)
    _write_json_atomic(figures_dir / "stage15_export_manifest.json", figure_manifest)
    return tables_dir, figures_dir


def _load_upstream(root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Perform strict upstream artifact audit and assemble all evaluation data."""
    required = (BENCHMARK_TRAIN, BENCHMARK_VAL, BENCHMARK_TEST, MATCHED_PATH, SHALLOW_PATH, SEQUENCE_CONFIG_PATH, LR_CONFIG_PATH)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Stage 15 requires all prepared data/config artifacts: " + ", ".join(missing))
    for path in (BENCHMARK_TRAIN, BENCHMARK_VAL, BENCHMARK_TEST):
        if not {"clean_text", "label"}.issubset(pd.read_csv(path, nrows=1).columns):
            raise KeyError(f"Benchmark split lacks clean_text/label: {path}")
    data = _load_context_data()
    # Verify benchmark preprocessing and max-length selection before treating
    # the prepared splits/configuration as authoritative inputs.
    stage1_manifest = eu.assert_current_stage_manifest(1, PROJECT_ROOT)
    stage1b_manifest = eu.assert_current_stage_manifest("1b", PROJECT_ROOT)
    # Verify preprocessing manifests before using dataset hashes as inputs.
    stage2_dir = RUN_ROOT / "stage_02"
    stage2_manifest_path = stage2_dir / eu.RUN_MANIFEST_NAME
    if not stage2_manifest_path.is_file():
        raise FileNotFoundError(f"Stage 2 manifest missing: {stage2_manifest_path}")
    stage2_manifest = _read_json(stage2_manifest_path)
    eu._validate_manifest_integrity(stage2_manifest)
    if str(stage2_manifest.get("stage")) != "2" or stage2_manifest.get("script_name") != eu.PIPELINE_SCRIPTS[2]:
        raise ValueError("Stage 2 preprocessing manifest has an incorrect identity")
    eu.assert_current_run_artifact(stage2_dir / eu.RUN_MANIFEST_NAME, expected_stage=2, expected_script=eu.PIPELINE_SCRIPTS[2], expected_fingerprint=str(stage2_manifest["fingerprint"]))
    # This validator additionally confirms the live utility/script/protocol/
    # pipeline fingerprints and all declared Stage 2 input/output hashes.
    eu.assert_current_stage_manifest(2, PROJECT_ROOT)
    _verify_manifest_inputs(stage2_manifest, root=PROJECT_ROOT)
    stage2_output_hashes = stage2_manifest.get("metadata", {}).get("output_sha256", {})
    if not isinstance(stage2_output_hashes, Mapping) or not stage2_output_hashes:
        raise ValueError("Stage 2 manifest is missing the processed-output SHA-256 inventory")
    for relative, expected_digest in stage2_output_hashes.items():
        output_path = PROJECT_ROOT / str(relative)
        if not output_path.is_file() or _sha(output_path) != expected_digest:
            raise ValueError(f"Stage 2 processed output is missing or stale: {output_path}")

    stage5 = _verify_stage5_bundle()
    stage11 = _verify_stage11_bundle()
    stage8_dir_bundle = _verify_stage_root(
        stage_dir=STAGE8_DIR, stage=8, script_name=eu.PIPELINE_SCRIPTS[8],
        required_files=("context_latency_measurements.json", "stage8_publish_receipt.json", "context_lr_trials.csv"),
        marker_name="stage8_complete.json", check_index_script_name=False,
    )
    # Validate both frozen shared config files against the Stage 8 publication receipt.
    sequence_doc = eu.load_shared_run_config(
        sequence_length_path=SEQUENCE_CONFIG_PATH, learning_rate_path=LR_CONFIG_PATH,
        require_context=True, require_context_lr=True,
    )
    stage8_receipt = _read_json(STAGE8_DIR / "stage8_publish_receipt.json")
    if stage8_receipt.get("sequence_length_config_sha256") != _sha(SEQUENCE_CONFIG_PATH) or stage8_receipt.get("learning_rate_config_sha256") != _sha(LR_CONFIG_PATH):
        raise ValueError("Current shared configs do not match Stage 8's publication receipt")
    decision = stage11["decision"]
    backbone = str(decision["decision"])
    c_best_map = {key: int(value) for key, value in decision["c_best_by_backbone"].items()}
    e2 = _load_e2_oof(data, stage11_decision=decision)
    e2b = _verify_stage12_13(stage11, data)
    e3 = _load_e3_oof(matched_data=data, backbone=backbone, c_best=c_best_map[backbone])
    e1 = _load_e1_results(pd.read_csv(BENCHMARK_TEST), stage5)

    # Enforce that Stage 14's stored C_best exactly matches Stage 11.
    stage14_manifest = _read_json(e3["bundle"]["dir"] / "e3_manifest.json")
    if int(stage14_manifest.get("selected_context_size_c_best", -1)) != c_best_map[backbone]:
        raise ValueError("Stage 14 was trained using a different C_best than the saved Stage 11 decision")

    stage1_manifest_path = RUN_ROOT / "stage_01" / eu.RUN_MANIFEST_NAME
    stage1b_manifest_path = RUN_ROOT / "stage_1b" / eu.RUN_MANIFEST_NAME
    input_paths: list[Path] = [*required, stage1_manifest_path, stage1b_manifest_path, stage2_manifest_path]
    input_paths.extend([STAGE5_DIR / eu.RUN_MANIFEST_NAME, STAGE5_DIR / "artifacts_sha256.json", *[STAGE5_DIR / name for name in STAGE5_OUTPUTS]])
    input_paths.extend([STAGE8_DIR / name for name in (eu.RUN_MANIFEST_NAME, "stage8_complete.json", "artifacts_sha256.json", "stage8_publish_receipt.json", "context_latency_measurements.json", "context_lr_trials.csv")])
    input_paths.extend([STAGE11_DIR / name for name in (eu.RUN_MANIFEST_NAME, "stage11_complete.json", "artifacts_sha256.json", "backbone_decision.json", "context_selection_indobertweet.json", "context_selection_xlmr.json")])
    input_paths.extend(e1["input_paths"])
    input_paths.extend(e2["input_paths"])
    input_paths.extend(e2b["input_paths"])
    input_paths.extend(e3["input_paths"])
    # Stable de-duplication while preserving source order.
    input_paths = list(dict.fromkeys(Path(path).resolve() for path in input_paths))
    for path in input_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Stage 15 provenance input is unavailable: {path}")
    return {
        "data": data, "stage1_manifest": stage1_manifest, "stage1b_manifest": stage1b_manifest,
        "stage2_manifest": stage2_manifest, "stage5": stage5, "stage8": stage8_dir_bundle,
        "stage11": stage11, "shared_config": sequence_doc, "e1": e1, "e2": e2, "e2b": e2b, "e3": e3,
        "input_paths": input_paths,
    }


def _make_stage15_manifest(upstream: Mapping[str, Any], tables: Mapping[str, pd.DataFrame], report: Mapping[str, Any]) -> dict[str, Any]:
    root = PROJECT_ROOT
    config = {
        "schema_version": eu.CONFIG_SCHEMA_VERSION, "run_id": eu.RUN_ID,
        "stage": STAGE, "script_name": SCRIPT_NAME,
        "selected_backbone": report["selected_backbone"], "selected_c_best": report["selected_c_best"],
        "c_best_by_backbone": report["c_best_by_backbone"],
        "primary_metric": "macro_f1", "bootstrap": {"n_bootstrap": N_BOOTSTRAP, "ci_level": CI_LEVEL, "random_state": BOOTSTRAP_SEED},
        "table_files": sorted(tables), "figure_files": sorted(report["summary"].get("figure_files", FIGURE_FILENAMES)),
        "strategy_decisions": report["strategy_decisions"],
    }
    metadata = {
        "purpose": "Chapter IV cross-experiment quantitative evaluation and Stage 15 training-strategy decision",
        "stage11_manifest_fingerprint": report["stage11_manifest_fingerprint"],
        "input_stage_manifest_fingerprints": report["summary"].get("input_stage_manifest_fingerprints", {}),
        "n_context_oof_examples": EXPECTED_MATCHED_ROWS,
        "e1_test_examples": int(len(upstream["e1"]["y_true"])),
    }
    return eu.make_run_manifest(stage=STAGE, script_name=SCRIPT_NAME, config=config,
                                input_paths=upstream["input_paths"], metadata=metadata, project_root=root)


def _publish_stage15(upstream: Mapping[str, Any], tables: Mapping[str, pd.DataFrame], report: Mapping[str, Any], *, stage_root_manifest: Mapping[str, Any], figure_source_dir: Path) -> None:
    # Never silently overwrite an earlier Stage 15 result bundle or canonical
    # decision: archive prior runs/exports before a deliberate fresh publication.
    output_dirs = (TABLES_EXPORT_DIR, FIGURES_EXPORT_DIR)
    for directory in output_dirs:
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError(f"Stage 15 report export directory is not empty: {directory}. Archive prior outputs before publishing.")
    if STRATEGY_DECISION_PATH.exists():
        raise FileExistsError(f"Canonical strategy decision already exists: {STRATEGY_DECISION_PATH}. Archive it before a clean Stage 15 publication.")

    eu.prepare_run_directory(STAGE_ROOT, stage_root_manifest, allow_resume=False)
    # Materialize tables and the already-rendered figures into the authoritative bundle.
    for filename, frame in tables.items():
        _write_csv_atomic(STAGE_ROOT / filename, frame)
    for filename, frame in tables.items():
        if filename not in TABLE_FILENAMES:
            raise AssertionError(f"Unexpected Stage 15 table filename: {filename}")
    figures_dir = STAGE_ROOT / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(figure_source_dir.glob("*.png")):
        shutil.copy2(path, figures_dir / path.name)
    _write_json_atomic(STAGE_ROOT / "stage15_summary.json", report["summary"])
    eu.save_training_strategy_decision(report["strategy_decisions"], path=STAGE_ROOT / "training_strategy_decision.json")
    decision_copy = _read_json(STAGE_ROOT / "training_strategy_decision.json")
    decision_copy["provenance"] = {
        "stage15_manifest_fingerprint": stage_root_manifest["fingerprint"],
        "stage15_output_dir": STAGE_ROOT.relative_to(PROJECT_ROOT).as_posix(),
        "stage11_manifest_fingerprint": report["stage11_manifest_fingerprint"],
        "strategy_input_post_id_sha256": eu.sha256_json(upstream["data"]["matched_ids"]),
    }
    _write_json_atomic(STAGE_ROOT / "training_strategy_decision.json", decision_copy)
    # Copy charts into a nested authoritative directory before creating the index.
    # All charts have already been rendered there by _create_figures().
    stage_root_manifest_fingerprint = str(stage_root_manifest["fingerprint"])
    # Save stage manifest is already present; artifact index excludes itself/marker.
    sha_map = {}
    for path in sorted(STAGE_ROOT.rglob("*")):
        if not path.is_file() or path.name in {"artifacts_sha256.json", "stage15_complete.json"}:
            continue
        sha_map[path.relative_to(STAGE_ROOT).as_posix()] = _sha(path)
    index_path = STAGE_ROOT / "artifacts_sha256.json"
    _write_json_atomic(index_path, {
        "schema_version": eu.CONFIG_SCHEMA_VERSION, "run_id": eu.RUN_ID, "stage": STAGE,
        "script_name": SCRIPT_NAME, "manifest_fingerprint": stage_root_manifest_fingerprint,
        "sha256": sha_map,
    })
    # The same canonical decision is published to the well-known location used
    # by Stage 17; the versioned copy + marker attest to its exact bytes.
    decision_root_doc = _read_json(STAGE_ROOT / "training_strategy_decision.json")
    _write_json_atomic(STRATEGY_DECISION_PATH, decision_root_doc)

    # Copy thesis-facing files to the dedicated report folders. The versioned
    # run bundle remains the provenance source of truth.
    TABLES_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    for filename in TABLE_FILENAMES:
        shutil.copy2(STAGE_ROOT / filename, TABLES_EXPORT_DIR / filename)
    shutil.copy2(STAGE_ROOT / "stage15_summary.json", TABLES_EXPORT_DIR / "stage15_summary.json")
    for path in sorted((STAGE_ROOT / "figures").glob("*.png")):
        shutil.copy2(path, FIGURES_EXPORT_DIR / path.name)
    _write_json_atomic(TABLES_EXPORT_DIR / "stage15_export_manifest.json", {
        "schema_version": eu.CONFIG_SCHEMA_VERSION, "run_id": eu.RUN_ID, "stage": STAGE,
        "script_name": SCRIPT_NAME, "stage15_manifest_fingerprint": stage_root_manifest_fingerprint,
        "files_sha256": {path.name: _sha(path) for path in sorted(TABLES_EXPORT_DIR.iterdir()) if path.is_file() and path.name != "stage15_export_manifest.json"},
    })
    _write_json_atomic(FIGURES_EXPORT_DIR / "stage15_export_manifest.json", {
        "schema_version": eu.CONFIG_SCHEMA_VERSION, "run_id": eu.RUN_ID, "stage": STAGE,
        "script_name": SCRIPT_NAME, "stage15_manifest_fingerprint": stage_root_manifest_fingerprint,
        "files_sha256": {path.name: _sha(path) for path in sorted(FIGURES_EXPORT_DIR.iterdir()) if path.is_file() and path.name != "stage15_export_manifest.json"},
    })
    index_path_sha = _sha(index_path)
    _write_json_atomic(STAGE_ROOT / "stage15_complete.json", {
        "schema_version": eu.CONFIG_SCHEMA_VERSION, "run_id": eu.RUN_ID, "stage": STAGE,
        "script_name": SCRIPT_NAME, "manifest_fingerprint": stage_root_manifest_fingerprint,
        "artifact_index_sha256": index_path_sha,
        "strategy_decision_sha256": _sha(STAGE_ROOT / "training_strategy_decision.json"),
        "canonical_strategy_decision_sha256": _sha(STRATEGY_DECISION_PATH),
        "selected_backbone": report["selected_backbone"], "selected_c_best": report["selected_c_best"],
        "output_table_count": len(TABLE_FILENAMES),
        "output_figure_count": len(list((STAGE_ROOT / "figures").glob("*.png"))),
    })
    eu.assert_current_run_artifact(STAGE_ROOT / "stage15_complete.json", expected_stage=STAGE,
                                   expected_script=SCRIPT_NAME, expected_fingerprint=stage_root_manifest_fingerprint)
    _verify_published_stage15()


def _verify_published_stage15() -> None:
    manifest = _read_json(STAGE_ROOT / eu.RUN_MANIFEST_NAME)
    fingerprint = str(manifest["fingerprint"])
    marker_path = STAGE_ROOT / "stage15_complete.json"
    index_path = STAGE_ROOT / "artifacts_sha256.json"
    marker = _read_json(marker_path)
    if marker.get("manifest_fingerprint") != fingerprint or marker.get("artifact_index_sha256") != _sha(index_path):
        raise ValueError("Stage 15 completion marker does not match the manifest/index")
    if marker.get("strategy_decision_sha256") != _sha(STAGE_ROOT / "training_strategy_decision.json"):
        raise ValueError("Stage 15 bundled strategy decision hash mismatch")
    if marker.get("canonical_strategy_decision_sha256") != _sha(STRATEGY_DECISION_PATH):
        raise ValueError("Canonical training strategy decision hash mismatch")
    index = _verify_artifact_index(
        index_path=index_path, artifact_root=STAGE_ROOT,
        expected_identity={"schema_version": eu.CONFIG_SCHEMA_VERSION, "run_id": eu.RUN_ID,
                         "stage": STAGE, "script_name": SCRIPT_NAME, "manifest_fingerprint": fingerprint},
    )
    if not (STAGE_ROOT / "training_strategy_decision.json").is_file():
        raise FileNotFoundError("Stage 15 strategy decision copy is missing")
    strategy = eu.load_training_strategy_decision(STRATEGY_DECISION_PATH)
    bundled = _read_json(STAGE_ROOT / "training_strategy_decision.json")
    if strategy.get("by_mode") != bundled.get("by_mode") or bundled.get("provenance", {}).get("stage15_manifest_fingerprint") != fingerprint:
        raise ValueError("Canonical/bundled Stage 15 strategy decisions differ or lack current provenance")
    if not index.get("sha256"):
        raise ValueError("Stage 15 artifact index is empty")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 15: cross-experiment quantitative evaluation and strategy selection.")
    parser.add_argument("--validate-only", action="store_true", help="Audit all upstream artifacts, calculate reports/strategy in memory, but do not write outputs.")
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP, help="Must remain 10,000 under the frozen protocol.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.n_bootstrap != N_BOOTSTRAP:
        raise ValueError(f"Frozen protocol requires exactly {N_BOOTSTRAP:,} paired bootstrap resamples")
    os.chdir(PROJECT_ROOT)
    if not (PROJECT_ROOT / "experiment_protocol.md").is_file() or not (PROJECT_ROOT / "pipeline.md").is_file():
        raise FileNotFoundError("experiment_protocol.md and pipeline.md must exist at the repository root")
    migration = eu.audit_pipeline_script_names(PROJECT_ROOT, require_all=False)
    if SCRIPT_NAME not in migration["present"]:
        raise FileNotFoundError(f"Authoritative Stage 15 script is missing from the registered scripts/ location")
    if not args.validate_only:
        eu.audit_pipeline_script_names(PROJECT_ROOT, require_all=True)
    print(f"Run ID: {eu.RUN_ID}")
    print("Auditing Stage 2, Stage 5, Stage 8, Stage 11, both Experiment II-A grids, and Stages 12-14...")
    upstream = _load_upstream(PROJECT_ROOT)
    tables, report = _make_report_tables(
        upstream=upstream, data=upstream["data"], e1=upstream["e1"], e2=upstream["e2"], stage11=upstream["stage11"],
        e2b=upstream["e2b"], e3=upstream["e3"],
    )
    print("\nExperiment I seed summary:")
    print(tables["e1_model_summary.csv"][["backbone", "accuracy_mean", "accuracy_sd", "macro_f1_mean", "macro_f1_sd", "weighted_f1_mean", "weighted_f1_sd"]].to_string(index=False))
    print("\nExperiment II-A context summary (pooled OOF):")
    print(tables["e2_context_summary.csv"][["backbone", "context_size", "n_examples", "accuracy", "macro_f1", "weighted_f1", "is_c_best"]].to_string(index=False))
    print("\nTraining strategy decisions:")
    print(tables["training_strategy_comparison.csv"].to_string(index=False))
    if args.validate_only:
        print("\nStage 15 validation-only passed: all upstream artifacts and hashes are valid, the required Chapter IV tables have been assembled in memory, and the two strategy decisions were computed. No files were written.")
        if migration["complete"]:
            print("Pipeline filename/directory audit: passed.")
        else:
            print(f"Pipeline filename/directory audit is not yet complete: missing={migration['missing']}; missing_directories={migration['missing_directories']}")
        return 0

    # Make the report-only table/figure destinations fail closed on a stale run.
    if TABLES_EXPORT_DIR.exists() and any(TABLES_EXPORT_DIR.iterdir()):
        raise FileExistsError(f"{TABLES_EXPORT_DIR} is not empty; archive prior exports before publication")
    if FIGURES_EXPORT_DIR.exists() and any(FIGURES_EXPORT_DIR.iterdir()):
        raise FileExistsError(f"{FIGURES_EXPORT_DIR} is not empty; archive prior exports before publication")
    if STAGE_ROOT.exists() and any(STAGE_ROOT.iterdir()):
        raise FileExistsError(f"{STAGE_ROOT} is not empty; archive the old Stage 15 bundle before publication")
    # Render charts in a temporary staging area so we can include the exact
    # figure inventory in the immutable manifest before creating Stage 15.
    with tempfile.TemporaryDirectory(prefix="stage15-figures-") as temporary_name:
        temporary_figure_dir = Path(temporary_name)
        figures = _create_figures(
            tables, e1=upstream["e1"], e2=upstream["e2"], e2b=upstream["e2b"], e3=upstream["e3"],
            report=report, figure_output_dir=temporary_figure_dir,
        )
        report["summary"]["figure_files"] = sorted(figures)
        input_sha = {path.relative_to(PROJECT_ROOT).as_posix(): _sha(path) for path in upstream["input_paths"]}
        report["summary"]["input_sha256"] = input_sha
        stage_manifest = _make_stage15_manifest(upstream, tables, report)
        if stage_manifest["config"].get("script_name") != SCRIPT_NAME:
            raise ValueError("Stage 15 manifest construction returned an inconsistent script identity")
        _publish_stage15(upstream, tables, report, stage_root_manifest=stage_manifest, figure_source_dir=temporary_figure_dir)
    print("\nStage 15 completed.")
    print(f"Authoritative bundle: {STAGE_ROOT.relative_to(PROJECT_ROOT).as_posix()}/")
    print(f"Thesis tables: {TABLES_EXPORT_DIR.relative_to(PROJECT_ROOT).as_posix()}/")
    print(f"Thesis figures: {FIGURES_EXPORT_DIR.relative_to(PROJECT_ROOT).as_posix()}/")
    print(f"Canonical strategy decision: {STRATEGY_DECISION_PATH.relative_to(PROJECT_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
