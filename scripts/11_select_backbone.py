"""Stage 11: select C_best per backbone, then select the deployment backbone.

Authoritative pipeline script: ``scripts/11_select_backbone.py``.

Requires completed, manifest-verified Stage 9 (IndoBERTweet) and Stage 10
(XLM-RoBERTa-base) grids: 5 context sizes x 5 shared folds x 2 seeds = 50 runs
per backbone. It does not train a model. It audits each run's manifest, marker,
metrics, predictions, probability array and output hashes before assembling
seed-ensembled pooled out-of-fold (OOF) predictions on the same ordered 900
matched examples.

The script then:
  1. Selects C_best independently for each backbone using the frozen paired
     OOF macro-F1 bootstrap/parsimony rule in experiment_utils.
  2. Selects the backbone using the C_best paired-bootstrap comparison, C0
     guard, and Stage 8 median inference latency measured at each backbone's
     selected context size.
  3. Writes the validated decision to results/backbone_decision.json and
     records its provenance plus the context decisions and grid fingerprints
     under results/runs/left_truncation_v2/stage_11/decision/.

Experiment I is informational only: if the current Stage 5 evaluation exists
and verifies successfully, its leader is recorded for discussion, never used
as an input to the Stage 11 selection rule.

Run from the repository root after Stages 1-10 have completed:
    python scripts/11_select_backbone.py --validate-only
    python scripts/11_select_backbone.py

The actual decision-writing run is blocked until the full pipeline filename /
directory audit passes. --validate-only performs the upstream artifact audit
and computes the same decisions in memory without writing decision files.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
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

STAGE = 11
SCRIPT_NAME = "scripts/11_select_backbone.py"
BACKBONE_STAGE = {"indobertweet": 9, "xlmr": 10}
CONTEXTS = tuple(range(5))
FOLDS = tuple(range(5))
SEEDS = tuple(eu.CV_SEEDS)
EXPECTED_RUNS_PER_BACKBONE = len(CONTEXTS) * len(FOLDS) * len(SEEDS)
EXPECTED_ROWS = eu.CONTEXT_MATCHED_EXPECTED_ROWS
BOOTSTRAP_REPLICATES = 10_000
CONFIDENCE_LEVEL = 95.0
BOOTSTRAP_SEED = 42

MATCHED_PATH = Path("data/processed/ctx_matched_subset.csv")
SEQUENCE_CONFIG_PATH = eu.SEQUENCE_LENGTH_CONFIG
LEARNING_RATE_CONFIG_PATH = eu.LEARNING_RATE_CONFIG
STAGE2_MANIFEST_PATH = eu.versioned_run_root(PROJECT_ROOT) / "stage_02" / eu.RUN_MANIFEST_NAME
STAGE8_DIR = eu.versioned_run_root(PROJECT_ROOT) / "stage_08" / "summary"
STAGE8_MANIFEST_PATH = STAGE8_DIR / eu.RUN_MANIFEST_NAME
STAGE8_MARKER_PATH = STAGE8_DIR / "stage8_complete.json"
STAGE8_INDEX_PATH = STAGE8_DIR / "artifacts_sha256.json"
STAGE8_RECEIPT_PATH = STAGE8_DIR / "stage8_publish_receipt.json"
STAGE8_LATENCY_PATH = STAGE8_DIR / "context_latency_measurements.json"
STAGE5_DIR = eu.versioned_run_root(PROJECT_ROOT) / "experiment_i" / "evaluation"
STAGE5_MANIFEST_PATH = STAGE5_DIR / eu.RUN_MANIFEST_NAME
STAGE5_INDEX_PATH = STAGE5_DIR / "artifacts_sha256.json"
STAGE5_SUMMARY_PATH = STAGE5_DIR / "e1_summary.json"
STAGE5_BOOTSTRAP_PATH = STAGE5_DIR / "e1_paired_bootstrap.csv"
DECISION_PATH = PROJECT_ROOT / eu.BACKBONE_DECISION_PATH
STAGE11_OUTPUT_DIR = eu.versioned_run_root(PROJECT_ROOT) / "stage_11" / "decision"
STAGE11_DECISION_COPY = STAGE11_OUTPUT_DIR / "backbone_decision.json"
STAGE11_INDEX_PATH = STAGE11_OUTPUT_DIR / "artifacts_sha256.json"
STAGE11_MARKER_PATH = STAGE11_OUTPUT_DIR / "stage11_complete.json"

LEGACY_OUTPUTS = (
    "results/raw/e2_C*_fold*_seed*_predictions.csv",
    "results/raw/e2_C*_fold*_seed*_probs.npy",
    "results/tables/e2_grid_metrics.csv",
    "models/E2_itft_context",
)
RUN_ARTIFACT_NAMES = (
    eu.RUN_MANIFEST_NAME,
    "run_complete.json",
    "artifacts_sha256.json",
    "predictions.csv",
    "probabilities.npy",
    "metrics.json",
)


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (PROJECT_ROOT / candidate).resolve()


def _read_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read valid JSON at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object at {path}")
    return value


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _legacy_paths() -> list[Path]:
    found: list[Path] = []
    for pattern in LEGACY_OUTPUTS:
        found.extend(p for p in PROJECT_ROOT.glob(pattern) if p.exists())
    return list(dict.fromkeys(found))


def _expected_run_dir(backbone: str, context: int, fold: int, seed: int) -> Path:
    stage = BACKBONE_STAGE[backbone]
    return (
        eu.versioned_run_root(PROJECT_ROOT)
        / f"stage_{stage:02d}"
        / f"e2_{backbone}"
        / f"C{context}"
        / f"fold_{fold:02d}"
        / f"seed_{seed}"
    )


def _validate_output_index(
    run_dir: Path,
    *,
    stage: int,
    backbone: str,
    context: int,
    fold: int,
    seed: int,
    fingerprint: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate a single grid run's manifest, marker, index, and hashed outputs."""
    required = [run_dir / name for name in RUN_ARTIFACT_NAMES]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete Stage {stage} grid run {run_dir}: missing {missing}")
    expected_names = set(RUN_ARTIFACT_NAMES)
    unexpected = {p.name for p in run_dir.iterdir() if p.is_file()} - expected_names
    if unexpected:
        raise ValueError(f"Unexpected file(s) in CV run directory {run_dir}: {sorted(unexpected)}")

    manifest = _read_json(run_dir / eu.RUN_MANIFEST_NAME)
    if manifest.get("fingerprint") != fingerprint:
        raise ValueError(f"Manifest fingerprint changed while auditing {run_dir}")
    marker_path = run_dir / "run_complete.json"
    eu.assert_current_run_artifact(
        marker_path,
        expected_stage=stage,
        expected_script=eu.PIPELINE_SCRIPTS[stage],
        expected_fingerprint=fingerprint,
    )
    marker = _read_json(marker_path)
    index = _read_json(run_dir / "artifacts_sha256.json")
    identity = {
        "schema_version": eu.CONFIG_SCHEMA_VERSION,
        "run_id": eu.RUN_ID,
        "stage": stage,
        "script_name": eu.PIPELINE_SCRIPTS[stage],
        "backbone": backbone,
        "context_size": context,
        "fold": fold,
        "seed": seed,
        "manifest_fingerprint": fingerprint,
    }
    for name, expected in identity.items():
        if marker.get(name) != expected or index.get(name) != expected:
            raise ValueError(f"Run {run_dir}: completion marker/index identity mismatch for {name}")
    if marker.get("artifact_index_sha256") != eu.sha256_file(run_dir / "artifacts_sha256.json"):
        raise ValueError(f"Run {run_dir}: completion marker has a stale artifact-index hash")
    indexed = index.get("sha256")
    if not isinstance(indexed, Mapping):
        raise ValueError(f"Run {run_dir}: artifacts_sha256.json has no sha256 map")
    for filename in ("predictions.csv", "probabilities.npy", "metrics.json"):
        path = run_dir / filename
        if indexed.get(filename) != eu.sha256_file(path):
            raise ValueError(f"Run {run_dir}: output hash mismatch for {filename}")

    cfg = manifest.get("config")
    if not isinstance(cfg, Mapping):
        raise ValueError(f"Run {run_dir}: manifest config is missing")
    cfg_expected = {
        "schema_version": eu.CONFIG_SCHEMA_VERSION,
        "run_id": eu.RUN_ID,
        "stage": stage,
        "script_name": eu.PIPELINE_SCRIPTS[stage],
        "backbone": backbone,
        "context_size": context,
        "fold": fold,
        "seed": seed,
        "truncation_side": "left",
    }
    for name, expected in cfg_expected.items():
        if str(cfg.get(name)) != str(expected):
            raise ValueError(f"Run {run_dir}: manifest config {name}={cfg.get(name)!r}; expected {expected!r}")
    if marker.get("post_id_sha256") is None:
        raise ValueError(f"Run {run_dir}: missing ordered held-out post-ID hash")
    return manifest, marker


def _pool_seed_probabilities(
    fold_indices: Sequence[np.ndarray],
    fold_probabilities: Sequence[np.ndarray],
    *,
    expected_rows: int = EXPECTED_ROWS,
) -> np.ndarray:
    """Pool five non-overlapping held-out probability arrays into canonical row order."""
    if len(fold_indices) != len(FOLDS) or len(fold_probabilities) != len(FOLDS):
        raise ValueError("OOF pooling requires exactly one held-out array for each of the five folds")
    pooled = eu.pool_out_of_fold_probabilities(
        list(fold_indices), [np.asarray(values) for values in fold_probabilities]
    )
    if pooled.shape != (expected_rows, len(eu.LABEL2ID)):
        raise ValueError(f"Pooled OOF probability shape {pooled.shape}; expected {(expected_rows, len(eu.LABEL2ID))}")
    if not np.isfinite(pooled).all() or (pooled < 0).any() or not np.allclose(pooled.sum(axis=1), 1.0, atol=1e-4, rtol=1e-4):
        raise ValueError("Pooled OOF probabilities contain invalid probability rows")
    return pooled


def _latency_at_selected_context(
    median_ms_by_context: Mapping[str, Mapping[str, float]],
    context_choices: Mapping[str, int],
) -> dict[str, float]:
    """Use each backbone's median latency at its own selected C_best."""
    if set(context_choices) != set(eu.BACKBONES):
        raise ValueError(f"Context choices must contain exactly {sorted(eu.BACKBONES)}")
    result = {}
    for backbone in eu.BACKBONES:
        context = int(context_choices[backbone])
        if context not in CONTEXTS:
            raise ValueError(f"Invalid selected context C{context} for {backbone}")
        entry = median_ms_by_context.get(f"C{context}")
        if not isinstance(entry, Mapping) or backbone not in entry:
            raise ValueError(f"Missing measured latency at C{context} for {backbone}")
        value = float(entry[backbone])
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"Latency at C{context} for {backbone} must be finite and positive")
        result[backbone] = value
    return result


def _load_and_audit_grid(
    backbone: str,
    matched_sorted: pd.DataFrame,
    canonical_y: np.ndarray,
    *,
    expected_model_revision: str,
    expected_tokenizer_revision: str,
    expected_max_length: int,
    expected_dataset_sha256: str,
    expected_fold_sha256: str,
) -> dict[str, Any]:
    """Audit one 50-run grid and return seed-pooled OOF probabilities by C0-C4."""
    stage = BACKBONE_STAGE[backbone]
    all_run_dirs = eu.versioned_run_root(PROJECT_ROOT) / f"stage_{stage:02d}" / f"e2_{backbone}"
    expected_keys = {
        (context, fold, seed)
        for context in CONTEXTS for fold in FOLDS for seed in SEEDS
    }
    discovered: set[tuple[int, int, int]] = set()
    if all_run_dirs.exists():
        for path in all_run_dirs.glob("C*/fold_*/seed_*"):
            if not path.is_dir():
                continue
            try:
                context = int(path.parent.parent.name.removeprefix("C"))
                fold = int(path.parent.name.removeprefix("fold_"))
                seed = int(path.name.removeprefix("seed_"))
            except ValueError:
                raise ValueError(f"Unrecognised CV run directory naming: {path}")
            discovered.add((context, fold, seed))
    if discovered != expected_keys:
        missing = sorted(expected_keys - discovered)
        extra = sorted(discovered - expected_keys)
        raise ValueError(
            f"Stage {stage} {backbone} run-directory inventory is incomplete or unexpected. "
            f"Missing={missing[:10]}, extra={extra[:10]}"
        )

    fold_ids = pd.to_numeric(matched_sorted[eu.CTX_FOLD_COLUMN], errors="raise").to_numpy(dtype=np.int64)
    canonical_ids = matched_sorted[eu.CTX_ID_COLUMN].astype(str).tolist()
    folds_indices = [np.flatnonzero(fold_ids == fold) for fold in FOLDS]
    run_records: list[dict[str, str]] = []
    input_paths: list[Path] = []
    raw_probabilities: dict[int, dict[int, list[np.ndarray | None]]] = {
        context: {seed: [None] * len(FOLDS) for seed in SEEDS} for context in CONTEXTS
    }
    manifest_fingerprints: dict[str, str] = {}

    for context in CONTEXTS:
        for fold in FOLDS:
            indices = folds_indices[fold]
            expected_fold_ids = [canonical_ids[index] for index in indices]
            expected_fold_y = canonical_y[indices]
            if len(indices) != 180:
                raise ValueError(f"Fold {fold} must hold out exactly 180 examples; got {len(indices)}")
            for seed in SEEDS:
                run_dir = _expected_run_dir(backbone, context, fold, seed)
                manifest_path = run_dir / eu.RUN_MANIFEST_NAME
                if not manifest_path.is_file():
                    raise FileNotFoundError(f"Missing Stage {stage} manifest: {manifest_path}")
                manifest = _read_json(manifest_path)
                fingerprint = str(manifest.get("fingerprint", ""))
                record = {"artifact_path": str(run_dir / "run_complete.json"), "manifest_fingerprint": fingerprint}
                run_records.append(record)
                verified_manifest, marker = _validate_output_index(
                    run_dir, stage=stage, backbone=backbone, context=context, fold=fold,
                    seed=seed, fingerprint=fingerprint,
                )
                run_config = verified_manifest.get("config", {})
                eu.validate_resolved_run_config(
                    run_config, stage=stage, backbone=backbone,
                    input_mode="contextual", context_size=context, require_shared_config=True,
                )
                revision_expectations = {
                    "hf_model_id": eu.get_backbone(backbone).hf_model_id,
                    "model_revision": expected_model_revision,
                    "tokenizer_revision": expected_tokenizer_revision,
                    "max_length_context": expected_max_length,
                    "dataset_sha256": expected_dataset_sha256,
                    "fold_assignment_sha256": expected_fold_sha256,
                }
                for field, expected_value in revision_expectations.items():
                    if str(run_config.get(field)) != str(expected_value):
                        raise ValueError(f"Manifest config {field} mismatch for {run_dir}: {run_config.get(field)!r} vs {expected_value!r}")
                pred_path = run_dir / "predictions.csv"
                probs_path = run_dir / "probabilities.npy"
                metrics_path = run_dir / "metrics.json"
                preds = pd.read_csv(pred_path)
                required_cols = {
                    eu.CTX_ID_COLUMN, "backbone", "context_size", "fold", "seed", "y_true", "y_pred"
                }
                missing_cols = sorted(required_cols - set(preds.columns))
                if missing_cols:
                    raise ValueError(f"Stage {stage} predictions {pred_path} are missing columns {missing_cols}")
                if preds[eu.CTX_ID_COLUMN].astype(str).tolist() != expected_fold_ids:
                    raise ValueError(f"Stage {stage} {backbone} C{context} fold {fold} prediction IDs are not in canonical order")
                for key, expected in (("backbone", backbone), ("context_size", context), ("fold", fold), ("seed", seed)):
                    values = preds[key].drop_duplicates().tolist()
                    if values != [expected]:
                        raise ValueError(f"Prediction file {pred_path} has inconsistent {key}: {values}")
                y_true = preds["y_true"].map(eu.LABEL2ID)
                y_pred = preds["y_pred"].map(eu.LABEL2ID)
                if y_true.isna().any() or y_pred.isna().any():
                    raise ValueError(f"Prediction labels in {pred_path} are outside the canonical N/AL/HS mapping")
                if not np.array_equal(y_true.to_numpy(dtype=np.int64), expected_fold_y):
                    raise ValueError(f"Ground-truth labels in {pred_path} differ from the current matched subset")
                probs = np.load(probs_path, allow_pickle=False)
                expected_shape = (len(indices), len(eu.LABEL2ID))
                if probs.shape != expected_shape or not np.issubdtype(probs.dtype, np.number) or not np.isfinite(probs).all():
                    raise ValueError(f"Invalid probability array shape/content in {probs_path}; expected {expected_shape}")
                probs = probs.astype(np.float64, copy=False)
                if (probs < 0).any() or (probs > 1).any() or not np.allclose(probs.sum(axis=1), 1.0, atol=1e-4, rtol=1e-4):
                    raise ValueError(f"Invalid softmax probabilities in {probs_path}")
                if not np.array_equal(probs.argmax(axis=1), y_pred.to_numpy(dtype=np.int64)):
                    raise ValueError(f"Stored y_pred differs from argmax(probabilities) in {pred_path}")
                metrics = _read_json(metrics_path)
                for key, expected in {
                    "schema_version": eu.CONFIG_SCHEMA_VERSION,
                    "run_id": eu.RUN_ID,
                    "stage": stage,
                    "script_name": eu.PIPELINE_SCRIPTS[stage],
                    "backbone": backbone,
                    "context_size": context,
                    "fold": fold,
                    "seed": seed,
                    "manifest_fingerprint": fingerprint,
                    "truncation_side": "left",
                }.items():
                    if str(metrics.get(key)) != str(expected):
                        raise ValueError(f"Metrics file {metrics_path} has inconsistent {key}")
                if int(metrics.get("n_held_out", -1)) != len(indices):
                    raise ValueError(f"Metrics file {metrics_path} has invalid n_held_out")
                if marker.get("post_id_sha256") != eu.sha256_json(expected_fold_ids):
                    raise ValueError(f"Completion marker {run_dir / 'run_complete.json'} has wrong held-out ID hash")

                raw_probabilities[context][seed][fold] = probs
                manifest_fingerprints[f"C{context}/fold_{fold:02d}/seed_{seed}"] = fingerprint
                input_paths.extend([
                    manifest_path, run_dir / "run_complete.json", run_dir / "artifacts_sha256.json",
                    pred_path, probs_path, metrics_path,
                ])

    grid_audit = eu.audit_context_cv_grid(run_records, backbone=backbone)
    probabilities_by_context: dict[int, list[np.ndarray]] = {}
    for context in CONTEXTS:
        seed_arrays = []
        for seed in SEEDS:
            fold_probabilities = raw_probabilities[context][seed]
            if any(value is None for value in fold_probabilities):
                raise RuntimeError(f"Internal pooling error: incomplete probability set for {backbone} C{context} seed {seed}")
            pooled = _pool_seed_probabilities(
                folds_indices, [np.asarray(value) for value in fold_probabilities],
                expected_rows=EXPECTED_ROWS,
            )
            seed_arrays.append(pooled)
        probabilities_by_context[context] = seed_arrays

    return {
        "backbone": backbone,
        "stage": stage,
        "grid_audit": grid_audit,
        "probabilities_by_context": probabilities_by_context,
        "manifest_fingerprints": manifest_fingerprints,
        "input_paths": input_paths,
        "canonical_ids": canonical_ids,
        "canonical_y": canonical_y,
    }


def _validate_stage8_latency_and_configs() -> dict[str, Any]:
    """Validate Stage 8's publication, shared configs, and per-context latency arrays."""
    for path in (
        STAGE8_MANIFEST_PATH, STAGE8_MARKER_PATH, STAGE8_INDEX_PATH,
        STAGE8_RECEIPT_PATH, STAGE8_LATENCY_PATH, _resolve(SEQUENCE_CONFIG_PATH),
        _resolve(LEARNING_RATE_CONFIG_PATH),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Required Stage 8 publication/configuration is missing: {path}")
    manifest = _read_json(STAGE8_MANIFEST_PATH)
    fingerprint = str(manifest.get("fingerprint", ""))
    marker = _read_json(STAGE8_MARKER_PATH)
    eu.assert_current_run_artifact(
        STAGE8_MARKER_PATH,
        expected_stage=8,
        expected_script="scripts/8_timing_pilot.py",
        expected_fingerprint=fingerprint,
    )
    if marker.get("manifest_fingerprint") != fingerprint:
        raise ValueError("Stage 8 completion marker and manifest fingerprints differ")
    if marker.get("artifact_index_sha256") != eu.sha256_file(STAGE8_INDEX_PATH):
        raise ValueError("Stage 8 completion marker has a stale artifact-index hash")
    index = _read_json(STAGE8_INDEX_PATH)
    for key, expected in {
        "schema_version": eu.CONFIG_SCHEMA_VERSION,
        "run_id": eu.RUN_ID,
        "stage": 8,
        "manifest_fingerprint": fingerprint,
    }.items():
        if str(index.get(key)) != str(expected):
            raise ValueError(f"Stage 8 artifact index mismatch for {key}")
    indexed_hashes = index.get("sha256")
    if not isinstance(indexed_hashes, Mapping):
        raise ValueError("Stage 8 artifact index has no sha256 map")
    for filename, digest in indexed_hashes.items():
        path = STAGE8_DIR / filename
        if not path.is_file() or eu.sha256_file(path) != digest:
            raise ValueError(f"Stage 8 artifact is missing or changed: {path}")
    sequence_path = _resolve(SEQUENCE_CONFIG_PATH)
    learning_rate_path = _resolve(LEARNING_RATE_CONFIG_PATH)
    if marker.get("sequence_length_config_sha256") != eu.sha256_file(sequence_path):
        raise ValueError("sequence_length.json changed after Stage 8 publication")
    if marker.get("learning_rate_config_sha256") != eu.sha256_file(learning_rate_path):
        raise ValueError("learning_rates.json changed after Stage 8 publication")
    shared = eu.load_shared_run_config(
        sequence_length_path=sequence_path,
        learning_rate_path=learning_rate_path,
        require_context=True,
        require_context_lr=True,
    )
    latency = _read_json(STAGE8_LATENCY_PATH)
    if latency.get("schema_version") != eu.CONFIG_SCHEMA_VERSION or latency.get("run_id") != eu.RUN_ID:
        raise ValueError("Stage 8 latency data uses an old schema or run namespace")
    if latency.get("manifest_fingerprint") != fingerprint:
        raise ValueError("Stage 8 latency data belongs to a different summary manifest")
    if latency.get("context_sizes") != list(CONTEXTS) or latency.get("truncation_side") != "left":
        raise ValueError("Stage 8 latency data must cover C0-C4 and use explicit left truncation")
    machine = str(latency.get("measurement_machine", "")).strip()
    if not machine:
        raise ValueError("Stage 8 latency artifact has no measurement_machine identifier")
    measurements = latency.get("measurements_ms_by_context")
    medians = latency.get("median_ms_by_context")
    if not isinstance(measurements, Mapping) or not isinstance(medians, Mapping):
        raise ValueError("Stage 8 latency artifact lacks raw measurement samples or median summaries")
    if set(latency.get("model_ids", {})) != set(eu.BACKBONES) or latency["model_ids"] != {b: eu.get_backbone(b).hf_model_id for b in eu.BACKBONES}:
        raise ValueError("Stage 8 latency model IDs do not match the registered backbone identifiers")
    seq_doc = shared["sequence_length"]
    lr_doc = shared["learning_rates"]
    expected_tokenizer_revisions = seq_doc.get("context_tokenizer_resolved_revisions")
    preflight = lr_doc.get("context_preflight")
    expected_model_revisions = preflight.get("model_revisions") if isinstance(preflight, Mapping) else None
    if not isinstance(expected_tokenizer_revisions, Mapping) or not isinstance(expected_model_revisions, Mapping):
        raise ValueError("Current shared configuration does not pin contextual model and tokenizer revisions")
    if latency.get("model_revisions") != expected_model_revisions:
        raise ValueError("Stage 8 latency model revisions differ from current contextual preflight")
    if latency.get("tokenizer_revisions") != expected_tokenizer_revisions:
        raise ValueError("Stage 8 latency tokenizer revisions differ from current shared length configuration")
    chosen_by_context: dict[str, dict[str, float]] = {}
    for context in CONTEXTS:
        key = f"C{context}"
        if key not in measurements or key not in medians:
            raise ValueError(f"Stage 8 latency artifact is missing {key}")
        sample_by_backbone = measurements[key]
        median_by_backbone = medians[key]
        if not isinstance(sample_by_backbone, Mapping) or set(sample_by_backbone) != set(eu.BACKBONES):
            raise ValueError(f"Stage 8 raw latency samples for {key} must include both backbones")
        if not isinstance(median_by_backbone, Mapping) or set(median_by_backbone) != set(eu.BACKBONES):
            raise ValueError(f"Stage 8 median latency for {key} must include both backbones")
        chosen_by_context[key] = {}
        for backbone in eu.BACKBONES:
            samples = np.asarray(sample_by_backbone[backbone], dtype=float)
            if samples.ndim != 1 or len(samples) < 20 or not np.isfinite(samples).all() or (samples <= 0).any():
                raise ValueError(f"Stage 8 latency samples for {key}/{backbone} require at least 20 finite positive measurements")
            computed_median = float(np.median(samples))
            recorded_median = float(median_by_backbone[backbone])
            if not math.isclose(computed_median, recorded_median, rel_tol=1e-8, abs_tol=1e-8):
                raise ValueError(f"Stage 8 median latency mismatch for {key}/{backbone}")
            chosen_by_context[key][backbone] = recorded_median
    receipt = _read_json(STAGE8_RECEIPT_PATH)
    if receipt.get("summary_manifest_fingerprint") != fingerprint or receipt.get("latency_measurements_sha256") != eu.sha256_file(STAGE8_LATENCY_PATH):
        raise ValueError("Stage 8 publish receipt does not match the latency artifact / manifest")
    return {
        "manifest": manifest,
        "manifest_fingerprint": fingerprint,
        "marker_path": STAGE8_MARKER_PATH,
        "index_path": STAGE8_INDEX_PATH,
        "latency_path": STAGE8_LATENCY_PATH,
        "latency_document": latency,
        "sequence_path": sequence_path,
        "learning_rate_path": learning_rate_path,
        "sequence_config": seq_doc,
        "learning_rate_config": lr_doc,
        "median_ms_by_context": chosen_by_context,
        "measurement_machine": machine,
        "input_paths": [
            STAGE8_MANIFEST_PATH, STAGE8_MARKER_PATH, STAGE8_INDEX_PATH,
            STAGE8_RECEIPT_PATH, STAGE8_LATENCY_PATH, sequence_path, learning_rate_path,
            *[STAGE8_DIR / filename for filename in indexed_hashes],
        ],
    }


def _stage5_exp1_leader() -> tuple[str | None, list[Path]]:
    """Read the Stage 5 result as discussion-only metadata, if it is present."""
    paths = [STAGE5_DIR / name for name in (
        "e1_results.csv", "e1_paired_bootstrap.csv", "e1_summary.json", "artifacts_sha256.json", eu.RUN_MANIFEST_NAME
    )]
    present = [path.is_file() for path in paths]
    if not any(present):
        return None, []
    if not all(present):
        missing = [str(path) for path in paths if not path.is_file()]
        raise FileNotFoundError(f"Stage 5 outputs are partial; archive/fix them before Stage 11: {missing}")
    manifest = _read_json(STAGE5_MANIFEST_PATH)
    fingerprint = str(manifest.get("fingerprint", ""))
    eu.assert_current_run_artifact(
        STAGE5_DIR / "e1_results.csv",
        expected_stage=5,
        expected_script="scripts/5_evaluate_e1.py",
        expected_fingerprint=fingerprint,
    )
    index = _read_json(STAGE5_INDEX_PATH)
    if index.get("manifest_fingerprint") != fingerprint or index.get("run_id") != eu.RUN_ID or index.get("stage") != 5:
        raise ValueError("Stage 5 output index is stale or belongs to a different run")
    for filename in ("e1_results.csv", "e1_paired_bootstrap.csv", "e1_summary.json"):
        if index.get("sha256", {}).get(filename) != eu.sha256_file(STAGE5_DIR / filename):
            raise ValueError(f"Stage 5 output hash mismatch for {filename}")
    paired = pd.read_csv(STAGE5_BOOTSTRAP_PATH)
    row = paired.loc[paired["metric"].astype(str) == "macro_f1"]
    if len(row) != 1:
        raise ValueError("Stage 5 must contain exactly one primary macro_f1 paired-bootstrap row")
    result = row.iloc[0]
    if str(result.get("comparison")) != "E1-XLMR-Target minus E1-IndoBERTweet-Target":
        raise ValueError("Stage 5 primary comparison direction/name differs from the declared Experiment I comparison")
    if not bool(result.get("significant")):
        return None, paths
    observed = float(result.get("observed_diff"))
    if not math.isfinite(observed) or observed == 0:
        raise ValueError("Stage 5 significant Experiment I macro-F1 difference has invalid observed_diff")
    return ("xlmr" if observed > 0 else "indobertweet"), paths


def _input_paths_stage2() -> list[Path]:
    manifest = eu.assert_current_stage_manifest(2, PROJECT_ROOT)
    return [STAGE2_MANIFEST_PATH, _resolve(MATCHED_PATH)]


def _collect_all_decision_inputs(
    stage8_audit: Mapping[str, Any],
    grid_outputs: Mapping[str, Mapping[str, Any]],
    stage5_paths: Sequence[Path],
) -> list[Path]:
    paths: list[Path] = []
    paths.extend(_input_paths_stage2())
    paths.extend(stage8_audit["input_paths"])
    paths.extend(stage5_paths)
    for backbone in eu.BACKBONES:
        paths.extend(grid_outputs[backbone]["input_paths"])
    # Preserve order but remove duplicates.
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        normalized = Path(path).resolve()
        if normalized not in seen:
            seen.add(normalized)
            unique.append(normalized)
    for path in unique:
        if not path.is_file():
            raise FileNotFoundError(f"Stage 11 input disappeared before manifest creation: {path}")
    return unique


def _prepare_outputs(
    *,
    decision: dict[str, Any],
    context_decisions: Mapping[str, Mapping[str, Any]],
    grid_audits: Mapping[str, Mapping[str, Any]],
    input_paths: Sequence[Path],
    stage8_audit: Mapping[str, Any],
    matched_audit: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    """Create the immutable Stage 11 manifest and validate/create output folder."""
    config = {
        "schema_version": eu.CONFIG_SCHEMA_VERSION,
        "run_id": eu.RUN_ID,
        "stage": STAGE,
        "script_name": SCRIPT_NAME,
        "input_mode": "contextual_oof_decision",
        "dataset_sha256": eu.sha256_file(_resolve(MATCHED_PATH)),
        "fold_assignment_sha256": matched_audit["fold_assignment_sha256"],
        "post_id_sha256": matched_audit["post_id_sha256"],
        "n_oof_examples": EXPECTED_ROWS,
        "max_length_context": int(stage8_audit["sequence_config"]["max_length_context"]),
        "truncation_side": "left",
        "backbones": list(eu.BACKBONES),
        "context_decisions": {key: dict(value) for key, value in context_decisions.items()},
        "grid_audits": {key: dict(value) for key, value in grid_audits.items()},
        "decision_result": dict(decision),
        "latency_ms": dict(decision["latency_ms"]),
        "latency_machine": decision["latency_machine"],
    }
    manifest = eu.make_run_manifest(
        stage=STAGE,
        script_name=SCRIPT_NAME,
        config=config,
        input_paths=input_paths,
        metadata={
            "purpose": "Stage 11 paired OOF context/backbone decision gate",
            "source_grid_fingerprints": decision["source_grid_fingerprints"],
            "context_rule_version": eu.CONTEXT_RULE_VERSION,
            "backbone_rule_version": eu.BACKBONE_RULE_VERSION,
            "selected_backbone": decision["decision"],
            "selected_context_by_backbone": decision["c_best_by_backbone"],
        },
        project_root=PROJECT_ROOT,
    )
    run_dir = STAGE11_OUTPUT_DIR
    if DECISION_PATH.exists():
        # Existing decisions can only be kept if they already have this exact
        # current Stage 11 manifest and a fully verified output inventory.
        if not _existing_decision_is_current(manifest):
            raise FileExistsError(
                f"{DECISION_PATH} already exists but is not a complete, matching current-run Stage 11 result. "
                "Archive the old decision and its stage_11/decision directory before regenerating it."
            )
        print(f"Existing Stage 11 decision passed exact manifest/hash verification: {DECISION_PATH}")
        return run_dir, manifest
    if run_dir.exists():
        existing_files = [p for p in run_dir.iterdir() if p.is_file()]
        if existing_files and not (run_dir / eu.RUN_MANIFEST_NAME).is_file():
            raise RuntimeError(f"Stage 11 output directory contains unmanifested files: {run_dir}; archive it")
        if (run_dir / eu.RUN_MANIFEST_NAME).is_file():
            old_manifest = _read_json(run_dir / eu.RUN_MANIFEST_NAME)
            if old_manifest.get("fingerprint") != manifest.get("fingerprint"):
                raise RuntimeError(f"Stage 11 output directory has stale manifest: {run_dir}; archive it")
            if any(p.name != eu.RUN_MANIFEST_NAME for p in run_dir.iterdir() if p.is_file()):
                raise RuntimeError(f"Stage 11 output directory contains partial prior artifacts: {run_dir}; archive it")
    eu.prepare_run_directory(run_dir, manifest, allow_resume=True)
    return run_dir, manifest


def _existing_decision_is_current(expected_manifest: Mapping[str, Any]) -> bool:
    """Only accept an existing decision when the complete current output bundle verifies."""
    try:
        manifest_path = STAGE11_OUTPUT_DIR / eu.RUN_MANIFEST_NAME
        if not manifest_path.is_file():
            return False
        existing_manifest = _read_json(manifest_path)
        if existing_manifest.get("fingerprint") != expected_manifest.get("fingerprint"):
            return False
        marker = _read_json(STAGE11_MARKER_PATH)
        index = _read_json(STAGE11_INDEX_PATH)
        eu.assert_current_run_artifact(
            STAGE11_MARKER_PATH, expected_stage=11, expected_script=SCRIPT_NAME,
            expected_fingerprint=str(expected_manifest["fingerprint"]),
        )
        if marker.get("artifact_index_sha256") != eu.sha256_file(STAGE11_INDEX_PATH):
            return False
        if marker.get("root_decision_sha256") != eu.sha256_file(DECISION_PATH):
            return False
        if eu.sha256_file(STAGE11_DECISION_COPY) != eu.sha256_file(DECISION_PATH):
            return False
        if index.get("manifest_fingerprint") != expected_manifest.get("fingerprint"):
            return False
        for name, digest in index.get("sha256", {}).items():
            path = STAGE11_OUTPUT_DIR / name
            if not path.is_file() or eu.sha256_file(path) != digest:
                return False
        return marker.get("run_id") == eu.RUN_ID and marker.get("stage") == 11
    except (OSError, KeyError, TypeError, ValueError, RuntimeError):
        return False


def _publish_decision_bundle(
    *,
    run_dir: Path,
    manifest: Mapping[str, Any],
    decision: Mapping[str, Any],
    context_decisions: Mapping[str, Mapping[str, Any]],
    grid_audits: Mapping[str, Mapping[str, Any]],
) -> None:
    """Write validated decision documents and a provenance/hash record."""
    decision_with_provenance = dict(decision)
    decision_with_provenance.update({
        "stage11_manifest_fingerprint": manifest["fingerprint"],
        "stage11_output_dir": run_dir.relative_to(PROJECT_ROOT).as_posix(),
    })
    for backbone in eu.BACKBONES:
        _write_json_atomic(run_dir / f"context_selection_{backbone}.json", context_decisions[backbone])
        _write_json_atomic(run_dir / f"grid_audit_{backbone}.json", grid_audits[backbone])
    eu.save_backbone_decision(decision_with_provenance, path=STAGE11_DECISION_COPY)
    eu.save_backbone_decision(decision_with_provenance, path=DECISION_PATH)
    tracked = [
        f"context_selection_{backbone}.json" for backbone in eu.BACKBONES
    ] + [
        f"grid_audit_{backbone}.json" for backbone in eu.BACKBONES
    ] + ["backbone_decision.json"]
    _write_json_atomic(run_dir / "artifacts_sha256.json", {
        "schema_version": eu.CONFIG_SCHEMA_VERSION,
        "run_id": eu.RUN_ID,
        "stage": STAGE,
        "script_name": SCRIPT_NAME,
        "manifest_fingerprint": manifest["fingerprint"],
        "sha256": {name: eu.sha256_file(run_dir / name) for name in tracked},
    })
    _write_json_atomic(run_dir / "stage11_complete.json", {
        "schema_version": eu.CONFIG_SCHEMA_VERSION,
        "run_id": eu.RUN_ID,
        "stage": STAGE,
        "script_name": SCRIPT_NAME,
        "manifest_fingerprint": manifest["fingerprint"],
        "artifact_index_sha256": eu.sha256_file(run_dir / "artifacts_sha256.json"),
        "root_decision_path": DECISION_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "root_decision_sha256": eu.sha256_file(DECISION_PATH),
        "decision_relative_path": "backbone_decision.json",
        "source_grid_fingerprints": dict(decision["source_grid_fingerprints"]),
        "c_best_by_backbone": dict(decision["c_best_by_backbone"]),
        "selected_backbone": decision["decision"],
    })
    eu.assert_current_run_artifact(
        run_dir / "stage11_complete.json",
        expected_stage=STAGE,
        expected_script=SCRIPT_NAME,
        expected_fingerprint=str(manifest["fingerprint"]),
    )
    # Round-trip through the public loader, including Stage 11 provenance checks.
    loaded = eu.load_backbone_decision_document(DECISION_PATH)
    if loaded != decision_with_provenance:
        raise RuntimeError("Stage 11 decision failed its own save/load integrity round-trip")


def build_decision_bundle(*, project_root: Path | str = ".", n_bootstrap: int = BOOTSTRAP_REPLICATES) -> dict[str, Any]:
    """Compute Stage 11 selections from actual upstream outputs without writing results."""
    global PROJECT_ROOT
    PROJECT_ROOT = Path(project_root).resolve()
    raw_path = _resolve(MATCHED_PATH)
    if not raw_path.is_file():
        raise FileNotFoundError(f"Matched contextual subset is missing: {raw_path}; complete Stage 2 first")
    stage2_manifest = eu.assert_current_stage_manifest(2, PROJECT_ROOT)
    raw_matched = pd.read_csv(raw_path)
    matched_audit = eu.validate_context_dataset(raw_matched, expected_rows=EXPECTED_ROWS)
    matched = eu.load_modelling_view(raw_matched, include_fold=True)
    matched[eu.CTX_FOLD_COLUMN] = pd.to_numeric(matched[eu.CTX_FOLD_COLUMN], errors="raise").astype(int)
    matched = matched.sort_values(eu.CTX_ID_COLUMN, key=lambda col: col.astype(str)).reset_index(drop=True)
    canonical_ids = matched[eu.CTX_ID_COLUMN].astype(str).tolist()
    canonical_y_series = matched["label"].map(eu.LABEL2ID)
    if canonical_y_series.isna().any():
        raise ValueError("Matched subset contains a label outside the canonical LABEL2ID mapping")
    canonical_y = canonical_y_series.to_numpy(dtype=np.int64)

    eu.assert_no_legacy_outputs(_legacy_paths())
    stage8_audit = _validate_stage8_latency_and_configs()
    if stage8_audit["latency_document"].get("matched_subset_sha256") != eu.sha256_file(raw_path):
        raise ValueError("Stage 8 latency measurements used a different matched contextual CSV")
    if stage8_audit["latency_document"].get("fold_assignment_sha256") != matched_audit["fold_assignment_sha256"]:
        raise ValueError("Stage 8 latency measurements used a different shared fold assignment")
    latency_doc = stage8_audit["latency_document"]
    if latency_doc.get("sample_post_ids_sha256") != eu.sha256_json(latency_doc.get("sample_post_ids", [])):
        raise ValueError("Stage 8 latency sample post-ID hash is invalid")
    canonical_fold_by_id = dict(zip(canonical_ids, matched[eu.CTX_FOLD_COLUMN].astype(int).tolist()))
    latency_sample_ids = [str(value) for value in latency_doc.get("sample_post_ids", [])]
    if not latency_sample_ids or any(post_id not in canonical_fold_by_id for post_id in latency_sample_ids):
        raise ValueError("Stage 8 latency sample IDs are empty or do not belong to the current matched subset")
    if any(canonical_fold_by_id[post_id] == 0 for post_id in latency_sample_ids):
        raise ValueError("Stage 8 latency sample included fold-0 held-out examples; this would violate the pre-declared pilot split")

    # Validate revisions and the resolved shared length against each logged grid
    # before reading probabilities. They must be exactly the Stage 8 revisions.
    revision_cfg = stage8_audit["learning_rate_config"].get("context_preflight", {})
    model_revisions = revision_cfg.get("model_revisions", {}) if isinstance(revision_cfg, Mapping) else {}
    tokenizer_revisions = stage8_audit["sequence_config"].get("context_tokenizer_resolved_revisions", {})
    if set(model_revisions) != set(eu.BACKBONES) or set(tokenizer_revisions) != set(eu.BACKBONES):
        raise ValueError("Stage 8 configurations must pin model and tokenizer revisions for both backbones")

    grid_outputs: dict[str, dict[str, Any]] = {}
    context_decisions: dict[str, dict[str, Any]] = {}
    for backbone in eu.BACKBONES:
        output = _load_and_audit_grid(
            backbone, matched, canonical_y,
            expected_model_revision=str(model_revisions[backbone]),
            expected_tokenizer_revision=str(tokenizer_revisions[backbone]),
            expected_max_length=int(stage8_audit["sequence_config"]["max_length_context"]),
            expected_dataset_sha256=eu.sha256_file(raw_path),
            expected_fold_sha256=matched_audit["fold_assignment_sha256"],
        )
        if output["canonical_ids"] != canonical_ids:
            raise ValueError(f"Internal ID alignment problem for {backbone}")
        grid_outputs[backbone] = output
        context_decisions[backbone] = eu.select_context_size_from_oof(
            canonical_y,
            output["probabilities_by_context"],
            canonical_ids,
            post_ids_by_context={context: canonical_ids for context in CONTEXTS},
            backbone=backbone,
            n_bootstrap=n_bootstrap,
            ci=CONFIDENCE_LEVEL,
            random_state=BOOTSTRAP_SEED,
            expected_n=EXPECTED_ROWS,
            expected_seed_count=len(SEEDS),
        )

    grid_audits = {backbone: grid_outputs[backbone]["grid_audit"] for backbone in eu.BACKBONES}
    latencies = _latency_at_selected_context(
        stage8_audit["median_ms_by_context"],
        {backbone: int(context_decisions[backbone]["selected_context"]) for backbone in eu.BACKBONES},
    )

    preds_c0 = {
        backbone: eu.ensemble_seed_probabilities(grid_outputs[backbone]["probabilities_by_context"][0])
        for backbone in eu.BACKBONES
    }
    preds_c_best = {
        backbone: eu.ensemble_seed_probabilities(
            grid_outputs[backbone]["probabilities_by_context"][int(context_decisions[backbone]["selected_context"])]
        )
        for backbone in eu.BACKBONES
    }
    exp1_leader, stage5_paths = _stage5_exp1_leader()
    decision = eu.select_backbone(
        canonical_y,
        preds_c_best=preds_c_best,
        preds_c0=preds_c0,
        latency_ms=latencies,
        exp1_leader=exp1_leader,
        n_bootstrap=n_bootstrap,
        ci=CONFIDENCE_LEVEL,
        random_state=BOOTSTRAP_SEED,
        context_decisions=context_decisions,
        context_sizes_by_backbone={b: int(context_decisions[b]["selected_context"]) for b in eu.BACKBONES},
        post_ids=canonical_ids,
        latency_machine=stage8_audit["measurement_machine"],
        grid_audits=grid_audits,
        post_ids_by_prediction={
            "C_best": {b: canonical_ids for b in eu.BACKBONES},
            "C0": {b: canonical_ids for b in eu.BACKBONES},
        },
        expected_n=EXPECTED_ROWS,
    )
    decision_inputs = _collect_all_decision_inputs(stage8_audit, grid_outputs, stage5_paths)
    if raw_path.resolve() not in {p.resolve() for p in decision_inputs}:
        decision_inputs.append(raw_path.resolve())
    return {
        "decision": decision,
        "context_decisions": context_decisions,
        "grid_audits": grid_audits,
        "grid_outputs": grid_outputs,
        "stage8_audit": stage8_audit,
        "matched_audit": matched_audit,
        "stage2_manifest": stage2_manifest,
        "input_paths": decision_inputs,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select C_best per backbone and make the Stage 11 backbone decision.")
    parser.add_argument("--validate-only", action="store_true", help="Audit all upstream artifacts and compute the decision without writing Stage 11 outputs.")
    parser.add_argument("--n-bootstrap", type=int, default=BOOTSTRAP_REPLICATES, help="Must remain 10,000 under the frozen protocol.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.n_bootstrap != BOOTSTRAP_REPLICATES:
        raise ValueError(f"The protocol requires exactly {BOOTSTRAP_REPLICATES:,} paired-bootstrap resamples")
    if not PROJECT_ROOT.joinpath("experiment_protocol.md").is_file() or not PROJECT_ROOT.joinpath("pipeline.md").is_file():
        raise FileNotFoundError("experiment_protocol.md and pipeline.md must exist at the repository root")
    if not (PROJECT_ROOT / SCRIPT_NAME).is_file():
        raise FileNotFoundError(f"Authoritative Stage 11 script is missing: {PROJECT_ROOT / SCRIPT_NAME}")
    print(f"Run ID: {eu.RUN_ID}")
    if not args.validate_only:
        # Avoid spending time on ten-thousand-resample bootstrap calculations if
        # the remaining scripts/directories have not yet been migrated.
        eu.audit_pipeline_script_names(PROJECT_ROOT, require_all=True)
    print("Auditing Stage 2 data, Stage 5 discussion-only results (when available), Stage 8 latency/configuration, and both 50-run CV grids...")
    bundle = build_decision_bundle(project_root=PROJECT_ROOT, n_bootstrap=args.n_bootstrap)
    decision = bundle["decision"]
    print("\nContext selection results:")
    for backbone in eu.BACKBONES:
        context_doc = bundle["context_decisions"][backbone]
        print(f"  {backbone}: provisional C{context_doc['provisional_top_context']} -> C{context_doc['selected_context']} ({context_doc['selection_reason']})")
        print(f"    pooled OOF macro-F1: {context_doc['pooled_oof_macro_f1']}")
    print("\nBackbone gate:")
    print(f"  C_best paired macro-F1 difference (XLM-R - IndoBERTweet): {decision['primary_c_best']}")
    print(f"  C0 guard: {decision['guard_c0']}")
    print(f"  median latency at own C_best (ms): {decision['latency_ms']}")
    print(f"  decision: {decision['decision']} ({decision['reason']})")

    migration = eu.audit_pipeline_script_names(PROJECT_ROOT, require_all=False)
    if args.validate_only:
        print("\nValidation-only complete: all upstream source artifacts are verified and the decision has been computed in memory; no decision files were written.")
        if migration["complete"]:
            print("Full pipeline filename/directory audit: passed.")
        else:
            print("Full pipeline filename/directory audit is not complete yet; decision publication remains blocked.")
            print(f"Missing scripts: {migration['missing']}")
            print(f"Missing directories: {migration['missing_directories']}")
            print(f"Old filenames still present: {migration['rename_leftovers']}")
        return

    # Follow the agreed policy: finish revision/testing of all stage files before
    # starting the clean run. Decision publication remains blocked until every
    # required script and extension directory is present under the current names.
    eu.audit_pipeline_script_names(PROJECT_ROOT, require_all=True)
    eu.validate_backbone_decision_document(decision)
    run_dir, manifest = _prepare_outputs(
        decision=decision,
        context_decisions=bundle["context_decisions"],
        grid_audits=bundle["grid_audits"],
        input_paths=bundle["input_paths"],
        stage8_audit=bundle["stage8_audit"],
        matched_audit=bundle["matched_audit"],
    )
    if _existing_decision_is_current(manifest):
        print(f"\nStage 11 is already complete and verified: {DECISION_PATH}")
        return
    _publish_decision_bundle(
        run_dir=run_dir,
        manifest=manifest,
        decision=decision,
        context_decisions=bundle["context_decisions"],
        grid_audits=bundle["grid_audits"],
    )
    print("\nStage 11 completed successfully.")
    print(f"Selected backbone: {decision['decision']}")
    print(f"C_best: {decision['c_best_by_backbone']}")
    print(f"Decision file: {DECISION_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Manifest: {(run_dir / eu.RUN_MANIFEST_NAME).relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
