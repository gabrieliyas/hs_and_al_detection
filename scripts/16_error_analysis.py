"""Stage 16: prediction-driven OOF error analysis for Chapter IV.

Primary comparison
------------------
Compare the selected backbone's Experiment II-A / ITFT target-only C0 predictions
with its Stage-11-selected C_best predictions on exactly the same 900 matched posts.
Both conditions use the same fold assignments, held-out rows, and seed ensemble, so
this is the cleanest available OOF contrast for examining the effect of context.

Outputs
-------
Authoritative, immutable bundle:
    results/runs/left_truncation_v2/stage_16/
Editable human-review worksheet (created once, outside the immutable bundle):
    results/manual_review/stage16_manual_review_sample.csv
Thesis tables/figures:
    results/tables/left_truncation_v2/stage16_*.csv
    results/figures/left_truncation_v2/stage16_*.png

The default manual-review size (90) is a target, not an unconditional quota. The
script adapts it to the available cases and uses reproducible, error-outcome-aware
allocation across the observed (outcome x context_type) strata. Frequencies in the
manually sampled worksheet are NOT population prevalence estimates. The full 900-row
ledger and quantitative tables are the population-level descriptive summaries.

After filling the editable worksheet, run:
    python scripts/16_error_analysis.py --finalize-review
This creates review-mechanism summaries separately from the immutable base bundle.
No model training is performed by this script.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from collections import Counter
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

STAGE = 16
SCRIPT_NAME = "scripts/16_error_analysis.py"
EXPECTED_ROWS = int(eu.CONTEXT_MATCHED_EXPECTED_ROWS)
CONDITIONS = tuple(range(5))
FOLDS = tuple(range(5))
SEEDS = tuple(eu.CV_SEEDS)
N_BOOTSTRAP = 10_000
CI_LEVEL = 95.0
BOOTSTRAP_SEED = 42
DEFAULT_REVIEW_SAMPLE_SIZE = 90
DEFAULT_REVIEW_SEED = 20261026

RUN_ROOT = eu.versioned_run_root(PROJECT_ROOT)
STAGE11_DIR = RUN_ROOT / "stage_11" / "decision"
STAGE15_DIR = RUN_ROOT / "stage_15"
STAGE_ROOT = RUN_ROOT / "stage_16"
MATCHED_PATH = PROJECT_ROOT / "data/processed/ctx_matched_subset.csv"
BACKBONE_DECISION_PATH = PROJECT_ROOT / eu.BACKBONE_DECISION_PATH
STRATEGY_DECISION_PATH = PROJECT_ROOT / eu.TRAINING_STRATEGY_DECISION_PATH
TABLES_DIR = PROJECT_ROOT / "results/tables" / eu.RUN_ID
FIGURES_DIR = PROJECT_ROOT / "results/figures" / eu.RUN_ID
REVIEW_DIR = PROJECT_ROOT / "results/manual_review"
REVIEW_WORKSHEET_PATH = REVIEW_DIR / "stage16_manual_review_sample.csv"

LABEL_CODES = tuple(eu.LABEL2ID.keys())
LABEL_IDS = tuple(eu.LABEL2ID.values())

# Human-applied hypotheses for the source of an observed model error. These are
# candidate review codes, not automatically inferred labels or verified findings.
ERROR_MECHANISMS = [
    {
        "code": "REFERENT_OR_TARGET_RESOLUTION",
        "label": "Referent or target resolution",
        "definition": "A parent turn clarifies who or what the target refers to, who is being addressed, or which entity a potentially harmful expression targets.",
        "include_when": "The error plausibly depends on an implicit person, group, entity, or antecedent that becomes clear from the thread.",
        "exclude_when": "The referent is already explicit in the target and context adds no meaningful resolution.",
    },
    {
        "code": "QUOTATION_OR_MENTION_VS_ENDORSEMENT",
        "label": "Quotation/mention versus endorsement",
        "definition": "The target quotes, reports, condemns, discusses, or reproduces harmful language rather than using it as an endorsed attack (or the reverse).",
        "include_when": "Speaker stance toward quoted or mentioned language is material to the gold label but may be missed by the model.",
        "exclude_when": "The post clearly directs the expression as an insult without a meaningful quotation or reporting distinction.",
    },
    {
        "code": "SARCASM_IRONY_OR_PRAGMATIC_REVERSAL",
        "label": "Sarcasm, irony, or pragmatic reversal",
        "definition": "Literal wording and intended communicative force differ because of sarcasm, irony, mock praise, joking, or pragmatic reversal.",
        "include_when": "The conversational exchange materially changes the likely interpretation of the target's tone or intent.",
        "exclude_when": "The wording is straightforward and the error is better explained by another mechanism.",
    },
    {
        "code": "PERSONAL_ABUSE_VS_GROUP_DIRECTED_HATE",
        "label": "Personal abuse versus group-directed hate",
        "definition": "The distinction between an insult or abusive attack and hate directed at a group is missed, or group identity is inferred incorrectly.",
        "include_when": "The error concerns the boundary between AL and HS or whether a group is the target.",
        "exclude_when": "The true issue is unrelated to target group or abuse-versus-hate distinctions.",
    },
    {
        "code": "LEXICAL_AMBIGUITY_OR_SLUR_SENSE",
        "label": "Lexical ambiguity or slur sense",
        "definition": "Slang, reclaimed language, polysemy, a potentially hateful term, or local usage is interpreted without the sense supplied by context.",
        "include_when": "A term has multiple plausible meanings or its harmful/non-harmful sense depends on discourse context.",
        "exclude_when": "The word's meaning is clear and the principal error concerns stance or target instead.",
    },
    {
        "code": "DISCOURSE_RELATION_OR_STANCE",
        "label": "Discourse relation or stance",
        "definition": "The relationship between turns—agreement, rebuttal, denial, continuation, provocation, response, or stance—is misunderstood.",
        "include_when": "A preceding turn changes whether the target agrees with, rejects, responds to, or amplifies a harmful statement.",
        "exclude_when": "There is no consequential relationship between the available parent turn(s) and the target.",
    },
    {
        "code": "IRRELEVANT_MISLEADING_OR_INSUFFICIENT_CONTEXT",
        "label": "Irrelevant, misleading, or insufficient context",
        "definition": "The supplied parent turns do not provide the required evidence, are only weakly related, conflict with the target, or may distract from the target's decisive cues.",
        "include_when": "The context window appears unhelpful, noisy, contradictory, or still insufficient to resolve the label reliably.",
        "exclude_when": "A more specific mechanism above better accounts for the error.",
    },
]
REVIEW_ONLY_CODES = [
    {"code": "OTHER", "label": "Other mechanism", "definition": "A clearly supported error mechanism not captured by the seven candidate codes."},
    {"code": "UNCERTAIN_NEEDS_ADJUDICATION", "label": "Uncertain / needs adjudication", "definition": "Available evidence does not support a reliable primary mechanism; retain notes and discuss or adjudicate."},
    {"code": "NOT_APPLICABLE_NO_ERROR", "label": "No error / correct control", "definition": "Use for a reviewed both-correct or same-condition-correct control; do not count it as an error mechanism."},
]
VALID_REVIEW_CODES = {item["code"] for item in ERROR_MECHANISMS + REVIEW_ONLY_CODES}


# ------------------------------- IO helpers ---------------------------------

def _read_json(path: Path) -> dict[str, Any]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read valid JSON at {path}: {exc}") from exc
    if not isinstance(obj, dict):
        raise ValueError(f"Expected a JSON object at {path}")
    return obj


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, path)


def _sha(path: Path) -> str:
    return eu.sha256_file(path)


def _safe_relative_path(root: Path, relative: str) -> Path:
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts:
        raise RuntimeError(f"Manifest contains an unsafe relative path: {relative}")
    path = (root / rel).resolve()
    if not path.is_relative_to(root.resolve()):
        raise RuntimeError(f"Manifest path escapes project root: {relative}")
    return path


def _verify_manifest_inputs(manifest: Mapping[str, Any], root: Path = PROJECT_ROOT) -> None:
    for relative, expected in manifest.get("input_sha256", {}).items():
        path = _safe_relative_path(root, str(relative))
        if not path.is_file():
            raise FileNotFoundError(f"Manifest input is missing: {path}")
        if _sha(path) != expected:
            raise ValueError(f"Manifest input hash changed: {path}")


def _verify_manifest_source_fingerprints(manifest: Mapping[str, Any], *, stage: int, root: Path = PROJECT_ROOT) -> None:
    expected_script = eu.PIPELINE_SCRIPTS[stage]
    source_files = {
        "utility_sha256": Path(eu.__file__).resolve(),
        "script_sha256": root / expected_script,
        "protocol_sha256": root / "experiment_protocol.md",
        "pipeline_sha256": root / "pipeline.md",
    }
    for key, path in source_files.items():
        if not path.is_file() or _sha(path) != manifest.get(key):
            raise RuntimeError(f"Stage {stage} artifact was produced using a different source fingerprint: {key}; rerun upstream stages after freezing the files")


def _verify_stage_bundle(stage_dir: Path, *, stage: int, script_name: str, marker_name: str, check_inputs: bool = True) -> dict[str, Any]:
    manifest_path = stage_dir / eu.RUN_MANIFEST_NAME
    index_path = stage_dir / "artifacts_sha256.json"
    marker_path = stage_dir / marker_name
    for path in (manifest_path, index_path, marker_path):
        if not path.is_file():
            raise FileNotFoundError(f"Stage {stage} bundle missing required artifact: {path}")
    manifest = _read_json(manifest_path)
    eu._validate_manifest_integrity(manifest)
    fingerprint = str(manifest["fingerprint"])
    if str(manifest.get("stage")) != str(stage) or manifest.get("script_name") != script_name:
        raise ValueError(f"Invalid Stage {stage} identity in {manifest_path}")
    _verify_manifest_source_fingerprints(manifest, stage=stage)
    if check_inputs:
        _verify_manifest_inputs(manifest)
    index = _read_json(index_path)
    for key, expected in {
        "schema_version": eu.CONFIG_SCHEMA_VERSION,
        "run_id": eu.RUN_ID,
        "stage": stage,
        "script_name": script_name,
        "manifest_fingerprint": fingerprint,
    }.items():
        if str(index.get(key)) != str(expected):
            raise ValueError(f"Stage {stage} artifact index mismatch for {key}: {index.get(key)!r}")
    hashes = index.get("sha256")
    if not isinstance(hashes, Mapping):
        raise ValueError(f"Stage {stage} artifact index has no sha256 map")
    for rel, expected_hash in hashes.items():
        path = stage_dir / str(rel)
        if not path.is_file() or _sha(path) != expected_hash:
            raise ValueError(f"Stage {stage} artifact hash mismatch/missing: {path}")
    marker = _read_json(marker_path)
    if marker.get("manifest_fingerprint") != fingerprint or marker.get("artifact_index_sha256") != _sha(index_path):
        raise ValueError(f"Stage {stage} completion marker does not match its manifest/index")
    eu.assert_current_run_artifact(marker_path, expected_stage=stage, expected_script=script_name, expected_fingerprint=fingerprint)
    return {"dir": stage_dir, "manifest": manifest, "fingerprint": fingerprint, "index": index, "marker": marker}


def _verify_leaf_run(run_dir: Path, *, stage: int, backbone: str, context_size: int, fold: int, seed: int,
                     expected_ids: Sequence[str], expected_y: np.ndarray, dataset_sha: str,
                     fold_assignment_sha: str, max_length_context: int) -> dict[str, Any]:
    required_names = (eu.RUN_MANIFEST_NAME, "run_complete.json", "artifacts_sha256.json", "predictions.csv", "probabilities.npy", "metrics.json")
    missing = [str(run_dir / name) for name in required_names if not (run_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete Stage {stage} CV run {run_dir}; missing {missing}")
    manifest = _read_json(run_dir / eu.RUN_MANIFEST_NAME)
    eu._validate_manifest_integrity(manifest)
    fingerprint = str(manifest["fingerprint"])
    if str(manifest.get("stage")) != str(stage) or manifest.get("script_name") != eu.PIPELINE_SCRIPTS[stage]:
        raise ValueError(f"Manifest stage/script mismatch in {run_dir}")
    _verify_manifest_source_fingerprints(manifest, stage=stage)
    _verify_manifest_inputs(manifest)
    cfg = manifest.get("config", {})
    expected_config = {
        "backbone": backbone,
        "context_size": context_size,
        "fold": fold,
        "seed": seed,
        "dataset_sha256": dataset_sha,
        "fold_assignment_sha256": fold_assignment_sha,
        "max_length_context": max_length_context,
        "truncation_side": "left",
    }
    for key, expected in expected_config.items():
        if str(cfg.get(key)) != str(expected):
            raise ValueError(f"Run config mismatch in {run_dir}: {key}={cfg.get(key)!r}; expected {expected!r}")
    index_path = run_dir / "artifacts_sha256.json"
    index = _read_json(index_path)
    expected_index = {
        "schema_version": eu.CONFIG_SCHEMA_VERSION, "run_id": eu.RUN_ID,
        "stage": stage, "script_name": eu.PIPELINE_SCRIPTS[stage],
        "backbone": backbone, "context_size": context_size, "fold": fold,
        "seed": seed, "manifest_fingerprint": fingerprint,
    }
    for key, expected in expected_index.items():
        if str(index.get(key)) != str(expected):
            raise ValueError(f"CV artifact-index mismatch in {run_dir}: {key}")
    for name in ("predictions.csv", "probabilities.npy", "metrics.json"):
        if index.get("sha256", {}).get(name) != _sha(run_dir / name):
            raise ValueError(f"CV artifact hash mismatch: {run_dir / name}")
    marker = _read_json(run_dir / "run_complete.json")
    for key, expected in {"manifest_fingerprint": fingerprint, "artifact_index_sha256": _sha(index_path), "backbone": backbone,
                          "context_size": context_size, "fold": fold, "seed": seed}.items():
        if str(marker.get(key)) != str(expected):
            raise ValueError(f"CV completion marker mismatch in {run_dir}: {key}")
    eu.assert_current_run_artifact(run_dir / "predictions.csv", expected_stage=stage,
                                   expected_script=eu.PIPELINE_SCRIPTS[stage], expected_fingerprint=fingerprint)

    pred = pd.read_csv(run_dir / "predictions.csv")
    required_pred_cols = {eu.CTX_ID_COLUMN, "y_true", "y_pred"}
    if not required_pred_cols.issubset(pred.columns):
        raise ValueError(f"Prediction file lacks {sorted(required_pred_cols - set(pred.columns))}: {run_dir}")
    if pred[eu.CTX_ID_COLUMN].astype(str).tolist() != [str(item) for item in expected_ids]:
        raise ValueError(f"OOF IDs are not in canonical held-out order: {run_dir}")
    y_true = _map_labels(pred["y_true"], f"{run_dir}/y_true")
    y_pred = _map_labels(pred["y_pred"], f"{run_dir}/y_pred")
    if not np.array_equal(y_true, expected_y):
        raise ValueError(f"OOF gold labels disagree with ctx_matched_subset.csv: {run_dir}")
    probs = np.load(run_dir / "probabilities.npy", allow_pickle=False)
    if probs.shape != (len(expected_ids), len(eu.LABEL2ID)):
        raise ValueError(f"Invalid probability array shape {probs.shape} in {run_dir}")
    if not np.isfinite(probs).all() or (probs < 0).any() or not np.allclose(probs.sum(axis=1), 1.0, atol=1e-4):
        raise ValueError(f"Invalid probability values in {run_dir}")
    if not np.array_equal(np.argmax(probs, axis=1), y_pred):
        raise ValueError(f"Prediction CSV and probability argmax disagree: {run_dir}")
    metric_doc = _read_json(run_dir / "metrics.json")
    for key, expected in {"backbone": backbone, "context_size": context_size, "fold": fold, "seed": seed,
                          "n_held_out": len(expected_ids), "dataset_sha256": dataset_sha,
                          "fold_assignment_sha256": fold_assignment_sha, "max_length_context": max_length_context,
                          "truncation_side": "left", "manifest_fingerprint": fingerprint}.items():
        if str(metric_doc.get(key)) != str(expected):
            raise ValueError(f"Metrics metadata mismatch in {run_dir}: {key}")
    return {"predictions": pred, "probabilities": probs, "manifest_fingerprint": fingerprint, "run_dir": run_dir}


def _map_labels(values: Sequence[Any] | pd.Series, where: str) -> np.ndarray:
    normalized = pd.Series(values).astype(str).str.strip()
    # Persisted CV prediction files use N/AL/HS. Accept exact numeric IDs only
    # as a defensive compatibility path; other vocabularies fail closed.
    mapped = normalized.map(eu.LABEL2ID)
    if mapped.isna().any():
        maybe_numeric = pd.to_numeric(normalized, errors="coerce")
        if maybe_numeric.notna().all() and maybe_numeric.astype(int).isin(LABEL_IDS).all():
            return maybe_numeric.astype(int).to_numpy(dtype=np.int64)
        bad = sorted(normalized[mapped.isna()].unique().tolist())
        raise ValueError(f"Unknown labels in {where}: {bad}")
    return mapped.to_numpy(dtype=np.int64)


def _label_name(value: int) -> str:
    return {0: "Neither", 1: "Abusive Language", 2: "Hate Speech"}[int(value)]


# ---------------------------- OOF data loading ------------------------------
def _load_upstream(*, review_sample_size: int, review_seed: int, validate_only: bool = False) -> dict[str, Any]:
    for path in (MATCHED_PATH, BACKBONE_DECISION_PATH, STRATEGY_DECISION_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Stage 16 requires upstream artifact: {path}")
    stage11 = _verify_stage_bundle(STAGE11_DIR, stage=11, script_name=eu.PIPELINE_SCRIPTS[11], marker_name="stage11_complete.json")
    canonical_decision = eu.load_backbone_decision_document(BACKBONE_DECISION_PATH)
    stage11_decision_path = STAGE11_DIR / "backbone_decision.json"
    if not stage11_decision_path.is_file() or _sha(stage11_decision_path) != _sha(BACKBONE_DECISION_PATH):
        raise ValueError("Canonical and versioned Stage 11 decision files differ")
    if canonical_decision.get("stage11_manifest_fingerprint") != stage11["fingerprint"]:
        raise ValueError("Canonical backbone decision does not reference the current Stage 11 manifest")

    # This utility validator verifies the Stage 15 manifest, all producer inputs,
    # the output index, completion marker, Stage 11 provenance, and strategy copy.
    strategy_decision = eu.load_training_strategy_decision(STRATEGY_DECISION_PATH, verify_provenance=True, project_root=PROJECT_ROOT)
    if strategy_decision.get("provenance", {}).get("stage11_manifest_fingerprint") != stage11["fingerprint"]:
        raise ValueError("Stage 15 strategy decision refers to a different Stage 11 manifest")
    if not (STAGE15_DIR / "stage15_summary.json").is_file():
        raise FileNotFoundError(f"Stage 15 summary is missing: {STAGE15_DIR / 'stage15_summary.json'}")

    raw = pd.read_csv(MATCHED_PATH)
    audit = eu.validate_context_dataset(raw, expected_rows=EXPECTED_ROWS)
    matched = raw.sort_values(eu.CTX_ID_COLUMN, key=lambda col: col.astype(str), kind="mergesort").reset_index(drop=True)
    if matched[eu.CTX_ID_COLUMN].astype(str).duplicated().any():
        raise ValueError("Matched subset contains duplicate post_id values")
    gold_ids = matched[eu.CTX_ID_COLUMN].astype(str).tolist()
    gold = _map_labels(matched["label"], str(MATCHED_PATH))
    folds = pd.to_numeric(matched[eu.CTX_FOLD_COLUMN], errors="raise").astype(int).to_numpy()
    id_to_index = {post_id: i for i, post_id in enumerate(gold_ids)}
    dataset_sha = _sha(MATCHED_PATH)
    fold_assignment_sha = audit["fold_assignment_sha256"]

    backbone = str(canonical_decision["decision"])
    c_best = int(canonical_decision["c_best_by_backbone"][backbone])
    contexts = tuple(dict.fromkeys((0, c_best)))
    stage_num = 9 if backbone == "indobertweet" else 10
    stage_grid_dir = RUN_ROOT / f"stage_{stage_num:02d}" / f"e2_{backbone}"
    grid_summary_path = stage_grid_dir / "e2_grid_summary.json"
    grid_metrics_path = stage_grid_dir / "e2_grid_metrics.csv"
    if not grid_summary_path.is_file() or not grid_metrics_path.is_file():
        raise FileNotFoundError(f"Stage {stage_num} grid summary is missing under {stage_grid_dir}")
    grid_summary = _read_json(grid_summary_path)
    if not grid_summary.get("complete") or int(grid_summary.get("n_runs", -1)) != 50 or int(grid_summary.get("expected_runs", -1)) != 50:
        raise ValueError(f"Stage {stage_num} grid summary does not represent a complete 50-run grid")
    if grid_summary.get("dataset_sha256") != dataset_sha or grid_summary.get("fold_assignment_sha256") != fold_assignment_sha:
        raise ValueError(f"Stage {stage_num} grid summary was produced from a different dataset/fold assignment")
    if grid_summary.get("truncation_side") != "left":
        raise ValueError(f"Stage {stage_num} grid summary does not verify context-first left truncation")
    if grid_summary.get("grid_metrics_sha256") != _sha(grid_metrics_path):
        raise ValueError(f"Stage {stage_num} grid metrics hash does not match its summary")
    stage8_manifest_path = RUN_ROOT / "stage_08" / "summary" / eu.RUN_MANIFEST_NAME
    stage8_manifest = _read_json(stage8_manifest_path)
    eu._validate_manifest_integrity(stage8_manifest)
    stage8_fingerprint = str(stage8_manifest["fingerprint"])
    if grid_summary.get("stage8_manifest_fingerprint") != stage8_fingerprint:
        raise ValueError(f"Stage {stage_num} grid summary refers to a different Stage 8 run")
    sequence_doc = eu.load_shared_run_config(
        sequence_length_path=PROJECT_ROOT / eu.SEQUENCE_LENGTH_CONFIG,
        learning_rate_path=PROJECT_ROOT / eu.LEARNING_RATE_CONFIG,
        require_context=True,
        require_context_lr=True,
    )
    max_length_context = int(sequence_doc["max_length_context"])

    prob_by_context: dict[int, np.ndarray] = {}
    fingerprints_by_context: dict[str, list[str]] = {f"C{c}": [] for c in contexts}
    input_paths: list[Path] = [MATCHED_PATH, BACKBONE_DECISION_PATH, STRATEGY_DECISION_PATH,
                               STAGE11_DIR / eu.RUN_MANIFEST_NAME, STAGE11_DIR / "stage11_complete.json",
                               STAGE11_DIR / "artifacts_sha256.json", stage_grid_dir / "e2_grid_summary.json",
                               stage_grid_dir / "e2_grid_metrics.csv", stage8_manifest_path,
                               PROJECT_ROOT / eu.SEQUENCE_LENGTH_CONFIG, PROJECT_ROOT / eu.LEARNING_RATE_CONFIG,
                               STAGE15_DIR / eu.RUN_MANIFEST_NAME, STAGE15_DIR / "stage15_summary.json",
                               STAGE15_DIR / "stage15_complete.json", STAGE15_DIR / "artifacts_sha256.json"]

    # Average probabilities across the two predeclared seeds for each held-out
    # post, after verifying row IDs and gold labels within every fold.
    for condition in contexts:
        pooled = np.full((len(matched), len(eu.LABEL2ID)), np.nan, dtype=float)
        fold_hits = np.zeros(len(matched), dtype=np.int64)
        for fold in FOLDS:
            positions = np.flatnonzero(folds == fold)
            expected_ids = [gold_ids[i] for i in positions]
            expected_y = gold[positions]
            seed_probs = []
            for seed in SEEDS:
                run_dir = stage_grid_dir / f"C{condition}" / f"fold_{fold:02d}" / f"seed_{seed}"
                verified = _verify_leaf_run(
                    run_dir, stage=stage_num, backbone=backbone, context_size=condition,
                    fold=fold, seed=seed, expected_ids=expected_ids, expected_y=expected_y,
                    dataset_sha=dataset_sha, fold_assignment_sha=fold_assignment_sha,
                    max_length_context=max_length_context,
                )
                seed_probs.append(verified["probabilities"])
                fingerprints_by_context[f"C{condition}"].append(verified["manifest_fingerprint"])
                input_paths.extend([run_dir / name for name in (eu.RUN_MANIFEST_NAME, "run_complete.json", "artifacts_sha256.json", "predictions.csv", "probabilities.npy", "metrics.json")])
            mean_probs = np.mean(np.stack(seed_probs, axis=0), axis=0)
            pooled[positions] = mean_probs
            fold_hits[positions] += 1
        if not np.all(fold_hits == 1) or not np.isfinite(pooled).all():
            raise ValueError(f"OOF pooling for C{condition} did not assign exactly one held-out probability row to every post")
        if not np.allclose(pooled.sum(axis=1), 1.0, atol=1e-4):
            raise ValueError(f"Pooled seed-ensemble probabilities do not sum to one for C{condition}")
        prob_by_context[condition] = pooled

    if int(canonical_decision.get("n_oof_examples", -1)) != len(matched):
        raise ValueError("Stage 11 decision has an unexpected OOF sample size")
    expected_id_hash = eu.sha256_json(gold_ids)
    if canonical_decision.get("post_id_sha256") != expected_id_hash:
        raise ValueError("Stage 11 selected context decision is not aligned to the current canonical post-ID ordering")

    manifest_inputs = list(dict.fromkeys(path.resolve() for path in input_paths))
    if any(not path.is_file() for path in manifest_inputs):
        missing = [str(path) for path in manifest_inputs if not path.is_file()]
        raise FileNotFoundError(f"Stage 16 provenance inputs are missing: {missing[:10]}")
    return {
        "matched": matched, "audit": audit, "gold": gold, "gold_ids": gold_ids, "folds": folds,
        "backbone": backbone, "c_best": c_best, "same_condition": c_best == 0,
        "probabilities": prob_by_context, "input_paths": manifest_inputs,
        "stage11_fingerprint": stage11["fingerprint"],
        "stage15_fingerprint": strategy_decision["provenance"]["stage15_manifest_fingerprint"],
        "strategy_decision": strategy_decision, "grid_fingerprints": fingerprints_by_context,
        "review_sample_size": review_sample_size, "review_seed": review_seed,
        "validate_only": validate_only,
    }


# ---------------------------- metric functions ------------------------------
def _classification_summary(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(LABEL_IDS), zero_division=0,
    )
    summary = {
        "n_examples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_recall_fscore_support(y_true, y_pred, labels=list(LABEL_IDS), average="macro", zero_division=0)[0]),
        "macro_recall": float(precision_recall_fscore_support(y_true, y_pred, labels=list(LABEL_IDS), average="macro", zero_division=0)[1]),
        "macro_f1": float(f1_score(y_true, y_pred, labels=list(LABEL_IDS), average="macro", zero_division=0)),
        "weighted_precision": float(precision_recall_fscore_support(y_true, y_pred, labels=list(LABEL_IDS), average="weighted", zero_division=0)[0]),
        "weighted_recall": float(precision_recall_fscore_support(y_true, y_pred, labels=list(LABEL_IDS), average="weighted", zero_division=0)[1]),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=list(LABEL_IDS), average="weighted", zero_division=0)),
    }
    per_class = []
    for i, code in enumerate(LABEL_CODES):
        per_class.append({"class_code": code, "class_name": _label_name(eu.LABEL2ID[code]),
                          "precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i]),
                          "support": int(support[i])})
    return summary, per_class


def _softmax_diagnostics(probs: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    sorted_probs = np.sort(probs, axis=1)
    entropy = -np.sum(np.clip(probs, 1e-12, 1.0) * np.log(np.clip(probs, 1e-12, 1.0)), axis=1)
    rows = []
    for idx, code in enumerate(LABEL_CODES):
        rows.append({"class_code": code, "class_name": _label_name(eu.LABEL2ID[code]),
                     "mean_probability": float(probs[:, idx].mean()), "mean_gold_probability": float(probs[y_true == idx, idx].mean()) if (y_true == idx).any() else np.nan,
                     "mean_confidence": float(probs.max(axis=1).mean()), "mean_margin": float((sorted_probs[:, -1] - sorted_probs[:, -2]).mean()),
                     "mean_entropy": float(entropy.mean()), "support_gold_class": int((y_true == idx).sum()),
                     "accuracy_within_gold_class": float((y_pred[y_true == idx] == y_true[y_true == idx]).mean()) if (y_true == idx).any() else np.nan})
    return pd.DataFrame(rows)


def _outcome_labels(gold: np.ndarray, pred_c0: np.ndarray, pred_best: np.ndarray, same_condition: bool) -> list[str]:
    if same_condition:
        return ["same_condition_correct" if (a == y) else "same_condition_wrong" for y, a in zip(gold, pred_c0)]
    outcomes: list[str] = []
    for y, p0, pb in zip(gold, pred_c0, pred_best):
        c0_ok, cb_ok = p0 == y, pb == y
        if c0_ok and cb_ok:
            outcomes.append("both_correct")
        elif (not c0_ok) and cb_ok:
            outcomes.append("context_fixed_error")
        elif c0_ok and (not cb_ok):
            outcomes.append("context_introduced_error")
        else:
            outcomes.append("both_wrong")
    return outcomes


def _build_case_ledger(upstream: Mapping[str, Any]) -> pd.DataFrame:
    matched = upstream["matched"].copy()
    gold = upstream["gold"]
    ledger = pd.DataFrame({
        "post_id": upstream["gold_ids"],
        "fold": upstream["folds"].astype(int),
        "gold_label": [eu.ID2LABEL[int(y)] for y in gold],
        "gold_label_name": [_label_name(int(y)) for y in gold],
        "selected_backbone": upstream["backbone"],
        "c_best": int(upstream["c_best"]),
        "comparison_available": not bool(upstream["same_condition"]),
    })
    pred_arrays: dict[int, np.ndarray] = {}
    for context, prefix in ((0, "c0"), (int(upstream["c_best"]), "c_best")):
        probs = upstream["probabilities"][context]
        pred = probs.argmax(axis=1)
        pred_arrays[context] = pred
        for class_index, code in enumerate(LABEL_CODES):
            ledger[f"{prefix}_prob_{code}"] = probs[:, class_index]
        ledger[f"{prefix}_predicted_label"] = [eu.ID2LABEL[int(x)] for x in pred]
        ledger[f"{prefix}_predicted_label_name"] = [_label_name(int(x)) for x in pred]
        ledger[f"{prefix}_correct"] = pred == gold
        ledger[f"{prefix}_confidence"] = probs.max(axis=1)
        ledger[f"{prefix}_gold_probability"] = probs[np.arange(len(gold)), gold]
        sorted_probs = np.sort(probs, axis=1)
        ledger[f"{prefix}_top2_margin"] = sorted_probs[:, -1] - sorted_probs[:, -2]
        ledger[f"{prefix}_entropy"] = -np.sum(np.clip(probs, 1e-12, 1.0) * np.log(np.clip(probs, 1e-12, 1.0)), axis=1)
    ledger["comparison_outcome"] = _outcome_labels(gold, pred_arrays[0], pred_arrays[int(upstream["c_best"])], bool(upstream["same_condition"]))
    ledger["c0_error"] = ~ledger["c0_correct"].astype(bool)
    ledger["c_best_error"] = ~ledger["c_best_correct"].astype(bool)
    # Join only post-level/context annotation and text fields, not any modelling
    # feature generated from predictions. The original strings are preserved.
    analysis_cols = [
        eu.CTX_ID_COLUMN, "context_type", "context_needed", "HS", "AL", "target_group",
        "target_group_attribute", "collection_method", "keyword_used", "post_date",
        "context_size_available", "parent_4", "parent_3", "parent_2", "parent_1", "target_text",
    ]
    for col in analysis_cols:
        if col in matched.columns and col != eu.CTX_ID_COLUMN:
            ledger[col] = matched[col].to_numpy()
    if "context_type" not in ledger.columns:
        ledger["context_type"] = "[missing context_type column]"
    ledger["context_type"] = ledger["context_type"].where(ledger["context_type"].notna() & ledger["context_type"].astype(str).str.strip().ne(""), "[missing/unspecified]").astype(str)
    # Keep the exact source label present as an explicit provenance check.
    ledger["source_label"] = matched["label"].astype(str).to_numpy()
    if not np.array_equal(_map_labels(ledger["source_label"], "ledger source_label"), gold):
        raise AssertionError("Ledger's joined source labels do not match the canonical gold labels")
    return ledger


def _metrics_tables(ledger: pd.DataFrame, upstream: Mapping[str, Any]) -> dict[str, pd.DataFrame]:
    gold = upstream["gold"]
    pred0 = upstream["probabilities"][0].argmax(axis=1)
    pred_best = upstream["probabilities"][int(upstream["c_best"])].argmax(axis=1)
    model_rows, per_class_rows, confusion_rows, prob_rows = [], [], [], []
    for name, context, pred, probs in [
        ("C0", 0, pred0, upstream["probabilities"][0]),
        ("C_best", int(upstream["c_best"]), pred_best, upstream["probabilities"][int(upstream["c_best"])]),
    ]:
        summary, per_class = _classification_summary(gold, pred)
        model_rows.append({"evaluation_model": name, "context_size": context, "backbone": upstream["backbone"],
                           "training_strategy": "ITFT (Experiment II-A)", "n_examples": len(gold), **summary})
        per_class_rows.extend({"evaluation_model": name, "context_size": context, "backbone": upstream["backbone"], **row} for row in per_class)
        cm = confusion_matrix(gold, pred, labels=list(LABEL_IDS))
        for i, gold_code in enumerate(LABEL_CODES):
            den = int(cm[i, :].sum())
            for j, pred_code in enumerate(LABEL_CODES):
                confusion_rows.append({"evaluation_model": name, "context_size": context,
                                       "gold_label": gold_code, "gold_label_name": _label_name(eu.LABEL2ID[gold_code]),
                                       "predicted_label": pred_code, "predicted_label_name": _label_name(eu.LABEL2ID[pred_code]),
                                       "count": int(cm[i, j]), "pct_within_gold_label": 100.0 * int(cm[i, j]) / den if den else 0.0,
                                       "gold_label_support": den})
        d = _softmax_diagnostics(probs, gold, pred)
        d.insert(0, "evaluation_model", name)
        d.insert(1, "context_size", context)
        prob_rows.append(d)
    paired_doc = eu.paired_bootstrap_f1_ci(gold, pred0, pred_best, n_bootstrap=N_BOOTSTRAP, ci=CI_LEVEL, random_state=BOOTSTRAP_SEED)
    paired_rows = []
    macro = paired_doc["macro_f1"]
    paired_rows.append({"comparison": "C_best_minus_C0", "metric": "macro_f1", **macro,
                        "n_examples": len(gold), "n_bootstrap": N_BOOTSTRAP, "ci_level": CI_LEVEL,
                        "bootstrap_seed": BOOTSTRAP_SEED,
                        "interpretation_note": "Paired OOF bootstrap conditional on the selected backbone/context; it does not correct selection optimism."})
    for class_code, values in paired_doc["per_class"].items():
        paired_rows.append({"comparison": "C_best_minus_C0", "metric": f"f1_{class_code}", **values,
                            "n_examples": len(gold), "n_bootstrap": N_BOOTSTRAP, "ci_level": CI_LEVEL,
                            "bootstrap_seed": BOOTSTRAP_SEED,
                            "interpretation_note": "Per-class intervals are descriptive; no multiplicity correction."})
    return {
        "evaluation_summary": pd.DataFrame(model_rows),
        "per_class_metrics": pd.DataFrame(per_class_rows),
        "confusion_matrix": pd.DataFrame(confusion_rows),
        "probability_diagnostics": pd.concat(prob_rows, ignore_index=True),
        "paired_bootstrap": pd.DataFrame(paired_rows),
    }


def _outcome_tables(ledger: pd.DataFrame, upstream: Mapping[str, Any]) -> dict[str, pd.DataFrame]:
    denominator = len(ledger)
    same_condition = bool(upstream["same_condition"])
    descriptions = {
        "both_correct": "C0 and C_best both match the gold label.",
        "context_fixed_error": "C0 is wrong and C_best matches the gold label.",
        "context_introduced_error": "C0 matches the gold label and C_best is wrong.",
        "both_wrong": "Neither condition matches the gold label.",
        "same_condition_correct": "C_best is C0; the shared prediction matches the gold label.",
        "same_condition_wrong": "C_best is C0; the shared prediction does not match the gold label.",
    }
    order = ["both_correct", "context_fixed_error", "context_introduced_error", "both_wrong"] if not same_condition else ["same_condition_correct", "same_condition_wrong"]
    outcomes = ledger["comparison_outcome"].value_counts().to_dict()
    outcome_rows = [{"comparison_outcome": outcome, "definition": descriptions[outcome], "count": int(outcomes.get(outcome, 0)),
                     "pct_of_all_oof_cases": 100.0 * int(outcomes.get(outcome, 0)) / denominator if denominator else 0.0,
                     "n_oof_cases": denominator, "context_comparison_available": not same_condition} for outcome in order]
    by_gold_rows = []
    for gold_code in LABEL_CODES:
        subset = ledger[ledger["gold_label"] == gold_code]
        for outcome in order:
            n = int((subset["comparison_outcome"] == outcome).sum())
            by_gold_rows.append({"gold_label": gold_code, "gold_label_name": _label_name(eu.LABEL2ID[gold_code]),
                                 "comparison_outcome": outcome, "count": n, "pct_within_gold_label": 100.0*n/len(subset) if len(subset) else 0.0,
                                 "gold_label_support": int(len(subset))})
    error_direction_rows = []
    for gold_code in LABEL_CODES:
        subset = ledger[ledger["gold_label"] == gold_code]
        n = len(subset)
        error_direction_rows.append({
            "gold_label": gold_code, "gold_label_name": _label_name(eu.LABEL2ID[gold_code]), "support": n,
            "c0_correct_n": int(subset["c0_correct"].sum()), "c0_accuracy_pct": 100.0*float(subset["c0_correct"].mean()) if n else 0.0,
            "c0_error_n": int(subset["c0_error"].sum()), "c0_error_pct": 100.0*float(subset["c0_error"].mean()) if n else 0.0,
            "c_best_correct_n": int(subset["c_best_correct"].sum()), "c_best_accuracy_pct": 100.0*float(subset["c_best_correct"].mean()) if n else 0.0,
            "c_best_error_n": int(subset["c_best_error"].sum()), "c_best_error_pct": 100.0*float(subset["c_best_error"].mean()) if n else 0.0,
            "context_fixed_n": (int((subset["comparison_outcome"] == "context_fixed_error").sum()) if not same_condition else np.nan),
            "context_introduced_n": (int((subset["comparison_outcome"] == "context_introduced_error").sum()) if not same_condition else np.nan),
            "both_wrong_n": (int((subset["comparison_outcome"] == "both_wrong").sum()) if not same_condition else np.nan),
            "comparison_available": not same_condition,
        })
    transition_rows = []
    grouped = ledger.groupby(["gold_label", "c0_predicted_label", "c_best_predicted_label"], dropna=False, sort=True)
    gold_supports = ledger["gold_label"].value_counts().to_dict()
    for (gold_code, p0, pb), frame in grouped:
        support = int(gold_supports.get(gold_code, 0))
        transition_rows.append({"gold_label": gold_code, "gold_label_name": _label_name(eu.LABEL2ID[gold_code]),
                                "c0_predicted_label": p0, "c_best_predicted_label": pb,
                                "transition_n": len(frame), "pct_within_gold_label": 100.0*len(frame)/support if support else 0.0,
                                "gold_label_support": support,
                                "c0_correct_n": int(frame["c0_correct"].sum()), "c_best_correct_n": int(frame["c_best_correct"].sum()),
                                "outcome_types": ";".join(sorted(frame["comparison_outcome"].unique().tolist()))})
    return {
        "outcome_summary": pd.DataFrame(outcome_rows),
        "outcome_by_gold_label": pd.DataFrame(by_gold_rows),
        "error_profile_by_gold_label": pd.DataFrame(error_direction_rows),
        "prediction_transitions": pd.DataFrame(transition_rows),
    }


def _context_type_tables(ledger: pd.DataFrame, upstream: Mapping[str, Any]) -> dict[str, pd.DataFrame]:
    same_condition = bool(upstream["same_condition"])
    outcome_order = ["both_correct", "context_fixed_error", "context_introduced_error", "both_wrong"] if not same_condition else ["same_condition_correct", "same_condition_wrong"]
    rows = []
    detailed_rows = []
    total = len(ledger)
    for context_type, subset in ledger.groupby("context_type", dropna=False, sort=True):
        n = len(subset)
        row: dict[str, Any] = {"context_type": context_type, "n_examples": n, "pct_of_900_oof_cases": 100.0*n/total if total else 0.0,
                               "c0_correct_n": int(subset["c0_correct"].sum()), "c0_accuracy_pct": 100.0*float(subset["c0_correct"].mean()) if n else 0.0,
                               "c0_error_n": int(subset["c0_error"].sum()), "c0_error_pct": 100.0*float(subset["c0_error"].mean()) if n else 0.0,
                               "c_best_correct_n": int(subset["c_best_correct"].sum()), "c_best_accuracy_pct": 100.0*float(subset["c_best_correct"].mean()) if n else 0.0,
                               "c_best_error_n": int(subset["c_best_error"].sum()), "c_best_error_pct": 100.0*float(subset["c_best_error"].mean()) if n else 0.0,
                               "comparison_available": not same_condition}
        for code in LABEL_CODES:
            row[f"gold_{code}_n"] = int((subset["gold_label"] == code).sum())
        for outcome in outcome_order:
            count = int((subset["comparison_outcome"] == outcome).sum())
            row[f"{outcome}_n"] = count
            row[f"{outcome}_pct_within_context_type"] = 100.0*count/n if n else 0.0
        rows.append(row)
        for gold_code, sub_gold in subset.groupby("gold_label", dropna=False, sort=True):
            denom = len(sub_gold)
            record = {"context_type": context_type, "gold_label": gold_code,
                      "gold_label_name": _label_name(eu.LABEL2ID[gold_code]), "n_examples": denom,
                      "pct_of_context_type": 100.0*denom/n if n else 0.0,
                      "c0_correct_n": int(sub_gold["c0_correct"].sum()),
                      "c0_accuracy_pct_within_type_and_gold": 100.0*float(sub_gold["c0_correct"].mean()) if denom else 0.0,
                      "c_best_correct_n": int(sub_gold["c_best_correct"].sum()),
                      "c_best_accuracy_pct_within_type_and_gold": 100.0*float(sub_gold["c_best_correct"].mean()) if denom else 0.0,
                      "context_comparison_available": not same_condition}
            for outcome in outcome_order:
                count_gold = int((sub_gold["comparison_outcome"] == outcome).sum())
                record[f"{outcome}_n"] = count_gold
                record[f"{outcome}_pct_within_type_and_gold"] = 100.0*count_gold/denom if denom else 0.0
            detailed_rows.append(record)
    return {"context_type_crosstab": pd.DataFrame(rows), "context_type_by_gold": pd.DataFrame(detailed_rows)}


# ------------------------- reproducible review sample -----------------------
def _allocate_capped(total: int, counts: Mapping[Any, int], *, minimum_one: bool, sqrt_weight: bool = True) -> dict[Any, int]:
    """Integer allocation with capacity caps and deterministic largest remainders."""
    keys = [k for k, n in counts.items() if int(n) > 0]
    result = {k: 0 for k in counts}
    if total <= 0 or not keys:
        return result
    total = min(int(total), sum(int(counts[k]) for k in keys))
    if minimum_one and total >= len(keys):
        for key in keys:
            result[key] = 1
        total -= len(keys)
    while total > 0:
        active = [key for key in keys if result[key] < int(counts[key])]
        if not active:
            break
        weights = {key: (math.sqrt(int(counts[key])) if sqrt_weight else float(counts[key])) for key in active}
        weight_sum = sum(weights.values())
        raw = {key: total * weights[key] / weight_sum for key in active}
        floors = {key: min(int(counts[key]) - result[key], int(math.floor(raw[key]))) for key in active}
        used = sum(floors.values())
        for key, amount in floors.items():
            result[key] += amount
        total -= used
        if total <= 0:
            break
        ranked = sorted(active, key=lambda key: (raw[key] - math.floor(raw[key]), weights[key], str(key)), reverse=True)
        progressed = False
        for key in ranked:
            if total <= 0:
                break
            if result[key] < int(counts[key]):
                result[key] += 1
                total -= 1
                progressed = True
        if not progressed:
            break
    return result


def _select_review_sample(ledger: pd.DataFrame, target_n: int, seed: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    if target_n < 1:
        raise ValueError("--target-review-sample-size must be >= 1")
    effective_target = min(int(target_n), len(ledger))
    rng = np.random.default_rng(int(seed))
    outcome_counts = ledger["comparison_outcome"].value_counts().to_dict()
    outcome_alloc = _allocate_capped(effective_target, outcome_counts, minimum_one=True, sqrt_weight=True)
    chosen_parts: list[pd.DataFrame] = []
    stratum_allocations: dict[str, int] = {}
    for outcome in sorted(outcome_alloc, key=str):
        quota = int(outcome_alloc[outcome])
        if quota <= 0:
            continue
        outcome_frame = ledger[ledger["comparison_outcome"] == outcome]
        context_counts = outcome_frame["context_type"].value_counts().to_dict()
        type_alloc = _allocate_capped(quota, context_counts, minimum_one=True, sqrt_weight=True)
        for context_type in sorted(type_alloc, key=str):
            n_take = int(type_alloc[context_type])
            if n_take <= 0:
                continue
            cell = outcome_frame[outcome_frame["context_type"] == context_type]
            selected_indices = rng.choice(cell.index.to_numpy(), size=n_take, replace=False)
            sampled = ledger.loc[selected_indices].copy()
            sampled["sampling_stratum"] = f"{outcome} | {context_type}"
            sampled["stratum_population_n"] = int(len(cell))
            sampled["stratum_selected_n"] = int(n_take)
            stratum_allocations[f"{outcome} | {context_type}"] = int(n_take)
            chosen_parts.append(sampled)
    if not chosen_parts:
        return pd.DataFrame(columns=list(ledger.columns) + ["sampling_stratum", "stratum_population_n", "stratum_selected_n"]), {
            "requested_n": int(target_n), "effective_n": 0, "seed": int(seed), "outcome_allocation": outcome_alloc,
            "stratum_allocation": stratum_allocations, "method": "two-stage square-root allocation by outcome then context_type; minimum one per nonempty group when quota permits",
        }
    sample = pd.concat(chosen_parts, ignore_index=True)
    if sample["post_id"].astype(str).duplicated().any():
        raise AssertionError("Manual review sampling selected a duplicate post_id")
    sample = sample.sort_values(["comparison_outcome", "context_type", "post_id"], kind="mergesort").reset_index(drop=True)
    # Human completion columns are intentionally blank; do not infer the mechanism.
    sample["primary_error_mechanism"] = ""
    sample["secondary_error_mechanism"] = ""
    sample["reviewer_confidence"] = ""
    sample["review_notes"] = ""
    sample["adjudication_status"] = "not_reviewed"
    sample["reviewer_conclusion"] = ""
    details = {
        "requested_n": int(target_n), "effective_n": int(len(sample)), "population_n": int(len(ledger)), "seed": int(seed),
        "method": "two-stage square-root allocation by prediction outcome then context_type; at least one per nonempty group when quota permits; samples without replacement within strata",
        "outcome_population_counts": {str(k): int(v) for k, v in outcome_counts.items()},
        "outcome_sample_allocation": {str(k): int(v) for k, v in outcome_alloc.items()},
        "outcome_x_context_type_sample_allocation": stratum_allocations,
        "prevalence_warning": "The manually selected sample is stratified for error-mechanism coverage and is not a simple random sample; do not interpret its proportions as population prevalence.",
    }
    return sample, details


def _make_codebook(ledger: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for context_type in sorted(ledger["context_type"].dropna().astype(str).unique().tolist()):
        rows.append({"code_family": "dataset_context_type", "code": context_type,
                     "label": context_type,
                     "definition": "Use the existing dataset annotation guide for this category; the script preserves the source category verbatim and does not infer a definition.",
                     "include_when": "Apply the existing annotation's meaning as defined in the project's annotation guide.",
                     "exclude_when": "Do not recode this source field during model-error-mechanism review.",
                     "annotation_status": "pre-existing dataset annotation; not reviewer-inferred"})
    for item in ERROR_MECHANISMS:
        rows.append({"code_family": "candidate_error_mechanism", **item,
                     "annotation_status": "human reviewer assigns only when supported by the target and available parent turns"})
    for item in REVIEW_ONLY_CODES:
        rows.append({"code_family": "review_status", **item, "include_when": "Use when the stated condition applies.", "exclude_when": "Use a specific mechanism when the evidence supports it.",
                     "annotation_status": "reviewer-selected fallback"})
    return pd.DataFrame(rows)


def _make_readme(upstream: Mapping[str, Any], sample_details: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "stage": STAGE, "run_id": eu.RUN_ID, "selected_backbone": upstream["backbone"],
        "c_best": int(upstream["c_best"]), "comparison": "Experiment II-A / ITFT, same selected backbone, C0 versus C_best",
        "comparison_available": not bool(upstream["same_condition"]),
        "n_oof_examples": EXPECTED_ROWS,
        "oof_probability_rule": "Within each held-out fold and condition, average the two seed probability vectors per post; predictions are argmax of the seed-mean probabilities.",
        "gold_label_source": "data/processed/ctx_matched_subset.csv label; every stored y_true was checked against the fold-held-out source labels.",
        "primary_metric": "macro-F1; accuracy, macro/weighted precision-recall-F1, per-class precision/recall/F1, confusion counts, probabilities and paired bootstrap are supplementary.",
        "paired_bootstrap": {"n_bootstrap": N_BOOTSTRAP, "ci_level": CI_LEVEL, "seed": BOOTSTRAP_SEED, "contrast": "C_best minus C0 in macro-F1; conditional on the selected context/backbone and therefore not a selection-optimism correction."},
        "outcome_definitions": {
            "both_correct": "C0 and C_best predict the gold label.",
            "context_fixed_error": "C0 is wrong and C_best predicts the gold label.",
            "context_introduced_error": "C0 predicts the gold label and C_best is wrong.",
            "both_wrong": "Both conditions are wrong.",
        } if not upstream["same_condition"] else {
            "same_condition_correct": "C_best resolved to C0, and the shared prediction is correct.",
            "same_condition_wrong": "C_best resolved to C0, and the shared prediction is wrong.",
            "note": "No distinct context-versus-target-only comparison is available when C_best=0."},
        "manual_review_sampling": dict(sample_details),
        "manual_review_warning": "The manual review sample is outcome/context-stratified, drawn from the full 900 OOF cases. It supports qualitative mechanism analysis; it does not estimate population prevalence.",
        "context_type_cross_tab_denominators": "Primary context-type table reports outcome percentages within each context_type; supplementary context_type x gold_label table reports percentages within each context_type/gold-label cell. Every percentage field names its denominator.",
        "source_provenance": {"stage11_manifest_fingerprint": upstream["stage11_fingerprint"], "stage15_manifest_fingerprint": upstream["stage15_fingerprint"],
                              "grid_run_fingerprints_by_context": upstream["grid_fingerprints"]},
        "human_review_codes": [item["code"] for item in ERROR_MECHANISMS] + [item["code"] for item in REVIEW_ONLY_CODES],
        "limitation": "C0/C_best are selected using the same OOF results used for this analysis. The paired bootstrap describes conditional OOF uncertainty; it does not remove selection optimism. Qualitative mechanism labels require human review.",
    }


def _create_figures(ledger: pd.DataFrame, out_dir: Path, *, same_condition: bool) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warning] matplotlib not installed; skipping Stage 16 PNG figures")
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    files = []
    outcome_order = ["both_correct", "context_fixed_error", "context_introduced_error", "both_wrong"] if not same_condition else ["same_condition_correct", "same_condition_wrong"]
    counts = ledger["comparison_outcome"].value_counts().reindex(outcome_order, fill_value=0)
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar([x.replace("_", "\n") for x in counts.index], counts.values)
    ax.set_ylabel("OOF cases")
    ax.set_title("Seed-ensembled OOF outcome distribution")
    ax.grid(axis="y", alpha=0.25)
    for bar, count in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{int(count)}\n({100*count/len(ledger):.1f}%)", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    filename = "stage16_outcome_distribution.png"
    fig.savefig(out_dir / filename, dpi=180)
    plt.close(fig)
    files.append(filename)

    context_summary = ledger.groupby("context_type", dropna=False).agg(
        n_examples=("post_id", "size"), c0_accuracy=("c0_correct", "mean"), c_best_accuracy=("c_best_correct", "mean")
    ).reset_index().sort_values("n_examples", ascending=True)
    fig, ax = plt.subplots(figsize=(10, max(4.5, 0.48*len(context_summary))))
    yloc = np.arange(len(context_summary))
    labels = [str(x) for x in context_summary["context_type"]]
    ax.barh(yloc - 0.18, context_summary["c0_accuracy"]*100, height=0.34, label="C0")
    ax.barh(yloc + 0.18, context_summary["c_best_accuracy"]*100, height=0.34, label=f"C{int(ledger['c_best'].iloc[0])}")
    ax.set_yticks(yloc, labels)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Accuracy within context type (%)")
    ax.set_title("OOF accuracy by annotated context type")
    ax.legend()
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    filename = "stage16_accuracy_by_context_type.png"
    fig.savefig(out_dir / filename, dpi=180)
    plt.close(fig)
    files.append(filename)
    return files


def _expected_output_names() -> tuple[str, ...]:
    return (
        "stage16_all_oof_cases.csv", "stage16_evaluation_summary.csv", "stage16_per_class_metrics.csv",
        "stage16_confusion_matrix.csv", "stage16_probability_diagnostics.csv", "stage16_paired_bootstrap.csv",
        "stage16_outcome_summary.csv", "stage16_outcome_by_gold_label.csv", "stage16_error_profile_by_gold_label.csv",
        "stage16_prediction_transitions.csv", "stage16_context_type_crosstab.csv", "stage16_context_type_by_gold_label.csv",
        "stage16_manual_review_sample_template.csv", "stage16_codebook.csv", "stage16_summary.json",
    )


def _make_stage_manifest(upstream: Mapping[str, Any], outputs: Mapping[str, pd.DataFrame], sample_details: Mapping[str, Any], figures: Sequence[str]) -> dict[str, Any]:
    config = {
        "schema_version": eu.CONFIG_SCHEMA_VERSION, "run_id": eu.RUN_ID,
        "stage": STAGE, "script_name": SCRIPT_NAME,
        "selected_backbone": upstream["backbone"], "selected_c_best": int(upstream["c_best"]),
        "comparison": "Experiment II-A ITFT C0 versus C_best; one common backbone and paired OOF posts",
        "primary_metric": "macro_f1", "paired_bootstrap": {"n_bootstrap": N_BOOTSTRAP, "ci_level": CI_LEVEL, "random_state": BOOTSTRAP_SEED},
        "review_sample": {"target_n": int(upstream["review_sample_size"]), "seed": int(upstream["review_seed"]), "allocation": sample_details},
        "output_csvs": sorted(outputs), "figure_files": list(figures),
    }
    metadata = {
        "purpose": "Stage 16 gold-label-aware OOF error analysis and reproducible qualitative review sampling",
        "stage11_manifest_fingerprint": upstream["stage11_fingerprint"],
        "stage15_manifest_fingerprint": upstream["stage15_fingerprint"],
        "n_oof_examples": EXPECTED_ROWS,
        "primary_comparison": "same selected backbone, Experiment II-A ITFT, C0 versus C_best",
        "source_grid_manifest_fingerprints_by_context": upstream["grid_fingerprints"],
        "manual_review_sample_is_prevalence_sample": False,
    }
    return eu.make_run_manifest(stage=STAGE, script_name=SCRIPT_NAME, config=config,
                                input_paths=upstream["input_paths"], metadata=metadata, project_root=PROJECT_ROOT)


def _publish(upstream: Mapping[str, Any], tables: Mapping[str, pd.DataFrame], sample: pd.DataFrame,
             sample_details: Mapping[str, Any], summary: Mapping[str, Any], figures: Sequence[str],
             figure_source_dir: Path, manifest: Mapping[str, Any]) -> None:
    # Never overwrite a prior Stage 16 bundle/export/worksheet. Archive old
    # results first so run lineage is unambiguous.
    if STAGE_ROOT.exists():
        raise FileExistsError(f"Stage 16 bundle already exists: {STAGE_ROOT}. Archive it before creating a fresh run.")
    if REVIEW_WORKSHEET_PATH.exists():
        raise FileExistsError(f"Editable human-review worksheet already exists: {REVIEW_WORKSHEET_PATH}. Archive it before publishing a new sample.")
    preexisting_stage16_tables = [TABLES_DIR / name for name in (*tables.keys(), "stage16_summary.json", "stage16_export_manifest.json") if (TABLES_DIR / name).exists()]
    preexisting_stage16_figures = [FIGURES_DIR / name for name in figures if (FIGURES_DIR / name).exists()]
    if preexisting_stage16_tables or preexisting_stage16_figures:
        raise FileExistsError("Stage 16 export filenames already exist; archive these files before publishing: " + ", ".join(map(str, preexisting_stage16_tables + preexisting_stage16_figures)))

    eu.prepare_run_directory(STAGE_ROOT, manifest, allow_resume=False)
    for filename, frame in tables.items():
        _write_csv(STAGE_ROOT / filename, frame)
    _write_json(STAGE_ROOT / "stage16_summary.json", dict(summary))
    fig_dir = STAGE_ROOT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for name in figures:
        source = figure_source_dir / name
        if not source.is_file():
            raise FileNotFoundError(f"Expected figure was not generated: {source}")
        shutil.copy2(source, fig_dir / name)

    # Immutable template is part of the provenance bundle. An editable copy is
    # exported to a separate directory so reviewer edits do not alter the bundle.
    _write_csv(STAGE_ROOT / "stage16_manual_review_sample_template.csv", sample)
    codebook_path = STAGE_ROOT / "stage16_codebook.csv"
    if not codebook_path.is_file():
        raise AssertionError("Codebook is missing from the output table mapping")

    index_hashes = {}
    for path in sorted(STAGE_ROOT.rglob("*")):
        if not path.is_file() or path.name in {"artifacts_sha256.json", "stage16_complete.json"}:
            continue
        index_hashes[path.relative_to(STAGE_ROOT).as_posix()] = _sha(path)
    index_path = STAGE_ROOT / "artifacts_sha256.json"
    _write_json(index_path, {"schema_version": eu.CONFIG_SCHEMA_VERSION, "run_id": eu.RUN_ID,
                             "stage": STAGE, "script_name": SCRIPT_NAME,
                             "manifest_fingerprint": manifest["fingerprint"], "sha256": index_hashes})
    _write_json(STAGE_ROOT / "stage16_complete.json", {"schema_version": eu.CONFIG_SCHEMA_VERSION, "run_id": eu.RUN_ID,
        "stage": STAGE, "script_name": SCRIPT_NAME, "manifest_fingerprint": manifest["fingerprint"],
        "artifact_index_sha256": _sha(index_path), "selected_backbone": upstream["backbone"],
        "selected_c_best": int(upstream["c_best"]), "n_oof_examples": len(tables["stage16_all_oof_cases.csv"]),
        "manual_review_sample_n": len(sample), "figure_count": len(figures)})
    eu.assert_current_run_artifact(STAGE_ROOT / "stage16_complete.json", expected_stage=STAGE,
                                   expected_script=SCRIPT_NAME, expected_fingerprint=str(manifest["fingerprint"]))

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for filename, frame in tables.items():
        shutil.copy2(STAGE_ROOT / filename, TABLES_DIR / filename)
    shutil.copy2(STAGE_ROOT / "stage16_summary.json", TABLES_DIR / "stage16_summary.json")
    for name in figures:
        shutil.copy2(STAGE_ROOT / "figures" / name, FIGURES_DIR / name)
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(STAGE_ROOT / "stage16_manual_review_sample_template.csv", REVIEW_WORKSHEET_PATH)
    _write_json(TABLES_DIR / "stage16_export_manifest.json", {
        "schema_version": eu.CONFIG_SCHEMA_VERSION, "run_id": eu.RUN_ID, "stage": STAGE,
        "script_name": SCRIPT_NAME, "stage16_manifest_fingerprint": manifest["fingerprint"],
        "files_sha256": {p.name: _sha(p) for p in sorted(TABLES_DIR.glob("stage16_*")) if p.is_file() and p.name != "stage16_export_manifest.json"},
    })


def _summarize_review_annotations(path: Path, *, project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Completed manual-review worksheet not found: {path}")
    # The base Stage 16 bundle is immutable. Reviewer input is a separate copy;
    # verify the original bundle and ensure reviewed rows/labels/outcomes are
    # exactly those selected by the recorded, deterministic sampling procedure.
    _verify_stage_bundle(STAGE_ROOT, stage=STAGE, script_name=SCRIPT_NAME, marker_name="stage16_complete.json")
    template_path = STAGE_ROOT / "stage16_manual_review_sample_template.csv"
    if not template_path.is_file():
        raise FileNotFoundError(f"Immutable Stage 16 review-sample template is missing: {template_path}")
    template = pd.read_csv(template_path, keep_default_na=False)
    review = pd.read_csv(path, keep_default_na=False)
    if set(review["post_id"].astype(str)) != set(template["post_id"].astype(str)) or len(review) != len(template):
        raise ValueError("Review worksheet post_id set differs from the immutable Stage 16 sample template; do not add or remove cases.")
    immutable_fields = ["post_id", "gold_label", "comparison_outcome", "context_type", "c0_predicted_label", "c_best_predicted_label"]
    template_by_id = template.set_index(template["post_id"].astype(str))
    review_by_id = review.set_index(review["post_id"].astype(str))
    for field in immutable_fields:
        if field in template_by_id.columns and field in review_by_id.columns:
            left = template_by_id[field].astype(str).sort_index()
            right = review_by_id[field].astype(str).sort_index()
            if not left.equals(right):
                raise ValueError(f"Review worksheet changed the immutable {field} values; restore them from the original sample template.")
    required = {"post_id", "comparison_outcome", "context_type", "gold_label", "primary_error_mechanism",
                "secondary_error_mechanism", "reviewer_confidence", "review_notes", "adjudication_status"}
    missing = required - set(review.columns)
    if missing:
        raise KeyError(f"Review worksheet lacks columns required for finalization: {sorted(missing)}")
    if review["post_id"].astype(str).duplicated().any():
        raise ValueError("Manual-review worksheet contains duplicate post_id values")
    primary = review["primary_error_mechanism"].astype(str).str.strip()
    secondary = review["secondary_error_mechanism"].astype(str).str.strip()
    status = review["adjudication_status"].astype(str).str.strip().str.casefold()
    reviewed_status = status.isin({"reviewed", "adjudicated", "complete", "completed"})
    has_mechanism = primary.ne("")
    invalid_codes = sorted(set(primary[has_mechanism]) - VALID_REVIEW_CODES)
    if invalid_codes:
        raise ValueError(f"Unknown primary_error_mechanism code(s): {invalid_codes}. See stage16_codebook.csv.")
    invalid_secondary = sorted(set(secondary[secondary.ne("")]) - VALID_REVIEW_CODES)
    if invalid_secondary:
        raise ValueError(f"Unknown secondary_error_mechanism code(s): {invalid_secondary}. See stage16_codebook.csv.")
    if not reviewed_status.any():
        raise ValueError("No completed annotations detected. Set adjudication_status to 'reviewed' for reviewed rows.")

    error_outcomes = {"context_fixed_error", "context_introduced_error", "both_wrong", "same_condition_wrong"}
    control_outcomes = {"both_correct", "same_condition_correct"}
    is_error = review["comparison_outcome"].astype(str).isin(error_outcomes)
    is_control = review["comparison_outcome"].astype(str).isin(control_outcomes)
    incomplete_error = reviewed_status & is_error & ~has_mechanism
    if incomplete_error.any():
        ids = review.loc[incomplete_error, "post_id"].astype(str).tolist()[:10]
        raise ValueError(f"Reviewed error cases require a primary mechanism code; missing for post_id(s): {ids}")
    wrongly_coded_control = reviewed_status & is_control & has_mechanism & primary.ne("NOT_APPLICABLE_NO_ERROR")
    if wrongly_coded_control.any():
        ids = review.loc[wrongly_coded_control, "post_id"].astype(str).tolist()[:10]
        raise ValueError(f"Correct-control cases must have a blank primary code or NOT_APPLICABLE_NO_ERROR; invalid coding for post_id(s): {ids}")

    valid_review = review[reviewed_status].copy()
    valid_review["primary_error_mechanism"] = valid_review["primary_error_mechanism"].astype(str).str.strip()
    mechanism_review = valid_review[
        valid_review["comparison_outcome"].astype(str).isin(error_outcomes)
        & valid_review["primary_error_mechanism"].ne("")
        & valid_review["primary_error_mechanism"].ne("NOT_APPLICABLE_NO_ERROR")
    ].copy()
    if mechanism_review.empty:
        raise ValueError("No completed, mechanism-coded error cases detected. Correct controls alone cannot produce an error-mechanism summary.")

    total_mechanism_reviewed = len(mechanism_review)
    mech_counts = mechanism_review["primary_error_mechanism"].value_counts().to_dict()
    summary_rows = []
    mechanism_codes = [item["code"] for item in ERROR_MECHANISMS] + ["OTHER", "UNCERTAIN_NEEDS_ADJUDICATION"]
    label_by_code = {item["code"]: item["label"] for item in ERROR_MECHANISMS + REVIEW_ONLY_CODES}
    for code in mechanism_codes:
        n = int(mech_counts.get(code, 0))
        summary_rows.append({"primary_error_mechanism": code, "label": label_by_code[code],
                             "mechanism_coded_error_cases_n": n, "pct_of_mechanism_coded_error_cases": 100.0*n/total_mechanism_reviewed if total_mechanism_reviewed else 0.0,
                             "denominator_mechanism_coded_error_cases": total_mechanism_reviewed,
                             "interpretation_warning": "Descriptive distribution among completed, stratified review error cases; not population prevalence."})
    detail_rows = []
    grouping = mechanism_review.groupby(["comparison_outcome", "context_type", "gold_label", "primary_error_mechanism"], dropna=False, sort=True)
    for keys, frame in grouping:
        outcome, context_type, gold_label, mechanism = keys
        detail_rows.append({"comparison_outcome": outcome, "context_type": context_type, "gold_label": gold_label,
                            "primary_error_mechanism": mechanism, "reviewed_cases_n": len(frame),
                            "post_ids": ";".join(frame["post_id"].astype(str).tolist())})

    coverage_rows = []
    for outcome, group in review.groupby("comparison_outcome", dropna=False, sort=True):
        is_reviewed = reviewed_status.loc[group.index]
        coded_error = is_reviewed & group["comparison_outcome"].astype(str).isin(error_outcomes) & primary.loc[group.index].ne("") & primary.loc[group.index].ne("NOT_APPLICABLE_NO_ERROR")
        coverage_rows.append({"comparison_outcome": outcome, "sample_cases_n": len(group), "reviewed_cases_n": int(is_reviewed.sum()),
                              "reviewed_pct_within_outcome": 100.0*float(is_reviewed.mean()) if len(group) else 0.0,
                              "mechanism_coded_error_cases_n": int(coded_error.sum()),
                              "not_yet_reviewed_n": int((~is_reviewed).sum()),
                              "interpretation_note": "Coverage of the stratified manual-review sample; not population prevalence."})

    outputs_dir = REVIEW_DIR
    summary_path = outputs_dir / "stage16_error_mechanism_summary.csv"
    detail_path = outputs_dir / "stage16_error_mechanisms_by_outcome_context_gold.csv"
    coverage_path = outputs_dir / "stage16_review_coverage_summary.csv"
    _write_csv(summary_path, pd.DataFrame(summary_rows))
    _write_csv(detail_path, pd.DataFrame(detail_rows, columns=["comparison_outcome", "context_type", "gold_label", "primary_error_mechanism", "reviewed_cases_n", "post_ids"]))
    _write_csv(coverage_path, pd.DataFrame(coverage_rows))
    # Preserve representative *error* cases only; correct predictions are
    # controls and cannot be presented as examples of an error mechanism.
    representative = mechanism_review.sort_values(
        ["comparison_outcome", "context_type", "gold_label", "primary_error_mechanism", "post_id"], kind="mergesort"
    ).groupby(["comparison_outcome", "context_type", "gold_label", "primary_error_mechanism"], dropna=False, sort=True).head(1)
    representative_path = outputs_dir / "stage16_representative_reviewed_cases.csv"
    _write_csv(representative_path, representative)

    # Duplicate finalized review tables into the thesis-table export folder;
    # they remain separate from the immutable Stage 16 base-run bundle.
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    exported_review_paths = []
    for source in (summary_path, detail_path, coverage_path, representative_path):
        destination = TABLES_DIR / source.name
        shutil.copy2(source, destination)
        exported_review_paths.append(destination)
    _write_json(TABLES_DIR / "stage16_review_export_manifest.json", {
        "schema_version": eu.CONFIG_SCHEMA_VERSION, "run_id": eu.RUN_ID, "stage": "16-review",
        "source_stage16_manifest_fingerprint": _read_json(STAGE_ROOT / eu.RUN_MANIFEST_NAME).get("fingerprint"),
        "worksheet_sha256": _sha(path),
        "files_sha256": {file.name: _sha(file) for file in exported_review_paths},
    })
    review_manifest = {
        "schema_version": eu.CONFIG_SCHEMA_VERSION, "run_id": eu.RUN_ID, "stage": "16-review",
        "source_stage16_manifest_fingerprint": _read_json(STAGE_ROOT / eu.RUN_MANIFEST_NAME).get("fingerprint") if (STAGE_ROOT / eu.RUN_MANIFEST_NAME).is_file() else None,
        "worksheet_path": str(path.resolve()), "worksheet_sha256": _sha(path),
        "completed_reviewed_cases_n": int(valid_review.shape[0]),
        "mechanism_coded_error_cases_n": int(total_mechanism_reviewed),
        "outputs_sha256": {p.name: _sha(p) for p in (summary_path, detail_path, coverage_path, representative_path)},
        "warning": "Manual mechanism findings apply to the reviewed stratified error sample only; do not present mechanism percentages as population prevalence.",
    }
    _write_json(outputs_dir / "stage16_review_manifest.json", review_manifest)
    return review_manifest

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 16: prediction-driven OOF error analysis.")
    parser.add_argument("--validate-only", action="store_true", help="Audit Stage 11/15 and C0/C_best OOF artifacts without writing Stage 16 outputs.")
    parser.add_argument("--target-review-sample-size", type=int, default=DEFAULT_REVIEW_SAMPLE_SIZE,
                        help=f"Target number of cases for manual review (default {DEFAULT_REVIEW_SAMPLE_SIZE}); capped by available cases and adaptively allocated.")
    parser.add_argument("--review-seed", type=int, default=DEFAULT_REVIEW_SEED, help="Fixed random seed for reproducible stratified review sampling.")
    parser.add_argument("--finalize-review", action="store_true", help="Summarize human annotations in results/manual_review/stage16_manual_review_sample.csv; do not rerun Stage 16.")
    parser.add_argument("--review-file", type=Path, default=None, help="Optional completed worksheet path for --finalize-review.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    os.chdir(PROJECT_ROOT)
    if args.finalize_review:
        review_path = args.review_file or REVIEW_WORKSHEET_PATH
        manifest = _summarize_review_annotations(review_path)
        print(f"Finalized manual review: {manifest['completed_reviewed_cases_n']} completed cases")
        print(f"Mechanism summary: {REVIEW_DIR / 'stage16_error_mechanism_summary.csv'}")
        print(f"Grouped details: {REVIEW_DIR / 'stage16_error_mechanisms_by_outcome_context_gold.csv'}")
        print(f"Representative reviewed cases: {REVIEW_DIR / 'stage16_representative_reviewed_cases.csv'}")
        print(f"Review coverage summary: {REVIEW_DIR / 'stage16_review_coverage_summary.csv'}")
        return 0

    if args.target_review_sample_size < 1:
        raise ValueError("--target-review-sample-size must be >= 1")
    upstream = _load_upstream(review_sample_size=args.target_review_sample_size, review_seed=args.review_seed, validate_only=args.validate_only)
    ledger = _build_case_ledger(upstream)
    sample, sample_details = _select_review_sample(ledger, args.target_review_sample_size, args.review_seed)
    tables = _metrics_tables(ledger, upstream)
    tables.update(_outcome_tables(ledger, upstream))
    tables.update(_context_type_tables(ledger, upstream))
    tables["stage16_all_oof_cases.csv"] = ledger
    tables["stage16_manual_review_sample_template.csv"] = sample
    tables["stage16_codebook.csv"] = _make_codebook(ledger)
    summary = _make_readme(upstream, sample_details)

    if args.validate_only:
        print("Stage 16 validation passed.")
        print(f"Selected backbone={upstream['backbone']}; C_best=C{upstream['c_best']}; OOF cases={len(ledger)}")
        print(f"Review target={args.target_review_sample_size}; effective sample={len(sample)}; seed={args.review_seed}")
        print("No files were written.")
        return 0

    figure_staging_dir = RUN_ROOT / ".stage16_figures_staging"
    if figure_staging_dir.exists():
        raise FileExistsError(f"Staging directory already exists: {figure_staging_dir}. Inspect/archive it before rerunning.")
    figures = _create_figures(ledger, figure_staging_dir, same_condition=bool(upstream["same_condition"]))
    # make_run_manifest only accepts the registered Stage 16 script and hashes
    # all source documents and inputs; output tables are hashed by the index.
    try:
        manifest = _make_stage_manifest(upstream, tables, sample_details, figures)
        _publish(upstream, tables, sample, sample_details, summary, figures, figure_staging_dir, manifest)
    finally:
        if figure_staging_dir.exists():
            shutil.rmtree(figure_staging_dir)
    print(f"Stage 16 completed. Authoritative bundle: {STAGE_ROOT}")
    print(f"Full OOF case ledger: {TABLES_DIR / 'stage16_all_oof_cases.csv'} ({len(ledger)} rows)")
    print(f"Thesis context-type crosstab: {TABLES_DIR / 'stage16_context_type_crosstab.csv'}")
    print(f"Editable manual review sample: {REVIEW_WORKSHEET_PATH} ({len(sample)} rows)")
    print("After review, run: python scripts/16_error_analysis.py --finalize-review")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
