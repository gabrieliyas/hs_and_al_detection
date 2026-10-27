"""Stage 8: contextual length selection, contextual LR tuning, and timing pilot.

Authoritative pipeline script: ``scripts/8_timing_pilot.py``.

This one script handles both registered backbones (IndoBERTweet and
XLM-RoBERTa-base) under the same pre-declared procedure. Run it on a GPU
runtime after Stages 1, 1b, 2, the benchmark LR preflight, and both ITFT runs
(Stages 6 and 7) are complete.

Responsibilities:
  1. Select a shared ``max_length_context`` from C4 examples in the 80%
     training partition carved from fold 0's CV training portion (rows whose
     ``fold != 0``). Both pinned tokenizers are used, without truncation, and
     the utility applies p95 -> round up to a multiple of 8 -> cap at 512.
  2. Tune contextual learning rate for each backbone across the entire shared
     ``experiment_utils.LR_SEARCH_GRID``. Each trial starts from that
     backbone's manifest-verified ITFT checkpoint, trains on the same 80%
     tuning partition, and selects its checkpoint by validation macro-F1 on
     the remaining 20%. The fold-0 held-out fold is never read for tuning.
  3. Run one additional C4 second-stage training-only timing pilot per
     backbone on the full fold-0 CV training portion (``fold != 0``), using
     its selected contextual LR and ITFT checkpoint. No held-out fold is
     evaluated or used for selection in the timing-only run.
  4. Save the shared contextual length and both contextual learning-rate
     selections only after all trials and timing pilots complete. Writes are
     validated and rolled back if either shared configuration fails validation.
  5. Estimate the Stage 9/10 runtime for 50 CV runs per backbone, including
     measured ITFT and Stage 8 overhead. Estimates are planning aids, not
     guarantees of Colab session duration.

The benchmark length and benchmark LR configuration must already exist and
must match the actual Stage 1 / Stage 1b / benchmark preflight inputs. The
contextual shared configuration is not partially published while trials are
still running. Exact, manifest-verified completed trials may be resumed after
an interruption; legacy/unmanifested outputs are never used.

Run from the repository root (normally in Google Colab with a GPU):
    python scripts/8_timing_pilot.py --validate-only
    python scripts/8_timing_pilot.py

Optional: ``--allow-cpu`` deliberately permits a slow CPU pilot.
"""
from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import platform
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


def _project_root() -> Path:
    here = Path(__file__).resolve().parent
    return here.parent if here.name.casefold() == "scripts" else here


PROJECT_ROOT = _project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import experiment_utils as eu  # noqa: E402

STAGE = 8
SCRIPT_NAME = "scripts/8_timing_pilot.py"
INPUT_MODE = "contextual"
PILOT_CONTEXT_SIZE = 4
PILOT_FOLD = 0
TUNING_VALIDATION_FRACTION = 0.20

# Match the contextual training grid's fixed settings. Only the LR is tuned.
NUM_EPOCHS = 8
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
GRADIENT_ACCUMULATION_STEPS = 1
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.0
LR_SCHEDULER = "linear"
OPTIMIZER_NAME = "AdamW"
CHECKPOINT_METRIC = "macro_f1"
LOSS_NAME = "unweighted_cross_entropy"
CLASS_WEIGHTING_CONTEXT = "none_balanced_matched_dataset"
SPLIT_SEED = 42
LATENCY_WARMUP_RUNS = 5
LATENCY_REPEATS_DEFAULT = 30
LATENCY_MIN_REPEATS = 20
LATENCY_BATCH_SIZE = 1

MATCHED_SUBSET_PATH = Path("data/processed/ctx_matched_subset.csv")
SEQUENCE_CONFIG_PATH = eu.SEQUENCE_LENGTH_CONFIG
LEARNING_RATE_CONFIG_PATH = eu.LEARNING_RATE_CONFIG
BENCHMARK_TRAIN_PATH = Path("data/processed/ib_train.csv")
BENCHMARK_VAL_PATH = Path("data/processed/ib_val.csv")
BENCHMARK_WEIGHTS_PATH = Path("results/tables/ib_class_weights.csv")

STAGE_ROOT = eu.versioned_run_root(PROJECT_ROOT) / "stage_08"
TOKEN_LENGTH_STATS_NAME = "context_token_length_stats.csv"
LR_TRIALS_NAME = "context_lr_trials.csv"
TIMING_SUMMARY_NAME = "context_timing_summary.csv"
SUMMARY_JSON_NAME = "context_pilot_summary.json"
LATENCY_MEASUREMENTS_NAME = "context_latency_measurements.json"
ARTIFACT_INDEX_NAME = "artifacts_sha256.json"
COMPLETION_MARKER_NAME = "stage8_complete.json"

LEGACY_OUTPUTS = (
    "results/tables/pilot_token_length_stats.csv",
    "results/tables/pilot_summary.csv",
    "results/tables/itft_metrics.csv",
    "models/E2_itft_context",
)
EXPECTED_CONTEXT_ROWS = 900
EXPECTED_FOLD_SIZE = 180
PLANNED_RUNS_PER_BACKBONE = len(range(5)) * 5 * len(eu.CV_SEEDS)  # 5 contexts x 5 folds x 2 seeds = 50


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _sha(path: str | Path) -> str:
    return eu.sha256_file(_resolve(path))


def _write_json_atomic(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_csv_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot read JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object in {path}")
    return value


def _lr_key(value: float) -> str:
    """Use the stable JSON key convention from the benchmark LR preflight."""
    return format(float(value), ".0e")


def _resolve_legacy_outputs() -> list[Path]:
    found: list[Path] = []
    for pattern in LEGACY_OUTPUTS:
        found.extend(path for path in PROJECT_ROOT.glob(pattern) if path.exists())
    return list(dict.fromkeys(found))


def _assert_no_legacy_stage8_outputs() -> None:
    eu.assert_no_legacy_outputs(_resolve_legacy_outputs())


def _load_pinned_benchmark_configs() -> dict[str, Any]:
    """Validate the current Stage 1b and benchmark LR preflight provenance."""
    paths = [
        _resolve(BENCHMARK_TRAIN_PATH),
        _resolve(BENCHMARK_VAL_PATH),
        _resolve(BENCHMARK_WEIGHTS_PATH),
        _resolve(SEQUENCE_CONFIG_PATH),
        _resolve(LEARNING_RATE_CONFIG_PATH),
    ]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(
                f"Stage 8 requires the completed benchmark preparation/preflight artifact {path}. "
                "Finish Stages 1 and 1b and scripts/preflight_tune_benchmark_lr.py first."
            )

    train_path, val_path, weights_path, sequence_path, lr_path = paths
    length_doc = eu.load_json_config(sequence_path)
    if "context_length_source_partition" in length_doc or "max_length_context" in length_doc:
        raise FileExistsError(
            f"{sequence_path} already contains a contextual length. Stage 8 only publishes once per clean run; "
            "archive the Stage 8 outputs and shared configs before a deliberate restart."
        )
    lr_raw = eu.load_json_config(lr_path)
    if "learning_rate_context" in lr_raw or "context_trial_macro_f1" in lr_raw:
        raise FileExistsError(
            f"{lr_path} already contains contextual LR results. Archive the historical configs before Stage 8."
        )

    # Utility validation checks schema, p95 calculation, the complete LR grid,
    # chosen benchmark LRs, and that the tuning seed is outside evaluation sets.
    shared = eu.load_shared_run_config(
        sequence_length_path=sequence_path,
        learning_rate_path=lr_path,
        require_context=False,
        require_context_lr=False,
    )
    length_doc = shared["sequence_length"]
    lr_doc = shared["learning_rates"]
    preflight = lr_doc.get("benchmark_preflight")
    if not isinstance(preflight, Mapping):
        raise ValueError("learning_rates.json is missing benchmark_preflight provenance")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    if not {"clean_text", "label"}.issubset(train_df.columns) or not {"clean_text", "label"}.issubset(val_df.columns):
        raise KeyError("Benchmark train/validation CSVs must contain clean_text and label")
    for name, frame in (("train", train_df), ("validation", val_df)):
        if frame.empty or frame["clean_text"].isna().any() or frame["label"].isna().any():
            raise ValueError(f"Benchmark {name} split contains empty rows or missing clean_text/label values")
        if set(frame["label"].astype(str)) != set(eu.LABEL2ID):
            raise ValueError(f"Benchmark {name} split does not contain all canonical labels {sorted(eu.LABEL2ID)}")
    train_sha, val_sha, weights_sha, length_sha = map(eu.sha256_file, paths[:4])
    dataset_sha = eu.sha256_json({"train_sha256": train_sha, "validation_sha256": val_sha})
    split_sha = eu.sha256_json({"train_sha256": train_sha, "validation_sha256": val_sha, "split_seed": SPLIT_SEED})
    model_ids = {key: eu.get_backbone(key).hf_model_id for key in eu.BACKBONES}
    revisions = length_doc.get("tokenizer_resolved_revisions")
    if not isinstance(revisions, Mapping) or set(revisions) != set(eu.BACKBONES):
        raise ValueError("Stage 1b must pin tokenizer_resolved_revisions for both registered backbones")
    if length_doc.get("tokenizer_model_ids") != model_ids:
        raise ValueError("Stage 1b tokenizer_model_ids differ from experiment_utils.BACKBONES")
    for backbone, revision in revisions.items():
        if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
            raise ValueError(f"Stage 1b must record a full 40-character tokenizer commit SHA for {backbone}")

    expected = {
        "train_csv_sha256": train_sha,
        "validation_csv_sha256": val_sha,
        "class_weights_csv_sha256": weights_sha,
        "sequence_length_config_sha256": length_sha,
        "benchmark_dataset_sha256": dataset_sha,
        "split_assignment_sha256": split_sha,
        "max_length_benchmark": length_doc["max_length_benchmark"],
        "truncation_side": "right",
        "model_ids": model_ids,
    }
    mismatch = {key: (preflight.get(key), value) for key, value in expected.items() if preflight.get(key) != value}
    if mismatch:
        raise ValueError(f"Benchmark LR preflight is not consistent with current Stage 1/1b inputs: {mismatch}")
    if preflight.get("uses_test_split") is not False:
        raise ValueError("Benchmark LR preflight must record uses_test_split=false")
    if length_doc.get("benchmark_training_split_sha256") != train_sha:
        raise ValueError("ib_train.csv differs from the training partition used by Stage 1b")
    if length_doc.get("benchmark_training_text_column") != "clean_text":
        raise ValueError("Stage 1b must record benchmark_training_text_column='clean_text'")
    if not isinstance(preflight.get("model_revisions"), Mapping) or set(preflight["model_revisions"]) != set(eu.BACKBONES):
        raise ValueError("Benchmark preflight must pin model_revisions for both backbones")
    for backbone in eu.BACKBONES:
        if str(preflight["model_revisions"][backbone]).lower() != str(revisions[backbone]).lower():
            raise ValueError(f"Model and tokenizer revisions differ for {backbone}; redo Stage 1b/preflight with matching pins")

    lr_seed = int(lr_doc["lr_tuning_seed"])
    if lr_seed in eu.EVAL_SEEDS_EXP1 or lr_seed in eu.CV_SEEDS:
        raise ValueError("lr_tuning_seed must be distinct from all Experiment I and contextual CV seeds")

    return {
        "sequence_length_doc": length_doc,
        "learning_rate_doc": lr_doc,
        "benchmark_train_sha256": train_sha,
        "benchmark_validation_sha256": val_sha,
        "class_weights_sha256": weights_sha,
        "sequence_config_sha256_before_stage8": length_sha,
        "learning_rate_config_sha256_before_stage8": eu.sha256_file(lr_path),
        "benchmark_dataset_sha256": dataset_sha,
        "benchmark_split_assignment_sha256": split_sha,
        "model_revisions": {key: str(value).lower() for key, value in preflight["model_revisions"].items()},
        "tokenizer_revisions": {key: str(value).lower() for key, value in revisions.items()},
        "benchmark_lr": {key: float(value) for key, value in lr_doc["learning_rate_benchmark"].items()},
        "tuning_seed": lr_seed,
        "benchmark_preflight_hyperparameters": dict(preflight.get("hyperparameters", {})),
        "paths": {
            "train": train_path,
            "validation": val_path,
            "weights": weights_path,
            "sequence_config": sequence_path,
            "learning_rate_config": lr_path,
        },
    }


def _load_matched_context_data(path: Path) -> tuple[pd.DataFrame, dict[str, Any], str]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing Stage 2 matched subset: {path}. Run scripts/2_preprocess_contextual.py first.")
    raw = pd.read_csv(path)
    validation = eu.validate_context_dataset(raw, expected_rows=EXPECTED_CONTEXT_ROWS)
    modelling = eu.load_modelling_view(raw, include_fold=True)
    modelling[eu.CTX_FOLD_COLUMN] = pd.to_numeric(modelling[eu.CTX_FOLD_COLUMN], errors="raise").astype(int)
    # Stable ordering makes the nested LR split invariant to saved CSV row order.
    modelling = modelling.sort_values(eu.CTX_ID_COLUMN, key=lambda col: col.astype(str)).reset_index(drop=True)
    dataset_sha = eu.sha256_file(path)
    return modelling, validation, dataset_sha


def make_context_tuning_split(matched: pd.DataFrame, tuning_seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return fold-0 CV training rows, then stable stratified 80/20 train/val.

    Despite the name ``pilot fold=0``, rows assigned to fold 0 are held out by
    the actual CV experiment and are deliberately excluded from this function.
    The tuning pool is exactly rows where fold != 0 (about 720 examples).
    """
    required = {eu.CTX_ID_COLUMN, "label", eu.CTX_FOLD_COLUMN}
    missing = required - set(matched.columns)
    if missing:
        raise KeyError(f"Matched subset lacks columns required for the Stage 8 split: {sorted(missing)}")
    if len(matched) != EXPECTED_CONTEXT_ROWS:
        raise ValueError(f"Expected the full {EXPECTED_CONTEXT_ROWS}-row matched subset before excluding fold 0; got {len(matched)}")
    fold_values = pd.to_numeric(matched[eu.CTX_FOLD_COLUMN], errors="coerce")
    if fold_values.isna().any() or not np.equal(fold_values, np.floor(fold_values)).all():
        raise ValueError("Matched subset fold column must contain integer values")
    observed_folds = fold_values.astype(int).value_counts().to_dict()
    expected_folds = {fold: EXPECTED_FOLD_SIZE for fold in range(5)}
    if observed_folds != expected_folds:
        raise ValueError(f"Expected five shared folds with {EXPECTED_FOLD_SIZE} rows each, got {observed_folds}")
    cv_train = matched.loc[fold_values.astype(int) != PILOT_FOLD].copy()
    if len(cv_train) != EXPECTED_CONTEXT_ROWS - EXPECTED_FOLD_SIZE:
        raise ValueError(f"Expected {EXPECTED_CONTEXT_ROWS - EXPECTED_FOLD_SIZE} rows from folds 1-4, got {len(cv_train)}")
    if (cv_train[eu.CTX_FOLD_COLUMN] == PILOT_FOLD).any():
        raise AssertionError("Fold-0 held-out examples leaked into contextual LR tuning")
    train_part, val_part = train_test_split(
        cv_train,
        test_size=TUNING_VALIDATION_FRACTION,
        random_state=int(tuning_seed),
        stratify=cv_train["label"].astype(str),
    )
    train_part = train_part.sort_values(eu.CTX_ID_COLUMN, key=lambda col: col.astype(str)).reset_index(drop=True)
    val_part = val_part.sort_values(eu.CTX_ID_COLUMN, key=lambda col: col.astype(str)).reset_index(drop=True)
    if set(train_part[eu.CTX_ID_COLUMN].astype(str)) & set(val_part[eu.CTX_ID_COLUMN].astype(str)):
        raise AssertionError("Contextual LR train/validation post IDs overlap")
    if len(train_part) + len(val_part) != len(cv_train):
        raise AssertionError("The contextual 80/20 split does not cover the CV training portion")
    return cv_train, train_part, val_part


def _id_hash(df: pd.DataFrame) -> str:
    return eu.sha256_json(sorted(df[eu.CTX_ID_COLUMN].astype(str).tolist()))


def _context_texts(
    df: pd.DataFrame, tokenizer, context_size: int = PILOT_CONTEXT_SIZE
) -> list[str]:
    """Build tokenizer-specific C0-C4 text in the shared chronological order."""
    separator = eu.sep_from_tokenizer(tokenizer)
    return [eu.build_context_text(row, context_size, sep=separator) for _, row in df.iterrows()]


def _select_latency_rows(tuning_train: pd.DataFrame, repeats: int) -> pd.DataFrame:
    """Choose the same deterministic, spread-out posts for all backbones/contexts."""
    if repeats < LATENCY_MIN_REPEATS:
        raise ValueError(f"Stage 8 latency requires at least {LATENCY_MIN_REPEATS} measured observations per backbone/context")
    ordered = tuning_train.sort_values(
        eu.CTX_ID_COLUMN, key=lambda col: col.astype(str)
    ).reset_index(drop=True)
    if repeats > len(ordered):
        raise ValueError(f"Requested {repeats} latency observations but only {len(ordered)} training examples are available")
    positions = np.linspace(0, len(ordered) - 1, num=repeats, dtype=int)
    sampled = ordered.iloc[positions].copy().reset_index(drop=True)
    if sampled[eu.CTX_ID_COLUMN].duplicated().any():
        raise AssertionError("Latency sample selection unexpectedly duplicated post IDs")
    return sampled


def _latency_machine_label(torch: Any, override: str | None = None) -> str:
    """Describe the common machine/device used for both backbone measurements."""
    if override is not None and override.strip():
        return override.strip()
    if torch.cuda.is_available():
        device = f"CUDA:{torch.cuda.get_device_name(0)}"
    else:
        device = f"CPU:{platform.processor() or platform.machine()}"
    return f"{platform.system()} {platform.machine()} | {device} | PyTorch {torch.__version__}"


def _synchronize_if_cuda(torch: Any, use_cuda: bool) -> None:
    if use_cuda:
        torch.cuda.synchronize()


def _measure_inference_latency_by_context(
    *,
    torch: Any,
    AutoModel: Any,
    tokenizers: Mapping[str, Any],
    sample_rows: pd.DataFrame,
    context_length: int,
    itft_by_backbone: Mapping[str, Any],
    use_cuda: bool,
    warmup_runs: int = LATENCY_WARMUP_RUNS,
    measurement_machine: str,
    matched_subset_path: Path,
) -> dict[str, Any]:
    """Measure single-post tokenization + forward-pass latency for C0-C4.

    Each measured call processes exactly one post without batch padding, with
    the same context length cap and explicit left truncation. The same ordered
    post IDs, runtime, precision and warm-up protocol are used for both
    backbones. Stage 8 uses manifest-verified ITFT checkpoints because their
    architecture and three-class inference head match downstream fine-tuned
    models; fine-tuning does not change the inference graph.
    """
    if warmup_runs < 1:
        raise ValueError("At least one latency warm-up is required")
    if set(tokenizers) != set(eu.BACKBONES) or set(itft_by_backbone) != set(eu.BACKBONES):
        raise ValueError("Latency measurement requires tokenizers and verified ITFT checkpoints for both backbones")
    if not measurement_machine.strip():
        raise ValueError("Latency measurement must identify the measurement machine")

    sample_ids = sample_rows[eu.CTX_ID_COLUMN].astype(str).tolist()
    measurements: dict[str, dict[str, list[float]]] = {f"C{c}": {} for c in range(5)}
    device = torch.device("cuda" if use_cuda else "cpu")
    for backbone in eu.BACKBONES:
        tokenizer = tokenizers[backbone]
        if getattr(tokenizer, "truncation_side", None) != "left":
            raise ValueError(f"Latency tokenizer for {backbone} must use explicit left truncation")
        print(f"\nMeasuring single-post latency for {backbone} on {measurement_machine}")
        model = AutoModel.from_pretrained(
            str(itft_by_backbone[backbone]["model_dir"]),
            num_labels=len(eu.LABEL2ID),
            id2label=eu.ID2LABEL,
            label2id=eu.LABEL2ID,
            ignore_mismatched_sizes=False,
        )
        model.to(device)
        model.eval()
        try:
            for context_size in range(5):
                texts = _context_texts(sample_rows, tokenizer, context_size=context_size)
                # Warm up each context size so shape-dependent kernels/caches
                # are initialized before recording measured observations.
                for warm_idx in range(warmup_runs):
                    encoded = tokenizer(
                        texts[warm_idx % len(texts)],
                        add_special_tokens=True, truncation=True,
                        max_length=context_length, padding=False, return_tensors="pt",
                    )
                    encoded = {key: value.to(device) for key, value in encoded.items()}
                    with torch.inference_mode():
                        _ = model(**encoded)
                _synchronize_if_cuda(torch, use_cuda)

                samples_ms: list[float] = []
                for text in texts:
                    _synchronize_if_cuda(torch, use_cuda)
                    started = time.perf_counter()
                    encoded = tokenizer(
                        text, add_special_tokens=True, truncation=True,
                        max_length=context_length, padding=False, return_tensors="pt",
                    )
                    encoded = {key: value.to(device) for key, value in encoded.items()}
                    with torch.inference_mode():
                        _ = model(**encoded)
                    _synchronize_if_cuda(torch, use_cuda)
                    elapsed_ms = (time.perf_counter() - started) * 1000.0
                    if not math.isfinite(elapsed_ms) or elapsed_ms <= 0:
                        raise RuntimeError(f"Invalid inference latency for {backbone} C{context_size}: {elapsed_ms}")
                    samples_ms.append(float(elapsed_ms))
                measurements[f"C{context_size}"][backbone] = samples_ms
                median = float(np.median(samples_ms))
                p95 = float(np.percentile(samples_ms, 95))
                print(f"  C{context_size}: median={median:.3f} ms; p95={p95:.3f} ms; n={len(samples_ms)}")
        finally:
            del model
            if use_cuda:
                torch.cuda.empty_cache()

    summary_by_context = {
        context: eu.summarize_latency_measurements(
            backbone_samples, measurement_machine=measurement_machine, min_repeats=LATENCY_MIN_REPEATS
        )
        for context, backbone_samples in measurements.items()
    }
    medians_by_context = {context: dict(summary["median_ms"]) for context, summary in summary_by_context.items()}
    return {
        "schema_version": eu.CONFIG_SCHEMA_VERSION,
        "run_id": eu.RUN_ID,
        "stage": STAGE,
        "script_name": SCRIPT_NAME,
        "measurement_machine": measurement_machine,
        "measurement_protocol": (
            "single-post inference; tokenizer + device transfer + model forward; no gradients; "
            "unpadded single-example input; tokenizer truncation cap=max_length_context; "
            "left truncation; float32 model weights/inference; fixed warm-up per context"
        ),
        "device": "cuda" if use_cuda else "cpu",
        "inference_batch_size": LATENCY_BATCH_SIZE,
        "precision": "float32",
        "warmup_runs_per_backbone_context": int(warmup_runs),
        "measurement_repeats_per_backbone_context": int(len(sample_rows)),
        "context_sizes": [0, 1, 2, 3, 4],
        "max_length_context": int(context_length),
        "truncation_side": "left",
        "sample_post_ids": sample_ids,
        "sample_post_ids_sha256": eu.sha256_json(sample_ids),
        "matched_subset_sha256": eu.sha256_file(matched_subset_path),
        "fold_assignment_sha256": eu.validate_context_dataset(
            pd.read_csv(matched_subset_path), expected_rows=EXPECTED_CONTEXT_ROWS
        )["fold_assignment_sha256"],
        "measurements_ms_by_context": measurements,
        "summary_by_context": summary_by_context,
        "median_ms_by_context": medians_by_context,
        "model_ids": {b: eu.get_backbone(b).hf_model_id for b in eu.BACKBONES},
        "model_revisions": {b: itft_by_backbone[b]["run_manifest"]["config"]["model_revision"] for b in eu.BACKBONES},
        "tokenizer_revisions": {b: itft_by_backbone[b]["run_manifest"]["config"]["tokenizer_revision"] for b in eu.BACKBONES},
        "itft_manifest_fingerprints": {b: itft_by_backbone[b]["manifest_fingerprint"] for b in eu.BACKBONES},
    }


def _token_length_stats(lengths: Sequence[int]) -> dict[str, float | int]:
    values = np.asarray(lengths, dtype=np.int64)
    if values.ndim != 1 or len(values) == 0 or (values <= 0).any():
        raise ValueError("Token-length statistics require a non-empty vector of positive lengths")
    return {
        "n": int(len(values)),
        "min": int(values.min()),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": int(values.max()),
    }


def _validate_itft_checkpoint(backbone: str, audit: Mapping[str, Any]) -> dict[str, Any]:
    """Verify Stage 6/7 manifests, source hashes and every saved checkpoint file."""
    stage = 6 if backbone == "indobertweet" else 7
    run_dir = eu.versioned_run_root(PROJECT_ROOT) / f"stage_{stage:02d}" / f"itft_{backbone}"
    model_dir = PROJECT_ROOT / "models" / "itft_checkpoint" / backbone
    manifest_path = run_dir / eu.RUN_MANIFEST_NAME
    marker_path = run_dir / "checkpoint_complete.json"
    index_path = run_dir / "artifacts_sha256.json"
    metrics_path = run_dir / "metrics.json"
    model_manifest_path = model_dir / "itft_checkpoint_manifest.json"
    for path in (manifest_path, marker_path, index_path, metrics_path, model_manifest_path, model_dir / "config.json"):
        if not path.exists():
            raise FileNotFoundError(f"Stage {stage} ITFT artifact is missing: {path}; complete the current Stage {stage} run first")

    run_manifest = _read_json(manifest_path)
    fingerprint = run_manifest.get("fingerprint")
    eu.assert_current_run_artifact(
        marker_path,
        expected_stage=stage,
        expected_script="scripts/6_train_itft.py",
        expected_fingerprint=str(fingerprint),
    )
    # Validate the producer's input hashes before Stage 8 changes the shared
    # sequence-length file by adding its contextual-length decision.
    input_hashes = run_manifest.get("input_sha256")
    if not isinstance(input_hashes, Mapping) or not input_hashes:
        raise ValueError(f"Stage {stage} ITFT manifest does not record input hashes")
    for stored_path, expected_hash in input_hashes.items():
        input_path = Path(str(stored_path))
        input_path = input_path if input_path.is_absolute() else PROJECT_ROOT / input_path
        if not input_path.is_file():
            raise FileNotFoundError(f"ITFT input recorded in manifest is missing: {input_path}")
        if eu.sha256_file(input_path) != expected_hash:
            raise ValueError(f"ITFT source input has changed since Stage {stage}: {input_path}")

    marker = _read_json(marker_path)
    model_manifest = _read_json(model_manifest_path)
    index = _read_json(index_path)
    metrics = _read_json(metrics_path)
    if marker.get("manifest_fingerprint") != fingerprint or marker.get("backbone") != backbone or int(marker.get("stage", -1)) != stage:
        raise ValueError(f"Stage {stage} completion marker does not match its run manifest")
    if model_manifest.get("manifest_fingerprint") != fingerprint or model_manifest.get("backbone") != backbone:
        raise ValueError(f"Stage {stage} checkpoint manifest belongs to another run/backbone")
    if metrics.get("stage") != stage or metrics.get("backbone") != backbone:
        raise ValueError(f"Stage {stage} metrics do not match the expected backbone")
    if index.get("manifest_fingerprint") != fingerprint or index.get("stage") != stage or index.get("backbone") != backbone:
        raise ValueError(f"Stage {stage} artifact hash index does not match the run manifest")
    expected_file_hashes = model_manifest.get("checkpoint_files_sha256")
    if not isinstance(expected_file_hashes, Mapping) or not expected_file_hashes:
        raise ValueError(f"Stage {stage} checkpoint manifest has no checkpoint_files_sha256 map")
    actual_file_hashes: dict[str, str] = {}
    for path in sorted(model_dir.rglob("*")):
        if not path.is_file() or path.name == "itft_checkpoint_manifest.json":
            continue
        actual_file_hashes[path.relative_to(model_dir).as_posix()] = eu.sha256_file(path)
    if dict(expected_file_hashes) != actual_file_hashes:
        raise ValueError(f"Stage {stage} model/tokenizer checkpoint hash mismatch; archive and rerun ITFT")
    indexed_hashes = index.get("sha256", {})
    if indexed_hashes.get("metrics.json") != eu.sha256_file(metrics_path):
        raise ValueError(f"Stage {stage} metrics hash does not match artifacts_sha256.json")
    if indexed_hashes.get("checkpoint_manifest_sha256") != eu.sha256_file(model_manifest_path):
        raise ValueError(f"Stage {stage} checkpoint-manifest hash does not match artifacts_sha256.json")
    if marker.get("checkpoint_manifest_sha256") != eu.sha256_file(model_manifest_path):
        raise ValueError(f"Stage {stage} completion marker has an inconsistent checkpoint-manifest hash")

    spec = eu.get_backbone(backbone)
    revision = audit["model_revisions"][backbone]
    tokenizer_revision = audit["tokenizer_revisions"][backbone]
    if model_manifest.get("hf_model_id") != spec.hf_model_id:
        raise ValueError(f"Stage {stage} ITFT checkpoint model ID differs from BACKBONES")
    if str(model_manifest.get("model_revision", "")).lower() != revision.lower():
        raise ValueError(f"Stage {stage} ITFT checkpoint model revision differs from Stage 1b/preflight")
    if str(model_manifest.get("tokenizer_revision", "")).lower() != tokenizer_revision.lower():
        raise ValueError(f"Stage {stage} ITFT checkpoint tokenizer revision differs from Stage 1b")
    if metrics.get("max_length_benchmark") != audit["sequence_length_doc"]["max_length_benchmark"]:
        raise ValueError(f"Stage {stage} ITFT checkpoint used a different benchmark sequence length")
    if not np.isclose(float(metrics.get("learning_rate_benchmark", -1)), audit["benchmark_lr"][backbone]):
        raise ValueError(f"Stage {stage} ITFT checkpoint used a different benchmark learning rate")
    elapsed = float(metrics.get("elapsed_seconds", 0))
    if not math.isfinite(elapsed) or elapsed <= 0:
        raise ValueError(f"Stage {stage} ITFT metrics lack a positive elapsed_seconds value")
    return {
        "stage": stage,
        "backbone": backbone,
        "run_dir": run_dir,
        "model_dir": model_dir,
        "run_manifest": run_manifest,
        "manifest_fingerprint": str(fingerprint),
        "marker_path": marker_path,
        "model_manifest_path": model_manifest_path,
        "metrics_path": metrics_path,
        "metrics": metrics,
        "elapsed_seconds": elapsed,
        "checkpoint_files_sha256": dict(expected_file_hashes),
    }


def _build_resolved_config(
    backbone: str,
    audit: Mapping[str, Any],
    matched_sha: str,
    fold_audit: Mapping[str, Any],
    context_length: int,
    tuning_train: pd.DataFrame,
    tuning_val: pd.DataFrame,
    learning_rate: float,
    *,
    mixed_precision: bool,
) -> dict[str, Any]:
    spec = eu.get_backbone(backbone)
    context = audit.get("context_length_provenance", {})
    return {
        "schema_version": eu.CONFIG_SCHEMA_VERSION,
        "run_id": eu.RUN_ID,
        "stage": STAGE,
        "script_name": SCRIPT_NAME,
        "experiment_id": f"E2-{spec.id_token}-ITFT-C4-LR-tuning",
        "backbone": backbone,
        "hf_model_id": spec.hf_model_id,
        "model_revision": audit["model_revisions"][backbone],
        "tokenizer_revision": audit["tokenizer_revisions"][backbone],
        "input_mode": INPUT_MODE,
        "input_order": "parents_oldest_to_newest_then_target",
        "context_order": "parent_4,parent_3,parent_2,parent_1,target",
        "dataset_sha256": matched_sha,
        "fold_assignment_sha256": fold_audit["fold_assignment_sha256"],
        "post_id_sha256": fold_audit["post_id_sha256"],
        "split_seed": SPLIT_SEED,
        "fold_column": eu.CTX_FOLD_COLUMN,
        "n_folds": 5,
        "cv_seeds": list(eu.CV_SEEDS),
        "eval_seeds_exp1": list(eu.EVAL_SEEDS_EXP1),
        "max_length_benchmark": int(audit["sequence_length_doc"]["max_length_benchmark"]),
        "max_length_context": int(context_length),
        "truncation_side": "left",
        "optimizer": OPTIMIZER_NAME,
        "weight_decay": WEIGHT_DECAY,
        "warmup_ratio": WARMUP_RATIO,
        "lr_scheduler": LR_SCHEDULER,
        "max_epochs": NUM_EPOCHS,
        "early_stopping_patience": None,
        "loss": LOSS_NAME,
        "class_weighting_benchmark": CLASS_WEIGHTING_CONTEXT,
        "checkpoint_metric": CHECKPOINT_METRIC,
        "lr_search_grid": [float(x) for x in eu.LR_SEARCH_GRID],
        "lr_tuning_seed": int(audit["tuning_seed"]),
        "mixed_precision": bool(mixed_precision),
        "per_device_batch_size": TRAIN_BATCH_SIZE,
        "eval_batch_size": EVAL_BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "effective_batch_size": TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS,
        "learning_rate_benchmark": float(audit["benchmark_lr"][backbone]),
        "learning_rate_context": float(learning_rate),
        "tuning_train_post_ids_sha256": _id_hash(tuning_train),
        "tuning_validation_post_ids_sha256": _id_hash(tuning_val),
        "tuning_train_rows": int(len(tuning_train)),
        "tuning_validation_rows": int(len(tuning_val)),
        "cv_train_pool_rows": int(EXPECTED_CONTEXT_ROWS - EXPECTED_FOLD_SIZE),
        "tuning_validation_fraction": TUNING_VALIDATION_FRACTION,
        "fold_0_held_out_used": False,
        "sequence_length_config_sha256_before_stage8": audit["sequence_config_sha256_before_stage8"],
        "learning_rate_config_sha256_before_stage8": audit["learning_rate_config_sha256_before_stage8"],
    }


def _metric_row_from_eval(metrics: Mapping[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for source, target in (
        ("eval_accuracy", "accuracy"),
        ("eval_macro_f1", "macro_f1"),
        ("eval_weighted_f1", "weighted_f1"),
        ("eval_f1_n", "f1_n"),
        ("eval_f1_al", "f1_al"),
        ("eval_f1_hs", "f1_hs"),
    ):
        value = float(metrics.get(source, float("nan")))
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"Invalid validation metric {source}: {value}")
        result[target] = value
    return result


def _compute_metrics(eval_pred: Any) -> dict[str, float]:
    logits, labels = eval_pred
    y_true = np.asarray(labels, dtype=np.int64)
    y_pred = np.asarray(logits).argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=[0, 1, 2], average="weighted", zero_division=0)),
        "f1_n": float(f1_score(y_true, y_pred, labels=[0], average="macro", zero_division=0)),
        "f1_al": float(f1_score(y_true, y_pred, labels=[1], average="macro", zero_division=0)),
        "f1_hs": float(f1_score(y_true, y_pred, labels=[2], average="macro", zero_division=0)),
    }


def _load_training_dependencies():
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, set_seed
    except ImportError as exc:
        raise ImportError(
            "Stage 8 requires PyTorch, Transformers, scikit-learn and pandas. Run the pilot in the configured Colab environment."
        ) from exc
    return torch, AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, set_seed


def _make_dataset_class(torch: Any):
    class TokenizedDataset(torch.utils.data.Dataset):
        def __init__(self, encodings: Mapping[str, Any], labels: Sequence[int]):
            self.encodings = encodings
            self.labels = np.asarray(labels, dtype=np.int64)

        def __len__(self) -> int:
            return len(self.labels)

        def __getitem__(self, index: int) -> dict[str, Any]:
            item = {key: value[index] for key, value in self.encodings.items()}
            item["labels"] = torch.tensor(int(self.labels[index]), dtype=torch.long)
            return item

    return TokenizedDataset


def _make_training_arguments(
    TrainingArguments: Any,
    *,
    output_dir: Path,
    learning_rate: float,
    seed: int,
    use_fp16: bool,
    disable_tqdm: bool,
    evaluate: bool,
):
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
        eval_key: "epoch" if evaluate else "no",
        "save_strategy": "epoch" if evaluate else "no",
        "logging_strategy": "epoch",
        "seed": int(seed),
        "data_seed": int(seed),
        "report_to": "none",
        "fp16": bool(use_fp16),
        "disable_tqdm": bool(disable_tqdm),
    }
    if evaluate:
        kwargs.update({
            "save_total_limit": 1,
            "load_best_model_at_end": True,
            "metric_for_best_model": CHECKPOINT_METRIC,
            "greater_is_better": True,
        })
    else:
        kwargs.update({
            "load_best_model_at_end": False,
        })
    for optional_key in ("data_seed", "lr_scheduler_type", "warmup_ratio", "disable_tqdm"):
        if optional_key not in signature:
            kwargs.pop(optional_key, None)
    return TrainingArguments(**kwargs)


def _verify_completed_run(
    run_dir: Path,
    result_name: str,
    *,
    backbone: str,
    expected_lr: float | None,
    kind: str,
) -> dict[str, Any] | None:
    """Return a validated completed result, or None for a manifest-only fresh run."""
    manifest_path = run_dir / eu.RUN_MANIFEST_NAME
    result_path = run_dir / result_name
    index_path = run_dir / ARTIFACT_INDEX_NAME
    untracked = [
        path for path in run_dir.rglob("*")
        if path.is_file() and path not in {manifest_path, result_path, index_path}
    ]
    if untracked:
        raise RuntimeError(
            f"Stage 8 run directory contains unexpected/partial outputs: {untracked}. Archive this directory; do not reuse them."
        )
    if not result_path.exists() and not index_path.exists():
        return None
    if not result_path.is_file() or not index_path.is_file() or not manifest_path.is_file():
        raise RuntimeError(f"Incomplete Stage 8 {kind} outputs in {run_dir}; archive this run directory before retrying")
    manifest = _read_json(manifest_path)
    fingerprint = str(manifest.get("fingerprint", ""))
    eu.assert_current_run_artifact(
        result_path,
        expected_stage=STAGE,
        expected_script=SCRIPT_NAME,
        expected_fingerprint=fingerprint,
    )
    result = _read_json(result_path)
    index = _read_json(index_path)
    if (
        result.get("manifest_fingerprint") != fingerprint
        or result.get("run_id") != eu.RUN_ID
        or result.get("stage") != STAGE
        or result.get("backbone") != backbone
        or result.get("kind") != kind
    ):
        raise RuntimeError(f"Completed Stage 8 {kind} result does not match current run manifest: {result_path}")
    if expected_lr is not None and not np.isclose(float(result.get("learning_rate", -1)), expected_lr):
        raise RuntimeError(f"Saved LR trial in {result_path} has the wrong candidate learning rate")
    if index.get("manifest_fingerprint") != fingerprint or index.get("backbone") != backbone:
        raise RuntimeError(f"Artifact index does not match Stage 8 run manifest: {index_path}")
    if index.get("sha256", {}).get(result_name) != eu.sha256_file(result_path):
        raise RuntimeError(f"Stage 8 {kind} result hash mismatch; archive the directory")
    if kind == "context_lr_trial":
        score = float(result.get("validation_macro_f1", float("nan")))
        elapsed = float(result.get("elapsed_seconds", float("nan")))
        if not (math.isfinite(score) and 0 <= score <= 1 and math.isfinite(elapsed) and elapsed > 0):
            raise RuntimeError(f"Invalid completed context LR trial result: {result_path}")
    if kind == "timing_only":
        elapsed = float(result.get("elapsed_seconds", float("nan")))
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise RuntimeError(f"Invalid timing-only result: {result_path}")
    return result


def _prepare_trial_run(
    backbone: str,
    learning_rate: float,
    audit: Mapping[str, Any],
    matched: pd.DataFrame,
    fold_audit: Mapping[str, Any],
    context_length: int,
    tuning_train: pd.DataFrame,
    tuning_val: pd.DataFrame,
    itft: Mapping[str, Any],
    *,
    mixed_precision: bool,
) -> tuple[Path, dict[str, Any]]:
    cfg = _build_resolved_config(
        backbone, audit, audit["matched_sha256"], fold_audit, context_length,
        tuning_train, tuning_val, learning_rate, mixed_precision=mixed_precision,
    )
    eu.validate_resolved_run_config(
        cfg, stage=STAGE, backbone=backbone, input_mode=INPUT_MODE,
        context_size=PILOT_CONTEXT_SIZE, require_shared_config=False,
    )
    run_dir = STAGE_ROOT / "context_lr" / backbone / f"lr_{_lr_key(learning_rate).replace('-', 'm')}"
    manifest = eu.make_run_manifest(
        stage=STAGE,
        script_name=SCRIPT_NAME,
        config=cfg,
        input_paths=[
            _resolve(MATCHED_SUBSET_PATH),
            audit["paths"]["sequence_config"],
            audit["paths"]["learning_rate_config"],
            itft["marker_path"],
            itft["model_manifest_path"],
            itft["metrics_path"],
        ],
        metadata={
            "kind": "context_lr_trial",
            "candidate_learning_rate": float(learning_rate),
            "context_size": PILOT_CONTEXT_SIZE,
            "fold_used_as_cv_test": False,
            "fold0_holdout_used": False,
            "tuning_train_post_ids_sha256": _id_hash(tuning_train),
            "tuning_validation_post_ids_sha256": _id_hash(tuning_val),
            "itft_manifest_fingerprint": itft["manifest_fingerprint"],
        },
        project_root=PROJECT_ROOT,
    )
    eu.prepare_run_directory(run_dir, manifest, allow_resume=True)
    return run_dir, manifest


def _run_context_lr_trial(
    *,
    torch: Any,
    AutoModel: Any,
    Trainer: Any,
    TrainingArguments: Any,
    set_seed: Any,
    Dataset: Any,
    backbone: str,
    learning_rate: float,
    tokenizer: Any,
    tuning_train_enc: Mapping[str, Any],
    tuning_val_enc: Mapping[str, Any],
    tuning_train: pd.DataFrame,
    tuning_val: pd.DataFrame,
    audit: Mapping[str, Any],
    fold_audit: Mapping[str, Any],
    context_length: int,
    matched: pd.DataFrame,
    itft: Mapping[str, Any],
    use_cuda: bool,
    disable_tqdm: bool,
) -> dict[str, Any]:
    run_dir, manifest = _prepare_trial_run(
        backbone, learning_rate, audit, matched, fold_audit, context_length,
        tuning_train, tuning_val, itft, mixed_precision=use_cuda,
    )
    completed = _verify_completed_run(
        run_dir, "trial_result.json", backbone=backbone,
        expected_lr=float(learning_rate), kind="context_lr_trial",
    )
    if completed is not None:
        completed["_result_path"] = str(run_dir / "trial_result.json")
        print(f"[resume verified] Stage 8 context LR trial {backbone} lr={learning_rate:g}")
        return completed

    train_labels = np.asarray([eu.LABEL2ID[str(v)] for v in tuning_train["label"]], dtype=np.int64)
    val_labels = np.asarray([eu.LABEL2ID[str(v)] for v in tuning_val["label"]], dtype=np.int64)
    train_ds = Dataset(tuning_train_enc, train_labels)
    val_ds = Dataset(tuning_val_enc, val_labels)
    spec = eu.get_backbone(backbone)
    set_seed(int(audit["tuning_seed"]))
    start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix=f"stage8-{backbone}-lr-", dir=PROJECT_ROOT) as temporary_dir:
        model = AutoModel.from_pretrained(
            str(itft["model_dir"]),
            num_labels=len(eu.LABEL2ID),
            id2label=eu.ID2LABEL,
            label2id=eu.LABEL2ID,
            ignore_mismatched_sizes=False,
        )
        training_args = _make_training_arguments(
            TrainingArguments,
            output_dir=Path(temporary_dir) / "trainer_checkpoints",
            learning_rate=learning_rate,
            seed=int(audit["tuning_seed"]),
            use_fp16=use_cuda,
            disable_tqdm=disable_tqdm,
            evaluate=True,
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            compute_metrics=_compute_metrics,
        )
        trainer.train()
        metrics = trainer.evaluate(val_ds)
        elapsed = time.perf_counter() - start
        selected_metrics = _metric_values(metrics)
        if not math.isfinite(selected_metrics["macro_f1"]) or not 0 <= selected_metrics["macro_f1"] <= 1:
            raise RuntimeError(f"Invalid validation macro-F1 for {backbone} at LR {learning_rate:g}")
        result = {
            "schema_version": eu.CONFIG_SCHEMA_VERSION,
            "run_id": eu.RUN_ID,
            "stage": STAGE,
            "script_name": SCRIPT_NAME,
            "kind": "context_lr_trial",
            "backbone": backbone,
            "model_id": spec.hf_model_id,
            "model_revision": audit["model_revisions"][backbone],
            "tokenizer_revision": audit["tokenizer_revisions"][backbone],
            "learning_rate": float(learning_rate),
            "tuning_seed": int(audit["tuning_seed"]),
            "context_size": PILOT_CONTEXT_SIZE,
            "max_length_context": int(context_length),
            "truncation_side": "left",
            "cv_training_pool_rows": int(len(matched.loc[matched[eu.CTX_FOLD_COLUMN] != PILOT_FOLD])),
            "training_rows": int(len(tuning_train)),
            "validation_rows": int(len(tuning_val)),
            "training_post_ids_sha256": _id_hash(tuning_train),
            "validation_post_ids_sha256": _id_hash(tuning_val),
            "fold_assignment_sha256": fold_audit["fold_assignment_sha256"],
            "matched_subset_sha256": audit["matched_sha256"],
            "uses_fold0_held_out": False,
            "validation_macro_f1": selected_metrics["macro_f1"],
            "validation_accuracy": selected_metrics["accuracy"],
            "validation_weighted_f1": selected_metrics["weighted_f1"],
            "validation_f1_n": selected_metrics["f1_n"],
            "validation_f1_al": selected_metrics["f1_al"],
            "validation_f1_hs": selected_metrics["f1_hs"],
            "best_global_step": int(getattr(trainer.state, "best_global_step", 0) or 0),
            "elapsed_seconds": round(float(elapsed), 3),
            "manifest_fingerprint": manifest["fingerprint"],
        }
        del trainer, model
    result_path = run_dir / "trial_result.json"
    _write_json_atomic(result_path, result)
    _write_json_atomic(run_dir / ARTIFACT_INDEX_NAME, {
        "schema_version": eu.CONFIG_SCHEMA_VERSION,
        "run_id": eu.RUN_ID,
        "stage": STAGE,
        "kind": "context_lr_trial",
        "backbone": backbone,
        "manifest_fingerprint": manifest["fingerprint"],
        "sha256": {"trial_result.json": eu.sha256_file(result_path)},
    })
    eu.assert_current_run_artifact(
        result_path,
        expected_stage=STAGE,
        expected_script=SCRIPT_NAME,
        expected_fingerprint=manifest["fingerprint"],
    )
    result["_result_path"] = str(result_path)
    if use_cuda:
        torch.cuda.empty_cache()
    return result


def _metric_values(metrics: Mapping[str, Any]) -> dict[str, float]:
    result = {}
    mapping = {
        "eval_accuracy": "accuracy", "eval_macro_f1": "macro_f1",
        "eval_weighted_f1": "weighted_f1", "eval_f1_n": "f1_n",
        "eval_f1_al": "f1_al", "eval_f1_hs": "f1_hs",
    }
    for source, target in mapping.items():
        value = float(metrics.get(source, float("nan")))
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"Validation metric {source!r} is invalid: {value}")
        result[target] = value
    return result


def _prepare_timing_run(
    backbone: str,
    audit: Mapping[str, Any],
    matched: pd.DataFrame,
    fold_audit: Mapping[str, Any],
    context_length: int,
    cv_train: pd.DataFrame,
    selected_lr: float,
    itft: Mapping[str, Any],
    *,
    mixed_precision: bool,
    trial_scores: Mapping[float, float],
) -> tuple[Path, dict[str, Any]]:
    cfg = _build_resolved_config(
        backbone, audit, audit["matched_sha256"], fold_audit, context_length,
        cv_train, cv_train.iloc[:0].copy(), selected_lr, mixed_precision=mixed_precision,
    )
    cfg.update({
        "experiment_id": f"E2-{eu.get_backbone(backbone).id_token}-ITFT-C4-timing-pilot",
        "kind": "timing_only",
        "learning_rate_context_trial_scores": {_lr_key(lr): float(score) for lr, score in trial_scores.items()},
        "timing_train_post_ids_sha256": _id_hash(cv_train),
        "timing_train_rows": int(len(cv_train)),
        "validation_not_used": True,
        "fold_0_held_out_used": False,
    })
    eu.validate_resolved_run_config(
        cfg, stage=STAGE, backbone=backbone, input_mode=INPUT_MODE,
        context_size=PILOT_CONTEXT_SIZE, require_shared_config=False,
    )
    run_dir = STAGE_ROOT / "timing" / backbone
    manifest = eu.make_run_manifest(
        stage=STAGE,
        script_name=SCRIPT_NAME,
        config=cfg,
        input_paths=[
            _resolve(MATCHED_SUBSET_PATH),
            audit["paths"]["sequence_config"],
            audit["paths"]["learning_rate_config"],
            itft["marker_path"],
            itft["model_manifest_path"],
            itft["metrics_path"],
        ],
        metadata={
            "kind": "timing_only",
            "selected_learning_rate": float(selected_lr),
            "context_size": PILOT_CONTEXT_SIZE,
            "planned_cv_runs_per_backbone": PLANNED_RUNS_PER_BACKBONE,
            "fold0_heldout_used": False,
            "itft_manifest_fingerprint": itft["manifest_fingerprint"],
        },
        project_root=PROJECT_ROOT,
    )
    eu.prepare_run_directory(run_dir, manifest, allow_resume=True)
    return run_dir, manifest


def _run_timing_only(
    *,
    torch: Any,
    AutoModel: Any,
    Trainer: Any,
    TrainingArguments: Any,
    set_seed: Any,
    Dataset: Any,
    backbone: str,
    tokenizer: Any,
    cv_train: pd.DataFrame,
    audit: Mapping[str, Any],
    fold_audit: Mapping[str, Any],
    matched: pd.DataFrame,
    context_length: int,
    selected_lr: float,
    trial_scores: Mapping[float, float],
    itft: Mapping[str, Any],
    use_cuda: bool,
    disable_tqdm: bool,
) -> dict[str, Any]:
    run_dir, manifest = _prepare_timing_run(
        backbone, audit, matched, fold_audit, context_length, cv_train,
        selected_lr, itft, mixed_precision=use_cuda, trial_scores=trial_scores,
    )
    completed = _verify_completed_run(
        run_dir, "timing_result.json", backbone=backbone,
        expected_lr=float(selected_lr), kind="timing_only",
    )
    if completed is not None:
        completed["_result_path"] = str(run_dir / "timing_result.json")
        completed["mixed_precision"] = bool(use_cuda)
        print(f"[resume verified] Stage 8 timing-only pilot for {backbone}")
        return completed

    labels = np.asarray([eu.LABEL2ID[str(v)] for v in cv_train["label"]], dtype=np.int64)
    texts = _context_texts(cv_train, tokenizer)
    enc = tokenizer(
        texts, truncation=True, max_length=context_length,
        padding="max_length", return_tensors="pt",
    )
    train_ds = Dataset(enc, labels)
    set_seed(int(audit["tuning_seed"]))
    start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix=f"stage8-{backbone}-timing-", dir=PROJECT_ROOT) as temporary_dir:
        model = AutoModel.from_pretrained(
            str(itft["model_dir"]),
            num_labels=len(eu.LABEL2ID),
            id2label=eu.ID2LABEL,
            label2id=eu.LABEL2ID,
            ignore_mismatched_sizes=False,
        )
        train_args = _make_training_arguments(
            TrainingArguments,
            output_dir=Path(temporary_dir) / "trainer_checkpoints",
            learning_rate=selected_lr,
            seed=int(audit["tuning_seed"]),
            use_fp16=use_cuda,
            disable_tqdm=disable_tqdm,
            evaluate=False,
        )
        trainer = Trainer(model=model, args=train_args, train_dataset=train_ds)
        trainer.train()
        elapsed = time.perf_counter() - start
        train_output_metrics = {k: float(v) for k, v in trainer.state.log_history[-1].items() if isinstance(v, (int, float))}
        del trainer, model
    result = {
        "schema_version": eu.CONFIG_SCHEMA_VERSION,
        "run_id": eu.RUN_ID,
        "stage": STAGE,
        "script_name": SCRIPT_NAME,
        "kind": "timing_only",
        "backbone": backbone,
        "model_id": eu.get_backbone(backbone).hf_model_id,
        "model_revision": audit["model_revisions"][backbone],
        "tokenizer_revision": audit["tokenizer_revisions"][backbone],
        "learning_rate": float(selected_lr),
        "tuning_seed": int(audit["tuning_seed"]),
        "context_size": PILOT_CONTEXT_SIZE,
        "max_length_context": int(context_length),
        "truncation_side": "left",
        "training_rows": int(len(cv_train)),
        "training_post_ids_sha256": _id_hash(cv_train),
        "fold_assignment_sha256": fold_audit["fold_assignment_sha256"],
        "matched_subset_sha256": audit["matched_sha256"],
        "fold_0_held_out_used": False,
        "validation_used": False,
        "elapsed_seconds": round(float(elapsed), 3),
        "trainer_last_log": train_output_metrics,
        "manifest_fingerprint": manifest["fingerprint"],
    }
    result_path = run_dir / "timing_result.json"
    _write_json_atomic(result_path, result)
    _write_json_atomic(run_dir / ARTIFACT_INDEX_NAME, {
        "schema_version": eu.CONFIG_SCHEMA_VERSION,
        "run_id": eu.RUN_ID,
        "stage": STAGE,
        "kind": "timing_only",
        "backbone": backbone,
        "manifest_fingerprint": manifest["fingerprint"],
        "sha256": {"timing_result.json": eu.sha256_file(result_path)},
    })
    eu.assert_current_run_artifact(
        result_path,
        expected_stage=STAGE,
        expected_script=SCRIPT_NAME,
        expected_fingerprint=manifest["fingerprint"],
    )
    result["_result_path"] = str(result_path)
    result["mixed_precision"] = bool(use_cuda)
    if use_cuda:
        torch.cuda.empty_cache()
    return result


def _build_context_learning_rate_document(
    original_lr_doc: Mapping[str, Any],
    scores_by_backbone: Mapping[str, Mapping[float, Mapping[str, Any]]],
    *,
    audit: Mapping[str, Any],
    matched_sha: str,
    fold_audit: Mapping[str, Any],
    context_length: int,
    tuning_train: pd.DataFrame,
    tuning_val: pd.DataFrame,
    itft_by_backbone: Mapping[str, Any],
    timing_by_backbone: Mapping[str, Any],
    latency_document: Mapping[str, Any],
) -> dict[str, Any]:
    if set(scores_by_backbone) != set(eu.BACKBONES):
        raise ValueError(f"Context LR scores must contain both registered backbones {sorted(eu.BACKBONES)}")
    selected: dict[str, float] = {}
    serialized: dict[str, dict[str, float]] = {}
    for backbone in eu.BACKBONES:
        if set(float(value) for value in scores_by_backbone[backbone]) != set(float(value) for value in eu.LR_SEARCH_GRID):
            raise ValueError(f"Context LR search for {backbone} must cover every candidate {eu.LR_SEARCH_GRID}")
        trial_scores = {float(lr): float(result["validation_macro_f1"]) for lr, result in scores_by_backbone[backbone].items()}
        selected[backbone] = eu.select_learning_rate(trial_scores, eu.LR_SEARCH_GRID)
        serialized[backbone] = {_lr_key(lr): trial_scores[lr] for lr in sorted(trial_scores)}

    updated = dict(original_lr_doc)
    updated["learning_rate_context"] = selected
    updated["context_trial_macro_f1"] = serialized
    updated["context_preflight"] = {
        "selection_rule": "highest validation macro-F1; exact ties go to the smaller learning rate",
        "trial_count_per_backbone": len(eu.LR_SEARCH_GRID),
        "total_trial_count": len(eu.LR_SEARCH_GRID) * len(eu.BACKBONES),
        "validation_metric": "macro_f1",
        "input_partition": "fold0_cv_training_portion_split_80_20",
        "fold_column": eu.CTX_FOLD_COLUMN,
        "excluded_cv_test_fold": PILOT_FOLD,
        "uses_fold0_held_out": False,
        "training_rows": int(len(tuning_train)),
        "validation_rows": int(len(tuning_val)),
        "training_post_ids_sha256": _id_hash(tuning_train),
        "validation_post_ids_sha256": _id_hash(tuning_val),
        "contextual_matched_csv_sha256": matched_sha,
        "fold_assignment_sha256": fold_audit["fold_assignment_sha256"],
        "max_length_context": int(context_length),
        "context_length_source_partition": "contextual_c4",
        "truncation_side": "left",
        "lr_tuning_seed": int(audit["tuning_seed"]),
        "model_ids": {b: eu.get_backbone(b).hf_model_id for b in eu.BACKBONES},
        "model_revisions": dict(audit["model_revisions"]),
        "tokenizer_revisions": dict(audit["tokenizer_revisions"]),
        "itft_manifest_fingerprints": {b: itft_by_backbone[b]["manifest_fingerprint"] for b in eu.BACKBONES},
        "timing_only_seconds": {b: float(timing_by_backbone[b]["elapsed_seconds"]) for b in eu.BACKBONES},
        "latency_measurement_machine": latency_document["measurement_machine"],
        "latency_measurement_repeats_per_backbone_context": int(latency_document["measurement_repeats_per_backbone_context"]),
        "latency_measurement_artifact": str(STAGE_ROOT / "summary" / LATENCY_MEASUREMENTS_NAME),
        "latency_median_ms_by_context": latency_document["median_ms_by_context"],
        "latency_measurement_manifest_fingerprints": dict(latency_document["itft_manifest_fingerprints"]),
        "benchmark_sequence_config_sha256_before_context_merge": audit["sequence_config_sha256_before_stage8"],
        "benchmark_learning_rate_config_sha256_before_context_merge": audit["learning_rate_config_sha256_before_stage8"],
        "hyperparameters": {
            "optimizer": OPTIMIZER_NAME,
            "weight_decay": WEIGHT_DECAY,
            "warmup_ratio": WARMUP_RATIO,
            "lr_scheduler": LR_SCHEDULER,
            "max_epochs": NUM_EPOCHS,
            "per_device_train_batch_size": TRAIN_BATCH_SIZE,
            "per_device_eval_batch_size": EVAL_BATCH_SIZE,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            "effective_batch_size": TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS,
            "loss": LOSS_NAME,
            "checkpoint_metric": CHECKPOINT_METRIC,
            "mixed_precision": bool(timing_by_backbone["indobertweet"].get("mixed_precision", False)),
        },
    }
    return updated


def _publish_shared_configs(
    *,
    selection: Mapping[str, Any],
    lr_doc: Mapping[str, Any],
    sequence_path: Path,
    lr_path: Path,
    sequence_metadata: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Publish the two shared config updates transactionally (best-effort rollback)."""
    original_sequence_bytes = sequence_path.read_bytes()
    original_lr_bytes = lr_path.read_bytes()
    try:
        eu.save_length_selection_config(selection, path=sequence_path)
        sequence_doc = eu.load_json_config(sequence_path)
        sequence_doc.update(dict(sequence_metadata))
        _write_json_atomic(sequence_path, sequence_doc)
        _write_json_atomic(lr_path, lr_doc)
        shared = eu.load_shared_run_config(
            sequence_length_path=sequence_path,
            learning_rate_path=lr_path,
            require_context=True,
            require_context_lr=True,
        )
        # The utility re-computes both selected values from every saved score.
        return shared["sequence_length"], shared["learning_rates"]
    except Exception:
        # Stage 8 must not leave only one of the two shared configs updated.
        sequence_path.write_bytes(original_sequence_bytes)
        lr_path.write_bytes(original_lr_bytes)
        raise


def _build_run_config_for_stage8_summary(
    audit: Mapping[str, Any],
    fold_audit: Mapping[str, Any],
    matched_sha: str,
    context_length: int,
    selected_lr: Mapping[str, float],
    *,
    mixed_precision: bool,
    tuning_train: pd.DataFrame,
    tuning_val: pd.DataFrame,
    latency_document: Mapping[str, Any],
) -> dict[str, Any]:
    cfg = _build_resolved_config(
        "indobertweet", audit, matched_sha, fold_audit, context_length,
        tuning_train, tuning_val, selected_lr["indobertweet"], mixed_precision=mixed_precision,
    )
    cfg.update({
        "experiment_id": "E2-contextual-pilot-summary",
        "kind": "contextual_pilot_summary",
        "learning_rate_context_by_backbone": {b: float(selected_lr[b]) for b in eu.BACKBONES},
        "latency_measurement_machine": latency_document["measurement_machine"],
        "latency_median_ms_by_context": latency_document["median_ms_by_context"],
        "latency_measurements_sha256": eu.sha256_json(latency_document["measurements_ms_by_context"]),
        "latency_sample_post_ids_sha256": latency_document["sample_post_ids_sha256"],
        "latency_measurement_repeats_per_backbone_context": int(latency_document["measurement_repeats_per_backbone_context"]),
        "contextual_lr_trial_count_per_backbone": len(eu.LR_SEARCH_GRID),
        "planned_cv_runs_per_backbone": PLANNED_RUNS_PER_BACKBONE,
        "sequence_length_config_sha256_before_stage8": audit["sequence_config_sha256_before_stage8"],
        "learning_rate_config_sha256_before_stage8": audit["learning_rate_config_sha256_before_stage8"],
    })
    eu.validate_resolved_run_config(
        cfg, stage=STAGE, backbone="indobertweet", input_mode=INPUT_MODE,
        context_size=PILOT_CONTEXT_SIZE, require_shared_config=False,
    )
    return cfg


def _build_summary_tables(
    *,
    token_stats: Mapping[str, Mapping[str, Any]],
    selection: Mapping[str, Any],
    trial_results: Mapping[str, Mapping[float, Mapping[str, Any]]],
    timing_results: Mapping[str, Mapping[str, Any]],
    itft_by_backbone: Mapping[str, Mapping[str, Any]],
    selected_lr: Mapping[str, float],
    latency_document: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    token_rows = []
    for backbone in eu.BACKBONES:
        row = {"backbone": backbone, "model_id": eu.get_backbone(backbone).hf_model_id}
        row.update(dict(token_stats[backbone]))
        row["p95_selected_length"] = selection["percentile_tokens"][backbone]
        row["truncation_rate_at_shared_max_length"] = selection["truncation_rate_at_max_length"][backbone]
        token_rows.append(row)
    lr_rows = []
    for backbone in eu.BACKBONES:
        for lr in sorted(float(x) for x in eu.LR_SEARCH_GRID):
            result = trial_results[backbone][lr]
            lr_rows.append({
                "run_id": eu.RUN_ID,
                "stage": STAGE,
                "backbone": backbone,
                "model_id": eu.get_backbone(backbone).hf_model_id,
                "model_revision": result["model_revision"],
                "tokenizer_revision": result["tokenizer_revision"],
                "learning_rate": lr,
                "selected": bool(np.isclose(lr, selected_lr[backbone])),
                "tuning_seed": int(result["tuning_seed"]),
                "training_rows": int(result["training_rows"]),
                "validation_rows": int(result["validation_rows"]),
                "validation_macro_f1": float(result["validation_macro_f1"]),
                "validation_accuracy": float(result["validation_accuracy"]),
                "validation_weighted_f1": float(result["validation_weighted_f1"]),
                "validation_f1_n": float(result["validation_f1_n"]),
                "validation_f1_al": float(result["validation_f1_al"]),
                "validation_f1_hs": float(result["validation_f1_hs"]),
                "elapsed_seconds": float(result["elapsed_seconds"]),
                "uses_fold0_held_out": False,
                "manifest_fingerprint": result["manifest_fingerprint"],
            })
    timing_rows = []
    combined_end_to_end_seconds = 0.0
    combined_grid_seconds = 0.0
    for backbone in eu.BACKBONES:
        timing = timing_results[backbone]
        itft_seconds = float(itft_by_backbone[backbone]["elapsed_seconds"])
        lr_tuning_seconds = sum(float(trial_results[backbone][lr]["elapsed_seconds"]) for lr in sorted(float(x) for x in eu.LR_SEARCH_GRID))
        pilot_seconds = float(timing["elapsed_seconds"])
        grid_seconds = pilot_seconds * PLANNED_RUNS_PER_BACKBONE
        end_to_end = itft_seconds + lr_tuning_seconds + pilot_seconds + grid_seconds
        combined_grid_seconds += grid_seconds
        combined_end_to_end_seconds += end_to_end
        timing_rows.append({
            "backbone": backbone,
            "model_id": eu.get_backbone(backbone).hf_model_id,
            "itft_seconds_paid_once": itft_seconds,
            "context_lr_tuning_seconds": lr_tuning_seconds,
            "timing_only_pilot_seconds": pilot_seconds,
            "estimated_seconds_per_cv_run": pilot_seconds,
            "planned_cv_runs_per_backbone": PLANNED_RUNS_PER_BACKBONE,
            "estimated_cv_grid_seconds": grid_seconds,
            "estimated_cv_grid_hours": grid_seconds / 3600,
            "estimated_end_to_end_seconds_including_itft_and_stage8": end_to_end,
            "estimated_end_to_end_hours_including_itft_and_stage8": end_to_end / 3600,
            "selection_context": "C4",
            "selected_learning_rate_context": float(selected_lr[backbone]),
            "latency_measurement_machine": latency_document["measurement_machine"],
            **{f"median_inference_latency_ms_C{c}": float(latency_document["median_ms_by_context"][f"C{c}"][backbone]) for c in range(5)},
            **{f"p95_inference_latency_ms_C{c}": float(latency_document["summary_by_context"][f"C{c}"]["backbones"][backbone]["p95_ms"]) for c in range(5)},
            "max_length_context": int(selection["max_length"]),
            "timing_run_used_held_out_fold": False,
            "estimate_note": "Planning estimate; excludes some tokenization, evaluation, transfer, and checkpoint I/O overhead.",
        })
    summary = {
        "schema_version": eu.CONFIG_SCHEMA_VERSION,
        "run_id": eu.RUN_ID,
        "stage": STAGE,
        "script_name": SCRIPT_NAME,
        "context_size_for_length_and_tuning": "C4",
        "length_source_partition": "contextual_c4",
        "tuning_partition": "fold != 0; stratified 80/20 by label",
        "fold_0_held_out_used": False,
        "max_length_context": int(selection["max_length"]),
        "length_percentile": eu.LENGTH_PERCENTILE,
        "length_percentile_tokens": dict(selection["percentile_tokens"]),
        "truncation_rate_at_max_length": dict(selection["truncation_rate_at_max_length"]),
        "contextual_learning_rates": {b: float(selected_lr[b]) for b in eu.BACKBONES},
        "latency_measurement_machine": latency_document["measurement_machine"],
        "latency_measurement_repeats_per_backbone_context": int(latency_document["measurement_repeats_per_backbone_context"]),
        "latency_median_ms_by_context": latency_document["median_ms_by_context"],
        "latency_summary_by_context": latency_document["summary_by_context"],
        "latency_measurements_sha256": eu.sha256_json(latency_document["measurements_ms_by_context"]),
        "latency_sample_post_ids_sha256": latency_document["sample_post_ids_sha256"],
        "trial_count_per_backbone": len(eu.LR_SEARCH_GRID),
        "total_context_lr_trials": len(eu.LR_SEARCH_GRID) * len(eu.BACKBONES),
        "cv_runs_per_backbone": PLANNED_RUNS_PER_BACKBONE,
        "estimated_future_cv_grid_hours_both_backbones": combined_grid_seconds / 3600,
        "estimated_end_to_end_hours_including_itft_and_stage8_both_backbones": combined_end_to_end_seconds / 3600,
        "itft_manifest_fingerprints": {b: itft_by_backbone[b]["manifest_fingerprint"] for b in eu.BACKBONES},
        "timing_only_manifest_fingerprints": {b: timing_results[b]["manifest_fingerprint"] for b in eu.BACKBONES},
        "interpretation_note": "Runtime values are planning estimates measured on the active machine; rerun timing pilot if the runtime/GPU, batch size, precision, or sequence length changes.",
    }
    return pd.DataFrame(token_rows), pd.DataFrame(lr_rows), pd.DataFrame(timing_rows), summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 8 contextual length selection, LR tuning and timing pilot.")
    parser.add_argument("--matched-csv", type=Path, default=MATCHED_SUBSET_PATH)
    parser.add_argument("--sequence-config", type=Path, default=SEQUENCE_CONFIG_PATH)
    parser.add_argument("--learning-rate-config", type=Path, default=LEARNING_RATE_CONFIG_PATH)
    parser.add_argument("--allow-cpu", action="store_true", help="Allow the eight LR trials, timing runs and latency measurements on CPU (very slow).")
    parser.add_argument("--disable-tqdm", action="store_true")
    parser.add_argument("--latency-repeats", type=int, default=LATENCY_REPEATS_DEFAULT, help="Single-post latency observations per backbone/context (minimum 20).")
    parser.add_argument("--latency-warmup", type=int, default=LATENCY_WARMUP_RUNS, help="Warm-up inference calls per backbone/context before recording latency.")
    parser.add_argument("--latency-machine", default=None, help="Optional explicit label for the common machine used for both backbones' latency measurements.")
    parser.add_argument("--validate-only", action="store_true", help="Validate data, configs and both ITFT checkpoints without loading tokenizers or training.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.latency_repeats < LATENCY_MIN_REPEATS:
        raise ValueError(f"--latency-repeats must be >= {LATENCY_MIN_REPEATS}")
    if args.latency_warmup < 1:
        raise ValueError("--latency-warmup must be >= 1")
    if args.latency_machine is not None and not args.latency_machine.strip():
        raise ValueError("--latency-machine cannot be an empty string")
    matched_path = _resolve(args.matched_csv)
    sequence_path = _resolve(args.sequence_config)
    lr_path = _resolve(args.learning_rate_config)
    if sequence_path != _resolve(SEQUENCE_CONFIG_PATH) or lr_path != _resolve(LEARNING_RATE_CONFIG_PATH):
        raise ValueError("Stage 8 must use the authoritative shared config paths from experiment_utils; custom config paths are unsupported.")
    _assert_no_legacy_stage8_outputs()
    audit = _load_pinned_benchmark_configs()
    matched, fold_audit, matched_sha = _load_matched_context_data(matched_path)
    cv_train, tuning_train, tuning_val = make_context_tuning_split(matched, audit["tuning_seed"])
    audit.update({
        "matched_sha256": matched_sha,
        "context_dataset_sha256": matched_sha,
        "fold_audit": fold_audit,
        "context_length_provenance": {
            "train_ids_sha256": _id_hash(tuning_train),
            "fold_assignment_sha256": fold_audit["fold_assignment_sha256"],
        },
        "paths": {
            **audit["paths"],
            "matched": matched_path,
            "sequence_config": sequence_path,
            "learning_rate_config": lr_path,
        },
    })
    itft_by_backbone = {backbone: _validate_itft_checkpoint(backbone, audit) for backbone in eu.BACKBONES}

    print(f"Run ID: {eu.RUN_ID}; authoritative stage: {STAGE}; script: {SCRIPT_NAME}")
    print(f"Matched subset rows: {len(matched)}; shared fold sizes: {fold_audit['fold_counts']}")
    print(f"Contextual LR split: fold != {PILOT_FOLD}, then stratified 80/20 => {len(tuning_train)} train / {len(tuning_val)} validation rows")
    print(f"Dedicated LR tuning seed: {audit['tuning_seed']}; fold-0 held-out examples: NOT USED")
    print(f"Benchmark max length: {audit['sequence_length_doc']['max_length_benchmark']}; source: benchmark_train")
    print(f"Model/tokenizer IDs and commits validated for: {', '.join(eu.BACKBONES)}")
    print("ITFT checkpoint manifests, metrics, input hashes, and all checkpoint file hashes verified.")

    if args.validate_only:
        migration = eu.audit_pipeline_script_names(PROJECT_ROOT, require_all=False)
        print("Stage 8 pre-run data/config/checkpoint audit passed. Tokenization and GPU trials were not run.")
        if migration["complete"]:
            print("Full pipeline script/directory audit: passed.")
        else:
            print("Full pipeline audit is not yet complete; actual Stage 8 training remains blocked until all scripts are migrated.")
            print(f"Missing scripts: {migration['missing']}")
            print(f"Missing directories: {migration['missing_directories']}")
            print(f"Old script names still present: {migration['rename_leftovers']}")
        return

    eu.audit_pipeline_script_names(PROJECT_ROOT, require_all=True)
    try:
        torch, AutoModel, AutoTokenizer, Trainer, TrainingArguments, set_seed = _load_training_dependencies()
    except ImportError:
        raise
    use_cuda = bool(torch.cuda.is_available())
    if not use_cuda and not args.allow_cpu:
        raise RuntimeError("No CUDA GPU detected. Use the configured Google Colab GPU runtime or pass --allow-cpu deliberately.")

    # Load both pinned tokenizers, assert their resolved commits, and measure
    # C4 lengths on the LR tuning TRAIN split only (not the 20% validation or
    # the fold-0 CV held-out partition).
    tokenizers: dict[str, Any] = {}
    length_vectors: dict[str, np.ndarray] = {}
    token_stats: dict[str, dict[str, Any]] = {}
    tokenizer_revision_resolved: dict[str, str] = {}
    for backbone, spec in eu.BACKBONES.items():
        revision = audit["tokenizer_revisions"][backbone]
        tokenizer = AutoTokenizer.from_pretrained(spec.hf_model_id, revision=revision, use_fast=True)
        eu.configure_tokenizer_for_input(tokenizer, INPUT_MODE)
        actual = getattr(tokenizer, "_commit_hash", None) or (getattr(tokenizer, "init_kwargs", {}) or {}).get("_commit_hash")
        if not actual or str(actual).lower() != revision.lower():
            raise RuntimeError(f"Tokenizer for {backbone} resolved to {actual!r}; expected pinned commit {revision!r}")
        tokenizer_revision_resolved[backbone] = str(actual).lower()
        texts = _context_texts(tuning_train, tokenizer)
        lengths = np.asarray([len(tokenizer(text, truncation=False, add_special_tokens=True)["input_ids"]) for text in texts], dtype=np.int64)
        length_vectors[backbone] = lengths
        token_stats[backbone] = _token_length_stats(lengths)
        tokenizers[backbone] = tokenizer
        print(f"{backbone}: tokenizer commit={actual}; C4 training input p95={token_stats[backbone]['p95']:.1f} tokens")

    selection = eu.choose_shared_max_length(
        length_vectors,
        percentile=eu.LENGTH_PERCENTILE,
        multiple_of=eu.LENGTH_ROUNDING_MULTIPLE,
        hard_cap=eu.MAX_SEQUENCE_LENGTH,
        partition="contextual_c4",
    )
    context_length = int(selection["max_length"])
    if selection["hard_cap_hit"]:
        print("WARNING: selected contextual max length reached the 512-token hard cap; inspect truncation rates before proceeding.")
    print(f"Selected shared max_length_context={context_length}; per-backbone truncation rates={selection['truncation_rate_at_max_length']}")

    Dataset = _make_dataset_class(torch)
    encoded_splits: dict[str, dict[str, Any]] = {}
    for backbone, tokenizer in tokenizers.items():
        train_texts = _context_texts(tuning_train, tokenizer)
        val_texts = _context_texts(tuning_val, tokenizer)
        train_enc = tokenizer(train_texts, truncation=True, max_length=context_length, padding="max_length", return_tensors="pt")
        val_enc = tokenizer(val_texts, truncation=True, max_length=context_length, padding="max_length", return_tensors="pt")
        encoded_splits[backbone] = {"train": train_enc, "validation": val_enc}

    # The effective batch size and all non-LR parameters must remain equivalent.
    parity_configs = {}
    for backbone in eu.BACKBONES:
        cfg = _build_resolved_config(
            backbone, audit, matched_sha, fold_audit, context_length,
            tuning_train, tuning_val, float(eu.LR_SEARCH_GRID[0]), mixed_precision=use_cuda,
        )
        eu.validate_resolved_run_config(
            cfg, stage=STAGE, backbone=backbone, input_mode=INPUT_MODE,
            context_size=PILOT_CONTEXT_SIZE, require_shared_config=False,
        )
        parity_configs[backbone] = cfg
    eu.check_config_parity(parity_configs["indobertweet"], parity_configs["xlmr"], input_mode=INPUT_MODE)

    scores_by_backbone: dict[str, dict[float, dict[str, Any]]] = {backbone: {} for backbone in eu.BACKBONES}
    for backbone in eu.BACKBONES:
        print(f"\n--- Stage 8 contextual LR search: {backbone} ---")
        for candidate_lr in eu.LR_SEARCH_GRID:
            result = _run_context_lr_trial(
                torch=torch, AutoModel=AutoModel, Trainer=Trainer, TrainingArguments=TrainingArguments,
                set_seed=set_seed, Dataset=Dataset, backbone=backbone, learning_rate=float(candidate_lr),
                tokenizer=tokenizers[backbone], tuning_train_enc=encoded_splits[backbone]["train"],
                tuning_val_enc=encoded_splits[backbone]["validation"], tuning_train=tuning_train,
                tuning_val=tuning_val, audit=audit, fold_audit=fold_audit, context_length=context_length,
                matched=matched, itft=itft_by_backbone[backbone], use_cuda=use_cuda,
                disable_tqdm=args.disable_tqdm,
            )
            scores_by_backbone[backbone][float(candidate_lr)] = result
            print(f"  lr={candidate_lr:g}: validation macro-F1={result['validation_macro_f1']:.4f} ({result['elapsed_seconds'] / 60:.2f} min)")
            if use_cuda:
                torch.cuda.empty_cache()

    selected_lr = {
        backbone: eu.select_learning_rate(
            {lr: float(result["validation_macro_f1"]) for lr, result in score_map.items()},
            eu.LR_SEARCH_GRID,
        )
        for backbone, score_map in scores_by_backbone.items()
    }
    print(f"\nSelected contextual learning rates: {selected_lr}")

    # Time one additional training-only C4 run on the full fold-0 CV training
    # portion for each backbone. Fold 0's held-out examples are never supplied.
    timing_by_backbone: dict[str, dict[str, Any]] = {}
    for backbone in eu.BACKBONES:
        timing_by_backbone[backbone] = _run_timing_only(
            torch=torch, AutoModel=AutoModel, Trainer=Trainer, TrainingArguments=TrainingArguments,
            set_seed=set_seed, Dataset=Dataset, backbone=backbone, tokenizer=tokenizers[backbone],
            cv_train=cv_train, audit=audit, fold_audit=fold_audit, matched=matched,
            context_length=context_length, selected_lr=selected_lr[backbone],
            trial_scores={lr: float(result["validation_macro_f1"]) for lr, result in scores_by_backbone[backbone].items()},
            itft=itft_by_backbone[backbone], use_cuda=use_cuda, disable_tqdm=args.disable_tqdm,
        )
        timing_by_backbone[backbone]["mixed_precision"] = bool(use_cuda)
        print(f"Timing-only {backbone}: {timing_by_backbone[backbone]['elapsed_seconds'] / 60:.2f} min for {len(cv_train)} rows")
        if use_cuda:
            torch.cuda.empty_cache()

    # Measure inference latency at every context depth. Stage 11 may select a
    # different C_best for each backbone, so a single C4-only latency value
    # would not match the pre-declared tie-break's deployed context size.
    latency_rows = _select_latency_rows(tuning_train, args.latency_repeats)
    latency_machine = _latency_machine_label(torch, args.latency_machine)
    latency_document = _measure_inference_latency_by_context(
        torch=torch, AutoModel=AutoModel, tokenizers=tokenizers,
        sample_rows=latency_rows, context_length=context_length,
        itft_by_backbone=itft_by_backbone, use_cuda=use_cuda,
        warmup_runs=args.latency_warmup, measurement_machine=latency_machine,
        matched_subset_path=matched_path,
    )
    print("\nMeasured single-post inference latency (median ms) by context:")
    for context in range(5):
        print(f"  C{context}: {latency_document['median_ms_by_context'][f'C{context}']}")

    lr_output = _build_context_learning_rate_document(
        audit["learning_rate_doc"], scores_by_backbone,
        audit=audit, matched_sha=matched_sha, fold_audit=fold_audit,
        context_length=context_length, tuning_train=tuning_train, tuning_val=tuning_val,
        itft_by_backbone=itft_by_backbone, timing_by_backbone=timing_by_backbone,
        latency_document=latency_document,
    )

    # Keep the exact pre-update configuration documents in memory. Publish
    # both shared files only after all 8 tuning trials and both timing runs pass.
    sequence_metadata = {
        "context_length_training_post_ids_sha256": _id_hash(tuning_train),
        "context_length_fold_assignment_sha256": fold_audit["fold_assignment_sha256"],
        "context_length_training_rows": int(len(tuning_train)),
        "context_length_validation_post_ids_sha256": _id_hash(tuning_val),
        "context_length_validation_rows": int(len(tuning_val)),
        "context_length_excluded_cv_fold": PILOT_FOLD,
        "context_length_input_policy": "C4 inputs from the 80% training split of rows with fold != 0; excludes tuning validation and fold-0 held-out examples",
        "context_tokenizer_model_ids": {b: eu.get_backbone(b).hf_model_id for b in eu.BACKBONES},
        "context_tokenizer_resolved_revisions": tokenizer_revision_resolved,
        "context_token_length_stats": token_stats,
        "context_matched_csv_sha256": matched_sha,
        "context_fold_assignment_sha256": fold_audit["fold_assignment_sha256"],
        "context_pilot_script": SCRIPT_NAME,
        "context_pilot_script_sha256": eu.sha256_file(_resolve(SCRIPT_NAME)),
        "context_pilot_utility_sha256": eu.sha256_file(_resolve("experiment_utils.py")),
        "context_pilot_manifest_fingerprints": {
            "lr_trials": {b: { _lr_key(lr): scores_by_backbone[b][lr]["manifest_fingerprint"] for lr in sorted(scores_by_backbone[b]) } for b in eu.BACKBONES},
            "timing_only": {b: timing_by_backbone[b]["manifest_fingerprint"] for b in eu.BACKBONES},
        },
        "contextual_learning_rates": {b: float(selected_lr[b]) for b in eu.BACKBONES},
    }

    # Prepare report tables before publishing the shared configuration. If a
    # filesystem write fails before publication, the configs remain benchmark-only.
    token_table, lr_table, timing_table, summary = _build_summary_tables(
        token_stats=token_stats, selection=selection, trial_results=scores_by_backbone,
        timing_results=timing_by_backbone, itft_by_backbone=itft_by_backbone,
        selected_lr=selected_lr, latency_document=latency_document,
    )
    summary.update({
        "matched_subset_sha256": matched_sha,
        "fold_assignment_sha256": fold_audit["fold_assignment_sha256"],
        "training_post_ids_sha256": _id_hash(tuning_train),
        "validation_post_ids_sha256": _id_hash(tuning_val),
        "benchmark_length_config_sha256_before_stage8": audit["sequence_config_sha256_before_stage8"],
        "benchmark_lr_config_sha256_before_stage8": audit["learning_rate_config_sha256_before_stage8"],
        "tokenizer_revisions": audit["tokenizer_revisions"],
        "model_revisions": audit["model_revisions"],
        "mixed_precision": use_cuda,
    })

    # Stage 8 aggregate record has a final configuration identity. Its input
    # manifest includes the frozen source data, the pre-Stage-8 configs, all
    # trial/timing receipts, and both verified ITFT checkpoint receipts.
    summary_cfg = _build_run_config_for_stage8_summary(
        audit, fold_audit, matched_sha, context_length, selected_lr,
        mixed_precision=use_cuda, tuning_train=tuning_train, tuning_val=tuning_val,
        latency_document=latency_document,
    )
    summary_cfg["context_lr_trial_macro_f1"] = {
        b: {_lr_key(lr): float(scores_by_backbone[b][lr]["validation_macro_f1"]) for lr in sorted(scores_by_backbone[b])}
        for b in eu.BACKBONES
    }
    summary_cfg["context_trial_manifest_fingerprints"] = {
        b: {_lr_key(lr): scores_by_backbone[b][lr]["manifest_fingerprint"] for lr in sorted(scores_by_backbone[b])}
        for b in eu.BACKBONES
    }
    summary_cfg["timing_only_seconds"] = {b: float(timing_by_backbone[b]["elapsed_seconds"]) for b in eu.BACKBONES}
    # Outputs are all beneath a dedicated versioned summary directory.
    summary_dir = STAGE_ROOT / "summary"
    summary_manifest = eu.make_run_manifest(
        stage=STAGE,
        script_name=SCRIPT_NAME,
        config=summary_cfg,
        input_paths=[
            matched_path,
            audit["paths"]["sequence_config"],
            audit["paths"]["learning_rate_config"],
            *[scores_by_backbone[b][lr]["_result_path"] for b in eu.BACKBONES for lr in sorted(scores_by_backbone[b])],
            *[timing_by_backbone[b]["_result_path"] for b in eu.BACKBONES],
            *[itft_by_backbone[b]["marker_path"] for b in eu.BACKBONES],
            *[itft_by_backbone[b]["model_manifest_path"] for b in eu.BACKBONES],
        ],
        metadata={
            "kind": "contextual_pilot_summary",
            "max_length_context": context_length,
            "selected_learning_rates": selected_lr,
            "length_partition": "contextual_c4",
            "fold0_heldout_used": False,
        },
        project_root=PROJECT_ROOT,
    )
    eu.prepare_run_directory(summary_dir, summary_manifest, allow_resume=True)
    allowed_summary_names = {
        eu.RUN_MANIFEST_NAME, TOKEN_LENGTH_STATS_NAME, LR_TRIALS_NAME,
        TIMING_SUMMARY_NAME, SUMMARY_JSON_NAME, LATENCY_MEASUREMENTS_NAME,
        ARTIFACT_INDEX_NAME, COMPLETION_MARKER_NAME,
        "stage8_publish_receipt.json",
    }
    unexpected = [p for p in summary_dir.iterdir() if p.is_file() and p.name not in allowed_summary_names]
    if unexpected:
        raise RuntimeError(f"Stage 8 summary contains unexpected files: {unexpected}; archive the summary directory")

    token_path = summary_dir / TOKEN_LENGTH_STATS_NAME
    lr_table_path = summary_dir / LR_TRIALS_NAME
    timing_path = summary_dir / TIMING_SUMMARY_NAME
    summary_path = summary_dir / SUMMARY_JSON_NAME
    _write_csv_atomic(token_path, token_table)
    _write_csv_atomic(lr_table_path, lr_table)
    _write_csv_atomic(timing_path, timing_table)
    _write_json_atomic(summary_path, summary)
    latency_path = summary_dir / LATENCY_MEASUREMENTS_NAME
    latency_document["manifest_fingerprint"] = summary_manifest["fingerprint"]
    _write_json_atomic(latency_path, latency_document)

    # Publish contextual fields to the two authoritative shared configs.
    _publish_shared_configs(
        selection=selection,
        lr_doc=lr_output,
        sequence_path=sequence_path,
        lr_path=lr_path,
        sequence_metadata=sequence_metadata,
    )
    final_sequence_sha = eu.sha256_file(sequence_path)
    final_lr_sha = eu.sha256_file(lr_path)
    publish_receipt = {
        "schema_version": eu.CONFIG_SCHEMA_VERSION,
        "run_id": eu.RUN_ID,
        "stage": STAGE,
        "script_name": SCRIPT_NAME,
        "summary_manifest_fingerprint": summary_manifest["fingerprint"],
        "sequence_length_config_sha256": final_sequence_sha,
        "learning_rate_config_sha256": final_lr_sha,
        "max_length_context": context_length,
        "learning_rate_context": selected_lr,
        "latency_measurements_file": str(latency_path.relative_to(PROJECT_ROOT)),
        "latency_measurements_sha256": eu.sha256_file(latency_path),
        "latency_measurement_machine": latency_document["measurement_machine"],
        "latency_median_ms_by_context": latency_document["median_ms_by_context"],
        "selected_context_lrs_validated": True,
    }
    _write_json_atomic(summary_dir / "stage8_publish_receipt.json", publish_receipt)

    # Hash all final summary artifacts except the index/marker being created.
    tracked_names = [TOKEN_LENGTH_STATS_NAME, LR_TRIALS_NAME, TIMING_SUMMARY_NAME, SUMMARY_JSON_NAME, LATENCY_MEASUREMENTS_NAME, "stage8_publish_receipt.json"]
    output_hashes = {name: eu.sha256_file(summary_dir / name) for name in tracked_names}
    _write_json_atomic(summary_dir / ARTIFACT_INDEX_NAME, {
        "schema_version": eu.CONFIG_SCHEMA_VERSION,
        "run_id": eu.RUN_ID,
        "stage": STAGE,
        "manifest_fingerprint": summary_manifest["fingerprint"],
        "sha256": output_hashes,
        "sequence_length_config_sha256": final_sequence_sha,
        "learning_rate_config_sha256": final_lr_sha,
    })
    _write_json_atomic(summary_dir / COMPLETION_MARKER_NAME, {
        "schema_version": eu.CONFIG_SCHEMA_VERSION,
        "run_id": eu.RUN_ID,
        "stage": STAGE,
        "script_name": SCRIPT_NAME,
        "manifest_fingerprint": summary_manifest["fingerprint"],
        "artifact_index_sha256": eu.sha256_file(summary_dir / ARTIFACT_INDEX_NAME),
        "sequence_length_config_sha256": final_sequence_sha,
        "learning_rate_config_sha256": final_lr_sha,
    })
    eu.assert_current_run_artifact(
        summary_dir / COMPLETION_MARKER_NAME,
        expected_stage=STAGE,
        expected_script=SCRIPT_NAME,
        expected_fingerprint=summary_manifest["fingerprint"],
    )
    # Reload and check the final shared configs once more before returning.
    final_shared = eu.load_shared_run_config(
        sequence_length_path=sequence_path,
        learning_rate_path=lr_path,
        require_context=True,
        require_context_lr=True,
    )
    if int(final_shared["sequence_length"]["max_length_context"]) != context_length:
        raise RuntimeError("Final sequence_length.json does not preserve the selected contextual maximum length")
    if final_shared["learning_rates"]["learning_rate_context"] != selected_lr:
        raise RuntimeError("Final learning_rates.json does not preserve the selected context rates")

    print("\nStage 8 completed successfully.")
    print(f"Shared max_length_context: {context_length}")
    print(f"Selected learning_rate_context: {selected_lr}")
    print(f"Contextual token statistics: {token_path.relative_to(PROJECT_ROOT)}")
    print(f"Context LR trials: {lr_table_path.relative_to(PROJECT_ROOT)}")
    print(f"Timing summary: {timing_path.relative_to(PROJECT_ROOT)}")
    print(f"Inference latency measurements (C0-C4): {latency_path.relative_to(PROJECT_ROOT)}")
    print(f"Stage 8 manifest: {(summary_dir / eu.RUN_MANIFEST_NAME).relative_to(PROJECT_ROOT)}")
    print(f"Stage 8 completion marker: {(summary_dir / COMPLETION_MARKER_NAME).relative_to(PROJECT_ROOT)}")
    print("Fold-0 held-out examples were not used for contextual length selection, LR tuning, training-time measurement, or inference latency sample selection.")


if __name__ == "__main__":
    main()
