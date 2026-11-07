"""Stage 5: evaluate the Experiment I Transformer backbone comparison.

Authoritative stage: 5 in experiment_utils.PIPELINE_SCRIPTS.

This stage compares E1-IndoBERTweet-Target with E1-XLMR-Target on the same
Ibrohim & Budi test partition. The SVM+TF-IDF script is retired/optional and
is deliberately excluded from the numbered Experiment I comparison.

Inputs are the five manifest-verified Stage 3 and Stage 4 runs under:
    results/runs/left_truncation_v2/experiment_i/indobertweet/seed_<seed>/
    results/runs/left_truncation_v2/experiment_i/xlmr/seed_<seed>/

The test partition's current row order and labels are checked against every
run. Per-seed metrics describe training stability; the head-to-head inferential
comparison uses seed-ensembled probabilities and a paired bootstrap (10,000
resamples, 95% CI), with the direction XLM-R minus IndoBERTweet.

Outputs are stored in the versioned Stage 5 directory:
    results/runs/left_truncation_v2/experiment_i/evaluation/
The output directory has its own manifest and artifact hash index. Historical
flat-path results/raw and results/tables files are never used as inputs or
resume signals.

Run from the repository root, normally on the local VS Code environment after
copying the complete Stage 3 and Stage 4 run directories back from Colab:
    python scripts/5_evaluate_e1.py --validate-only
    python scripts/5_evaluate_e1.py
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def _project_root() -> Path:
    here = Path(__file__).resolve().parent
    return here.parent if here.name.casefold() == "scripts" else here


PROJECT_ROOT = _project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import experiment_utils as eu  # noqa: E402

STAGE = 5
SCRIPT_NAME = "scripts/5_evaluate_e1.py"
INPUT_MODE = "target_only"
EXPECTED_TEST_ROWS = 1910
N_BOOTSTRAP = 10_000
CI_LEVEL = 95.0
BOOTSTRAP_SEED = 42

TEST_PATH = PROJECT_ROOT / "data/processed/ib_test.csv"
TRAIN_PATH = PROJECT_ROOT / "data/processed/ib_train.csv"
VALIDATION_PATH = PROJECT_ROOT / "data/processed/ib_val.csv"
CLASS_WEIGHTS_PATH = PROJECT_ROOT / "results/tables/ib_class_weights.csv"
SEQUENCE_LENGTH_PATH = PROJECT_ROOT / eu.SEQUENCE_LENGTH_CONFIG
LEARNING_RATE_PATH = PROJECT_ROOT / eu.LEARNING_RATE_CONFIG

# eu.load_shared_run_config() returns a mapping with the top-level keys
# "learning_rates", "run_id" and "sequence_length". Each entry below is
# (section of shared_config, key inside that section, key in the run config).
# A value is either shared by both backbones or a per-backbone mapping.
SHARED_CONFIG_COMPARISONS: tuple[tuple[str, str, str], ...] = (
    ("sequence_length", "max_length_benchmark", "max_length_benchmark"),
    ("learning_rates", "learning_rate_benchmark", "learning_rate_benchmark"),
)

# learning_rates.json -> "benchmark_preflight": (key there, key in the run config).
PREFLIGHT_HASH_KEYS: tuple[tuple[str, str], ...] = (
    ("benchmark_dataset_sha256", "dataset_sha256"),
    ("train_csv_sha256", "train_split_sha256"),
    ("validation_csv_sha256", "validation_split_sha256"),
    ("class_weights_csv_sha256", "class_weights_sha256"),
    ("split_assignment_sha256", "split_assignment_sha256"),
    ("sequence_length_config_sha256", "sequence_length_config_sha256"),
)
# learning_rates.json -> "benchmark_preflight" -> "hyperparameters".
PREFLIGHT_HYPERPARAMETER_KEYS: tuple[tuple[str, str], ...] = (
    ("checkpoint_metric", "checkpoint_metric"),
    ("early_stopping_patience", "early_stopping_patience"),
    ("effective_batch_size", "effective_batch_size"),
    ("gradient_accumulation_steps", "gradient_accumulation_steps"),
    ("loss", "loss"),
    ("lr_scheduler", "lr_scheduler"),
    ("max_epochs", "max_epochs"),
    ("mixed_precision", "mixed_precision"),
    ("optimizer", "optimizer"),
    ("per_device_eval_batch_size", "eval_batch_size"),
    ("per_device_train_batch_size", "per_device_batch_size"),
    ("warmup_ratio", "warmup_ratio"),
    ("weight_decay", "weight_decay"),
)
LEGACY_OUTPUTS = (
    "results/raw/e1_svm_test_predictions.csv",
    "results/raw/e1_transformer_seed*_predictions.csv",
    "results/raw/e1_transformer_seed*_probs.npy",
    "results/tables/e1_final_results.csv",
    "results/tables/e1_significance_test.csv",
    "results/tables/e1_transformer_confusion_matrix.csv",
)
OUTPUT_NAMES = (
    "e1_seed_metrics.csv",
    "e1_results.csv",
    "e1_ensemble_metrics.csv",
    "e1_paired_bootstrap.csv",
    "e1_confusion_matrix_indobertweet.csv",
    "e1_confusion_matrix_xlmr.csv",
    "e1_summary.json",
)


def _sha(path: Path) -> str:
    return eu.sha256_file(path)


def _resolve_legacy_outputs() -> list[Path]:
    found: list[Path] = []
    for pattern in LEGACY_OUTPUTS:
        found.extend(path for path in PROJECT_ROOT.glob(pattern) if path.exists())
    return list(dict.fromkeys(found))


def _write_json_atomic(path: Path, document: Mapping[str, Any]) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temp_path, path)


def validate_probability_array(probabilities: np.ndarray, *, expected_rows: int, name: str) -> np.ndarray:
    """Validate a persisted three-class softmax array and return float64 view."""
    values = np.asarray(probabilities)
    expected_shape = (expected_rows, len(eu.LABEL2ID))
    if values.shape != expected_shape:
        raise ValueError(f"{name} has shape {values.shape}; expected {expected_shape}")
    if not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all():
        raise ValueError(f"{name} contains non-numeric or non-finite probabilities")
    values = values.astype(np.float64, copy=False)
    if (values < -1e-6).any() or (values > 1.0 + 1e-6).any():
        raise ValueError(f"{name} has values outside [0, 1]")
    row_sums = values.sum(axis=1)
    if not np.allclose(row_sums, 1.0, rtol=1e-4, atol=1e-4):
        raise ValueError(f"{name} rows do not sum to one within tolerance")
    return values


def _labels_to_ids(values: pd.Series, *, description: str) -> np.ndarray:
    mapped = values.astype(str).map(eu.LABEL2ID)
    if mapped.isna().any():
        unexpected = sorted(values[mapped.isna()].astype(str).unique().tolist())
        raise ValueError(f"{description} contains non-canonical labels: {unexpected}")
    return mapped.to_numpy(dtype=np.int64)


def _load_test_partition() -> tuple[pd.DataFrame, np.ndarray, str]:
    if not TEST_PATH.is_file():
        raise FileNotFoundError(
            f"{TEST_PATH} is missing. Stage 5 requires the same processed ib_test.csv "+
            "used to train/evaluate Stages 3 and 4; transfer it from the canonical repository."
        )
    test_df = pd.read_csv(TEST_PATH)
    if len(test_df) != EXPECTED_TEST_ROWS:
        raise ValueError(
            f"ib_test.csv has {len(test_df)} rows; protocol expects {EXPECTED_TEST_ROWS}. "
            "Do not evaluate runs against a regenerated or different test split."
        )
    if not {"clean_text", "label"}.issubset(test_df.columns):
        raise KeyError("ib_test.csv must contain clean_text and label")
    if test_df["clean_text"].isna().any() or test_df["clean_text"].astype(str).str.strip().eq("").any():
        raise ValueError("ib_test.csv contains missing or blank clean_text values")
    if test_df["clean_text"].astype(str).duplicated().any():
        raise ValueError("ib_test.csv contains duplicate clean_text values")
    y_true = _labels_to_ids(test_df["label"], description="ib_test.csv label")
    if set(y_true.tolist()) != set(eu.ID2LABEL):
        raise ValueError("ib_test.csv must contain all three canonical classes N, AL and HS")
    return test_df, y_true, _sha(TEST_PATH)


def _validate_input_hashes(manifest: Mapping[str, Any], *, run_dir: Path) -> None:
    """Re-hash every recorded input used to produce an Experiment I run."""
    input_hashes = manifest.get("input_sha256")
    if not isinstance(input_hashes, Mapping) or not input_hashes:
        raise ValueError(f"Run manifest has no input SHA-256 map: {run_dir}")
    for stored_path, expected_hash in input_hashes.items():
        path = Path(str(stored_path))
        path = path if path.is_absolute() else PROJECT_ROOT / path
        if not path.is_file():
            raise FileNotFoundError(
                f"An input recorded by {run_dir} is unavailable: {stored_path}. "
                "Keep the canonical prepared splits and configs in the repository."
            )
        actual = _sha(path)
        if actual != expected_hash:
            raise ValueError(
                f"Input hash mismatch for {stored_path} in {run_dir}: "
                f"expected {expected_hash}, observed {actual}. Re-run the producing stage."
            )


def _verify_artifact_index(run_dir: Path, *, stage: int, script_name: str, fingerprint: str) -> dict[str, Any]:
    index_path = run_dir / "artifacts_sha256.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"Missing artifact hash index: {index_path}")
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unreadable artifact hash index: {index_path}") from exc
    expected_header = {
        "schema_version": eu.CONFIG_SCHEMA_VERSION,
        "run_id": eu.RUN_ID,
        "stage": stage,
        "script_name": script_name,
        "manifest_fingerprint": fingerprint,
    }
    for key, expected in expected_header.items():
        if index.get(key) != expected:
            raise ValueError(f"{index_path}: {key}={index.get(key)!r}, expected {expected!r}")
    hashes = index.get("sha256")
    if not isinstance(hashes, Mapping):
        raise ValueError(f"{index_path} does not contain a sha256 mapping")
    for filename in ("predictions.csv", "probabilities.npy", "metrics.json"):
        path = run_dir / filename
        expected = hashes.get(filename)
        if not path.is_file() or not isinstance(expected, str):
            raise FileNotFoundError(f"Missing output/hash entry: {path}")
        actual = _sha(path)
        if actual != expected:
            raise ValueError(f"Output hash mismatch for {path}; output may have been modified")
    return index


def _read_json_object(path: Path) -> dict[str, Any]:
    """Read a JSON file whose top level must be an object."""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unreadable JSON file: {path}") from exc
    if not isinstance(document, dict):
        raise ValueError(f"{path} must contain a JSON object at the top level")
    return document


def _values_match(observed: Any, expected: Any) -> bool:
    """Equality that compares floats with the same tolerance as the metric checks."""
    if isinstance(observed, float) or isinstance(expected, float):
        try:
            return math.isclose(float(observed), float(expected), rel_tol=0.0, abs_tol=1e-12)
        except (TypeError, ValueError):
            return False
    return bool(observed == expected)


def _backbone_value(value: Any, *, name: str, backbone: str) -> Any:
    """Return value, or its entry for backbone when it is a per-backbone mapping."""
    if isinstance(value, Mapping):
        if backbone not in value:
            raise KeyError(
                f"{name} has no entry for backbone {backbone!r}; "
                f"available: {sorted(str(key) for key in value)}"
            )
        return value[backbone]
    return value


def _select_learning_rate(trials: Any, *, backbone: str) -> float:
    """Re-apply the documented rule: highest validation macro-F1, exact ties to the smaller rate."""
    if not isinstance(trials, Mapping) or not trials:
        raise ValueError(f"{backbone}: benchmark_trial_macro_f1 has no learning-rate trials")
    scored: list[tuple[float, float]] = []
    for rate_text, score in trials.items():
        try:
            scored.append((float(rate_text), float(score)))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{backbone}: non-numeric learning-rate trial {rate_text!r}: {score!r}") from exc
    best_rate, _ = min(scored, key=lambda pair: (-pair[1], pair[0]))
    return best_rate


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def _validate_run_config_against_current_inputs(
    config: Mapping[str, Any],
    *,
    backbone: str,
    test_sha256: str,
    shared_config: Mapping[str, Any],
) -> None:
    """Check one run's recorded config against the inputs that are current now.

    _validate_input_hashes() already re-hashes every file listed in the
    manifest's input_sha256 map. This function adds independent checks of the
    run *config* against the files as they are now:

    1. Hashes recorded in the config match the current files, so the config
       and input_sha256 cannot disagree with each other.
    2. The config agrees with the contents of sequence_length.json (schema and
       run ID, max length, training-split hash, target-only truncation side,
       this backbone's tokenizer model ID and resolved revision).
    3. The config agrees with the contents of learning_rates.json (this
       backbone's learning rate, which must also be the best trial under the
       documented selection rule; search grid; tuning seed).
    4. The tuning preflight recorded in learning_rates.json used the same
       data hashes, model revisions, hyperparameters, max length and truncation
       side as the run, and never used the test split.
    5. The values in shared_config (the loader's view of both files) equal the
       run's recorded values.
    """
    if not isinstance(shared_config, Mapping):
        raise TypeError(
            "shared_config must be a Mapping[str, Any] but eu.load_shared_run_config() "
            f"returned {type(shared_config).__name__}; update the annotation and "
            "step 5 of this function to match its actual return type."
        )

    # 1. Hashes recorded in the config versus the files on disk now.
    if config.get("test_split_sha256") != test_sha256:
        raise ValueError(f"{backbone}: run config test_split_sha256 does not match the current ib_test.csv")
    current_files = {
        "train_split_sha256": TRAIN_PATH,
        "validation_split_sha256": VALIDATION_PATH,
        "class_weights_sha256": CLASS_WEIGHTS_PATH,
        "sequence_length_config_sha256": SEQUENCE_LENGTH_PATH,
        "learning_rate_config_sha256": LEARNING_RATE_PATH,
    }
    for key, path in current_files.items():
        if not path.is_file():
            raise FileNotFoundError(f"{backbone}: current input for {key} is missing: {_display_path(path)}")
        recorded = config.get(key)
        observed = _sha(path)
        if recorded != observed:
            raise ValueError(
                f"{backbone}: run config {key} does not match current {_display_path(path)}: "
                f"recorded {recorded!r}, observed {observed}. If only line endings differ, "
                "normalise the file to LF; otherwise re-run the producing stage."
            )

    # 2. Run config versus the contents of the current sequence-length config.
    sequence = _read_json_object(SEQUENCE_LENGTH_PATH)
    if sequence.get("schema_version") != eu.CONFIG_SCHEMA_VERSION or sequence.get("run_id") != eu.RUN_ID:
        raise ValueError(
            f"{_display_path(SEQUENCE_LENGTH_PATH)} has schema/run ID "
            f"{sequence.get('schema_version')!r}/{sequence.get('run_id')!r}; "
            f"expected {eu.CONFIG_SCHEMA_VERSION!r}/{eu.RUN_ID!r}"
        )
    if sequence.get("max_length_benchmark") != config.get("max_length_benchmark"):
        raise ValueError(
            f"{backbone}: run max_length_benchmark={config.get('max_length_benchmark')!r} but the current "
            f"sequence-length config selects {sequence.get('max_length_benchmark')!r}"
        )
    if sequence.get("benchmark_training_split_sha256") != config.get("train_split_sha256"):
        raise ValueError(
            f"{backbone}: the sequence length was selected on a different training split "
            "than the one this run was trained on"
        )
    if sequence.get("target_only_truncation_side") != config.get("truncation_side"):
        raise ValueError(
            f"{backbone}: run truncation_side={config.get('truncation_side')!r} but the sequence-length "
            f"config specifies target_only_truncation_side={sequence.get('target_only_truncation_side')!r}"
        )
    tokenizer_ids = sequence.get("tokenizer_model_ids")
    tokenizer_revisions = sequence.get("tokenizer_resolved_revisions")
    for label, table in (("tokenizer_model_ids", tokenizer_ids), ("tokenizer_resolved_revisions", tokenizer_revisions)):
        if not isinstance(table, Mapping) or backbone not in table:
            raise ValueError(f"{_display_path(SEQUENCE_LENGTH_PATH)}: {label} has no entry for {backbone!r}")
    if tokenizer_ids[backbone] != config.get("hf_model_id"):
        raise ValueError(
            f"{backbone}: run hf_model_id={config.get('hf_model_id')!r} but the sequence length was "
            f"selected with tokenizer {tokenizer_ids[backbone]!r}"
        )
    if tokenizer_revisions[backbone] != config.get("tokenizer_revision"):
        raise ValueError(
            f"{backbone}: run tokenizer_revision={config.get('tokenizer_revision')!r} but the sequence length "
            f"was selected with revision {tokenizer_revisions[backbone]!r}"
        )

    # 3. Run config versus the contents of the current learning-rate config.
    rates = _read_json_object(LEARNING_RATE_PATH)
    rates_name = _display_path(LEARNING_RATE_PATH)
    if rates.get("schema_version") != eu.CONFIG_SCHEMA_VERSION or rates.get("run_id") != eu.RUN_ID:
        raise ValueError(
            f"{rates_name} has schema/run ID {rates.get('schema_version')!r}/{rates.get('run_id')!r}; "
            f"expected {eu.CONFIG_SCHEMA_VERSION!r}/{eu.RUN_ID!r}"
        )
    learning_rate = config.get("learning_rate_benchmark")
    if isinstance(learning_rate, bool) or not isinstance(learning_rate, (int, float)):
        raise ValueError(f"{backbone}: run config learning_rate_benchmark is missing or non-numeric")
    selected_rate = _backbone_value(
        rates.get("learning_rate_benchmark"), name=f"{rates_name} learning_rate_benchmark", backbone=backbone,
    )
    if not _values_match(learning_rate, selected_rate):
        raise ValueError(
            f"{backbone}: run learning_rate_benchmark={learning_rate!r} but {rates_name} selects {selected_rate!r}"
        )
    trials = rates.get("benchmark_trial_macro_f1")
    if not isinstance(trials, Mapping) or backbone not in trials:
        raise ValueError(f"{rates_name}: benchmark_trial_macro_f1 has no entry for {backbone!r}")
    rederived_rate = _select_learning_rate(trials[backbone], backbone=backbone)
    if not _values_match(selected_rate, rederived_rate):
        raise ValueError(
            f"{backbone}: {rates_name} selects {selected_rate!r}, but its own trials select {rederived_rate!r} "
            "under 'highest validation macro-F1; exact ties go to the smaller learning rate'"
        )
    grid = config.get("lr_search_grid")
    file_grid = rates.get("lr_search_grid")
    if not isinstance(grid, list) or not isinstance(file_grid, list) or len(grid) != len(file_grid) or not all(
        _values_match(recorded_rate, file_rate) for recorded_rate, file_rate in zip(grid, file_grid)
    ):
        raise ValueError(f"{backbone}: run lr_search_grid={grid!r} differs from {rates_name} grid {file_grid!r}")
    trial_rates = sorted(float(rate_text) for rate_text in trials[backbone])
    if len(trial_rates) != len(grid) or not all(
        _values_match(trial_rate, grid_rate) for trial_rate, grid_rate in zip(trial_rates, sorted(float(r) for r in grid))
    ):
        raise ValueError(f"{backbone}: the trials in {rates_name} do not cover exactly the search grid {grid!r}")
    if config.get("lr_tuning_seed") != rates.get("lr_tuning_seed"):
        raise ValueError(
            f"{backbone}: run lr_tuning_seed={config.get('lr_tuning_seed')!r} but {rates_name} used "
            f"{rates.get('lr_tuning_seed')!r}"
        )

    # 4. The tuning preflight recorded alongside the learning rates.
    preflight = rates.get("benchmark_preflight")
    if not isinstance(preflight, Mapping):
        raise ValueError(f"{rates_name} has no benchmark_preflight section")
    if preflight.get("uses_test_split") is not False:
        raise ValueError(f"{rates_name}: benchmark_preflight must record uses_test_split=false")
    for preflight_key, config_key in PREFLIGHT_HASH_KEYS:
        if preflight.get(preflight_key) != config.get(config_key):
            raise ValueError(
                f"{backbone}: {rates_name} benchmark_preflight.{preflight_key} differs from the run's {config_key}; "
                "the learning rate was tuned on different data or configuration than this run used"
            )
    hyperparameters = preflight.get("hyperparameters")
    if not isinstance(hyperparameters, Mapping):
        raise ValueError(f"{rates_name}: benchmark_preflight has no hyperparameters section")
    for preflight_key, config_key in PREFLIGHT_HYPERPARAMETER_KEYS:
        if preflight_key not in hyperparameters or config_key not in config:
            raise ValueError(
                f"{backbone}: cannot compare hyperparameter {preflight_key!r} (preflight) with {config_key!r} (run config); "
                "one of them is missing"
            )
        if not _values_match(config[config_key], hyperparameters[preflight_key]):
            raise ValueError(
                f"{backbone}: run {config_key}={config[config_key]!r} but the tuning preflight used "
                f"{preflight_key}={hyperparameters[preflight_key]!r}"
            )
    for table_key, config_key in (("model_ids", "hf_model_id"), ("model_revisions", "model_revision")):
        preflight_value = _backbone_value(
            preflight.get(table_key), name=f"{rates_name} benchmark_preflight.{table_key}", backbone=backbone,
        )
        if preflight_value != config.get(config_key):
            raise ValueError(
                f"{backbone}: run {config_key}={config.get(config_key)!r} but the tuning preflight used {preflight_value!r}"
            )
    for key in ("max_length_benchmark", "truncation_side"):
        if preflight.get(key) != config.get(key):
            raise ValueError(
                f"{backbone}: run {key}={config.get(key)!r} but the tuning preflight used {preflight.get(key)!r}"
            )

    # 5. Cross-check the loader's view (shared_config) of both files.
    if shared_config.get("run_id") != eu.RUN_ID:
        raise ValueError(f"shared_config run_id={shared_config.get('run_id')!r}, expected {eu.RUN_ID!r}")
    for section_name, key, config_key in SHARED_CONFIG_COMPARISONS:
        section = shared_config.get(section_name)
        if not isinstance(section, Mapping) or key not in section:
            available = sorted(str(name) for name in section) if isinstance(section, Mapping) else type(section).__name__
            raise ValueError(
                f"shared_config[{section_name!r}] does not provide {key!r}; found {available}. "
                "Update SHARED_CONFIG_COMPARISONS to the names returned by eu.load_shared_run_config()."
            )
        expected = _backbone_value(
            section[key], name=f"shared_config[{section_name!r}][{key!r}]", backbone=backbone,
        )
        if not _values_match(config.get(config_key), expected):
            raise ValueError(
                f"{backbone}: run config {config_key}={config.get(config_key)!r} differs from the current "
                f"shared configuration value {expected!r}"
            )


def _load_seed_run(
    *, backbone: str,
    seed: int,
    test_df: pd.DataFrame,
    y_true_test: np.ndarray,
    test_sha256: str,
    shared_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Load a current, hash-verified, row-aligned Stage 3 or Stage 4 run."""
    stage = 3 if backbone == "indobertweet" else 4
    script_name = eu.PIPELINE_SCRIPTS[stage]
    run_dir = eu.versioned_run_root(PROJECT_ROOT) / "experiment_i" / backbone / f"seed_{seed}"
    pred_path = run_dir / "predictions.csv"
    probs_path = run_dir / "probabilities.npy"
    metrics_path = run_dir / "metrics.json"
    manifest_path = run_dir / eu.RUN_MANIFEST_NAME
    required = (pred_path, probs_path, metrics_path, manifest_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Stage {stage} run for {backbone}, seed {seed} is incomplete: {missing}. "
            "Copy the complete versioned run directory from Colab; do not use historical flat outputs."
        )

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unreadable manifest or metrics JSON in {run_dir}") from exc

    # Confirms manifest integrity and that the producing utility, script and
    # frozen protocol/pipeline files exactly match this evaluation repository.
    fingerprint_from_metrics = metrics.get("run_manifest_fingerprint")
    if not isinstance(fingerprint_from_metrics, str):
        raise ValueError(f"Missing run_manifest_fingerprint in {metrics_path}")
    eu.assert_current_run_artifact(
        pred_path,
        expected_stage=stage,
        expected_script=script_name,
        expected_fingerprint=fingerprint_from_metrics,
    )
    if manifest.get("fingerprint") != fingerprint_from_metrics:
        raise ValueError(f"Metrics and run manifest fingerprints differ for {run_dir}")
    _validate_input_hashes(manifest, run_dir=run_dir)
    _verify_artifact_index(run_dir, stage=stage, script_name=script_name, fingerprint=fingerprint_from_metrics)

    config = manifest.get("config")
    if not isinstance(config, Mapping):
        raise ValueError(f"Run manifest config is missing or invalid in {manifest_path}")
    expected_identity = {
        "schema_version": eu.CONFIG_SCHEMA_VERSION,
        "run_id": eu.RUN_ID,
        "stage": stage,
        "script_name": script_name,
        "backbone": backbone,
        "experiment_id": eu.experiment1_id(backbone),
        "hf_model_id": eu.get_backbone(backbone).hf_model_id,
        "seed": seed,
        "input_mode": INPUT_MODE,
        "truncation_side": "right",
        "max_length_benchmark": config.get("max_length_benchmark"),
    }
    for key, expected in expected_identity.items():
        if config.get(key) != expected:
            raise ValueError(f"Run config mismatch for {run_dir}: {key}={config.get(key)!r}, expected {expected!r}")
    _validate_run_config_against_current_inputs(
        config, backbone=backbone, test_sha256=test_sha256, shared_config=shared_config,
    )
    if tuple(config.get("eval_seeds_exp1", ())) != tuple(eu.EVAL_SEEDS_EXP1):
        raise ValueError(f"Unexpected Experiment I seed list in {manifest_path}")
    if config.get("test_split_sha256") != test_sha256:
        raise ValueError(f"Run config test split hash does not match current ib_test.csv in {run_dir}")
    if metrics.get("schema_version") != eu.CONFIG_SCHEMA_VERSION or metrics.get("run_id") != eu.RUN_ID:
        raise ValueError(f"Metrics JSON has incompatible schema/run ID in {metrics_path}")
    for key, value in (
        ("stage", stage),
        ("script_name", script_name),
        ("experiment_id", eu.experiment1_id(backbone)),
        ("backbone", backbone),
        ("seed", seed),
        ("model_revision", config.get("model_revision")),
        ("tokenizer_revision", config.get("tokenizer_revision")),
        ("max_length_benchmark", config.get("max_length_benchmark")),
        ("truncation_side", "right"),
        ("learning_rate_benchmark", config.get("learning_rate_benchmark")),
        ("n_test", len(test_df)),
        ("test_row_order_sha256", eu.sha256_json(np.arange(len(test_df), dtype=np.int64).tolist())),
    ):
        observed = metrics.get(key)
        if isinstance(value, float):
            if observed is None or not np.isclose(float(observed), value, rtol=0, atol=1e-12):
                raise ValueError(f"Metrics/config mismatch for {key} in {metrics_path}: {observed!r} vs {value!r}")
        elif observed != value:
            raise ValueError(f"Metrics/config mismatch for {key} in {metrics_path}: {observed!r} vs {value!r}")

    predictions = pd.read_csv(pred_path)
    if not {"row_id", "y_true", "y_pred"}.issubset(predictions.columns):
        raise KeyError(f"{pred_path} must contain row_id, y_true and y_pred")
    n_test = len(test_df)
    if len(predictions) != n_test:
        raise ValueError(f"{pred_path} has {len(predictions)} rows; expected {n_test}")
    expected_row_ids = np.arange(n_test, dtype=np.int64)
    row_ids = pd.to_numeric(predictions["row_id"], errors="coerce").to_numpy()
    if not np.array_equal(row_ids, expected_row_ids):
        raise ValueError(f"{pred_path} row_id must be exactly 0..{n_test - 1} in test-file order")
    y_true_saved = _labels_to_ids(predictions["y_true"], description=f"{pred_path} y_true")
    y_pred_saved = _labels_to_ids(predictions["y_pred"], description=f"{pred_path} y_pred")
    if not np.array_equal(y_true_saved, y_true_test):
        raise ValueError(f"Saved y_true in {pred_path} does not match the current ib_test.csv row order")

    probabilities = np.load(probs_path, allow_pickle=False)
    probabilities = validate_probability_array(probabilities, expected_rows=n_test, name=str(probs_path))
    y_pred_probs = probabilities.argmax(axis=1).astype(np.int64)
    if not np.array_equal(y_pred_saved, y_pred_probs):
        raise ValueError(f"y_pred column and probability argmax disagree in {pred_path}")

    # Recompute rather than blindly trust the metrics JSON; saved values are
    # checked against probabilities and the canonical held-out labels.
    recomputed = _prediction_metrics(y_true_test, y_pred_probs)
    saved_test_metrics = metrics.get("test_metrics")
    if not isinstance(saved_test_metrics, Mapping):
        raise ValueError(f"Missing test_metrics in {metrics_path}")
    # Stage 3/4 store the per-class scores with lower-case class names (f1_n,
    # f1_al, f1_hs) while the recomputation uses the canonical N/AL/HS labels
    # (f1_N, ...). Compare case-insensitively; the three names stay distinct.
    saved_lookup = {str(name).casefold(): value for name, value in saved_test_metrics.items()}
    for key, expected in recomputed.items():
        observed = saved_lookup.get(key.casefold())
        if observed is None:
            raise ValueError(
                f"Saved test metric {key!r} is missing from {metrics_path}; "
                f"saved keys: {sorted(str(name) for name in saved_test_metrics)}; "
                f"recomputed keys: {sorted(recomputed)}"
            )
        if not np.isclose(float(observed), expected, rtol=0, atol=1e-6):
            raise ValueError(
                f"Saved test metric {key} differs from recomputation in {metrics_path}: "
                f"saved {float(observed)!r}, recomputed {expected!r}, "
                f"absolute difference {abs(float(observed) - expected):.3g}"
            )

    return {
        "backbone": backbone,
        "seed": seed,
        "run_dir": run_dir,
        "manifest": manifest,
        "metrics": metrics,
        "predictions": predictions,
        "probabilities": probabilities,
        "y_pred": y_pred_probs,
        "test_sha256": test_sha256,
        "run_manifest_fingerprint": fingerprint_from_metrics,
    }


def _prediction_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=[0, 1, 2], average="weighted", zero_division=0)),
        **{f"f1_{label}": score for label, score in eu.per_class_f1(y_true, y_pred).items()},
    }


def _build_result_tables(
    y_true: np.ndarray,
    runs: Mapping[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    seeds = list(eu.EVAL_SEEDS_EXP1)
    backbones = ("indobertweet", "xlmr")
    probabilities = {key: [item["probabilities"] for item in runs[key]] for key in backbones}

    for backbone in backbones:
        if [item["seed"] for item in runs[backbone]] != seeds:
            raise ValueError(f"{backbone}: loaded runs are not in the canonical Experiment I seed order")

    comparison = eu.compare_backbones_exp1(
        y_true,
        probabilities["indobertweet"],
        probabilities["xlmr"],
        name_a="indobertweet",
        name_b="xlmr",
        seeds=seeds,
        n_bootstrap=N_BOOTSTRAP,
        ci=CI_LEVEL,
        random_state=BOOTSTRAP_SEED,
    )

    seed_tables: list[pd.DataFrame] = []
    result_rows: list[dict[str, Any]] = []
    ensemble_rows: list[dict[str, Any]] = []
    confusion_matrices: dict[str, pd.DataFrame] = {}

    for backbone in backbones:
        table = comparison["per_seed"][backbone].copy()
        table.insert(0, "experiment_id", eu.experiment1_id(backbone))
        table.insert(0, "backbone", backbone)
        seed_tables.append(table)

        summary = comparison["per_seed_summary"][backbone]
        result_row: dict[str, Any] = {
            "backbone": backbone,
            "experiment_id": eu.experiment1_id(backbone),
            "model_id": eu.get_backbone(backbone).hf_model_id,
            "n_seeds": len(seeds),
        }
        for metric in table.columns:
            if metric in {"backbone", "experiment_id", "seed"}:
                continue
            result_row[f"{metric}_mean"] = float(summary.loc[metric, "mean"])
            result_row[f"{metric}_std"] = float(summary.loc[metric, "std"])
        result_rows.append(result_row)

        mean_probs = probabilities[backbone][0].copy()
        for array in probabilities[backbone][1:]:
            mean_probs += array
        mean_probs /= len(probabilities[backbone])
        ensemble_pred = eu.ensemble_seed_probabilities(probabilities[backbone])
        if not np.array_equal(ensemble_pred, mean_probs.argmax(axis=1)):
            raise AssertionError("Utility ensemble prediction disagrees with mean-probability argmax")
        ensemble_row = {
            "backbone": backbone,
            "experiment_id": eu.experiment1_id(backbone),
            **_prediction_metrics(y_true, ensemble_pred),
        }
        ensemble_rows.append(ensemble_row)
        cm = confusion_matrix(y_true, ensemble_pred, labels=[0, 1, 2])
        cm_df = pd.DataFrame(cm, index=[eu.ID2LABEL[i] for i in range(3)], columns=[eu.ID2LABEL[i] for i in range(3)])
        cm_df.index.name = "true_label"
        cm_df.columns.name = "predicted_label"
        confusion_matrices[f"e1_confusion_matrix_{backbone}.csv"] = cm_df

    bootstrap = comparison["bootstrap"]
    bootstrap_rows: list[dict[str, Any]] = []
    for metric_name, result in [("macro_f1", bootstrap["macro_f1"])] + [
        (f"f1_{label}", info) for label, info in bootstrap["per_class"].items()
    ]:
        bootstrap_rows.append({
            "comparison": "E1-XLMR-Target minus E1-IndoBERTweet-Target",
            "metric": metric_name,
            **result,
            "inferential_status": "primary" if metric_name == "macro_f1" else "descriptive_unadjusted",
        })

    return {
        "seed_metrics": pd.concat(seed_tables, ignore_index=True),
        "results": pd.DataFrame(result_rows),
        "ensemble_metrics": pd.DataFrame(ensemble_rows),
        "paired_bootstrap": pd.DataFrame(bootstrap_rows),
        "confusion_matrices": confusion_matrices,
        "comparison": comparison,
    }


def _verify_existing_outputs(output_dir: Path, manifest: Mapping[str, Any]) -> bool:
    """Verify that an existing Stage 5 result set is complete and unchanged."""
    index_path = output_dir / "artifacts_sha256.json"
    output_files = [output_dir / name for name in OUTPUT_NAMES]
    exists_any_output = any(path.exists() for path in output_files) or index_path.exists()
    if not exists_any_output:
        return False
    if not index_path.is_file():
        raise RuntimeError(
            f"Partial/unindexed Stage 5 outputs exist in {output_dir}; archive this directory rather than reusing them."
        )
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unreadable Stage 5 output hash index: {index_path}") from exc
    if index.get("run_id") != eu.RUN_ID or index.get("stage") != STAGE or index.get("script_name") != SCRIPT_NAME:
        raise ValueError(f"Stage 5 output hash index has a stale run/stage/script identity: {index_path}")
    if index.get("manifest_fingerprint") != manifest.get("fingerprint"):
        raise RuntimeError("Stage 5 results were produced from a different run fingerprint; archive them before recalculating")
    hashes = index.get("sha256")
    if not isinstance(hashes, Mapping):
        raise ValueError("Stage 5 output hash index lacks a sha256 mapping")
    for path in output_files:
        expected = hashes.get(path.name)
        if not path.is_file() or not isinstance(expected, str) or _sha(path) != expected:
            raise RuntimeError(f"Stage 5 output is missing or changed: {path}")
    eu.assert_current_run_artifact(
        output_dir / "e1_results.csv", expected_stage=STAGE, expected_script=SCRIPT_NAME,
        expected_fingerprint=str(manifest["fingerprint"]),
    )
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the paired Experiment I IndoBERTweet versus XLM-R comparison.")
    parser.add_argument(
        "--validate-only", action="store_true",
        help="Validate current Stage 3/4 artifacts, hashes and parity without writing Stage 5 outputs.",
    )
    parser.add_argument(
        "--allow-incomplete-pipeline", action="store_true",
        help="For isolated testing only; skip the full pipeline-name audit. Does not bypass artifact/config checks.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if SCRIPT_NAME != eu.PIPELINE_SCRIPTS.get(STAGE):
        raise RuntimeError(f"Stage registry mismatch: {STAGE} is registered as {eu.PIPELINE_SCRIPTS.get(STAGE)!r}")
    if not (PROJECT_ROOT / SCRIPT_NAME).is_file():
        raise FileNotFoundError(f"Place this script at the authoritative path {PROJECT_ROOT / SCRIPT_NAME}")

    legacy = _resolve_legacy_outputs()
    eu.assert_no_legacy_outputs(legacy)

    migration_audit = eu.audit_pipeline_script_names(PROJECT_ROOT, require_all=False)
    if not args.validate_only and not args.allow_incomplete_pipeline:
        eu.audit_pipeline_script_names(PROJECT_ROOT, require_all=True)

    test_df, y_true, test_sha256 = _load_test_partition()
    shared_config = eu.load_shared_run_config(
        sequence_length_path=SEQUENCE_LENGTH_PATH,
        learning_rate_path=LEARNING_RATE_PATH,
        require_context=False,
        require_context_lr=False,
    )
    seeds = list(eu.EVAL_SEEDS_EXP1)
    runs: dict[str, list[dict[str, Any]]] = {key: [] for key in ("indobertweet", "xlmr")}

    for backbone in ("indobertweet", "xlmr"):
        for seed in seeds:
            item = _load_seed_run(
                backbone=backbone,
                seed=int(seed),
                test_df=test_df,
                y_true_test=y_true,
                test_sha256=test_sha256,
                shared_config=shared_config,
            )
            runs[backbone].append(item)
        print(f"Verified Stage {3 if backbone == 'indobertweet' else 4} artifacts for {backbone}: {seeds}")

    # Enforce same-stage backbone parity at each paired seed, and a fixed shared
    # data/split/configuration hash across all ten runs.
    for seed_index, seed in enumerate(seeds):
        cfg_a = runs["indobertweet"][seed_index]["manifest"]["config"]
        cfg_b = runs["xlmr"][seed_index]["manifest"]["config"]
        eu.check_config_parity(cfg_a, cfg_b, input_mode=INPUT_MODE)
        for key in (
            "dataset_sha256", "train_split_sha256", "validation_split_sha256", "test_split_sha256",
            "class_weights_sha256", "sequence_length_config_sha256", "learning_rate_config_sha256",
            "split_assignment_sha256", "max_length_benchmark", "eval_seeds_exp1",
        ):
            if cfg_a.get(key) != cfg_b.get(key):
                raise ValueError(f"Paired backbone config mismatch for seed {seed}: {key}")

    # Ensure every run agrees with the exact same test data and every seed's
    # gold labels / row order. _load_seed_run validates against ib_test.csv.
    all_y_true = [item["predictions"]["y_true"].astype(str).tolist() for b in runs for item in runs[b]]
    if any(labels != all_y_true[0] for labels in all_y_true[1:]):
        raise ValueError("Experiment I runs do not share one identical y_true row sequence")

    # A validation-only invocation proves that all ten source runs are current,
    # aligned and configuration-comparable, and exercises the metric/bootstrap
    # code, but it deliberately creates no Stage 5 outputs or manifest.
    if args.validate_only:
        _build_result_tables(y_true, runs)
        print("Stage 5 validation passed: ten current seed runs, row-order checks, artifact hashes, and backbone parity are valid.")
        print("Paired bootstrap computation also passed; no Stage 5 output files were written.")
        if not migration_audit["complete"]:
            print("Pipeline filename audit is still incomplete:")
            print(f"  Missing scripts: {migration_audit['missing']}")
            print(f"  Missing directories: {migration_audit['missing_directories']}")
        return

    input_paths: list[Path] = [TEST_PATH]
    run_fingerprints: dict[str, dict[str, str]] = {}
    for backbone, items in runs.items():
        run_fingerprints[backbone] = {}
        for item in items:
            run_fingerprints[backbone][str(item["seed"])] = str(item["run_manifest_fingerprint"])
            run_dir = item["run_dir"]
            input_paths.extend([
                run_dir / "predictions.csv", run_dir / "probabilities.npy", run_dir / "metrics.json",
                run_dir / "artifacts_sha256.json", run_dir / eu.RUN_MANIFEST_NAME,
            ])

    evaluation_config = {
        "schema_version": eu.CONFIG_SCHEMA_VERSION,
        "run_id": eu.RUN_ID,
        "stage": STAGE,
        "script_name": SCRIPT_NAME,
        "experiment_id": "E1-Backbone-Comparison",
        "input_mode": INPUT_MODE,
        "backbones": list(eu.BACKBONES),
        "experiment_ids": {key: eu.experiment1_id(key) for key in eu.BACKBONES},
        "seeds": seeds,
        "test_split_sha256": test_sha256,
        "n_test": len(test_df),
        "bootstrap": {"n_resamples": N_BOOTSTRAP, "ci_level": CI_LEVEL, "random_state": BOOTSTRAP_SEED,
                      "difference_direction": "XLM-R minus IndoBERTweet", "primary_metric": "macro_f1"},
        "source_run_manifest_fingerprints": run_fingerprints,
    }
    output_dir = eu.versioned_run_root(PROJECT_ROOT) / "experiment_i" / "evaluation"
    manifest = eu.make_run_manifest(
        stage=STAGE,
        script_name=SCRIPT_NAME,
        config=evaluation_config,
        input_paths=input_paths,
        metadata={
            "purpose": "Experiment I per-seed summary and paired backbone bootstrap",
            "source_stage_manifests": run_fingerprints,
            "test_row_order_sha256": eu.sha256_json(np.arange(len(test_df), dtype=np.int64).tolist()),
        },
        project_root=PROJECT_ROOT,
    )
    eu.prepare_run_directory(output_dir, manifest, allow_resume=True)

    if _verify_existing_outputs(output_dir, manifest):
        print(f"Stage 5 outputs already exist and passed manifest/hash verification: {output_dir}")
        return

    tables = _build_result_tables(y_true, runs)
    tables["seed_metrics"].to_csv(output_dir / "e1_seed_metrics.csv", index=False)
    tables["results"].to_csv(output_dir / "e1_results.csv", index=False)
    tables["ensemble_metrics"].to_csv(output_dir / "e1_ensemble_metrics.csv", index=False)
    tables["paired_bootstrap"].to_csv(output_dir / "e1_paired_bootstrap.csv", index=False)
    for filename, matrix in tables["confusion_matrices"].items():
        matrix.to_csv(output_dir / filename)

    summary_doc = {
        "schema_version": eu.CONFIG_SCHEMA_VERSION,
        "run_id": eu.RUN_ID,
        "stage": STAGE,
        "script_name": SCRIPT_NAME,
        "evaluation_manifest_fingerprint": manifest["fingerprint"],
        "test_split_sha256": test_sha256,
        "test_row_order_sha256": eu.sha256_json(np.arange(len(test_df), dtype=np.int64).tolist()),
        "n_test": len(test_df),
        "seeds": seeds,
        "experiment_ids": {key: eu.experiment1_id(key) for key in eu.BACKBONES},
        "mean_std_seed_results": tables["results"].to_dict(orient="records"),
        "ensemble_results": tables["ensemble_metrics"].to_dict(orient="records"),
        "paired_bootstrap": tables["paired_bootstrap"].to_dict(orient="records"),
        "source_run_manifest_fingerprints": run_fingerprints,
        "interpretation_note": (
            "The pre-declared inferential comparison is the paired 95% bootstrap CI for macro-F1 "
            "of seed-ensembled predictions, XLM-R minus IndoBERTweet. Per-class intervals are descriptive."
        ),
    }
    _write_json_atomic(output_dir / "e1_summary.json", summary_doc)

    artifact_hashes = {
        "schema_version": eu.CONFIG_SCHEMA_VERSION,
        "run_id": eu.RUN_ID,
        "stage": STAGE,
        "script_name": SCRIPT_NAME,
        "manifest_fingerprint": manifest["fingerprint"],
        "sha256": {name: _sha(output_dir / name) for name in OUTPUT_NAMES},
    }
    _write_json_atomic(output_dir / "artifacts_sha256.json", artifact_hashes)
    # Verify the Stage 5 artifact itself against the current utility/script/docs.
    eu.assert_current_run_artifact(
        output_dir / "e1_results.csv",
        expected_stage=STAGE,
        expected_script=SCRIPT_NAME,
        expected_fingerprint=str(manifest["fingerprint"]),
    )
    _verify_existing_outputs(output_dir, manifest)

    print("Stage 5 evaluation completed.")
    print(f"Run ID: {eu.RUN_ID}")
    print(f"Test split SHA-256: {test_sha256}")
    print(f"Verified runs per backbone: {len(seeds)}")
    print(f"Bootstrap: {N_BOOTSTRAP:,} paired resamples; {CI_LEVEL:.1f}% CI; direction XLM-R minus IndoBERTweet")
    print(f"Outputs: {output_dir.relative_to(PROJECT_ROOT).as_posix()}/")
    if not migration_audit["complete"]:
        print("Note: the pipeline registry audit is incomplete; this was a validate-only/isolated run.")


if __name__ == "__main__":
    main()
