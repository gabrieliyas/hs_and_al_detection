"""Stage 2: validate and prepare the custom 1,800-post contextual dataset.

This is intentionally not the benchmark cleaning pipeline. It does not remove
short target posts, deduplicate silently, or strip URLs/mentions. The source
CSV must satisfy the confirmed annotation/collection assumptions exactly.

Run from the repository root or from any working directory:
    python scripts/2_preprocess_contextual.py

Canonical outputs (created only when absent):
    data/processed/ctx_cleaned.csv
    data/processed/ctx_matched_subset.csv
    data/processed/ctx_shallow_pool.csv
    results/tables/left_truncation_v2/ctx_distribution.csv
    results/runs/left_truncation_v2/stage_02/_run_manifest.json

The run manifest fingerprints this script, experiment_utils.py, the protocol,
the pipeline table, the raw source CSV, and all four output files. Stage 2
refuses to overwrite existing outputs or reuse an old stage manifest.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import experiment_utils as eu  # noqa: E402

RAW_PATH = PROJECT_ROOT / "data/raw/contextual_dataset.csv"
OUT_DIR = PROJECT_ROOT / "data/processed"
TABLE_DIR = PROJECT_ROOT / "results/tables" / eu.RUN_ID
STAGE_RUN_DIR = eu.versioned_run_root(PROJECT_ROOT) / "stage_02"

OUTPUT_PATHS = {
    "data/processed/ctx_cleaned.csv": OUT_DIR / "ctx_cleaned.csv",
    "data/processed/ctx_matched_subset.csv": OUT_DIR / "ctx_matched_subset.csv",
    "data/processed/ctx_shallow_pool.csv": OUT_DIR / "ctx_shallow_pool.csv",
    f"results/tables/{eu.RUN_ID}/ctx_distribution.csv": TABLE_DIR / "ctx_distribution.csv",
}


def _normalise_raw_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Map annotated full label names to canonical N/AL/HS without losing provenance."""
    out = df.copy()
    out["label_raw"] = out["label"].astype(str)
    normalized = out["label_raw"].str.strip().str.casefold()
    out["label"] = normalized.map(eu.RAW_CTX_LABEL_MAP)
    unmapped = out["label"].isna()
    if unmapped.any():
        bad_values = sorted(normalized[unmapped].unique().tolist())
        raise ValueError(
            "Contextual CSV contains labels outside the expected raw vocabulary "
            f"{sorted(eu.RAW_CTX_LABEL_MAP)}: {bad_values}"
        )
    return out


def _ensure_fresh_outputs() -> None:
    """Refuse to overwrite historical or unmanifested preprocessing outputs."""
    existing = [str(path.relative_to(PROJECT_ROOT)) for path in OUTPUT_PATHS.values() if path.exists()]
    if existing:
        raise FileExistsError(
            "Stage 2 will not overwrite existing files. Archive historical files before the clean rerun: "
            + ", ".join(existing)
        )
    if STAGE_RUN_DIR.exists():
        raise FileExistsError(
            f"Stage 2 run directory already exists: {STAGE_RUN_DIR}. Archive the old stage directory "
            "rather than deleting or reusing its manifest."
        )


def _assign_shared_folds(matched: pd.DataFrame) -> pd.DataFrame:
    """Assign folds once in stable post_id order, stratified by context_need × label."""
    result = matched.sort_values(eu.CTX_ID_COLUMN, key=lambda col: col.astype(str), kind="mergesort").copy()
    result = result.reset_index(drop=True)
    strat_key = result["context_needed"].astype(int).astype(str) + "_" + result["label"].astype(str)
    splitter = StratifiedKFold(
        n_splits=eu.CONTEXT_N_SPLITS,
        shuffle=True,
        random_state=eu.CONTEXT_FOLD_SEED,
    )
    folds = pd.Series(-1, index=result.index, dtype="int64")
    for fold_id, (_, heldout_indices) in enumerate(splitter.split(result, strat_key)):
        folds.iloc[heldout_indices] = fold_id
    if (folds == -1).any():
        raise AssertionError("Fold assignment failed: at least one matched post has no fold")
    result[eu.CTX_FOLD_COLUMN] = folds.astype(int)
    return result


def main() -> None:
    if not RAW_PATH.is_file():
        raise FileNotFoundError(
            f"Raw contextual dataset not found: {RAW_PATH}. Export the finalized annotated dataset to CSV first."
        )

    _ensure_fresh_outputs()
    raw_sha256 = eu.sha256_file(RAW_PATH)
    df = pd.read_csv(RAW_PATH)
    print(f"[1/8] Loaded {len(df):,} raw contextual rows.")

    missing = sorted(set(eu.CTX_ALL_RAW_COLUMNS) - set(df.columns))
    if missing:
        raise KeyError(
            f"Raw contextual CSV is missing required columns: {missing}. "
            "Compare the CSV header with CTX_ALL_RAW_COLUMNS in experiment_utils.py."
        )
    extra = sorted(set(df.columns) - set(eu.CTX_ALL_RAW_COLUMNS))
    if extra:
        print(f"  Note: preserving {len(extra)} extra source column(s) for traceability: {extra}")
    print("[2/8] Required schema columns are present.")

    df = _normalise_raw_labels(df)
    source_audit = eu.validate_context_source_dataset(df)
    print("[3/8] Contextual source validation passed: IDs, labels, parent availability, duplicates and distributions.")

    matched = df.loc[pd.to_numeric(df["context_size_available"], errors="coerce") == 4].copy()
    shallow = df.loc[pd.to_numeric(df["context_size_available"], errors="coerce").isin([1, 2, 3])].copy()
    if len(matched) != eu.CONTEXT_MATCHED_EXPECTED_ROWS or len(shallow) != eu.CONTEXT_SHALLOW_EXPECTED_ROWS:
        raise ValueError(
            "Expected a disjoint 900/900 matched/shallow split; "
            f"got matched={len(matched)} and shallow={len(shallow)}"
        )
    matched_ids = set(matched[eu.CTX_ID_COLUMN].astype(str))
    shallow_ids = set(shallow[eu.CTX_ID_COLUMN].astype(str))
    if matched_ids & shallow_ids:
        raise ValueError("Matched subset and shallow pool overlap by post_id")
    if matched_ids | shallow_ids != set(df[eu.CTX_ID_COLUMN].astype(str)):
        raise ValueError("Matched subset and shallow pool do not cover all source post IDs")

    matched = _assign_shared_folds(matched)
    matched_audit = eu.validate_context_dataset(matched)
    shallow_audit = eu.validate_context_shallow_pool(shallow)
    print("[4/8] Matched/shallow partitions validated; stable shared five-fold assignment verified.")
    print(f"       Matched fold counts: {matched_audit['fold_counts']}")
    print(f"       Shallow availability counts: {shallow_audit['context_size_available_counts']}")

    # Stable presentation order is not used for fold generation; post_id and fold
    # remain intact so the fixed assignment is identical across later contexts.
    cleaned_saved = df.sample(frac=1, random_state=eu.CONTEXT_SHUFFLE_SEED).reset_index(drop=True)
    matched_saved = matched.sample(frac=1, random_state=eu.CONTEXT_SHUFFLE_SEED).reset_index(drop=True)
    shallow_saved = shallow.sample(frac=1, random_state=eu.CONTEXT_SHUFFLE_SEED).reset_index(drop=True)
    distribution = eu.three_way_context_table(df)
    outputs = {
        "data/processed/ctx_cleaned.csv": (cleaned_saved, False),
        "data/processed/ctx_matched_subset.csv": (matched_saved, False),
        "data/processed/ctx_shallow_pool.csv": (shallow_saved, False),
        f"results/tables/{eu.RUN_ID}/ctx_distribution.csv": (distribution, True),
    }

    for destination in OUTPUT_PATHS.values():
        destination.parent.mkdir(parents=True, exist_ok=True)

    # Prepare byte-identical temporary outputs first so the immutable manifest
    # includes the digest of every final output before anything is published.
    with tempfile.TemporaryDirectory(prefix=".stage2-contextual-", dir=PROJECT_ROOT) as temp_name:
        temp_dir = Path(temp_name)
        output_hashes: dict[str, str] = {}
        temp_outputs: dict[str, Path] = {}
        for relative_path, (frame, include_index) in outputs.items():
            temp_path = temp_dir / Path(relative_path).name
            frame.to_csv(temp_path, index=include_index)
            output_hashes[relative_path] = eu.sha256_file(temp_path)
            temp_outputs[relative_path] = temp_path

        config = {
            "schema_version": eu.CONFIG_SCHEMA_VERSION,
            "run_id": eu.RUN_ID,
            "stage": 2,
            "script_name": eu.PIPELINE_SCRIPTS[2],
            "dataset_sha256": raw_sha256,
            "n_rows_raw": len(df),
            "post_id_sha256": source_audit["post_id_sha256"],
            "matched_post_id_sha256": matched_audit["post_id_sha256"],
            "fold_assignment_sha256": matched_audit["fold_assignment_sha256"],
            "shallow_post_id_sha256": shallow_audit["post_id_sha256"],
            "fold_seed": eu.CONTEXT_FOLD_SEED,
            "shuffle_seed": eu.CONTEXT_SHUFFLE_SEED,
            "n_splits": eu.CONTEXT_N_SPLITS,
            "stratification_columns": ["context_needed", "label"],
            "raw_label_map": eu.RAW_CTX_LABEL_MAP,
        }
        metadata = {
            "source_audit": source_audit,
            "matched_audit": matched_audit,
            "shallow_audit": shallow_audit,
            "partition_disjoint": True,
            "output_sha256": output_hashes,
            "output_rows": {
                "data/processed/ctx_cleaned.csv": len(cleaned_saved),
                "data/processed/ctx_matched_subset.csv": len(matched_saved),
                "data/processed/ctx_shallow_pool.csv": len(shallow_saved),
                f"results/tables/{eu.RUN_ID}/ctx_distribution.csv": int(distribution.shape[0]),
            },
        }
        manifest = eu.make_run_manifest(
            stage=2,
            script_name=eu.PIPELINE_SCRIPTS[2],
            config=config,
            input_paths=[RAW_PATH],
            metadata=metadata,
            project_root=PROJECT_ROOT,
        )
        eu.prepare_run_directory(STAGE_RUN_DIR, manifest, allow_resume=False)

        # Each individual rename is atomic on the same filesystem. If the
        # process fails partway, the next run refuses to overwrite partial
        # outputs and the incomplete run must be archived/investigated.
        for relative_path, temp_path in temp_outputs.items():
            destination = OUTPUT_PATHS[relative_path]
            if destination.exists():
                raise FileExistsError(f"Refusing to overwrite newly appeared output: {destination}")
            os.replace(temp_path, destination)

    verified_manifest = eu.assert_current_stage_manifest(2, PROJECT_ROOT)
    print("[5/8] Stage 2 manifest and live input/output hashes verified.")
    print("[6/8] Wrote validated contextual partitions:")
    for relative_path, destination in OUTPUT_PATHS.items():
        print(f"  {relative_path} ({destination.stat().st_size:,} bytes)")
    print(f"[7/8] Manifest fingerprint: {verified_manifest['fingerprint']}")
    print(f"[8/8] Stage 2 completed. Run ID: {eu.RUN_ID}")
    print("Reminder: downstream scripts must use load_modelling_view(); analysis-only columns must not reach tokenization.")


if __name__ == "__main__":
    main()
