"""
2_preprocess_contextual.py

Stage 2: preprocess the custom contextual dataset for Experiments II and III.

This is deliberately NOT the benchmark cleaning pipeline (see
1_preprocess_benchmark.py). The custom dataset was collected and annotated
differently, so different assumptions hold:
  - No deduplication: already checked (manually, in Excel) to have no
    duplicate target posts.
  - No short-text removal: a 2-3 word target post is a meaningful, deliberate
    case, not noise -- it is often exactly the situation where conversational
    context is doing the work. Removing it would strip out part of the
    dataset's core contribution.
  - No blind URL/mention stripping: the dataset was collected and
    anonymized as part of annotation, not scraped via keyword search the
    way the benchmark was.

What this script DOES do:
  1. Validate the raw file has every expected column (fails loud, not
     silently, on a schema mismatch).
  2. Verify label == f(HS, AL) via the priority mapping rule.
  3. Verify no duplicate target_text (should be zero; raises if not, since
     that would contradict the stated collection process rather than being
     an expected cleaning step).
  4. Hard-check total counts and the context_needed x label distribution
     against the confirmed final collection numbers.
  5. Split into the 900-example matched subset (context_size_available==4)
     and the 900-example shallow pool (context_size_available in {1,2,3}).
  6. Assign the shared 5-fold column to the matched subset only (per
     docs/dataset_split_and_weighting.md Section 2.4).
  7. Shuffle all saved files (seeded) -- the raw collection order is
     systematic (by parent count / label), not random.

Usage:
    python scripts/2_preprocess_contextual.py

Input:
    data/raw/contextual_dataset.csv

Outputs:
    data/processed/ctx_cleaned.csv          -- full 1,800-row dataset, all columns
    data/processed/ctx_matched_subset.csv   -- 900 rows, context_size_available==4, includes `fold`
    data/processed/ctx_shallow_pool.csv     -- 900 rows, context_size_available in {1,2,3}
    results/tables/ctx_distribution.csv     -- three-way distribution table
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiment_utils import (  # noqa: E402
    CTX_ALL_RAW_COLUMNS,
    CTX_ID_COLUMN,
    RAW_CTX_LABEL_MAP,
    three_way_context_table,
    validate_label_consistency,
)

RAW_PATH = Path("data/raw/contextual_dataset.csv")
OUT_DIR = Path("data/processed")
RESULTS_DIR = Path("results/tables")
SHUFFLE_SEED = 42

# Confirmed final collection numbers (see README.md). Hard checks: if the
# real file doesn't match these, something changed since collection was
# reported complete, and that needs investigating before proceeding --
# not silently absorbing a different dataset.
EXPECTED_TOTAL = 1800
EXPECTED_LABEL_TOTALS = {"N": 600, "AL": 600, "HS": 600}
EXPECTED_CONTEXT_NEEDED_CELLS = {
    # (context_needed, label): count
    (1, "N"): 180, (1, "AL"): 300, (1, "HS"): 365,
    (0, "N"): 420, (0, "AL"): 300, (0, "HS"): 235,
}


def shuffle_dataset(df: pd.DataFrame, random_state: int = SHUFFLE_SEED) -> pd.DataFrame:
    """Shuffle row order (seeded) -- removes the systematic collection
    order (by parent count / label) before saving. post_id is preserved
    as a column, so original order is always recoverable via sort."""
    return df.sample(frac=1, random_state=random_state).reset_index(drop=True)


def main() -> None:
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Raw contextual dataset not found at {RAW_PATH}. Export the "
            "final annotated spreadsheet to CSV and place it there."
        )

    df = pd.read_csv(RAW_PATH)
    print(f"[1/8] Raw rows loaded: {len(df)}")

    # --- Step 1: schema validation (allowlist-driven) ---
    missing = [c for c in CTX_ALL_RAW_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(
            f"Raw file is missing expected column(s): {missing}.\n"
            f"Expected schema (see experiment_utils.CTX_ALL_RAW_COLUMNS):\n"
            f"  {CTX_ALL_RAW_COLUMNS}\n"
            "Check the CSV's header row, or update CTX_MODELLING_COLUMNS / "
            "CTX_ANALYSIS_ONLY_COLUMNS in experiment_utils.py if the real "
            "column names differ from what was assumed here."
        )
    extra = [c for c in df.columns if c not in CTX_ALL_RAW_COLUMNS]
    if extra:
        print(f"  Note: {len(extra)} column(s) present but not in the "
              f"expected schema (kept, unused): {extra}")
    print("[2/8] Schema validated -- all expected columns present.")

    # --- Step 1b: normalize raw label spelling to the canonical N/AL/HS codes ---
    # The annotated export spells labels out in full ("neither",
    # "abusive_language", "hate_speech") rather than using the short codes
    # used everywhere else in this pipeline (LABEL2ID, Experiment I's own
    # cleaned dataset, all docs). This is an ingestion-normalization step,
    # not a change to the underlying data: the HS/AL 0/1 indicator columns
    # are untouched, and the original spelling is kept as `label_raw` for
    # traceability. Fails loud on any value outside the expected vocabulary,
    # rather than silently producing NaN that would confuse the checks below.
    df["label_raw"] = df["label"]
    df["label"] = df["label"].map(RAW_CTX_LABEL_MAP)
    unmapped = df[df["label"].isna()]
    if len(unmapped):
        bad_values = sorted(unmapped["label_raw"].astype(str).unique())
        raise ValueError(
            f"{len(unmapped)} row(s) have a `label` value outside the "
            f"expected raw vocabulary {sorted(RAW_CTX_LABEL_MAP)}. "
            f"Unexpected value(s): {bad_values}. Check for typos in the "
            "annotated export, or update RAW_CTX_LABEL_MAP in "
            "experiment_utils.py if the raw spelling has changed."
        )
    print(
        "[2b/8] Label spelling normalized to canonical codes ("
        + ", ".join(f"{k}->{v}" for k, v in RAW_CTX_LABEL_MAP.items())
        + "); original spelling kept as `label_raw`."
    )

    # --- Step 2: label consistency check ---
    validate_label_consistency(df)
    print("[3/8] Label consistency check passed (label == f(HS, AL) for all rows).")

    # --- Step 3: duplicate check (should be zero; not a cleaning step) ---
    n_dupes = df.duplicated(subset="target_text").sum()
    if n_dupes > 0:
        raise ValueError(
            f"Found {n_dupes} duplicate target_text value(s), but the "
            "dataset was expected to have none (per manual Excel check). "
            "Investigate before proceeding -- do not silently deduplicate, "
            "since that would change which posts are in the dataset "
            "without a documented reason."
        )
    print("[4/8] Duplicate check passed -- no duplicate target_text values.")

    # --- Step 4: hard distribution checks ---
    if len(df) != EXPECTED_TOTAL:
        raise AssertionError(f"Total rows = {len(df)}, expected {EXPECTED_TOTAL}.")

    label_totals = df["label"].value_counts().to_dict()
    for label, expected in EXPECTED_LABEL_TOTALS.items():
        actual = label_totals.get(label, 0)
        if actual != expected:
            raise AssertionError(
                f"Label '{label}' has {actual} rows, expected {expected}."
            )

    cell_counts = df.groupby(["context_needed", "label"]).size().to_dict()
    mismatched_cells = []
    for (needed, label), expected in EXPECTED_CONTEXT_NEEDED_CELLS.items():
        actual = cell_counts.get((needed, label), 0)
        if actual != expected:
            mismatched_cells.append((needed, label, actual, expected))
    if mismatched_cells:
        detail = "\n".join(
            f"  context_needed={n}, label={l}: got {a}, expected {e}"
            for n, l, a, e in mismatched_cells
        )
        raise AssertionError(
            f"context_needed x label distribution does not match the "
            f"confirmed collection numbers:\n{detail}"
        )
    print("[5/8] Distribution checks passed -- matches confirmed collection numbers "
          "(1,800 total; 600/600/600 per class; context-needed 845/955 split).")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Step 5: three-way distribution table (for Chapter III reporting) ---
    dist_table = three_way_context_table(df)
    dist_table.to_csv(RESULTS_DIR / "ctx_distribution.csv")
    print("\nThree-way distribution (context_size_available x context_needed x label):")
    print(dist_table.to_string())

    # --- Step 6: split into matched subset and shallow pool ---
    matched = df[df["context_size_available"] == 4].copy()
    shallow = df[df["context_size_available"].isin([1, 2, 3])].copy()
    print(f"\n[6/8] Matched subset (context_size_available==4): {len(matched)} rows "
          f"(expected 900)")
    print(f"       Shallow pool (context_size_available in 1-3): {len(shallow)} rows "
          f"(expected 900)")
    if len(matched) != 900 or len(shallow) != 900:
        raise AssertionError(
            "Matched subset / shallow pool sizes do not match the expected "
            "900 / 900 split. Check context_size_available values."
        )

    # --- Step 7: shared fold assignment (matched subset only) ---
    # Per docs/dataset_split_and_weighting.md Section 2.4: fold assignment
    # happens once, on the 900 base examples, BEFORE any C0-C4 truncated
    # view is constructed downstream. Stratified by (context_needed, label)
    # jointly, since context_size_available is constant (=4) within this
    # subset and therefore not a useful stratification variable here.
    strat_key = matched["context_needed"].astype(str) + "_" + matched["label"]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    matched["fold"] = -1
    for fold_id, (_, test_idx) in enumerate(skf.split(matched, strat_key)):
        matched.iloc[test_idx, matched.columns.get_loc("fold")] = fold_id
    assert (matched["fold"] == -1).sum() == 0, "Every row must receive a fold assignment."

    fold_sizes = matched["fold"].value_counts().sort_index()
    print(f"\n[7/8] Fold assignment complete. Fold sizes: {fold_sizes.to_dict()}")

    # --- Step 8: shuffle and save ---
    df_shuffled = shuffle_dataset(df)
    matched_shuffled = shuffle_dataset(matched)
    shallow_shuffled = shuffle_dataset(shallow)

    df_shuffled.to_csv(OUT_DIR / "ctx_cleaned.csv", index=False)
    matched_shuffled.to_csv(OUT_DIR / "ctx_matched_subset.csv", index=False)
    shallow_shuffled.to_csv(OUT_DIR / "ctx_shallow_pool.csv", index=False)

    print(f"\n[8/8] Saved (all shuffled, seed={SHUFFLE_SEED}):")
    print(f"  data/processed/ctx_cleaned.csv        ({len(df_shuffled)} rows)")
    print(f"  data/processed/ctx_matched_subset.csv ({len(matched_shuffled)} rows, +fold column)")
    print(f"  data/processed/ctx_shallow_pool.csv   ({len(shallow_shuffled)} rows)")
    print(f"  results/tables/ctx_distribution.csv")
    print(
        f"\nReminder: {CTX_ID_COLUMN} is preserved through the shuffle, so "
        "original collection order is always recoverable by sorting on it. "
        "Downstream training scripts must load data via "
        "experiment_utils.load_modelling_view(), never raw column access, "
        "to keep analysis-only columns (context_needed, target_group, "
        "context_type, etc.) out of the model input."
    )


if __name__ == "__main__":
    main()
