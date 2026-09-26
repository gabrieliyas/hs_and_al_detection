"""
1_preprocess_benchmark.py

Stage 1: preprocess the Ibrohim & Budi (2019) benchmark dataset for
Experiment I.

Pipeline (documented in docs/dataset_split_and_weighting.md, Section 1):
  1. Load raw re_dataset.csv.
  2. Remove exact duplicate tweets.
  3. Strip USER/URL placeholder tokens and raw URLs (noise from the
     original keyword-based collection method), collapse whitespace.
  4. Remove tweets that are empty or under 3 words after stripping.
  5. Apply the 3-class label mapping (HS takes priority over Abusive).
  6. Stratified 70/15/15 train/val/test split.
  7. Compute class weights from the TRAINING partition only.

This pipeline is specific to the benchmark dataset. It must NOT be reused
on the custom contextual dataset (see 2_preprocess_contextual.py) --
that dataset is already deduplicated and anonymized, and short target
posts are a deliberate, meaningful part of its design (a short target is
often exactly the case where conversational context is needed), so the
same cleaning steps would actively damage it rather than clean it.

Usage:
    python scripts/1_preprocess_benchmark.py

Input:
    data/raw/re_dataset.csv

Outputs:
    data/processed/ib_cleaned.csv   -- full cleaned + labeled dataset
    data/processed/ib_train.csv
    data/processed/ib_val.csv
    data/processed/ib_test.csv
    results/tables/ib_class_weights.csv
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiment_utils import LABEL2ID, compute_class_weights  # noqa: E402

# Expected final row count after cleaning, per docs/dataset_split_and_weighting.md
# Section 1.1. This is a hard check, not an estimate: if the raw file or the
# cleaning logic ever drifts, this assertion is what catches it early,
# rather than silently producing a dataset that no longer matches the
# methodology chapter's stated numbers.
EXPECTED_CLEANED_COUNT = 12_746

RAW_PATH = Path("data/raw/re_dataset.csv")
OUT_DIR = Path("data/processed")
RESULTS_DIR = Path("results/tables")


def strip_placeholders(text: str) -> str:
    """
    Remove raw URLs and the USER/URL placeholder tokens the original
    dataset uses to mask mentions and links, then collapse whitespace.
    This targets noise introduced by the benchmark's keyword-based
    collection method -- it is not applied to the custom contextual
    dataset, which was collected and anonymized differently.
    """
    text = str(text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\bUSER\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\bURL\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def map_label(row: pd.Series) -> str:
    """Three-class mapping: HS takes priority over Abusive (see
    docs/annotation_guidelines.md and the Davidson et al. 2017 precedent
    documented in docs/dataset_split_and_weighting.md)."""
    if row["HS"] == 1:
        return "HS"
    elif row["Abusive"] == 1:
        return "AL"
    else:
        return "N"


def main() -> None:
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {RAW_PATH}. Download re_dataset.csv "
            "from https://github.com/okkyibrohim/id-multi-label-hate-speech-"
            "and-abusive-language-detection and place it there."
        )

    # The raw file is not valid UTF-8 throughout (a handful of tweets
    # contain raw byte sequences from emoji/encoding artifacts) -- latin-1
    # reads it without error and without altering the ASCII content that
    # matters for cleaning and modelling.
    df = pd.read_csv(RAW_PATH, encoding="latin-1", engine="python", on_bad_lines="skip")
    n_raw = len(df)
    print(f"[1/6] Raw rows loaded: {n_raw}")

    # Step 2: exact duplicate removal
    n_before = len(df)
    df = df.drop_duplicates(subset="Tweet", keep="first")
    print(f"[2/6] Removed {n_before - len(df)} exact duplicate tweets -> {len(df)} rows")

    # Step 3: strip placeholders/URLs
    df = df.copy()
    df["clean_text"] = df["Tweet"].apply(strip_placeholders)

    # Step 4: remove empty / under-3-word tweets (after stripping)
    df["word_count"] = df["clean_text"].apply(lambda t: len(t.split()) if t else 0)
    n_before = len(df)
    df = df[df["word_count"] >= 3].copy()
    print(f"[3-4/6] Removed {n_before - len(df)} empty/short (<3 word) tweets -> {len(df)} rows")

    # Hard check against the methodology chapter's documented figure.
    if len(df) != EXPECTED_CLEANED_COUNT:
        raise AssertionError(
            f"Cleaned row count is {len(df)}, expected {EXPECTED_CLEANED_COUNT}. "
            "This means the raw file or a cleaning step has changed since "
            "docs/dataset_split_and_weighting.md was written. Do not proceed "
            "to Stage 3 until this is understood -- either update the "
            "documented figure (with justification) or find what changed."
        )
    print(f"[5/6] Row count matches documented figure: {EXPECTED_CLEANED_COUNT}")

    # Step 5: label mapping
    df["label"] = df.apply(map_label, axis=1)
    df = df.rename(columns={"Tweet": "text_raw"})
    df = df[["text_raw", "clean_text", "HS", "Abusive", "label"]]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "ib_cleaned.csv", index=False)

    label_counts = df["label"].value_counts()
    label_pct = (label_counts / len(df) * 100).round(2)
    print("\nLabel distribution (cleaned, full dataset):")
    for c in ["N", "AL", "HS"]:
        print(f"  {c}: {label_counts[c]} ({label_pct[c]}%)")

    # Step 6: stratified 70/15/15 split
    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df["label"], random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["label"], random_state=42
    )

    train_df.to_csv(OUT_DIR / "ib_train.csv", index=False)
    val_df.to_csv(OUT_DIR / "ib_val.csv", index=False)
    test_df.to_csv(OUT_DIR / "ib_test.csv", index=False)

    print(f"\n[6/6] Stratified split -- train: {len(train_df)}, "
          f"val: {len(val_df)}, test: {len(test_df)}")

    # Step 7: class weights from the TRAINING partition only
    weights_by_id = compute_class_weights(train_df["label"])
    id2label = {v: k for k, v in LABEL2ID.items()}
    weights_table = pd.DataFrame(
        [{"label": id2label[i], "weight": w} for i, w in weights_by_id.items()]
    )
    weights_table.to_csv(RESULTS_DIR / "ib_class_weights.csv", index=False)

    print("\nClass weights (from training partition only):")
    print(weights_table.to_string(index=False))
    print(
        "\nExpected (per docs/dataset_split_and_weighting.md): "
        "N=0.7456, AL=2.5528, HS=0.7893"
    )

    print("\nDone. Outputs written to data/processed/ and results/tables/.")


if __name__ == "__main__":
    main()
