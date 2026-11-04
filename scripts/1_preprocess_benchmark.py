"""Stage 1: preprocess the Ibrohim & Budi (2019) benchmark dataset.

Pipeline
--------
1. Load the original ``data/raw/re_dataset.csv`` file.
2. Clean and normalize tweet text (Unicode NFKC, remove raw URLs and the
   dataset's USER/URL placeholders, and collapse whitespace).
3. Apply the three-class mapping (HS takes priority over Abusive).
4. Audit missing/invalid text and label values. Invalid text rows are
   excluded and recorded; invalid label values stop the run.
5. Deduplicate on the final ``clean_text`` value before splitting. Apply the
   configured conflict policy to groups with more than one mapped label.
6. Perform a stratified 70/15/15 train/validation/test split.
7. Compute class weights from the training split only.
8. Save the processed datasets, class weights, and preprocessing audit.

Conflict policy is controlled by CONFLICT_POLICY:
    exclude    Exclude all rows in every conflicting clean_text group (default).
    error      Stop before splitting if any conflicting group is found.
    adjudicate Require a documented decision in data/annotations/ib_duplicate_label_adjudication.csv for every conflicting group.

This benchmark-specific cleaning must not be reused for the custom
contextual dataset, where short target posts are meaningful data.

Run from the repository root:
    python scripts/1_preprocess_benchmark.py

Input:
    data/raw/re_dataset.csv

Outputs:
    data/processed/ib_cleaned.csv
    data/processed/ib_train.csv
    data/processed/ib_val.csv
    data/processed/ib_test.csv
    results/tables/ib_class_weights.csv
    results/tables/ib_preprocessing_audit.json
    results/tables/ib_preprocessing_audit.csv

Diagnostic outputs (created when needed):
    data/processed/ib_invalid_rows.csv
    data/processed/ib_duplicate_label_conflicts.csv
    data/processed/ib_duplicate_label_conflict_summary.csv
    data/processed/ib_duplicate_label_adjudication_template.csv
    data/processed/ib_conflict_excluded_rows.csv
    data/processed/ib_conflict_resolution_log.csv

Optional manual adjudication input (used only when CONFLICT_POLICY="adjudicate"):
    data/annotations/ib_duplicate_label_adjudication.csv
"""

from __future__ import annotations

import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from experiment_utils import LABEL2ID, compute_class_weights  # noqa: E402

RAW_PATH = PROJECT_ROOT / "data/raw/re_dataset.csv"
OUT_DIR = PROJECT_ROOT / "data/processed"
RESULTS_DIR = PROJECT_ROOT / "results/tables"

RANDOM_STATE = 42
MIN_WORDS = 3
CLASS_ORDER = ["N", "AL", "HS"]
SPLIT_RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}

# Conflict policy:
#   "exclude"    -> exclude EVERY row in each conflicting clean_text group.
#   "error"      -> stop and require the conflicts to be resolved explicitly.
#   "adjudicate" -> read data/annotations/ib_duplicate_label_adjudication.csv;
#                   each conflict group must have an explicit adjudicate/exclude
#                   decision, rationale, and reviewer.
CONFLICT_POLICY = "exclude"
ADJUDICATION_PATH = PROJECT_ROOT / "data/annotations/ib_duplicate_label_adjudication.csv"
ADJUDICATION_TEMPLATE_PATH = OUT_DIR / "ib_duplicate_label_adjudication_template.csv"
ADJUDICATION_COLUMNS = [
    "clean_text", "action", "adjudicated_label", "rationale", "reviewer"
]

STAGE1_OUTPUTS = [
    OUT_DIR / "ib_cleaned.csv",
    OUT_DIR / "ib_train.csv",
    OUT_DIR / "ib_val.csv",
    OUT_DIR / "ib_test.csv",
    OUT_DIR / "ib_invalid_rows.csv",
    OUT_DIR / "ib_duplicate_label_conflicts.csv",
    OUT_DIR / "ib_duplicate_label_conflict_summary.csv",
    OUT_DIR / "ib_duplicate_label_adjudication_template.csv",
    OUT_DIR / "ib_conflict_excluded_rows.csv",
    OUT_DIR / "ib_conflict_resolution_log.csv",
    RESULTS_DIR / "ib_class_weights.csv",
    RESULTS_DIR / "ib_preprocessing_audit.json",
    RESULTS_DIR / "ib_preprocessing_audit.csv",
]


def clean_and_normalize_text(value: Any) -> str:
    """Normalize benchmark text without changing its letter casing."""
    if not isinstance(value, str):
        return ""

    text = unicodedata.normalize("NFKC", value)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\bUSER\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\bURL\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def map_label(hs: int, abusive: int) -> str:
    """Map the original binary annotations to N/AL/HS; HS has priority."""
    if hs == 1:
        return "HS"
    if abusive == 1:
        return "AL"
    return "N"


def _class_distribution(frame: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    """Return counts and percentages in the fixed N, AL, HS order."""
    counts = frame["label"].value_counts().reindex(CLASS_ORDER, fill_value=0)
    total = len(frame)
    return {
        label: {
            "count": int(counts[label]),
            "percentage": round(float(counts[label] / total * 100), 2) if total else 0.0,
        }
        for label in CLASS_ORDER
    }


def _write_audit(audit: dict[str, Any], audit_records: list[dict[str, Any]]) -> None:
    """Write both a detailed JSON audit and a compact, tabular CSV audit."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with (RESULTS_DIR / "ib_preprocessing_audit.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)

    pd.DataFrame(
        audit_records, columns=["section", "split", "label", "metric", "value"]
    ).to_csv(RESULTS_DIR / "ib_preprocessing_audit.csv", index=False)


def _add_count(
    records: list[dict[str, Any]], metric: str, value: int, section: str = "row_counts"
) -> None:
    records.append(
        {"section": section, "split": "all", "label": "", "metric": metric, "value": int(value)}
    )


def _add_distribution(
    records: list[dict[str, Any]], split_name: str, distribution: dict[str, dict[str, float | int]]
) -> None:
    for label in CLASS_ORDER:
        records.append(
            {
                "section": "class_distribution",
                "split": split_name,
                "label": label,
                "metric": "count",
                "value": distribution[label]["count"],
            }
        )
        records.append(
            {
                "section": "class_distribution",
                "split": split_name,
                "label": label,
                "metric": "percentage",
                "value": distribution[label]["percentage"],
            }
        )


def _clear_previous_stage1_outputs() -> None:
    """Prevent outputs from an earlier run being mistaken for current ones."""
    for path in STAGE1_OUTPUTS:
        if path.exists():
            path.unlink()


def _write_conflict_diagnostics(valid_df: pd.DataFrame, conflict_keys: list[str]) -> pd.DataFrame:
    """Write full-source conflict rows, a group summary, and an edit template."""
    if not conflict_keys:
        return pd.DataFrame()

    conflict_rows = valid_df[valid_df["clean_text"].isin(conflict_keys)].copy()
    conflict_rows = conflict_rows.sort_values(["clean_text", "source_record"])
    conflict_rows.to_csv(OUT_DIR / "ib_duplicate_label_conflicts.csv", index=False)

    summary = (
        conflict_rows.groupby("clean_text", sort=False)
        .agg(
            duplicate_rows=("clean_text", "size"),
            source_records=("source_record", lambda values: ", ".join(map(str, values))),
            raw_text_variants=(
                "text_raw",
                lambda values: " || ".join(dict.fromkeys(values.astype(str))),
            ),
            mapped_labels=(
                "label",
                lambda values: " | ".join(sorted(set(values.astype(str)))),
            ),
        )
        .reset_index()
    )
    summary.to_csv(OUT_DIR / "ib_duplicate_label_conflict_summary.csv", index=False)

    template = pd.DataFrame(
        {
            "clean_text": list(conflict_keys),
            "action": [""] * len(conflict_keys),
            "adjudicated_label": [""] * len(conflict_keys),
            "rationale": [""] * len(conflict_keys),
            "reviewer": [""] * len(conflict_keys),
        },
        columns=ADJUDICATION_COLUMNS,
    )
    template.to_csv(ADJUDICATION_TEMPLATE_PATH, index=False)
    return conflict_rows


def _load_adjudication_decisions(conflict_keys: list[str]) -> dict[str, dict[str, str]]:
    """Load and strictly validate one explicit decision for every conflict group."""
    if not ADJUDICATION_PATH.exists():
        raise FileNotFoundError(
            f"Conflict policy is 'adjudicate', but {ADJUDICATION_PATH} does not exist. "
            f"A blank template was generated at {ADJUDICATION_TEMPLATE_PATH}. "
            "Copy/fill that template at the adjudication path and rerun Stage 1."
        )

    decisions_df = pd.read_csv(
        ADJUDICATION_PATH, dtype=str, keep_default_na=False, encoding="utf-8"
    )
    missing_columns = sorted(set(ADJUDICATION_COLUMNS) - set(decisions_df.columns))
    if missing_columns:
        raise ValueError(
            "The adjudication file is missing required column(s): "
            + ", ".join(missing_columns)
        )

    decisions_df = decisions_df[ADJUDICATION_COLUMNS].copy()
    decisions_df["clean_text"] = decisions_df["clean_text"].astype(str)
    decisions_df["action"] = decisions_df["action"].str.strip().str.lower()
    decisions_df["adjudicated_label"] = (
        decisions_df["adjudicated_label"].str.strip().str.upper()
    )
    decisions_df["rationale"] = decisions_df["rationale"].str.strip()
    decisions_df["reviewer"] = decisions_df["reviewer"].str.strip()

    if decisions_df["clean_text"].duplicated().any():
        duplicates = decisions_df.loc[
            decisions_df["clean_text"].duplicated(keep=False), "clean_text"
        ].tolist()
        raise ValueError(
            "The adjudication file contains duplicate clean_text decision rows: "
            f"{duplicates[:3]}"
        )

    supplied_keys = set(decisions_df["clean_text"])
    required_keys = set(conflict_keys)
    missing_keys = required_keys - supplied_keys
    extra_keys = supplied_keys - required_keys
    if missing_keys or extra_keys:
        details = []
        if missing_keys:
            details.append(f"missing decisions for {len(missing_keys)} current conflict group(s)")
        if extra_keys:
            details.append(f"{len(extra_keys)} decision row(s) do not match a current conflict group")
        raise ValueError(
            "The adjudication file must contain exactly one row for every current "
            "conflict group: " + "; ".join(details)
        )

    invalid_actions = sorted(set(decisions_df["action"]) - {"adjudicate", "exclude"})
    if invalid_actions:
        raise ValueError(
            "Invalid adjudication action(s): " + ", ".join(invalid_actions)
            + ". Allowed actions are 'adjudicate' and 'exclude'."
        )

    for row in decisions_df.itertuples(index=False):
        if not row.rationale:
            raise ValueError(f"Missing rationale for conflict group: {row.clean_text[:80]!r}")
        if not row.reviewer:
            raise ValueError(f"Missing reviewer for conflict group: {row.clean_text[:80]!r}")
        if row.action == "adjudicate" and row.adjudicated_label not in CLASS_ORDER:
            raise ValueError(
                f"Invalid adjudicated_label for conflict group {row.clean_text[:80]!r}. "
                f"Use one of {CLASS_ORDER}."
            )
        if row.action == "exclude" and row.adjudicated_label:
            raise ValueError(
                f"Leave adjudicated_label blank when action='exclude' for "
                f"{row.clean_text[:80]!r}."
            )

    return {
        row.clean_text: {
            "action": row.action,
            "adjudicated_label": row.adjudicated_label,
            "rationale": row.rationale,
            "reviewer": row.reviewer,
        }
        for row in decisions_df.itertuples(index=False)
    }


def main() -> None:
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {RAW_PATH}. Place the original "
            "re_dataset.csv file in data/raw/."
        )

    # latin-1 retains the original script's handling of legacy byte/emoji
    # encoding artefacts. A bad-line callback makes skipped malformed rows
    # countable in the audit instead of silently hiding that information.
    skipped_bad_lines: list[list[str]] = []

    def collect_bad_line(bad_line: list[str]) -> None:
        skipped_bad_lines.append(bad_line)
        return None

    df = pd.read_csv(
        RAW_PATH,
        encoding="latin-1",
        engine="python",
        on_bad_lines=collect_bad_line,
    )
    n_raw = len(df)
    # One-based ordinal among successfully parsed CSV data records (not physical
    # file line numbers). This enables conflict decisions to be traced to source rows.
    df.insert(0, "source_record", range(1, n_raw + 1))

    if CONFLICT_POLICY not in {"exclude", "error", "adjudicate"}:
        raise ValueError(
            f"Unknown CONFLICT_POLICY={CONFLICT_POLICY!r}. "
            "Choose 'exclude', 'error', or 'adjudicate'."
        )

    required_columns = {"Tweet", "HS", "Abusive"}
    missing_columns = sorted(required_columns.difference(df.columns))
    if missing_columns:
        raise ValueError(
            "The raw dataset is missing required column(s): "
            + ", ".join(missing_columns)
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _clear_previous_stage1_outputs()

    audit: dict[str, Any] = {
        "status": "running",
        "input_file": str(RAW_PATH.relative_to(PROJECT_ROOT)),
        "encoding": "latin-1",
        "malformed_csv_lines_skipped": len(skipped_bad_lines),
        "configuration": {
            "text_normalization": "Unicode NFKC; remove http/https URLs and standalone USER/URL placeholders; collapse whitespace; preserve casing",
            "minimum_word_count": MIN_WORDS,
            "label_mapping": {
                "HS": "HS (takes priority when HS=1)",
                "Abusive": "AL when HS=0 and Abusive=1",
                "neither": "N when HS=0 and Abusive=0",
            },
            "deduplication_key": "clean_text",
            "deduplication_stage": "after text/label validation and before data splitting",
            "conflict_policy": CONFLICT_POLICY,
            "adjudication_file": str(ADJUDICATION_PATH.relative_to(PROJECT_ROOT)),
            "adjudication_actions": ["adjudicate", "exclude"],
            "split_ratios": SPLIT_RATIOS,
            "split_random_state": RANDOM_STATE,
            "class_order": CLASS_ORDER,
            "class_weight_source": "training split only",
        },
        "row_counts": {"raw_rows_loaded": n_raw},
        "class_distributions": {},
        "class_weights": {},
        "errors": [],
    }
    audit_records: list[dict[str, Any]] = []
    _add_count(audit_records, "raw_rows_loaded", n_raw)
    _add_count(audit_records, "malformed_csv_lines_skipped", len(skipped_bad_lines))

    print(f"[1/8] Raw rows loaded: {n_raw}")
    if skipped_bad_lines:
        print(f"      Malformed CSV lines skipped: {len(skipped_bad_lines)}")

    # Step 2: clean/normalize text.
    df = df.copy()
    df["text_raw"] = df["Tweet"]
    df["clean_text"] = df["text_raw"].apply(clean_and_normalize_text)
    df["word_count"] = df["clean_text"].map(lambda text: len(text.split()))
    print("[2/8] Text cleaning and normalization completed.")

    # Parse the source annotations without silently coercing invalid values
    # into a valid class. Original values are retained for the diagnostic CSV.
    hs_numeric = pd.to_numeric(df["HS"], errors="coerce")
    abusive_numeric = pd.to_numeric(df["Abusive"], errors="coerce")
    hs_valid = hs_numeric.notna() & hs_numeric.isin([0, 1])
    abusive_valid = abusive_numeric.notna() & abusive_numeric.isin([0, 1])
    valid_label_mask = hs_valid & abusive_valid

    # Step 3: apply the mapping to rows whose source labels are interpretable.
    df["label"] = pd.Series(pd.NA, index=df.index, dtype="object")
    if valid_label_mask.any():
        df.loc[valid_label_mask, "label"] = [
            map_label(int(hs), int(abusive))
            for hs, abusive in zip(
                hs_numeric.loc[valid_label_mask],
                abusive_numeric.loc[valid_label_mask],
            )
        ]
    print("[3/8] Three-class label mapping applied (HS priority).")

    # Step 4: validate texts and labels. Text rows that are missing, empty,
    # or shorter than the existing three-word benchmark rule are excluded.
    raw_text_missing = (
        df["text_raw"].isna()
        | df["text_raw"].astype("string").str.strip().eq("").fillna(True)
    )
    clean_text_empty = df["clean_text"].eq("")
    too_short = df["word_count"] < MIN_WORDS
    invalid_text_mask = raw_text_missing | clean_text_empty | too_short
    invalid_label_mask = ~valid_label_mask

    invalid_rows = df.loc[invalid_text_mask | invalid_label_mask].copy()
    if not invalid_rows.empty:
        invalid_rows["text_issue"] = ""
        invalid_rows.loc[raw_text_missing.loc[invalid_rows.index], "text_issue"] = "missing/blank raw text"
        invalid_rows.loc[~raw_text_missing.loc[invalid_rows.index] & clean_text_empty.loc[invalid_rows.index], "text_issue"] = "empty after cleaning"
        invalid_rows.loc[~raw_text_missing.loc[invalid_rows.index] & ~clean_text_empty.loc[invalid_rows.index] & too_short.loc[invalid_rows.index], "text_issue"] = f"fewer than {MIN_WORDS} words after cleaning"
        invalid_rows["label_issue"] = ""
        invalid_rows.loc[~hs_valid.loc[invalid_rows.index], "label_issue"] = "HS is missing or not binary (0/1)"
        invalid_rows.loc[~abusive_valid.loc[invalid_rows.index], "label_issue"] = invalid_rows.loc[~abusive_valid.loc[invalid_rows.index], "label_issue"].apply(
            lambda current: (current + "; " if current else "") + "Abusive is missing or not binary (0/1)"
        )
        invalid_rows.to_csv(OUT_DIR / "ib_invalid_rows.csv", index=False)

    counts = {
        "missing_or_blank_raw_text_rows": int(raw_text_missing.sum()),
        "empty_after_cleaning_rows": int(clean_text_empty.sum()),
        "under_minimum_word_count_rows": int((~raw_text_missing & ~clean_text_empty & too_short).sum()),
        "invalid_text_rows_excluded": int(invalid_text_mask.sum()),
        "invalid_label_rows": int(invalid_label_mask.sum()),
        "rows_with_any_text_or_label_issue": int((invalid_text_mask | invalid_label_mask).sum()),
    }
    audit["row_counts"].update(counts)
    for metric, value in counts.items():
        _add_count(audit_records, metric, value)

    print(
        "[4/8] Validation: "
        f"{counts['invalid_text_rows_excluded']} invalid text row(s) to exclude; "
        f"{counts['invalid_label_rows']} invalid label row(s)."
    )

    if invalid_label_mask.any():
        message = (
            f"Found {int(invalid_label_mask.sum())} row(s) with missing or invalid "
            "HS/Abusive values. See data/processed/ib_invalid_rows.csv. "
            "Correct the source data or resolve these rows before rerunning Stage 1."
        )
        audit["status"] = "failed"
        audit["errors"].append(message)
        _write_audit(audit, audit_records)
        raise ValueError(message)

    # Exclude invalid text rows, retaining only rows with a usable text and
    # a valid mapped label for the duplicate-label audit.
    valid_df = df.loc[~invalid_text_mask].copy()
    if valid_df.empty:
        message = "No valid benchmark rows remain after text validation."
        audit["status"] = "failed"
        audit["errors"].append(message)
        _write_audit(audit, audit_records)
        raise ValueError(message)

    # Step 5: inspect duplicates globally on normalized text BEFORE splitting.
    # Keep all original source columns here so the conflict report is auditable.
    grouped = valid_df.groupby("clean_text", sort=False, dropna=False)
    group_sizes = grouped.size()
    label_cardinality = grouped["label"].nunique(dropna=False)
    duplicate_group_keys = group_sizes[group_sizes > 1].index.tolist()
    conflict_keys = label_cardinality[label_cardinality > 1].index.tolist()
    same_label_duplicate_keys = group_sizes.index.intersection(
        label_cardinality[(label_cardinality == 1) & (group_sizes > 1)].index
    ).tolist()

    conflict_rows = _write_conflict_diagnostics(valid_df, conflict_keys)
    duplicate_occurrences = int(valid_df.duplicated(subset="clean_text", keep="first").sum())
    same_label_duplicate_occurrences = int(
        valid_df[valid_df["clean_text"].isin(same_label_duplicate_keys)]
        .duplicated(subset="clean_text", keep="first")
        .sum()
    )

    audit["row_counts"].update(
        {
            "valid_text_rows_before_deduplication": int(len(valid_df)),
            "duplicated_clean_text_groups": int(len(duplicate_group_keys)),
            "same_label_duplicate_groups": int(len(same_label_duplicate_keys)),
            "duplicate_occurrences_total": duplicate_occurrences,
            "same_label_duplicate_occurrences_removed": same_label_duplicate_occurrences,
            "conflicting_clean_text_groups": int(len(conflict_keys)),
            "rows_in_conflicting_groups": int(len(conflict_rows)),
        }
    )
    for metric in (
        "valid_text_rows_before_deduplication",
        "duplicated_clean_text_groups",
        "same_label_duplicate_groups",
        "duplicate_occurrences_total",
        "same_label_duplicate_occurrences_removed",
        "conflicting_clean_text_groups",
        "rows_in_conflicting_groups",
    ):
        _add_count(audit_records, metric, audit["row_counts"][metric])

    print(
        "[5/8] Duplicate audit: "
        f"{len(duplicate_group_keys)} duplicate group(s), "
        f"{len(same_label_duplicate_keys)} same-label group(s), "
        f"{len(conflict_keys)} conflicting group(s)."
    )

    if conflict_keys and CONFLICT_POLICY == "error":
        message = (
            f"Found {len(conflict_keys)} normalized text group(s) with conflicting "
            "three-class labels. CONFLICT_POLICY='error' stops processing. "
            "Inspect data/processed/ib_duplicate_label_conflicts.csv and "
            "data/processed/ib_duplicate_label_conflict_summary.csv."
        )
        audit["status"] = "failed"
        audit["errors"].append(message)
        _write_audit(audit, audit_records)
        raise ValueError(message)

    decisions: dict[str, dict[str, str]] = {}
    if conflict_keys and CONFLICT_POLICY == "adjudicate":
        try:
            decisions = _load_adjudication_decisions(conflict_keys)
        except (FileNotFoundError, ValueError) as exc:
            message = str(exc)
            audit["status"] = "failed"
            audit["errors"].append(message)
            _write_audit(audit, audit_records)
            raise

    # Deduplicate the non-conflicting portion safely. Every remaining text
    # group has a single mapped label, so keeping its first source record is safe.
    non_conflict_df = valid_df[~valid_df["clean_text"].isin(conflict_keys)].copy()
    safe_dedup_df = non_conflict_df.drop_duplicates(
        subset="clean_text", keep="first"
    ).copy()
    rows_removed_same_label = int(len(non_conflict_df) - len(safe_dedup_df))
    conflict_excluded_frames: list[pd.DataFrame] = []
    adjudicated_frames: list[pd.DataFrame] = []
    resolution_records: list[dict[str, Any]] = []
    rows_collapsed_by_adjudication = 0

    for clean_text in conflict_keys:
        group = conflict_rows[conflict_rows["clean_text"] == clean_text].sort_values(
            "source_record"
        )
        source_labels = sorted(set(group["label"].astype(str)))
        source_records = ", ".join(map(str, group["source_record"].tolist()))

        if CONFLICT_POLICY == "exclude":
            action = "exclude"
            excluded = group.copy()
            excluded["resolution_action"] = "exclude"
            excluded["resolution_rationale"] = (
                "Excluded in full because the same clean_text has conflicting source-mapped labels."
            )
            conflict_excluded_frames.append(excluded)
            resolution_records.append(
                {
                    "clean_text": clean_text,
                    "source_records": source_records,
                    "group_rows": int(len(group)),
                    "source_labels": " | ".join(source_labels),
                    "action": action,
                    "adjudicated_label": "",
                    "rationale": "Unresolved conflicting annotations; entire group excluded by configured policy.",
                    "reviewer": "",
                    "retained_source_record": "",
                }
            )

        elif CONFLICT_POLICY == "adjudicate":
            decision = decisions[clean_text]
            action = decision["action"]
            if action == "exclude":
                excluded = group.copy()
                excluded["resolution_action"] = "exclude"
                excluded["resolution_rationale"] = decision["rationale"]
                conflict_excluded_frames.append(excluded)
                retained_source_record = ""
            else:
                # Retain one deterministic source record as the representative.
                # Preserve its original HS/Abusive values; only the experiment
                # target 'label' is changed and explicitly marked as adjudicated.
                representative = group.iloc[[0]].copy()
                representative.loc[:, "label"] = decision["adjudicated_label"]
                adjudicated_frames.append(representative)
                rows_collapsed_by_adjudication += int(len(group) - 1)
                retained_source_record = int(representative.iloc[0]["source_record"])

            resolution_records.append(
                {
                    "clean_text": clean_text,
                    "source_records": source_records,
                    "group_rows": int(len(group)),
                    "source_labels": " | ".join(source_labels),
                    "action": action,
                    "adjudicated_label": decision["adjudicated_label"],
                    "rationale": decision["rationale"],
                    "reviewer": decision["reviewer"],
                    "retained_source_record": retained_source_record,
                }
            )

    if conflict_excluded_frames:
        excluded_df = pd.concat(conflict_excluded_frames, ignore_index=True)
        excluded_df.to_csv(OUT_DIR / "ib_conflict_excluded_rows.csv", index=False)
    else:
        pd.DataFrame(
            columns=list(valid_df.columns) + ["resolution_action", "resolution_rationale"]
        ).to_csv(OUT_DIR / "ib_conflict_excluded_rows.csv", index=False)

    resolution_columns = [
        "clean_text", "source_records", "group_rows", "source_labels", "action",
        "adjudicated_label", "rationale", "reviewer", "retained_source_record",
    ]
    pd.DataFrame(resolution_records, columns=resolution_columns).to_csv(
        OUT_DIR / "ib_conflict_resolution_log.csv", index=False
    )

    final_parts = [safe_dedup_df]
    if adjudicated_frames:
        final_parts.extend(adjudicated_frames)
    dedup_df = pd.concat(final_parts, ignore_index=True)
    dedup_df = dedup_df.sort_values("source_record").reset_index(drop=True)

    # Preserve the existing five-column model-facing schema so downstream
    # training/preflight scripts remain compatible. Source-record provenance
    # and any manual decision are retained in the conflict-resolution log.
    output_columns = ["text_raw", "clean_text", "HS", "Abusive", "label"]
    dedup_df = dedup_df[output_columns].copy()

    rows_excluded_by_conflict = int(
        sum(len(frame) for frame in conflict_excluded_frames)
    )
    audit["row_counts"].update(
        {
            "rows_removed_by_same_label_deduplication": rows_removed_same_label,
            "rows_excluded_by_conflict_policy": rows_excluded_by_conflict,
            "conflict_groups_excluded": sum(
                1 for item in resolution_records if item["action"] == "exclude"
            ),
            "conflict_groups_adjudicated": sum(
                1 for item in resolution_records if item["action"] == "adjudicate"
            ),
            "rows_collapsed_by_adjudication": rows_collapsed_by_adjudication,
            "cleaned_rows_after_conflict_resolution_and_deduplication": int(len(dedup_df)),
            "cleaned_rows_after_deduplication": int(len(dedup_df)),
            "rows_removed_total_from_valid_text_set": int(len(valid_df) - len(dedup_df)),
        }
    )
    for metric in (
        "rows_removed_by_same_label_deduplication",
        "rows_excluded_by_conflict_policy",
        "conflict_groups_excluded",
        "conflict_groups_adjudicated",
        "rows_collapsed_by_adjudication",
        "cleaned_rows_after_conflict_resolution_and_deduplication",
        "cleaned_rows_after_deduplication",
        "rows_removed_total_from_valid_text_set",
    ):
        _add_count(audit_records, metric, audit["row_counts"][metric])

    if dedup_df["clean_text"].duplicated().any():
        message = "Internal error: duplicate clean_text values remain after conflict resolution."
        audit["status"] = "failed"
        audit["errors"].append(message)
        _write_audit(audit, audit_records)
        raise AssertionError(message)

    print(
        f"      Conflict policy '{CONFLICT_POLICY}': "
        f"excluded {rows_excluded_by_conflict} row(s), "
        f"adjudicated {audit['row_counts']['conflict_groups_adjudicated']} group(s)."
    )
    print(
        "      Final global deduplication: "
        f"{len(dedup_df)} unique text(s) remain; "
        f"{len(valid_df) - len(dedup_df)} valid row(s) removed or consolidated."
    )

    # Step 6: stratified 70/15/15 split. Deduplication has already occurred
    # across the complete eligible dataset, so a clean_text cannot cross splits.
    try:
        train_df, temporary_df = train_test_split(
            dedup_df,
            test_size=0.30,
            stratify=dedup_df["label"],
            random_state=RANDOM_STATE,
        )
        validation_df, test_df = train_test_split(
            temporary_df,
            test_size=0.50,
            stratify=temporary_df["label"],
            random_state=RANDOM_STATE,
        )
    except ValueError as exc:
        message = (
            "The stratified 70/15/15 split could not be created. Check whether "
            "each class has enough unique examples after deduplication. "
            f"Underlying error: {exc}"
        )
        audit["status"] = "failed"
        audit["errors"].append(message)
        _write_audit(audit, audit_records)
        raise ValueError(message) from exc

    split_frames = {
        "train": train_df,
        "validation": validation_df,
        "test": test_df,
    }
    for split_name, split_df in split_frames.items():
        split_df = split_df.reset_index(drop=True)
        split_frames[split_name] = split_df
        distribution = _class_distribution(split_df)
        audit["class_distributions"][split_name] = distribution
        audit["row_counts"][f"{split_name}_rows"] = int(len(split_df))
        _add_count(audit_records, f"{split_name}_rows", len(split_df))
        _add_distribution(audit_records, split_name, distribution)

    cleaned_distribution = _class_distribution(dedup_df)
    audit["class_distributions"]["cleaned_all"] = cleaned_distribution
    _add_distribution(audit_records, "cleaned_all", cleaned_distribution)
    audit["row_counts"]["total_split_rows"] = int(
        sum(len(frame) for frame in split_frames.values())
    )
    _add_count(audit_records, "total_split_rows", audit["row_counts"]["total_split_rows"])

    # Explicitly verify that no normalized text occurs in more than one split.
    train_texts = set(split_frames["train"]["clean_text"])
    validation_texts = set(split_frames["validation"]["clean_text"])
    test_texts = set(split_frames["test"]["clean_text"])
    cross_split_overlap = {
        "train_validation": len(train_texts & validation_texts),
        "train_test": len(train_texts & test_texts),
        "validation_test": len(validation_texts & test_texts),
    }
    audit["cross_split_clean_text_overlap"] = cross_split_overlap
    for pair, overlap_count in cross_split_overlap.items():
        _add_count(audit_records, f"cross_split_overlap_{pair}", overlap_count, section="leakage_check")
    if any(cross_split_overlap.values()):
        message = f"Unexpected clean_text overlap across splits: {cross_split_overlap}"
        audit["status"] = "failed"
        audit["errors"].append(message)
        _write_audit(audit, audit_records)
        raise AssertionError(message)

    if audit["row_counts"]["total_split_rows"] != len(dedup_df):
        message = "Split row counts do not sum to the deduplicated dataset size."
        audit["status"] = "failed"
        audit["errors"].append(message)
        _write_audit(audit, audit_records)
        raise AssertionError(message)

    print(
        "[6/8] Stratified split completed: "
        f"train={len(train_df)}, validation={len(validation_df)}, test={len(test_df)}."
    )
    print("      Cross-split clean_text overlap: 0")

    # Step 7: class weights from training data only.
    if set(train_df["label"].unique()) != set(CLASS_ORDER):
        message = (
            "The training split does not contain all three classes; class weights "
            "cannot be generated reliably. Check the post-deduplication class counts."
        )
        audit["status"] = "failed"
        audit["errors"].append(message)
        _write_audit(audit, audit_records)
        raise ValueError(message)

    weights_by_id = compute_class_weights(train_df["label"])
    id2label = {value: key for key, value in LABEL2ID.items()}
    weights_table = pd.DataFrame(
        [
            {"label": id2label[class_id], "weight": float(weight)}
            for class_id, weight in weights_by_id.items()
        ]
    )
    weights_table["_order"] = weights_table["label"].map(
        {label: index for index, label in enumerate(CLASS_ORDER)}
    )
    weights_table = weights_table.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    weights_table.to_csv(RESULTS_DIR / "ib_class_weights.csv", index=False)
    audit["class_weights"] = {
        row.label: float(row.weight) for row in weights_table.itertuples(index=False)
    }
    for label, weight in audit["class_weights"].items():
        audit_records.append(
            {
                "section": "class_weights",
                "split": "train_only",
                "label": label,
                "metric": "weight",
                "value": weight,
            }
        )
    print("[7/8] Training-only class weights:")
    print(weights_table.to_string(index=False))

    # Step 8: save the final datasets and completed audit.
    dedup_df.to_csv(OUT_DIR / "ib_cleaned.csv", index=False)
    split_frames["train"].to_csv(OUT_DIR / "ib_train.csv", index=False)
    split_frames["validation"].to_csv(OUT_DIR / "ib_val.csv", index=False)
    split_frames["test"].to_csv(OUT_DIR / "ib_test.csv", index=False)

    audit["status"] = "completed"
    audit["outputs"] = [
        "data/processed/ib_cleaned.csv",
        "data/processed/ib_train.csv",
        "data/processed/ib_val.csv",
        "data/processed/ib_test.csv",
        "data/processed/ib_duplicate_label_conflicts.csv",
        "data/processed/ib_duplicate_label_conflict_summary.csv",
        "data/processed/ib_duplicate_label_adjudication_template.csv",
        "data/processed/ib_conflict_excluded_rows.csv",
        "data/processed/ib_conflict_resolution_log.csv",
        "results/tables/ib_class_weights.csv",
        "results/tables/ib_preprocessing_audit.json",
        "results/tables/ib_preprocessing_audit.csv",
    ]
    _write_audit(audit, audit_records)

    print("[8/8] Processed datasets and preprocessing audit saved.")
    print("\nFinal cleaned class distribution:")
    for label in CLASS_ORDER:
        item = cleaned_distribution[label]
        print(f"  {label}: {item['count']} ({item['percentage']}%)")
    print("\nFinal split sizes:")
    for split_name in ("train", "validation", "test"):
        print(f"  {split_name}: {len(split_frames[split_name])}")
    print("\nDone. Stage 1 outputs are current; prior Stage 1 outputs were cleared at run start.")


if __name__ == "__main__":
    main()
