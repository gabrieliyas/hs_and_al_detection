"""Stage 1b: select a shared benchmark maximum sequence length.

This script runs AFTER Stage 1 preprocessing and BEFORE benchmark LR tuning,
Experiment I, or ITFT. It uses ONLY data/processed/ib_train.csv, tokenizes the
clean_text column with both registered backbone tokenizers without truncation,
and writes the shared benchmark-length config through experiment_utils.py.

Run from the repository root or from any working directory:
    python scripts/1b_select_benchmark_max_length.py

A pre-existing sequence_length.json is never overwritten. Archive the old
configuration and its associated run outputs manually before a fresh Stage 1b.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd


def _project_root() -> Path:
    """Resolve the repository root whether this file is in root or scripts/."""
    here = Path(__file__).resolve().parent
    return here.parent if here.name.lower() == "scripts" else here


PROJECT_ROOT = _project_root()
sys.path.insert(0, str(PROJECT_ROOT))

from experiment_utils import (  # noqa: E402
    BACKBONES,
    CONFIG_SCHEMA_VERSION,
    LENGTH_PERCENTILE,
    RUN_ID,
    SEQUENCE_LENGTH_CONFIG,
    choose_shared_max_length,
    save_length_selection_config,
    sha256_file,
)

DEFAULT_TRAIN_CSV = Path("data/processed/ib_train.csv")
DEFAULT_AUDIT_CSV = Path("results/tables/left_truncation_v2/benchmark_length_audit.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select max_length_benchmark from the benchmark training split only, "
            "using both registered tokenizer backbones."
        )
    )
    parser.add_argument(
        "--train-csv", type=Path, default=DEFAULT_TRAIN_CSV,
        help="Stage 1 training split (default: data/processed/ib_train.csv).",
    )
    parser.add_argument(
        "--output-config", type=Path, default=SEQUENCE_LENGTH_CONFIG,
        help="Shared sequence-length config output path.",
    )
    parser.add_argument(
        "--audit-csv", type=Path, default=DEFAULT_AUDIT_CSV,
        help="Human-readable per-backbone token-length audit CSV.",
    )
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Tokenizer batch size; affects speed, not results.")
    parser.add_argument(
        "--indobertweet-revision", default=None,
        help="Optional Hugging Face revision/commit for IndoBERTweet. Prefer a full commit hash.",
    )
    parser.add_argument(
        "--xlmr-revision", default=None,
        help="Optional Hugging Face revision/commit for XLM-RoBERTa-base. Prefer a full commit hash.",
    )
    return parser.parse_args()


def _resolve_tokenizer_revision(tokenizer: Any, requested_revision: str | None, backbone: str) -> str:
    """Return a resolved commit SHA, refusing an unpinned/unrecordable revision."""
    init_kwargs = getattr(tokenizer, "init_kwargs", {}) or {}
    resolved = (
        getattr(tokenizer, "_commit_hash", None)
        or init_kwargs.get("_commit_hash")
        or init_kwargs.get("commit_hash")
    )
    if isinstance(resolved, str) and resolved.strip():
        return resolved.strip()

    # Offline/local loading may not expose the resolved commit. Accept a
    # caller-supplied full SHA as an exact, auditable revision; do not mistake
    # a moving tag such as "main" for a pinned revision.
    if requested_revision and re.fullmatch(r"[0-9a-fA-F]{40}", requested_revision.strip()):
        return requested_revision.strip().lower()

    raise RuntimeError(
        f"Could not determine the resolved Hugging Face commit for {backbone}. "
        "For reproducibility, rerun with the tokenizer's full 40-character "
        "commit SHA via its --*-revision argument, or verify that the installed "
        "Transformers version exposes tokenizer.init_kwargs['_commit_hash']."
    )


def _load_train_texts(train_csv: Path) -> tuple[pd.DataFrame, list[str]]:
    if not train_csv.exists():
        raise FileNotFoundError(
            f"Training split not found: {train_csv}. Run Stage 1 ("
            "1_preprocess_benchmark.py) successfully before Stage 1b."
        )

    df = pd.read_csv(train_csv)
    required = {"clean_text", "label"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(
            f"{train_csv} lacks required column(s) {missing}. Stage 1 must write "
            "a clean_text field and canonical N/AL/HS labels."
        )
    if df.empty:
        raise ValueError(f"Training split is empty: {train_csv}")

    labels = set(df["label"].dropna().astype(str).unique())
    expected_labels = {"N", "AL", "HS"}
    if labels != expected_labels:
        raise ValueError(
            f"Training split labels must be exactly {sorted(expected_labels)}; "
            f"found {sorted(labels)}. Recheck Stage 1 label mapping/split."
        )
    if df["label"].isna().any():
        raise ValueError("Training split contains missing labels.")

    texts = df["clean_text"].astype("string").fillna("").str.strip().tolist()
    blank_count = sum(not text for text in texts)
    if blank_count:
        raise ValueError(
            f"Training split has {blank_count} empty clean_text value(s). "
            "Do not select a sequence length until Stage 1 output is corrected."
        )
    return df, texts


def _token_lengths(tokenizer: Any, texts: list[str], batch_size: int, backbone: str) -> list[int]:
    """Count tokenized lengths including special tokens, without truncation."""
    lengths: list[int] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            add_special_tokens=True,
            truncation=False,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        input_ids = encoded.get("input_ids")
        if input_ids is None or len(input_ids) != len(batch):
            raise RuntimeError(
                f"Tokenizer for {backbone} returned unexpected input_ids at rows "
                f"{start}:{start + len(batch)}."
            )
        lengths.extend(len(ids) for ids in input_ids)

    if len(lengths) != len(texts) or not lengths or any(n <= 0 for n in lengths):
        raise RuntimeError(f"Invalid token-length results for {backbone}.")
    return lengths


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be a positive integer")

    train_csv = args.train_csv if args.train_csv.is_absolute() else PROJECT_ROOT / args.train_csv
    output_config = args.output_config if args.output_config.is_absolute() else PROJECT_ROOT / args.output_config
    audit_csv = args.audit_csv if args.audit_csv.is_absolute() else PROJECT_ROOT / args.audit_csv

    # Fail before downloading tokenizers or doing expensive work if an old or
    # current config already exists. The utility performs a second guard at save.
    if output_config.exists():
        raise FileExistsError(
            f"Refusing to overwrite {output_config}. Archive the existing sequence-length "
            "config and its dependent runs, then rerun Stage 1b. Historical or current "
            "configs must never be silently reused."
        )
    if audit_csv.exists():
        raise FileExistsError(
            f"Refusing to overwrite {audit_csv}. Archive the existing audit file first."
        )

    df, texts = _load_train_texts(train_csv)
    train_sha256 = sha256_file(train_csv)
    print(f"Training examples: {len(df):,}")
    print(f"Training split SHA-256: {train_sha256}")
    print(f"Length-selection partition: benchmark_train (training split only)")
    print(f"Protocol percentile: p{LENGTH_PERCENTILE:g}")

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "Stage 1b requires Hugging Face Transformers. Install the project "
            "requirements or run: pip install transformers"
        ) from exc

    requested_revisions = {
        "indobertweet": args.indobertweet_revision,
        "xlmr": args.xlmr_revision,
    }
    lengths_by_backbone: dict[str, list[int]] = {}
    revision_by_backbone: dict[str, str] = {}
    tokenizer_class_by_backbone: dict[str, str] = {}

    for backbone, spec in BACKBONES.items():
        requested_revision = requested_revisions[backbone]
        print(f"\nLoading {spec.display_name} tokenizer: {spec.hf_model_id}")
        kwargs: dict[str, Any] = {"use_fast": True}
        if requested_revision:
            kwargs["revision"] = requested_revision
        try:
            tokenizer = AutoTokenizer.from_pretrained(spec.hf_model_id, **kwargs)
        except Exception as exc:
            raise RuntimeError(
                f"Unable to load tokenizer for {backbone} ({spec.hf_model_id}). "
                "Check the model ID, network/Hugging Face access, tokenizer files, "
                "and any configured revision."
            ) from exc

        revision_by_backbone[backbone] = _resolve_tokenizer_revision(
            tokenizer, requested_revision, backbone
        )
        tokenizer_class_by_backbone[backbone] = tokenizer.__class__.__name__
        lengths = _token_lengths(tokenizer, texts, args.batch_size, backbone)
        lengths_by_backbone[backbone] = lengths
        p95 = float(pd.Series(lengths).quantile(0.95, interpolation="linear"))
        print(
            f"  resolved revision: {revision_by_backbone[backbone]}\n"
            f"  tokenizer class: {tokenizer_class_by_backbone[backbone]}\n"
            f"  token lengths: min={min(lengths)}, median={pd.Series(lengths).median():.1f}, "
            f"p95={p95:.2f}, max={max(lengths)}"
        )

    selection = choose_shared_max_length(
        lengths_by_backbone,
        percentile=LENGTH_PERCENTILE,
        partition="benchmark_train",
    )
    print(f"\nSelected shared max_length_benchmark: {selection['max_length']}")
    print("Estimated truncation rate at selected limit:")
    for backbone in BACKBONES:
        print(
            f"  {backbone}: "
            f"{selection['truncation_rate_at_max_length'][backbone] * 100:.2f}%"
        )

    # Persist via the utility's strict schema/selection validation. It writes
    # contextual='left' and target-only='right' policies for later stages.
    saved_config = save_length_selection_config(selection, path=output_config)

    # Add provenance metadata without changing the utility-owned selection fields.
    doc = json.loads(saved_config.read_text(encoding="utf-8"))
    doc["benchmark_training_split_sha256"] = train_sha256
    doc["benchmark_training_rows"] = int(len(df))
    doc["benchmark_training_text_column"] = "clean_text"
    doc["tokenizer_model_ids"] = {b: BACKBONES[b].hf_model_id for b in BACKBONES}
    doc["tokenizer_resolved_revisions"] = revision_by_backbone
    doc["tokenizer_classes"] = tokenizer_class_by_backbone
    doc["token_count_includes_special_tokens"] = True
    doc["selection_script"] = Path(__file__).name
    doc["selection_script_version"] = "1.0"
    doc["tokenizer_batch_size"] = int(args.batch_size)
    saved_config.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Human-readable artifact for the methodology/results audit trail.
    audit_rows = []
    for backbone, spec in BACKBONES.items():
        series = pd.Series(lengths_by_backbone[backbone], dtype="int64")
        audit_rows.append({
            "run_id": RUN_ID,
            "partition": "benchmark_train",
            "backbone": backbone,
            "model_id": spec.hf_model_id,
            "tokenizer_revision": revision_by_backbone[backbone],
            "tokenizer_class": tokenizer_class_by_backbone[backbone],
            "n_training_examples": len(texts),
            "train_csv_sha256": train_sha256,
            "min_tokens": int(series.min()),
            "median_tokens": float(series.median()),
            "p95_tokens": float(series.quantile(0.95, interpolation="linear")),
            "max_tokens": int(series.max()),
            "max_length_benchmark": int(selection["max_length"]),
            "truncation_rate_at_max_length": float(selection["truncation_rate_at_max_length"][backbone]),
            "percentile": LENGTH_PERCENTILE,
            "token_count_includes_special_tokens": True,
        })
    audit_df = pd.DataFrame(audit_rows)
    audit_csv.parent.mkdir(parents=True, exist_ok=True)
    audit_df.to_csv(audit_csv, index=False)

    print(f"\nWrote shared length configuration: {saved_config}")
    print(f"Wrote human-readable audit: {audit_csv}")
    print("Truncation policy saved: target-only benchmark='right'; contextual inputs='left'.")
    print("Next required step: run preflight_tune_benchmark_lr.py before Experiment I.")


if __name__ == "__main__":
    main()
