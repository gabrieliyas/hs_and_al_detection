"""
experiment_utils.py

Shared utilities for the Indonesian hate-speech / abusive-language thesis pipeline.

Single sources of truth for labels, model/input construction, backbone IDs, the
20-stage pipeline/script registry, fair-comparison configuration audits, paired
bootstrap decisions, and versioned run manifests.

Critical protocol guarantees:
- Context-first inputs are truncated from the LEFT so the target post remains.
- The context choice uses paired OOF bootstrap intervals and the pre-declared
  parsimony rule; it must never be chosen from mean F1 alone.
- The Stage 11 backbone gate requires positive measured median latency for both
  backbones and records the result in a decision JSON carrying the run ID.
- Experiments from the old unversioned output paths cannot satisfy the new run
  manifest checks. The corrected sweep uses `results/runs/left_truncation_v2/`.

Design notes (terminology as revised for the thesis):
- Experiment I: Transformer backbone comparison (IndoBERTweet vs. XLM-RoBERTa-
  base), target-only, trained + evaluated only on the Ibrohim & Budi dataset.
  The earlier SVM+TF-IDF baseline is retired (see experiment_protocol.md).
- Backbones are a factor in every experiment. Experiment IDs, the C_best
  resolver, and the backbone decision rule below are all backbone-aware, and
  every IndoBERTweet ID is unchanged from the earlier single-backbone version.
- Experiment II: Context-Aware WITH Intermediate Task Fine-Tuning (ITFT) -- the
  model is first fine-tuned on Ibrohim & Budi (the intermediate task), then
  further fine-tuned on the custom contextual dataset (matched 4-parent
  subset), truncated to N parent messages.
- Experiment III: Context-Aware WITHOUT Intermediate Task Fine-Tuning -- fine-
  tuned on the custom contextual dataset only, skipping the intermediate
  Ibrohim & Budi fine-tuning stage.
  "C_best" must be resolved to a concrete context size (e.g. C3) as soon as
  the Experiment II ablation identifies it -- never stored as the literal
  string "best".
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


# ---------------------------------------------------------------------------
# 0. Label mapping -- single source of truth for label <-> integer id.
# Import this everywhere in the Transformer pipeline (and only if needed in the
# optional legacy SVM script) so no experiment accidentally uses a different int
# convention. The dataset file itself should keep the string label
# ("N"/"AL"/"HS") for human readability; convert to int only at the
# data-loading step below.
# ---------------------------------------------------------------------------

LABEL2ID = {"N": 0, "AL": 1, "HS": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# The custom contextual dataset's raw annotated export spells labels out in
# full ("neither", "abusive_language", "hate_speech") rather than using the
# short codes above. This is purely a raw-file vocabulary difference -- the
# canonical codes ("N"/"AL"/"HS") stay the single source of truth used
# everywhere else in the pipeline (including Experiment I's own
# 1_preprocess_benchmark.py output). 2_preprocess_contextual.py maps the raw
# spelling onto these codes once, at load time, before any validation or
# distribution check runs.
RAW_CTX_LABEL_MAP = {
    "neither": "N",
    "abusive_language": "AL",
    "hate_speech": "HS",
}


# ---------------------------------------------------------------------------
# 0.1 Custom contextual dataset -- column allowlist (single source of truth).
#
# Two lists, not one drop-list, because several "non-modelling" columns are
# still needed downstream -- just never as model input:
#   - MODELLING columns: what actually reaches the tokenizer/model.
#   - ANALYSIS_ONLY columns: kept in the master file, joined back in by
#     post_id at EVALUATION time (subgroup analysis, error analysis,
#     Chapter IV descriptive tables) -- never passed to any training script.
#
# Every contextual training/inference script (Stages 8 onward) must build its model input via
# load_modelling_view() below rather than dropping columns ad hoc. A
# blacklist ("drop everything except the label") fails open -- a new column
# added later would silently leak into the model. An allowlist fails safe.
# ---------------------------------------------------------------------------

CTX_ID_COLUMN = "post_id"

CTX_MODELLING_COLUMNS = [
    "parent_4", "parent_3", "parent_2", "parent_1", "target_text",
    "context_size_available", "label",
]
# Added by 2_preprocess_contextual.py to the matched subset only; not part
# of the raw schema, but a legitimate modelling-adjacent column once present
# (used for CV split assignment, never as a model feature).
CTX_FOLD_COLUMN = "fold"

CTX_ANALYSIS_ONLY_COLUMNS = [
    "context_needed", "HS", "AL", "target_group", "target_group_attribute",
    "context_type", "collection_method", "keyword_used", "post_date",
]

CTX_ALL_RAW_COLUMNS = [CTX_ID_COLUMN] + CTX_MODELLING_COLUMNS + CTX_ANALYSIS_ONLY_COLUMNS


def load_modelling_view(df: pd.DataFrame, include_fold: bool = True) -> pd.DataFrame:
    """
    Select ONLY the columns a training script is allowed to see, plus
    post_id (kept for joining predictions back to analysis-only columns
    at evaluation time, but must itself never be tokenized as a feature).

    Raises if any allowlisted column is missing -- fails loud rather than
    silently training on a partial schema.
    """
    wanted = [CTX_ID_COLUMN] + CTX_MODELLING_COLUMNS
    if include_fold:
        wanted.append(CTX_FOLD_COLUMN)
    missing = [c for c in wanted if c not in df.columns]
    if missing:
        raise KeyError(
            f"load_modelling_view: missing required column(s) {missing}. "
            "Check that the input file is ctx_matched_subset.csv (post-"
            "Stage-2) and that column names match CTX_MODELLING_COLUMNS "
            "in experiment_utils.py."
        )
    return df[wanted].copy()


# Per docs/project_summary.md's explicit C0-C4 definitions: context size n
# uses parent_n .. parent_1 (furthest back first, i.e. chronological order),
# then the target. C0 is target-only. Single source of truth -- both
# 8_timing_pilot.py and 9_train_e2_ablation.py build their input text
# through this function, so the two can never silently disagree on how a
# given C-level's input is constructed.
_CONTEXT_PARENT_COLUMNS = {
    0: [],
    1: ["parent_1"],
    2: ["parent_2", "parent_1"],
    3: ["parent_3", "parent_2", "parent_1"],
    4: ["parent_4", "parent_3", "parent_2", "parent_1"],
}


def sep_from_tokenizer(tokenizer) -> str:
    """
    Return the separator string for build_context_text() from a Hugging Face
    tokenizer's own special token ("[SEP]" for IndoBERTweet, "</s>" for
    XLM-R). Always use this instead of hard-coding "[SEP]", so the two
    backbones receive structurally equivalent inputs.
    """
    if tokenizer.sep_token is None:
        raise ValueError("Tokenizer has no sep_token; cannot build context text.")
    return f" {tokenizer.sep_token} "


def build_context_text(row, context_size: int, sep: str = " [SEP] ") -> str:
    """
    Build the input text for a given C-level (0-4) from one row of the
    matched contextual subset.

    `sep` should be the tokenizer's own special separator token at call
    time (e.g. f" {tokenizer.sep_token} "), not this default -- the
    default here is a plain placeholder for callers that just want to
    inspect the concatenated text (e.g. for a token-length pilot) without
    loading a tokenizer.
    """
    if context_size not in _CONTEXT_PARENT_COLUMNS:
        raise ValueError(f"context_size must be 0-4, got {context_size}")
    if hasattr(row, "index") and "context_size_available" in row.index and not pd.isna(row["context_size_available"]):
        available = int(row["context_size_available"])
        if context_size > available:
            raise ValueError(
                f"C{context_size} requested for post with only {available} available parent(s)"
            )
    cols = _CONTEXT_PARENT_COLUMNS[context_size]
    pieces = []
    for col in cols + ["target_text"]:
        value = row[col]
        if pd.isna(value) or not str(value).strip():
            raise ValueError(f"Missing/blank text in required context field {col!r}")
        pieces.append(str(value).strip())
    return sep.join(pieces)


def compute_class_weights(train_labels, label2id: dict = LABEL2ID) -> dict:
    """
    Compute 'balanced' class weights from a TRAINING partition's labels
    ONLY -- never from the full pre-split dataset. See
    dataset_split_and_weighting.md for the full reasoning and worked
    example on the Ibrohim & Budi dataset.

    Formula: w_c = n_train / (n_classes * count_c)

    Returns a dict keyed by integer label id (via label2id), in id order,
    ready to convert to a tensor for torch.nn.CrossEntropyLoss(weight=...).
    """
    counts = pd.Series(train_labels).value_counts()
    n_train = len(train_labels)
    n_classes = len(label2id)
    weights_by_name = {c: n_train / (n_classes * cnt) for c, cnt in counts.items()}
    weights_by_id = {label2id[name]: w for name, w in weights_by_name.items()}
    return dict(sorted(weights_by_id.items()))


# ---------------------------------------------------------------------------
# 1. Experiment ID registry
# ---------------------------------------------------------------------------

# 1.0 Backbone registry. Two candidate backbones are compared throughout.
# `hf_model_id` values are the public Hugging Face hub identifiers as far as
# I know them -- confirm both load (AutoTokenizer / AutoModelForSequence
# Classification) on first use, and record the exact revision hash in the
# run log for reproducibility.

@dataclass(frozen=True)
class BackboneSpec:
    key: str           # short key used in code, file names, and result tables
    id_token: str      # token used inside experiment IDs
    display_name: str  # name used in the thesis text
    hf_model_id: str


BACKBONES: dict[str, BackboneSpec] = {
    "indobertweet": BackboneSpec(
        "indobertweet", "IndoBERTweet", "IndoBERTweet",
        "indolem/indobertweet-base-uncased",
    ),
    "xlmr": BackboneSpec(
        "xlmr", "XLMR", "XLM-RoBERTa-base", "FacebookAI/xlm-roberta-base",
    ),
}
DEFAULT_BACKBONE = "indobertweet"

# The corrected sweep must use this namespace. Old root-level CSV/NPY files
# are retained only as historical evidence and must never satisfy run checks.
RUN_NAMESPACE = "left_truncation_v2"
RUN_ID = RUN_NAMESPACE
RUN_ROOT_PARTS = ("results", "runs", RUN_NAMESPACE)
SEQUENCE_LENGTH_CONFIG = Path("results/config/sequence_length.json")
LEARNING_RATE_CONFIG = Path("results/config/learning_rates.json")
BACKBONE_DECISION_PATH = Path("results/backbone_decision.json")
TRAINING_STRATEGY_DECISION_PATH = Path("results/training_strategy_decision.json")
RUN_MANIFEST_NAME = "_run_manifest.json"
CONFIG_SCHEMA_VERSION = "thesis-run-config-v3"
CONTEXT_RULE_VERSION = "paired-oof-depth-tradeoff-v2"
BACKBONE_RULE_VERSION = "v3-paired-context-and-latency-gate"
LENGTH_PERCENTILE = 95.0
LENGTH_ROUNDING_MULTIPLE = 8
MAX_SEQUENCE_LENGTH = 512

# Stage/script registry is synchronized with experiment_protocol.md and
# pipeline_revised.md. Stages 6/7 and 9/10 invoke one parameterized script
# once per backbone. The SVM file is deliberately not in the ordered stages.
PIPELINE_SCRIPTS = {
    1: "scripts/1_preprocess_benchmark.py",
    "1b": "scripts/1b_select_benchmark_max_length.py",
    2: "scripts/2_preprocess_contextual.py",
    "preflight": "scripts/preflight_tune_benchmark_lr.py",
    3: "scripts/3_train_e1_indobertweet.py",
    4: "scripts/4_train_e1_xlmr.py",
    5: "scripts/5_evaluate_e1.py",
    6: "scripts/6_train_itft.py",
    7: "scripts/6_train_itft.py",
    8: "scripts/8_timing_pilot.py",
    9: "scripts/9_train_e2_ablation.py",
    10: "scripts/9_train_e2_ablation.py",
    11: "scripts/11_select_backbone.py",
    12: "scripts/12_train_e2_retrain_full.py",
    13: "scripts/13_e2b_generalization.py",
    14: "scripts/14_train_e3_direct.py",
    15: "scripts/15_evaluate_all.py",
    16: "scripts/16_error_analysis.py",
    17: "scripts/17_train_final_models.py",
    18: "extension/backend",
    19: "extension/frontend",
    20: "scripts/20_test_extension_e2e.py",
}
OPTIONAL_LEGACY_SCRIPTS = ("scripts/3_train_e1.svm",)
PIPELINE_DIRECTORIES = {18: "extension/backend", 19: "extension/frontend"}
CONTEXTUAL_TRAINING_STAGES = (8, 9, 10, 12, 13, 14, 17)
DECISION_DEPENDENT_STAGES = (12, 13, 14, 15, 16, 17)

# Existing script path -> authoritative replacement. All runnable Python
# stages live under `scripts/`; the registry, manifest identity and audit use
# repository-relative POSIX paths so running from another working directory
# cannot create a different script identity.
SCRIPT_RENAMES = {
    "scripts/4_train_e1_indobertweet.py": "scripts/3_train_e1_indobertweet.py",
    "scripts/6_itft.py": "scripts/6_train_itft.py",
    "scripts/7_timing_pilot.py": "scripts/8_timing_pilot.py",
    "scripts/8_train_e2_ablation.py": "scripts/9_train_e2_ablation.py",
    "scripts/9_train_e2_retrain_full.py": "scripts/12_train_e2_retrain_full.py",
    "scripts/10_e2b_generalization.py": "scripts/13_e2b_generalization.py",
    "scripts/11_train_e3_context_only.py": "scripts/14_train_e3_direct.py",
    "scripts/12_evaluate_all.py": "scripts/15_evaluate_all.py",
    "scripts/13_error_analysis.py": "scripts/16_error_analysis.py",
    "scripts/14_train_final_models.py": "scripts/17_train_final_models.py",
}


def audit_pipeline_script_names(project_root: str | Path = ".", *, require_all: bool = True) -> dict:
    """Check that the project has the authoritative stage scripts on disk.

    `require_all=False` returns a report without raising, useful while the
    repository is being migrated. The final pre-run check should use the
    default `require_all=True`; old filenames are reported as migration
    leftovers and never count as satisfying a required stage.
    """
    root = Path(project_root)
    required = sorted({name for name in PIPELINE_SCRIPTS.values() if name.endswith(".py")})
    present = [name for name in required if (root / name).is_file()]
    missing = [name for name in required if name not in present]
    rename_leftovers = {
        old: new for old, new in SCRIPT_RENAMES.items()
        if (root / old).exists() and not (root / new).exists()
    }
    directories = {stage: name for stage, name in PIPELINE_DIRECTORIES.items() if (root / name).is_dir()}
    missing_directories = {stage: name for stage, name in PIPELINE_DIRECTORIES.items() if stage not in directories}
    report = {
        "complete": not missing and not rename_leftovers and not missing_directories,
        "required_scripts": required,
        "present": present,
        "missing": missing,
        "required_directories": PIPELINE_DIRECTORIES,
        "present_directories": directories,
        "missing_directories": missing_directories,
        "rename_leftovers": rename_leftovers,
        "optional_legacy_scripts": [name for name in OPTIONAL_LEGACY_SCRIPTS if (root / name).exists()],
        "run_id": RUN_ID,
    }
    if require_all and not report["complete"]:
        raise FileNotFoundError(
            "Pipeline script audit failed. Missing authoritative scripts: "
            f"{missing}. Missing required directories: {missing_directories}. "
            f"Files still using old names: {rename_leftovers}. "
            "Do not start the rerun until the script migration is complete."
        )
    return report


def assert_no_legacy_outputs(paths: Sequence[str | Path]) -> None:
    """Fail if legacy/unversioned outputs exist where a stage might resume them.

    Every migrated training/evaluation script must call this before checking
    whether an experiment is complete. The shared utility cannot prevent reuse
    by a legacy script that bypasses this function and reads a flat historical
    path directly. This function never deletes historical evidence; archive it
    outside the active output tree. New artifacts must be created beneath
    `results/runs/left_truncation_v2/` and pass manifest fingerprint checks.
    """
    found = [str(Path(path)) for path in paths if Path(path).exists()]
    if found:
        raise RuntimeError(
            "Legacy outputs exist and must not be used as resume signals: "
            + ", ".join(found)
            + ". Archive them, then rerun into the versioned run namespace."
        )


def get_backbone(key: str) -> BackboneSpec:
    """Return the BackboneSpec for `key`, failing loudly on a typo."""
    if key not in BACKBONES:
        raise KeyError(f"Unknown backbone {key!r}. Expected one of {sorted(BACKBONES)}.")
    return BACKBONES[key]


def experiment1_id(backbone: str = DEFAULT_BACKBONE) -> str:
    """Experiment I (target-only on Ibrohim & Budi), e.g. 'E1-XLMR-Target'."""
    return f"E1-{get_backbone(backbone).id_token}-Target"


def experiment2_id(n: int, backbone: str = DEFAULT_BACKBONE) -> str:
    """Experiment II (ITFT + context size n), e.g. 'E2-XLMR-ITFT-C4'."""
    if n not in range(5):
        raise ValueError(f"context size must be 0-4, got {n}")
    return f"E2-{get_backbone(backbone).id_token}-ITFT-C{n}"


def experiment3_id(n: int, backbone: str = DEFAULT_BACKBONE) -> str:
    """Experiment III (no ITFT + context size n), e.g. 'E3-XLMR-C4'."""
    if n not in range(5):
        raise ValueError(f"context size must be 0-4, got {n}")
    return f"E3-{get_backbone(backbone).id_token}-C{n}"


# Experiment I -- benchmark on Ibrohim & Budi only (target post, no
# conversational context). "svm" is a RETIRED entry kept only so that older
# scripts do not raise KeyError; do not use it in new code or in results.
EXPERIMENT1_IDS = {
    "svm": "E1-SVM-TFIDF",  # retired baseline
    "transformer": experiment1_id("indobertweet"),  # legacy key
    **{key: experiment1_id(key) for key in BACKBONES},
}

# Experiment II -- Context-Aware with Intermediate Task Fine-Tuning (ITFT):
# fine-tuned on Ibrohim & Budi first, then on the custom contextual dataset
# (matched 4-parent subset), context window ablation C0..C4.
# Legacy IndoBERTweet-only dicts are preserved; use experiment2_id() /
# experiment3_id() for any backbone-aware code.
EXPERIMENT2_IDS = {n: experiment2_id(n, "indobertweet") for n in range(5)}

# Experiment III -- Context-Aware without ITFT: fine-tuned on the custom
# contextual dataset only, no intermediate Ibrohim & Budi fine-tuning stage
EXPERIMENT3_IDS = {n: experiment3_id(n, "indobertweet") for n in range(5)}


def resolve_best_context_id(
    experiment2_scores: dict[int, float],
    backbone: str = DEFAULT_BACKBONE,
) -> tuple[int, str, str]:
    """Deprecated fail-closed shim: mean scores alone cannot resolve C_best.

    The function remains named for clear errors in old scripts, but it refuses
    to select a context from means. Use `select_context_size_from_oof()` with
    paired bootstrap intervals, then call `experiment2_id()` and
    `experiment3_id()` for the resolved numeric context.
    """
    get_backbone(backbone)
    if not experiment2_scores:
        raise ValueError("experiment2_scores must not be empty")
    raise RuntimeError(
        "resolve_best_context_id(score_dict) is disabled: Stage 11 must select "
        "C_best with paired pooled-OOF bootstrap intervals. Use "
        "select_context_size_from_oof(); downstream stages must load "
        "results/backbone_decision.json instead of recomputing C_best."
    )


def summarize_context_cv_scores(
    grid_df: pd.DataFrame,
    metric: str = "macro_f1",
) -> pd.DataFrame:
    """
    Summarize an Experiment II C0-C4 CV grid by context size.

    The expected grid contains one row per (condition, fold, seed) run.
    The returned table reports the mean and sample standard deviation of
    the selected metric across all fold/seed runs for each C-level.

    This function does not decide whether the grid is complete; callers
    that require a complete 5-fold x 2-seed grid should validate that
    separately before using the summary for model selection.
    """
    if "condition" not in grid_df.columns or metric not in grid_df.columns:
        raise KeyError(
            f"grid_df must contain `condition` and `{metric}` columns"
        )

    summary = (
        grid_df.groupby("condition")[metric]
        .agg(mean="mean", std=lambda x: x.std(ddof=1))
        .reset_index()
        .sort_values("condition")
        .reset_index(drop=True)
    )
    return summary


def select_best_context_from_cv(
    grid_df: pd.DataFrame,
    metric: str = "macro_f1",
    backbone: str = DEFAULT_BACKBONE,
    *,
    y_true=None,
    probabilities_by_context: Mapping[int, Sequence[np.ndarray]] | None = None,
    post_ids=None,
    post_ids_by_context: Mapping[int, Sequence[Any]] | None = None,
    n_bootstrap: int = 10_000,
    ci: float = 95.0,
    random_state: int = 42,
) -> tuple[int, str, str, pd.DataFrame]:
    """Select a context size using the paired-bootstrap parsimony rule.

    A mean-only selector is deliberately disallowed. Callers must supply
    `y_true`, the pooled out-of-fold probability matrix for each seed and C
    condition, and the post-ID alignment. This fail-closed behaviour prevents
    Stage 12 or another downstream script from silently reselecting context
    size based only on a CSV summary.

    `grid_df` is retained for a compact descriptive mean/std table; it is not
    the inferential source for choosing the context window.
    """
    summary = summarize_context_cv_scores(grid_df, metric=metric)
    if y_true is None or probabilities_by_context is None or post_ids is None:
        raise RuntimeError(
            "Mean-only context selection is disabled by experiment_protocol.md. "
            "Supply y_true, probabilities_by_context (pooled OOF probabilities "
            "for every seed and C0-C4), and post_ids, or load the saved Stage 11 "
            "decision with load_backbone_decision_document()."
        )
    selection = select_context_size_from_oof(
        y_true=y_true,
        probabilities_by_context=probabilities_by_context,
        post_ids=post_ids,
        post_ids_by_context=post_ids_by_context,
        backbone=backbone,
        n_bootstrap=n_bootstrap,
        ci=ci,
        random_state=random_state,
    )
    n = int(selection["selected_context"])
    return n, experiment2_id(n, backbone), experiment3_id(n, backbone), summary


# ---------------------------------------------------------------------------
# 1.1 Backbone parity: what must be identical, and what may differ.
#
# A paired backbone comparison is fair only if every training choice is held
# identical and just the model/tokenizer and the learning rate differ -- the
# learning rate being chosen from the SAME grid with the SAME tuning budget
# (not copied from one backbone to the other, because the optimal rate is
# model-dependent). Each training run should dump its resolved configuration
# as a dict with the keys below, and check_config_parity() is called before
# any cross-backbone comparison is reported.
# ---------------------------------------------------------------------------

PARITY_KEYS = (
    "run_id", "input_mode", "dataset_sha256", "fold_assignment_sha256",
    "context_order", "split_seed", "fold_column", "n_folds",
    "eval_seeds_exp1", "cv_seeds",
    "max_length_benchmark", "max_length_context", "truncation_side",
    "optimizer", "weight_decay", "warmup_ratio", "lr_scheduler",
    "max_epochs", "early_stopping_patience",
    "loss", "class_weighting_benchmark", "checkpoint_metric",
    "lr_search_grid", "lr_tuning_seed", "mixed_precision",
)

# Allowed to differ. Batch size is split into two keys so that gradient
# accumulation can keep the EFFECTIVE batch size identical if XLM-R needs a
# smaller per-device batch to fit in GPU memory.
BACKBONE_SPECIFIC_KEYS = (
    "hf_model_id", "learning_rate_benchmark", "learning_rate_context",
    "per_device_batch_size", "gradient_accumulation_steps",
)

# Shared learning-rate search space (applies to both backbones, both stages).
# Experiment I seeds and contextual CV seeds are frozen by the protocol; LR
# tuning uses a distinct seed recorded in learning_rates.json and must not
# overlap with either evaluation seed set.
# ASSUMPTION: this grid brackets the usual BERT-base fine-tuning range and
# adds 1e-5 because XLM-R is reported to prefer lower rates. It MUST contain
# the learning rate already used by the existing IndoBERTweet runs; if it does
# not, add that value to the grid (for both backbones) before tuning.
LR_SEARCH_GRID = (1e-5, 2e-5, 3e-5, 5e-5)
EVAL_SEEDS_EXP1 = (42, 123, 2024, 7, 99)
CV_SEEDS = (42, 123)


def _normalise(value):
    """Lists/tuples/arrays compare equal if their elements do."""
    if isinstance(value, (list, tuple, np.ndarray)):
        return tuple(_normalise(v) for v in value)
    return value


def _is_sha256_hex(value: Any) -> bool:
    """Return whether a value is a 64-character hexadecimal SHA-256 digest."""
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _is_positive_finite(value: Any) -> bool:
    """Return whether a value can be represented as finite positive float."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(numeric) and numeric > 0


def check_config_parity(
    cfg_a: Mapping,
    cfg_b: Mapping,
    *,
    input_mode: str | None = None,
) -> None:
    """Audit two backbone configs against the same-stage fairness contract.

    Target-only benchmark comparison (Experiment I) uses explicit right
    truncation and `max_length_benchmark`. Contextual runs use explicit left
    truncation and `max_length_context`. The maximum length for a later stage
    is not required during the earlier Experiment I comparison.
    """
    mode_a = cfg_a.get("input_mode")
    mode_b = cfg_b.get("input_mode")
    if mode_a != mode_b:
        raise ValueError(f"Input mode differs between backbones: {mode_a!r} vs {mode_b!r}")
    mode = input_mode or mode_a
    if mode not in {"target_only", "contextual"}:
        raise ValueError("input_mode must be explicitly set to 'target_only' or 'contextual'")
    if input_mode is not None and (mode_a != input_mode or mode_b != input_mode):
        raise ValueError(f"Both configs must declare input_mode={input_mode!r}")

    common_required = (
        "schema_version", "run_id", "input_mode", "dataset_sha256", "split_seed",
        "eval_seeds_exp1", "optimizer", "weight_decay", "warmup_ratio",
        "lr_scheduler", "max_epochs", "early_stopping_patience", "loss",
        "class_weighting_benchmark", "checkpoint_metric", "lr_search_grid",
        "lr_tuning_seed", "mixed_precision", "input_order", "truncation_side",
    )
    mode_required = (
        ("max_length_benchmark", "split_assignment_sha256")
        if mode == "target_only"
        else ("max_length_benchmark", "max_length_context", "fold_column", "fold_assignment_sha256", "n_folds", "cv_seeds", "context_order")
    )
    backbone_specific_required = (
        ("hf_model_id", "learning_rate_benchmark", "per_device_batch_size", "gradient_accumulation_steps")
        if mode == "target_only"
        else BACKBONE_SPECIFIC_KEYS
    )
    required = common_required + mode_required + backbone_specific_required
    missing = sorted({k for cfg in (cfg_a, cfg_b) for k in required if k not in cfg})
    if missing:
        raise KeyError(f"check_config_parity: missing key(s) {missing}")

    expected_side = "right" if mode == "target_only" else "left"
    for name, cfg in (("cfg_a", cfg_a), ("cfg_b", cfg_b)):
        if cfg["schema_version"] != CONFIG_SCHEMA_VERSION:
            raise ValueError(
                f"{name} uses schema {cfg['schema_version']!r}; expected {CONFIG_SCHEMA_VERSION!r}"
            )
        if cfg["run_id"] != RUN_ID:
            raise ValueError(f"{name} belongs to {cfg['run_id']!r}; expected {RUN_ID!r}")
        if cfg["truncation_side"] != expected_side:
            raise ValueError(
                f"{name}: {mode} inputs require truncation_side={expected_side!r}, "
                f"got {cfg['truncation_side']!r}"
            )
        for key in ("dataset_sha256", "split_assignment_sha256") if mode == "target_only" else ("dataset_sha256", "fold_assignment_sha256"):
            if not cfg.get(key):
                raise ValueError(f"{name} missing reproducibility audit field {key!r}")
        if tuple(cfg["eval_seeds_exp1"]) != EVAL_SEEDS_EXP1:
            raise ValueError(f"{name}: Experiment I seeds must be {EVAL_SEEDS_EXP1}")
        if mode == "target_only" and int(cfg["split_seed"]) != 42:
            raise ValueError(f"{name}: benchmark split_seed must be 42")
        if mode == "contextual":
            if int(cfg["n_folds"]) != 5 or tuple(cfg["cv_seeds"]) != CV_SEEDS:
                raise ValueError(f"{name}: contextual CV requires 5 folds and seeds {CV_SEEDS}")

    # Fields shared between backbones for the current input mode.
    parity_keys = list(common_required) + list(mode_required) + ["truncation_side"]
    problems = {
        k: (cfg_a[k], cfg_b[k])
        for k in parity_keys
        if _normalise(cfg_a[k]) != _normalise(cfg_b[k])
    }

    eff_a = cfg_a["per_device_batch_size"] * cfg_a["gradient_accumulation_steps"]
    eff_b = cfg_b["per_device_batch_size"] * cfg_b["gradient_accumulation_steps"]
    if eff_a != eff_b:
        problems["effective_batch_size"] = (eff_a, eff_b)

    for tag, cfg in (("a", cfg_a), ("b", cfg_b)):
        grid = _normalise(cfg["lr_search_grid"])
        lr_keys = ("learning_rate_benchmark",) if mode == "target_only" else ("learning_rate_benchmark", "learning_rate_context")
        for key in lr_keys:
            if not any(np.isclose(cfg[key], g) for g in grid):
                problems[f"{key} ({tag}) not in lr_search_grid"] = (cfg[key], grid)

    if problems:
        detail = "; ".join(f"{k}: {v[0]!r} vs {v[1]!r}" for k, v in problems.items())
        raise ValueError(f"Backbone configurations are not comparable -- {detail}")


def select_learning_rate(
    val_scores: Mapping[float, float],
    grid: Sequence[float] = LR_SEARCH_GRID,
) -> float:
    """
    Pick the learning rate with the highest validation macro-F1.

    Enforces the equal-tuning-budget rule: `val_scores` must contain a score
    for EVERY value in `grid`, so neither backbone can be evaluated on fewer
    candidates than the other. Exact ties go to the smaller learning rate
    (more conservative, deterministic).
    """
    got = sorted(val_scores)
    want = sorted(grid)
    if len(got) != len(want) or not np.allclose(got, want):
        raise ValueError(
            f"Equal-budget rule violated: scores given for {got}, "
            f"but the shared grid is {want}."
        )
    return float(max(got, key=lambda lr: (val_scores[lr], -lr)))


def choose_shared_max_length(
    lengths_by_backbone: Mapping[str, Sequence[int]],
    percentile: float = 95.0,
    multiple_of: int = 8,
    hard_cap: int = 512,
    *,
    partition: str,
) -> dict:
    """
    Choose ONE max_length shared by both backbones for a given stage.

    Args:
        lengths_by_backbone: backbone key -> token counts (including special
            tokens) of the stage's inputs under THAT backbone's tokenizer.
            For the benchmark stage use the training partition only; for the
            contextual stage use the concatenated C4 inputs (the longest
            condition) of the pilot fold or the matched subset.
        percentile: coverage target (default 95th percentile).
        multiple_of: round up to a multiple of this (GPU-friendly shapes).
        hard_cap: the models' position limit.

    The value is the LARGER of the two backbones' percentile lengths, so
    neither tokenizer is truncated more than the coverage target. The
    result is frozen in the run configuration and used identically for both
    backbones (PARITY_KEYS).

    Returns a dict with `max_length`, the per-backbone percentile lengths,
    and the per-backbone truncation rate at the chosen value.
    """
    allowed_partitions = {"benchmark_train", "contextual_c4"}
    if partition not in allowed_partitions:
        raise ValueError(
            "Length selection must explicitly declare partition='benchmark_train' "
            "or partition='contextual_c4'. Benchmark length selection may use only "
            "the benchmark training split; contextual length selection uses C4 inputs "
            "for the pre-declared training/pilot population. Validation/test/held-out "
            "benchmark text is prohibited."
        )
    if set(lengths_by_backbone) != set(BACKBONES):
        raise ValueError(
            f"Token lengths are required for every backbone {sorted(BACKBONES)}, "
            f"got {sorted(lengths_by_backbone)}."
        )
    raw_arrays = {b: np.asarray(v, dtype=float) for b, v in lengths_by_backbone.items()}
    invalid_lengths = {
        b: a for b, a in raw_arrays.items()
        if a.ndim != 1 or len(a) == 0 or not np.isfinite(a).all()
        or (a <= 0).any() or not np.equal(a, np.floor(a)).all()
    }
    if invalid_lengths:
        raise ValueError("Each tokenizer's token lengths must be a non-empty vector of positive finite integers")
    arrays = {b: a.astype(int) for b, a in raw_arrays.items()}
    if (percentile, multiple_of, hard_cap) != (LENGTH_PERCENTILE, LENGTH_ROUNDING_MULTIPLE, MAX_SEQUENCE_LENGTH):
        raise ValueError(
            "Length selection must follow the frozen protocol rule: "
            f"p{LENGTH_PERCENTILE:g}, round to {LENGTH_ROUNDING_MULTIPLE}, cap at {MAX_SEQUENCE_LENGTH}."
        )
    pct = {b: float(np.percentile(a, percentile)) for b, a in arrays.items()}
    chosen = int(np.ceil(max(pct.values()) / multiple_of) * multiple_of)
    chosen = min(chosen, hard_cap)
    return {
        "run_id": RUN_ID,
        "partition": partition,
        "max_length": chosen,
        "percentile": percentile,
        "percentile_tokens": pct,
        "truncation_rate_at_max_length": {
            b: float(np.mean(a > chosen)) for b, a in arrays.items()
        },
        "hard_cap_hit": chosen == hard_cap,
    }


def save_length_selection_config(
    selection: Mapping[str, Any],
    path: str | Path = SEQUENCE_LENGTH_CONFIG,
) -> Path:
    """Persist a validated benchmark or contextual length selection.

    Stage 1b writes `partition="benchmark_train"`. Stage 8 calls this again
    with `partition="contextual_c4"`; that call must find a valid benchmark
    selection already saved, then merge the contextual fields without altering
    the benchmark choice. Historical-schema files are rejected, not upgraded.
    """
    partition = selection.get("partition")
    if partition not in {"benchmark_train", "contextual_c4"}:
        raise ValueError("selection must declare partition='benchmark_train' or 'contextual_c4'")
    chosen = selection.get("max_length")
    if isinstance(chosen, bool) or not isinstance(chosen, (int, np.integer)):
        raise ValueError("selected max_length must be an integer")
    if not 8 <= int(chosen) <= MAX_SEQUENCE_LENGTH or int(chosen) % LENGTH_ROUNDING_MULTIPLE:
        raise ValueError("selected max_length must be 8..512 and rounded to a multiple of 8")
    if float(selection.get("percentile", -1)) != LENGTH_PERCENTILE:
        raise ValueError("length selection must use the protocol's 95th percentile")
    pct = selection.get("percentile_tokens")
    rates = selection.get("truncation_rate_at_max_length")
    if not isinstance(pct, Mapping) or set(pct) != set(BACKBONES):
        raise ValueError("selection must record p95 token lengths for both backbones")
    if not isinstance(rates, Mapping) or set(rates) != set(BACKBONES):
        raise ValueError("selection must record truncation rates for both backbones")
    try:
        pct_values = {b: float(pct[b]) for b in BACKBONES}
        rate_values = {b: float(rates[b]) for b in BACKBONES}
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid percentile/truncation statistics") from exc
    if any(not math.isfinite(v) or v <= 0 for v in pct_values.values()):
        raise ValueError("p95 token lengths must be finite and positive")
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in rate_values.values()):
        raise ValueError("truncation rates must be finite fractions in 0..1")
    calculated = min(
        int(math.ceil(max(pct_values.values()) / LENGTH_ROUNDING_MULTIPLE) * LENGTH_ROUNDING_MULTIPLE),
        MAX_SEQUENCE_LENGTH,
    )
    if calculated != int(chosen):
        raise ValueError(f"selected max_length {chosen} disagrees with saved p95 values (expected {calculated})")
    path = Path(path)
    if partition == "benchmark_train":
        # Do not silently preserve an old contextual value when restarting Stage 1b.
        if path.exists():
            old = load_json_config(path)
            if old.get("schema_version") != CONFIG_SCHEMA_VERSION or old.get("run_id") != RUN_ID:
                raise ValueError("Existing sequence-length config is historical; archive it before Stage 1b")
            if old.get("benchmark_length_source_partition") != "benchmark_train":
                raise ValueError("Existing sequence-length config has no valid benchmark training partition declaration")
            raise FileExistsError("sequence_length.json already exists for this run; archive it before regenerating Stage 1b")
        doc = {
            "schema_version": CONFIG_SCHEMA_VERSION,
            "run_id": RUN_ID,
            "truncation_side": "left",
            "target_only_truncation_side": "right",
            "benchmark_length_source_partition": "benchmark_train",
            "max_length_benchmark": int(chosen),
            "benchmark_length_percentile": LENGTH_PERCENTILE,
            "benchmark_length_percentile_tokens": pct_values,
            "benchmark_truncation_rate_at_max_length": rate_values,
            "benchmark_hard_cap_hit": bool(selection.get("hard_cap_hit", int(chosen) == MAX_SEQUENCE_LENGTH)),
        }
    else:
        if not path.exists():
            raise FileNotFoundError("Stage 8 cannot save contextual length before Stage 1b has saved benchmark length")
        doc = load_json_config(path, required_keys=[
            "schema_version", "run_id", "benchmark_length_source_partition", "max_length_benchmark",
            "benchmark_length_percentile", "benchmark_length_percentile_tokens", "benchmark_truncation_rate_at_max_length",
            "truncation_side", "target_only_truncation_side",
        ])
        if doc["schema_version"] != CONFIG_SCHEMA_VERSION or doc["run_id"] != RUN_ID:
            raise ValueError("Existing benchmark length config is historical; archive it and rerun Stage 1b")
        if doc["benchmark_length_source_partition"] != "benchmark_train":
            raise ValueError("Benchmark max_length was not selected from the training partition only")
        if "context_length_source_partition" in doc:
            raise FileExistsError("Contextual length is already recorded; archive config before rerunning Stage 8")
        doc.update({
            "context_length_source_partition": "contextual_c4",
            "max_length_context": int(chosen),
            "context_length_percentile": LENGTH_PERCENTILE,
            "context_length_percentile_tokens": pct_values,
            "context_truncation_rate_at_max_length": rate_values,
            "context_hard_cap_hit": bool(selection.get("hard_cap_hit", int(chosen) == MAX_SEQUENCE_LENGTH)),
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(doc), indent=2, sort_keys=True), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# 2. Evaluation helpers
# ---------------------------------------------------------------------------

def ensemble_seed_probabilities(prob_arrays: list[np.ndarray]) -> np.ndarray:
    """
    Average softmax probability arrays across multiple random-seed runs and
    return the argmax as a single representative prediction set.

    Use this to build each side of a paired_bootstrap_ci call when a model
    was trained with multiple seeds (e.g. IndoBERTweet vs. XLM-R in
    Experiment I) -- averaging probabilities uses information from every
    seed rather than arbitrarily picking one run's predictions.
    Report per-seed mean +/- std (via summarize_seed_runs) separately as
    the training-stability statistic; use this ensembled prediction set
    specifically for the head-to-head significance test.

    Args:
        prob_arrays: list of (n_examples, n_classes) arrays, one per seed,
            all from the same fixed test set.

    Returns:
        (n_examples,) array of predicted class ids.
    """
    stacked = np.stack(prob_arrays, axis=0)  # (n_seeds, n_examples, n_classes)
    mean_probs = stacked.mean(axis=0)
    return mean_probs.argmax(axis=1)


def paired_bootstrap_ci(
    y_true,
    preds_a,
    preds_b,
    metric_fn=None,
    n_bootstrap: int = 10_000,
    ci: float = 95.0,
    random_state: int = 42,
) -> dict:
    """
    Bootstrap CI for the difference in a metric (default: macro F1) between two
    models' predictions on the SAME test set (paired comparison).

    Resamples test examples with replacement `n_bootstrap` times, recomputes the
    metric for both models' predictions on each resample, and reports a CI for
    (metric_b - metric_a). If the CI excludes 0, treat the difference as
    statistically credible at the chosen confidence level.

    Use this for:
    - Experiment I: IndoBERTweet-Target (preds_a) vs. XLM-R-Target (preds_b) on
      the fixed test set, using seed-ensembled predictions. (For the
      per-class CIs as well, use paired_bootstrap_f1_ci below, which gives
      the identical macro-F1 interval.)
    - Experiment III: pooled out-of-fold predictions for C0 (preds_a) vs. C_best (preds_b).
    """
    if metric_fn is None:
        metric_fn = lambda y, p: f1_score(y, p, average="macro")

    rng = np.random.default_rng(random_state)
    y_true = np.asarray(y_true)
    preds_a = np.asarray(preds_a)
    preds_b = np.asarray(preds_b)
    n = len(y_true)

    diffs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        diffs[i] = metric_fn(y_true[idx], preds_b[idx]) - metric_fn(y_true[idx], preds_a[idx])

    alpha = (100 - ci) / 2
    lower, upper = np.percentile(diffs, [alpha, 100 - alpha])
    observed_diff = metric_fn(y_true, preds_b) - metric_fn(y_true, preds_a)

    return {
        "observed_diff": float(observed_diff),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "ci_level": ci,
        "n_bootstrap": n_bootstrap,
        "significant": not (lower <= 0 <= upper),
    }


def summarize_seed_runs(metric_values: list[float]) -> dict:
    """
    Aggregate a metric (e.g. macro F1) across multiple random seeds for a
    single condition. Use this instead of reporting a single-seed number
    for any transformer result.
    """
    arr = np.asarray(metric_values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "n_seeds": len(arr),
        "values": arr.tolist(),
    }


def per_class_f1(y_true, y_pred) -> dict[str, float]:
    """
    Per-class F1 keyed by class code ("N", "AL", "HS"). Inputs are integer
    class ids (LABEL2ID), e.g. the argmax of a softmax output.
    """
    scores = f1_score(
        y_true, y_pred, labels=list(range(len(ID2LABEL))),
        average=None, zero_division=0,
    )
    return {ID2LABEL[i]: float(s) for i, s in enumerate(scores)}


def seed_metrics_table(y_true, prob_arrays: list[np.ndarray], seeds=None) -> pd.DataFrame:
    """
    One row per random seed: accuracy, macro/weighted F1 and per-class F1,
    computed on the same fixed test set. This is the "per-seed results" table
    for Experiment I.

    Args:
        y_true: (n_examples,) integer class ids.
        prob_arrays: one (n_examples, n_classes) softmax array per seed.
        seeds: the seed values (for the `seed` column); defaults to 0..k-1.
    """
    seeds = list(seeds) if seeds is not None else list(range(len(prob_arrays)))
    if len(seeds) != len(prob_arrays):
        raise ValueError("`seeds` and `prob_arrays` must have the same length")

    rows = []
    for seed, probs in zip(seeds, prob_arrays):
        preds = np.asarray(probs).argmax(axis=1)
        row = {
            "seed": seed,
            "accuracy": float(accuracy_score(y_true, preds)),
            "macro_f1": float(f1_score(y_true, preds, average="macro")),
            "weighted_f1": float(f1_score(y_true, preds, average="weighted")),
        }
        row.update({f"f1_{c}": v for c, v in per_class_f1(y_true, preds).items()})
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_seed_metrics(table: pd.DataFrame) -> pd.DataFrame:
    """Mean and sample standard deviation (ddof=1) of every column of
    seed_metrics_table() except `seed`; one row per metric."""
    metrics = [c for c in table.columns if c != "seed"]
    return pd.DataFrame({
        "mean": table[metrics].mean(),
        "std": table[metrics].std(ddof=1),
    })


def _f1_per_class_vector(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """Per-class F1 via a confusion matrix (a 0 where a class has no support
    and no predictions) -- fast enough for 10,000 bootstrap resamples."""
    cm = np.bincount(y_true * n_classes + y_pred, minlength=n_classes ** 2)
    cm = cm.reshape(n_classes, n_classes).astype(float)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    denom = 2 * tp + fp + fn
    return np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)


def paired_bootstrap_f1_ci(
    y_true,
    preds_a,
    preds_b,
    n_bootstrap: int = 10_000,
    ci: float = 95.0,
    random_state: int = 42,
) -> dict:
    """
    Paired bootstrap CI for the difference (b - a) in macro-F1 AND in each
    class's F1, from one shared stream of resamples.

    The macro-F1 interval is identical to paired_bootstrap_ci() with the
    same `random_state` (same resampling stream, same metric); this function
    only adds the per-class intervals and runs faster.

    Interpretation: the macro-F1 interval is the single pre-declared test.
    The per-class intervals are descriptive (they show WHERE a difference
    arises) and are not corrected for multiple comparisons, so do not treat
    a lone per-class interval excluding 0 as confirmatory.

    Inputs are integer class ids (LABEL2ID). Returns
    {"macro_f1": {...}, "per_class": {"N": {...}, "AL": {...}, "HS": {...}}},
    each entry having the same keys as paired_bootstrap_ci().
    """
    y_true = np.asarray(y_true, dtype=int)
    preds_a = np.asarray(preds_a, dtype=int)
    preds_b = np.asarray(preds_b, dtype=int)
    k = len(ID2LABEL)
    for name, arr in (("y_true", y_true), ("preds_a", preds_a), ("preds_b", preds_b)):
        if arr.min() < 0 or arr.max() >= k:
            raise ValueError(f"{name} must contain class ids in 0..{k - 1}")

    rng = np.random.default_rng(random_state)
    n = len(y_true)
    diffs = np.empty((n_bootstrap, k + 1))  # columns 0..k-1 per class, k = macro
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        fa = _f1_per_class_vector(y_true[idx], preds_a[idx], k)
        fb = _f1_per_class_vector(y_true[idx], preds_b[idx], k)
        diffs[i, :k] = fb - fa
        diffs[i, k] = fb.mean() - fa.mean()

    obs_a = _f1_per_class_vector(y_true, preds_a, k)
    obs_b = _f1_per_class_vector(y_true, preds_b, k)
    observed = np.append(obs_b - obs_a, obs_b.mean() - obs_a.mean())

    alpha = (100 - ci) / 2

    def pack(col: int) -> dict:
        lower, upper = np.percentile(diffs[:, col], [alpha, 100 - alpha])
        return {
            "observed_diff": float(observed[col]),
            "ci_lower": float(lower),
            "ci_upper": float(upper),
            "ci_level": ci,
            "n_bootstrap": n_bootstrap,
            "significant": bool(not (lower <= 0 <= upper)),
        }

    return {
        "macro_f1": pack(k),
        "per_class": {ID2LABEL[c]: pack(c) for c in range(k)},
    }


def compare_backbones_exp1(
    y_true,
    probs_a: list[np.ndarray],
    probs_b: list[np.ndarray],
    name_a: str = "indobertweet",
    name_b: str = "xlmr",
    seeds=None,
    n_bootstrap: int = 10_000,
    ci: float = 95.0,
    random_state: int = 42,
) -> dict:
    """
    Experiment I report for two backbones on the SAME fixed test partition.

    Two statistics answering two different questions, reported together:
    - per-seed results and their mean +/- std (training stability), and
    - a paired bootstrap on seed-ENSEMBLED predictions (head-to-head test),
      for macro-F1 and for each class's F1. Difference direction is
      name_b minus name_a.

    Returns a dict with the per-seed tables, their summaries, the ensembled
    macro/per-class F1 of each model, and the bootstrap result.
    """
    table_a = seed_metrics_table(y_true, probs_a, seeds)
    table_b = seed_metrics_table(y_true, probs_b, seeds)
    ens_a = ensemble_seed_probabilities(probs_a)
    ens_b = ensemble_seed_probabilities(probs_b)

    return {
        "difference": f"{name_b} minus {name_a}",
        "per_seed": {name_a: table_a, name_b: table_b},
        "per_seed_summary": {
            name_a: summarize_seed_metrics(table_a),
            name_b: summarize_seed_metrics(table_b),
        },
        "ensemble_macro_f1": {
            name_a: float(f1_score(y_true, ens_a, average="macro")),
            name_b: float(f1_score(y_true, ens_b, average="macro")),
        },
        "ensemble_per_class_f1": {
            name_a: per_class_f1(y_true, ens_a),
            name_b: per_class_f1(y_true, ens_b),
        },
        "bootstrap": paired_bootstrap_f1_ci(
            y_true, ens_a, ens_b,
            n_bootstrap=n_bootstrap, ci=ci, random_state=random_state,
        ),
    }


# ---------------------------------------------------------------------------
# 2.1 Pre-declared decision rules (Stage 11 backbone gate; training strategy).
#
# Both rules are deliberately mechanical so the outcome cannot be steered by
# looking at results first. Commit this file BEFORE any XLM-R Experiment II
# run starts; the commit is the evidence that the rule was declared in
# advance. The threshold values below are declared ASSUMPTIONS (judgement
# calls, not established standards) -- change them now, not after seeing data.
# ---------------------------------------------------------------------------

BACKBONE_RULE_VERSION = "v3-paired-context-and-latency-gate"
LATENCY_TOLERANCE = 0.10  # latency differences within 10% count as "equal"
DEFAULT_STRATEGY_ON_TIE = "direct"  # parsimony: skip ITFT unless it credibly helps



def summarize_latency_measurements(
    measurements_ms: Mapping[str, Sequence[float]],
    *,
    measurement_machine: str,
    min_repeats: int = 20,
) -> dict:
    """Summarize repeated single-post inference latencies for Stage 8/11.

    Measurements must be collected on the same intended backend machine with
    the same warm-up and timing protocol. Each backbone needs at least
    `min_repeats` finite, strictly positive observations.
    """
    if not isinstance(measurement_machine, str) or not measurement_machine.strip():
        raise ValueError("measurement_machine must identify the intended inference machine")
    if set(measurements_ms) != set(BACKBONES):
        raise ValueError(f"Latency measurements must contain exactly {sorted(BACKBONES)}")
    out = {}
    for backbone, values in measurements_ms.items():
        arr = np.asarray(values, dtype=float)
        if arr.ndim != 1 or len(arr) < min_repeats:
            raise ValueError(f"{backbone}: need at least {min_repeats} latency measurements")
        if not np.isfinite(arr).all() or (arr <= 0).any():
            raise ValueError(f"{backbone}: latency samples must be finite and > 0 ms")
        out[backbone] = {
            "n": int(len(arr)),
            "median_ms": float(np.median(arr)),
            "mean_ms": float(np.mean(arr)),
            "p95_ms": float(np.percentile(arr, 95)),
            "min_ms": float(np.min(arr)),
            "max_ms": float(np.max(arr)),
        }
    return {
        "run_id": RUN_ID,
        "measurement_machine": measurement_machine.strip(),
        "backbones": out,
        "median_ms": {b: out[b]["median_ms"] for b in BACKBONES},
    }


def select_backbone(
    y_true,
    preds_c_best: Mapping[str, np.ndarray],
    preds_c0: Mapping[str, np.ndarray],
    latency_ms: Mapping[str, float] | None = None,
    exp1_leader: str | None = None,
    n_bootstrap: int = 10_000,
    ci: float = 95.0,
    random_state: int = 42,
    context_decisions: Mapping[str, Mapping[str, Any]] | None = None,
    context_sizes_by_backbone: Mapping[str, int] | None = None,
    post_ids: Sequence[Any] | None = None,
    latency_machine: str | None = None,
    grid_audits: Mapping[str, Mapping[str, Any]] | None = None,
    post_ids_by_prediction: Mapping[str, Mapping[str, Sequence[Any]]] | None = None,
    expected_n: int | None = 900,
) -> dict:
    """
    Stage 11: choose the backbone used by Stages 12-17, by this fixed rule.

    Inputs are POOLED OUT-OF-FOLD, seed-ensembled predictions (integer class
    ids) on the same 900 matched examples, from Experiment II-A:
      - preds_c_best[b]: backbone b's predictions at ITS OWN C_best,
      - preds_c0[b]:     backbone b's predictions at C0 (target-only mode).
    Build them with pool_out_of_fold_probabilities() ->
    ensemble_seed_probabilities().

    Rule (v2):
      1. PRIMARY: paired bootstrap on macro-F1 at C_best (difference =
         xlmr - indobertweet). If the 95% CI excludes 0, the higher backbone
         is the leader.
      2. GUARD: the leader must not be CREDIBLY worse in target-only mode
         (C0): its C0 difference CI must not lie entirely on the losing side.
         A failed guard means "no clear winner".
      3. TIE-BREAK (no clear winner): the backbone with the lower measured
         median inference latency, if the relative difference exceeds
         LATENCY_TOLERANCE; otherwise the default, IndoBERTweet.

    `latency_ms` is mandatory and must contain finite positive median latency
    in milliseconds for both backbones. `context_decisions`, the 50-run grid
    audit for each backbone, and ordered post IDs for both the C_best and C0
    prediction arrays are also mandatory. These checks tie the gate to the
    exact audited Experiment II-A grid, not just to summary scores.

    Experiment I is NOT an input to the decision (it is target-only on a
    different dataset, whereas deployment is conversation-aware on X); pass
    `exp1_leader` only to record whether the two agree, for the discussion.

    Returns a JSON-serialisable dict; persist it with
    save_backbone_decision() so Stages 12-17 read the decision from a file.
    """
    for name, preds in (("preds_c_best", preds_c_best), ("preds_c0", preds_c0)):
        if set(preds) != set(BACKBONES):
            raise ValueError(f"{name} must contain exactly {sorted(BACKBONES)}")
    # Protocol guard: never silently default when latency is missing. This
    # measurement is required even when the primary comparison is decisive,
    # so the decision artifact is complete and auditable.
    if latency_ms is None:
        raise ValueError(
            "Stage 11 requires measured median inference latency for both "
            "backbones; pass latency_ms={'indobertweet': ..., 'xlmr': ...}."
        )
    if set(latency_ms) != set(BACKBONES):
        raise ValueError(f"latency_ms must contain exactly {sorted(BACKBONES)}")
    latency_ms = {k: float(v) for k, v in latency_ms.items()}
    invalid_latency = {
        k: v for k, v in latency_ms.items()
        if not math.isfinite(v) or v <= 0
    }
    if invalid_latency:
        raise ValueError(
            "Stage 11 latency values must be finite, positive median milliseconds: "
            f"{invalid_latency}"
        )
    if context_decisions is None or set(context_decisions) != set(BACKBONES):
        raise ValueError(
            "Stage 11 must include a saved/validated context decision for each "
            f"backbone: {sorted(BACKBONES)}"
        )
    if context_sizes_by_backbone is None or set(context_sizes_by_backbone) != set(BACKBONES):
        raise ValueError(
            "Stage 11 requires the actual context size used to generate each C_best prediction map"
        )
    for backbone, ctx_decision in context_decisions.items():
        validate_context_selection_document(ctx_decision, expected_backbone=backbone)
        selected = ctx_decision["selected_context"]
        if int(context_sizes_by_backbone[backbone]) != int(selected):
            raise ValueError(
                f"C_best predictions for {backbone} used C{context_sizes_by_backbone[backbone]}, "
                f"but its context decision selects C{selected}"
            )
    if not isinstance(latency_machine, str) or not latency_machine.strip():
        raise ValueError("Stage 11 must record the intended latency measurement machine")
    if grid_audits is None or set(grid_audits) != set(BACKBONES):
        raise ValueError("Stage 11 requires a successful audited Experiment II-A grid for both backbones")
    grid_shared_fields = ("dataset_sha256", "fold_assignment_sha256", "max_length_context", "truncation_side")
    for field in grid_shared_fields:
        values = {str(grid_audits[b].get(field)) for b in BACKBONES} if grid_audits is not None and set(grid_audits) == set(BACKBONES) else set()
        if len(values) != 1:
            raise ValueError(f"Stage 11 grids differ between backbones on shared field {field!r}")
    for backbone, audit in grid_audits.items():
        if audit.get("complete") is not True or audit.get("run_id") != RUN_ID or audit.get("backbone") != backbone:
            raise ValueError(f"Experiment II-A grid audit is incomplete or stale for {backbone}")
        if int(audit.get("n_runs", -1)) != 50 or int(audit.get("expected_runs", -1)) != 50:
            raise ValueError(f"Experiment II-A grid audit for {backbone} must verify exactly 50 runs")
        if audit.get("contexts") != [0, 1, 2, 3, 4] or audit.get("folds") != [0, 1, 2, 3, 4] or audit.get("seeds") != [42, 123]:
            raise ValueError(f"Experiment II-A grid audit for {backbone} does not match the protocol grid")
        if not _is_sha256_hex(audit.get("grid_fingerprint")):
            raise ValueError(f"Experiment II-A grid audit for {backbone} is missing a valid grid_fingerprint")
    if post_ids is None:
        raise ValueError("Stage 11 requires the canonical ordered post_ids to verify OOF alignment")
    canonical_ids = [str(x) for x in post_ids]
    if len(set(canonical_ids)) != len(canonical_ids) or any(not x or x.lower() == "nan" for x in canonical_ids):
        raise ValueError("Stage 11 post_ids must be non-null and unique")
    if expected_n is not None and len(canonical_ids) != expected_n:
        raise ValueError(f"Stage 11 expected {expected_n} post IDs, received {len(canonical_ids)}")
    expected_id_hash = sha256_json(canonical_ids)
    if post_ids_by_prediction is None or set(post_ids_by_prediction) != {"C_best", "C0"}:
        raise ValueError("Stage 11 requires ordered post-ID lists for both C_best and C0 prediction sets")
    for condition in ("C_best", "C0"):
        id_map = post_ids_by_prediction[condition]
        if not isinstance(id_map, Mapping) or set(id_map) != set(BACKBONES):
            raise ValueError(f"post_ids_by_prediction[{condition!r}] must contain both backbones")
        for backbone, raw_ids in id_map.items():
            ids = [str(x) for x in raw_ids]
            if ids != canonical_ids:
                raise ValueError(f"{condition}/{backbone} prediction post-ID order differs from the canonical matched-subset order")
    for backbone, ctx_decision in context_decisions.items():
        if ctx_decision.get("post_id_sha256") != expected_id_hash:
            raise ValueError(f"Context OOF decision for {backbone} is not aligned to the canonical post IDs")
    y = np.asarray(y_true, dtype=int).reshape(-1)
    if len(y) != len(canonical_ids):
        raise ValueError("y_true length differs from canonical post_ids")
    if (y < 0).any() or (y >= len(LABEL2ID)).any():
        raise ValueError("y_true contains invalid class IDs")
    for name, pred_map in (("C_best", preds_c_best), ("C0", preds_c0)):
        for backbone, pred in pred_map.items():
            arr = np.asarray(pred, dtype=int).reshape(-1)
            if len(arr) != len(y):
                raise ValueError(f"{name}/{backbone} predictions do not match the OOF label length")
            if (arr < 0).any() or (arr >= len(LABEL2ID)).any():
                raise ValueError(f"{name}/{backbone} predictions contain invalid class IDs")

    a, b = "indobertweet", "xlmr"  # difference is always b - a
    kw = dict(n_bootstrap=n_bootstrap, ci=ci, random_state=random_state)
    primary = paired_bootstrap_f1_ci(y_true, preds_c_best[a], preds_c_best[b], **kw)["macro_f1"]
    guard = paired_bootstrap_f1_ci(y_true, preds_c0[a], preds_c0[b], **kw)["macro_f1"]

    leader = None
    if primary["significant"]:
        leader = b if primary["observed_diff"] > 0 else a

    guard_ok = None
    if leader == b:
        guard_ok = not (guard["ci_upper"] < 0)  # xlmr credibly worse at C0?
    elif leader == a:
        guard_ok = not (guard["ci_lower"] > 0)  # indobertweet credibly worse at C0?

    latency_rel = (latency_ms[b] - latency_ms[a]) / latency_ms[a]

    if leader is not None and guard_ok:
        decision, reason = leader, "primary_significant"
    elif latency_rel is not None and abs(latency_rel) > LATENCY_TOLERANCE:
        decision, reason = (b if latency_rel < 0 else a), "tiebreak_latency"
    else:
        decision, reason = a, "default_indobertweet"

    if leader is None:
        primary_outcome = "not_significant"
    else:
        primary_outcome = "significant" if guard_ok else "significant_but_guard_failed"

    return {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "rule_version": BACKBONE_RULE_VERSION,
        "difference": f"{b} minus {a}",
        "decision": decision,
        "reason": reason,
        "primary_outcome": primary_outcome,
        "primary_c_best": primary,
        "guard_c0": guard,
        "leader_before_guard": leader,
        "guard_ok": guard_ok,
        "run_id": RUN_ID,
        "context_rule_version": CONTEXT_RULE_VERSION,
        "context_decisions": {k: _jsonable(v) for k, v in context_decisions.items()},
        "source_grid_fingerprints": {k: grid_audits[k]["grid_fingerprint"] for k in BACKBONES},
        "source_grid_run_counts": {k: int(grid_audits[k]["n_runs"]) for k in BACKBONES},
        "c_best_by_backbone": {
            k: int(context_decisions[k]["selected_context"]) for k in BACKBONES
        },
        "c_best_predictions_verified": {
            k: int(context_sizes_by_backbone[k]) for k in BACKBONES
        },
        "latency_ms": dict(latency_ms),
        "latency_machine": latency_machine.strip(),
        "post_id_sha256": expected_id_hash,
        "n_oof_examples": len(y),
        "latency_relative_difference": float(latency_rel),
        "latency_tolerance": LATENCY_TOLERANCE,
        "exp1_leader": exp1_leader,
        "concordant_with_experiment1": (
            exp1_leader == decision if exp1_leader in BACKBONES else None
        ),
    }


def save_backbone_decision(decision: dict, path=BACKBONE_DECISION_PATH) -> Path:
    """Validate and persist a complete Stage 11 decision document.

    A decision without both context selections, a run ID, paired-bootstrap
    intervals, and positive latency measurements is rejected.
    """
    path = Path(path)
    validate_backbone_decision_document(decision)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(decision), indent=2, sort_keys=True), encoding="utf-8")
    return path


def validate_context_selection_document(
    context_doc: Mapping[str, Any],
    *,
    expected_backbone: str | None = None,
) -> None:
    """Validate a saved C0-C4 selection against the declared paired-CI rule."""
    required = {
        "schema_version", "run_id", "context_rule_version", "backbone", "selected_context",
        "provisional_top_context", "pooled_oof_macro_f1", "seed_count_by_context",
        "paired_comparisons_top_minus_shallower", "selection_reason",
        "n_oof_examples", "n_bootstrap", "ci_level", "post_id_sha256",
        "selected_experiment2_id", "selected_experiment3_id",
    }
    missing = sorted(required - set(context_doc))
    if missing:
        raise ValueError(f"Context decision is incomplete; missing {missing}")
    if context_doc["schema_version"] != CONFIG_SCHEMA_VERSION:
        raise ValueError("Context decision uses a stale/unsupported schema version")
    if context_doc["run_id"] != RUN_ID or context_doc["context_rule_version"] != CONTEXT_RULE_VERSION:
        raise ValueError("Context decision belongs to a stale run or unsupported context rule")
    backbone = str(context_doc["backbone"])
    get_backbone(backbone)
    if expected_backbone is not None and backbone != expected_backbone:
        raise ValueError(f"Context decision is for {backbone!r}, expected {expected_backbone!r}")
    if int(context_doc["n_oof_examples"]) != 900:
        raise ValueError("Context selection must use exactly 900 matched OOF examples")
    if int(context_doc["n_bootstrap"]) != 10_000 or float(context_doc["ci_level"]) != 95.0:
        raise ValueError("Context selection must use 10,000 paired-bootstrap resamples and a 95% CI")
    if not context_doc["post_id_sha256"]:
        raise ValueError("Context selection is missing the ordered post-ID hash")

    expected_keys = {f"C{i}" for i in range(5)}
    scores = context_doc["pooled_oof_macro_f1"]
    seeds = context_doc["seed_count_by_context"]
    if not isinstance(scores, Mapping) or set(scores) != expected_keys:
        raise ValueError("Context selection must report pooled OOF macro-F1 for every C0-C4")
    if not isinstance(seeds, Mapping) or set(seeds) != expected_keys:
        raise ValueError("Context selection must report seed counts for every C0-C4")
    for key, score in scores.items():
        try:
            numeric = float(score)
        except (TypeError, ValueError):
            raise ValueError(f"Invalid pooled OOF macro-F1 for {key}: {score!r}")
        if not math.isfinite(numeric) or not 0 <= numeric <= 1:
            raise ValueError(f"Invalid pooled OOF macro-F1 for {key}: {score!r}")
        if int(seeds[key]) != 2:
            raise ValueError(f"{key} must be ensembled from exactly two CV seeds")

    # Exact-score ties prefer the shallowest context, matching the selector.
    top = max(range(5), key=lambda c: float(scores[f"C{c}"]))
    if int(context_doc["provisional_top_context"]) != top:
        raise ValueError("Provisional top context does not match the pooled OOF macro-F1 values")
    comparisons = context_doc["paired_comparisons_top_minus_shallower"]
    expected_comparison_keys = {f"C{top}_minus_C{c}" for c in range(top)}
    if not isinstance(comparisons, Mapping) or set(comparisons) != expected_comparison_keys:
        raise ValueError("Paired-bootstrap CIs are missing for one or more shallower contexts")
    for c in range(top):
        key = f"C{top}_minus_C{c}"
        interval = comparisons[key]
        try:
            lower = float(interval["ci_lower"])
            upper = float(interval["ci_upper"])
            observed = float(interval["observed_diff"])
            reps = int(interval["n_bootstrap"])
            level = float(interval["ci_level"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid paired-bootstrap interval for {key}") from exc
        if not (math.isfinite(lower) and math.isfinite(upper) and math.isfinite(observed)) or lower > upper:
            raise ValueError(f"Invalid confidence interval for {key}")
        expected_observed = float(scores[f"C{top}"]) - float(scores[f"C{c}"])
        if not math.isclose(observed, expected_observed, rel_tol=1e-8, abs_tol=1e-8):
            raise ValueError(f"Observed paired F1 difference for {key} does not match the saved scores")
        if reps != 10_000 or level != 95.0:
            raise ValueError(f"{key} must use 10,000 paired-bootstrap resamples and a 95% CI")
        includes_zero = lower <= 0.0 <= upper
        if bool(interval.get("includes_zero")) != includes_zero:
            raise ValueError(f"Stored includes_zero flag disagrees with the CI for {key}")
        if bool(interval.get("significant")) == includes_zero:
            raise ValueError(f"Stored significance flag disagrees with the CI for {key}")
    candidate_order = ([top - 2, top - 1] if top >= 2 else ([top - 1] if top >= 1 else []))
    expected_selected = top
    expected_reason = "top_scoring_context_retained"
    for c in candidate_order:
        if comparisons[f"C{top}_minus_C{c}"]["includes_zero"]:
            expected_selected = c
            expected_reason = f"tradeoff_C{c}_CI_includes_zero_vs_C{top}"
            break
    if int(context_doc["selected_context"]) != expected_selected:
        raise ValueError(
            f"Selected C{context_doc['selected_context']} violates the paired-CI trade-off rule; "
            f"expected C{expected_selected}"
        )
    if context_doc["selection_reason"] != expected_reason:
        raise ValueError("Context selection_reason does not match the paired-CI trade-off rule")
    if context_doc["selected_experiment2_id"] != experiment2_id(expected_selected, backbone):
        raise ValueError("Context decision has an inconsistent Experiment II identifier")
    if context_doc["selected_experiment3_id"] != experiment3_id(expected_selected, backbone):
        raise ValueError("Context decision has an inconsistent Experiment III identifier")


def validate_backbone_decision_document(document: Mapping[str, Any]) -> None:
    required = {
        "schema_version", "run_id", "rule_version", "context_rule_version", "decision", "reason",
        "primary_c_best", "guard_c0", "latency_ms", "latency_machine", "context_decisions",
        "c_best_by_backbone", "c_best_predictions_verified", "latency_relative_difference", "latency_tolerance",
        "post_id_sha256", "n_oof_examples", "source_grid_fingerprints", "source_grid_run_counts",
    }
    missing = sorted(required - set(document))
    if missing:
        raise ValueError(f"Invalid Stage 11 decision document; missing {missing}")
    if document["schema_version"] != CONFIG_SCHEMA_VERSION:
        raise ValueError(
            f"Decision schema {document['schema_version']!r} is unsupported; "
            f"expected {CONFIG_SCHEMA_VERSION!r}."
        )
    if document["rule_version"] != BACKBONE_RULE_VERSION:
        raise ValueError(
            f"Decision rule {document['rule_version']!r} is stale/unsupported; "
            f"expected {BACKBONE_RULE_VERSION!r}."
        )
    if document["context_rule_version"] != CONTEXT_RULE_VERSION:
        raise ValueError("Decision uses an unsupported context-selection rule version")
    if document["run_id"] != RUN_ID:
        raise ValueError(
            f"Decision belongs to run {document['run_id']!r}, not the required {RUN_ID!r}."
        )
    get_backbone(document["decision"])
    if set(document["context_decisions"]) != set(BACKBONES):
        raise ValueError("Decision must contain context decisions for both backbones")
    if set(document["c_best_by_backbone"]) != set(BACKBONES):
        raise ValueError("c_best_by_backbone must contain both backbones")
    if set(document["c_best_predictions_verified"]) != set(BACKBONES):
        raise ValueError("c_best_predictions_verified must verify the C_best prediction inputs for both backbones")
    if document["c_best_predictions_verified"] != document["c_best_by_backbone"]:
        raise ValueError("C_best predictions were not generated at each backbone's selected context")
    grid_fingerprints = document["source_grid_fingerprints"]
    grid_counts = document["source_grid_run_counts"]
    if not isinstance(grid_fingerprints, Mapping) or set(grid_fingerprints) != set(BACKBONES):
        raise ValueError("Decision must record a source grid fingerprint for both backbones")
    if not isinstance(grid_counts, Mapping) or set(grid_counts) != set(BACKBONES):
        raise ValueError("Decision must record the source CV grid run count for both backbones")
    for b in BACKBONES:
        if not _is_sha256_hex(grid_fingerprints[b]):
            raise ValueError(f"Invalid source grid fingerprint for {b}")
        if int(grid_counts[b]) != 50:
            raise ValueError(f"Source grid for {b} must represent exactly 50 runs")
    context_hashes = {document["context_decisions"][b].get("post_id_sha256") for b in BACKBONES}
    if context_hashes != {document["post_id_sha256"]}:
        raise ValueError("Backbone context decisions are not aligned to the same ordered post IDs")
    for b in BACKBONES:
        c = document["c_best_by_backbone"][b]
        ctx_doc = document["context_decisions"][b]
        validate_context_selection_document(ctx_doc, expected_backbone=b)
        if c not in range(5) or int(ctx_doc["selected_context"]) != c:
            raise ValueError(f"Inconsistent/invalid C_best for {b}: {c!r}")
    if not isinstance(document["latency_machine"], str) or not document["latency_machine"].strip():
        raise ValueError("Decision must identify the latency measurement machine")
    if not document["post_id_sha256"] or int(document["n_oof_examples"]) != 900:
        raise ValueError("Decision must record aligned post IDs for the 900-example OOF set")
    latency = document["latency_ms"]
    if not isinstance(latency, Mapping) or set(latency) != set(BACKBONES):
        raise ValueError("Decision must include measured median latency for both backbones")
    bad = {k: latency[k] for k in BACKBONES if not _is_positive_finite(latency[k])}
    if bad:
        raise ValueError(f"Decision has invalid latency values: {bad}")
    if not isinstance(document["primary_c_best"], Mapping) or not isinstance(document["guard_c0"], Mapping):
        raise ValueError("Decision must include C_best and C0 paired-bootstrap intervals")
    for name in ("primary_c_best", "guard_c0"):
        interval = document[name]
        if int(interval.get("n_bootstrap", -1)) != 10_000 or float(interval.get("ci_level", -1)) != 95.0:
            raise ValueError(f"{name} must use the protocol's 10,000 paired-bootstrap resamples and 95% CI")
        lower, upper = interval.get("ci_lower"), interval.get("ci_upper")
        try:
            lower, upper = float(lower), float(upper)
        except (TypeError, ValueError):
            raise ValueError(f"{name} contains an invalid confidence interval")
        if not (math.isfinite(lower) and math.isfinite(upper)) or lower > upper:
            raise ValueError(f"{name} contains an invalid confidence interval")
        expected_significant = not (lower <= 0.0 <= upper)
        if bool(interval.get("significant")) != expected_significant:
            raise ValueError(f"{name} significance flag disagrees with its confidence interval")
    primary = document["primary_c_best"]
    guard = document["guard_c0"]
    leader = None
    if bool(primary.get("significant")):
        try:
            observed_primary = float(primary["observed_diff"])
        except (TypeError, ValueError, KeyError) as exc:
            raise ValueError("primary_c_best has an invalid observed difference") from exc
        if not math.isfinite(observed_primary) or observed_primary == 0:
            raise ValueError("A significant primary interval must have a non-zero finite observed difference")
        leader = "xlmr" if observed_primary > 0 else "indobertweet"
    if leader == "xlmr":
        guard_ok = not (float(guard["ci_upper"]) < 0)
    elif leader == "indobertweet":
        guard_ok = not (float(guard["ci_lower"]) > 0)
    else:
        guard_ok = None
    lat = document["latency_ms"]
    expected_latency_rel = (float(lat["xlmr"]) - float(lat["indobertweet"])) / float(lat["indobertweet"])
    try:
        saved_latency_rel = float(document["latency_relative_difference"])
        saved_tolerance = float(document["latency_tolerance"])
    except (TypeError, ValueError) as exc:
        raise ValueError("Decision has invalid latency tie-break metadata") from exc
    if not math.isclose(saved_latency_rel, expected_latency_rel, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("Saved latency_relative_difference disagrees with the measured medians")
    if not math.isclose(saved_tolerance, LATENCY_TOLERANCE, rel_tol=0, abs_tol=1e-12):
        raise ValueError("Decision uses a stale latency tolerance")
    if leader is not None and guard_ok:
        expected_decision, expected_reason = leader, "primary_significant"
    elif abs(expected_latency_rel) > LATENCY_TOLERANCE:
        expected_decision = "xlmr" if expected_latency_rel < 0 else "indobertweet"
        expected_reason = "tiebreak_latency"
    else:
        expected_decision, expected_reason = "indobertweet", "default_indobertweet"
    if document["decision"] != expected_decision or document["reason"] != expected_reason:
        raise ValueError(
            "Saved backbone choice/reason does not follow the current paired-CI, C0 guard, and latency rule"
        )
    if document.get("leader_before_guard") != leader or document.get("guard_ok") != guard_ok:
        raise ValueError("Saved Stage 11 leader/C0-guard metadata is inconsistent with the paired intervals")


def load_backbone_decision_document(path=BACKBONE_DECISION_PATH) -> dict:
    """Read/validate the complete Stage 11 JSON, rejecting historical files."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run Stage 11 ({PIPELINE_SCRIPTS[11]}) first."
        )
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"Cannot read valid Stage 11 decision at {path}: {exc}") from exc
    validate_backbone_decision_document(doc)
    return doc


def load_backbone_decision(path=BACKBONE_DECISION_PATH) -> str:
    """Return the selected backbone, validating the complete current-run gate."""
    return load_backbone_decision_document(path)["decision"]


def selected_context_for_backbone(backbone: str, path=BACKBONE_DECISION_PATH) -> int:
    """Return the Stage 11 C_best for a backbone; historical/invalid gates fail."""
    get_backbone(backbone)
    return int(load_backbone_decision_document(path)["c_best_by_backbone"][backbone])


def select_training_strategy(
    y_true,
    preds_itft,
    preds_direct,
    n_bootstrap: int = 10_000,
    ci: float = 95.0,
    random_state: int = 42,
    *,
    post_ids: Sequence[Any] | None = None,
    post_ids_itft: Sequence[Any] | None = None,
    post_ids_direct: Sequence[Any] | None = None,
    expected_n: int | None = 900,
) -> dict:
    """
    Choose the final training strategy for ONE deployment mode (C0 or
    C_best) of the selected backbone: ITFT (Experiment II) or direct
    fine-tuning (Experiment III).

    Inputs are pooled out-of-fold, seed-ensembled predictions on the same
    900 examples and the same folds. Rule (v1): ITFT is chosen only if its
    macro-F1 is higher AND the paired-bootstrap CI (ITFT - direct) excludes
    0; direct fine-tuning is chosen if it is credibly higher; with no
    credible difference the parsimonious default (DEFAULT_STRATEGY_ON_TIE)
    is used. Call once per mode -- the two modes may select differently.
    """
    if post_ids is None or post_ids_itft is None or post_ids_direct is None:
        raise ValueError("Training-strategy selection requires canonical and per-strategy post ID order")
    canonical = [str(x) for x in post_ids]
    if len(set(canonical)) != len(canonical):
        raise ValueError("Canonical post IDs must be unique")
    if post_ids_itft is None or [str(x) for x in post_ids_itft] != canonical:
        raise ValueError("ITFT OOF predictions are not aligned to canonical post IDs")
    if post_ids_direct is None or [str(x) for x in post_ids_direct] != canonical:
        raise ValueError("Direct OOF predictions are not aligned to canonical post IDs")
    if expected_n is not None and len(canonical) != expected_n:
        raise ValueError(f"Expected {expected_n} matched OOF IDs, got {len(canonical)}")
    if len(np.asarray(y_true).reshape(-1)) != len(canonical):
        raise ValueError("y_true length differs from canonical post IDs")
    result = paired_bootstrap_f1_ci(
        y_true, preds_direct, preds_itft,
        n_bootstrap=n_bootstrap, ci=ci, random_state=random_state,
    )["macro_f1"]
    if result["significant"] and result["observed_diff"] > 0:
        strategy, reason = "itft", "itft_credibly_better"
    elif result["significant"]:
        strategy, reason = "direct", "direct_credibly_better"
    else:
        strategy, reason = DEFAULT_STRATEGY_ON_TIE, "no_credible_difference_default"
    return {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "run_id": RUN_ID,
        "rule_version": BACKBONE_RULE_VERSION,
        "post_id_sha256": sha256_json(canonical),
        "n_oof_examples": len(canonical),
        "difference": "itft minus direct",
        "strategy": strategy,
        "reason": reason,
        "bootstrap": result,
    }



def save_training_strategy_decision(
    decision_by_mode: Mapping[str, Mapping[str, Any]],
    path=TRAINING_STRATEGY_DECISION_PATH,
) -> Path:
    """Persist the separately selected ITFT/direct strategy for C0 and C_best."""
    if set(decision_by_mode) != {"target_only", "context_aware"}:
        raise ValueError("Strategy decision must contain target_only and context_aware modes")
    for mode, doc in decision_by_mode.items():
        if doc.get("schema_version") != CONFIG_SCHEMA_VERSION:
            raise ValueError(f"Strategy result for {mode} uses a stale/unsupported schema")
        if doc.get("run_id") != RUN_ID:
            raise ValueError(f"Strategy result for {mode} is missing current run_id={RUN_ID!r}")
        if doc.get("strategy") not in {"itft", "direct"}:
            raise ValueError(f"Invalid strategy for {mode}: {doc.get('strategy')!r}")
    out = {"schema_version": CONFIG_SCHEMA_VERSION, "run_id": RUN_ID, "rule_version": BACKBONE_RULE_VERSION, "by_mode": _jsonable(decision_by_mode)}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    return path


def load_training_strategy_decision(path=TRAINING_STRATEGY_DECISION_PATH) -> dict:
    """Load only a valid current-run Stage 15 strategy decision."""
    doc = load_json_config(path, required_keys=("schema_version", "run_id", "rule_version", "by_mode"))
    if doc["schema_version"] != CONFIG_SCHEMA_VERSION or doc["run_id"] != RUN_ID or doc["rule_version"] != BACKBONE_RULE_VERSION:
        raise ValueError("Strategy decision is stale or uses a different decision-rule version")
    modes = doc["by_mode"]
    if set(modes) != {"target_only", "context_aware"}:
        raise ValueError("Strategy decision is missing target_only/context_aware")
    for mode, item in modes.items():
        if item.get("strategy") not in {"itft", "direct"}:
            raise ValueError(f"Invalid strategy stored for {mode}")
    return doc


def three_way_context_table(
    df,
    context_needed_col: str = "context_needed",
    label_col: str = "label",
    parent_count_col: str = "context_size_available",
):
    """
    Cross-tabulate context_size_available x (context_needed, label).

    Run this:
    - Periodically during annotation (e.g. every ~200 new posts), to catch
      under-filled combinations while there is still time to target
      collection toward them.
    - Once on the full dataset, to confirm the overall 600/600/600
      class-balanced label distribution and the planned context-needed
      breakdown.
    - Filtered to context_size_available == 4 only, right before Experiment
      II training starts -- this confirms the matched 4-parent subset used
      for the C0-C4 ablation is not skewed on context_needed x label,
      even if the full dataset looks fine.
    """
    return pd.crosstab(
        index=df[parent_count_col],
        columns=[df[context_needed_col], df[label_col]],
        margins=True,
        margins_name="Total",
    )


def flag_small_cells(df, group_cols: list[str], min_count: int = 20):
    """
    Flag context_needed x label (x parent_count) combinations with fewer
    than `min_count` examples. Default of 20 is sized for 5-fold CV: a cell
    of 20 leaves ~4 examples per fold's held-out slice, which is the rough
    floor for a per-cell F1 to mean anything at all. Prefer >=50 per cell
    in the matched subset if your annotation budget allows it.
    """
    counts = df.groupby(group_cols).size().reset_index(name="count")
    return counts[counts["count"] < min_count]


def validate_label_consistency(
    df: pd.DataFrame,
    hs_col: str = "HS",
    al_col: str = "AL",
    label_col: str = "label",
) -> None:
    """
    Verify that `label` is consistent with the HS-priority mapping rule
    (HS=1 -> HS; else AL=1 -> AL; else N) applied to the HS/AL columns.
    Raises with the offending post_id(s) if any row's stored label
    disagrees with what the mapping rule would produce -- catches manual
    annotation-sheet edits or copy/paste errors before they reach training.
    """
    expected = df.apply(
        lambda r: "HS" if r[hs_col] == 1 else ("AL" if r[al_col] == 1 else "N"),
        axis=1,
    )
    mismatches = df[expected != df[label_col]]
    if len(mismatches):
        ids = mismatches[CTX_ID_COLUMN].tolist() if CTX_ID_COLUMN in df.columns else mismatches.index.tolist()
        raise ValueError(
            f"{len(mismatches)} row(s) where `{label_col}` does not match "
            f"the HS/AL priority mapping. Affected post_id(s): {ids}"
        )


def pool_out_of_fold_predictions(fold_indices: list[np.ndarray], fold_preds: list[np.ndarray]):
    """
    Combine per-fold held-out predictions (from k-fold CV) into a single
    pooled prediction array covering the full dataset, in original example
    order. Use the result as `preds_a` / `preds_b` in paired_bootstrap_ci
    for the Experiment III (C0 vs. C_best) comparison, so the bootstrap
    draws from the full matched subset rather than only 5 fold-level F1 values.

    Args:
        fold_indices: list of arrays of original example indices held out in each fold.
        fold_preds: list of arrays of predictions for those held-out examples,
            same order/length as the corresponding fold_indices entry.
    """
    all_indices = np.concatenate(fold_indices)
    all_preds = np.concatenate(fold_preds)
    order = np.argsort(all_indices)
    return all_preds[order]


def pool_out_of_fold_probabilities(
    fold_indices: list[np.ndarray],
    fold_probs: list[np.ndarray],
) -> np.ndarray:
    """
    Probability counterpart of pool_out_of_fold_predictions(): combine each
    fold's held-out (n_fold, n_classes) softmax arrays into one
    (n_examples, n_classes) array in original example order. Call it ONCE
    PER SEED, then pass the per-seed results to ensemble_seed_probabilities()
    to obtain the seed-ensembled pooled predictions used by select_backbone()
    and select_training_strategy().

    Raises if any example appears in more than one fold's held-out set.
    """
    all_indices = np.concatenate(fold_indices)
    if len(np.unique(all_indices)) != len(all_indices):
        raise ValueError("An example appears in more than one held-out fold")
    all_probs = np.concatenate(fold_probs, axis=0)
    return all_probs[np.argsort(all_indices)]

# ---------------------------------------------------------------------------
# 3. Paired-bootstrap context-size selection (Stage 11)
# ---------------------------------------------------------------------------

def select_context_size_from_oof(
    y_true,
    probabilities_by_context: Mapping[int, Sequence[np.ndarray]],
    post_ids: Sequence[Any],
    *,
    post_ids_by_context: Mapping[int, Sequence[Any]] | None = None,
    backbone: str = DEFAULT_BACKBONE,
    n_bootstrap: int = 10_000,
    ci: float = 95.0,
    random_state: int = 42,
    expected_n: int | None = 900,
    expected_seed_count: int = 2,
) -> dict:
    """Choose C0-C4 from paired, seed-ensembled pooled OOF probabilities.

    `probabilities_by_context[c]` must contain one pooled (n_examples, 3)
    probability matrix per CV seed. Each seed matrix is already in the fixed
    `post_ids` order; all 5 folds are pooled before seed ensembling. If provided,
    `post_ids_by_context` is checked condition-by-condition to catch misaligned
    output files before paired resampling.

    Rule (experiment_protocol.md Section 5.1): first identify the context with
    highest pooled OOF macro-F1 and compare it with EVERY shallower condition
    using paired bootstrap intervals. For a provisional top C>=2, the pre-
    declared deployment trade-off checks C(top-2) first, then C(top-1); choose
    the first of those two candidates whose F1(top)-F1(candidate) 95% CI
    includes zero. For C4 this is explicitly C2, then C3. If neither qualifies,
    keep the top context. C0/C1 comparisons remain in the report but do not
    override the specific C4-vs-C2/C3 decision rule. This is not equivalence.
    """
    get_backbone(backbone)
    y = np.asarray(y_true, dtype=int).reshape(-1)
    ids = np.asarray([str(x) for x in post_ids], dtype=object)
    n = len(y)
    if n == 0 or len(ids) != n:
        raise ValueError("y_true and post_ids must have the same non-zero length")
    if (y < 0).any() or (y >= len(LABEL2ID)).any():
        raise ValueError(f"y_true must contain class IDs 0..{len(LABEL2ID) - 1}")
    if n_bootstrap < 1 or not (0 < ci < 100):
        raise ValueError("n_bootstrap must be >=1 and ci must be between 0 and 100")
    if expected_seed_count < 1:
        raise ValueError("expected_seed_count must be at least 1")
    if pd.isna(ids).any() or len(pd.unique(ids)) != len(ids):
        raise ValueError("post_ids must be non-null and unique")
    if expected_n is not None and n != expected_n:
        raise ValueError(f"Expected {expected_n} matched OOF examples, received {n}")
    if set(probabilities_by_context) != set(range(5)):
        raise ValueError("probabilities_by_context must contain exactly C0, C1, C2, C3, C4")
    if post_ids_by_context is None:
        raise ValueError(
            "post_ids_by_context is required for Stage 11 so each condition's "
            "OOF probabilities can be verified against the same ordered 900 posts"
        )
    if set(post_ids_by_context) != set(range(5)):
        raise ValueError("post_ids_by_context must contain post IDs for every C0-C4 condition")

    ensemble_probs: dict[int, np.ndarray] = {}
    predictions: dict[int, np.ndarray] = {}
    n_seed_by_context: dict[str, int] = {}
    for c in range(5):
        seed_arrays = [np.asarray(a, dtype=float) for a in probabilities_by_context[c]]
        if len(seed_arrays) != expected_seed_count:
            raise ValueError(
                f"C{c} must contain exactly {expected_seed_count} seed-ensembled pooled OOF arrays; "
                f"got {len(seed_arrays)}"
            )
        for seed_i, probs in enumerate(seed_arrays):
            if probs.shape != (n, len(LABEL2ID)):
                raise ValueError(
                    f"C{c} seed {seed_i}: probability shape {probs.shape}; expected {(n, len(LABEL2ID))}"
                )
            if not np.isfinite(probs).all() or (probs < 0).any():
                raise ValueError(f"C{c} seed {seed_i}: probabilities must be finite and non-negative")
            sums = probs.sum(axis=1)
            if not np.allclose(sums, 1.0, atol=1e-3, rtol=1e-3):
                raise ValueError(f"C{c} seed {seed_i}: each probability row must sum to 1")
        run_ids = np.asarray([str(x) for x in post_ids_by_context[c]], dtype=object)
        if len(run_ids) != n or not np.array_equal(run_ids, ids):
            raise ValueError(
                f"C{c} post-ID order differs from the canonical matched-subset order"
            )
        ensemble_probs[c] = np.mean(np.stack(seed_arrays, axis=0), axis=0)
        predictions[c] = ensemble_probs[c].argmax(axis=1)
        n_seed_by_context[str(c)] = len(seed_arrays)

    scores = {
        c: float(f1_score(y, predictions[c], labels=[0, 1, 2], average="macro", zero_division=0))
        for c in range(5)
    }
    # max() iterates in ascending C order; exact ties therefore prefer the
    # shallowest window as specified by the protocol.
    top_c = max(range(5), key=lambda c: scores[c])
    comparisons: dict[str, dict] = {}
    for c in range(top_c):
        # args are (candidate, top), so observed difference = F1(top)-F1(candidate).
        # The same random_state and same y/order produce the same resampling
        # indices for every condition comparison (paired bootstrap).
        comparison = paired_bootstrap_f1_ci(
            y, predictions[c], predictions[top_c],
            n_bootstrap=n_bootstrap, ci=ci, random_state=random_state,
        )["macro_f1"]
        comparison["includes_zero"] = bool(
            comparison["ci_lower"] <= 0.0 <= comparison["ci_upper"]
        )
        comparisons[f"C{top_c}_minus_C{c}"] = comparison

    selected_c = top_c
    selected_reason = "top_scoring_context_retained"
    # The protocol pre-declares a practical trade-off between the two-level-
    # shallower condition and the immediately shallower condition. In the
    # common C4-top case, C4-vs-C2 is checked first; only if it excludes zero
    # do we use C4-vs-C3 to consider C3. C0/C1 intervals are still reported.
    candidate_order = ([top_c - 2, top_c - 1] if top_c >= 2 else ([top_c - 1] if top_c >= 1 else []))
    for c in candidate_order:
        if comparisons[f"C{top_c}_minus_C{c}"]["includes_zero"]:
            selected_c = c
            selected_reason = f"tradeoff_C{c}_CI_includes_zero_vs_C{top_c}"
            break

    return {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "run_id": RUN_ID,
        "context_rule_version": CONTEXT_RULE_VERSION,
        "backbone": backbone,
        "selected_context": int(selected_c),
        "selected_experiment2_id": experiment2_id(selected_c, backbone),
        "selected_experiment3_id": experiment3_id(selected_c, backbone),
        "provisional_top_context": int(top_c),
        "pooled_oof_macro_f1": {f"C{c}": scores[c] for c in range(5)},
        "seed_count_by_context": {f"C{c}": n_seed_by_context[str(c)] for c in range(5)},
        "paired_comparisons_top_minus_shallower": comparisons,
        "selection_reason": selected_reason,
        "n_oof_examples": n,
        "n_bootstrap": n_bootstrap,
        "ci_level": ci,
        "post_id_sha256": sha256_json(ids.tolist()),
        "interpretation_note": (
            "A CI containing zero means the data did not resolve a gain at the "
            "chosen confidence level; it does not establish equivalence."
        ),
    }


# ---------------------------------------------------------------------------
# 4. Explicit tokenizer policy and shared-config audits
# ---------------------------------------------------------------------------

def configure_tokenizer_for_input(tokenizer, input_mode: str):
    """Set and verify truncation side rather than relying on tokenizer defaults.

    Context-first inputs (parents followed by target) require left truncation.
    Target-only benchmark inputs use right truncation explicitly. Returns the
    tokenizer for convenient use during model setup.
    """
    expected = {"contextual": "left", "target_only": "right"}
    if input_mode not in expected:
        raise ValueError(f"input_mode must be one of {sorted(expected)}, got {input_mode!r}")
    tokenizer.truncation_side = expected[input_mode]
    if tokenizer.truncation_side != expected[input_mode]:
        raise RuntimeError(
            f"Tokenizer truncation_side must be {expected[input_mode]!r} for {input_mode} inputs"
        )
    return tokenizer


def validate_truncation_policy(config: Mapping[str, Any], *, input_mode: str) -> None:
    """Fail before training when the configured truncation side is unsafe."""
    expected = {"contextual": "left", "target_only": "right"}
    if input_mode not in expected:
        raise ValueError(f"Unknown input_mode {input_mode!r}")
    got = config.get("truncation_side")
    if got != expected[input_mode]:
        raise ValueError(
            f"Unsafe truncation configuration: {input_mode} requires "
            f"truncation_side={expected[input_mode]!r}, got {got!r}"
        )


def load_json_config(path: str | Path, *, required_keys: Sequence[str] = ()) -> dict:
    """Load a shared JSON config and fail on missing/invalid fields."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Required shared configuration does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"Invalid JSON configuration {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Configuration at {path} must be a JSON object")
    missing = sorted(set(required_keys) - set(value))
    if missing:
        raise ValueError(f"Configuration {path} is missing keys: {missing}")
    return value


def load_shared_run_config(
    sequence_length_path: str | Path = SEQUENCE_LENGTH_CONFIG,
    learning_rate_path: str | Path = LEARNING_RATE_CONFIG,
    *,
    require_context: bool = False,
    require_context_lr: bool = False,
) -> dict:
    """Load the shared configuration fields available at the current stage.

    Stages 3-7 run before the contextual length/LR pilot, so they require only
    the benchmark sequence length and benchmark learning rates. Stages 9+
    require both length values and both tuned rates.
    """
    length_keys = [
        "schema_version", "run_id", "max_length_benchmark", "truncation_side",
        "target_only_truncation_side", "benchmark_length_source_partition",
        "benchmark_length_percentile", "benchmark_length_percentile_tokens",
        "benchmark_truncation_rate_at_max_length",
    ]
    if require_context:
        length_keys.extend([
            "max_length_context", "context_length_source_partition", "context_length_percentile",
            "context_length_percentile_tokens", "context_truncation_rate_at_max_length",
        ])
    rate_keys = [
        "schema_version", "run_id", "lr_search_grid", "learning_rate_benchmark",
        "benchmark_trial_macro_f1", "lr_tuning_seed",
    ]
    if require_context_lr:
        rate_keys.extend(["learning_rate_context", "context_trial_macro_f1"])
    lengths = load_json_config(sequence_length_path, required_keys=length_keys)
    rates = load_json_config(learning_rate_path, required_keys=rate_keys)
    for name, doc in (("sequence_length", lengths), ("learning_rates", rates)):
        if doc["schema_version"] != CONFIG_SCHEMA_VERSION:
            raise ValueError(
                f"{name} config schema is {doc['schema_version']!r}; expected {CONFIG_SCHEMA_VERSION!r}; regenerate it"
            )
        if doc["run_id"] != RUN_ID:
            raise ValueError(
                f"{name} config belongs to {doc['run_id']!r}, expected {RUN_ID!r}; regenerate it for the corrected sweep"
            )
    if lengths["benchmark_length_source_partition"] != "benchmark_train":
        raise ValueError("max_length_benchmark must declare benchmark_length_source_partition='benchmark_train'")
    if require_context and lengths["context_length_source_partition"] != "contextual_c4":
        raise ValueError("max_length_context must declare context_length_source_partition='contextual_c4'")
    length_groups = [("benchmark", lengths["max_length_benchmark"], lengths["benchmark_length_percentile_tokens"], lengths["benchmark_truncation_rate_at_max_length"], lengths["benchmark_length_percentile"])]
    if require_context:
        length_groups.append(("context", lengths["max_length_context"], lengths["context_length_percentile_tokens"], lengths["context_truncation_rate_at_max_length"], lengths["context_length_percentile"]))
    for name, value, pct_doc, trunc_doc, percentile in length_groups:
        if isinstance(value, bool) or not isinstance(value, int) or not (8 <= value <= MAX_SEQUENCE_LENGTH):
            raise ValueError(f"max_length_{name if name == 'context' else 'benchmark'} must be an integer in 8..{MAX_SEQUENCE_LENGTH}, got {value!r}")
        if value % LENGTH_ROUNDING_MULTIPLE:
            raise ValueError(f"max_length for {name} must be rounded to a multiple of {LENGTH_ROUNDING_MULTIPLE}")
        if float(percentile) != LENGTH_PERCENTILE:
            raise ValueError(f"{name} length selection must use p{LENGTH_PERCENTILE:g}")
        if not isinstance(pct_doc, Mapping) or set(pct_doc) != set(BACKBONES):
            raise ValueError(f"{name} p95 token stats must contain both backbones")
        if not isinstance(trunc_doc, Mapping) or set(trunc_doc) != set(BACKBONES):
            raise ValueError(f"{name} truncation stats must contain both backbones")
        try:
            pct_values = {b: float(pct_doc[b]) for b in BACKBONES}
            trunc_values = {b: float(trunc_doc[b]) for b in BACKBONES}
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid saved {name} length/truncation statistics") from exc
        if any(not math.isfinite(v) or v <= 0 for v in pct_values.values()):
            raise ValueError(f"Invalid saved {name} p95 token lengths")
        if any(not math.isfinite(v) or not 0 <= v <= 1 for v in trunc_values.values()):
            raise ValueError(f"Invalid saved {name} truncation rates")
        expected_length = min(int(math.ceil(max(pct_values.values()) / LENGTH_ROUNDING_MULTIPLE) * LENGTH_ROUNDING_MULTIPLE), MAX_SEQUENCE_LENGTH)
        if expected_length != value:
            raise ValueError(f"Saved max_length_{name} does not match its p95 stats; expected {expected_length}")
    if lengths["truncation_side"] != "left":
        raise ValueError("Shared pipeline config must explicitly declare contextual truncation_side='left'")
    if lengths["target_only_truncation_side"] != "right":
        raise ValueError(
            "Shared pipeline config must explicitly declare target_only_truncation_side='right'"
        )
    for key in ("max_length_benchmark", "max_length_context") if require_context else ("max_length_benchmark",):
        if key in lengths and lengths[key] % LENGTH_ROUNDING_MULTIPLE != 0:
            raise ValueError(f"{key} must be rounded to a multiple of {LENGTH_ROUNDING_MULTIPLE}")
    grid = tuple(float(x) for x in rates["lr_search_grid"])
    if tuple(sorted(grid)) != tuple(sorted(LR_SEARCH_GRID)):
        raise ValueError(f"LR grid differs from the frozen protocol grid: {grid} vs {LR_SEARCH_GRID}")
    tuning_seed = rates["lr_tuning_seed"]
    if isinstance(tuning_seed, bool) or not isinstance(tuning_seed, int):
        raise ValueError("lr_tuning_seed must be an integer, not a historical or implicit default")
    if tuning_seed in EVAL_SEEDS_EXP1 or tuning_seed in CV_SEEDS:
        raise ValueError(
            f"lr_tuning_seed={tuning_seed} overlaps an evaluation/CV seed; "
            "use a dedicated tuning seed not used in Experiment I or contextual CV"
        )
    for b in BACKBONES:
        for field, trials_field in ([
            ("learning_rate_benchmark", "benchmark_trial_macro_f1"),
        ] + ([ ("learning_rate_context", "context_trial_macro_f1") ] if require_context_lr else [])):
            value = rates[field].get(b) if isinstance(rates[field], Mapping) else None
            if value is None or not any(np.isclose(float(value), g) for g in grid):
                raise ValueError(f"{field}[{b!r}] is missing or outside the shared grid")
            trial_doc = rates[trials_field].get(b) if isinstance(rates[trials_field], Mapping) else None
            if not isinstance(trial_doc, Mapping):
                raise ValueError(f"{trials_field}[{b!r}] must store validation macro-F1 for every grid candidate")
            try:
                trial_scores = {float(k): float(v) for k, v in trial_doc.items()}
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid LR trial scores for {b} / {trials_field}") from exc
            if any(not math.isfinite(score) or not 0 <= score <= 1 for score in trial_scores.values()):
                raise ValueError(f"LR validation macro-F1 scores must be finite values in 0..1 for {b}")
            selected = select_learning_rate(trial_scores, grid)
            if not np.isclose(float(value), selected, rtol=0, atol=1e-12):
                raise ValueError(
                    f"Saved {field}[{b}]={value} is not the best rate supported by {trials_field}; "
                    f"expected {selected} under the frozen tie-break"
                )
    return {"run_id": RUN_ID, "sequence_length": lengths, "learning_rates": rates}


def validate_resolved_run_config(
    config: Mapping[str, Any],
    *,
    stage: int | str,
    backbone: str | None = None,
    input_mode: str | None = None,
    context_size: int | None = None,
    require_shared_config: bool = False,
) -> None:
    """Audit run metadata against pipeline/stage, naming, truncation and LR rules.

    Training/evaluation scripts should call this before touching any output.
    It intentionally rejects old runs/configs without the version tag or
    script identity rather than guessing that they are compatible.
    """
    if stage not in PIPELINE_SCRIPTS:
        raise ValueError(f"Stage {stage!r} is not in the authoritative pipeline registry")
    expected_script = PIPELINE_SCRIPTS[stage]
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError(
            f"Run config schema must be {CONFIG_SCHEMA_VERSION!r}; got {config.get('schema_version')!r}"
        )
    if config.get("run_id") != RUN_ID:
        raise ValueError(f"Run config must have run_id={RUN_ID!r}; got {config.get('run_id')!r}")
    if config.get("stage") != stage and str(config.get("stage")) != str(stage):
        raise ValueError(f"Run config stage {config.get('stage')!r} does not match {stage!r}")
    if Path(str(config.get("script_name", ""))).as_posix() != expected_script:
        raise ValueError(
            f"Stage {stage} must use script {expected_script!r}; "
            f"got {config.get('script_name')!r}"
        )
    if backbone is not None:
        spec = get_backbone(backbone)
        if config.get("backbone") != backbone:
            raise ValueError(f"Run config backbone does not match requested {backbone!r}")
        if config.get("hf_model_id") != spec.hf_model_id:
            raise ValueError(
                f"Run config hf_model_id must be {spec.hf_model_id!r} for {backbone}"
            )
    if context_size is not None and context_size not in range(5):
        raise ValueError(f"context_size must be 0-4, got {context_size}")
    if input_mode is not None:
        validate_truncation_policy(config, input_mode=input_mode)
    for key in ("dataset_sha256", "tokenizer_revision", "model_revision"):
        if not config.get(key):
            raise ValueError(f"Run config missing reproducibility audit field {key!r}")
    if input_mode == "contextual" and not config.get("fold_assignment_sha256"):
        raise ValueError("Contextual run config missing fold_assignment_sha256")
    if require_shared_config:
        context_mode = input_mode == "contextual"
        shared = load_shared_run_config(
            require_context=context_mode,
            require_context_lr=context_mode,
        )
        lengths = shared["sequence_length"]
        if config.get("max_length_benchmark") != lengths["max_length_benchmark"]:
            raise ValueError("Run max_length_benchmark differs from the shared frozen config")
        if context_mode and config.get("max_length_context") != lengths["max_length_context"]:
            raise ValueError("Run max_length_context differs from the shared frozen config")
        rates = shared["learning_rates"]
        if backbone and config.get("learning_rate_benchmark") != rates["learning_rate_benchmark"][backbone]:
            raise ValueError("Run benchmark LR differs from the shared tuned LR")
        if context_mode and backbone and config.get("learning_rate_context") != rates["learning_rate_context"][backbone]:
            raise ValueError("Run contextual LR differs from the shared tuned LR")
    if stage in DECISION_DEPENDENT_STAGES:
        decision = load_backbone_decision_document()
        if backbone is not None and backbone != decision["decision"]:
            raise ValueError(
                f"Stage {stage} must use the Stage 11 selected backbone {decision['decision']!r}, "
                f"not {backbone!r}"
            )
        if stage == 17:
            load_training_strategy_decision()


# ---------------------------------------------------------------------------
# 5. Versioned run manifests: reject old results and stale checkpoints
# ---------------------------------------------------------------------------

def _jsonable(value: Any):
    """Convert common numpy/path/dataclass values to stable JSON-safe values."""
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except (ValueError, TypeError):
            pass
    if hasattr(value, "tolist") and callable(value.tolist):
        try:
            return value.tolist()
        except (ValueError, TypeError):
            pass
    if hasattr(value, "__dataclass_fields__"):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("NaN/Infinity cannot appear in a run manifest")
    return value


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Stream a SHA-256 digest of a file without loading it all into memory."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Cannot hash missing input file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    """Stable SHA-256 digest of JSON-serialisable input/config metadata."""
    payload = json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def make_run_manifest(
    *,
    stage: int | str,
    script_name: str,
    config: Mapping[str, Any],
    input_paths: Sequence[str | Path] = (),
    metadata: Mapping[str, Any] | None = None,
    project_root: str | Path = ".",
) -> dict:
    """Create the immutable manifest used to identify a reproducible run.

    A manifest cannot be produced unless the registered stage script and the
    two frozen study documents are present. This prevents a stale/renamed
    script from generating apparently current outputs.
    """
    if stage not in PIPELINE_SCRIPTS:
        raise ValueError(f"Unknown pipeline stage: {stage!r}")
    expected_script = PIPELINE_SCRIPTS[stage]
    if Path(script_name).as_posix() != expected_script:
        raise ValueError(f"Stage {stage} expects script {expected_script!r}, not {script_name!r}")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError(
            f"Run config must declare schema_version={CONFIG_SCHEMA_VERSION!r}; "
            f"got {config.get('schema_version')!r}"
        )
    if config.get("run_id") != RUN_ID:
        raise ValueError(f"Run config uses {config.get('run_id')!r}, expected explicit run_id={RUN_ID!r}")
    if str(config.get("stage")) != str(stage):
        raise ValueError(f"Run config must declare stage={stage!r}")
    if Path(str(config.get("script_name", ""))).as_posix() != expected_script:
        raise ValueError(f"Run config must declare script_name={expected_script!r}")
    root = Path(project_root).resolve()
    script_path = root / expected_script
    protocol_path = root / "experiment_protocol.md"
    pipeline_path = root / "pipeline_revised.md"
    for required_path in (script_path, protocol_path, pipeline_path):
        if not required_path.is_file():
            raise FileNotFoundError(
                f"Cannot create a run manifest until required source exists: {required_path}. "
                "Run audit_pipeline_script_names() and complete the script migration first."
            )
    input_hashes = {}
    for raw_path in input_paths:
        input_path = Path(raw_path)
        if not input_path.is_absolute():
            input_path = root / input_path
        input_hashes[str(input_path.relative_to(root) if input_path.is_relative_to(root) else input_path)] = sha256_file(input_path)
    utility_path = Path(__file__).resolve()
    payload = {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "run_id": RUN_ID,
        "utility_sha256": sha256_file(utility_path),
        "script_sha256": sha256_file(script_path),
        "protocol_sha256": sha256_file(protocol_path),
        "pipeline_sha256": sha256_file(pipeline_path),
        "stage": stage,
        "script_name": expected_script,
        "config": _jsonable(dict(config)),
        "input_sha256": input_hashes,
        "metadata": _jsonable(dict(metadata or {})),
    }
    payload["fingerprint"] = sha256_json(payload)
    return payload


def versioned_run_root(project_root: str | Path = ".") -> Path:
    """Return `<project_root>/results/runs/left_truncation_v2`."""
    return Path(project_root) / Path(*RUN_ROOT_PARTS)


def _validate_manifest_integrity(manifest: Mapping[str, Any]) -> None:
    """Validate schema, namespace and self-fingerprint of a stored manifest."""
    if manifest.get("schema_version") != CONFIG_SCHEMA_VERSION or manifest.get("run_id") != RUN_ID:
        raise RuntimeError("Run manifest belongs to a historical/incompatible schema or namespace")
    expected = manifest.get("fingerprint")
    if not _is_sha256_hex(expected):
        raise RuntimeError("Run manifest has no valid SHA-256 fingerprint")
    payload = {k: v for k, v in manifest.items() if k != "fingerprint"}
    if sha256_json(payload) != expected:
        raise RuntimeError("Run manifest fingerprint is invalid; do not reuse potentially modified outputs")
    for key in ("stage", "script_name", "utility_sha256", "script_sha256", "protocol_sha256", "pipeline_sha256", "input_sha256", "config"):
        if key not in manifest:
            raise RuntimeError(f"Run manifest is incomplete; missing {key!r}")
    stage = manifest["stage"]
    if stage not in PIPELINE_SCRIPTS:
        raise RuntimeError(f"Run manifest contains an unknown stage {stage!r}")
    if manifest["script_name"] != PIPELINE_SCRIPTS[stage]:
        raise RuntimeError("Run manifest script name does not match the authoritative pipeline registry")
    cfg = manifest["config"]
    if not isinstance(cfg, Mapping):
        raise RuntimeError("Run manifest config must be a mapping")
    if cfg.get("schema_version") != CONFIG_SCHEMA_VERSION or cfg.get("run_id") != RUN_ID:
        raise RuntimeError("Run manifest nested config has a stale schema or namespace")
    if str(cfg.get("stage")) != str(stage) or Path(str(cfg.get("script_name", ""))).as_posix() != manifest["script_name"]:
        raise RuntimeError("Run manifest nested stage/script identity is inconsistent")
    for key in ("utility_sha256", "script_sha256", "protocol_sha256", "pipeline_sha256"):
        if not _is_sha256_hex(manifest[key]):
            raise RuntimeError(f"Run manifest is missing a valid {key} source fingerprint")
    if not isinstance(manifest["input_sha256"], Mapping):
        raise RuntimeError("Run manifest input_sha256 must be a mapping")
    for filename, digest in manifest["input_sha256"].items():
        if not isinstance(filename, str) or not _is_sha256_hex(digest):
            raise RuntimeError("Run manifest contains an invalid input data fingerprint")


def prepare_run_directory(
    run_dir: str | Path,
    manifest: Mapping[str, Any],
    *,
    allow_resume: bool = True,
) -> Path:
    """Create or validate a stage/condition run directory; never reuse old runs.

    - The directory must live beneath `results/runs/left_truncation_v2/`.
    - A non-empty directory without a manifest is considered historical/unsafe
      and is rejected, not silently resumed.
    - An existing manifest resumes only when its fingerprint is identical.
    - Different inputs/configs require a new run directory or a new namespace.
    """
    run_dir = Path(run_dir)
    parts = run_dir.resolve().parts
    wanted = RUN_ROOT_PARTS
    if not any(tuple(parts[i:i + len(wanted)]) == wanted for i in range(max(0, len(parts) - len(wanted) + 1))):
        raise ValueError(
            f"Run outputs must live below {'/'.join(RUN_ROOT_PARTS)!r}; received {run_dir}"
        )
    try:
        _validate_manifest_integrity(manifest)
    except RuntimeError as exc:
        raise ValueError(f"New run manifest failed integrity validation: {exc}") from exc
    manifest_path = run_dir / RUN_MANIFEST_NAME
    if run_dir.exists():
        files = [p for p in run_dir.rglob("*") if p.is_file() and p != manifest_path]
        if files and not manifest_path.exists():
            raise RuntimeError(
                f"Refusing to reuse {run_dir}: it contains files but no {RUN_MANIFEST_NAME}. "
                "Archive historical outputs and use a clean versioned run directory."
            )
        if manifest_path.exists():
            try:
                existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"Unreadable run manifest {manifest_path}: {exc}") from exc
            _validate_manifest_integrity(existing)
            if existing.get("fingerprint") != manifest.get("fingerprint"):
                raise RuntimeError(
                    f"Refusing to reuse {run_dir}: its run manifest does not match the current "
                    "configuration/data hashes. Use a new run directory or archive it."
                )
            if not allow_resume:
                raise FileExistsError(f"Run directory already exists and resume is disabled: {run_dir}")
            return run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(_jsonable(manifest), indent=2, sort_keys=True), encoding="utf-8")
    return run_dir


def find_run_manifest(path: str | Path) -> Path:
    """Find the nearest ancestor run manifest for a run artifact."""
    path = Path(path).resolve()
    start = path if path.is_dir() else path.parent
    for parent in (start, *start.parents):
        candidate = parent / RUN_MANIFEST_NAME
        if candidate.is_file():
            return candidate
        # Stop at namespace root; outside it an artifact must never be trusted.
        if tuple(parent.parts[-len(RUN_ROOT_PARTS):]) == RUN_ROOT_PARTS:
            break
    raise FileNotFoundError(
        f"No {RUN_MANIFEST_NAME} found for {path}; root-level/historical artifacts are not reusable."
    )


def assert_current_run_artifact(
    path: str | Path,
    *,
    expected_stage: int | str,
    expected_script: str,
    expected_fingerprint: str,
) -> dict:
    """Fail-closed verification before a downstream script consumes an artifact.

    All expected identifiers are mandatory: a caller may not merely accept any
    artifact that happens to carry a current schema or run ID.
    """
    artifact = Path(path)
    if not artifact.exists():
        raise FileNotFoundError(f"Required current-run artifact does not exist: {artifact}")
    manifest_path = find_run_manifest(artifact)
    try:
        doc = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot read artifact manifest {manifest_path}: {exc}") from exc
    _validate_manifest_integrity(doc)
    # A self-consistent historical manifest is still stale if the utility,
    # registered stage script, or frozen protocol documents have changed.
    parts = manifest_path.resolve().parts
    root = None
    for i in range(len(parts) - 2):
        if tuple(parts[i:i + 3]) == RUN_ROOT_PARTS:
            root = Path(*parts[:i])
            break
    if root is None:
        raise RuntimeError(f"Manifest {manifest_path} is outside the active run namespace")
    current_sources = {
        "utility_sha256": sha256_file(Path(__file__).resolve()),
        "script_sha256": sha256_file(root / expected_script) if (root / expected_script).is_file() else None,
        "protocol_sha256": sha256_file(root / "experiment_protocol.md") if (root / "experiment_protocol.md").is_file() else None,
        "pipeline_sha256": sha256_file(root / "pipeline_revised.md") if (root / "pipeline_revised.md").is_file() else None,
    }
    for key, digest in current_sources.items():
        if digest is None or doc.get(key) != digest:
            raise RuntimeError(
                f"Artifact {artifact} was produced with stale/missing source {key}; "
                "rerun it under the current utility, stage script, protocol and pipeline."
            )
    if str(doc.get("stage")) != str(expected_stage):
        raise RuntimeError(f"Artifact {artifact} came from stage {doc.get('stage')}, not {expected_stage}")
    if doc.get("script_name") != expected_script:
        raise RuntimeError(f"Artifact {artifact} was created by {doc.get('script_name')}, not {expected_script}")
    if expected_script != PIPELINE_SCRIPTS.get(expected_stage):
        raise RuntimeError(f"Expected script {expected_script!r} is not the registered script for stage {expected_stage}")
    if not _is_sha256_hex(expected_fingerprint):
        raise ValueError("expected_fingerprint must be the 64-character SHA-256 run fingerprint recorded by the producing stage")
    if doc.get("fingerprint") != expected_fingerprint:
        raise RuntimeError(f"Artifact {artifact} belongs to a different run fingerprint")
    return doc


def audit_context_cv_grid(
    run_records: Sequence[Mapping[str, Any]],
    *,
    backbone: str,
    required_folds: Sequence[int] = (0, 1, 2, 3, 4),
    required_seeds: Sequence[int] = (42, 123),
    required_contexts: Sequence[int] = (0, 1, 2, 3, 4),
) -> dict:
    """Verify the full 50-run Stage 9/10 grid using actual run manifests.

    Each record must provide `artifact_path` and `manifest_fingerprint`. This
    function calls `assert_current_run_artifact()` itself (using the correct
    stage/script for the backbone), then derives run identity/config fields
    from the verified manifest rather than trusting caller-supplied metadata.
    The returned `grid_fingerprint` identifies the sorted set of run manifests
    and the shared dataset/fold/length/truncation configuration.
    """
    get_backbone(backbone)
    if not run_records:
        raise ValueError("Experiment II-A grid audit received no run records")
    stage = 9 if backbone == "indobertweet" else 10
    expected_script = PIPELINE_SCRIPTS[stage]
    required_fields = {"artifact_path", "manifest_fingerprint"}
    problems: list[str] = []
    seen: set[tuple[int, int, int]] = set()
    verified_rows: list[dict[str, Any]] = []
    expected = {
        (int(c), int(f), int(seed))
        for c in required_contexts for f in required_folds for seed in required_seeds
    }

    for i, rec in enumerate(run_records):
        missing = sorted(required_fields - set(rec))
        if missing:
            problems.append(f"record {i} missing {missing}")
            continue
        try:
            manifest = assert_current_run_artifact(
                rec["artifact_path"],
                expected_stage=stage,
                expected_script=expected_script,
                expected_fingerprint=rec["manifest_fingerprint"],
            )
        except (FileNotFoundError, ValueError, RuntimeError, OSError) as exc:
            problems.append(f"record {i} artifact failed provenance check: {exc}")
            continue
        cfg = manifest.get("config")
        if not isinstance(cfg, Mapping):
            problems.append(f"record {i} has no resolved config in its manifest")
            continue
        cfg_required = {
            "schema_version", "run_id", "stage", "script_name", "backbone", "context_size",
            "fold", "seed", "dataset_sha256", "fold_assignment_sha256",
            "max_length_context", "truncation_side",
        }
        missing_cfg = sorted(cfg_required - set(cfg))
        if missing_cfg:
            problems.append(f"record {i} manifest config missing {missing_cfg}")
            continue
        if cfg["backbone"] != backbone:
            problems.append(f"record {i} manifest has backbone {cfg['backbone']!r}")
        if cfg["stage"] != stage and str(cfg["stage"]) != str(stage):
            problems.append(f"record {i} manifest has stage {cfg['stage']!r}, expected {stage}")
        if cfg["script_name"] != expected_script:
            problems.append(f"record {i} manifest uses script {cfg['script_name']!r}")
        if cfg["schema_version"] != CONFIG_SCHEMA_VERSION or cfg["run_id"] != RUN_ID:
            problems.append(f"record {i} manifest config belongs to an old schema/run")
        if cfg["truncation_side"] != "left":
            problems.append(f"record {i} does not use explicit left truncation")
        if not _is_sha256_hex(cfg.get("dataset_sha256")) or not _is_sha256_hex(cfg.get("fold_assignment_sha256")):
            problems.append(f"record {i} is missing valid dataset/fold SHA-256 hashes")
        try:
            context, fold, seed = int(cfg["context_size"]), int(cfg["fold"]), int(cfg["seed"])
            ml = int(cfg["max_length_context"])
        except (TypeError, ValueError):
            problems.append(f"record {i} has non-integer context/fold/seed/length values")
            continue
        if context not in range(5) or fold not in range(5) or seed not in required_seeds:
            problems.append(f"record {i} has unexpected context/fold/seed {(context, fold, seed)}")
        if ml < 8 or ml > MAX_SEQUENCE_LENGTH or ml % LENGTH_ROUNDING_MULTIPLE != 0:
            problems.append(f"record {i} has invalid max_length_context={ml}")
        key = (context, fold, seed)
        if key in seen:
            problems.append(f"duplicate context/fold/seed run {key}")
        seen.add(key)
        verified_rows.append({
            "schema_version": cfg["schema_version"],
            "run_id": cfg["run_id"],
            "backbone": backbone,
            "context_size": context,
            "fold": fold,
            "seed": seed,
            "manifest_fingerprint": manifest["fingerprint"],
            "dataset_sha256": cfg["dataset_sha256"],
            "fold_assignment_sha256": cfg["fold_assignment_sha256"],
            "max_length_context": ml,
            "truncation_side": cfg["truncation_side"],
        })

    missing_runs = sorted(expected - seen)
    unexpected_runs = sorted(seen - expected)
    if missing_runs:
        problems.append(f"missing {len(missing_runs)} expected runs: {missing_runs[:10]}")
    if unexpected_runs:
        problems.append(f"unexpected run keys: {unexpected_runs[:10]}")
    for key in ("dataset_sha256", "fold_assignment_sha256", "max_length_context", "truncation_side"):
        values = {str(r[key]) for r in verified_rows}
        if len(values) > 1:
            problems.append(f"{key} varies across the grid: {sorted(values)[:5]}")
    if problems:
        raise ValueError("Experiment II-A grid audit failed: " + " | ".join(problems))

    verified_rows.sort(key=lambda r: (r["context_size"], r["fold"], r["seed"]))
    fingerprint_rows = [
        {key: row[key] for key in (
            "schema_version", "run_id", "backbone", "context_size", "fold", "seed",
            "manifest_fingerprint", "dataset_sha256", "fold_assignment_sha256",
            "max_length_context", "truncation_side",
        )}
        for row in verified_rows
    ]
    return {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "run_id": RUN_ID,
        "backbone": backbone,
        "stage": stage,
        "script_name": expected_script,
        "n_runs": len(verified_rows),
        "expected_runs": len(expected),
        "complete": True,
        "contexts": list(required_contexts),
        "folds": list(required_folds),
        "seeds": list(required_seeds),
        "dataset_sha256": verified_rows[0]["dataset_sha256"],
        "fold_assignment_sha256": verified_rows[0]["fold_assignment_sha256"],
        "max_length_context": verified_rows[0]["max_length_context"],
        "truncation_side": "left",
        "grid_fingerprint": sha256_json(fingerprint_rows),
        "manifest_fingerprints": [row["manifest_fingerprint"] for row in verified_rows],
    }


# Canonical Stage 2 dataset expectations. Keep these here so the preprocessing
# script and downstream validators cannot drift apart.
CONTEXT_RAW_EXPECTED_ROWS = 1800
CONTEXT_MATCHED_EXPECTED_ROWS = 900
CONTEXT_SHALLOW_EXPECTED_ROWS = 900
CONTEXT_EXPECTED_LABEL_COUNTS = {"N": 600, "AL": 600, "HS": 600}
CONTEXT_EXPECTED_NEEDED_LABEL_CELLS = {
    (1, "N"): 180, (1, "AL"): 300, (1, "HS"): 365,
    (0, "N"): 420, (0, "AL"): 300, (0, "HS"): 235,
}
CONTEXT_FOLD_SEED = 42
CONTEXT_SHUFFLE_SEED = 42
CONTEXT_N_SPLITS = 5
_CONTEXT_PARENT_COLUMNS = {
    1: ["parent_1"],
    2: ["parent_2", "parent_1"],
    3: ["parent_3", "parent_2", "parent_1"],
    4: ["parent_4", "parent_3", "parent_2", "parent_1"],
}
_MISSING_TEXT_MARKERS = {"", "nan", "none", "null", "<na>", "n/a", "na"}


def _text_is_present(series: pd.Series) -> pd.Series:
    """Return true for non-empty text, excluding common CSV missing markers."""
    normalized = series.fillna("").astype(str).str.strip()
    return ~normalized.str.casefold().isin(_MISSING_TEXT_MARKERS)


def _validate_parent_availability(df: pd.DataFrame, *, require_target_text: bool = True) -> None:
    parent_columns = ["parent_4", "parent_3", "parent_2", "parent_1"]
    required = parent_columns + (["target_text"] if require_target_text else [])
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise KeyError(f"Context data missing parent/target text columns: {missing}")
    if require_target_text and not _text_is_present(df["target_text"]).all():
        bad = df.loc[~_text_is_present(df["target_text"]), CTX_ID_COLUMN].astype(str).tolist()[:10]
        raise ValueError(f"Empty target_text in contextual dataset; example post_id values: {bad}")

    if "context_size_available" not in df.columns:
        raise KeyError("Context data is missing context_size_available")
    available = pd.to_numeric(df["context_size_available"], errors="coerce")
    if available.isna().any() or not np.equal(available, np.floor(available)).all():
        raise ValueError("context_size_available must contain integers only")
    df["context_size_available"] = available.astype(int)
    available = df["context_size_available"]
    if not available.isin([1, 2, 3, 4]).all():
        bad_values = sorted(set(available[~available.isin([1, 2, 3, 4])].tolist()))
        raise ValueError(f"context_size_available must be in 1..4; found {bad_values}")

    present = {col: _text_is_present(df[col]) for col in parent_columns}
    for idx, row_index in enumerate(df.index):
        count = int(available.loc[row_index])
        expected_cols = set(_CONTEXT_PARENT_COLUMNS[count])
        actual_cols = {col for col in parent_columns if bool(present[col].loc[row_index])}
        if actual_cols != expected_cols:
            post_id = df.at[row_index, CTX_ID_COLUMN] if CTX_ID_COLUMN in df.columns else row_index
            raise ValueError(
                f"Parent availability mismatch for post_id={post_id!r}: "
                f"context_size_available={count}, populated parents={sorted(actual_cols)}, "
                f"expected={sorted(expected_cols)}"
            )


def validate_context_source_dataset(
    df: pd.DataFrame,
    *,
    expected_rows: int = CONTEXT_RAW_EXPECTED_ROWS,
    expected_label_counts: Mapping[str, int] | None = None,
    expected_needed_label_cells: Mapping[tuple[int, str], int] | None = None,
) -> dict:
    """Validate the full Stage 2 dataset after raw label spelling is normalized.

    The function validates rather than cleans/deduplicates: it deliberately
    preserves short target posts and raises if collection assumptions fail.
    """
    missing = sorted(set(CTX_ALL_RAW_COLUMNS) - set(df.columns))
    if missing:
        raise KeyError(f"Context source dataset is missing required columns: {missing}")
    if len(df) != expected_rows:
        raise ValueError(f"Expected {expected_rows} contextual rows, got {len(df)}")
    if df[CTX_ID_COLUMN].isna().any() or not df[CTX_ID_COLUMN].astype(str).str.strip().ne("").all():
        raise ValueError("Contextual post_id must be non-null and non-empty")
    if not df[CTX_ID_COLUMN].is_unique:
        duplicated = df.loc[df[CTX_ID_COLUMN].duplicated(keep=False), CTX_ID_COLUMN].astype(str).tolist()[:10]
        raise ValueError(f"Duplicate post_id values in contextual dataset: {duplicated}")
    if df["target_text"].isna().any() or not df["target_text"].astype(str).str.strip().ne("").all():
        raise ValueError("target_text must be non-empty for every contextual example")
    duplicated_text = df.duplicated(subset="target_text", keep=False)
    if duplicated_text.any():
        examples = df.loc[duplicated_text, CTX_ID_COLUMN].astype(str).tolist()[:10]
        raise ValueError(f"Duplicate target_text values found; example post_id values: {examples}")

    unexpected_labels = sorted(set(df["label"].astype(str)) - set(LABEL2ID))
    if unexpected_labels:
        raise ValueError(f"Labels must already be normalized to {sorted(LABEL2ID)}; got {unexpected_labels}")

    # Validate binary indicator values before applying the priority mapping.
    for col in ("HS", "AL", "context_needed"):
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.isna().any() or not numeric.isin([0, 1]).all():
            raise ValueError(f"{col} must contain only binary 0/1 values")
        df[col] = numeric.astype(int)
    validate_label_consistency(df)
    _validate_parent_availability(df)

    expected_label_counts = dict(expected_label_counts or CONTEXT_EXPECTED_LABEL_COUNTS)
    observed_label_counts = df["label"].value_counts().to_dict()
    if observed_label_counts != expected_label_counts:
        raise ValueError(
            f"Unexpected contextual label totals: expected {expected_label_counts}, got {observed_label_counts}"
        )

    expected_cells = dict(expected_needed_label_cells or CONTEXT_EXPECTED_NEEDED_LABEL_CELLS)
    observed_cells = df.groupby(["context_needed", "label"]).size().to_dict()
    mismatches = {
        key: {"expected": int(expected), "observed": int(observed_cells.get(key, 0))}
        for key, expected in expected_cells.items()
        if int(observed_cells.get(key, 0)) != int(expected)
    }
    unexpected_cells = {key: int(count) for key, count in observed_cells.items() if key not in expected_cells}
    if mismatches or unexpected_cells:
        raise ValueError(
            "context_needed × label distribution differs from the confirmed dataset counts; "
            f"mismatches={mismatches}, unexpected_cells={unexpected_cells}"
        )

    availability_counts = pd.to_numeric(df["context_size_available"]).astype(int).value_counts().to_dict()
    if availability_counts.get(4, 0) != CONTEXT_MATCHED_EXPECTED_ROWS or sum(
        count for available, count in availability_counts.items() if available in {1, 2, 3}
    ) != CONTEXT_SHALLOW_EXPECTED_ROWS or set(availability_counts) - {1, 2, 3, 4}:
        raise ValueError(
            "Expected 900 records with four available parents and 900 with one to three; "
            f"got {availability_counts}"
        )
    return {
        "n_rows": len(df),
        "post_id_sha256": sha256_json(sorted(df[CTX_ID_COLUMN].astype(str).tolist())),
        "label_counts": {str(k): int(v) for k, v in df["label"].value_counts().sort_index().items()},
        "context_size_available_counts": {str(k): int(v) for k, v in sorted(availability_counts.items())},
        "context_needed_label_counts": {
            f"{int(needed)}:{label}": int(count)
            for (needed, label), count in sorted(observed_cells.items(), key=lambda item: (int(item[0][0]), str(item[0][1])))
        },
    }


def validate_context_dataset(
    df: pd.DataFrame,
    *,
    expected_rows: int = CONTEXT_MATCHED_EXPECTED_ROWS,
    expected_fold_counts: Mapping[int, int] | None = None,
) -> dict:
    """Audit the matched C0-C4 subset before CV; never assigns/reassigns folds.

    Fold hashes are canonicalized by post_id, making them invariant to later
    display/shuffle order while still detecting any reassignment.
    """
    required = {
        CTX_ID_COLUMN, "target_text", "label", CTX_FOLD_COLUMN,
        "context_size_available", "parent_4", "parent_3", "parent_2", "parent_1",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"Context dataset missing required columns: {missing}")
    if len(df) != expected_rows:
        raise ValueError(f"Expected {expected_rows} matched examples, got {len(df)}")
    if df[CTX_ID_COLUMN].isna().any() or not df[CTX_ID_COLUMN].astype(str).str.strip().ne("").all():
        raise ValueError("Context dataset post_id must be non-null and non-empty")
    if not df[CTX_ID_COLUMN].is_unique:
        raise ValueError("Context dataset post_id must be unique")
    if df.duplicated(subset="target_text").any():
        raise ValueError("Duplicate target_text values found in matched context subset")
    unexpected_labels = sorted(set(df["label"].astype(str)) - set(LABEL2ID))
    if unexpected_labels:
        raise ValueError(f"Unexpected labels: {unexpected_labels}")

    fold_values = pd.to_numeric(df[CTX_FOLD_COLUMN], errors="coerce")
    if fold_values.isna().any() or not np.equal(fold_values, np.floor(fold_values)).all():
        raise ValueError("Fold assignment must contain integer values for every matched example")
    if sorted(fold_values.astype(int).unique().tolist()) != [0, 1, 2, 3, 4]:
        raise ValueError("Expected shared folds 0-4")
    if not df[CTX_FOLD_COLUMN].equals(fold_values.astype(int)):
        # Avoid accepting string/float folds that only happen to coerce to integers.
        df = df.copy()
        df[CTX_FOLD_COLUMN] = fold_values.astype(int)
    available = pd.to_numeric(df["context_size_available"], errors="coerce")
    if available.isna().any() or not (available == 4).all():
        raise ValueError("ctx_matched_subset.csv must contain only posts with exactly four available parents")
    _validate_parent_availability(df)

    expected_class_count = expected_rows // 3 if expected_rows % 3 == 0 else None
    if expected_class_count is not None:
        expected_labels = {label: expected_class_count for label in LABEL2ID}
        observed_labels = df["label"].value_counts().to_dict()
        if any(int(observed_labels.get(label, 0)) != count for label, count in expected_labels.items()):
            raise ValueError(f"Matched subset must be balanced: expected {expected_labels}, got {observed_labels}")
    if expected_fold_counts is None and expected_rows % CONTEXT_N_SPLITS == 0:
        expected_fold_counts = {fold: expected_rows // CONTEXT_N_SPLITS for fold in range(CONTEXT_N_SPLITS)}
    fold_series = fold_values.astype(int)
    if expected_fold_counts is not None:
        actual = fold_series.value_counts().to_dict()
        for fold, expected_count in expected_fold_counts.items():
            if int(actual.get(int(fold), 0)) != int(expected_count):
                raise ValueError(f"Fold {fold} has {actual.get(int(fold), 0)} rows; expected {expected_count}")

    canonical = df.assign(**{CTX_FOLD_COLUMN: fold_series}).sort_values(CTX_ID_COLUMN, key=lambda col: col.astype(str))
    return {
        "n_rows": len(df),
        "n_unique_post_id": int(df[CTX_ID_COLUMN].nunique()),
        "label_counts": {str(k): int(v) for k, v in df["label"].value_counts().sort_index().items()},
        "fold_counts": {str(k): int(v) for k, v in fold_series.value_counts().sort_index().items()},
        "post_id_sha256": sha256_json(canonical[CTX_ID_COLUMN].astype(str).tolist()),
        "fold_assignment_sha256": sha256_json(canonical[[CTX_ID_COLUMN, CTX_FOLD_COLUMN]].astype(str).values.tolist()),
    }


def validate_context_shallow_pool(
    df: pd.DataFrame,
    *,
    expected_rows: int = CONTEXT_SHALLOW_EXPECTED_ROWS,
) -> dict:
    """Validate availability-aware inference pool without assigning CV folds."""
    required = {CTX_ID_COLUMN, "target_text", "label", "context_size_available", "parent_4", "parent_3", "parent_2", "parent_1"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"Shallow context pool missing required columns: {missing}")
    if len(df) != expected_rows:
        raise ValueError(f"Expected {expected_rows} shallow-pool examples, got {len(df)}")
    if df[CTX_ID_COLUMN].isna().any() or not df[CTX_ID_COLUMN].astype(str).str.strip().ne("").all() or not df[CTX_ID_COLUMN].is_unique:
        raise ValueError("Shallow pool post_id values must be non-null, non-empty and unique")
    if df["target_text"].isna().any() or not df["target_text"].astype(str).str.strip().ne("").all():
        raise ValueError("Shallow pool target_text must be non-empty")
    labels = df["label"].astype(str)
    unexpected = sorted(set(labels) - set(LABEL2ID))
    if unexpected:
        raise ValueError(f"Unexpected shallow-pool labels: {unexpected}")
    counts = labels.value_counts().to_dict()
    expected_counts = {label: expected_rows // 3 for label in LABEL2ID}
    if expected_rows % 3 == 0 and counts != expected_counts:
        raise ValueError(f"Shallow pool must be class balanced: expected {expected_counts}, got {counts}")
    available = pd.to_numeric(df["context_size_available"], errors="coerce")
    if available.isna().any() or not available.isin([1, 2, 3]).all():
        raise ValueError("Shallow pool must contain context_size_available values 1, 2 or 3 only")
    _validate_parent_availability(df)
    return {
        "n_rows": len(df),
        "post_id_sha256": sha256_json(sorted(df[CTX_ID_COLUMN].astype(str).tolist())),
        "label_counts": {str(k): int(v) for k, v in labels.value_counts().sort_index().items()},
        "context_size_available_counts": {str(k): int(v) for k, v in available.astype(int).value_counts().sort_index().items()},
    }


def assert_current_stage_manifest(stage: int | str, project_root: str | Path = ".") -> dict:
    """Validate a stage-level manifest and all declared input/output hashes.

    Unlike model artifact checks, this also supports preprocessing stages whose
    authoritative output files are stored at stable data/processed paths. It
    verifies the current utility/script/protocol/pipeline fingerprints and the
    live hashes of the raw inputs and outputs.
    """
    if stage not in PIPELINE_SCRIPTS:
        raise ValueError(f"Unknown stage: {stage!r}")
    root = Path(project_root).resolve()
    # Numeric/letter stages use canonical folder labels: stage_02 and stage_1b.
    stage_label = "1b" if stage == "1b" else f"{int(stage):02d}" if isinstance(stage, int) or str(stage).isdigit() else str(stage)
    manifest_path = root / Path(*RUN_ROOT_PARTS) / f"stage_{stage_label}" / RUN_MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Stage {stage} manifest not found: {manifest_path}")
    try:
        doc = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot read stage manifest {manifest_path}: {exc}") from exc
    _validate_manifest_integrity(doc)
    if str(doc.get("stage")) != str(stage):
        raise RuntimeError(f"Manifest {manifest_path} describes stage {doc.get('stage')}, not {stage}")
    expected_script = PIPELINE_SCRIPTS[stage]
    source_files = {
        "utility_sha256": Path(__file__).resolve(),
        "script_sha256": root / expected_script,
        "protocol_sha256": root / "experiment_protocol.md",
        "pipeline_sha256": root / "pipeline_revised.md",
    }
    for key, path in source_files.items():
        if not path.is_file() or sha256_file(path) != doc.get(key):
            raise RuntimeError(f"Stage {stage} manifest has stale or missing source fingerprint: {key}")
    for relative_path, expected_hash in doc.get("input_sha256", {}).items():
        path = Path(relative_path)
        if not path.is_absolute():
            path = root / path
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise RuntimeError(f"Stage {stage} input has changed or is missing: {relative_path}")
    outputs = doc.get("metadata", {}).get("output_sha256")
    if not isinstance(outputs, Mapping) or not outputs:
        raise RuntimeError(f"Stage {stage} manifest has no output_sha256 metadata")
    for relative_path, expected_hash in outputs.items():
        path = Path(relative_path)
        if not path.is_absolute():
            path = root / path
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise RuntimeError(f"Stage {stage} output has changed or is missing: {relative_path}")
    return doc

