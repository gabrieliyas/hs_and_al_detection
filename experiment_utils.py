"""
experiment_utils.py

Shared utilities for the Indonesian hate-speech / abusive-language thesis pipeline:
1. An explicit experiment-ID registry (avoids ambiguous names like "IndoBERTweet-I"
   vs. "IndoBERTweet-I+C0", and avoids leaving "C_best" as a literal symbolic string).
2. Evaluation helpers: paired bootstrap CI for comparing two models' predictions,
   and seed-aggregation for reporting mean +/- std across random seeds.

Design notes (terminology as revised for the thesis):
- Experiment I: trained + evaluated only on the Ibrohim & Budi dataset (baseline).
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

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


# ---------------------------------------------------------------------------
# 0. Label mapping -- single source of truth for label <-> integer id.
# Import this everywhere (SVM pipeline, all IndoBERTweet variants across
# Experiments I/II/III) so no experiment accidentally uses a different int
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
# Every training script (Stage 8 onward) must build its model input via
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
# 7_timing_pilot.py and 8_train_e2_ablation.py build their input text
# through this function, so the two can never silently disagree on how a
# given C-level's input is constructed.
_CONTEXT_PARENT_COLUMNS = {
    0: [],
    1: ["parent_1"],
    2: ["parent_2", "parent_1"],
    3: ["parent_3", "parent_2", "parent_1"],
    4: ["parent_4", "parent_3", "parent_2", "parent_1"],
}


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
    cols = _CONTEXT_PARENT_COLUMNS[context_size]
    pieces = [str(row[c]) for c in cols] + [str(row["target_text"])]
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

# Experiment I -- Baseline: benchmark on Ibrohim & Budi only (target post,
# no conversational context)
EXPERIMENT1_IDS = {
    "svm": "E1-SVM-TFIDF",
    "transformer": "E1-IndoBERTweet-Target",
}

# Experiment II -- Context-Aware with Intermediate Task Fine-Tuning (ITFT):
# fine-tuned on Ibrohim & Budi first, then on the custom contextual dataset
# (matched 4-parent subset), context window ablation C0..C4
EXPERIMENT2_IDS = {n: f"E2-IndoBERTweet-ITFT-C{n}" for n in range(5)}

# Experiment III -- Context-Aware without ITFT: fine-tuned on the custom
# contextual dataset only, no intermediate Ibrohim & Budi fine-tuning stage
EXPERIMENT3_IDS = {n: f"E3-IndoBERTweet-C{n}" for n in range(5)}


def resolve_best_context_id(
    experiment2_scores: dict[int, float],
) -> tuple[int, str, str]:
    """
    Resolve the best context size from Experiment II ablation results.

    Args:
        experiment2_scores: mapping of context size (0-4) -> mean macro-F1
            from the Experiment II ablation (averaged across folds/seeds).

    Returns:
        (best_n, experiment2_id, experiment3_id), where the IDs are the
        concrete Experiment II and Experiment III identifiers for the same
        context size. Never store the literal placeholder "C_best" as a
        model/experiment identifier.
    """
    if not experiment2_scores:
        raise ValueError("experiment2_scores must not be empty")

    invalid = [n for n in experiment2_scores if n not in range(5)]
    if invalid:
        raise ValueError(
            f"Invalid Experiment II context size(s): {invalid}. Expected 0-4."
        )

    # max() returns the first occurrence on an exact tie. Since callers
    # should supply context sizes in ascending order, this provides the
    # deterministic tie-break of preferring the smaller context window.
    best_n = max(experiment2_scores, key=experiment2_scores.get)
    return best_n, EXPERIMENT2_IDS[best_n], EXPERIMENT3_IDS[best_n]


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
) -> tuple[int, str, str, pd.DataFrame]:
    """
    Select the best C-level from the Experiment II CV grid.

    Selection is based only on the mean metric across the available
    fold/seed runs. Exact ties are resolved in favour of the smaller
    context window (lower C), giving a deterministic and more parsimonious
    choice.

    Returns:
        (best_n, experiment2_id, experiment3_id, summary_df)
    """
    summary = summarize_context_cv_scores(grid_df, metric=metric)
    if summary.empty:
        raise ValueError("Experiment II CV grid contains no rows")

    scores = {
        int(row["condition"]): float(row["mean"])
        for _, row in summary.iterrows()
    }
    best_n, e2_id, e3_id = resolve_best_context_id(scores)
    return best_n, e2_id, e3_id, summary


# ---------------------------------------------------------------------------
# 2. Evaluation helpers
# ---------------------------------------------------------------------------

def ensemble_seed_probabilities(prob_arrays: list[np.ndarray]) -> np.ndarray:
    """
    Average softmax probability arrays across multiple random-seed runs and
    return the argmax as a single representative prediction set.

    Use this to build the "transformer" side of a paired_bootstrap_ci call
    against a deterministic baseline (e.g. SVM) when the transformer was
    trained with multiple seeds -- averaging probabilities uses information
    from every seed rather than arbitrarily picking one run's predictions.
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
    - Experiment I: SVM (preds_a) vs. IndoBERTweet-Target (preds_b) on the fixed test set.
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
