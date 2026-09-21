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


def resolve_best_context_id(experiment2_scores: dict[int, float]) -> tuple[int, str]:
    """
    Resolve "C_best" to a concrete context size based on Experiment II
    ablation results.

    Args:
        experiment2_scores: mapping of context size (0-4) -> mean macro-F1
            from the Experiment II ablation (averaged across folds/seeds).

    Returns:
        (best_n, experiment3_id) where experiment3_id is e.g.
        "E3-IndoBERTweet-C3". Use this returned string everywhere
        downstream -- never store the literal placeholder "C_best" in
        filenames, configs, or result tables.
    """
    best_n = max(experiment2_scores, key=experiment2_scores.get)
    return best_n, EXPERIMENT3_IDS[best_n]


# ---------------------------------------------------------------------------
# 2. Evaluation helpers
# ---------------------------------------------------------------------------

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
    Cross-tabulate parent_count_available x (context_needed, label).

    Run this:
    - Periodically during annotation (e.g. every ~200 new posts), to catch
      under-filled combinations while there is still time to target
      collection toward them.
    - Once on the full dataset, to confirm the overall 500/500/500 x
      context-needed distribution holds as designed.
    - Filtered to parent_count_available == 4 only, right before Experiment
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
