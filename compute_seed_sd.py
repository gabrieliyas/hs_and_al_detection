"""Estimate the seed-to-seed standard deviation of macro-F1 ([SD_seed]).

Why this exists
---------------
Section 3.9.1 justifies using two random seeds by comparing two sources of
uncertainty in the pooled macro-F1:

* sampling noise from the finite 900-post evaluation set (about 1.0 point
  for accuracy under the stated assumptions), which more seeds cannot reduce;
* initialization / data-order noise, which averaging over s seeds reduces
  in proportion to 1/s.

[SD_seed] is an empirical estimate of the second source, taken from runs
that already exist. It needs, for each condition (e.g. one C-level of
Experiment II or III) and each seed, the held-out softmax probabilities of
every fold. The same seeds and folds must be used for every condition.

Usage
-----
    runs = {
        "E3-IndoBERTweet-C0": {
            1: [fold0_probs, fold1_probs, ...],   # (n_fold, 3) arrays
            2: [fold0_probs, fold1_probs, ...],
        },
        "E3-IndoBERTweet-C4": {...},
    }
    table = seed_sd_table(y_true, fold_indices, runs)
    print(table)
    print(summarize_seed_sd(table))

``y_true`` is the (900,) array of integer labels in the original example
order and ``fold_indices`` is the list of held-out index arrays, one per fold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from experiment_utils import pool_out_of_fold_probabilities

# {condition: {seed: [probability array of fold 0, fold 1, ...]}}
RunProbs = dict[str, dict[int, list[np.ndarray]]]


def _macro_f1(y_true: np.ndarray, probs: np.ndarray) -> float:
    """Macro-F1 of the argmax prediction of one probability array."""
    return float(f1_score(y_true, probs.argmax(axis=1), average="macro"))


def seed_sd_table(
    y_true,
    fold_indices: list[np.ndarray],
    runs: RunProbs,
) -> pd.DataFrame:
    """One row per condition with the seed SD in macro-F1 percentage points.

    pooled_sd:     SD across seeds of the macro-F1 computed on the pooled
                   out-of-fold predictions of that seed. This is the number
                   to compare with the ~1.0-point sampling SE.
    fold_level_sd: root-mean-square over folds of the SD across seeds of the
                   per-fold macro-F1. It is larger than pooled_sd because a
                   fold has only 180 posts; report it only as a secondary
                   figure.
    """
    y_true = np.asarray(y_true)
    rows = []
    for condition, by_seed in runs.items():
        seeds = sorted(by_seed)
        if len(seeds) < 2:
            raise ValueError(f"{condition}: at least two seeds are required")

        # Pooled out-of-fold macro-F1, one value per seed.
        pooled = [
            _macro_f1(y_true, pool_out_of_fold_probabilities(fold_indices, by_seed[s]))
            for s in seeds
        ]

        # Per-fold macro-F1, shape (n_seeds, n_folds).
        fold_f1 = np.array([
            [_macro_f1(y_true[idx], by_seed[s][k]) for k, idx in enumerate(fold_indices)]
            for s in seeds
        ])
        sd_per_fold = fold_f1.std(axis=0, ddof=1)

        rows.append({
            "condition": condition,
            "n_seeds": len(seeds),
            "pooled_sd": 100 * float(np.std(pooled, ddof=1)),
            "fold_level_sd": 100 * float(np.sqrt(np.mean(sd_per_fold ** 2))),
        })
    return pd.DataFrame(rows)


def summarize_seed_sd(table: pd.DataFrame) -> dict[str, float]:
    """Combine conditions into one [SD_seed] value.

    With two seeds each condition gives a variance estimate with only one
    degree of freedom, so a single condition is very noisy. Variances are
    therefore averaged across conditions (root mean square), which assumes
    the seed noise is similar across conditions.
    """
    return {
        "n_conditions": int(len(table)),
        "SD_seed_pooled": float(np.sqrt(np.mean(table["pooled_sd"] ** 2))),
        "SD_seed_fold_level": float(np.sqrt(np.mean(table["fold_level_sd"] ** 2))),
        "min_condition_pooled_sd": float(table["pooled_sd"].min()),
        "max_condition_pooled_sd": float(table["pooled_sd"].max()),
    }
