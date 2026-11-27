"""Command-line runner for compute_seed_sd.py ([SD_seed] estimate).

Place this file and compute_seed_sd.py in the repository root, next to
experiment_utils.py, and run from that folder:

    python run_seed_sd.py --demo      # proves the script runs (synthetic data)
    python run_seed_sd.py             # real run on your saved probabilities

Real run: for every condition, seed and fold it loads one .npy file holding
the held-out softmax probabilities, shape (n_examples_in_fold, 3), with rows
in the original row order of data/processed/ctx_matched_subset.csv (the order
of ``np.where(fold == k)``). Edit --pattern if your files are named or stored
differently.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from compute_seed_sd import RunProbs, seed_sd_table, summarize_seed_sd
from experiment_utils import LABEL2ID

DEFAULT_CSV = "data/processed/ctx_matched_subset.csv"
DEFAULT_PATTERN = "results/predictions/{condition}/seed{seed}_fold{fold}.npy"
DEFAULT_CONDITIONS = [f"E3-IndoBERTweet-C{n}" for n in range(5)]
DEFAULT_SEEDS = (1, 2)
OUTPUT_CSV = Path("results/tables/seed_sd.csv")


def load_real_inputs(
    csv_path: str,
    pattern: str,
    conditions: list[str],
    seeds: tuple[int, ...],
) -> tuple[np.ndarray, list[np.ndarray], RunProbs]:
    """Read labels, fold assignment and per-run probabilities from disk."""
    df = pd.read_csv(csv_path)
    y_true = df["label"].map(LABEL2ID).to_numpy()
    fold_values = sorted(df["fold"].unique())
    fold_indices = [np.where(df["fold"].to_numpy() == f)[0] for f in fold_values]

    runs: RunProbs = {}
    missing: list[str] = []
    for condition in conditions:
        runs[condition] = {}
        for seed in seeds:
            arrays = []
            for fold_pos, fold_value in enumerate(fold_values):
                path = Path(pattern.format(condition=condition, seed=seed, fold=fold_value))
                if not path.exists():
                    missing.append(str(path))
                    continue
                probs = np.load(path)
                expected = (len(fold_indices[fold_pos]), len(LABEL2ID))
                if probs.shape != expected:
                    raise ValueError(f"{path}: shape {probs.shape}, expected {expected}")
                arrays.append(probs)
            runs[condition][seed] = arrays

    if missing:
        shown = "\n  ".join(missing[:10])
        raise FileNotFoundError(
            f"{len(missing)} probability file(s) not found. First ones:\n  {shown}\n"
            "Check --pattern, --conditions and --seeds."
        )
    return y_true, fold_indices, runs


def make_demo_inputs() -> tuple[np.ndarray, list[np.ndarray], RunProbs]:
    """Synthetic data only: the printed numbers have no meaning."""
    rng = np.random.default_rng(0)
    n, n_folds = 900, 5
    y_true = np.repeat([0, 1, 2], n // 3)
    order = rng.permutation(n)
    fold_indices = [np.sort(order[i::n_folds]) for i in range(n_folds)]

    def one_run() -> list[np.ndarray]:
        out = []
        for idx in fold_indices:
            logits = rng.normal(0, 1.0, (len(idx), 3))
            logits[np.arange(len(idx)), y_true[idx]] += 1.5
            exp = np.exp(logits)
            out.append(exp / exp.sum(axis=1, keepdims=True))
        return out

    runs: RunProbs = {f"demo-C{c}": {1: one_run(), 2: one_run()} for c in range(5)}
    return y_true, fold_indices, runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true", help="run on synthetic data")
    parser.add_argument("--csv", default=DEFAULT_CSV)
    parser.add_argument("--pattern", default=DEFAULT_PATTERN)
    parser.add_argument("--conditions", nargs="+", default=DEFAULT_CONDITIONS)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    args = parser.parse_args()

    if args.demo:
        print("*** DEMO: synthetic data, the numbers below are meaningless ***\n")
        y_true, fold_indices, runs = make_demo_inputs()
    else:
        y_true, fold_indices, runs = load_real_inputs(
            args.csv, args.pattern, args.conditions, tuple(args.seeds)
        )

    table = seed_sd_table(y_true, fold_indices, runs)
    summary = summarize_seed_sd(table)

    print(table.round(3).to_string(index=False))
    print("\nSummary (macro-F1 percentage points):")
    for key, value in summary.items():
        print(f"  {key}: {value:.3f}" if isinstance(value, float) else f"  {key}: {value}")

    if not args.demo:
        OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(OUTPUT_CSV, index=False)
        print(f"\nSaved per-condition table to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
