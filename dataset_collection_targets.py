"""
dataset_collection_targets.py

Per-cell collection targets for the custom contextual dataset (parent_count x
context_needed x label), and a progress-tracking function to compare against
what has actually been annotated so far.

Run `collection_progress_table` periodically during annotation (e.g. every
~150-200 new posts) so you always know which specific cell to prioritize
sourcing next, rather than discovering a shortfall only after collection
is "done."
"""

import pandas as pd

# (parent_count_available, context_needed, label) -> target count
# Derived from the revised distribution: depths 1-3 unchanged from the
# original plan, depth 4 increased from 200/label to 300/label to support
# a healthier C0-C4 ablation subset (900 examples instead of 600).
TARGET_TABLE = {
    (1, 1, "N"): 15,  (1, 1, "AL"): 25,  (1, 1, "HS"): 25,
    (1, 0, "N"): 35,  (1, 0, "AL"): 25,  (1, 0, "HS"): 25,
    (2, 1, "N"): 30,  (2, 1, "AL"): 50,  (2, 1, "HS"): 60,
    (2, 0, "N"): 70,  (2, 0, "AL"): 50,  (2, 0, "HS"): 40,
    (3, 1, "N"): 45,  (3, 1, "AL"): 75,  (3, 1, "HS"): 90,
    (3, 0, "N"): 105, (3, 0, "AL"): 75,  (3, 0, "HS"): 60,
    (4, 1, "N"): 90,  (4, 1, "AL"): 150, (4, 1, "HS"): 190,
    (4, 0, "N"): 210, (4, 0, "AL"): 150, (4, 0, "HS"): 110,
}


def collection_progress_table(
    df,
    target_table: dict = TARGET_TABLE,
    parent_count_col: str = "context_size_available",
    context_needed_col: str = "context_needed",
    label_col: str = "label",
) -> pd.DataFrame:
    """
    Compare currently-annotated counts against the per-cell targets above.

    Returns a table with one row per (parent_count, context_needed, label)
    cell showing: target, collected so far, remaining needed, and percent
    complete. Sort by `remaining` descending to see what to prioritize next.
    """
    actual = df.groupby([parent_count_col, context_needed_col, label_col]).size()

    rows = []
    for (parent_count, context_needed, label), target in target_table.items():
        collected = actual.get((parent_count, context_needed, label), 0)
        rows.append({
            "parent_count": parent_count,
            "context_needed": context_needed,
            "label": label,
            "target": target,
            "collected": collected,
            "remaining": max(target - collected, 0),
            "pct_complete": round(100 * collected / target, 1) if target else None,
        })

    return pd.DataFrame(rows).sort_values(
        ["parent_count", "context_needed", "label"]
    ).reset_index(drop=True)
