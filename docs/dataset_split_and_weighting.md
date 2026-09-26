# Dataset Splitting Strategy and Class Weight Calculations

This document records finalized decisions and verified calculations for
dataset splitting and class weighting across Experiments I, II, and III.
It is the reference for these decisions during implementation and thesis
writing (Chapter III methodology).

---

## 1. Ibrohim & Budi dataset (Experiment I)

### 1.1 Cleaned dataset size -- reproduced, not borrowed

The commonly-cited "13,126-13,130 rows after cleaning" figure is real: a
2026 paper (*A Comparative Study of PyCaret AutoML and CNN-BiLSTM for
Binary Hate Speech Detection in Indonesian Twitter*) explicitly reports a
13,130-row cleaned table derived from the original 13,169-tweet corpus.

However, that figure reflects **that paper's own, unspecified cleaning
criteria** -- not a canonical count published by Ibrohim & Budi (2019)
themselves. When the preprocessing pipeline already committed to for this
thesis (remove duplicate tweets; strip `USER`/`URL` placeholder tokens and
raw URLs; remove tweets with fewer than 3 words after stripping; remove
empty texts) is actually run against the real `re_dataset.csv`, the result
is **12,746 rows**, not 13,130:

| Step | Rows removed | Rows remaining |
|---|---|---|
| Raw corpus | -- | 13,169 |
| Exact duplicate tweets removed | 146 | 13,023 |
| Empty after stripping USER/URL placeholders | 6 | 13,017 |
| Fewer than 3 words after stripping placeholders | 271 | **12,746** |

**Decision: use 12,746 as the Experiment I dataset size**, since it is the
number that is actually reproducible from the documented pipeline, rather
than adopting an external figure whose exact cleaning steps aren't
disclosed and can't be independently verified. State this reasoning
explicitly in Chapter III: report both figures (12,746 from this thesis's
own pipeline vs. the externally-cited 13,130) and note the discrepancy is
expected, since different studies apply different (and often unstated)
cleaning thresholds to the same source corpus.

### 1.2 Three-class label distribution (on the cleaned 12,746 rows)

Using the priority mapping (`HS=1` -> HS; else `Abusive=1` -> AL; else N):

| Class | Count | % |
|---|---|---|
| Neither (N) | 5,699 | 44.71% |
| Hate Speech (HS) | 5,383 | 42.23% |
| Abusive (AL) | 1,664 | 13.06% |

(Nearly identical proportions to the raw, uncleaned corpus -- 44.5% / 42.2%
/ 13.3% -- confirming the cleaning step removes low-quality tweets roughly
proportionally across classes rather than disproportionately affecting any
one class.)

### 1.3 Split ratio and stratification

**Decision: 70% train / 15% validation / 15% test, stratified by the
3-class label.**

Stratification is recommended as a strictly-better default here: it costs
nothing (`train_test_split(..., stratify=y)`), and removes any risk of the
smallest class (Abusive, 13.06%) landing at a meaningfully different
proportion in the training partition purely by chance.

Verified split sizes on the 12,746-row cleaned dataset:

| Partition | Rows | % |
|---|---|---|
| Train | 8,922 | 70.0% |
| Validation | 1,912 | 15.0% |
| Test | 1,912 | 15.0% |

Train partition label counts (stratified, `random_state=42`):

| Class | Count | % |
|---|---|---|
| N | 3,989 | 44.71% |
| HS | 3,768 | 42.23% |
| AL | 1,165 | 13.06% |

### 1.4 Class weight calculation -- computed from the TRAINING partition only

**Principle (confirmed):** weights must be computed after splitting, from
the training partition's realized label counts -- never from the full
dataset before splitting. With a stratified split, the training partition
mirrors the population closely, so the difference is usually small; but
computing from the train partition is the methodologically correct
practice regardless, and becomes materially important if stratification
is ever skipped. Illustration, using an unstratified split on the same
data (`random_state=7`, for comparison only -- not the actual pipeline):

| Class | Stratified train % | Non-stratified train % (illustrative) |
|---|---|---|
| N | 44.71% | 44.65% |
| HS | 42.23% | 42.19% |
| AL | 13.06% | 13.16% |

Formula used ("balanced" weighting, equivalent to scikit-learn's
`class_weight="balanced"`):

```
w_c = n_train / (n_classes * count_c)
```

**Final class weights for Experiment I** (computed from the stratified
training partition, n_train = 8,922, n_classes = 3):

| Class | count_c (train) | Weight w_c |
|---|---|---|
| N | 3,989 | 0.7456 |
| AL | 1,165 | 2.5528 |
| HS | 3,768 | 0.7893 |

Implementation: `experiment_utils.compute_class_weights(train_labels)`
returns these weights keyed by integer label id (via `LABEL2ID`), ready to
pass into `torch.nn.CrossEntropyLoss(weight=...)`.

```python
from experiment_utils import compute_class_weights, LABEL2ID
import torch

weights_by_id = compute_class_weights(train_df["label"])
weight_tensor = torch.tensor(
    [weights_by_id[i] for i in range(len(LABEL2ID))], dtype=torch.float
)
loss_fn = torch.nn.CrossEntropyLoss(weight=weight_tensor)
```

---

## 2. Custom contextual dataset (Experiments II and III)

### 2.0 Raw label spelling -- normalized at load time, not a data change

The annotated export spells the `label` column out in full (`neither`,
`abusive_language`, `hate_speech`), rather than using the short codes
(`N`/`AL`/`HS`) used everywhere else in this pipeline, including Experiment
I's own cleaned dataset, `LABEL2ID`, and every other decision in this
document. `2_preprocess_contextual.py` maps the raw spelling onto the
canonical codes once, immediately after loading the raw file and before any
validation or distribution check runs (`experiment_utils.RAW_CTX_LABEL_MAP`).
This is an ingestion-normalization step, not a reinterpretation of the
data: the underlying `HS`/`AL` 0/1 indicator columns are untouched, and the
original spelling is retained as an additional `label_raw` column in the
processed output for traceability -- it is analysis-only and is excluded
from the model input by `load_modelling_view()`, the same way as the other
non-modelling columns.

The short codes (`N`/`AL`/`HS`) were kept as the canonical vocabulary,
rather than adopting the raw dataset's spelling as canonical, because that
vocabulary is already load-bearing across Experiment I's own cleaned
dataset, `LABEL2ID`/`ID2LABEL`, and the rest of this document; changing it
would require re-touching and re-verifying that already-tested pipeline for
no methodological benefit.

### 2.1 No weighted loss needed

**Decision: do not apply class weighting to the custom contextual
dataset.** Unlike Ibrohim & Budi, this dataset was collected via an
explicit quota scheme that deliberately produced a balanced distribution
-- 600 Neither / 600 Abusive / 600 Hate Speech out of 1,800 total, held
consistent across every parent-count stratum by design (see
`dataset_collection_targets.py` and the final collected distribution in
`README.md`). Since the class imbalance problem this technique solves
does not exist in this dataset, applying it would add unnecessary
complexity with no benefit -- and could even distort training if applied
on top of an already-balanced set. Weighted loss is an Experiment I
(Ibrohim & Budi)-only concern.

### 2.2 Splitting strategy: stratified k-fold, not a fixed train/val/test split

**Decision: 5-fold stratified cross-validation on the matched 4-parent
subset (context_size_available == 4, 900 examples), stratified by
(context_needed, label).** A single fixed split was rejected earlier in
favor of k-fold specifically because of this subset's small size relative
to the number of comparisons it needs to support (see prior discussion on
statistical power for the C0-C4 ablation and the context-needed / label
subgroup analyses).

### 2.3 `context_size_available` and fold leakage -- resolved by construction, not by adding a stratification variable

This is worth stating precisely, since the risk is real but the fix is
mechanical rather than a new stratification variable:

- The C0-C4 ablation conditions are **not five separate pools of
  examples** -- they are five different *truncated views of the same 900
  base examples* (each base example has context_size_available == 4, and
  C0/C1/C2/C3/C4 are constructed by truncating how many of its parent
  messages are included).
- Therefore, **fold assignment must happen once, at the base-example
  level, before generating the truncated views** -- a single `fold`
  column (0-4) added to `ctx_matched_subset.csv`, reused identically for
  every C-level. `context_size_available` itself does not need to be a
  stratification variable, because it is already constant (=4) within
  this subset by construction.
- **The actual failure mode to avoid:** running `train_test_split` or
  k-fold assignment independently for each C-level would risk the same
  underlying post landing in the training fold for one context size and
  the test fold for another -- e.g., post #123 in train for C2 but in
  test for C3. Since all five views share the same label and largely
  overlapping text, this would be a real data leakage bug, not just an
  inconsistency. The single shared `fold` column prevents this by
  construction.
- If the full 1,800-post dataset (including the parent_count 1-3 strata)
  is ever used for a broader training purpose beyond the core ablation,
  stratifying that split by `context_size_available` in addition to
  `label` would become a genuinely separate, relevant decision at that
  point -- it is not currently needed for the core Experiment II/III
  design as scoped.

### 2.4 Fold assignment implementation

```python
from sklearn.model_selection import StratifiedKFold
import pandas as pd

matched = pd.read_csv("data/processed/ctx_matched_subset.csv")  # 900 rows, all context_size_available == 4
strat_key = matched["context_needed"].astype(str) + "_" + matched["label"]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
matched["fold"] = -1
for fold_id, (_, test_idx) in enumerate(skf.split(matched, strat_key)):
    matched.loc[test_idx, "fold"] = fold_id

matched.to_csv("data/processed/ctx_matched_subset.csv", index=False)
```

This `fold` column, once written, is the single source of truth for which
examples are held out in which fold -- reused identically across every
C0-C4 truncation and across both Experiment II and Experiment III, so
their comparisons remain on matched data as required.
