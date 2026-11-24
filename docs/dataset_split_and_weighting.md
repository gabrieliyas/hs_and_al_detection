# Dataset Splitting Strategy and Class-Weight Calculations

This document records the dataset-preparation, partitioning, fold-assignment,
and class-weighting decisions for the current Transformer pipeline. It is a
reference for implementation and Chapter III methodology. The canonical
implementation is defined by `experiment_utils.py` and the registered Stage 1
and Stage 2 preprocessing scripts.

> **Version and reporting note (24 November 2026).** The benchmark preprocessing
> procedure normalizes text before deduplication and audits duplicate texts with
> conflicting mapped labels. The benchmark figures in Sections 1.3 and 1.4 and the
> contextual figures in Section 2.4 were reconciled against the successful Stage 1
> audit (`ib_preprocessing_audit.csv`, `ib_class_weights.csv`) and the Stage 2
> distribution table (`ctx_distribution.csv`). They supersede the values in earlier
> versions of this document, which came from a previous preprocessing run and
> were internally inconsistent. If the raw benchmark file, cleaning logic,
> conflict policy, or split configuration changes, rerun Stage 1 and replace
> these figures with the new audit values.

---

## 1. Ibrohim & Budi benchmark dataset

### 1.1 Stage 1 preprocessing and cleaning

**Input:** `data/raw/re_dataset.csv`  
**Script:** `scripts/1_preprocess_benchmark.py`  
**Outputs:**

- `data/processed/ib_cleaned.csv`
- `data/processed/ib_train.csv`
- `data/processed/ib_val.csv`
- `data/processed/ib_test.csv`
- `results/tables/ib_class_weights.csv`
- `results/tables/ib_preprocessing_audit.json`
- `results/tables/ib_preprocessing_audit.csv`

The current preprocessing procedure is:

1. Read the original CSV using the configured legacy-compatible `latin-1`
   decoding. Malformed CSV records skipped by the parser are counted in the
   preprocessing audit rather than being silently forgotten.
2. Normalize tweet text with Unicode NFKC, remove raw HTTP/HTTPS URLs and
   standalone `USER` and `URL` placeholder tokens (case-insensitive), collapse
   whitespace, and preserve letter casing. The result is stored as
   `clean_text`; the source text is retained as `text_raw`.
3. Map the two source indicator columns to the canonical three-class label:
   `HS=1` takes priority and maps to `HS`; otherwise `Abusive=1` maps to `AL`;
   otherwise the row maps to `N`.
4. Validate text and labels. Missing/blank text, text that becomes empty after
   cleaning, and text shorter than three whitespace-separated words are
   excluded and recorded in `data/processed/ib_invalid_rows.csv`. `HS` and
   `Abusive` must both be valid binary values (`0` or `1`); invalid or missing
   label values stop preprocessing and must be resolved rather than silently
   assigned a class.
5. Deduplicate globally on the final normalized `clean_text` value **before
   splitting**. This prevents normalized duplicates from being distributed
   across train, validation, and test partitions.
6. Resolve groups where the same `clean_text` has conflicting mapped labels
   using the explicit `CONFLICT_POLICY` in the Stage 1 script. The current
   default is `exclude`: all rows in each conflicting group are excluded. The
   alternative `error` policy stops the run. The `adjudicate` policy requires
   one documented decision per conflicting group in
   `data/annotations/ib_duplicate_label_adjudication.csv`, including an action,
   rationale, and reviewer. An adjudicated group retains one deterministic
   representative row and uses the explicitly adjudicated three-class label;
   an excluded group contributes no row. No majority-vote or arbitrary label
   selection is performed automatically.
7. For duplicate normalized texts whose mapped labels agree, retain one
   deterministic source record. Record duplicate-group counts, removed rows,
   conflict-group counts, exclusions/adjudications, and the selected conflict
   policy in the audit.
8. Save the cleaned dataset and preprocessing audit. The script clears its
   previous Stage 1 output files at the beginning of a new run to prevent old
   outputs from being mistaken for the current result.

The benchmark-specific three-word minimum and placeholder removal are **not**
reused for the custom contextual dataset. Short contextual target posts are
valid research examples and must be preserved.

### 1.2 Three-class label mapping

The canonical mapping is shared through `experiment_utils.py`:

| Canonical label | Meaning | Source condition |
|---|---|---|
| `N` | Neither | `HS=0` and `Abusive=0` |
| `AL` | Abusive Language | `HS=0` and `Abusive=1` |
| `HS` | Hate Speech | `HS=1`, regardless of `Abusive` |

The priority rule is important because a source record can have both binary
indicators set to one. Such a record is mapped to `HS` consistently; the
pipeline does not create a fourth class.

### 1.3 Train, validation, and test splitting

**Decision: stratified 70% / 15% / 15% split**, using the canonical three-class
`label` as the stratification variable and `random_state=42`.

The split is created only after text validation, normalized-text deduplication,
and conflict handling have completed. This order prevents the same normalized
text from appearing in more than one partition and ensures that split
stratification reflects the final eligible dataset.

The cleaned row count, split sizes, and class distributions are **data-run
results** recorded by Stage 1 in `results/tables/ib_preprocessing_audit.json` and
`results/tables/ib_preprocessing_audit.csv`. The figures below were taken from
that audit and are internally consistent: the class counts sum to each partition
size, and the partition sizes sum to the cleaned total. Stage 1 also reports zero
overlapping normalized texts between every pair of partitions.

| Partition | Split ratio | Rows | `N` | `AL` | `HS` |
|---|---:|---:|---:|---:|---:|
| Training | 70% | 8,913 | 3,986 (44.72%) | 1,164 (13.06%) | 3,763 (42.22%) |
| Validation | 15% | 1,910 | 854 (44.71%) | 249 (13.04%) | 807 (42.25%) |
| Test | 15% | 1,910 | 854 (44.71%) | 250 (13.09%) | 806 (42.20%) |
| **Total** | **100%** | **12,733** | **5,694 (44.72%)** | **1,663 (13.06%)** | **5,376 (42.22%)** |

Reconciliation of the cleaning steps (Stage 1 audit):

| Step | Rows |
|---|---:|
| Raw rows loaded | 13,169 |
| Malformed CSV lines skipped | 0 |
| Excluded for invalid text (36 empty after cleaning; 273 under three words) | −309 |
| Valid rows before deduplication | 12,860 |
| Removed as same-label duplicates (102 groups) | −117 |
| Excluded by the `exclude` conflict policy (5 groups, 10 rows) | −10 |
| **Cleaned rows after deduplication and conflict resolution** | **12,733** |

Values quoted in earlier versions of this document (12,746 cleaned rows;
8,922 / 1,912 / 1,912 split; class weights 0.7456, 2.5528, 0.7893) came from a
previous preprocessing run and must not be cited.

#### 1.3.1 Rationale for the 70/15/15 ratio

The ratio is a trade-off between training-set size and the precision of the
estimate on the test partition, not a proven optimum. The standard error of a
proportion is \(\mathrm{SE}=\sqrt{p(1-p)/n}\). The table compares three common
ratios on the 12,733 cleaned rows, assuming accuracy 0.90, `AL` recall 0.80,
and the observed `AL` share of 13.06%:

| Ratio | Train | Test | `AL` in test | SE accuracy (points) | SE `AL` recall (points) |
|---|---:|---:|---:|---:|---:|
| 60/20/20 | 7,640 | 2,547 | 333 | 0.59 | 2.19 |
| **70/15/15** | **8,913** | **1,910** | **250** | **0.69** | **2.53** |
| 80/10/10 | 10,186 | 1,273 | 166 | 0.84 | 3.10 |

Relative to 70/15/15, 80/10/10 adds about 14% training data but increases the
test standard error by about 22% and leaves only about 166 `AL` posts for the
minority-class estimate. 60/20/20 lowers the test standard error by about 13%
but removes about 14% of the training data (1,273 posts). Preferring more
training data over the extra test precision is an assumption that was not tested
separately. Only 70/15/15 is used in the pipeline.

The validation partition exists so that the learning rate, the checkpoint, and
early stopping can be selected without using the test partition. Selecting on the
test partition would make the reported test score optimistic. At 1,910 posts the
validation partition has a standard error of about 0.69 points in accuracy under
the same assumptions.

### 1.4 Class weights: computed from the training partition only

**Decision: use balanced class weights for the benchmark-based Transformer
training only.** The weights are calculated after the split, using the actual
class counts in `ib_train.csv`. They must not be calculated from the complete
cleaned dataset, validation partition, or test partition.

For class \(c\), the balanced weight is:

\[
 w_c = \frac{n_{\mathrm{train}}}{K\,n_c}
\]

where:

- \(n_{\mathrm{train}}\) is the number of examples in the training partition;
- \(K=3\) is the number of canonical classes;
- \(n_c\) is the number of training examples belonging to class \(c\).

For the current training partition (8,913 rows), the computed weights are:

| Class | Training count | Weight |
|---|---:|---:|
| `N` | 3,986 | 0.7454 |
| `AL` | 1,164 | 2.5524 |
| `HS` | 3,763 | 0.7895 |

These values are saved in `results/tables/ib_class_weights.csv` and agree with
the formula above (for example, `AL`: 8,913 / (3 × 1,164) = 2.5524).

This is the same inverse-frequency weighting principle as scikit-learn's
`class_weight="balanced"`. The shared function
`experiment_utils.compute_class_weights(train_labels)` returns weights keyed by
canonical integer label ID (`N=0`, `AL=1`, `HS=2`), which are ordered by ID when
converted to a PyTorch loss-weight tensor.

```python
from experiment_utils import LABEL2ID, compute_class_weights
import torch

weights_by_id = compute_class_weights(train_df["label"])
weight_tensor = torch.tensor(
    [weights_by_id[LABEL2ID[label]] for label in ("N", "AL", "HS")],
    dtype=torch.float32,
)
loss_fn = torch.nn.CrossEntropyLoss(weight=weight_tensor)
```

The current Stage 1 script saves the resulting values to
`results/tables/ib_class_weights.csv` and includes the weight values in the
preprocessing audit. The class counts used in the formula must be taken from
`ib_train.csv` and checked against the audit. A weight table is valid only for
the exact training split from which it was calculated; if Stage 1 is rerun and
the split changes, the weights must be regenerated too.

The same benchmark-derived weighting policy is used for both backbones to
preserve parity. It applies to Experiment I and the benchmark-based ITFT
initialization stages. Do not carry these benchmark weights over to contextual
fine-tuning.

### 1.5 Benchmark maximum sequence length (Stage 1b)

After Stage 1 has produced the fixed benchmark training partition, Stage 1b
runs `scripts/1b_select_benchmark_max_length.py`. It uses **only**
`ib_train.csv` and both backbone tokenizers to select one shared
`max_length_benchmark`; it does not change the dataset split or class weights.
The selection is based on the larger of the two tokenizers' 95th-percentile
training-text lengths, rounded up to a multiple of eight and capped at 512.
The chosen value and supporting statistics are saved in
`results/config/sequence_length.json` and must be frozen before benchmark
learning-rate preflight, Experiment I, and ITFT.

For target-only benchmark inputs, truncation is explicitly right-sided. For
context-first inputs, the target is placed last, so contextual truncation is
explicitly left-sided to preserve the target. These are input-construction
policies; they do not alter the split or weight calculation.

---

## 2. Custom contextual dataset (Experiments II and III)

### 2.1 Label normalization and preservation of annotation fields

**Input:** `data/raw/contextual_dataset.csv`  
**Script:** `scripts/2_preprocess_contextual.py`

The raw annotated export uses the full label spellings `neither`,
`abusive_language`, and `hate_speech`. Stage 2 maps these values to the
canonical codes `N`, `AL`, and `HS` through
`experiment_utils.RAW_CTX_LABEL_MAP`. The original spelling is preserved in
`label_raw` for traceability; it is not a model input.

Stage 2 checks the required raw schema, validates that the mapped `label`
agrees with the binary `HS` and `AL` indicators under the same HS-priority
rule, verifies unique `post_id` and `target_text` values, and checks the
expected class, context-needed, and parent-availability distributions. It does
not apply the benchmark's placeholder-removal and three-word filtering rules,
and it does not silently deduplicate target posts. A duplicate target post or
an inconsistent annotation is a data-integrity issue that must be investigated.

The modelling allowlist in `experiment_utils.py` separates model inputs from
analysis-only annotations. `post_id` is retained for prediction alignment but
is not tokenized. Fields such as `context_needed`, `HS`, `AL`, `target_group`,
`target_group_attribute`, `context_type`, `collection_method`, `keyword_used`,
and `post_date` are preserved for dataset reporting and error analysis, but
must not be passed to the model as input features.

### 2.2 Verified collection distribution

The supplied contextual dataset contains 1,800 posts, with 600 examples in
each canonical class. The distribution by number of available parent messages
is also balanced by class:

| Available parent messages | `N` | `AL` | `HS` | Total |
|---:|---:|---:|---:|---:|
| 1 | 50 | 50 | 50 | 150 |
| 2 | 100 | 100 | 100 | 300 |
| 3 | 150 | 150 | 150 | 450 |
| 4 | 300 | 300 | 300 | 900 |
| **Total** | **600** | **600** | **600** | **1,800** |

The full-dataset `context_needed` by class distribution is:

| `context_needed` | `N` | `AL` | `HS` | Total |
|---|---:|---:|---:|---:|
| 0 — context not needed | 420 | 300 | 235 | 955 |
| 1 — context needed | 180 | 300 | 365 | 845 |
| **Total** | **600** | **600** | **600** | **1,800** |

These are dataset-collection counts, not model-performance results. The
parent-count-by-class values in the first table were calculated directly from
the supplied `contextual_dataset.csv`. Stage 2 hard-checks the overall
three-class totals, the full-dataset `context_needed` × `label` cells, and the
parent-availability totals; it does not currently declare every
parent-count × class cell as a separate hard-coded assertion. Stage 2 also
produces `ctx_distribution.csv` for Chapter III reporting.

### 2.3 No class weighting for the contextual dataset

**Decision: do not apply class weighting to contextual fine-tuning.** The full
1,800-post contextual dataset has 600 examples per class. The parent-count
strata are also balanced: 150, 300, 450, and 900 examples respectively, with
one-third of each stratum assigned to each class. The 900-post matched subset
and 900-post shallow pool each contain 300 examples per class.

Consequently, the contextual training/evaluation design does not use the
benchmark-derived weights. Experiment II's contextual fine-tuning, Experiment
III's direct contextual fine-tuning, and Stage 17 final-model training use
unweighted cross-entropy. The benchmark weights remain limited to benchmark
training and ITFT on the benchmark training partition.

### 2.4 Matched subset, shallow pool, and fold assignment

Stage 2 partitions the 1,800 contextual posts by `context_size_available`:

| Processed dataset | Selection rule | Rows | Intended use |
|---|---|---:|---|
| `ctx_cleaned.csv` | All validated contextual posts | 1,800 | Stage 17 final-model training |
| `ctx_matched_subset.csv` | Exactly four available parents | 900 | Main C0–C4 ablation and matched Experiment III cross-validation |
| `ctx_shallow_pool.csv` | One, two, or three available parents | 900 | Secondary availability-aware generalization analysis |

The matched subset is the main complete-case population for comparing C0,
C1, C2, C3, and C4. These are five different input views of the **same 900
posts**, not five independently sampled datasets. The shallow pool is disjoint
from the matched subset and is used for the secondary Stage 13 analysis; it
does not select `C_best`.

A single five-fold assignment is generated once on the 900-row matched subset,
before any C0–C4 text views are constructed. Stage 2 uses
`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` with the joint
stratification key:

```python
strat_key = matched["context_needed"].astype(str) + "_" + matched["label"]
```

The six strata (`context_needed` × `label`) in the matched subset, with the
number of examples per fold in parentheses (each cell divides evenly by five):

| `context_needed` | `N` | `AL` | `HS` | Total |
|---|---:|---:|---:|---:|
| 0 — context not needed | 210 (42) | 150 (30) | 110 (22) | 470 (94) |
| 1 — context needed | 90 (18) | 150 (30) | 190 (38) | 430 (86) |
| **Total** | **300 (60)** | **300 (60)** | **300 (60)** | **900 (180)** |

The smallest stratum is `N` with `context_needed = 1` (90 posts, 18 per fold).
The disjoint shallow pool (one to three parents) has the following composition:

| `context_needed` | `N` | `AL` | `HS` | Total |
|---|---:|---:|---:|---:|
| 0 | 210 | 150 | 125 | 485 |
| 1 | 90 | 150 | 175 | 415 |
| **Total** | **300** | **300** | **300** | **900** |

The resulting `fold` column contains five folds of 180 examples each. In every
iteration one fold (180 examples) is held out and the other four (720 examples)
are used for training; each example is held out exactly once. Each fold is
trained with two random seeds. The rationale for five folds, two seeds, and
cross-validation instead of a single split is given in Subsections 3.4.4 and
3.9.1. This
same stored fold assignment is reused for all C0–C4 configurations, both
backbones, Experiment II (ITFT plus contextual fine-tuning), and Experiment
III (direct contextual fine-tuning). Do not independently reshuffle or
reassign folds for a context size, backbone, or experiment. Doing so would
break the paired comparisons and could create leakage between the truncated
views of the same underlying post.

### 2.5 Contextual input construction and full-data training

For a contextual input with \(C=n\), parent messages are concatenated from
`parent_n` down to `parent_1` (oldest to newest), followed by `target_text`.
C0 contains only `target_text`. The shared input builder in
`experiment_utils.build_context_text()` defines this order. Each backbone uses
its own tokenizer separator token; separator tokens must not be hard-coded
across backbones.

Context-first inputs place the target at the end. Therefore, contextual
truncation is explicitly left-sided so that the oldest parent content is
removed first and the target is retained. Stage 8 selects one shared
`max_length_context` from the C4 training-side token distributions of both
backbones; this value is used consistently for the contextual experiments and
final context-aware model.

The 900-post matched subset is used for the main cross-validation and
selection analyses. Stage 17 subsequently trains two distinct deployment
models on all 1,800 contextual posts: one target-only C0 model and one
context-aware model using the concrete `C_best` chosen at Stage 11. The full
1,800-post dataset is not a held-out test set; Stage 17 training metrics must
not be presented as independent generalization results. Use the held-out
cross-validation/OOF evaluation and the separately defined system evaluation
for performance claims.

---

## 3. Reporting and reproducibility requirements

For Chapter III, report:

1. The raw benchmark row count, malformed-line count, invalid-text count,
   invalid-label count, duplicate-normalized-text counts, conflict-handling
   policy and outcome, and final eligible benchmark row count from the Stage 1
   audit.
2. The final benchmark class distribution and the actual train/validation/test
   sizes and per-class counts from the same successful Stage 1 run.
3. The training-partition class counts and the corresponding numeric weights
   from `ib_class_weights.csv`; do not quote the obsolete values from an older
   preprocessing run.
4. The contextual class and parent-availability distributions, matched/shallow
   partition sizes, and shared fold-assignment procedure.
5. The rule that contextual losses are unweighted while benchmark/ITFT losses
   use training-only benchmark class weights.

Keep the complete preprocessing audit files with the run outputs. If the raw
benchmark file, cleaning logic, conflict policy, or split configuration
changes, rerun Stage 1 and replace the reported counts and weights with the
new audit results before executing downstream benchmark training.
