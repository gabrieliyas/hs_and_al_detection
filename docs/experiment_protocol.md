# Experiment Execution Protocol

Companion to `dataset_split_and_weighting.md`. Records execution-level
decisions for Experiments I, II, and III: seed counts, significance
testing methodology, compute budgeting, and the C0-C4 ablation analysis
design. Reference for implementation and for the methodology chapter.

---

## 1. Experiment I

- **Seeds:** 5 random seeds for IndoBERTweet-Target (varying classification-
  head init and data shuffling), on the single fixed 70/15/15 split. Report
  mean +/- std via `experiment_utils.summarize_seed_runs`. 2 seeds was
  considered and rejected -- a standard deviation from 2 points carries
  almost no information; 3 is the practical floor, 5 is affordable here
  since Experiment I has no repeated ITFT stage or fold grid, and is
  recommended.
- **SVM+TF-IDF:** single deterministic run. Set `random_state` explicitly
  on the SVM (and on any stochastic step in the TF-IDF pipeline) --
  determinism is a property of the fixed configuration, not an inherent
  guarantee of "being an SVM."
- **SVM vs. transformer significance test:** the transformer has 5 seed-
  level prediction sets on the test set, not one, so a single arbitrary
  seed should not be fed into the paired bootstrap. Use
  `experiment_utils.ensemble_seed_probabilities` to average softmax
  probabilities across the 5 seeds and take the argmax as the
  representative prediction set, then run `paired_bootstrap_ci` between
  that and SVM's predictions. The per-seed mean +/- std (training
  stability) and the ensembled-prediction significance test (head-to-head
  comparison) are two separate statistics answering two separate
  questions -- both are reported, using different aggregations by design.

---

## 2. Experiments II and III

- **Timing pilot:** before committing to the full grid, run exactly one
  (condition, fold) pair -- e.g. C4, fold 0 -- and time two things
  separately: (a) the one-time ITFT stage on the full Ibrohim & Budi
  training set, and (b) a single second-stage fine-tune on that fold's
  ~720-example train portion. Total budget estimate = ITFT time (paid
  once) + (single second-stage run time x total planned runs across
  conditions x folds x seeds).
- **`max_length` decision happens inside this same pilot**, not
  separately: tokenize the pilot fold with the actual IndoBERTweet
  tokenizer, inspect the real token-length distribution for the
  concatenated context + target, and choose `max_length` to cover it
  (e.g. 95th percentile) without excessive padding waste. This also
  directly affects the timing estimate above, since longer `max_length`
  means slower runs.
- **Colab session limits:** check the total timing estimate against
  Colab's session runtime limits. If the full sweep exceeds one session,
  plan for checkpoint/resume logic across sessions rather than
  discovering this mid-sweep.
- **Fold assignment sharing:** confirmed -- the same `fold` column (from
  `ctx_matched_subset.csv`) is reused identically across Experiment II
  (ITFT+C) and Experiment III (C-only), and across every C0-C4 context
  configuration. See `dataset_split_and_weighting.md` Section 2.3-2.4 for
  the implementation.
- **C0-C4 construction order:** fold assignment happens first, once, on
  the 900 base examples. The C0-C4 truncated views are generated
  afterward, at data-loading/tokenization time, from each example's
  already-assigned fold -- never the reverse. This is what prevents the
  same underlying post from landing in train for one context size and
  test for another.

---

## 3. C0-C4 ablation: main analysis vs. secondary analysis

Two candidate designs were considered:

- **Nested availability-aware cohorts:** compare C0 vs. Cn using the
  maximal subset with >= n parents available (C0-vs-C1 on 1,800; C0-vs-C2
  on 1,650; C0-vs-C3 on 1,350; C0-vs-C4 on 900).
- **Complete-case (matched-subset) analysis:** compare all of C0 through
  C4 on the identical 900 examples with exactly 4 parents available
  (the existing matched subset, shared fold assignments).

**Decision: complete-case analysis is the MAIN analysis.** The nested
design changes which population is evaluated at every comparison point,
which confounds "effect of context depth" with "which examples happen to
have that much depth available" -- e.g. the >=3 subset systematically
excludes shallow-thread posts that the >=2 subset includes. This is the
same confound the shared-fold-assignment principle (Section 2, above; and
`dataset_split_and_weighting.md` Section 2.3) already exists to prevent --
the nested design reintroduces it at the cohort-selection stage rather
than the fold stage. A genuine dose-response claim ("does performance
plateau or decline past a certain depth") requires holding the underlying
examples constant across all compared levels, which only the complete-
case design does.

**The nested availability-aware analysis is a SECONDARY analysis**, kept
for a different, complementary purpose: it reflects realistic deployment
conditions (most real conversations will not have 4 parents available,
which matters for the browser-extension demo's generalizability claim).
When reporting it, each depth-specific comparison (C0-vs-C1, C0-vs-C2,
etc.) must be presented as an independent, standalone finding on its own
population -- NOT plotted as points on one continuous curve alongside
each other, since doing so would silently reintroduce the population
confound the complete-case analysis was specifically designed to avoid.

**Practical consequence:** the headline F1-vs-context-window-size plot
(contribution: optimal context size / diminishing returns) is built from
the complete-case analysis only. The nested analysis, if included,
appears as a separate table/discussion in Chapter IV framed around
real-world context availability, not as an alternative version of the
same curve.
