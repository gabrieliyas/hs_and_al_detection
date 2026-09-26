"""
7_timing_pilot.py

Stage 7: the timing / max_length pilot required by
docs/experiment_protocol.md Section 2, before committing to the full
Experiment II grid (Stage 8: C0-C4 x 5 folds x 2 seeds = 50 runs).

Per that section, this pilot does three things together, on exactly one
(condition, fold) pair -- C4, fold 0, the example the doc itself gives:
  1. Tokenizes the pilot fold's C4 text (parent_4..parent_1 + target, via
     experiment_utils.build_context_text) with the REAL IndoBERTweet
     tokenizer, with NO truncation, and inspects the actual token-length
     distribution -- this is what decides max_length (95th percentile,
     rounded up to a multiple of 8), not a guess made in advance.
  2. Times a single second-stage fine-tune on that fold's ~720-example
     train portion (folds != 0), STARTING FROM THE ITFT CHECKPOINT (Stage
     6's output) -- this mirrors exactly what each of the 50 Experiment II
     runs will do, which a pilot starting from the raw pretrained model
     would not.
  3. Combines that per-run time with Stage 6's own logged ITFT time
     (results/tables/itft_metrics.csv) into a total Experiment II budget
     estimate: ITFT time (paid once) + single_run_time * 50 -- for you to
     compare against your Colab tier's actual session-length limit (this
     script does not hard-code a specific limit, since it varies by tier
     and changes over time).

This pilot run's own checkpoint is NOT saved -- it exists only to produce
the timing number and the max_length recommendation; the actual 50
Experiment II runs happen in Stage 8.

Usage (inside a cloned repo, on a GPU runtime, AFTER Stage 6 has produced
models/E2_itft_context/):
    python scripts/7_timing_pilot.py

Input:
    data/processed/ctx_matched_subset.csv   (Stage 2 output; has the `fold` column)
    models/E2_itft_context/                 (Stage 6 output -- the ITFT checkpoint)
    results/tables/itft_metrics.csv         (Stage 6 output -- for the ITFT timing figure)

Outputs:
    results/tables/pilot_token_length_stats.csv   -- min/mean/median/p90/p95/p99/max
    results/tables/pilot_summary.csv              -- chosen max_length, timings, budget estimate
"""

from __future__ import annotations

import math
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiment_utils import (  # noqa: E402
    ID2LABEL,
    LABEL2ID,
    build_context_text,
    compute_class_weights,
)

MATCHED_SUBSET_PATH = Path("data/processed/ctx_matched_subset.csv")
ITFT_CHECKPOINT_DIR = Path("models/E2_itft_context")
ITFT_METRICS_PATH = Path("results/tables/itft_metrics.csv")
TABLES_DIR = Path("results/tables")

# The doc's own example pilot pair.
PILOT_CONDITION = 4  # C4
PILOT_FOLD = 0
SEED = 42

# From docs/decisions.md: "Experiment II-A: C0-C4 ablation, 5-fold CV, 2 seeds"
N_CONDITIONS_E2 = 5   # C0..C4
N_FOLDS_E2 = 5
N_SEEDS_E2 = 2
TOTAL_PLANNED_RUNS_E2 = N_CONDITIONS_E2 * N_FOLDS_E2 * N_SEEDS_E2  # 50

TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
LEARNING_RATE = 2e-5
NUM_EPOCHS = 4


class TokenizedTextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings: dict, labels: np.ndarray):
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(int(self.labels[idx]))
        return item


class WeightedLossTrainer(Trainer):
    def __init__(self, *args, class_weights: torch.Tensor, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred) -> dict:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }


def round_up_to_multiple(n: int, multiple: int = 8) -> int:
    return int(math.ceil(n / multiple) * multiple)


def main() -> None:
    if not MATCHED_SUBSET_PATH.exists():
        raise FileNotFoundError(
            f"{MATCHED_SUBSET_PATH} not found. Run scripts/2_preprocess_contextual.py first."
        )
    if not (ITFT_CHECKPOINT_DIR / "config.json").exists():
        raise FileNotFoundError(
            f"{ITFT_CHECKPOINT_DIR} not found or incomplete. Run scripts/6_itft.py first "
            "and copy models/E2_itft_context/ onto this machine -- this pilot must start "
            "from the ITFT checkpoint to give a realistic timing estimate for Stage 8 "
            "(see module docstring point 2)."
        )
    if not torch.cuda.is_available():
        print("WARNING: no GPU detected -- the timing measurement below will not be "
              "representative of the actual Colab GPU run time.")

    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(MATCHED_SUBSET_PATH)
    print(f"[1/6] Loaded matched subset ({len(df)} rows).")

    pilot_train = df[df["fold"] != PILOT_FOLD].copy()
    pilot_held_out = df[df["fold"] == PILOT_FOLD].copy()
    print(f"       Pilot pair: C{PILOT_CONDITION}, fold {PILOT_FOLD} -- "
          f"train {len(pilot_train)} rows, held-out {len(pilot_held_out)} rows.")

    tokenizer = AutoTokenizer.from_pretrained(str(ITFT_CHECKPOINT_DIR))

    # --- Step 1: token-length distribution (no truncation) -> choose max_length ---
    texts_for_length_check = pd.concat([pilot_train, pilot_held_out]).apply(
        lambda r: build_context_text(r, PILOT_CONDITION, sep=f" {tokenizer.sep_token} "), axis=1
    )
    lengths = texts_for_length_check.apply(
        lambda t: len(tokenizer(t, truncation=False)["input_ids"])
    )
    stats = {
        "min": int(lengths.min()),
        "mean": float(lengths.mean()),
        "median": float(lengths.median()),
        "p90": float(np.percentile(lengths, 90)),
        "p95": float(np.percentile(lengths, 95)),
        "p99": float(np.percentile(lengths, 99)),
        "max": int(lengths.max()),
    }
    chosen_max_length = round_up_to_multiple(int(math.ceil(stats["p95"])), multiple=8)
    print(f"\n[2/6] Token length distribution for C{PILOT_CONDITION} "
          f"(parent_4..parent_1 + target, no truncation):")
    for k, v in stats.items():
        print(f"       {k}: {v:.1f}" if isinstance(v, float) else f"       {k}: {v}")
    print(f"       -> chosen max_length = {chosen_max_length} "
          f"(95th percentile {stats['p95']:.1f}, rounded up to a multiple of 8)")

    pd.DataFrame([stats]).to_csv(TABLES_DIR / "pilot_token_length_stats.csv", index=False)

    # --- Step 2: tokenize the pilot pair at the chosen max_length ---
    def tokenize_condition(sub_df: pd.DataFrame) -> dict:
        texts = sub_df.apply(
            lambda r: build_context_text(r, PILOT_CONDITION, sep=f" {tokenizer.sep_token} "),
            axis=1,
        ).tolist()
        enc = tokenizer(
            texts, truncation=True, max_length=chosen_max_length, padding="max_length",
            return_tensors="pt",
        )
        return dict(enc)

    train_labels_id = pilot_train["label"].map(LABEL2ID).to_numpy()
    held_out_labels_id = pilot_held_out["label"].map(LABEL2ID).to_numpy()

    train_enc = tokenize_condition(pilot_train)
    held_out_enc = tokenize_condition(pilot_held_out)
    train_ds = TokenizedTextDataset(train_enc, train_labels_id)
    held_out_ds = TokenizedTextDataset(held_out_enc, held_out_labels_id)
    print(f"[3/6] Tokenized pilot train/held-out at max_length={chosen_max_length}.")

    # The custom contextual dataset is exactly balanced (docs/decisions.md)
    # -- class weights recomputed here anyway, from THIS pilot fold's train
    # portion, rather than assumed 1.0, since a single CV fold's ~720 rows
    # need not be perfectly balanced even though the full 900-row subset is.
    class_weights_by_id = compute_class_weights(pilot_train["label"])
    class_weights_tensor = torch.tensor(
        [class_weights_by_id[i] for i in range(len(LABEL2ID))], dtype=torch.float
    )

    # --- Step 3: time a single second-stage fine-tune, starting from ITFT ---
    print(f"\n[4/6] Timing a single second-stage fine-tune "
          f"(C{PILOT_CONDITION}, fold {PILOT_FOLD}, seed {SEED}) "
          f"starting from the ITFT checkpoint...")
    set_seed(SEED)
    t0 = time.time()

    model = AutoModelForSequenceClassification.from_pretrained(
        str(ITFT_CHECKPOINT_DIR), num_labels=len(LABEL2ID), id2label=ID2LABEL, label2id=LABEL2ID,
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        args = TrainingArguments(
            output_dir=tmp_dir,
            num_train_epochs=NUM_EPOCHS,
            per_device_train_batch_size=TRAIN_BATCH_SIZE,
            per_device_eval_batch_size=EVAL_BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="no",  # pilot run -- no checkpoint kept, see module docstring
            logging_steps=50,
            seed=SEED,
            report_to="none",
            fp16=torch.cuda.is_available(),
        )
        trainer = WeightedLossTrainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=held_out_ds,
            compute_metrics=compute_metrics,
            class_weights=class_weights_tensor,
        )
        trainer.train()
        held_out_metrics = trainer.evaluate(held_out_ds)

    single_run_minutes = (time.time() - t0) / 60
    print(f"[5/6] Single second-stage run: {single_run_minutes:.2f} min "
          f"(held-out macro_f1={held_out_metrics['eval_macro_f1']:.4f}, "
          "not a CV result -- this fold's held-out set was also used for "
          "epoch-level model selection here, unlike the real Stage 8 runs).")

    # --- Step 4: combine with Stage 6's logged ITFT time into a total budget estimate ---
    if ITFT_METRICS_PATH.exists():
        itft_minutes = pd.read_csv(ITFT_METRICS_PATH)["train_minutes"].iloc[0]
    else:
        itft_minutes = None
        print(f"\nWARNING: {ITFT_METRICS_PATH} not found -- total budget estimate below "
              "will omit the (paid-once) ITFT time.")

    total_second_stage_minutes = single_run_minutes * TOTAL_PLANNED_RUNS_E2
    total_minutes = total_second_stage_minutes + (itft_minutes or 0)
    print(f"\n[6/6] Experiment II budget estimate:")
    print(f"       ITFT (paid once): {itft_minutes if itft_minutes is not None else 'unknown'} min")
    print(f"       Single second-stage run: {single_run_minutes:.2f} min")
    print(f"       Planned runs (C0-C4 x {N_FOLDS_E2} folds x {N_SEEDS_E2} seeds): "
          f"{TOTAL_PLANNED_RUNS_E2}")
    print(f"       Total second-stage time: {total_second_stage_minutes:.1f} min "
          f"({total_second_stage_minutes/60:.1f} h)")
    print(f"       TOTAL estimate: {total_minutes:.1f} min ({total_minutes/60:.1f} h)")
    print(
        "\n       Compare this against your Colab tier's actual session-length "
        "limit (this script does not assume a specific number, since it varies "
        "by tier and changes over time). If the total exceeds one session, "
        "Stage 8 needs checkpoint/resume logic across sessions rather than "
        "discovering that mid-sweep (docs/experiment_protocol.md Section 2)."
    )

    pd.DataFrame([{
        "pilot_condition": f"C{PILOT_CONDITION}",
        "pilot_fold": PILOT_FOLD,
        "chosen_max_length": chosen_max_length,
        "itft_minutes": itft_minutes,
        "single_run_minutes": round(single_run_minutes, 2),
        "total_planned_runs_e2": TOTAL_PLANNED_RUNS_E2,
        "total_second_stage_minutes": round(total_second_stage_minutes, 1),
        "total_estimate_minutes": round(total_minutes, 1),
        "total_estimate_hours": round(total_minutes / 60, 2),
    }]).to_csv(TABLES_DIR / "pilot_summary.csv", index=False)

    print(f"\nSaved results/tables/pilot_token_length_stats.csv and "
          f"results/tables/pilot_summary.csv.")
    print(
        "\nNext: send me both CSVs (or the console output above) -- Stage 8 will "
        f"be written using max_length={chosen_max_length} and the timing estimate "
        "to decide checkpoint/resume granularity across the 50 runs."
    )


if __name__ == "__main__":
    main()
