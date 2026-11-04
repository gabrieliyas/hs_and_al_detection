import re
import unicodedata
from pathlib import Path

import pandas as pd

raw_path = Path("data/raw/re_dataset.csv")
output_path = Path(
    "data/processed/ib_duplicate_label_conflicts_full_source.csv"
)

df = pd.read_csv(
    raw_path,
    encoding="latin-1",
    engine="python",
    on_bad_lines="error",
)

# Assign a stable record number for tracing source records.
df.insert(0, "source_record", range(1, len(df) + 1))

# Reproduce Stage 1 text normalization.
def clean_text(value):
    if not isinstance(value, str):
        return ""

    text = unicodedata.normalize("NFKC", value)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\bUSER\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\bURL\b", " ", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()

df["text_raw"] = df["Tweet"]
df["clean_text"] = df["text_raw"].apply(clean_text)

# Reproduce the original three-class mapping.
hs = pd.to_numeric(df["HS"], errors="coerce")
abusive = pd.to_numeric(df["Abusive"], errors="coerce")

valid_labels = (
    hs.notna()
    & hs.isin([0, 1])
    & abusive.notna()
    & abusive.isin([0, 1])
)

df["_mapped_label"] = pd.NA

df.loc[valid_labels, "_mapped_label"] = [
    "HS" if int(h) == 1 else
    "AL" if int(a) == 1 else
    "N"
    for h, a in zip(hs[valid_labels], abusive[valid_labels])
]

# Apply the same text validity criteria as Stage 1.
word_count = df["clean_text"].str.split().str.len()

valid_text = (
    df["text_raw"].notna()
    & df["text_raw"].astype("string").str.strip().ne("").fillna(False)
    & df["clean_text"].ne("")
    & word_count.ge(3)
)

eligible = df.loc[valid_labels & valid_text].copy()

# Identify normalized texts associated with multiple labels.
label_counts = eligible.groupby("clean_text")["_mapped_label"].nunique()
conflict_keys = label_counts[label_counts > 1].index

conflicts = eligible[
    eligible["clean_text"].isin(conflict_keys)
].sort_values(["clean_text", "source_record"])

# Preserve ALL original annotation columns.
output_path.parent.mkdir(parents=True, exist_ok=True)
conflicts.to_csv(output_path, index=False)

# Generate a compact comparison summary.
summary = (
    conflicts.groupby("clean_text", sort=False)
    .agg(
        duplicate_rows=("_mapped_label", "size"),
        source_records=(
            "source_record",
            lambda x: ", ".join(map(str, x))
        ),
        raw_text_variants=(
            "text_raw",
            lambda x: " || ".join(dict.fromkeys(x.astype(str)))
        ),
        mapped_labels=(
            "_mapped_label",
            lambda x: " | ".join(sorted(set(x.astype(str))))
        ),
    )
    .reset_index()
)

summary.to_csv(
    output_path.with_name(
        "ib_duplicate_label_conflicts_full_summary.csv"
    ),
    index=False,
)

print(f"Conflicting records exported: {len(conflicts)}")
print(f"Conflicting groups: {len(conflict_keys)}")
print(f"Detailed report: {output_path}")