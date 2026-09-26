"""
5_evaluate_e1.py

Aggregate Experiment I results: seed mean/std, SVM-vs-transformer paired bootstrap significance test

TODO: implement once the relevant preceding stage is complete.
"""

import pandas as pd
df = pd.read_csv("data/raw/contextual_dataset.csv")
print(df[["label", "HS", "AL"]].dtypes)
print()
print("label unique:", df["label"].unique()[:10])
print("HS unique:", df["HS"].unique())
print("AL unique:", df["AL"].unique())
print()
print(df[["label", "HS", "AL"]].head(5).to_string())