import os
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

# 1. Load all CSVs
file_paths = {
    "lr": "./results/zero_shot_prediction_agreement_lr.csv",
    "svm": "./results/zero_shot_prediction_agreement_svm.csv",
    "rf": "./results/zero_shot_prediction_agreement_rf.csv",
    "xgb": "./results/zero_shot_prediction_agreement_xgb.csv",
    "knn": "./results/zero_shot_prediction_agreement_knn.csv"
}

dfs = []
for model, path in file_paths.items():
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    df = pd.read_csv(path)
    dfs.append(df)
df_all = pd.concat(dfs, ignore_index=True)

# 2. Compute each model's consensus vote per protein
agg = df_all.groupby(["protein_id", "model_type"]).agg(
    votes=pd.NamedAgg(column="pred_label", aggfunc=lambda x: x.value_counts().idxmax()),
    true_label=("true_label", "first"),  # assume consistent
    total_votes=("pred_label", "count"),
    votes_for_winner=("pred_label", lambda x: x.value_counts().max())
).reset_index()

agg["agreement_ratio"] = agg["votes_for_winner"] / agg["total_votes"]

# 3. Evaluate global system performance
thresholds = [0.75, 0.8, 0.9, 0.95]
min_models = [1, 2, 3, 4, 5]  # Minimum number of models agreeing for consensus
rows = []

for thresh in thresholds:
    sub = agg[agg["agreement_ratio"] >= thresh]
    for k in min_models:
        # proteins with consensus from at least k models
        grp = sub.groupby("protein_id")
        valid = grp.filter(lambda g: len(g) >= k)
        if valid.empty:
            rows.append((k, thresh, 0.0, 0.0, 0.0))
            continue
        y_true = valid["true_label"]
        y_pred = valid["votes"]  # majority consensus per model
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="micro", zero_division=0)
        rows.append((k, thresh, p, r, f1))

results = pd.DataFrame(rows, columns=["min_models", "threshold", "precision", "recall", "f1"])

# Pivot into table form
pivot = results.pivot(index="min_models", columns="threshold", values="f1")
print(pivot)

# Export to LaTeX style table
print(pivot.to_latex(float_format="%.3f"))
