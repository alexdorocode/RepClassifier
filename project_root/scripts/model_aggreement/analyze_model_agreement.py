import os
import pandas as pd

# Load all CSVs from model results
file_paths = {
    "lr": "./results/zero_shot_prediction_agreement_lr.csv",
    "svm": "./results/zero_shot_prediction_agreement_svm.csv",
    "rf": "./results/zero_shot_prediction_agreement_rf.csv",
    "xgb": "./results/zero_shot_prediction_agreement_xgb.csv",
    "knn": "./results/zero_shot_prediction_agreement_knn.csv"
}

dfs = []
for model, path in file_paths.items():
    if os.path.exists(path):
        df = pd.read_csv(path)
        df["model_type"] = model
        dfs.append(df)
    else:
        print(f"Missing file: {path}")

# Merge everything
df_all = pd.concat(dfs, ignore_index=True)

# Compute agreement per (protein, model_type)
agg = (
    df_all.groupby(["protein_id", "model_type", "true_label"])
    .agg(correct_count=("is_correct", "sum"), total=("is_correct", "count"))
    .reset_index()
)
agg["agreement_ratio"] = agg["correct_count"] / agg["total"]

# Analyze thresholds
thresholds = [0.75, 0.8, 0.85, 0.9, 0.95]
model_agreements_required = [1, 2, 3, 4, 5]  # Minimum number of models agreeing for consensus
summary = []

for threshold in thresholds:
    for min_models in model_agreements_required:
        prot_group = agg[agg["agreement_ratio"] >= threshold]
        per_protein_count = prot_group.groupby("protein_id").size().reset_index(name="num_agreeing_models")
        filtered = per_protein_count[per_protein_count["num_agreeing_models"] >= min_models]
        summary.append({
            "threshold": threshold,
            "min_models_required": min_models,
            "proteins_accepted": len(filtered)
        })

summary_df = pd.DataFrame(summary)
print("\n=== Agreement Summary ===")
print(summary_df)

# Identify models with poor internal consistency
model_variability = (
    agg.groupby("model_type")["agreement_ratio"]
    .agg(["mean", "std"])
    .rename(columns={"mean": "avg_agreement_ratio", "std": "agreement_variability"})
)

print("\n=== Model Self-Agreement Analysis ===")
print(model_variability)

# Optionally save to CSVs
summary_df.to_csv("./results/agreement_summary.csv", index=False)
model_variability.to_csv("./results/model_self_agreement.csv", index=False)
