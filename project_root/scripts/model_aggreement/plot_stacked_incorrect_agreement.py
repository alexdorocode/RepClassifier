import pandas as pd
import matplotlib.pyplot as plt
import os

# File loading (same as before)

# Define file paths
file_paths = {
    "lr": "./results/zero_shot_prediction_agreement_lr.csv",
    "svm": "./results/zero_shot_prediction_agreement_svm.csv",
    "rf": "./results/zero_shot_prediction_agreement_rf.csv",
    "xgb": "./results/zero_shot_prediction_agreement_xgb.csv",
    "knn": "./results/zero_shot_prediction_agreement_knn.csv"
}

all_data = []
for model_type, path in file_paths.items():
    if os.path.exists(path):
        df = pd.read_csv(path)
        df["model_type"] = model_type
        all_data.append(df)
df_all = pd.concat(all_data, ignore_index=True)

# 1. Compute agreement counts per protein+model
agg = (
    df_all.groupby(["protein_id", "model_type"])
    .agg(correct_runs=("is_correct","sum"), total_runs=("is_correct","count"))
    .reset_index()
)
agg["agreement_ratio"] = agg["correct_runs"] / agg["total_runs"]

# 2. Keep only proteins where agreement ≥ 80%
agg80 = agg[agg["agreement_ratio"] >= 0.8]

# 3. Mark incorrect cases: where majority vote was wrong
#    We assume "correct_runs" counts true predictions.
agg80["incorrect_agreement_count"] = agg80.apply(
    lambda r: r.total_runs - r.correct_runs if r.agreement_ratio >= 0.8 else 0,
    axis=1
)

# 4. Pivot: proteins × model, value = number of incorrect-agreeing runs
pivot = agg80.pivot(index="protein_id", columns="model_type", values="incorrect_agreement_count")
pivot = pivot.fillna(0).astype(int)

# 5. Sum per protein and sort ascending (least incorrect to most)
pivot["total_incorrect_agreements"] = pivot.sum(axis=1)
pivot_sorted = pivot.sort_values("total_incorrect_agreements", ascending=True).drop(columns="total_incorrect_agreements")
print(pivot_sorted.head())
# Plot stacked bar
fig, ax = plt.subplots(figsize=(20, 6))
pivot_sorted.plot(kind="bar", stacked=True, colormap="tab20", width=1.0, ax=ax)

ax.set_ylabel("Number of Incorrect-Agreeing Runs")
ax.set_xlabel("Proteins (ordered by total incorrect agreement)")
ax.set_title("Proteins with ≥80% Agreement but Incorrect Prediction")

# Optional: no x ticks
ax.set_xticks([])

# Save/show
plt.tight_layout()
plt.savefig("./results/incorrect_agreement_proteins_80.png", dpi=300)
plt.show()
