import pandas as pd
import matplotlib.pyplot as plt
import os

# Define file paths
file_paths = {
    "lr": "./results/zero_shot_prediction_agreement_lr.csv",
    "svm": "./results/zero_shot_prediction_agreement_svm.csv",
    "rf": "./results/zero_shot_prediction_agreement_rf.csv",
    "xgb": "./results/zero_shot_prediction_agreement_xgb.csv",
    "knn": "./results/zero_shot_prediction_agreement_knn.csv"
}

# Load and combine available files
all_data = []
for model_type, path in file_paths.items():
    if os.path.exists(path):
        df = pd.read_csv(path)
        df["model_type"] = model_type
        all_data.append(df)
    else:
        print(f"⚠️ File not found and skipped: {path}")

# Continue only if some data was loaded
if not all_data:
    raise RuntimeError("❌ No valid CSV files found in ./results/. Please check paths.")

df_all = pd.concat(all_data, ignore_index=True)

# Compute agreement ratio per (protein, model)
agg = (
    df_all.groupby(["protein_id", "model_type"])["is_correct"]
    .agg(["sum", "count"])
    .reset_index()
)
agg["agreement_ratio"] = agg["sum"] / agg["count"]
agg["agreed"] = (agg["agreement_ratio"] >= 0.8).astype(int)

# Pivot: proteins x model_type with 0/1 agreement flags
pivot = agg.pivot(index="protein_id", columns="model_type", values="agreed").fillna(0).astype(int)

# Sort proteins by total agreement across models
pivot["total"] = pivot.sum(axis=1)
pivot_sorted = pivot.sort_values("total", ascending=False).drop(columns="total")

# Plot stacked bar chart
fig, ax = plt.subplots(figsize=(20, 6))
pivot_sorted.plot(kind="bar", stacked=True, colormap="tab20", width=1.0, ax=ax)

# Labels and title
ax.set_ylabel("Models with ≥80% Agreement")
ax.set_xlabel("Proteins (sorted by number of agreeing models)")
ax.set_title("Protein-Level Agreement (≥80%) Across Available Models")
ax.legend(title="Model Type", bbox_to_anchor=(1.01, 1), loc="upper left")

# Vertical lines every 10% (from 50% to 100%)
num_proteins = pivot_sorted.shape[0]
for frac in range(50, 101, 10):
    pos = int(num_proteins * frac / 100)
    ax.axvline(pos, color='gray', linestyle='--', alpha=0.3)
    ax.text(pos, ax.get_ylim()[1] * 1.01, f"{frac}%", rotation=90, va='top', ha='center', fontsize=8, color='gray')

# Horizontal lines every 20%
y_max = ax.get_ylim()[1]
for frac in range(20, 101, 20):
    y = y_max * frac / 100
    ax.axhline(y, color='gray', linestyle='--', alpha=0.3)
    ax.text(-5, y, f"{frac}%", va='center', ha='left', fontsize=8, color='gray')

# Hide x-ticks for readability
ax.set_xticks([])

# Save and show plot
plt.tight_layout()
output_path = "./results/stacked_agreement_per_protein_80pct_threshold.png"
plt.savefig(output_path, dpi=300)
plt.show()
print(f"✅ Plot saved to: {output_path}")
