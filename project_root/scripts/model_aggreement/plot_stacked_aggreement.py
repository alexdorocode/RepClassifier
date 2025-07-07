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

# Load and combine data
all_data = []
for model_type, path in file_paths.items():
    if not os.path.exists(path):
        print(f"⚠️ File not found: {path}")
        continue
    df = pd.read_csv(path)
    df["model_type"] = model_type
    all_data.append(df)

if not all_data:
    raise RuntimeError("❌ No CSV files loaded. Check your paths.")

# Combine into single DataFrame
df_all = pd.concat(all_data, ignore_index=True)

# Filter to only correct predictions
df_correct = df_all[df_all["is_correct"] == 1]

# Group by protein and model_type
grouped = df_correct.groupby(["protein_id", "model_type"]).size().unstack(fill_value=0)

# Order proteins by total correct counts
grouped["total"] = grouped.sum(axis=1)
grouped_sorted = grouped.sort_values("total", ascending=False).drop(columns="total")

# Plot stacked bar
fig, ax = plt.subplots(figsize=(20, 6))
grouped_sorted.plot(kind="bar", stacked=True, colormap="tab20", width=1.0, ax=ax)

# Axis labels and title
ax.set_ylabel("Number of Correct Predictions")
ax.set_xlabel("Proteins (sorted by total correct predictions)")
ax.set_title("Correct Classifications per Protein (Zero-Shot, Stacked by Model)")
ax.legend(title="Model Type", bbox_to_anchor=(1.01, 1), loc="upper left")

# Get total number of proteins
num_proteins = grouped_sorted.shape[0]

# Add vertical lines from 50% to 100%, every 10%
for frac in range(50, 91, 10):
    pos = int(num_proteins * frac / 100)
    ax.axvline(pos, color='gray', linestyle='--', alpha=0.3)
    ax.text(pos, ax.get_ylim()[1]*1.01, f"{frac}%", rotation=90, va='top', ha='center', fontsize=8, color='gray')

# Add horizontal lines every 20% of y max
y_max = ax.get_ylim()[1]
for frac in range(20, 101, 20):
    y = y_max * frac / 100
    ax.axhline(y, color='gray', linestyle='--', alpha=0.3)
    ax.text(-5, y, f"{frac}%", va='center', ha='left', fontsize=8, color='gray')

# Hide x-tick labels for clarity
ax.set_xticks([])

# Adjust layout and save
plt.tight_layout()
output_path = "./results/stacked_correct_predictions_per_protein.png"
plt.savefig(output_path, dpi=300)
plt.show()
print(f"✅ Plot saved to: {output_path}")

# Save and show plot
output_path = "./results/stacked_correct_predictions_per_protein.png"
plt.savefig(output_path, dpi=300)
plt.show()
print(f"✅ Plot saved to: {output_path}")
