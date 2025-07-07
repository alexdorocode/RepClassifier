import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support

# 1. Load data
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
    df["model_type"] = model
    dfs.append(df)
df = pd.concat(dfs, ignore_index=True)
print(f"Loaded {len(df)} rows from {len(dfs)} models.")

# 2. Compute within-model consensus per protein
print("Computing within-model consensus...")
agg = df.groupby(["protein_id", "model_type", "true_label"]).agg(
    votes=("pred_label", lambda x: x.mode()[0]),
    cnt_votes=("pred_label", "count"),
    max_votes=("pred_label", lambda x: x.value_counts().iloc[0])
).reset_index()
agg["agreement_ratio"] = agg["max_votes"] / agg["cnt_votes"]

# 3. Utility to evaluate policies
def evaluate_policy(policy, thresholds, min_k=None):
    results = []
    for thr in thresholds:
        print(f"Evaluating policy: {policy}, threshold: {thr:.2f}")
        sub = agg[agg["agreement_ratio"] >= thr]
        protein_ids = sub["protein_id"].unique()
        total_correct = total_classified = 0
        
        for pid, grp in sub.groupby("protein_id"):
            if policy == "democratic":
                votes = grp["votes"].value_counts()
                top_votes = votes.max()
                # tie → skip (unclassified)
                if list(votes).count(top_votes) > 1:
                    continue  # skip if tie in votes
                label = votes.idxmax()

            elif policy == "democratic_strong_models":
                votes = grp["votes"].value_counts()
                top_votes = votes.max()
                # tie → skip (unclassified)
                if list(votes).count(top_votes) > 1:
                    model_types = grp["model_type"].values
                    if "svm" not in model_types:
                        continue  # skip if neither xgb nor svm classified
                    elif "svm" in model_types:
                        strong_model = "svm"
                    print(f"Using {strong_model} vote for {pid} due to tie in votes.")
                    label = grp[grp["model_type"] == strong_model]["votes"].mode()[0]
                else:
                    label = votes.idxmax()

            else:  # policy == "min_k"
                votes = grp["votes"].value_counts()
                if votes.max() < min_k:
                    continue
                label = votes.idxmax()

            true = grp["true_label"].iloc[0]
            total_classified += 1
            if label == true:
                total_correct += 1

        results.append((thr*100, total_correct, total_classified))
    return pd.DataFrame(results, columns=["threshold", "correct", "classified"])

# 4. Prepare thresholds
thresholds = np.arange(0.501, 0.999, 0.005)

# Evaluate
dem = evaluate_policy("democratic", thresholds)
dem_strong = evaluate_policy("democratic_strong_models", thresholds)
min_k_vals = {k: evaluate_policy("min_k", thresholds, min_k=k) for k in [3, 5]}

# 5. Plot results
plt.figure(figsize=(10,6))
plt.plot(dem["threshold"], dem["classified"], 'b-', label="Majority‑Voting: Classified")
plt.plot(dem["threshold"], dem["correct"], 'b-.', label="Majority‑Voting: Correct")
plt.plot(dem_strong["threshold"], dem_strong["classified"], 'g-', label="Majority‑Voting Tie‑Breaker (SVM): Classified")
plt.plot(dem_strong["threshold"], dem_strong["correct"], 'g-.', label="Majority‑Voting Tie‑Breaker (SVM): Correct")

# Example for min_k=1 (change or loop others)
colors = ['r', 'm', 'y', 'k']

for idx, (k, mk) in enumerate(min_k_vals.items()):
    color = colors[idx % len(colors)]
    plt.plot(mk["threshold"], mk["classified"], f'{color}-', label=f"Min-{k}: Classified")
    plt.plot(mk["threshold"], mk["correct"], f'{color}-.', label=f"Min-{k}: Correct")

# Set y-axis limits and ticks
plt.ylim(0, 316)
yticks = np.arange(0, 317, 10)
plt.yticks(yticks)

# Format y-tick labels to show count and percentage
def ytick_fmt(x, pos):
    pct = 100 * x / 316
    return f"{int(x)} ({pct:.0f}%)"

plt.gca().set_yticklabels([ytick_fmt(y, None) for y in yticks])


plt.xlabel("Within-Model Agreement Threshold (%)")
plt.ylabel("% of Proteins")
plt.title("Classification Performance by Agreement Policy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
