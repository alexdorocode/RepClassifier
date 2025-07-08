# Final commit – Master’s Thesis by Àlex Domínguez Roig

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Load all data
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

# 2. Compute per-model, per-protein consensus
agg = df.groupby(["model_type", "protein_id", "true_label"]).agg(
    pred_majority=("pred_label", lambda x: x.mode()[0]),
    total_votes=("pred_label", "count"),
    top_votes=("pred_label", lambda x: x.value_counts().max())
).reset_index()
agg["agree_ratio"] = agg["top_votes"] / agg["total_votes"]

total_proteins = len(agg["protein_id"].unique())  # Or set this to your total protein count

def compute_counts(model_df):
    thresholds = np.arange(0.501, 0.999, 0.005)
    counts = {
        "threshold": [],
        "TP": [],
        "TN": [],
        "FP": [],
        "FN": [],
        "correct": [],
        "incorrect": [],
        "unclassified": [],
        "classifiable": [],
        "f1": [],
    }
    proteins = model_df["protein_id"].unique()
    for thr in thresholds:
        sub = model_df[model_df["agree_ratio"] >= thr]
        TP = TN = FP = FN = 0

        for pid in proteins:
            dfp = sub[sub["protein_id"] == pid]
            if dfp.empty:
                continue
            true = dfp["true_label"].iloc[0]
            pred = dfp["pred_majority"].iloc[0]
            # Assuming binary classification with labels 0 (negative) and 1 (positive)
            if pred == 1 and true == 1:
                TP += 1
            elif pred == 0 and true == 0:
                TN += 1
            elif pred == 1 and true == 0:
                FP += 1
            elif pred == 0 and true == 1:
                FN += 1

        classifiable = TP + TN + FP + FN
        unclassified = len(proteins) - classifiable
        correct = TP + TN
        incorrect = FP + FN

        # F1 score calculation (binary)
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        counts["threshold"].append(thr)
        counts["TP"].append(TP)
        counts["TN"].append(TN)
        counts["FP"].append(FP)
        counts["FN"].append(FN)
        counts["correct"].append(correct)
        counts["incorrect"].append(incorrect)
        counts["unclassified"].append(unclassified)
        counts["classifiable"].append(classifiable)
        counts["f1"].append(f1*total_proteins)  # Scale F1 by total proteins for consistency

    return pd.DataFrame(counts)

# Assign a unique color to each model
model_colors = {model: color for model, color in zip(file_paths.keys(), plt.cm.tab10.colors)}

line_styles = {
    "correct": "-",
    "incorrect": "--",
    "unclassified": ":",
    "f1": "-.",
    "classifiable": "-"
}

plt.figure(figsize=(12, 8))
for model in file_paths.keys():
    print(f"Processing model: {model}")
    dfm = agg[agg["model_type"] == model]
    stats = compute_counts(dfm)
    color = model_colors[model]
    #plt.plot(stats["threshold"], stats["correct"], label=f"{model.upper()} Correct", linestyle=line_styles["correct"], color=color)
    #plt.plot(stats["threshold"], stats["incorrect"], label=f"{model.upper()} Incorrect", linestyle=line_styles["incorrect"], color=color)
    #plt.plot(stats["threshold"], stats["unclassified"], label=f"{model.upper()} Unclassified", linestyle=line_styles["unclassified"], color=color)
    plt.plot(stats["threshold"], stats["f1"], label=f"{model.upper()} F1 over Classifiable Proteins", linestyle=line_styles["f1"], color=color)
    plt.plot(stats["threshold"], stats["classifiable"], label=f"{model.upper()} Classifiable Proteins", linestyle=line_styles["classifiable"], color=color)


# Set y-axis limits and ticks
plt.ylim(0, 316)
yticks = np.arange(0, 317, 10)
plt.yticks(yticks)

# Format y-tick labels to show count and percentage
def ytick_fmt(x, pos):
    pct = 100 * x / total_proteins if total_proteins else 0
    return f"{pct:.0f}%"

plt.gca().set_yticklabels([ytick_fmt(y, None) for y in yticks])

plt.xlabel("Within-Model Agreement Threshold")
plt.ylabel("% of total")
plt.title("F1 score over Classifiable Proteins by Agreement Threshold")
plt.legend(loc='lower left')
plt.grid(True, axis='y', which='both', linestyle=':', linewidth=0.7)
plt.tight_layout()
plt.show()
