# Final commit – Master’s Thesis by Àlex Domínguez Roig

import pandas as pd
import ast
import os

def process_results(csv_paths):
    for path in csv_paths:
        df = pd.read_csv(path)

        print(f"🔍 Processing {path}...")

        # Drop full nan columns
        df = df.dropna(axis=1, how='all')

        # find the cv folds column
        cv_folds_col = [col for col in df.columns if "cv_folds" in col.lower()]
        print(f"Found cv_folds column: {cv_folds_col}")
        
        # Fix empty values
        df["cv_folds"] = df[f"{cv_folds_col[0]}"]
        df["cv_balance"] = df["training.cross_val_balance"].fillna("none")

        # Parse sequence embeddings
        seq_embs = df["classifier_definition.sequence_embeddings"].apply(ast.literal_eval)
        df["esm_dim"] = seq_embs.apply(lambda x: x.get("ESM", {}).get("target_dim", None))
        df["prot_dim"] = seq_embs.apply(lambda x: x.get("ProtT5", {}).get("target_dim", None))
        df["prost_dim"] = seq_embs.apply(lambda x: x.get("ProstT5", {}).get("target_dim", None))

        # Parse GO embeddings
        go_embs = df["classifier_definition.go_embeddings"].apply(ast.literal_eval)
        df["go_in"] = go_embs.apply(lambda x: x.get("GeOKG", {}).get("input_dim", None))
        df["go_emb"] = go_embs.apply(lambda x: x.get("GeOKG", {}).get("emb_dim", None))
        df["go_agg"] = go_embs.apply(lambda x: x.get("GeOKG", {}).get("aggregation_strategy", None))
        df["go_cat"] = go_embs.apply(lambda x: x.get("GeOKG", {}).get("go_categories", None))

        # Drop original nested dicts
        df = df.drop(columns=[
            "classifier_definition.sequence_embeddings",
            "classifier_definition.go_embeddings",
            "training.cross_val_balance",
            f"{cv_folds_col[0]}",
        ])

        # Sort by cv_avg_f1
        df = df.sort_values(by="cv_avg_f1", ascending=False)

        print("✅ Successfully processed the file.")
        print("Head of the DataFrame:")
        print(df.head())

        # Save with prefix
        filename = os.path.basename(path)
        output_name = f"../RESULTS/clean_{filename}"
        df.to_csv(output_name, index=False)
        print(f"✅ Saved cleaned file to {output_name}")

# Example usage
csv_files = [
    "../RESULTS/wandb_phase_1_knn.csv",
    "../RESULTS/wandb_phase_1_rf.csv",
    "../RESULTS/wandb_phase_1_svm.csv",
    "../RESULTS/wandb_phase_1_xgb.csv",
    "../RESULTS/wandb_phase_1_lr.csv"
]
processed_dfs = process_results(csv_files)