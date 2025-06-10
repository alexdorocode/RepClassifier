import pandas as pd
import ast
import os

def process_results(csv_paths):
    for path in csv_paths:
        df = pd.read_csv(path)

        print(f"🔍 Processing {path}...")

        # Drop full nan columns
        df = df.dropna(axis=1, how='all')

        rename_map = {
            "kernel_config.coef0": "coef0",
            "kernel_config.degree": "degree",
            "kernel_config.gamma": "gamma",
            "kernel_config.kernel": "kernel",
        }
        df = df.rename(columns=rename_map)
        

        columns_to_drop = [
            "classifier_definition.sequence_embeddings",
            "classifier_definition.go_embeddings"
        ]

        # Find the cv folds column
        cv_folds_col = [col for col in df.columns if "cv_folds" in col.lower()]
        if cv_folds_col:
            columns_to_drop.append(f"{cv_folds_col[0]}")

        # Drop original nested dicts
        df = df.drop(columns=columns_to_drop, errors='ignore')
        print(f"Length of DataFrame after dropping columns: {len(df)}")

        """
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
        """

        # Sort by cv_avg_f1
        df = df.sort_values(by="cv_avg_f1", ascending=False)

        print("✅ Successfully processed the file.")
        print("Head of the DataFrame:")
        print(df.head())

        print("Columns in the DataFrame:")
        print(df.columns.tolist())

        # Save with prefix
        filename = os.path.basename(path)
        output_name = f"clean_{filename}"
        df.to_csv(output_name, index=False)
        print(f"✅ Saved cleaned file to {output_name}")


# Example usage
csv_files = [
#    "../RESULTS/wandb_phase_2_params_knn.csv",
#    "../RESULTS/wandb_phase_2_params_rf.csv",
#    "../RESULTS/wandb_phase_2_params_svm.csv",
#    "../RESULTS/wandb_phase_2_params_xgb.csv",
#    "../RESULTS/wandb_phase_2_params_lr.csv",
    "../RESULTS/wandb_phase_2_xgb.csv",
    "../RESULTS/wandb_phase_2_rf.csv",
    "../RESULTS/wandb_phase_2_svm.csv",
    "../RESULTS/wandb_phase_2_knn.csv",
    "../RESULTS/wandb_phase_2_lr.csv"
]
processed_dfs = process_results(csv_files)