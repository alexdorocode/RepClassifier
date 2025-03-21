import os
import torch
import pandas as pd
from torch.utils.data import Dataset

class ProteinDataset(Dataset):
    """
    Handles protein dataset preprocessing, consistency checking, and integration with PyTorch's Dataset API.
    """
    def __init__(self, dataframe, embeddings, attention_weights,
                 target_column='Class', id_column='UniProt IDs',
                 solve_inconsistencies=False, save_path="./OUTPUTS/"):
        
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)
        
        # Validate inputs
        DatasetUtils.check_arguments(dataframe, embeddings, attention_weights, target_column, id_column)
        
        print("Checking consistency...")
        self.dataframe, self.embeddings, self.attention_weights = DatasetUtils.ensure_consistency(
            dataframe, embeddings, attention_weights, id_column, solve_inconsistencies
        )
        print("Consistency checked.")

        self.labels = dataframe[target_column].tolist()
        self.ids = dataframe[id_column].tolist()

        self.display_report(target_column, id_column)

    def display_report(self, target_column, id_column):
        """Prints a summary of the dataset's structure."""
        print("\nProteinDataset Report:")
        print(f" - Number of samples: {len(self.dataframe)}")
        print(f" - Number of embeddings: {len(self.embeddings)}")
        print(f" - Number of attention weights: {len(self.attention_weights)}")
        print(f" - Target column: {target_column}")
        print(f" - ID column: {id_column}")
        print(f" - Save path: {self.save_path}\n")

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        """Returns embeddings, attention weights, and labels as tensors."""
        return (torch.tensor(self.embeddings[idx]), torch.tensor(self.attention_weights[idx])), torch.tensor(self.labels[idx], dtype=torch.float)

    def drop_duplicates(self, column: str, inplace: bool = True):
        """Removes duplicate rows from the dataset based on a specific column."""
        print(f"Samples before deduplication: {len(self.dataframe)}")
        self.dataframe.drop_duplicates(subset=[column], inplace=inplace)
        print(f"Samples after deduplication: {len(self.dataframe)}")

    def dropna(self):
        """Removes samples with missing embeddings or attention weights."""
        valid_indices = [i for i, (emb, attn) in enumerate(zip(self.embeddings, self.attention_weights)) if emb is not None and attn is not None]
        self.embeddings = [self.embeddings[i] for i in valid_indices]
        self.attention_weights = [self.attention_weights[i] for i in valid_indices]
        self.labels = [self.labels[i] for i in valid_indices]
        self.ids = [self.ids[i] for i in valid_indices]


class DatasetUtils:
    """Helper class to manage consistency checks and dataset validation."""
    
    @staticmethod
    def check_arguments(dataframe, embeddings, attention_weights, target_column, id_column):
        if not isinstance(dataframe, pd.DataFrame):
            raise ValueError("dataframe must be a pandas DataFrame.")
        if not isinstance(embeddings, dict) or not isinstance(attention_weights, dict):
            raise ValueError("embeddings and attention_weights must be dictionaries.")
        if not isinstance(target_column, str) or not isinstance(id_column, str):
            raise ValueError("target_column and id_column must be strings.")
        if id_column not in dataframe.columns or target_column not in dataframe.columns:
            raise ValueError(f"Missing columns: Ensure '{id_column}' and '{target_column}' are in the dataframe.")

    @staticmethod
    def check_duplicates(dataframe, id_column):
        duplicates = dataframe[id_column].duplicated()
        if duplicates.any():
            print(f"Warning: {duplicates.sum()} duplicate IDs found in dataframe.")
            return True
        return False

    @staticmethod
    def ensure_consistency(dataframe, embeddings, attention_weights, id_column, solve_inconsistencies):
        """Ensures dataset consistency by aligning IDs and removing duplicates if necessary."""
        if DatasetUtils.check_duplicates(dataframe, id_column) and solve_inconsistencies:
            print("Removing duplicate entries...")
            dataframe.drop_duplicates(subset=[id_column], inplace=True)

        df_ids = set(dataframe[id_column])
        emb_ids = set(embeddings.keys())
        attn_ids = set(attention_weights.keys())

        print(f" - DataFrame IDs: {len(df_ids)}")
        print(f" - Embeddings IDs: {len(emb_ids)}")
        print(f" - Attention Weights IDs: {len(attn_ids)}")

        if df_ids != emb_ids or df_ids != attn_ids:
            print("Warning: Inconsistencies found between dataframe, embeddings, and attention_weights.")
            if solve_inconsistencies:
                common_ids = df_ids & emb_ids & attn_ids
                print(f"Resolving inconsistencies. Keeping {len(common_ids)} common samples.")
                dataframe = dataframe[dataframe[id_column].isin(common_ids)]
                embeddings = {k: embeddings[k] for k in common_ids}
                attention_weights = {k: attention_weights[k].flatten() for k in common_ids}
            else:
                print("Inconsistencies detected but not resolved. Consider enabling `solve_inconsistencies=True`.")

        print(f" - DataFrame IDs: {len(df_ids)}")
        print(f" - Embeddings IDs: {len(emb_ids)}")
        print(f" - Attention Weights IDs: {len(attn_ids)}")

        return dataframe, list(embeddings.values()), list(attention_weights.values())
