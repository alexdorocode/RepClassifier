import os
import torch
import pandas as pd
from torch.utils.data import Dataset


class RepresentationDataset(Dataset):
    """
    Handles protein dataset preprocessing, consistency checking, and integration with PyTorch's Dataset API.
    Stores labels, embeddings, attention weights, and ids as dictionaries indexed by UniProt IDs.
    """
    def __init__(self, dataframe, embeddings, attention_weights,
                 target_column='Class', id_column='UniProt IDs',
                 solve_inconsistencies=False, save_path="./OUTPUTS/"):

        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)

        # Validate inputs
        DatasetUtils.check_arguments(dataframe, embeddings, attention_weights, target_column, id_column)

        print("Checking consistency...")
        self.dataframe, self.embeddings, self.attention_weights, self.labels, self.ids = DatasetUtils.ensure_consistency(
            dataframe, embeddings, attention_weights, target_column, id_column, solve_inconsistencies
        )
        print("Consistency checked.")

        self.display_report(target_column, id_column)

    def display_report(self, target_column, id_column):
        print("\nProteinDataset Report:")
        print(f" - Number of samples: {len(self.ids)}")
        print(f" - Number of embeddings: {len(self.embeddings)}")
        print(f" - Number of attention weights: {len(self.attention_weights)}")
        print(f" - Target column: {target_column}")
        print(f" - ID column: {id_column}")
        print(f" - Save path: {self.save_path}\n")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id_ = list(self.ids.keys())[idx]
        return (
            torch.tensor(self.embeddings[id_]),
            torch.tensor(self.attention_weights[id_])
        ), torch.tensor(self.labels[id_], dtype=torch.float)

    def get_embeddings(self):
        return list(self.embeddings.values())

    def get_attention_weights(self):
        return list(self.attention_weights.values())

    def get_labels(self):
        return list(self.labels.values())

    def get_ids(self):
        return list(self.ids.values())

    def get_attribute(self, attribute_name):
        if attribute_name not in self.dataframe.columns:
            raise ValueError(f"Attribute '{attribute_name}' not found in dataframe.")
        return self.dataframe.set_index('UniProt IDs').loc[list(self.ids.keys()), attribute_name].tolist()


class DatasetUtils:
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
    def ensure_consistency(dataframe, embeddings, attention_weights, target_column, id_column, solve_inconsistencies):
        if DatasetUtils.check_duplicates(dataframe, id_column) and solve_inconsistencies:
            print("Removing duplicate entries...")
            dataframe = dataframe.drop_duplicates(subset=[id_column])

        df_ids = set(dataframe[id_column])
        emb_ids = set(embeddings.keys())
        attn_ids = set(attention_weights.keys())

        if df_ids != emb_ids or df_ids != attn_ids:
            print("Warning: Inconsistencies found between dataframe, embeddings, and attention_weights.")
            if solve_inconsistencies:
                common_ids = df_ids & emb_ids & attn_ids
                print(f"Resolving inconsistencies. Keeping {len(common_ids)} common samples.")
                dataframe = dataframe[dataframe[id_column].isin(common_ids)]
                embeddings = {k: embeddings[k] for k in common_ids}
                attention_weights = {k: attention_weights[k].flatten() for k in common_ids}
            else:
                print("Inconsistencies detected but not resolved. Consider enabling solve_inconsistencies=True.")

        labels = {row[id_column]: row[target_column] for _, row in dataframe.iterrows()}
        ids = {id_: id_ for id_ in dataframe[id_column]}
        return dataframe, embeddings, attention_weights, labels, ids
