import os
import torch
import pandas as pd
from torch.utils.data import Dataset


class RawDataset(Dataset):
    """
    Handles protein dataset preprocessing, consistency checking, and integration with PyTorch's Dataset API.
    Stores labels, embeddings, attention weights, and metadata indexed by UniProt IDs.
    """
    def __init__(self, dataframe, embeddings, attention_weights,
                 target_column='Class', id_column='UniProt IDs',
                 solve_inconsistencies=False, save_path="./OUTPUTS/"):

        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)

        # Validate and store input
        DatasetUtils.check_arguments(dataframe, embeddings, attention_weights, target_column, id_column)

        self.dataframe, self.embeddings, self.attention_weights, self.labels, self.ids = DatasetUtils.ensure_consistency(
            dataframe, embeddings, attention_weights, target_column, id_column, solve_inconsistencies
        )

        self.id_column = id_column
        self.target_column = target_column

        self.lengths = self._compute_lengths()
        self.display_report()

    def _compute_lengths(self):
        if 'Amino Acid Sequence' in self.dataframe.columns:
            return self.dataframe.set_index(self.id_column)['Amino Acid Sequence'].apply(len).to_dict()
        return {id_: 0 for id_ in self.ids}  # fallback to 0 if unavailable

    def display_report(self):
        print("\n📊 RawDataset Report")
        print(f" - Samples: {len(self.ids)}")
        print(f" - Embeddings: {len(self.embeddings)}")
        print(f" - Attention Weights: {len(self.attention_weights)}")
        print(f" - Target Column: {self.target_column}")
        print(f" - ID Column: {self.id_column}")
        print(f" - Save Path: {self.save_path}")
        if self.lengths:
            lengths = list(self.lengths.values())
            print(f" - Min Seq Length: {min(lengths)}")
            print(f" - Max Seq Length: {max(lengths)}")
            print(f" - Mean Seq Length: {sum(lengths) / len(lengths):.2f}")
        print()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id_ = list(self.ids.keys())[idx]
        return (
            torch.tensor(self.embeddings[id_]),
            torch.tensor(self.attention_weights[id_])
        ), torch.tensor(self.labels[id_], dtype=torch.float)

    # ========= Accessors =========

    def get_embeddings(self):
        return list(self.embeddings.values())

    def get_attention_weights(self):
        return list(self.attention_weights.values())

    def get_labels(self):
        return list(self.labels.values())

    def get_ids(self):
        return list(self.ids.values())

    def get_lengths(self):
        return [self.lengths.get(id_, 0) for id_ in self.ids]

    def get_attributes(self):
        return list(self.dataframe.columns)

    def get_attribute(self, attribute_name):
        if attribute_name not in self.dataframe.columns:
            raise ValueError(f"❌ Attribute '{attribute_name}' not found in dataframe.")
        return self.dataframe.set_index(self.id_column).loc[list(self.ids.keys()), attribute_name].tolist()

    def has_attribute(self, attribute_name):
        return attribute_name in self.dataframe.columns


# ========= Utility Class =========

class DatasetUtils:
    @staticmethod
    def check_arguments(dataframe, embeddings, attention_weights, target_column, id_column):
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("dataframe must be a pandas DataFrame.")
        if not isinstance(embeddings, dict) or not isinstance(attention_weights, dict):
            raise TypeError("embeddings and attention_weights must be dictionaries.")
        if id_column not in dataframe.columns or target_column not in dataframe.columns:
            raise ValueError(f"DataFrame must include '{id_column}' and '{target_column}'.")

    @staticmethod
    def check_duplicates(dataframe, id_column):
        duplicates = dataframe[id_column].duplicated()
        if duplicates.any():
            print(f"⚠️ Warning: {duplicates.sum()} duplicate IDs found in dataframe.")
            return True
        return False

    @staticmethod
    def ensure_consistency(dataframe, embeddings, attention_weights, target_column, id_column, solve_inconsistencies):
        if DatasetUtils.check_duplicates(dataframe, id_column) and solve_inconsistencies:
            print("🧹 Removing duplicate entries and rows with NaN values...")
            dataframe = dataframe.drop_duplicates(subset=[id_column]).dropna()

        df_ids = set(dataframe[id_column])
        emb_ids = set(embeddings)
        attn_ids = set(attention_weights)

        if df_ids != emb_ids or df_ids != attn_ids:
            print("⚠️ Inconsistencies found between dataframe, embeddings, and attention_weights.")
            if solve_inconsistencies:
                common_ids = df_ids & emb_ids & attn_ids
                print(f"Keeping {len(common_ids)} common samples.")
                dataframe = dataframe[dataframe[id_column].isin(common_ids)]
                embeddings = {k: embeddings[k] for k in common_ids}
                attention_weights = {k: attention_weights[k] for k in common_ids}
            else:
                print("Not resolving inconsistencies. Set solve_inconsistencies=True to auto-fix.")

        labels = {row[id_column]: row[target_column] for _, row in dataframe.iterrows()}
        ids = {id_: id_ for id_ in dataframe[id_column]}

        return dataframe, embeddings, attention_weights, labels, ids
