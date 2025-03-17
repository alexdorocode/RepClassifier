import os
from matplotlib import pyplot as plt
import torch
import seaborn as sns # type: ignore
import numpy as np
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.decomposition import PCA # type: ignore
from sklearn.manifold import TSNE # type: ignore
from sklearn.cluster import KMeans # type: ignore

from torch.utils.data import Dataset
import pandas as pd

class ProteinDataset(Dataset):
    def __init__(self, dataframe, embeddings, attention_weights,
                value_column_embedding, value_column_attention,
                id_embedding, id_attention_weights,
                target_column='Class', id_column='UniProt IDs',
                solve_inconsitence=False, save_path="./OUTPUTS/"):
        
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)

        self.dataframe = dataframe
        self.embeddings = embeddings[value_column_embedding]
        self.attention_weights = attention_weights[value_column_attention]

        print("Checking consistency...")
        self.ensure_consistency(solve_inconsitence, target_column, id_column, id_embedding, id_attention_weights)

        self.labels = self.dataframe[target_column].tolist()
        self.ids = self.dataframe[id_column].tolist()

        self.display_report()

    def check_arguments(self, dataframe, embeddings, attention_weights,
                        id_embedding, id_attention_weights, 
                        value_column_embedding, value_column_attention,
                        target_column, id_column):
        
        if not isinstance(dataframe, pd.DataFrame):
            raise ValueError("dataframe must be a pandas DataFrame.")
        
        if not isinstance(embeddings, dict):
            raise ValueError("embeddings must be a dictionary.")
        if not isinstance(attention_weights, dict):
            raise ValueError("attention_weights must be a dictionary.")
        
        if not isinstance(id_embedding, str):
            raise ValueError("id_embedding must be a string.")
        if not isinstance(id_attention_weights, str):
            raise ValueError("id_attention_weights must be a string.")
        if not isinstance(value_column_embedding, str):
            raise ValueError("value_column_embedding must be a string.")
        if not isinstance(value_column_attention, str):
            raise ValueError("value_column_attention must be a string.")
        if not isinstance(target_column, str):
            raise ValueError("target_column must be a string.")
        if not isinstance(id_column, str):
            raise ValueError("id_column must be a string.")

        if id_column not in dataframe.columns:
            raise ValueError("id_column not found in dataframe.")
        if target_column not in dataframe.columns:
            raise ValueError("target_column not found in dataframe.")
        if id_embedding not in embeddings.columns:
            raise ValueError("id_embedding not found in embeddings.")
        if id_attention_weights not in attention_weights.columns:    
            raise ValueError("id_attention_weights not found in attention_weights.")

    def ensure_consistency(self, solve_inconsitence,
                            target_column, id_column, id_embedding, id_attention_weights):
        
        if self.check_duplicates(target_column) and solve_inconsitence:
            print("Removing duplicates...")
            old_len = len(self.dataframe)
            self.dataframe.drop_duplicates(inplace=True)
            print(f"Removed {old_len - len(self.dataframe)} duplicates.")

        self.check_and_solve_ids_consistency(id_column, id_embedding, id_attention_weights,
                                        solve_inconsitence)
        
    def check_duplicates(self, target_column):
        duplicates = self.dataframe[target_column].duplicated()
        if duplicates.any():
            print("Warning: The dataframe contains duplicates.")
            print(f"Number of duplicates: {duplicates.sum()}")
            print("Use the remove_duplicates function to remove duplicates.")

    def check_main_columns_exist(self, target_column, id_column):
        if target_column not in self.dataframe.columns or id_column not in self.dataframe.columns:
            raise ValueError(f"Dataframe must contain the columns '{target_column}' and '{id_column}'.")

    def check_and_solve_ids_consistency(self, id_column, id_embedding, id_attention_weights, 
                                        solve_inconsitence):
        
        df_ids = set(self.dataframe[id_column])
        emb_ids = set(self.embeddings[id_embedding])
        attn_ids = set(self.attention_weights[id_attention_weights])
        
        if df_ids != emb_ids or df_ids != attn_ids:
            print("Warning: Inconsistency found between dataframe, embeddings, and attention_weights IDs.")
            if solve_inconsitence:
                common_ids = df_ids & emb_ids & attn_ids
                self.dataframe = self.dataframe[self.dataframe[id_column].isin(common_ids)]
                self.embeddings = {k: v for k, v in self.embeddings.items() if k in common_ids}
                self.attention_weights = {k: v for k, v in self.attention_weights.items() if k in common_ids}


    def display_report(self):
        print("ProteinDataset Report:")
        print(f"Number of samples: {len(self.dataframe)}")
        print(f"Number of embeddings: {len(self.embeddings)}")
        print(f"Number of attention weights: {len(self.attention_weights)}")
        print(f"Target column values: {self.labels.unique()}")
        print(f"ID column: {self.id}")
        print(f"Save path: {self.save_path}")

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Convertir de numpy a tensor cuando se accede.
        return (torch.tensor(self.embeddings[idx]), torch.tensor(self.attention_weights[idx])), torch.tensor(self.labels[idx], dtype=torch.float)

    def drop_duplicates(self, column: str):
        """Drop duplicate rows based on a specific column in the dataframe."""
        self.dataframe.drop_duplicates(subset=[column], inplace=True)

    def dropna(self):
        """Drop rows with NaN or None values in embeddings or attention layers."""
        valid_indices = [i for i, (emb, attn) in enumerate(zip(self.embeddings, self.attention_weights)) if emb is not None and attn is not None]
        self.embeddings = [self.embeddings[i] for i in valid_indices]
        self.attention_weights = [self.attention_weights[i] for i in valid_indices]
        self.labels = [self.labels[i] for i in valid_indices]
        self.id = [self.id[i] for i in valid_indices]

    def split_train_test(self, test_size=0.2):
        """Split the dataset into train and test sets."""
        train_indices, test_indices = train_test_split(range(len(self.dataframe)), test_size=test_size, random_state=42, stratify=self.labels)
        train_dataset = torch.utils.data.Subset(self, train_indices)
        test_dataset = torch.utils.data.Subset(self, test_indices)
        return train_dataset, test_dataset

    def plot_tsne(self, attribute='Class', combined=False):
        """Plot TSNE for embeddings, attention_weights, or both combined."""
        if combined:
            data = np.concatenate([self.embeddings, self.attention_weights], axis=1)
        else:
            data = self.embeddings

        # Ensure data is a NumPy array
        data = np.array(data)

        # Check the shape of the data
        print(f"Data shape: {data.shape}")

        tsne = TSNE(n_components=2, random_state=42)
        tsne_results = tsne.fit_transform(data)

        plt.figure(figsize=(16, 10))
        sns.scatterplot(
            x=tsne_results[:, 0], y=tsne_results[:, 1],
            hue=self.dataframe[attribute],
            palette=sns.color_palette("hsv", len(self.dataframe[attribute].unique())),
            legend="full",
            alpha=0.3
        )
        plt.title(f'TSNE plot for {attribute}')
        plt.show()

    def plot_pca(self, attribute='Class', combined=False):
        """Plot PCA for embeddings, attention_weights, or both combined."""
        if combined:
            data = np.concatenate([self.embeddings, self.attention_weights], axis=1)
        else:
            data = self.embeddings

        # Ensure data is a NumPy array
        data = np.array(data)

        pca = PCA(n_components=2)
        pca_results = pca.fit_transform(data)

        plt.figure(figsize=(16, 10))
        sns.scatterplot(
            x=pca_results[:, 0], y=pca_results[:, 1],
            hue=self.dataframe[attribute],
            palette=sns.color_palette("hsv", len(self.dataframe[attribute].unique())),
            legend="full",
            alpha=0.3
        )
        plt.title(f'PCA plot for {attribute}')
        plt.show()

    def plot_pca_variance(self, combined=False):
        """Plot PCA variance explained for embeddings, attention_weights, or both combined."""
        if combined:
            data = np.concatenate([self.embeddings, self.attention_weights], axis=1)
        else:
            data = self.embeddings

        # Ensure data is a NumPy array
        data = np.array(data)

        pca = PCA()
        pca.fit(data)
        explained_variance = pca.explained_variance_ratio_

        plt.figure(figsize=(16, 10))
        plt.plot(np.cumsum(explained_variance))
        plt.xlabel('Number of Components')
        plt.ylabel('Variance Explained')
        plt.title('PCA Variance Explained')
        plt.show()

    def plot_kmeans(self, n_clusters=3, attribute='Class', combined=False):
        """Plot k-means clustering for embeddings, attention_weights, or both combined."""
        if combined:
            data = np.concatenate([self.embeddings, self.attention_weights], axis=1)
        else:
            data = self.embeddings

        # Ensure data is a NumPy array
        data = np.array(data)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(data)
        clusters = kmeans.labels_

        plt.figure(figsize=(16, 10))
        sns.scatterplot(
            x=data[:, 0], y=data[:, 1],
            hue=clusters,
            palette=sns.color_palette("hsv", n_clusters),
            legend="full",
            alpha=0.3
        )
        plt.title(f'K-means clustering with {n_clusters} clusters')
        plt.show()
