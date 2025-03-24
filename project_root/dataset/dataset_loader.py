import pandas as pd
import numpy as np
import os

class DatasetLoader:
    """
    Handles loading of CSV dataset, embeddings, and attention weights.
    """

    def __init__(self, dataset_path):
        """
        Initialize with the base dataset directory path.

        Args:
            dataset_path (str): Path to the dataset folder containing CSV and .npy files.
        """
        self.dataset_path = dataset_path

    def load_dataframe(self, filename="predictor_dataset.csv"):
        """Loads a CSV file into a Pandas DataFrame."""
        file_path = os.path.join(self.dataset_path, filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        return pd.read_csv(file_path)

    def load_numpy_dict(self, filename):
        """Loads a NumPy dictionary from a .npy file."""
        file_path = os.path.join(self.dataset_path, filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Numpy file not found: {file_path}")
        return np.load(file_path, allow_pickle=True).item()

    def load_embeddings_and_attention(self, embedding_file="prott5_embeddings.npy",
                                      attention_file="prott5_attention_layers.npy"):
        """Loads embeddings and attention weights, transforming them into usable dictionaries."""
        prott5_embeddings = self.load_numpy_dict(embedding_file)
        prott5_attention_weights = self.load_numpy_dict(attention_file)

        embeddings_dict = {id_: emb for id_, emb in zip(prott5_embeddings['UniProt IDs'], prott5_embeddings['embeddings'])}
        attention_weights_dict = {id_: attn for id_, attn in zip(prott5_attention_weights['UniProt IDs'], prott5_attention_weights['attention_layers'])}

        for attention_weights in attention_weights_dict.values():
            print(attention_weights.shape)

        return embeddings_dict, attention_weights_dict
