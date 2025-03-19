import os
import torch

from torch.utils.data import Dataset
import pandas as pd

class ProteinDataset(Dataset):
    def __init__(self, dataframe, embeddings, attention_weights,
                target_column='Class', id_column='UniProt IDs',
                solve_inconsitence=False,
                save_path="./OUTPUTS/"):
        
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)

        self.dataframe = dataframe

        print("Checking consistency...")
        self.ensure_consistency(embeddings, attention_weights, id_column, solve_inconsitence)
        print("Consistency checked.")

        self.labels = self.dataframe[target_column].tolist()
        self.ids = self.dataframe[id_column].tolist()
        self.save_path = save_path

        self.display_report(target_column, id_column)

    def check_arguments(self, dataframe, embeddings, attention_weights,
                        target_column, id_column):
        
        if not isinstance(dataframe, pd.DataFrame):
            raise ValueError("dataframe must be a pandas DataFrame.")
        
        if not isinstance(embeddings, dict):
            raise ValueError("embeddings must be a dictionary.")
        if not isinstance(attention_weights, dict):
            raise ValueError("attention_weights must be a dictionary.")
        
        if not isinstance(target_column, str):
            raise ValueError("target_column must be a string.")
        if not isinstance(id_column, str):
            raise ValueError("id_column must be a string.")

        if id_column not in dataframe.columns:
            raise ValueError("id_column not found in dataframe.")
        if target_column not in dataframe.columns:
            raise ValueError("target_column not found in dataframe.")

    def ensure_consistency(self, embeddings, attention_weights, id_column, solve_inconsitence):
        
        if self.check_duplicates(id_column) and solve_inconsitence:
            print("Removing duplicates...")
            old_len = len(self.dataframe)
            self.dataframe.drop_duplicates(subset=[id_column], inplace=True)
            print(f"Removed {old_len - len(self.dataframe)} duplicates.")

        self.check_and_solve_ids_consistency(embeddings, attention_weights, id_column, solve_inconsitence)
        
    def check_duplicates(self, id_column):
        duplicates = self.dataframe[id_column].duplicated()
        if duplicates.any():
            print("Warning: The dataframe contains duplicates.")
            print(f"Number of duplicates: {duplicates.sum()}")
            print("Use the remove_duplicates function to remove duplicates.")
            return True
        return False

    def check_and_solve_ids_consistency(self, embeddings, attention_weights, id_column, solve_inconsitence):
        df_ids = set(self.dataframe[id_column])
        emb_ids = set(embeddings.keys())
        attn_ids = set(attention_weights.keys())
        
        if df_ids != emb_ids or df_ids != attn_ids:
            print("Warning: Inconsistency found between dataframe, embeddings, and attention_weights IDs.")
            if solve_inconsitence:
                print("Solving inconsistency...")
                common_ids = df_ids & emb_ids & attn_ids
                self.dataframe = self.dataframe[self.dataframe[id_column].isin(common_ids)]
                self.embeddings = [embeddings[k] for k in common_ids]
                self.attention_weights = [attention_weights[k] for k in common_ids]

    def display_report(self, target_column, id_column):
        print("ProteinDataset Report:")
        print(f"Number of samples: {len(self.dataframe)}")
        print(f"Number of embeddings: {len(self.embeddings)}")
        print(f"Number of attention weights: {len(self.attention_weights)}")
        print(f"Target column: {target_column}")
        print(f"ID column: {id_column}")
        print(f"Save path: {self.save_path}")

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Convertir de numpy a tensor cuando se accede.
        return (torch.tensor(self.embeddings[idx]), torch.tensor(self.attention_weights[idx])), torch.tensor(self.labels[idx], dtype=torch.float)

    def drop_duplicates(self, column: str, inplace: bool = True):
        """Drop duplicate rows based on a specific column in the dataframe."""
        print(f"Number of samples before removing duplicates: {len(self.dataframe)}")
        self.dataframe.drop_duplicates(subset=[column], inplace=inplace)
        print(f"Number of samples after removing duplicates: {len(self.dataframe)}")

    def dropna(self):
        """Drop rows with NaN or None values in embeddings or attention layers."""
        valid_indices = [i for i, (emb, attn) in enumerate(zip(self.embeddings, self.attention_weights)) if emb is not None and attn is not None]
        self.embeddings = [self.embeddings[i] for i in valid_indices]
        self.attention_weights = [self.attention_weights[i] for i in valid_indices]
        self.labels = [self.labels[i] for i in valid_indices]
        self.id = [self.id[i] for i in valid_indices]



"""
        Combine embeddings and attention_weights by applying PCA trimming to attention_weights.
        # Flatten attention weights with progress bar
    def new_combine_by_pca_trimming(self, pca_components=100):

        flattened_attention_weights = [attn.flatten() for attn in tqdm(self.attention_weights, desc="Flattening attention weights")]
        
        # Ensure all flattened attention weights have the same shape
        max_length = max(len(attn) for attn in flattened_attention_weights)
        padded_attention_weights = [np.pad(attn, (0, max_length - len(attn)), 'constant') for attn in tqdm(flattened_attention_weights, desc="Padding attention weights")]
        
        # Convert flattened attention weights to NumPy array
        flattened_attention_weights_array = np.array(padded_attention_weights)
        
        # Apply PCA to the entire matrix of flattened attention weights
        pca = PCA(n_components=min(pca_components, flattened_attention_weights_array.shape[1]))
        reduced_attention_weights = pca.fit_transform(flattened_attention_weights_array)
        
        # Convert embeddings to NumPy arrays
        embeddings_array = np.array(self.embeddings)
        
        # Check the shapes of the arrays
        print(f"Embeddings shape: {embeddings_array.shape}")
        print(f"Reduced attention weights shape: {reduced_attention_weights.shape}")
        
        # Concatenate embeddings and reduced attention weights along the feature dimension
        combined_data = np.concatenate([embeddings_array, reduced_attention_weights], axis=1)
        
        return combined_data

    def calculate_best_pca_components(self, data, method='threshold', threshold=0.95):
        Calculate the best number of PCA components using a specified method.
        
        Args:
            data (np.ndarray): The data to perform PCA on.
            method (str): The method to use ('threshold' or 'derivative').
            threshold (float): The variance threshold for the 'threshold' method.
        
        Returns:
            int: The best number of PCA components.
        """"""
        pca = PCA()
        pca.fit(data)
        explained_variance = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        
        if method == 'threshold':
            best_n_components = np.argmax(cumulative_variance >= threshold) + 1
        elif method == 'derivative':
            derivatives = np.diff(cumulative_variance)
            best_n_components = np.argmax(derivatives) + 1
        else:
            raise ValueError("Invalid method. Use 'threshold' or 'derivative'.")
        
        return best_n_components

    def get_best_pca_components_for_embeddings(self, method='threshold', threshold=0.95):
        """"""Get the best number of PCA components for embeddings.""""""
        embeddings_array = np.array(self.embeddings)
        return self.calculate_best_pca_components(embeddings_array, method, threshold)

    def get_best_pca_components_for_attention_weights(self, method='threshold', threshold=0.95):
        """"""Get the best number of PCA components for attention_weights.""""""
        flattened_attention_weights = [attn.flatten() for attn in self.attention_weights]
        max_length = max(len(attn) for attn in flattened_attention_weights)
        padded_attention_weights = [np.pad(attn, (0, max_length - len(attn)), 'constant') for attn in flattened_attention_weights]
        flattened_attention_weights_array = np.array(padded_attention_weights)
        return self.calculate_best_pca_components(flattened_attention_weights_array, method, threshold)

    def get_best_pca_components_for_combined_data(self, method='threshold', threshold=0.95):
        combined_data = self.combine_by_pca_trimming()
        return self.calculate_best_pca_components(combined_data, method, threshold)"
        """#Get the best number of PCA components for combined embeddings and attention_weights."""
        