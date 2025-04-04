from project_root.dataset.representation_dataset import RepresentationDataset
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns  # type: ignore
from sklearn.cluster import KMeans  # type: ignore

from project_root.utils.feature_processor import (
    flatten_attention_weights,
    pad_attention_weights,
    process_embeddings_and_attention,
)


class WrappedRepresentationDataset(RepresentationDataset):
    def __init__(self, dataset,
                 process_attention_weights = True,
                 reduce_method=None,
                 pca_method='threshold',
                 random_projection_dim=1000,
                 random_projection_method='global',
                 threshold=0.95,
                 attributes_to_one_hot=None):
        """
        Initializes WrappedProteinDataset with optional dimensionality reduction.

        Args:
            dataset (ProteinDataset): A pre-existing instance of ProteinDataset.
            reduce_method (str): Type of dimensionality reduction to apply ('pca' or 'tsne').
            random_projection_dim (int): Target dimensionality for random projection before PCA.
        """
        self.dataset = dataset  # Store original dataset reference
        self.one_hot_encoded_attributes = {}
        self.integer_encoded_attributes = {}

        if attributes_to_one_hot:
            for attribute in attributes_to_one_hot:
                if attribute not in self.dataset.get_attributes():
                    raise ValueError(f"Attribute '{attribute}' not found in dataset.")
                self.one_hot_encode_attribute(attribute)


        print("Converting embeddings and attention weights to NumPy arrays...")
        embeddings_array = np.array(self.dataset.get_embeddings())
        
        if process_attention_weights:
            flattened_attention_weights = flatten_attention_weights(self.dataset.get_attention_weights())
            padded_attention_weights = pad_attention_weights(flattened_attention_weights)
        else:
            padded_attention_weights = None

        reduced_embeddings, reduced_attention_weights = process_embeddings_and_attention(
            embeddings_array,
            padded_attention_weights,
            process_attention_weights,
            reduce_method=reduce_method,
            pca_method=pca_method,
            threshold=threshold,
            random_projection_dim=random_projection_dim,
            random_projection_method=random_projection_method
        )

        self.embeddings = reduced_embeddings
        self.attention_weights = reduced_attention_weights
        if self.attention_weights is not None:
            self.combined_embeddings_and_attention = np.concatenate([self.embeddings, self.attention_weights], axis=1)

        if attributes_to_one_hot:
            for attribute in attributes_to_one_hot:
                if attribute not in self.dataset.get_attributes():
                    raise ValueError(f"Attribute '{attribute}' not found in dataset.")
                self.one_hot_encode_attribute(attribute)

    def select_data(self,
                    embedding=False,
                    attention_weights=False,
                    id_column=False,
                    target_column=False,
                    additional_columns=None,
                    length_column=False,
                    one_hot_columns=None):
        """Select data for visualization and print the column order."""

        if attention_weights and self.attention_weights is None:
            raise ValueError("Attention weights are not available for this dataset.")

        if embedding and attention_weights:
            data = self.combined_embeddings_and_attention
        elif embedding:
            data = self.embeddings
        elif attention_weights:
            data = self.attention_weights
        else:
            data = []

        column_order = []
        current_column_index = 0

        if target_column:
            print("Adding labels to data...")
            labels = np.array(self.dataset.get_labels()).reshape(-1, 1)
            print(f"Shape data before adding: {data.shape} | Shape labels: {labels.shape}")
            data = np.concatenate([labels, data], axis=1)
            column_order.append("target_column")
            current_column_index += 1

        if additional_columns:
            print("Adding additional columns to data...")
            for column in additional_columns:
                attribute = np.array(self.dataset.get_attribute(column))
                if attribute.ndim == 1:
                    attribute = attribute.reshape(-1, 1)
                print(f"Shape data before adding: {data.shape} | Shape column: {attribute.shape}")
                data = np.concatenate([attribute, data], axis=1)
                column_order.append(column)
                current_column_index += attribute.shape[1]


        if length_column:
            print("Adding lengths to data...")
            lengths = np.array(self.dataset.get_lengths()).reshape(-1, 1)
            print(f"Shape data before adding: {data.shape} | Shape lengths: {lengths.shape}")
            data = np.concatenate([lengths, data], axis=1)
            column_order.append("length_column")
            current_column_index += 1

        if id_column:
            print("Adding IDs to data...")
            ids = np.array(self.dataset.get_ids()).reshape(-1, 1)
            print(f"Shape data before adding: {data.shape} | Shape IDs: {ids.shape}")
            data = np.concatenate([ids, data], axis=1)
            column_order.append("id_column")
            current_column_index += 1

        if one_hot_columns:
            print("Adding one-hot encoded attributes to data...")
            for column in one_hot_columns:
                if column not in self.one_hot_encoded_attributes:
                    self.one_hot_encode_attribute(column)
                one_hot_data = self.one_hot_encoded_attributes[column]
                data = np.concatenate([one_hot_data, data], axis=1)
                column_order.append(f"one_hot_{column}")
                current_column_index += one_hot_data.shape[1]

        print(f"Final data shape: {data.shape}")
        print("Column order in the resulting dataset:")
        for index, column_name in enumerate(column_order):
            print(f"Column {index}: {column_name}")

        return data


    def one_hot_encode_attribute(self, attribute):
        """ One-hot encode a specified attribute."""
        print(f"One-hot encoding attribute: {attribute}")
        if attribute in self.one_hot_encoded_attributes:
            print(f"Attribute '{attribute}' is already one-hot encoded.")
            return
        attribute_data = self.dataset.get_attribute(attribute)
        unique_values = set()
        for go_terms in attribute_data:
            for term in go_terms:
                if isinstance(term, str):
                    unique_values.add(term)
                else:
                    raise ValueError(f"Invalid term type: {type(term)}. Expected string.")
    
        # Map unique values to integers
        value_to_int = {value: idx for idx, value in enumerate(unique_values)}
    
        # Convert attribute_data to integer indices
        integer_encoded = []
        for go_terms in attribute_data:
            row_indices = [value_to_int[term] for term in go_terms if term in value_to_int]
            integer_encoded.append(row_indices)
    
        # Create one-hot encoded array
        one_hot_encoded = np.zeros((len(attribute_data), len(unique_values)))
        for i, row_indices in enumerate(integer_encoded):
            one_hot_encoded[i, row_indices] = 1
    
        self.one_hot_encoded_attributes[attribute] = one_hot_encoded
        self.integer_encoded_attributes[attribute] = value_to_int

"""
    def plot_kmeans(self, n_clusters=3, attribute='Class', embedding=False, attention_weights=False):
        data = self.select_data(embedding, attention_weights)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(data)
        clusters = kmeans.labels_

        plt.figure(figsize=(16, 10))
        sns.scatterplot(
            x=data[:, 0],
            y=data[:, 1],
            hue=clusters,
            style=self.dataset.dataframe[attribute],
            palette=sns.color_palette("hsv", n_clusters),
            legend="full",
            alpha=0.7
        )
        plt.title(f'K-means clustering with {n_clusters} clusters')
        plt.show()


    def plot_correration_heatmap(self, embedding=False, attention_weights=False):
        data = self.select_data(embedding, attention_weights)
        corr = np.corrcoef(data, rowvar=False)
        plt.figure(figsize=(16, 10))
        sns.heatmap(corr, annot=False, cmap='coolwarm')
        plt.title('Correlation Heatmap')
        plt.show()
"""