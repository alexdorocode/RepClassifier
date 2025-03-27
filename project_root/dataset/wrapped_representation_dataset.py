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
    def __init__(self, dataset, process_attention_weights = True, reduce_method=None, pca_method='threshold', random_projection_dim=1000, random_projection_method='global', threshold=0.95):
        """
        Initializes WrappedProteinDataset with optional dimensionality reduction.

        Args:
            dataset (ProteinDataset): A pre-existing instance of ProteinDataset.
            reduce_method (str): Type of dimensionality reduction to apply ('pca' or 'tsne').
            random_projection_dim (int): Target dimensionality for random projection before PCA.
        """
        self.dataset = dataset  # Store original dataset reference

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

    def select_data(self, embedding=False, attention_weights=False, id_column=False, target_column=False, additional_columns=None):
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
            raise ValueError("At least one of 'embedding' or 'attention_weights' must be True.")
        
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
                attribute = np.array(self.dataset.get_attribute(column)).reshape(-1, 1)
                print(f"Shape data before adding: {data.shape} | Shape column: {attribute.shape}")
                data = np.concatenate([attribute, data], axis=1)
                column_order.append(column)
                current_column_index += 1

        if id_column:
            print("Adding IDs to data...")
            ids = np.array(self.dataset.get_ids()).reshape(-1, 1)
            print(f"Shape data before adding: {data.shape} | Shape IDs: {ids.shape}")
            data = np.concatenate([ids, data], axis=1)
            column_order.append("id_column")
            current_column_index += 1

        print(f"Final data shape: {data.shape}")
        print("Column order in the resulting dataset:")
        for index, column_name in enumerate(column_order):
            print(f"Column {index}: {column_name}")
    
        return data
        

    def plot_kmeans(self, n_clusters=3, attribute='Class', embedding=False, attention_weights=False):
        """Apply K-Means clustering and visualize results."""

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
        """Plot a heatmap of the correlation between features."""
        data = self.select_data(embedding, attention_weights)
        corr = np.corrcoef(data, rowvar=False)
        plt.figure(figsize=(16, 10))
        sns.heatmap(corr, annot=False, cmap='coolwarm')
        plt.title('Correlation Heatmap')
        plt.show()
