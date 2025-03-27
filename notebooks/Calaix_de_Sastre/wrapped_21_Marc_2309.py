from project_root.dataset.representation_dataset import ProteinDataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # type: ignore
from sklearn.cluster import KMeans  # type: ignore

from project_root.utils.feature_processor import (
    flatten_attention_weights,
    pad_attention_weights,
    apply_random_projection,
    apply_pca,
    apply_tsne,
)


class WrappedProteinDataset(ProteinDataset):
    def __init__(self, dataset, reduce_method=None, pca_method='threshold', random_projection_dim=1000, threshold=0.95):
        """
        Initializes WrappedProteinDataset with optional dimensionality reduction.

        Args:
            dataset (ProteinDataset): A pre-existing instance of ProteinDataset.
            reduce_method (str): Type of dimensionality reduction to apply ('pca' or 'tsne').
            random_projection_dim (int): Target dimensionality for random projection before PCA.
        """
        self.dataset = dataset  # Store original dataset reference

        print("Converting embeddings and attention weights to NumPy arrays...")
        embeddings_array = np.array(self.dataset.embeddings)
        flattened_attention_weights = flatten_attention_weights(self.dataset.attention_weights)
        padded_attention_weights = pad_attention_weights(flattened_attention_weights)

        if random_projection_dim < padded_attention_weights.shape[1]:
            print(f"Applying random projection to reduce attention weights from {padded_attention_weights.shape[1]} to {random_projection_dim} dimensions...")
            attention_weights_array = apply_random_projection(padded_attention_weights, n_components=random_projection_dim)
        else:
            attention_weights_array = padded_attention_weights

        print(f"Applying dimensionality reduction using {reduce_method}...")
        if reduce_method == 'pca':
            reduced_embeddings = apply_pca(embeddings_array, method=pca_method, threshold=threshold)
            reduced_attention_weights = apply_pca(attention_weights_array, method=pca_method, threshold=threshold)
        elif reduce_method == 'tsne':
            perplexity = min(30, len(embeddings_array) - 1)
            reduced_embeddings = apply_tsne(embeddings_array, perplexity=perplexity)
            reduced_attention_weights = apply_tsne(attention_weights_array, perplexity=perplexity)
        else:
            reduced_embeddings = embeddings_array
            reduced_attention_weights = attention_weights_array

        self.embeddings = reduced_embeddings
        self.attention_weights = reduced_attention_weights
        self.combined_embeddings_and_attention = np.concatenate([self.embeddings, self.attention_weights], axis=1)

    def select_data(self, embedding=True, attention_weights=True):
        """Select data for visualization."""
        if embedding and attention_weights:
            return self.combined_embeddings_and_attention
        elif embedding:
            return self.embeddings
        elif attention_weights:
            return self.attention_weights
        else:
            raise ValueError("At least one of 'embedding' or 'attention_weights' must be True.")

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