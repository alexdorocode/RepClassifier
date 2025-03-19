from project_root.dataset.protein_dataset import ProteinDataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # type: ignore
from sklearn.decomposition import PCA  # type: ignore
from sklearn.manifold import TSNE  # type: ignore
from sklearn.cluster import KMeans  # type: ignore


class WrappedProteinDataset(ProteinDataset):
    def __init__(self, dataset, reduce_method='pca', **kwargs):
        """
        Initializes WrappedProteinDataset with optional dimensionality reduction.

        Args:
            dataset (ProteinDataset): A pre-existing instance of ProteinDataset.
            reduce_method (str): Type of dimensionality reduction to apply ('pca' or 'tsne').
        """
        self.dataset = dataset  # Store original dataset reference

        # Convert embeddings & attention weights to NumPy arrays
        embeddings_array = np.array(list(self.dataset.embeddings.values()))  # Convert dict values to array
        attention_weights_array = np.array([attn.flatten() for attn in self.dataset.attention_weights.values()])

        # Apply dimensionality reduction if specified
        if reduce_method:
            if reduce_method == 'pca':
                reduced_embeddings = self.pca_embeddings_reduction(embeddings_array)
                reduced_attention_weights = self.pca_attention_weights_reduction(attention_weights_array)
            elif reduce_method == 'tsne':
                perplexity = min(30, len(embeddings_array) - 1)
                reduced_embeddings = TSNE(n_components=2, perplexity=perplexity, random_state=42).fit_transform(embeddings_array)
                reduced_attention_weights = TSNE(n_components=2, perplexity=perplexity, random_state=42).fit_transform(attention_weights_array)
            else:
                raise ValueError("Invalid reduce_method. Use 'pca' or 'tsne'.")

            # Convert back to dictionary format
            self.embeddings = {key: reduced_embeddings[i] for i, key in enumerate(self.dataset.ids)}
            self.attention_weights = {key: reduced_attention_weights[i] for i, key in enumerate(self.dataset.ids)}

    # ========================= #
    #  PCA-BASED REDUCTION      #
    # ========================= #

    def get_best_pca_components(self, data, method='threshold', threshold=0.95):
        """Determine the optimal number of PCA components using variance threshold or derivative method."""
        pca = PCA().fit(data)
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

        if method == 'threshold':
            return np.argmax(cumulative_variance >= threshold) + 1
        elif method == 'derivative':
            return np.argmax(np.diff(cumulative_variance)) + 1
        else:
            raise ValueError("Invalid method. Use 'threshold' or 'derivative'.")

    def apply_pca(self, data, method='derivative', pca_components=100):
        """Apply PCA reduction with dynamically determined or fixed component count."""
        best_n_components = (
            min(pca_components, data.shape[1])
            if method == 'custom'
            else self.get_best_pca_components(data, method)
        )

        return PCA(n_components=best_n_components).fit_transform(data)

    def pca_attention_weights_reduction(self, attention_weights_array, method='derivative', pca_components=100):
        """Reduce attention weights using PCA."""
        return self.apply_pca(attention_weights_array, method, pca_components)

    def pca_embeddings_reduction(self, embeddings_array, method='derivative', pca_components=100):
        """Reduce embeddings using PCA."""
        return self.apply_pca(embeddings_array, method, pca_components)

    def combine_by_pca_trimming(self, pca_components=100):
        """Combine embeddings and reduced attention weights."""
        reduced_attention_weights = self.pca_attention_weights_reduction(
            self.dataset.attention_weights, method='custom', pca_components=pca_components
        )
        return np.concatenate([np.array(list(self.embeddings.values())), reduced_attention_weights], axis=1)

    # ========================= #
    #  DATA VISUALIZATION       #
    # ========================= #

    def plot_embedding_reduction(self, method='tsne', attribute='Class', combined=False):
        """Plot dimensionality reduction results using PCA or t-SNE."""
        data = (
            self.combine_by_pca_trimming()
            if combined
            else np.array(list(self.embeddings.values()))
        )

        # Ensure perplexity is valid for t-SNE
        if method == 'tsne':
            perplexity = min(30, data.shape[0] - 1)
            reduced_data = TSNE(n_components=2, perplexity=perplexity, random_state=42).fit_transform(data)
        elif method == 'pca':
            reduced_data = PCA(n_components=2).fit_transform(data)
        else:
            raise ValueError("Invalid method. Use 'pca' or 'tsne'.")

        plt.figure(figsize=(16, 10))
        sns.scatterplot(
            x=reduced_data[:, 0],
            y=reduced_data[:, 1],
            hue=self.dataset.dataframe[attribute],
            palette=sns.color_palette("hsv", len(self.dataset.dataframe[attribute].unique())),
            legend="full",
            alpha=0.3
        )
        plt.title(f'{method.upper()} plot for {attribute}')
        plt.show()

    def plot_pca_variance(self, combined=False):
        """Plot the explained variance of PCA components."""
        data = self.combine_by_pca_trimming() if combined else np.array(list(self.embeddings.values()))

        pca = PCA().fit(data)
        explained_variance = np.cumsum(pca.explained_variance_ratio_)

        plt.figure(figsize=(16, 10))
        plt.plot(explained_variance)
        plt.xlabel('Number of Components')
        plt.ylabel('Variance Explained')
        plt.title('PCA Variance Explained')
        plt.show()

    # ========================= #
    #  CLUSTERING               #
    # ========================= #

    def plot_kmeans(self, n_clusters=3, attribute='Class', combined=False):
        """Apply K-Means clustering and visualize results."""
        data = self.combine_by_pca_trimming() if combined else np.array(list(self.embeddings.values()))

        kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(data)
        clusters = kmeans.labels_

        plt.figure(figsize=(16, 10))
        sns.scatterplot(
            x=data[:, 0],
            y=data[:, 1],
            hue=clusters,
            palette=sns.color_palette("hsv", n_clusters),
            legend="full",
            alpha=0.3
        )
        plt.title(f'K-means clustering with {n_clusters} clusters')
        plt.show()
