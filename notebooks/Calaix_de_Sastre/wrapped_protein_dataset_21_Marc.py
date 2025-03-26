from project_root.dataset.protein_dataset import ProteinDataset
from project_root.utils.feature_processor import pad_attention_weights
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # type: ignore
from sklearn.decomposition import PCA  # type: ignore
from sklearn.manifold import TSNE  # type: ignore
from sklearn.cluster import KMeans  # type: ignore
from sklearn.random_projection import SparseRandomProjection  # type: ignore
from tqdm import tqdm  # type: ignore


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
        # Convert embeddings & attention weights to NumPy arrays
        embeddings_array = np.array(self.dataset.embeddings)  # Convert dict values to array

        # Flatten and pad attention weights to ensure consistent shape
        flattened_attention_weights = (self.dataset.attention_weights)
        attention_weights_array = np.array(pad_attention_weights(flattened_attention_weights))

        # Apply random projection to reduce dimensionality if needed
        if random_projection_dim < attention_weights_array.shape[1]:
            print(f"Applying random projection to reduce attention weights from {attention_weights_array.shape[1]} to {random_projection_dim} dimensions...")
            transformer = SparseRandomProjection(n_components=random_projection_dim)
            attention_weights_array = transformer.fit_transform(attention_weights_array)

        # Apply dimensionality reduction if specified
        print(f"Applying dimensionality reduction using {reduce_method}...")
        if reduce_method:
            if reduce_method == 'pca':
                print("Applying PCA reduction to embeddings")
                reduced_embeddings = self.pca_embeddings_reduction(embeddings_array, method=pca_method, threshold=threshold)
                print("Applying PCA reduction to attention weights")
                reduced_attention_weights = self.pca_attention_weights_reduction(attention_weights_array, method=pca_method, threshold=threshold)
            elif reduce_method == 'tsne':
                print("Applying t-SNE reduction to embeddings")
                perplexity = min(30, len(embeddings_array) - 1)
                reduced_embeddings = TSNE(n_components=2, perplexity=perplexity, random_state=42).fit_transform(embeddings_array)
                print("Applying t-SNE reduction to attention weights")
                reduced_attention_weights = TSNE(n_components=2, perplexity=perplexity, random_state=42).fit_transform(attention_weights_array)
            else:
                raise ValueError("Invalid reduce_method. Use 'pca' or 'tsne'.")

            self.embeddings = reduced_embeddings
            self.attention_weights = reduced_attention_weights
            
            # Convert back to dictionary format
            # self.embeddings = {key: reduced_embeddings[i] for i, key in enumerate(self.dataset.ids)}
            # self.attention_weights = {key: reduced_attention_weights[i] for i, key in enumerate(self.dataset.ids)}

        else:
            self.embeddings = embeddings_array
            self.attention_weights = attention_weights_array

        self.combined_embeddings_and_attention = np.concatenate([np.array(self.embeddings), np.array(self.attention_weights)], axis=1)

    # ========================= #
    #  PCA-BASED REDUCTION      #
    # ========================= #

    def get_best_pca_components(self, data, method='threshold', threshold=0.95):
        """Determine the optimal number of PCA components using variance threshold or derivative method."""
        pca = PCA().fit(data)
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

        if method == 'threshold':
            components = np.argmax(cumulative_variance >= threshold) + 1
            print(f"The number of components required to explain the variance using the threshold method ({threshold}) is:")
            print(components)
            return components
        elif method == 'derivative':
            components = np.arange(1, len(cumulative_variance) + 1)
            print("The number of components required to explain the variance using the derivative method is:")
            print(components)
            return components
        else:
            raise ValueError("Invalid method. Use 'threshold' or 'derivative'.")

    def apply_pca(self, data, method='custom', pca_components=100, threshold=0.95):
        """Apply PCA reduction with dynamically determined or fixed component count."""
        best_n_components = (
            min(pca_components, len(data))
            if method == 'custom'
            else self.get_best_pca_components(data, method, threshold=threshold)
        )

        return PCA(n_components=best_n_components).fit_transform(data)

    def pca_attention_weights_reduction(self, attention_weights_array, method='derivative', pca_components=100, threshold=0.95):
        """Reduce attention weights using PCA."""
        return self.apply_pca(attention_weights_array, method, pca_components, threshold)

    def pca_embeddings_reduction(self, embeddings_array, method='derivative', pca_components=100, threshold=0.95):
        """Reduce embeddings using PCA."""
        return self.apply_pca(embeddings_array, method, pca_components, threshold)

    def get_pca(self, embedding=False, attention_weights=False):
        """Plot PCA variance explained by each component."""

        if embedding and not attention_weights:
            data = self.embeddings
        elif attention_weights and not embedding:
            data = self.attention_weights
        elif embedding and attention_weights:
            data = self.combined_embeddings_and_attention
        else:
            raise ValueError("At least one of 'embedding' or 'attention_weights' must be True.")

        pca = PCA().fit(data)
        return pca

    # ========================= #
    #  DATA VISUALIZATION       #
    # ========================= #

    def plot_pca_variance(self, embeddings=False, attention_weights=False, thresholds=[0.95], show_second_derivative=False):
        """Plot PCA variance explained by each component with multiple thresholds and optional second derivative."""
    
        if embeddings and not attention_weights:
            data = self.embeddings
            text = 'Embeddings'
        elif attention_weights and not embeddings:
            data = self.attention_weights
            text = 'Attention Weights'
        elif embeddings and attention_weights:
            data = self.combined_embeddings_and_attention
            text = 'Combined Embeddings and Attention Weights'
        else:
            raise ValueError("At least one of 'embedding' or 'attention_weights' must be True.")
    
        pca = PCA().fit(data)
        pca_variance = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(pca_variance)
    
        plt.figure(figsize=(16, 10))
        plt.plot(cumulative_variance, label='Cumulative Variance')
    
        # Plot thresholds and their corresponding components
        for threshold in thresholds:
            n_components = np.argmax(cumulative_variance >= threshold) + 1
            plt.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold: {threshold}')
            plt.axvline(x=n_components - 1, color='b', linestyle='--', label=f'Components: {n_components} for {threshold}')
            plt.scatter(n_components - 1, cumulative_variance[n_components - 1], color='b')
            plt.text(n_components - 1, cumulative_variance[n_components - 1], f'{n_components}', fontsize=12, ha='right')
    
        # Optionally plot the second derivative
        if show_second_derivative:
            second_derivative = np.diff(np.diff(cumulative_variance))
            plt.plot(second_derivative, label='Second Derivative', linestyle='--')
    
        plt.xlabel('Number of Components')
        plt.ylabel('Variance (%)')
        plt.title(f'PCA Variance Explained by Components of {text}')
        plt.legend()
        plt.show()

    # ========================= #
    #  CLUSTERING               #
    # ========================= #

    def plot_kmeans(self, n_clusters=3, attribute='Class', embedding = False, attention_weights=False):
        """Apply K-Means clustering and visualize results."""
        
        if embedding and not attention_weights:
            data = self.embeddings
        elif attention_weights and not embedding:
            data = self.attention_weights
        elif embedding and attention_weights:
            data = self.combined_embeddings_and_attention
        else:
            raise ValueError("At least one of 'embedding' or 'attention_weights' must be True.")
    
        kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(data)
        clusters = kmeans.labels_
    
        plt.figure(figsize=(16, 10))
        sns.scatterplot(
            x=data[:, 0],
            y=data[:, 1],
            hue=clusters,
            style=self.dataset[attribute],
            palette=sns.color_palette("hsv", n_clusters),
            legend="full",
            alpha=0.7
        )
        plt.title(f'K-means clustering with {n_clusters} clusters')
        plt.show()