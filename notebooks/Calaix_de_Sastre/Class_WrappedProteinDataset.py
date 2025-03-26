from Class_ProteinDataset import ProteinDataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # type: ignore
from sklearn.decomposition import PCA # type: ignore
from sklearn.manifold import TSNE # type: ignore
from sklearn.cluster import KMeans # type: ignore

class WrappedProteinDataset(ProteinDataset):
    def __init__(self, dataset, reduce_method='pca'):
        super().__init__(dataset)

        if reduce_method == 'pca':
            self.reduce_pca()
        elif reduce_method == 'tsne':
            self.reduce_tsne()
        else:
            raise ValueError('Invalid reduce_method')
    
    def get_best_pca_components(self, data, method='threshold', threshold=0.95):
        """Get the best number of PCA components using a specified method."""

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

    def pca_attention_weights_reduction(self, method='derivative', pca_components=100):

        # Flatten attention weights
        flattened_attention_weights = [attn.flatten() for attn in self.attention_weights]
        
        # Ensure all flattened attention weights have the same shape
        max_length = max(len(attn) for attn in flattened_attention_weights)
        padded_attention_weights = [np.pad(attn, (0, max_length - len(attn)), 'constant') for attn in flattened_attention_weights]
        
        # Convert flattened attention weights to NumPy array
        flattened_attention_weights_array = np.array(padded_attention_weights)
        
        # Calcuclate the best number of PCA components
        if method == 'custom':
            # Use the provided number of PCA components
            best_n_components = min(pca_components, flattened_attention_weights_array.shape[1])
        else:
            # Calculate the best number of PCA components using the derivative method or threshold
            best_n_components = self.get_best_pca_components(flattened_attention_weights_array, method)
        
        # Apply PCA to the entire matrix of flattened attention weights
        pca = PCA(n_components=best_n_components)
        reduced_attention_weights = pca.fit_transform(flattened_attention_weights_array)

        return reduced_attention_weights

    def pca_embeddings_reduction(self, method='derivative', pca_components=100):
        """Reduce the dimensionality of embeddings using PCA."""
        
        # Convert embeddings to NumPy array
        embeddings_array = np.array(self.embeddings)

        if method == 'custom':
            # Use the provided number of PCA components
            best_n_components = min(pca_components, embeddings_array.shape[1])
        else:
            # Calculate the best number of PCA components using the derivative method
            best_n_components = self.get_best_pca_components(embeddings_array, method)
        
        # Apply PCA to the embeddings
        pca = PCA(n_components=best_n_components)
        reduced_embeddings = pca.fit_transform(embeddings_array)
        
        return reduced_embeddings

    def combine_by_pca_trimming(self, pca_components=100):
        """Combine embeddings and attention_weights by applying PCA trimming to attention_weights."""
        # Flatten attention weights
        flattened_attention_weights = [attn.flatten() for attn in self.attention_weights]
        
        # Ensure all flattened attention weights have the same shape
        max_length = max(len(attn) for attn in flattened_attention_weights)
        padded_attention_weights = [np.pad(attn, (0, max_length - len(attn)), 'constant') for attn in flattened_attention_weights]
        
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

    def plot_tsne(self, attribute='Class', combined=False):
        """Plot TSNE for embeddings, attention_weights, or both combined."""
        if combined:
            # Ensure embeddings and attention_weights have the same number of samples
            if len(self.embeddings) != len(self.attention_weights):
                raise ValueError("Embeddings and attention weights must have the same number of samples.")
            
            # Combine embeddings and attention_weights by applying PCA trimming
            data = self.combine_by_pca_trimming()
        else:
            data = np.array(self.embeddings)

        # Check the shape of the data
        print(f"Data shape: {data.shape}")

        # Set perplexity to a value less than the number of samples if needed
        perplexity = min(30, data.shape[0] - 1)

        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
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
            # Ensure embeddings and attention_weights have the same number of samples
            if len(self.embeddings) != len(self.attention_weights):
                raise ValueError("Embeddings and attention weights must have the same number of samples.")
            
            # Combine embeddings and attention_weights by applying PCA trimming
            data = self.combine_by_pca_trimming()
        else:
            data = np.array(self.embeddings)

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
            # Ensure embeddings and attention_weights have the same number of samples
            if len(self.embeddings) != len(self.attention_weights):
                raise ValueError("Embeddings and attention weights must have the same number of samples.")
            
            # Combine embeddings and attention_weights by applying PCA trimming
            data = self.combine_by_pca_trimming()
        else:
            data = np.array(self.embeddings)

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
