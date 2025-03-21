import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.random_projection import SparseRandomProjection


def flatten_attention_weights(attention_weights):
    """
    Flattens a list of attention weight matrices.
    """
    return [attn.flatten() for attn in attention_weights]


def pad_attention_weights(flattened_attention_weights):
    """
    Pads flattened attention weights to ensure uniform length.
    """
    max_length = max(len(attn) for attn in flattened_attention_weights)
    return np.array([
        np.pad(attn, (0, max_length - len(attn)), 'constant')
        for attn in flattened_attention_weights
    ])


def apply_random_projection(data, n_components=1000):
    """
    Applies sparse random projection to reduce dimensionality.
    """
    transformer = SparseRandomProjection(n_components=n_components)
    return transformer.fit_transform(data)


def get_best_pca_components(data, method='threshold', threshold=0.95):
    """
    Determine optimal number of PCA components.
    """
    pca = PCA().fit(data)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

    if method == 'threshold':
        return np.argmax(cumulative_variance >= threshold) + 1
    elif method == 'derivative':
        return np.argmax(np.diff(cumulative_variance)) + 1
    else:
        raise ValueError("Invalid method. Use 'threshold' or 'derivative'.")


def apply_pca(data, method='threshold', pca_components=100, threshold=0.95):
    """
    Apply PCA reduction using threshold or fixed component count.
    """
    best_n_components = (
        min(pca_components, data.shape[1])
        if method == 'custom'
        else get_best_pca_components(data, method, threshold)
    )
    return PCA(n_components=best_n_components).fit_transform(data)


def apply_tsne(data, n_components=2, perplexity=30, random_state=42):
    """
    Apply t-SNE for visualization.
    """
    return TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state).fit_transform(data) 
