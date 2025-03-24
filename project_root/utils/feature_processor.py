import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.random_projection import SparseRandomProjection

def process_embeddings_and_attention(
    embeddings_array,
    attention_weights_array,
    reduce_method=None,
    pca_method='threshold',
    threshold=0.95,
    random_projection_dim=1000,
    random_projection_method='global'
):
    """
    Applies optional random projection and dimensionality reduction to embeddings and attention weights.
    """

    if random_projection_dim < attention_weights_array.shape[1]:
        print(f"Applying random projection to reduce attention weights from {attention_weights_array.shape[1]} to {random_projection_dim} dimensions...")
        if random_projection_method == 'global':
            attention_weights_array = apply_random_projection_globaly(attention_weights_array, n_components=random_projection_dim)
        elif random_projection_method == 'by_prot':
            attention_weights_array = apply_random_projection_by_prot(attention_weights_array, n_components=random_projection_dim)
        else:
            raise ValueError(f"Unknown random_projection_method: {random_projection_method}")

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

    return reduced_embeddings, reduced_attention_weights



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


def apply_random_projection_globaly(data, n_components=1000):
    """
    Applies sparse random projection to reduce dimensionality.
    """
    transformer = SparseRandomProjection(n_components=n_components)
    return transformer.fit_transform(data)


def apply_random_projection_by_prot(data, n_components=1000):
    """
    Applies sparse random projection to each protein individually.

    Args:
        data (np.ndarray): Array of shape (n_proteins, seq_len, feature_dim)
        n_components (int): Target dimensionality

    Returns:
        np.ndarray: Array of shape (n_proteins, seq_len, n_components)
    """
    """
    return np.array([
        SparseRandomProjection(n_components=n_components).fit_transform(prot)
        for prot in tqdm(data, desc="Applying projection per protein")
    ])
    """
    RuntimeError("Not implemented yet")
    return None


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
