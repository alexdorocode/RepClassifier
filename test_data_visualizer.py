import numpy as np
import pytest
from sklearn.decomposition import PCA
from project_root.utils.visualization import DataVisualizer

# Generate dummy 2D dataset for PCA and clustering
data = np.random.rand(50, 10)
pca_model = PCA(n_components=5).fit(data)
reduced_data = pca_model.transform(data)

# ---- Test Setup ----
@pytest.fixture(scope="module")
def pca():
    return pca_model

@pytest.fixture(scope="module")
def pca_data():
    return reduced_data

# ---- Tests ----
def test_plot_variance_explained(pca):
    DataVisualizer.plot_variance_explained(pca, threshold=0.8)  # Should produce a plot without error

def test_plot_scree(pca):
    DataVisualizer.plot_scree(pca)  # Should produce a scree plot without error

def test_get_loadings(pca):
    loadings = DataVisualizer.get_loadings(pca)
    assert loadings.shape == pca.components_.shape

def test_plot_feature_importance(pca):
    DataVisualizer.plot_feature_importance(pca)  # Should produce plot with default feature names


def test_get_variance_contribution(pca):
    contrib = DataVisualizer.get_variance_contribution(pca)
    assert 'variance_contribution' in contrib
    assert 'cumulative_variance' in contrib
    assert len(contrib['variance_contribution']) == pca.n_components_

def test_plot_variance_contribution(pca):
    DataVisualizer.plot_variance_contribution(pca)

def test_plot_kmeans(pca_data):
    DataVisualizer.plot_kmeans(pca_data, n_clusters=3)  # Should produce clustering scatter plot

def test_plot_correlation_heatmap(pca_data):
    DataVisualizer.plot_correration_heatmap(pca_data)  # Should produce a heatmap plot
