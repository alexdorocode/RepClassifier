import matplotlib.pyplot as plt
import numpy as np

class PCAExplainability:
    """Provides tools to interpret PCA results."""
    
    @staticmethod
    def plot_variance_explained(pca):
        """Plots the explained variance of PCA components."""
        plt.figure(figsize=(8, 5))
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel('Number of Components')
        plt.ylabel('Explained Variance')
        plt.title('PCA Explained Variance')
        plt.grid()
        plt.show()
