import matplotlib.pyplot as plt
import numpy as np

class PCAExplainability:
    """Provides tools to interpret PCA results."""
    
    @staticmethod
    def plot_variance_explained(pca, title = 'PCA Variance Explained by Components', threshold=0.95):
        """
        Plots the cumulative explained variance of PCA components.
        
        The plot shows the cumulative explained variance as more components are added. 
        This helps determine how many components are required to explain a certain 
        amount of variance in the dataset.

        Parameters:
        pca (sklearn.decomposition.PCA): Fitted PCA object with explained_variance_ratio_.
        """

        # Find the number of components where the cumulative variance meets the threshold
        n_components = np.argmax(pca.explained_variance_ratio_ >= threshold) + 1
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        text = f'{len(pca.explained_variance_ratio_)} components'

        plt.figure(figsize=(16, 10))
        plt.plot(cumulative_variance, label='Cumulative Variance')
        plt.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold: {threshold}')
        plt.axvline(x=n_components - 1, color='b', linestyle='--', label=f'Components: {n_components}')
        plt.scatter(n_components - 1, cumulative_variance[n_components - 1], color='b')
        plt.text(n_components - 1, cumulative_variance[n_components - 1], f'{n_components}', fontsize=12, ha='right')
        plt.xlabel('Number of Components')
        plt.ylabel('Variance (%)')
        plt.title(f'{title} (Threshold: {threshold})')
        plt.legend()
        plt.show()

    @staticmethod
    def plot_scree(pca):
        """
        Plots the scree plot for PCA.
        
        The scree plot visualizes the eigenvalues of each principal component, 
        which indicates the amount of variance captured by each component.
        The "elbow" of the plot can help determine the number of components 
        to retain.

        Parameters:
        pca (sklearn.decomposition.PCA): Fitted PCA object with components_ and explained_variance_ratio_.
        """
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, marker='o')
        plt.xlabel('Principal Component')
        plt.ylabel('Variance Explained')
        plt.title('PCA Scree Plot')
        plt.grid()
        plt.show()

    @staticmethod
    def get_loadings(pca):
        """
        Retrieves the loadings (eigenvectors) for the principal components.
        
        The loadings are the coefficients of the original features for each 
        principal component. A high absolute value of a loading indicates that 
        a feature has a strong contribution to the corresponding component.

        Parameters:
        pca (sklearn.decomposition.PCA): Fitted PCA object with components_.
        
        Returns:
        np.ndarray: The loadings for each principal component.
        """
        return pca.components_

    @staticmethod
    def plot_feature_importance(pca, feature_names=None):
        """
        Plots the feature importance based on PCA loadings.
        
        This plot shows the absolute value of the loadings for each feature 
        across all components. Features with high loadings on the components 
        contribute more to the variance explained by the PCA.

        Parameters:
        pca (sklearn.decomposition.PCA): Fitted PCA object with components_.
        feature_names (list, optional): List of feature names. If None, the 
                                        features will be labeled as 'Feature 1', 'Feature 2', etc.
        """
        loadings = pca.components_
        num_features = loadings.shape[1]
        feature_names = feature_names if feature_names is not None else [f"Feature {i+1}" for i in range(num_features)]
        
        # Plot absolute values of loadings for each feature
        plt.figure(figsize=(10, 6))
        for i, feature in enumerate(feature_names):
            plt.plot(np.abs(loadings[:, i]), label=feature)
        plt.xlabel('Principal Component')
        plt.ylabel('Absolute Loading Value')
        plt.title('Feature Importance based on PCA Loadings')
        plt.legend(loc='best')
        plt.grid()
        plt.show()

    @staticmethod
    def get_variance_contribution(pca):
        """
        Returns the variance contribution of each component.
        
        This function calculates the proportion of total variance explained by each 
        principal component based on its eigenvalue, as well as the cumulative contribution.

        Parameters:
        pca (sklearn.decomposition.PCA): Fitted PCA object with explained_variance_ratio_.
        
        Returns:
        dict: A dictionary containing variance contributions for each component and cumulative variance.
        """
        explained_variance = pca.explained_variance_
        total_variance = np.sum(explained_variance)
        variance_contribution = explained_variance / total_variance
        cumulative_variance = np.cumsum(variance_contribution)
        
        return {
            "variance_contribution": variance_contribution,
            "cumulative_variance": cumulative_variance
        }
    
    @staticmethod
    def plot_variance_contribution(pca):
        """
        Plots the variance contribution of each principal component.
        
        This plot shows how much variance each principal component contributes 
        to the overall dataset, allowing for insight into the importance of each component.

        Parameters:
        pca (sklearn.decomposition.PCA): Fitted PCA object with explained_variance_.
        """
        variance_contribution = PCAExplainability.get_variance_contribution(pca)['variance_contribution']
        plt.figure(figsize=(8, 5))
        plt.bar(range(1, len(variance_contribution) + 1), variance_contribution)
        plt.xlabel('Principal Component')
        plt.ylabel('Variance Contribution')
        plt.title('PCA Variance Contribution per Component')
        plt.grid()
        plt.show()
