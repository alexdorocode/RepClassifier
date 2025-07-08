# Final commit – Master’s Thesis by Àlex Domínguez Roig

import shap # type: ignore
import torch # type: ignore
from captum.attr import IntegratedGradients # type: ignore
import numpy as np # type: ignore
import pandas as pd # type: ignore

class ExplainabilityModule:
    """
    Unified explainability interface for different model types.
    Supports SHAP for tree/linear models, Integrated Gradients for MLP, and KernelExplainer for KNN.
    """

    def __init__(self, model, model_type, device='cpu'):
        """
        Initialize the ExplainabilityModule.

        :param model: Trained model object
        :param model_type: Model type string (e.g., 'mlp', 'logistic', 'svm', 'xgboost', 'random_forest', 'knn')
        :param device: Device to use ('cpu' or 'cuda')
        """
        self.model = model
        self.model_type = model_type.lower()
        self.device = device

    def explain(self, X, feature_names=None, target=None):
        """
        Compute feature attributions for the given input and model type.

        :param X: Input data (numpy array or torch tensor)
        :param feature_names: List of feature names
        :param target: Target class index (for MLP/IG)
        :return: DataFrame with feature attributions
        """
        if self.model_type in ['logistic', 'svm']:
            return self._explain_linear(X, feature_names)
        elif self.model_type == 'xgboost':
            return self._explain_xgboost(X, feature_names)
        elif self.model_type == "mlp":
            return self._explain_mlp(X, feature_names, target=target)
        elif self.model_type == 'random_forest':
            return self._explain_random_forest(X, feature_names)
        elif self.model_type == 'knn':
            return self._explain_knn(X, feature_names)
        else:
            raise ValueError(f"Explainability not supported for model type: {self.model_type}")

    def _build_summary_df(self, shap_values, feature_names, n_samples):
        """
        Build a summary DataFrame from SHAP or attribution values.

        :param shap_values: SHAP or attribution values (array or list)
        :param feature_names: List of feature names
        :param n_samples: Number of samples
        :return: DataFrame with attributions per feature and sample
        """
        # Handle list (e.g., for binary classification with KernelExplainer)
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        # Reduce 3D arrays to 2D (pick class 1 or mean across classes)
        if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]  # or .mean(axis=2)
        # If 1D, reshape to (n_samples, 1)
        if isinstance(shap_values, np.ndarray) and shap_values.ndim == 1:
            shap_values = shap_values.reshape(-1, 1)
        summary_df = pd.DataFrame(shap_values, columns=feature_names)
        summary_df['sample'] = range(n_samples)
        return summary_df

    def _explain_linear(self, X, feature_names):
        """
        Explain linear models (logistic regression, SVM) using SHAP LinearExplainer.

        :param X: Input data
        :param feature_names: List of feature names
        :return: DataFrame with SHAP values
        """
        explainer = shap.LinearExplainer(self.model, X)
        shap_values = explainer.shap_values(X)
        return self._build_summary_df(shap_values, feature_names, len(X))

    def _explain_xgboost(self, X, feature_names):
        """
        Explain XGBoost models using SHAP TreeExplainer.

        :param X: Input data
        :param feature_names: List of feature names
        :return: DataFrame with SHAP values
        """
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X)
        return self._build_summary_df(shap_values, feature_names, len(X))

    def _explain_mlp(self, X, feature_names, target=None):
        """
        Explain MLP models using Integrated Gradients.

        :param X: Input data (numpy array or torch tensor)
        :param feature_names: List of feature names
        :param target: Target class index
        :return: DataFrame with attributions
        """
        if isinstance(X, np.ndarray):
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        else:
            X_tensor = X.to(self.device)
        X_tensor.requires_grad = True
        ig = IntegratedGradients(self.model)
        attributions, _ = ig.attribute(X_tensor, target=target, return_convergence_delta=True)
        attributions = attributions.detach().cpu().numpy()
        return self._build_summary_df(attributions, feature_names, len(X_tensor))

    def _explain_random_forest(self, X, feature_names):
        """
        Explain Random Forest models using SHAP TreeExplainer.

        :param X: Input data
        :param feature_names: List of feature names
        :return: DataFrame with SHAP values
        """
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X)
        return self._build_summary_df(shap_values, feature_names, len(X))

    def _explain_knn(self, X, feature_names):
        """
        Explain KNN models using SHAP KernelExplainer.

        :param X: Input data
        :param feature_names: List of feature names
        :return: DataFrame with SHAP values
        """
        explainer = shap.KernelExplainer(self.model.predict_proba, shap.sample(X, 100))
        shap_values = explainer.shap_values(X)
        return self._build_summary_df(shap_values, feature_names, len(X))