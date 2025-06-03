import shap
import torch
from captum.attr import IntegratedGradients, Saliency
import numpy as np
import pandas as pd

class ExplainabilityModule:
    def __init__(self, model, model_type, device='cpu'):
        self.model = model
        self.model_type = model_type.lower()
        self.device = device

    def explain(self, X, feature_names=None, target=None):
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
        explainer = shap.LinearExplainer(self.model, X)
        shap_values = explainer.shap_values(X)
        return self._build_summary_df(shap_values, feature_names, len(X))

    def _explain_xgboost(self, X, feature_names):
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X)
        return self._build_summary_df(shap_values, feature_names, len(X))

    def _explain_mlp(self, X, feature_names, target=None):
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
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X)
        return self._build_summary_df(shap_values, feature_names, len(X))

    def _explain_knn(self, X, feature_names):
        explainer = shap.KernelExplainer(self.model.predict_proba, shap.sample(X, 100))
        shap_values = explainer.shap_values(X)
        return self._build_summary_df(shap_values, feature_names, len(X))