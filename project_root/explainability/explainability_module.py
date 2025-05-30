# project_root/explainability/explainability_module.py

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
        """
        X: np.ndarray or torch.Tensor, input data for explanation
        feature_names: list of feature names
        """
        if self.model_type in ['logistic', 'svm']:
            return self._explain_linear(X, feature_names)
        elif self.model_type == 'xgboost':
            return self._explain_xgboost(X, feature_names)
        elif self.model_type == "mlp":
            return self._explain_mlp(X, feature_names, target=target)
        else:
            raise ValueError(f"Explainability not supported for model type: {self.model_type}")

    def _explain_linear(self, X, feature_names):
        # SHAP LinearExplainer or Coefficients
        explainer = shap.LinearExplainer(self.model, X)
        shap_values = explainer.shap_values(X)
        summary_df = pd.DataFrame(shap_values, columns=feature_names)
        summary_df['sample'] = range(len(X))
        return summary_df

    def _explain_xgboost(self, X, feature_names):
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X)
        summary_df = pd.DataFrame(shap_values, columns=feature_names)
        summary_df['sample'] = range(len(X))
        return summary_df

    def _explain_mlp(self, X, feature_names, target=None):
        # Convert numpy to torch tensor if needed
        if isinstance(X, np.ndarray):
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        else:
            X_tensor = X.to(self.device)

        X_tensor.requires_grad = True
        # Integrated Gradients
        ig = IntegratedGradients(self.model)
        attributions, _ = ig.attribute(X_tensor, target=target, return_convergence_delta=True)
        attributions = attributions.detach().cpu().numpy()

        # Optional: Also compute Saliency
        # saliency = Saliency(self.model)
        # saliency_attr = saliency.attribute(X_tensor).detach().cpu().numpy()

        summary_df = pd.DataFrame(attributions, columns=feature_names)
        summary_df['sample'] = range(len(X_tensor))
        return summary_df
