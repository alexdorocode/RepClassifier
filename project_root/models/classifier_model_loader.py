# project_root/models/classifier_model.py

import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xgb
import lightgbm as lgb
from project_root.models.mlp_protein_classifier import MLPProteinClassifier

class ClassifierModelLoader:
    def __init__(self, model_type, input_size, output_size, device='cpu', model_params=None):
        """
        Args:
            model_type (str): 'logistic', 'svm', 'xgboost', 'lightgbm', or 'mlp'
            input_size (int): Input feature size
            output_size (int): Number of output classes
            device (str): 'cpu' or 'cuda'
            model_params (dict): Additional model-specific parameters
        """
        self.model_type = model_type.lower()
        self.input_size = input_size
        self.output_size = output_size
        self.device = device
        self.model_params = model_params if model_params is not None else {}
        self.model = self._initialize_model()

    def _initialize_model(self):
        if self.model_type == 'logistic':
            return LogisticRegression(max_iter=self.model_params.get('max_iter', 1000))
        elif self.model_type == 'svm':
            return SVC(kernel=self.model_params.get('kernel', 'rbf'), probability=True)
        elif self.model_type == 'xgboost':
            return xgb.XGBClassifier(**self.model_params)
        elif self.model_type == 'mlp':
            num_hidden_layers = self.model_params.get('num_hidden_layers', 2)
            dropout_rate = self.model_params.get('dropout_rate', 0.1)
            hidden_layers_mode = self.model_params.get('hidden_layers_mode', 'quadratic_increase')
            custom_hidden_layers = self.model_params.get('custom_hidden_layers', None)
            return MLPProteinClassifier(
                device=self.device,
                input_size=self.input_size,
                output_size=self.output_size,
                num_hidden_layers=num_hidden_layers,
                dropout_rate=dropout_rate,
                hidden_layers_mode=hidden_layers_mode,
                custom_hidden_layers=custom_hidden_layers
            )
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")

    def get_model(self):
        return self.model
