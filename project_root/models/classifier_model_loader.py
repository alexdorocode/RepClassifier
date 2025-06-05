import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
import lightgbm as lgb
from project_root.models.mlp_protein_classifier import MLPProteinClassifier

class ClassifierModelLoader:
    def __init__(self, model_type, input_size=None, output_size=None, device='cpu', model_params=None):
        """
        Args:
            model_type (str): One of ['logistic', 'svm', 'xgboost', 'lightgbm', 'knn', 'random_forest', 'mlp']
            input_size (int): Input feature size (needed for MLP)
            output_size (int): Number of output classes (needed for MLP)
            device (str): 'cpu' or 'cuda'
            model_params (dict): Model-specific parameters
        """
        self.model_type = model_type.lower()
        self.input_size = input_size
        self.output_size = output_size
        self.device = device
        self.model_params = model_params if model_params is not None else {}
        self.model = self._initialize_model()

    def _initialize_model(self):
        if self.model_type == 'logistic':
            return LogisticRegression(**self.model_params)

        elif self.model_type == 'svm':
            return SVC(**self.model_params)

        elif self.model_type == 'xgboost':
            return xgb.XGBClassifier(**self.model_params)

        elif self.model_type == 'lightgbm':
            return lgb.LGBMClassifier(**self.model_params)

        elif self.model_type == 'knn':
            return KNeighborsClassifier(**self.model_params)

        elif self.model_type == 'random_forest':
            return RandomForestClassifier(**self.model_params)

        elif self.model_type == 'mlp':
            # Extract MLP-specific parameters with defaults
            num_hidden_layers = self.model_params.get('num_hidden_layers', 2)
            dropout_rate = self.model_params.get('dropout_rate', 0.1)
            hidden_layers_mode = self.model_params.get('hidden_layers_mode', 'quadratic_increase')
            custom_hidden_layers = self.model_params.get('custom_hidden_layers', None)
            activation_function = self.model_params.get('activation_function', 'ReLU')
            use_batch_norm = self.model_params.get('use_batch_norm', False)
            output_activation = self.model_params.get('output_activation', None)
            initialization = self.model_params.get('initialization', None)

            if self.input_size is None or self.output_size is None:
                raise ValueError("MLP requires input_size and output_size to be specified.")

            return MLPProteinClassifier(
                device=self.device,
                input_size=self.input_size,
                output_size=self.output_size,
                num_hidden_layers=num_hidden_layers,
                dropout_rate=dropout_rate,
                hidden_layers_mode=hidden_layers_mode,
                custom_hidden_layers=custom_hidden_layers,
                activation_function=activation_function,
                use_batch_norm=use_batch_norm,
                output_activation=output_activation,
                initialization=initialization
            )
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")

    def get_model(self):
        return self.model
