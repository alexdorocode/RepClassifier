import hydra
from omegaconf import DictConfig, OmegaConf
import os
import sys
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from project_root.models.classifier_model_loader import ClassifierModelLoader
from project_root.training.trainer_module import TrainerModule
from project_root.explainability.explainability_module import ExplainabilityModule
from project_root.training.tracker_module import TrackerModule
from project_root.dataset.dataset_config import DatasetConfigReader
from project_root.dataset.dataset_handler import DatasetHandler

class ClassifierLauncher:
    def __init__(self, cfg: DictConfig, random_seed=None):
        self.cfg = cfg
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Initialize config reader to access parsed dictionaries
        self.config_reader = DatasetConfigReader(cfg)

        self.random_seed = random_seed
        self._set_all_seeds(self.random_seed)

        print(f"🔍 Setingg random seed to {self.random_seed} for reproducibility.")

        # Initialize dataset
        self._initialize_dataset()

        # Initialize model
        self._initialize_model()

        # Initialize tracker
        tracker_cfg = self.config_reader.tracker
        self.tracker = TrackerModule(
            project_name=tracker_cfg['project_name'],
            run_name=tracker_cfg['run_name'],
            offline=tracker_cfg.get('offline', False),
            config=self.config_reader,  # You can store the entire config_reader as a reference
            random_seed=self.random_seed,
            tags=['phase3_develop', 'classifier', f'{self.config_reader.model["type"]}']
            )
        

    def _initialize_dataset(self):
        handler = DatasetHandler(self.config_reader)
        self.classifier_dataset = handler.load_classifier_dataset()
        X, y = self.classifier_dataset.get_X_y()

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.config_reader.training['test_size_ratio'], random_state=self.random_seed, stratify=y
        )
        
    def _initialize_model(self):
        input_size = self.classifier_dataset.features.shape[1]
        output_size = len(torch.unique(self.classifier_dataset.labels))
        model_cfg = self.config_reader.model
        self.model_loader = ClassifierModelLoader(
            model_type=model_cfg['type'],
            input_size=input_size,
            output_size=output_size,
            device=self.device,
            model_params=model_cfg['params']
        )
        self.model = self.model_loader.get_model()

    def _set_all_seeds(self,seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def run(self):
        try:
            training_cfg = self.config_reader.training
            mlp_params = training_cfg.get('mlp_params', {})
            trainer = TrainerModule(
                model=self.model,
                model_type=self.config_reader.model['type'],
                device=self.device,
                learning_rate=mlp_params['learning_rate'] if 'learning_rate' in mlp_params else None,
                num_epochs=mlp_params['num_epochs'] if 'num_epochs' in mlp_params else None,
                cv_folds=training_cfg['cv_folds'],
                tracker=self.tracker,
                optimizer_name=mlp_params['optimizer'] if 'optimizer' in mlp_params else None,
                criterion_name=mlp_params['criterion'] if 'criterion' in mlp_params else None,
                early_stopping_patience=mlp_params['early_stopping_patience'] if 'early_stopping_patience' in mlp_params else None,
                batch_size=mlp_params['batch_size'] if 'batch_size' in mlp_params else None,
                random_seed=self.random_seed,
            )
            avg_acc, avg_f1, avg_precision, avg_recall, fold_metrics = trainer.cross_validate(self.X_train, self.y_train, self.classifier_dataset.balance_values)
            print(f"✅ Training done! Accuracy: {avg_acc:.2%}, F1: {avg_f1:.2%}, Precision: {avg_precision:.2%}, Recall: {avg_recall:.2%}")

            # Explainability (optional)
            if self.config_reader.explainability:
                explainer = ExplainabilityModule(self.model, self.config_reader.model['type'], device=self.device)
                explanation_df = explainer.explain(
                    self.classifier_dataset.features[:5], 
                    feature_names=self.classifier_dataset.feature_cols,
                    target=1  # or 0, depending on which class you want to explain
                )
                print(f"🧠 Explainability output:\n{explanation_df.head()}")

        finally:
            self.tracker.finish()

