# Final commit – Master’s Thesis by Àlex Domínguez Roig

import os

class DatasetConfigReader:
    """
    Reads and organizes dataset and experiment configuration for downstream processing.

    :param cfg: Configuration object (e.g., from Hydra/OmegaConf)
    """

    def __init__(self, cfg):
        """
        Initialize the DatasetConfigReader.

        :param cfg: Configuration object with dataset, features, embeddings, and training info.
        """
        self.root = cfg.dataset.root_dir

        if cfg.dataset.unified_dataset is not None:
            self.file = cfg.dataset.unified_dataset.file
            self.id_col = cfg.dataset.unified_dataset.id_col
            self.label_col = cfg.dataset.unified_dataset.label_col
            self.organism_col = cfg.dataset.unified_dataset.organism_col
            self.sequence_col = cfg.dataset.unified_dataset.sequence_col
            self.metrics_col = cfg.dataset.unified_dataset.metrics_col

        self.features_to_process = cfg.features_to_process
        
        self.paths = {
            "embedding_sequence_paths": cfg.paths.embedding_sequence_paths,
            "autoencoder_paths": cfg.paths.autoencoder_paths,
            "autoencoded_seq_embeddings": cfg.paths.autoencoded_seq_embeddings,
            "autoencoded_go_embeddings": cfg.paths.autoencoded_go_embeddings
        }

        self.classifier_definition = {
            "label_col": cfg.classifier_definition.label_col,
            "features_col": cfg.classifier_definition.features_col,
            "sequence_embeddings": cfg.classifier_definition.sequence_embeddings,
            "go_embeddings": cfg.classifier_definition.go_embeddings,
            "balance_col": cfg.classifier_definition.balance_col,
            "organism_discrimination_strategy": cfg.classifier_definition.organism_discrimination_strategy,
            "production_mode": cfg.classifier_definition.production_mode,
        }

        self.model = {
            "type": cfg.classifier_definition.model.type,
            "params": cfg.classifier_definition.model.params
        }

        self.training = {
            "cv_folds": cfg.classifier_definition.training.cv_folds,
            "cross_val_balance": cfg.classifier_definition.training.cross_val_balance,
            "test_size_ratio": cfg.classifier_definition.training.test_size_ratio,
            "mlp_params": cfg.classifier_definition.training.mlp_params if 'mlp_params' in cfg.classifier_definition.training else {}
        }
        self.tracker = {
            "project_name": cfg.classifier_definition.tracker.project_name,
            "run_name": cfg.classifier_definition.tracker.run_name,
            "offline": cfg.classifier_definition.tracker.get('offline', False)
        }
        self.explainability = cfg.classifier_definition.explainability
        self.random_seed = cfg.classifier_definition.get('random_seed') if 'random_seed' in cfg.classifier_definition else None
        
    def get_as_dict(self):
        """
        Returns the configuration as a dictionary.

        :return: Dictionary of all relevant configuration fields.
        """
        return {
            "root": self.root,
            "file": self.file,
            "id_col": self.id_col,
            "label_col": self.label_col,
            "organism_col": self.organism_col,
            "sequence_col": self.sequence_col,
            "metrics_col": self.metrics_col,
            "features_to_process": self.features_to_process,
            "paths": self.paths,
            "classifier_definition": self.classifier_definition,
            "model": self.model,
            "training": self.training,
            "tracker": self.tracker,
            "explainability": self.explainability,
            "random_seed": self.random_seed
        }
    
    def get_random_seed(self):
        """
        Returns the random seed from the configuration.

        :return: Random seed (int or None)
        """
        return  self.random_seed