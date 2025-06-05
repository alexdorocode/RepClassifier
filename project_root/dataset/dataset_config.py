import os

# dataset/dataset_config.py
class DatasetConfigReader:
    def __init__(self, cfg):

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
            "mlp_params": cfg.classifier_definition.training.mlp_params,
        }
        self.tracker = {
            "project_name": cfg.classifier_definition.tracker.project_name,
            "run_name": cfg.classifier_definition.tracker.run_name,
            "offline": cfg.classifier_definition.tracker.get('offline', False)
        }
        self.explainability = cfg.classifier_definition.explainability
        