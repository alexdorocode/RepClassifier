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
        #self.sequence_embeddings = cfg.sequence_embeddings
        #self.go_embeddings = cfg.go_embeddings

        self.paths = {
            "embedding_sequence_paths": cfg.paths.embedding_sequence_paths,
            "autoencoder_paths": cfg.paths.autoencoder_paths,
            "autoencoded_seq_embeddings": cfg.paths.autoencoded_seq_embeddings,
            "autoencoded_go_embeddings": cfg.paths.autoencoded_go_embeddings
        }

        self.experiment_definition = {
            "experiment_name": cfg.experiment_definition.experiment_name,
            "label_col": cfg.experiment_definition.label_col,
            "features_col": cfg.experiment_definition.features_col,
            "sequence_embeddings": cfg.experiment_definition.sequence_embeddings,
            "go_embeddings": cfg.experiment_definition.go_embeddings,
            "organism_discrimination_strategy": cfg.experiment_definition.organism_discrimination_strategy,
        }