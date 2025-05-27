import os

# dataset/dataset_config.py
class DatasetConfigReader:
    def __init__(self, cfg):

        # Initialize configuration parameters
        for key, value in cfg.items():
            print(f"Setting config parameter: {key} = {value}")

        self.root = cfg.dataset.root_dir

        if cfg.dataset.unified_dataset is not None:
            self.file = cfg.dataset.unified_dataset.file
            self.id_col = cfg.dataset.unified_dataset.id_col
            self.label_col = cfg.dataset.unified_dataset.label_col
            self.organism_col = cfg.dataset.unified_dataset.organism_col
            self.sequence_col = cfg.dataset.unified_dataset.sequence_col
            self.metrics_col = cfg.dataset.unified_dataset.metrics_col

        self.metrics_to_use = cfg.metrics_to_use
        self.sequence_embeddings = cfg.sequence_embeddings
        self.go_embeddings = cfg.go_embeddings

        self.paths = {
            "embedding_sequence_paths": cfg.paths.embedding_sequence_paths,
            "autoencoder_paths": cfg.paths.autoencoder_paths,
            "autoencoded_seq_embeddings": cfg.paths.autoencoded_seq_embeddings,
            "autoencoded_go_embeddings": cfg.paths.autoencoded_go_embeddings
        }