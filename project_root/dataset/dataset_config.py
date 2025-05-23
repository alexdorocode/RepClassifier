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

        self.embeddings = {
            "esm": cfg.embeddings.esm,
            "prott5": cfg.embeddings.prott5,
            "prostt5": cfg.embeddings.prostt5,
            "geokg": cfg.embeddings.geokg,
        }
        self.emb_dir = cfg.embeddings.emb_dir
