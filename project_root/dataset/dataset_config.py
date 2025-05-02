import os

# dataset/dataset_config.py
class DatasetConfigReader:
    def __init__(self, cfg):
        self.root = cfg.dataset.root_dir
        self.paths = {
            "uniprot_ids": os.path.join(self.root, cfg.dataset.raw.uniprot_ids),
            "labels": os.path.join(self.root, cfg.dataset.raw.labels),
            "sequences": os.path.join(self.root, cfg.dataset.raw.sequences),
            "tokens_3di": os.path.join(self.root, cfg.dataset.raw.tokens_3di),
        }
        self.embeddings = {
            "esm": cfg.embeddings.esm,
            "prott5": cfg.embeddings.prott5,
            "prostt5": cfg.embeddings.prostt5,
            "geokg": cfg.embeddings.geokg,
        }
        self.emb_dir = cfg.embeddings.emb_dir
