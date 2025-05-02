# dataset/raw_dataset.py

import pandas as pd
import numpy as np

class RawDataset:
    """
    Container for raw protein dataset elements:
    UniProt IDs, labels, sequences, 3Di tokens, and optional embeddings.
    """

    def __init__(self, raw_data: dict, embeddings: dict = None):
        """
        Initializes a RawDataset object.

        Args:
            raw_data (dict): Dictionary containing raw DataFrames keyed by 'uniprot_ids', 'labels', 'sequences', 'tokens_3di'.
            embeddings (dict, optional): Dictionary with keys ['esm', 'prott5', 'prostt5', 'geokg'], each mapping to npy-loaded dicts.
        """
        self.uniprot_ids = raw_data.get("uniprot_ids", pd.DataFrame())
        self.labels = raw_data.get("labels", pd.DataFrame())
        self.sequences = raw_data.get("sequences", pd.DataFrame())
        self.tokens_3di = raw_data.get("tokens_3di", pd.DataFrame())
        self.embeddings = embeddings if embeddings is not None else {}

    def __len__(self):
        return len(self.uniprot_ids)

    def summary(self):
        return {
            "num_ids": len(self.uniprot_ids),
            "has_labels": not self.labels.empty,
            "has_sequences": not self.sequences.empty,
            "has_3di_tokens": not self.tokens_3di.empty,
            "embeddings_loaded": list(self.embeddings.keys())
        }

    def get_attribute(self, name):
        mapping = {
            "aa_sequence": self.sequences,
            "labels": self.labels,
            "uniprot_ids": self.uniprot_ids,
            "3di_tokens": self.tokens_3di,
            "go_annotations": getattr(self, "go_annotations", None),
            "protT5_embedding": self.embeddings.get("prott5"),
            "prostT5_embedding": self.embeddings.get("prostt5"),
            "esm2_embedding": self.embeddings.get("esm"),
        }

        if name not in mapping or mapping[name] is None:
            raise KeyError(f"Attribute '{name}' not found in dataset.")

        return mapping[name]

