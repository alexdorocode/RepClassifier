import pandas as pd
import numpy as np

class RawDataset:
    """
    Container for raw protein dataset elements:
    UniProt IDs, labels, sequences, metrics, and optional embeddings.
    """

    def __init__(self, dataset: pd.DataFrame, 
                 id_col: str, label_col: str, 
                 organism_col: str, metrics_col: list, sequence_col: str #,
                 # embeddings: dict = None
                 ):
        """
        Initializes a RawDataset object.

        Args:
            dataset (pd.DataFrame): The raw dataset as a DataFrame.
            id_col (str): Column name for UniProt IDs.
            label_col (str): Column name for labels.
            organism_col (str): Column name for organism information.
            metrics_col (list): List of column names for metrics.
            embeddings (dict, optional): Dictionary with keys ['esm', 'prott5', 'prostt5', 'geokg'], each mapping to npy-loaded dicts.
        """
        self.dataset = dataset
        self.uniprot_ids = dataset[id_col]
        self.labels = dataset[label_col]
        self.organisms = dataset[organism_col]
        self.metrics = dataset[metrics_col] if metrics_col else pd.DataFrame()
        self.sequences = dataset[sequence_col]
        self.main_columns = {
            "id_col" : id_col,
            "label_col" : label_col,
            "organism_col" : organism_col,
            "metrics_col" : metrics_col,
            "sequence_col" : sequence_col
        }
        
        # self.embeddings = embeddings if embeddings is not None else {}

    def __len__(self):
        return len(self.uniprot_ids)

    def summary(self):
        return {
            "num_ids": len(self.uniprot_ids),
            "has_labels": not self.labels.empty,
            "has_organisms": not self.organisms.empty,
            "num_metrics": self.metrics.shape[1] if not self.metrics.empty else 0,
            # "embeddings_loaded": list(self.embeddings.keys())
            "length_biggest_than_3000": len(self.sequences[self.sequences.str.len() > 3000]),
            "length_biggest_than_2000": len(self.sequences[self.sequences.str.len() > 2000]),
            "length_biggest_than_1000": len(self.sequences[self.sequences.str.len() > 1000]),
        }

    def get_attribute(self, name):
        mapping = {
            "uniprot_ids": self.uniprot_ids,
            "labels": self.labels,
            "organisms": self.organisms,
            "metrics": self.metrics,
            #"protT5_embedding": self.embeddings.get("prott5"),
            #"prostT5_embedding": self.embeddings.get("prostt5"),
            #"esm2_embedding": self.embeddings.get("esm"),
        }

        if name not in mapping or mapping[name] is None:
            raise KeyError(f"Attribute '{name}' not found in dataset.")

        return mapping[name]