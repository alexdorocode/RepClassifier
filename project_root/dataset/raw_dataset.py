import pandas as pd
import numpy as np

class RawDataset:
    """
    Container for raw protein dataset elements:
    UniProt IDs, labels, sequences, metrics, and optional embeddings.

    :param dataset: The raw dataset as a DataFrame.
    :param id_col: Column name for UniProt IDs.
    :param label_col: Column name for labels.
    :param organism_col: Column name for organism information.
    :param metrics_col: List of column names for metrics.
    :param sequence_col: Column name for protein sequences.
    """

    def __init__(self, dataset: pd.DataFrame, 
                 id_col: str, label_col: str, 
                 organism_col: str, metrics_col: list, sequence_col: str
                 ):
        """
        Initializes a RawDataset object.

        :param dataset: The raw dataset as a DataFrame.
        :param id_col: Column name for UniProt IDs.
        :param label_col: Column name for labels.
        :param organism_col: Column name for organism information.
        :param metrics_col: List of column names for metrics.
        :param sequence_col: Column name for protein sequences.
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
        """
        Returns the number of samples in the dataset.

        :return: Number of UniProt IDs (int)
        """
        return len(self.uniprot_ids)

    def summary(self):
        """
        Returns a summary of the dataset, including counts and sequence length stats.

        :return: Dictionary with summary statistics
        """
        return {
            "num_ids": len(self.uniprot_ids),
            "has_labels": not self.labels.empty,
            "has_organisms": not self.organisms.empty,
            "num_metrics": self.metrics.shape[1] if not self.metrics.empty else 0,
            # "embeddings_loaded": list(self.embeddings.keys())
            "length_biggest_than_3000": len(self.sequences[self.sequences.str.len() > 3000]),
            "length_biggest_than_2000": len(self.sequences[self.sequences.str.len() > 2000]),
            "length_biggest_than_1000": len(self.sequences[self.sequences.str.len() > 1000]),
        }

    def get_attribute(self, name):
        """
        Retrieve a specific attribute from the dataset.

        :param name: Name of the attribute ('uniprot_ids', 'labels', 'organisms', 'metrics')
        :return: The requested attribute
        :raises KeyError: If the attribute is not found
        """
        mapping = {
            "uniprot_ids": self.uniprot_ids,
            "labels": self.labels,
            "organisms": self.organisms,
            "metrics": self.metrics,
            # "protT5_embedding": self.embeddings.get("prott5"),
            # "prostT5_embedding": self.embeddings.get("prostt5"),
            # "esm2_embedding": self.embeddings.get("esm"),
        }

        if name not in mapping or mapping[name] is None:
            raise KeyError(f"Attribute '{name}' not found in dataset.")

        return