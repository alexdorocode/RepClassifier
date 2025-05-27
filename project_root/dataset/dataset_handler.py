import os
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import yaml  # We'll need this to parse YAML if you load from YAML files
from project_root.dataset.features.embedding_loader import SequenceEmbeddingLoader, GOEmbeddingLoader

class DatasetHandler:
    def __init__(self, config_reader):
        self.config = config_reader

    def load_raw(self):
        root_dir = self.config.root
        unified_dataset = self.config.file
        dataset_path = os.path.join(root_dir, unified_dataset)

        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset file {dataset_path} does not exist.")

        print(f"Loading dataset from {dataset_path}")
        df = pd.read_csv(dataset_path)

        id_col = self.config.id_col
        label_col = self.config.label_col
        organism_col = self.config.organism_col
        sequence_col = self.config.sequence_col
        metrics_col = self.config.metrics_col

        required_columns = [id_col, label_col, organism_col] + metrics_col
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in the dataset: {missing_columns}")

        self.id_col = df[id_col]

        return {
            "dataset": df,
            "id_col": id_col,
            "label_col": label_col,
            "sequence_col": sequence_col,
            "organism_col": organism_col,
            "metrics_col": metrics_col,
        }

    def load_embedding_loaders(self):
        """
        Load configurations for embedding loaders (Sequence and GO) and return loader instances.
        """
        # Access paths from the nested paths dictionary
        seq_emb_cfg = self.config.paths["embedding_sequence_paths"]
        ae_cfg = self.config.paths["autoencoder_paths"]
        autoencoded_seq_emb = self.config.paths["autoencoded_seq_embeddings"]
        go_emb_cfg = self.config.paths["autoencoded_go_embeddings"]

        # If go_emb_cfg is a YAML path (string), load it; otherwise, assume it's already a dict
        if isinstance(go_emb_cfg, str):
            with open(go_emb_cfg, "r") as f:
                go_emb_cfg = yaml.safe_load(f)['autoencoded_go_embeddings']

        # Create loader instances
        sequence_loader = SequenceEmbeddingLoader(
            embedding_sequence_paths=seq_emb_cfg,
            ae_paths=ae_cfg,
            autoencoded_seq_embeddings=autoencoded_seq_emb,
            autoencoded_go_embeddings=go_emb_cfg
        )

        go_loader = GOEmbeddingLoader(
            embedding_sequence_paths=seq_emb_cfg,  # Optional: you can pass {} or None if not used by GO
            ae_paths=ae_cfg,
            autoencoded_seq_embeddings=autoencoded_seq_emb,
            autoencoded_go_embeddings=go_emb_cfg
        )

        return {
            "sequence_loader": sequence_loader,
            "go_loader": go_loader
        }
    