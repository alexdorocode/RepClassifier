import os
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import yaml  # We'll need this to parse YAML if you load from YAML files
from project_root.dataset.raw_dataset import RawDataset
from project_root.dataset.processed_dataset import ProcessedDataset
from project_root.dataset.features.embedding_loader import SequenceEmbeddingLoader, GOEmbeddingLoader

class DatasetHandler:
    def __init__(self, config_reader):
        self.config = config_reader
        
        self._build_processed_dataset()
        print("🔧 Processed dataset created successfully.")

        print("📦 DatasetHandler initialized successfully.")
        a = self.load_experimental_dataset(config=self.config.experiment_definition)
        print(a)
        print("📦 DatasetHandler loading experimental dataset.")

    def _build_raw(self):
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

        return RawDataset(
            dataset=df,
            id_col=id_col,
            label_col=label_col,
            organism_col=organism_col,
            sequence_col=sequence_col,
            metrics_col=metrics_col
        )
    
    def _build_embedding_loaders(self):
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

        return sequence_loader, go_loader
    
    def _build_processed_dataset(self):
        """
        Load the processed dataset based on the raw dataset and configuration.
        """
        
        print("🔍 Initializing DatasetHandler")
        raw_dataset = self._build_raw()
        print("📥 Raw dataset loaded successfully.")
        sequence_loader, go_loader = self._build_embedding_loaders()
        print("🔍 Embedding loaders initialized successfully.")
        
        self.processed_dataset = ProcessedDataset(
            raw_dataset= raw_dataset,
            config=self.config.features_to_process,
            seq_loader= sequence_loader,
            go_loader= go_loader
        )

    def load_experimental_dataset(self, config):
        """
        Load the experimental dataset based on the configuration.
        """
        print("🔍 Loading experimental dataset...")
        return self.processed_dataset.get_dataset(config)