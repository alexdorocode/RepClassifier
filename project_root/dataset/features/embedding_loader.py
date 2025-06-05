from abc import ABC, abstractmethod
import pandas as pd
import torch
import gc
import os
from project_root.models.autoencoder import Autoencoder  # Adjust import

class EmbeddingLoader(ABC):
    def __init__(self, 
                 embedding_sequence_paths = None, 
                 ae_paths = None,
                 autoencoded_seq_embeddings = None,
                 autoencoded_go_embeddings = None):
        self.embedding_sequence_paths = embedding_sequence_paths
        self.ae_paths = ae_paths
        self.autoencoded_seq_embeddings = autoencoded_seq_embeddings
        self.autoencoded_go_embeddings = autoencoded_go_embeddings

    @abstractmethod
    def load(self, **kwargs):
        """Load embeddings based on model and dimension information."""
        pass

    def _load_flat_csv_embeddings(self, path, id_col=0, target_dim=None, ae_model=None):
        """Shared method to load embeddings from flattened CSV."""
        print(f"📥 Loading flat CSV embeddings from {path}...")
        df = pd.read_csv(path, dtype={df.columns[id_col]: str})
        embedding_cols = df.columns[1:]
        df['embedding'] = df[embedding_cols].apply(lambda row: torch.tensor(row.values, dtype=torch.float32), axis=1)
        embeddings = df[[df.columns[id_col], 'embedding']]

        if target_dim and ae_model:
            print(f"🔄 Compressing embeddings to {target_dim} dimensions...")
            with torch.no_grad():
                embeddings['embedding'] = embeddings['embedding'].apply(lambda x: ae_model.encode(x.unsqueeze(0)).squeeze(0).cpu())
            del ae_model
            gc.collect()
            torch.cuda.empty_cache()

        return embeddings

    def _load_autoencoder(self, model_name, input_dim, target_dim):
        ae_info = self.ae_paths[model_name]
        for model in ae_info["models"]:
            if model["in_out_dim"][0] == input_dim and model["in_out_dim"][1] == target_dim:
                path = f"{ae_info['folder_path']}/{model['file_name']}"
                ae = Autoencoder.load_from_checkpoint(
                    path=path,
                    input_dim=input_dim,
                    latent_dim=target_dim,
                    activation=model["activation"]
                )
                ae.eval()
                return ae
        raise ValueError(f"No autoencoder found for model {model_name} with input_dim={input_dim} and target_dim={target_dim}")

class SequenceEmbeddingLoader(EmbeddingLoader):
    def load(self, model_name: str, target_dim: int = None, use_autoencoder=False, autoencoded_embeddings=False):
        
        print(f"🚀 Loading sequence embeddings for {model_name}...")
        print(f"Target dimension: {target_dim}, Use autoencoder: {use_autoencoder}, Autoencoded embeddings: {autoencoded_embeddings}")

        if use_autoencoder:
            
            emb_path = self.embedding_sequence_paths[model_name]

            print(f"🚀 Loading raw sequence embeddings from {emb_path} and compressing...")
            input_dim = pd.read_csv(emb_path, nrows=1).shape[1] - 1
            ae_model = self._load_autoencoder(model_name, input_dim, target_dim) if target_dim else None
            return self._load_flat_csv_embeddings(emb_path, target_dim=target_dim, ae_model=ae_model)

        # Load precomputed autoencoded embeddings
        print(f"🚀 Loading precomputed autoencoded sequence embeddings for {model_name}...")

        # Find the entry with the correct dimension
        precomputed_list = self.autoencoded_seq_embeddings[model_name]
        precomputed_entry = next((item for item in precomputed_list if item["dim"] == target_dim), None)
        if precomputed_entry is None:
            raise ValueError(f"No precomputed embedding found for {model_name} with dim={target_dim}")

        folder_path = self.autoencoded_seq_embeddings["folder_path"]
        precomputed_path = os.path.join(folder_path, precomputed_entry["file_name"])
        return pd.read_csv(precomputed_path)

class GOEmbeddingLoader(EmbeddingLoader):
    def load(self, input_dim: int, emb_dim: int, 
             aggregated_dim: int, aggregation_strategy: str, 
             go_categories: list, use_autoencoder=False, autoencoded_embeddings=False):
        path = self._resolve_go_embedding_path(
            input_dim=input_dim,
            emb_dim=emb_dim,
            aggregated_dim=aggregated_dim,
            aggregation_strategy=aggregation_strategy,
            go_categories=go_categories,
            use_autoencoder=use_autoencoder,
            autoencoded_embeddings=autoencoded_embeddings
        )
        print(f"🚀 Loading GO embeddings from {path}...")
        return pd.read_csv(path)

    def _resolve_go_embedding_path(self, input_dim, emb_dim, 
                                   aggregated_dim, aggregation_strategy, go_categories, 
                                   use_autoencoder, autoencoded_embeddings):
        folder_path = self.autoencoded_go_embeddings['folder_path']
        file_entries = self.autoencoded_go_embeddings['file_paths']['autoencoded_embeddings'] if autoencoded_embeddings else self.autoencoded_go_embeddings['file_paths']['non_autoencoded_embeddings']

        for entry in file_entries:
            # Compare each key, assuming all are integers or strings as needed
            if (entry['input_dim'] == input_dim and
                entry['emb_dim'] == emb_dim and
                entry['aggreated_dim'] == aggregated_dim and
                entry['aggregation_strategy'] == aggregation_strategy and
                entry['go_categories'] == '_'.join(go_categories) if isinstance(entry['go_categories'], str) else entry['go_categories'] == go_categories):
                return f"{folder_path}/{entry['path']}"

        # If not found
        raise ValueError(f"No embedding path found for input_dim={input_dim}, emb_dim={emb_dim}, "
                         f"aggregated_dim={aggregated_dim}, strategy={aggregation_strategy}, categories={go_categories}, "
                         f"use_autoencoder={use_autoencoder}")


