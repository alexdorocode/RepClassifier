from abc import ABC, abstractmethod
import pandas as pd # type: ignore
import torch # type: ignore
import gc
import os
from project_root.models.autoencoder import Autoencoder

class EmbeddingLoader(ABC):
    """
    Abstract base class for embedding loaders.
    Handles loading and (optionally) compressing embeddings using autoencoders.
    """

    def __init__(self, 
                 embedding_sequence_paths=None, 
                 ae_paths=None,
                 autoencoded_seq_embeddings=None,
                 autoencoded_go_embeddings=None):
        """
        Initialize the EmbeddingLoader.

        :param embedding_sequence_paths: Dict of paths to sequence embeddings.
        :param ae_paths: Dict of autoencoder model paths.
        :param autoencoded_seq_embeddings: Dict of precomputed autoencoded sequence embeddings.
        :param autoencoded_go_embeddings: Dict of precomputed autoencoded GO embeddings.
        """
        self.embedding_sequence_paths = embedding_sequence_paths
        self.ae_paths = ae_paths
        self.autoencoded_seq_embeddings = autoencoded_seq_embeddings
        self.autoencoded_go_embeddings = autoencoded_go_embeddings

    @abstractmethod
    def load(self, **kwargs):
        """
        Load embeddings based on model and dimension information.
        """
        pass

    def _load_flat_csv_embeddings(self, path, id_col=0, target_dim=None, ae_model=None):
        """
        Shared method to load embeddings from flattened CSV.

        :param path: Path to the CSV file.
        :param id_col: Index of the ID column.
        :param target_dim: Target dimension for compression.
        :param ae_model: Autoencoder model for compression.
        :return: DataFrame with ID and embedding columns.
        """
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
        """
        Loads an autoencoder model for the given model name and dimensions.

        :param model_name: Name of the model.
        :param input_dim: Input dimension.
        :param target_dim: Target (latent) dimension.
        :return: Loaded Autoencoder instance.
        :raises ValueError: If no matching autoencoder is found.
        """
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
    """
    Loader for sequence embeddings, with optional autoencoder compression.
    """

    def load(self, model_name: str, target_dim, use_autoencoder=False, autoencoded_embeddings=False):
        """
        Load sequence embeddings for a given model and dimension.

        :param model_name: Name of the sequence embedding model.
        :param target_dim: Target dimension for embeddings.
        :param use_autoencoder: If True, compress embeddings using an autoencoder.
        :param autoencoded_embeddings: If True, load precomputed autoencoded embeddings.
        :return: DataFrame with embeddings.
        """
        print(f"🚀 Loading sequence embeddings for {model_name}...")
        print(f"Target dimension: {target_dim}, Use autoencoder: {use_autoencoder}, Autoencoded embeddings: {autoencoded_embeddings}")

        if use_autoencoder:
            emb_path = self.embedding_sequence_paths[model_name]
            print(f"🚀 Loading raw sequence embeddings from {emb_path} and compressing...")
            input_dim = pd.read_csv(emb_path, nrows=1).shape[1] - 1
            ae_model = self._load_autoencoder(model_name, input_dim, target_dim) if target_dim else None
            return self._load_flat_csv_embeddings(emb_path, target_dim=target_dim, ae_model=ae_model)

        # Find the entry with the correct dimension
        precomputed_list = self.autoencoded_seq_embeddings[model_name]
        precomputed_entry = next((item for item in precomputed_list if item["dim"] == target_dim), None)
        if precomputed_entry is None:
            raise ValueError(f"No precomputed embedding found for {model_name} with dim={target_dim}")

        folder_path = self.autoencoded_seq_embeddings["folder_path"]
        precomputed_path = os.path.join(folder_path, precomputed_entry["file_name"])
        return pd.read_csv(precomputed_path)

class GOEmbeddingLoader(EmbeddingLoader):
    """
    Loader for GO embeddings, with support for aggregation and autoencoding.
    """

    def load(self, input_dim: int, emb_dim: int, 
             aggregated_dim: int, aggregation_strategy: str, 
             go_categories: list, use_autoencoder=False, autoencoded_embeddings=False):
        """
        Load GO embeddings with the specified configuration.

        :param input_dim: Input dimension.
        :param emb_dim: Embedding dimension.
        :param aggregated_dim: Aggregated dimension.
        :param aggregation_strategy: Aggregation strategy (e.g., 'mean_pooling').
        :param go_categories: List of GO categories.
        :param use_autoencoder: If True, use autoencoder for compression.
        :param autoencoded_embeddings: If True, load precomputed autoencoded embeddings.
        :return: DataFrame with GO embeddings.
        """
        print(f"🚀 Loading GO embeddings with input_dim={input_dim}, emb_dim={emb_dim},    "
                f"aggregated_dim={aggregated_dim}, aggregation_strategy={aggregation_strategy}, "
                f"go_categories={go_categories}, use_autoencoder={use_autoencoder}, autoencoded_embeddings={autoencoded_embeddings}")
        
        if aggregated_dim is None and aggregation_strategy is not None:
            if aggregation_strategy == 'mean_pooling':
                aggregated_dim = emb_dim
            else:
                aggregated_dim = emb_dim * 10

        path = self._resolve_go_embedding_path(
            input_dim=input_dim,
            emb_dim=emb_dim,
            aggregated_dim=aggregated_dim,
            aggregation_strategy=aggregation_strategy,
            go_categories=go_categories,
            autoencoded_embeddings=autoencoded_embeddings
        )
        print(f"🚀 Loading GO embeddings from {path}...")
        if os.path.getsize(path) == 0:
            raise ValueError(f"Embedding file {path} is empty!")
        return pd.read_csv(path)

    def _resolve_go_embedding_path(self, input_dim, emb_dim, 
                                   aggregated_dim, aggregation_strategy, go_categories, 
                                    autoencoded_embeddings):
        """
        Resolves the file path for the requested GO embedding configuration.

        :param input_dim: Input dimension.
        :param emb_dim: Embedding dimension.
        :param aggregated_dim: Aggregated dimension.
        :param aggregation_strategy: Aggregation strategy.
        :param go_categories: List of GO categories.
        :param autoencoded_embeddings: If True, use autoencoded embeddings.
        :return: Path to the embedding file.
        :raises ValueError: If no matching embedding file is found.
        """
        folder_path = self.autoencoded_go_embeddings['folder_path']
        file_entries = self.autoencoded_go_embeddings['file_paths']['autoencoded_embeddings'] if autoencoded_embeddings else self.autoencoded_go_embeddings['file_paths']['non_autoencoded_embeddings']

        if aggregation_strategy == 'mean_pooling':
            aggregation_strategy = 'mean_pool'

        for entry in file_entries:
            # Compare each key, assuming all are integers or strings as needed
            if (entry['input_dim'] == input_dim and
                entry['emb_dim'] == emb_dim and
                entry['aggregated_dim'] == aggregated_dim and
                entry['aggregation_strategy'] == aggregation_strategy and
                (entry['go_categories'] == '_'.join(go_categories) if isinstance(entry['go_categories'], str) else entry['go_categories'] == go_categories)):
                return f"{folder_path}/{entry['path']}"
        
        raise ValueError(f"No embedding path found for input_dim={input_dim}, emb_dim={emb_dim}, "
                         f"aggregated_dim={aggregated_dim}, strategy={aggregation_strategy}, categories={go_categories}")