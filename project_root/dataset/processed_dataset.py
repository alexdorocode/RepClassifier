from project_root.dataset.raw_dataset import RawDataset
from project_root.dataset.features.metric_processor import MetricProcessor
from project_root.dataset.features.embedding_loader import SequenceEmbeddingLoader, GOEmbeddingLoader


class ProcessedDataset:
    def __init__(self, raw_dataset: RawDataset, config, features_to_process=None):
        """
        Initializes the ProcessedDataset for selected feature processing.

        Args:
            raw_dataset (RawDataset): Instance containing raw input data.
            config (DictConfig or dict): Configuration with embedding/normalization parameters.
            features_to_process (list or None): If None, process all available features.
        """
        self.raw_dataset = raw_dataset
        self.config = config
        self.features_to_process = set(features_to_process) if features_to_process else self.all_possible_features

        # Lazy placeholders
        self._sequence_embeddings = None
        self._go_embeddings = None
        self._processed_metrics = None

        # Initialize loaders with appropriate configs
        self.seq_loader = SequenceEmbeddingLoader(
            embedding_cfg=config.embedding_paths,
            ae_cfg=config.autoencoder,
            model_name=config.sequence_embedding_name,
            target_dim=getattr(config.sequence_embedding, "target_dim", None)
        )

        self.go_loader = GOEmbeddingLoader(
            embedding_cfg=config.embedding_paths,
            ae_cfg=config.autoencoder,
            model_name="GeoKG",  # Assuming this is the key used for GO
            target_dim=getattr(config.go, "target_dim", None),
            dim=config.go.dim,
            pooling=config.go.pooling,
            go_type=config.go.go_type
        )

        self._process_features()

    @property
    def all_possible_features(self):
        return {"metrics", "sequence_embeddings", "go_embeddings"}

    def _process_features(self):
        self.feature_outputs = {}

        if "metrics" in self.features_to_process:
            self.feature_outputs["metrics"] = self._process_metrics()

    def _process_metrics(self):
        metric_cols = self.config.metrics.columns
        print(f"Processing metrics: {metric_cols}")
        processor = MetricProcessor(
            df=self.raw_dataset.dataset,
            metric_cols=metric_cols,
            nan_strategy=self.config.metrics.nan_strategy,
            scaler=self.config.metrics.scaler_type,
            trim_config=getattr(self.config.metrics, "trim_config", None)
        )
        return processor.handle_nans().normalize().get_processed_df()

    @property
    def sequence_embeddings(self):
        if self._sequence_embeddings is None:
            self._sequence_embeddings = self.seq_loader.load()
        return self._sequence_embeddings

    @property
    def go_embeddings(self):
        if self._go_embeddings is None:
            self._go_embeddings = self.go_loader.load()
        return self._go_embeddings

    def get_feature(self, name):
        if name not in self.feature_outputs:
            raise KeyError(f"Feature '{name}' not found or not processed.")
        return self.feature_outputs[name]
