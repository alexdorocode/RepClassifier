from project_root.dataset.raw_dataset import RawDataset
from project_root.dataset.features.metric_processor import MetricProcessor
from project_root.dataset.features.embedding_loader import SequenceEmbeddingLoader, GOEmbeddingLoader


class ProcessedDataset:
    def __init__(self, raw_dataset: RawDataset, 
                 config, seq_loader: SequenceEmbeddingLoader = None,
                 go_loader: GOEmbeddingLoader = None,
                 features_to_process=None):
        """
        Initializes the ProcessedDataset for selected feature processing.

        Args:
            raw_dataset (RawDataset): Instance containing raw input data.
            config (DictConfig or dict): Configuration with embedding/normalization parameters.
            features_to_process (list or None): If None, process all available features.
        """
        self.raw_dataset = raw_dataset
        self.config = config

        # Load embedding loaders if not provided
        self.seq_loader = seq_loader
        self.go_loader = go_loader

        # Lazy placeholders
        self._sequence_embeddings = None
        self._go_embeddings = None
        self._processed_metrics = None

        self.processed_df = self._process_metrics()

    def _process_metrics(self):
        metric_cols = self.config.columns
        print(f"Processing metrics: {metric_cols}")
        processor = MetricProcessor(
            df=self.raw_dataset.dataset,
            metric_cols=metric_cols,
            nan_strategy=self.config.nan_strategy,
            scaler=self.config.scaler_type,
            trim_config=self.config.trim_config
        )
        return processor.handle_nans().normalize().get_processed_df()

    def get_dataset(self, config):
        """
        Returns the processed dataset based on the provided configuration.

        Args:
            config (dict): Configuration for the dataset processing.

        Returns:
            pd.DataFrame: Processed dataset.
        """
        label_col = config['label_col']
        features_col = config['features_col']
        sequence_embeddings = config['sequence_embeddings']
        go_embeddings = config['go_embeddings']
        organism_discrimination_strategy = config['organism_discrimination_strategy']
        
        print(f"Returning dataset with label_col: {label_col}, features_col: {features_col}, "
                f"sequence_embeddings: {sequence_embeddings}, go_embeddings: {go_embeddings}, " 
                f"organism_discrimination_strategy: {organism_discrimination_strategy}")



'''

    @property
    def all_possible_features(self):
        return {"metrics", "sequence_embeddings", "go_embeddings"}

    def _process_features(self):
        self.feature_outputs = {}

        if "metrics" in self.features_to_process:
            self.feature_outputs["metrics"] = self._process_metrics()

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

        '''