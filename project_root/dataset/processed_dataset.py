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
        
        self.config = config
        self.processed_df = self._process_metrics(raw_dataset)
        self.main_columns = raw_dataset.main_columns

        # Load embedding loaders if not provided
        self.seq_loader = seq_loader
        self.go_loader = go_loader

        # Lazy placeholders
        self._sequence_embeddings = None
        self._go_embeddings = None
        self._processed_metrics = None

    def _process_metrics(self, raw_dataset: RawDataset):
        metric_cols = self.config.columns
        print(f"Processing metrics: {metric_cols}")
        processor = MetricProcessor(
            df=raw_dataset.dataset,
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

        print("Accessions on processed dataset: ", len(self.processed_df))
        print("Accessions on selected dataset: ", len(self._selected_accessions(organism_discrimination_strategy)))
        
        '''
        embeddings = _load_embeddings(
            accessions=self.processed_df.selected_accessions(organism_discrimination_strategy),
            sequence_embeddings=self.sequence_embeddings,
            go_embeddings=self.go_embeddings,
            organism_discrimination_strategy=organism_discrimination_strategy
        )
        '''

    def _selected_accessions(self, organism_discrimination_strategy=None):
        """
        Returns the accessions of the processed dataset based on the organism discrimination strategy.
    
        Args:
            organism_discrimination_strategy (dict): Strategy for organism discrimination.
    
        Returns:
            pd.Series: Accessions of the processed dataset.
        """
        
        # Use dictionary access, not attribute access
        use_selected = organism_discrimination_strategy.get('use_selected_organisms')
        use_top = organism_discrimination_strategy.get('use_top_organisms')
        
        if use_selected is not None:
            selected_organisms = use_selected
        elif use_top is not None:
            selected_organisms = self._top_n_organisms(use_top)['organism'].tolist()
        else:
            # If neither is set, return all accessions
            print("No organism discrimination strategy set, returning all accessions.")
            return self.processed_df.id_col
    
        return self._get_accessions_by_organism(selected_organisms)
    
    def _top_n_organisms(self, n):
        """
        Returns the top n organisms with the most samples in the dataframe.
        
        Args:
            df (pd.DataFrame): DataFrame containing an 'organism' column.
            n (int): Number of top organisms to return.
        
        Returns:
            pd.DataFrame: A dataframe with two columns: 'organism' and 'count', sorted by count descending.
        """
        # Count samples per organism
        organism_counts = self.processed_df['organism'].value_counts().reset_index()
        organism_counts.columns = ['organism', 'count']
        print(f"Organism_counts: {organism_counts.columns}")
        
        # Return top n organisms
        return organism_counts.head(n)
    
    def _get_accessions_by_organism(self, selected_organisms):
        """
        Returns accessions filtered by the selected organisms.

        Args:
            selected_organisms (list): List of organisms to filter by.

        Returns:
            pd.Series: Accessions corresponding to the selected organisms.
        """
        print(f"Selected organisms: {selected_organisms}")
        return self.processed_df[self.processed_df['organism'].isin(selected_organisms)][self.main_columns['id_col']]
    

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