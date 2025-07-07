from project_root.dataset.raw_dataset import RawDataset
from project_root.dataset.features.metric_processor import MetricProcessor
from project_root.dataset.features.embedding_loader import SequenceEmbeddingLoader, GOEmbeddingLoader

class ProcessedDataset:
    """
    Handles feature processing, embedding loading, and dataset construction for protein classification.
    Supports zero-shot dataset creation and flexible feature/embedding selection.

    :param raw_dataset: Instance containing raw input data.
    :param config: Configuration with embedding/normalization parameters.
    :param classifier_dataset_config: Optional config for classifier dataset.
    :param seq_loader: Loader for sequence embeddings.
    :param go_loader: Loader for GO embeddings.
    :param build_zero_shot_dataset: If True, builds a zero-shot dataset split.
    """

    def __init__(self, raw_dataset: RawDataset, 
                 config, classifier_dataset_config=None,
                 seq_loader: SequenceEmbeddingLoader = None,
                 go_loader: GOEmbeddingLoader = None,
                 build_zero_shot_dataset: bool = False):
        """
        Initializes the ProcessedDataset for selected feature processing.

        :param raw_dataset: Instance containing raw input data.
        :param config: Configuration with embedding/normalization parameters.
        :param classifier_dataset_config: Optional config for classifier dataset.
        :param seq_loader: Loader for sequence embeddings.
        :param go_loader: Loader for GO embeddings.
        :param build_zero_shot_dataset: If True, builds a zero-shot dataset split.
        """
        self.config = config
        self.processed_df = self._process_metrics(raw_dataset)
        self.main_columns = raw_dataset.main_columns
        self.build_zero_shot_dataset = build_zero_shot_dataset

        # Load embedding loaders if not provided
        self.seq_loader = seq_loader
        self.go_loader = go_loader

        # Lazy placeholders
        self._sequence_embeddings = None
        self._go_embeddings = None
        self._processed_metrics = None

        self._build_processed_dataset(classifier_dataset_config)

    def get_dataset(self):
        """
        Returns the processed dataset as a pandas DataFrame.

        :return: Processed dataset with embeddings and metrics (pd.DataFrame)
        """
        return self.df

    def get_zero_shot_dataset(self):
        """
        Returns the zero-shot dataset as a pandas DataFrame.

        :return: Zero-shot dataset with embeddings and metrics (pd.DataFrame)
        :raises ValueError: If zero-shot dataset is not enabled.
        """
        if not self.build_zero_shot_dataset:
            raise ValueError("Zero-shot dataset is not enabled in the configuration.")
        return self.zero_shot_df

    def _process_metrics(self, raw_dataset: RawDataset):
        """
        Processes and normalizes metrics from the raw dataset.

        :param raw_dataset: RawDataset instance
        :return: Processed DataFrame
        """
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

    def _build_processed_dataset(self, config, debug=False):
        """
        Builds the processed dataset and, if enabled, the zero-shot dataset.

        :param config: Configuration for the dataset processing.
        :param debug: If True, prints debug information.
        """
        self.selected_accessions = self._selected_accessions(config['organism_discrimination_strategy'])
        self.df = self._build_dataset(config, self.selected_accessions, debug=debug)

        print('----' * 20)
        print(f"Zero-shot dataset enabled: {self.build_zero_shot_dataset}")

        if self.build_zero_shot_dataset:
            # For zero-shot, use all accessions not in the selected ones
            print("Creating zero-shot dataset, using all accessions not in the selected ones.")
            all_accessions = self.processed_df[self.main_columns['id_col']]
            self.zero_shot_accesions = all_accessions[~all_accessions.isin(self.selected_accessions)]
            self.zero_shot_df = self._build_dataset(config, self.zero_shot_accesions, debug=debug)

    def _build_dataset(self, config, selected_accessions, debug=False):
        """
        Builds the dataset based on the provided configuration and selected accessions.

        :param config: Configuration for the dataset processing.
        :param selected_accessions: Accessions to include in the dataset (pd.Series).
        :param debug: If True, prints debug information.
        :return: Processed dataset (pd.DataFrame)
        """
        if debug:
            print(f"Building dataset with {len(selected_accessions)} accessions.")
        
        # Extract embeddings
        embeddings = self._load_embeddings(
            accessions=selected_accessions,
            sequence_embeddings=config['sequence_embeddings'],
            go_embeddings=config['go_embeddings']
        )
        
        # Start with the selected features from the processed DataFrame
        df = self.processed_df[self.processed_df[self.main_columns['id_col']].isin(selected_accessions)].copy()
        
        if debug:
            print("Getting dataset with the following columns:")
            print(f"Main ID column: {self.main_columns['id_col']}")
            print(f"Features columns: {config['features_col']}")
            print(f"Label column: {config['label_col']}")
            print(f"Balance column: {config['balance_col']}")
            
        df = df[[self.main_columns['id_col']] + config['features_col'] + [config['label_col']] + [config['balance_col']]]
        df.set_index(self.main_columns['id_col'], inplace=True)
        
        # Add each embedding as a new column (embedding tensor or array)
        for emb_name, emb_dict in embeddings.items():
            # emb_dict: {accession: embedding}
            df[emb_name] = df.index.map(emb_dict)

        return df

    def _selected_accessions(self, organism_discrimination_strategy=None):
        """
        Returns the accessions of the processed dataset based on the organism discrimination strategy.

        :param organism_discrimination_strategy: Strategy for organism discrimination (dict)
        :return: Accessions of the processed dataset (pd.Series)
        """
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

        :param n: Number of top organisms to return.
        :return: DataFrame with columns: 'organism', 'count', sorted by count descending.
        """
        organism_counts = self.processed_df['organism'].value_counts().reset_index()
        organism_counts.columns = ['organism', 'count']
        return organism_counts.head(n)
    
    def _get_accessions_by_organism(self, selected_organisms):
        """
        Returns accessions filtered by the selected organisms.

        :param selected_organisms: List of organisms to filter by.
        :return: Accessions corresponding to the selected organisms (pd.Series)
        """
        print(f"Selected organisms: {selected_organisms}")
        return self.processed_df[self.processed_df['organism'].isin(selected_organisms)][self.main_columns['id_col']]
    
    def _extract_embeddings(self, df, accessions, accession_col):
        """
        Extracts embeddings from a DataFrame for the given accessions.

        :param df: DataFrame containing embeddings.
        :param accessions: Accessions to extract.
        :param accession_col: Column name for accessions.
        :return: Dict mapping accession to embedding.
        """
        filtered = df[df[accession_col].isin(accessions)]
        if "embedding" in filtered.columns:
            return filtered.set_index(accession_col)["embedding"].to_dict()
        else:
            emb_cols = filtered.columns[1:]
            return (
                filtered.set_index(accession_col)[emb_cols]
                .apply(lambda row: row.values.astype(float), axis=1)
                .to_dict()
            )
    
    def _load_sequence_embeddings(self, accessions, sequence_embeddings):
        """
        Loads sequence embeddings for the given accessions and embedding configs.

        :param accessions: Accessions to load embeddings for.
        :param sequence_embeddings: Sequence embedding configuration dict.
        :return: Dict of embeddings.
        """
        seq_embeddings = {}
        for model_name, seq_cfg in sequence_embeddings.items():
            target_dim = seq_cfg.get("target_dim")
            use_autoencoder = seq_cfg.get("use_autoencoder")
            autoencoded_embeddings = seq_cfg.get("autoencoded_embeddings")
            df = self.seq_loader.load(
                model_name=model_name, 
                target_dim=target_dim, 
                use_autoencoder=use_autoencoder,
                autoencoded_embeddings=autoencoded_embeddings
            )
            accession_col = df.columns[0]  # Assuming first column is accession
            print(f"Filtered sequence embeddings for {model_name}: {df.shape}")
            seq_embeddings[model_name] = self._extract_embeddings(df, accessions, accession_col)
        return seq_embeddings
    
    def _load_go_embeddings(self, accessions, go_embeddings):
        """
        Loads GO embeddings for the given accessions and embedding configs.

        :param accessions: Accessions to load embeddings for.
        :param go_embeddings: GO embedding configuration dict.
        :return: Dict of embeddings.
        """
        go_embs = {}
        for model_name, go_cfg in go_embeddings.items():
            print(f"Loading GO embeddings for model: {model_name}")
            print(f"GO configuration: {go_cfg}")

            input_dim = go_cfg.get("input_dim")
            emb_dim = go_cfg.get("emb_dim")
            aggregated_dim = go_cfg.get("aggregated_dim")
            aggregation_strategy = go_cfg.get("aggregation_strategy")
            go_categories = go_cfg.get("go_categories")
            autoencoded_embeddings = go_cfg.get("autoencoded_embeddings", False)
            if isinstance(go_categories, str):
                go_categories = go_categories.split("_")
            df = self.go_loader.load(
                input_dim=input_dim,
                emb_dim=emb_dim,
                aggregated_dim=aggregated_dim,
                aggregation_strategy=aggregation_strategy,
                go_categories=go_categories,
                autoencoded_embeddings=autoencoded_embeddings
            )
            accession_col = df.columns[0]  # Assuming first column is accession
            print(f"Filtered GO embeddings for {model_name}: {df.shape}")
            go_embs[model_name + "_GO"] = self._extract_embeddings(df, accessions, accession_col)
        return go_embs
    
    def _load_embeddings(self, accessions, sequence_embeddings, go_embeddings):
        """
        Loads all embeddings (sequence and GO) for the given accessions.

        :param accessions: Accessions to load embeddings for.
        :param sequence_embeddings: Sequence embedding configuration dict.
        :param go_embeddings: GO embedding configuration dict.
        :return: Dict of all embeddings.
        """
        all_embeddings = {}
        all_embeddings.update(self._load_sequence_embeddings(accessions, sequence_embeddings))
        all_embeddings.update(self._load_go_embeddings(accessions, go_embeddings))
        return all_embeddings
    
    def get_zero_shot_accessions(self):
        """
        Returns the accessions of the zero-shot dataset.

        :return: Accessions of the zero-shot dataset (pd.Series)
        :raises ValueError: If zero-shot dataset is not enabled.
        """
        if not self.build_zero_shot_dataset:
            raise ValueError("Zero-shot dataset is not enabled in the configuration.")
        return self.zero_shot_accesions

    def get_accessions(self):
        """
        Returns the accessions of the processed dataset.

        :return: Accessions of the processed dataset (pd.Series)
        """
        return self.selected_accessions