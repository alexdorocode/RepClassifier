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

        '''
        main_columns = {
            "id_col" :
            "label_col" :
            "organism_col" :
            "metrics_col" :
            "sequence_col" :
        }
        '''

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
        balance_col = config['balance_col']
        organism_discrimination_strategy = config['organism_discrimination_strategy']
        
        selected_accessions = self._selected_accessions(organism_discrimination_strategy)
        
        embeddings = self._load_embeddings(
            accessions=selected_accessions,
            sequence_embeddings=sequence_embeddings,
            go_embeddings=go_embeddings
        )
        
        # Start with the selected features from the processed DataFrame
        df = self.processed_df[self.processed_df[self.main_columns['id_col']].isin(selected_accessions)].copy()
        print("Getting dataset with the following columns:")
        print(f"Main ID column: {self.main_columns['id_col']}")
        print(f"Features columns: {features_col}")
        print(f"Label column: {label_col}")
        print(f"Balance column: {balance_col}")
        df = df[[self.main_columns['id_col']] + features_col + [label_col] + [balance_col]]
        df.set_index(self.main_columns['id_col'], inplace=True)
        
        # Add each embedding as a new column (embedding tensor or array)
        for emb_name, emb_dict in embeddings.items():
            # emb_dict: {accession: embedding}
            df[emb_name] = df.index.map(emb_dict)

        # Optionally, reset index if you want accession as a column
        # df.reset_index(inplace=True)
        
        return df

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
    
    def _extract_embeddings(self, df, accessions, accession_col):
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
        go_embs = {}
        for model_name, go_cfg in go_embeddings.items():
            # Extract configuration parameters for GO embeddings

            print(f"Loading GO embeddings for model: {model_name}")

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
        all_embeddings = {}
        all_embeddings.update(self._load_sequence_embeddings(accessions, sequence_embeddings))
        all_embeddings.update(self._load_go_embeddings(accessions, go_embeddings))
        return all_embeddings

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