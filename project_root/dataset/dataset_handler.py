# dataset/dataset_handler.py
import os
import numpy as np # type: ignore
import pandas as pd # type: ignore

class DatasetHandler:
    def __init__(self, config_reader):
        self.config = config_reader
        
    def load_raw(self):
        # Access the unified dataset configuration
        root_dir = self.config.root
        unified_dataset = self.config.file

        # Construct the full path to the dataset file
        dataset_path = os.path.join(root_dir, unified_dataset)

        # Check if the dataset file exists
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset file {dataset_path} does not exist.")

        # Load the dataset
        print(f"Loading dataset from {dataset_path}")
        df = pd.read_csv(dataset_path)

        # Extract relevant columns based on the configuration
        id_col = self.config.id_col
        label_col = self.config.label_col
        organism_col = self.config.organism_col
        sequence_col = self.config.sequence_col
        metrics_col = self.config.metrics_col

        # Ensure all required columns exist in the dataset
        required_columns = [id_col, label_col, organism_col] + metrics_col
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in the dataset: {missing_columns}")

        self.id_col = df[id_col]

        # Return the loaded dataset and relevant columns
        return {
            "dataset": df,
            "id_col": id_col,
            "label_col": label_col,
            "sequence_col": sequence_col,
            "organism_col": organism_col,
            "metrics_col": metrics_col,
        }

    def load_embeddings(self):
        emb_data = {}
        # Load embeddings from the specified directory
        emb_folder = self.config.emb_dir
        if not os.path.exists(emb_folder):
            raise FileNotFoundError(f"Embedding directory {emb_folder} does not exist.")
        
        print(f"Loading embeddings from {emb_folder}")

        for name, emb_info in self.config.embeddings.items():
            # Check if the embedding info is provided
            print(f"Loading {name} embeddings")
            print(emb_info)
            if emb_info:
                file = emb_info.get("file")
                id_col = emb_info.get("id_col")
                emb_col = emb_info.get("emb_col")

                path = os.path.join(emb_folder, file)
                print(f"Loading {name} embeddings from {path}")

                if not self.check_embeddings(path, id_col, emb_col):
                    raise ValueError(f"Invalid embedding file {file} or columns {id_col}, {emb_col}.")

                df = pd.read_csv(path)
                embedding = df[emb_col].apply(lambda x: np.fromstring(x[1:-1], sep=',')).tolist()
                emb_data[name] = np.array(embedding, dtype=np.float32)
                
        return emb_data
    
    def check_embeddings(self, file, id_col, emb_col):
        # Check if the embeddings are in the correct format
        df = pd.read_csv(file)
        if id_col not in df.columns or emb_col not in df.columns:
            raise ValueError(f"Columns {id_col} or {emb_col} not found in {file}.")
        
        # Check if the embeddings are valid numpy arrays
        for i, row in df.iterrows():
            try:
                np.fromstring(row[emb_col][1:-1], sep=',')
            except Exception as e:
                raise ValueError(f"Invalid embedding at row {i}: {e}")
        
        return True