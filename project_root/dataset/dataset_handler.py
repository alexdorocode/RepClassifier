# dataset/dataset_handler.py
import os
import numpy as np # type: ignore
import pandas as pd # type: ignore

class DatasetHandler:
    def __init__(self, config_reader):
        self.config = config_reader

    def load_raw(self):
        dfs = {}
        for key, path in self.config.paths.items():
            dfs[key] = pd.read_csv(path)
        return dfs

    def load_embeddings(self):
        emb_data = {}
        for name, file in self.config.embeddings.items():
            if file:
                path = os.path.join(self.config.emb_dir, name, file)
                emb_data[name] = np.load(path, allow_pickle=True).item()
        return emb_data
