# Final commit – Master’s Thesis by Àlex Domínguez Roig

# load_dataset.py
import hydra # type: ignore
from omegaconf import DictConfig # type: ignore
from project_root.dataset.dataset_handler import DatasetHandler
from project_root.dataset.dataset_config import DatasetConfigReader
from project_root.dataset.raw_dataset import RawDataset

import os
import sys

# Add the project root directory to PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
if project_root not in sys.path:
    sys.path.append(project_root)

@hydra.main(config_path="../config", config_name="config_load", version_base="1.3")
def load(cfg: DictConfig):
    config_reader = DatasetConfigReader(cfg)
    handler = DatasetHandler(config_reader)

    dataset_info = handler.load_raw()

    # Access the loaded DataFrame and column names
    id_col = dataset_info["id_col"]
    label_col = dataset_info["label_col"]
    organism_col = dataset_info["organism_col"]
    sequence_col = dataset_info["sequence_col"]
    metrics_col = dataset_info["metrics_col"]

    print("Dataset loaded successfully!")
    print("ID Column:", id_col)
    print("Label Column:", label_col)
    print("Organism Column:", organism_col)
    print("Sequence Column:", sequence_col)
    print("Metrics Columns:", metrics_col)
    print(dataset_info['dataset'].head())

    # embeddings = handler.load_embeddings() if cfg.options.use_embeddings else {}

    dataset = RawDataset(
        dataset = dataset_info["dataset"],
        id_col = id_col,
        label_col = label_col,
        organism_col = organism_col,
        metrics_col = metrics_col,
        sequence_col = sequence_col,
        #embeddings = embeddings
        )
    
    print(dataset.summary())

if __name__ == "__main__":
    load()
