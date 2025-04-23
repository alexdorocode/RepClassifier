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

    raw_data = handler.load_raw() if cfg.options.load_from_raw else {}
    embeddings = handler.load_embeddings() if cfg.options.use_embeddings else {}

    dataset = RawDataset(raw_data, embeddings)
    print(dataset.summary())

if __name__ == "__main__":
    load()
