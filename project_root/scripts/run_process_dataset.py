# scripts/process_dataset.py

import hydra # type: ignore
from omegaconf import DictConfig # type: ignore
from project_root.dataset.dataset_handler import DatasetHandler
from project_root.dataset.dataset_config import DatasetConfigReader
from project_root.dataset.raw_dataset import RawDataset
from project_root.dataset.processed_dataset import ProcessedDataset

import os
import sys

# Add the project root directory to PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
if project_root not in sys.path:
    sys.path.append(project_root)

@hydra.main(config_path="../config", config_name="config_process", version_base="1.3")
def process(cfg: DictConfig):
    config_reader = DatasetConfigReader(cfg)
    handler = DatasetHandler(config_reader)

    raw_data = handler.load_raw() if cfg.options.load_from_raw else {}
    embeddings = handler.load_embeddings() if cfg.options.use_embeddings else {}

    raw_dataset = RawDataset(raw_data, embeddings)
    print("📦 Raw dataset summary:")
    print(raw_dataset.summary())

    processed_dataset = ProcessedDataset(raw_dataset, features_to_process=cfg.features.to_process)
    print("✅ Processed features:")
    for name in processed_dataset.get_all_features().keys():
        print(f"  • {name}")

if __name__ == "__main__":
    process()
