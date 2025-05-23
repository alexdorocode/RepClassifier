# run_process_dataset.py

import hydra  # type: ignore
from omegaconf import DictConfig  # type: ignore
import os
import sys

from project_root.dataset.dataset_handler import DatasetHandler
from project_root.dataset.dataset_config import DatasetConfigReader
from project_root.dataset.raw_dataset import RawDataset
from project_root.dataset.processed_dataset import ProcessedDataset

# Add the project root directory to PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
if project_root not in sys.path:
    sys.path.append(project_root)

@hydra.main(config_path="../config", config_name="config_process", version_base="1.3")
def process(cfg: DictConfig):
    print("⚙️  Initializing raw dataset...")
    config_reader = DatasetConfigReader(cfg)
    handler = DatasetHandler(config_reader)

    dataset_info = handler.load_raw()

    dataset = RawDataset(
        dataset=dataset_info["dataset"],
        id_col=dataset_info["id_col"],
        label_col=dataset_info["label_col"],
        organism_col=dataset_info["organism_col"],
        metrics_col=dataset_info["metrics_col"],
        sequence_col=dataset_info["sequence_col"],
    )

    print(dataset.summary())

    print("\n🔧 Creating processed dataset...")
    processed = ProcessedDataset(
        raw_dataset=dataset,
        config=cfg,
        features_to_process=["metrics", "sequence_embeddings", "go_embeddings"]
    )

    metrics_df = processed.get_feature("metrics")
    print("\n📊 Normalized metrics preview:")
    print(metrics_df.head())

    #print("\n📥 Sequence Embeddings shape:", processed.sequence_embeddings.shape)
    #print("📥 GO Embeddings shape:", processed.go_embeddings.shape)

if __name__ == "__main__":
    process()
