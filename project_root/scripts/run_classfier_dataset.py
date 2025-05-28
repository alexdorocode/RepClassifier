import hydra  # type: ignore
from omegaconf import DictConfig  # type: ignore
import os
import sys
import gc
import torch
import pandas as pd

from project_root.dataset.dataset_config import DatasetConfigReader
from project_root.dataset.dataset_handler import DatasetHandler
from project_root.dataset.raw_dataset import RawDataset
from project_root.dataset.processed_dataset import ProcessedDataset

# Add the project root directory to PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
if project_root not in sys.path:
    sys.path.append(project_root)

@hydra.main(config_path="../config", config_name="config_experiment", version_base="1.3")
def test_sequence_loader(cfg: DictConfig):
    print("🔍 Initializing DatasetConfigReader and DatasetHandler...")
    config_reader = DatasetConfigReader(cfg)
    handler = DatasetHandler(config_reader)

    """
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



    loaders = handler.load_embedding_loaders()

    seq_loader = loaders['sequence_loader']
    go_loader = loaders['go_loader']

    # === SEQUENCE EMBEDDING LOADER ===
    print("🔍 Testing SequenceEmbeddingLoader...")
    model_name_collection = cfg.sequence_embedding.model_name
    target_dim_collection = cfg.sequence_embedding.target_dim

    for model_name in model_name_collection:
        for target_dim in target_dim_collection:
            print(f"\n🔍 Testing model: {model_name} with target dimension: {target_dim}")
            embeddings_df = seq_loader.load(
                model_name=model_name,
                target_dim=target_dim,
                use_autoencoder=True  # Assuming we want to use autoencoder compression
            )
            if embeddings_df.empty:
                print(f"⚠️ No embeddings found for model {model_name}. Skipping...")
                continue
            print(f"✅ Loaded shape: {embeddings_df.columns.tolist()} with {len(embeddings_df)} rows.")
            # Optional: save embeddings if needed
            del embeddings_df
            gc.collect()
            print("🧹 Cleaned up resources.")

    # === GO EMBEDDING LOADER ===
    print("🔍 Testing GOEmbeddingLoader...")
    go_models_collection = cfg.go_embeddings.model_name
    input_dim_collection = cfg.go_embeddings.input_dim
    emb_dim_collection = cfg.go_embeddings.emb_dim
    aggregated_dim_collection = cfg.go_embeddings.aggregated_dim
    aggregation_strategy_collection = cfg.go_embeddings.aggregation_strategy
    go_categories_collection = cfg.go_embeddings.go_categories
    use_autoencoder = cfg.go_embeddings.use_autoencoder

    print("🔍 GO models to test:", go_models_collection)
    print("🔍 Input dimensions to test:", input_dim_collection)
    print("🔍 Embedding dimensions to test:", emb_dim_collection)
    print("🔍 Aggregated dimensions to test:", aggregated_dim_collection)
    print("🔍 Aggregation strategies to test:", aggregation_strategy_collection)
    print("🔍 GO categories to test:", go_categories_collection)

    for model_name in go_models_collection:
        for input_dim in input_dim_collection:
            for emb_dim in emb_dim_collection:
                for aggregated_dim in aggregated_dim_collection:
                    for aggregation_strategy in aggregation_strategy_collection:
                        for go_categories in go_categories_collection:
                            print(f"\n🔍 Testing GO model: {model_name} with input_dim: {input_dim}, emb_dim: {emb_dim}, aggregated_dim: {aggregated_dim}, aggregation_strategy: {aggregation_strategy}, go_categories: {go_categories}")
                            embeddings_df = go_loader.load(
                                input_dim=input_dim,
                                emb_dim=emb_dim,
                                aggregated_dim=aggregated_dim,
                                aggregation_strategy=aggregation_strategy,
                                go_categories=go_categories,
                                use_autoencoder=use_autoencoder
                            )
                            if embeddings_df.empty:
                                print(f"⚠️ No embeddings found for model {model_name}, input_dim: {input_dim}, emb_dim: {emb_dim}, aggregated_dim: {aggregated_dim}, aggregation_strategy: {aggregation_strategy}, go_categories: {go_categories}. Skipping...")
                                continue
                            print(f"✅ Loaded shape: {embeddings_df.columns.tolist()} with {len(embeddings_df)} rows.")
                            # Optional: save embeddings if needed
                            del embeddings_df
                            gc.collect()
                            print("🧹 Cleaned up resources.")
    """
                            
if __name__ == "__main__":
    test_sequence_loader()
