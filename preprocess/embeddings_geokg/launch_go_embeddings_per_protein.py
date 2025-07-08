# Final commit – Master’s Thesis by Àlex Domínguez Roig

import os
import json
from go_embeddings_per_protein import generate_go_embeddings_per_protein

# Base configuration
BASE_CONFIG = {
    "input_csv": "../DATASETS/BioData_backup_alex/go_annotations_with_category.csv",
    "id_col": "accession_code",
    "go_col": "go_id",
    "category_col": "category",
    "one_letter_categories": True,
    "two_letter_categories": False,
    "embedding_folder": "../DATASETS/embeddings/GeOKG/goa_embedding/",
    "entity_map": "../GeOKG/GeOKG/src_data/GO/entities.tsv",
    "autoencoder_path": "../DATASETS/embeddings/GeOKG/autoencoders/",
    "max_go_terms_per_protein": 10
}

# Settings
categories_list = [["C"], ["F"], ["P"], ["C", "F"], ["C", "F", "P"]]
embedding_dimensions = ["50", "100", "200", "500", "1000"]
big_embedding_dimensions = ["1000", "500", "200"]
aggregation_strategies = ["mean_pool", "padding"]

# Available autoencoders
available_autoencoders = {
    ("1000", "128"), ("1000", "32"),
    ("500", "128"), ("500", "64"), ("500", "32"),
    ("200", "128"), ("200", "64"), ("200", "32"),
}

# Launcher
for categories in categories_list:
    for embedding_dim in embedding_dimensions:
        
        # Create base config for this loop
        config = BASE_CONFIG.copy()
        config["categories_to_embed"] = ",".join(categories)
        config["embedding_dimentions"] = embedding_dim
        
        ### First: without AE
        config["dimensionality_reduction"] = "none"
        
        for agg_strategy in aggregation_strategies:
            
            if embedding_dim in big_embedding_dimensions and agg_strategy == "padding":
                # Skip padding for big embeddings
                continue
            else:
                # Run with mean pooling or padding
                print("-" * 20)
                config["aggregation_strategy"] = agg_strategy
                print(f"Running: categories={categories}, emb_dim={embedding_dim}, AE=None, agg={agg_strategy}")
                generate_go_embeddings_per_protein(config)

        ### Then: with AE if available
        for (in_dim, out_dim) in available_autoencoders:
            if in_dim == embedding_dim:
                config["dimensionality_reduction"] = "autoencoder"
                config["input_dim"] = int(in_dim)
                config["output_dim"] = int(out_dim)
                for agg_strategy in aggregation_strategies:
                    print("-" * 20)
                    config["aggregation_strategy"] = agg_strategy
                    print(f"Running: categories={categories}, emb_dim={embedding_dim}, AE=IN_{in_dim}_OUT_{out_dim}, agg={agg_strategy}")
                    generate_go_embeddings_per_protein(config)
