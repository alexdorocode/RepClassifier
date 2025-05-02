import pandas as pd
import numpy as np
import argparse
import os
import json
from collections import defaultdict
from datetime import date
import torch
import torch.nn as nn

from preprocess.embeddings_geokg.train_go_autoencoder_wandb import GOAutoencoder

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Extract GO embeddings using GeOKG")
parser.add_argument("config_file", help="Path to the JSON configuration file")
args = parser.parse_args()

# Load configuration from JSON file
with open(args.config_file, "r") as f:
    config = json.load(f)

# Normalize categories if provided
selected_categories = None
if config.get("categories_to_embed"):
    selected_categories = [cat.strip() for cat in config["categories_to_embed"].split(",")]
    if config.get("two_letter_categories"):
        # Convert to one-letter code for filtering
        two_to_one = {"CC": "C", "MF": "F", "BP": "P"}
        selected_categories = [two_to_one.get(cat, cat) for cat in selected_categories]

# Load entity2id
print("Loading entity2id mapping...")
entity2id = {}
df_entities = pd.read_csv(config["entity_map"], sep="\t")
for idx, row in df_entities.iterrows():
    entity2id[row["term"]] = idx

# Load embeddings
print("Loading embeddings from:", config["embedding_folder"])

if config["embedding_dimentions"] not in ["50", "100", "200", "500", "1000"]:
    raise ValueError("Invalid embedding dimensions. Choose from 50, 100, 200, 500, or 1000.")

embedding_path = os.path.join(config["embedding_folder"], f"GeOKG_{config['embedding_dimentions']}dim.npy")
embedding_matrix = np.load(embedding_path)

# Load annotation data
print("Loading annotations from:", config["input_csv"])
df = pd.read_csv(config["input_csv"])
print(f"Loaded {len(df[config["id_col"]])} entries from the input CSV")


# Filter if categories are provided
if selected_categories and config.get("category_col"):
    df = df[df[config["category_col"]].isin(selected_categories)]
    print(f"Filtered to {len(df)} entries matching categories: {selected_categories}")

# Collect embeddings
print("Extracting embeddings...")
protein_to_vectors = defaultdict(list)
missing = 0
for _, row in df.iterrows():
    pid = row[config["id_col"]]
    go = row[config["go_col"]]
    if go in entity2id:
        emb = embedding_matrix[entity2id[go]]
        protein_to_vectors[pid].append(emb)
    else:
        missing += 1

print(f"Skipped {missing} GO terms not found in entity2id")

# Set up output path
output_dir = f"../DATASETS/embeddings/GeOKG/{date.today()}"
os.makedirs(output_dir, exist_ok=True)

# Build filename suffix
suffix = f"_{config['embedding_dimentions']}dim_{config['reduction_strategy']}"
if selected_categories:
    suffix += "_" + "_".join(selected_categories) + "_go_terms"

embedding_dim = int(config["embedding_dimentions"])

if config["reduction_strategy"] == "autoencoder":
    print("Reducing using autoencoder...")
    ae_path = os.path.join(config["autoencoder_path"], f"geokg_{embedding_dim}dim_autoencoder.pt")
    if not os.path.exists(ae_path):
        raise FileNotFoundError(f"Autoencoder not found at {ae_path}")
    
    ae_model = GOAutoencoder(input_dim=embedding_dim, latent_dim=50)  # or your actual latent_dim
    ae_model.load_state_dict(torch.load(ae_path, map_location=torch.device('cpu')))
    ae_model.eval()

    encoder = ae_model.encoder

    result = {}
    for pid, vectors in protein_to_vectors.items():
        if vectors:
            vectors_np = np.array(vectors)
            tensor = torch.tensor(vectors_np, dtype=torch.float32)
            with torch.no_grad():
                encoded = encoder(tensor).numpy()
            result[pid] = np.mean(encoded, axis=0)

    df_out = pd.DataFrame.from_dict(result, orient="index")
    df_out.index.name = "UniProt_ID"
    output_path = os.path.join(output_dir, f"geokg_encoded_embeddings{suffix}.csv")
    df_out.to_csv(output_path)

elif config["reduction_strategy"] == "mean_pool":
    print("Applying mean pooling...")

    # Print protein_to_vectors lenghts
    for pid, vectors in protein_to_vectors.items():
        print(f"{pid}: {len(vectors)} vectors")

    result = {pid: np.mean(vectors, axis=0) for pid, vectors in protein_to_vectors.items() if vectors}
    df_out = pd.DataFrame.from_dict(result, orient="index")
    df_out.index.name = "UniProt_ID"
    output_path = os.path.join(output_dir, f"geokg_mean_embeddings{suffix}.csv")
    df_out.to_csv(output_path)

elif config["reduction_strategy"] == "none":
    print("Saving raw GO term vectors (no dimensionality reduction)...")
    result = {pid: vectors for pid, vectors in protein_to_vectors.items() if vectors}
    output_path = os.path.join(output_dir, f"geokg_raw_embeddings{suffix}.npz")
    np.savez_compressed(output_path, **result)

else:
    raise ValueError(f"Unsupported reduction strategy: {config['reduction_strategy']}")

print("Saved embeddings to:", output_path)

print(f"The output file length is: {df_out.shape}")