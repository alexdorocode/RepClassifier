# Final commit – Master’s Thesis by Àlex Domínguez Roig

import pandas as pd # type: ignore
import numpy as np # type: ignore
import os
from collections import defaultdict
from datetime import date
import torch # type: ignore

from preprocess.embeddings_geokg.train_go_autoencoder_wandb import GOAutoencoder

def generate_go_embeddings_per_protein(config):
    selected_categories = None
    if config.get("categories_to_embed"):
        selected_categories = [cat.strip() for cat in config["categories_to_embed"].split(",")]
        if config.get("two_letter_categories"):
            two_to_one = {"CC": "C", "MF": "F", "BP": "P"}
            selected_categories = [two_to_one.get(cat, cat) for cat in selected_categories]

    print("Loading entity2id mapping...")
    entity2id = {}
    df_entities = pd.read_csv(config["entity_map"], sep="\t")
    for idx, row in df_entities.iterrows():
        entity2id[row["term"]] = idx

    print("Loading embeddings from:", config["embedding_folder"])
    if config["embedding_dimentions"] not in ["50", "100", "200", "500", "1000"]:
        raise ValueError("Invalid embedding dimensions. Choose from 50, 100, 200, 500, or 1000.")
    embedding_path = os.path.join(config["embedding_folder"], f"GeOKG_{config['embedding_dimentions']}dim.npy")
    embedding_matrix = np.load(embedding_path)

    print("Loading annotations from:", config["input_csv"])
    df = pd.read_csv(config["input_csv"])
    print(f"Loaded {len(df[config['id_col']].unique())} entries from the input CSV")

    if selected_categories and config.get("category_col"):
        df = df[df[config["category_col"]].isin(selected_categories)]
        print(f"Filtered to {len(df)} entries matching categories: {selected_categories}")

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

    output_dir = f"../DATASETS/embeddings/GeOKG/{date.today()}"
    os.makedirs(output_dir, exist_ok=True)

    # Build filename suffix
    suffix = f"_{config['embedding_dimentions']}dim"
    if config["dimensionality_reduction"] == "autoencoder":
        suffix += f"_go_ae_IN_{config['input_dim']}dim_OUT_{config['output_dim']}dim"
    suffix += f"_{config['aggregation_strategy']}"

    if selected_categories:
        suffix += "_" + "_".join(selected_categories) + "_go_terms"

    # Apply dimensionality reduction if needed
    if config["dimensionality_reduction"] == "autoencoder":
        print("Applying autoencoder dimensionality reduction...")
        ae_model_path = os.path.join(config["autoencoder_path"],
                                     f"geokg_IN_{config['input_dim']}dim_OUT_{config['output_dim']}dim.pt")
        if not os.path.exists(ae_model_path):
            raise FileNotFoundError(f"Autoencoder not found at {ae_model_path}")

        ae_model = GOAutoencoder(input_dim=config["input_dim"], latent_dim=config["output_dim"])
        ae_model.load_state_dict(torch.load(ae_model_path, map_location=torch.device('cpu')))
        ae_model.eval()
        encoder = ae_model.encoder

        reduced_protein_to_vectors = {}
        for pid, vectors in protein_to_vectors.items():
            if vectors:
                tensor = torch.tensor(np.array(vectors), dtype=torch.float32)
                with torch.no_grad():
                    encoded = encoder(tensor).numpy()
                reduced_protein_to_vectors[pid] = encoded
    else:
        print("No dimensionality reduction applied.")
        reduced_protein_to_vectors = protein_to_vectors

    # Apply reduction strategy
    if config["aggregation_strategy"] == "mean_pool":
        print("Applying mean pooling...")
        result = {pid: np.mean(vectors, axis=0) for pid, vectors in reduced_protein_to_vectors.items() if np.asarray(vectors).size > 0}
        df_out = pd.DataFrame.from_dict(result, orient="index")
        df_out.index.name = "UniProt_ID"
        output_path = os.path.join(output_dir, f"geokg_embeddings{suffix}.csv")
        df_out.to_csv(output_path)


    elif config["aggregation_strategy"] == "padding":
        print("Applying padding strategy...")
        max_len = config.get("max_go_terms_per_protein", 10)  # Default to 10 if not provided
        padded_result = {}
        for pid, vectors in reduced_protein_to_vectors.items():
            if len(vectors) > max_len:
                vectors = vectors[:max_len]
            else:
                padding = np.zeros((max_len - len(vectors), vectors[0].shape[0]))
                vectors = np.vstack([vectors, padding])
            
            vectors = np.array(vectors)
            padded_result[pid] = vectors.flatten()
        df_out = pd.DataFrame.from_dict(padded_result, orient="index")
        df_out.index.name = "UniProt_ID"
        output_path = os.path.join(output_dir, f"geokg_embeddings{suffix}.csv")
        df_out.to_csv(output_path)

    elif config["aggregation_strategy"] == "none":
        print("Saving raw GO term vectors without aggregation...")
        result = {pid: vectors for pid, vectors in reduced_protein_to_vectors.items() if vectors}
        output_path = os.path.join(output_dir, f"geokg_raw_embeddings{suffix}.npz")
        np.savez_compressed(output_path, **result)

    else:
        raise ValueError(f"Unsupported reduction strategy: {config['aggregation_strategy']}")

    print("Saved embeddings to:", output_path)
    if config["aggregation_strategy"] in ["mean_pool", "padding"]:
        print(f"The output file shape is: {df_out.shape}")
