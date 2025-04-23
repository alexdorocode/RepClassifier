import pandas as pd
import numpy as np
import argparse
import os
from collections import defaultdict
from datetime import date

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Extract GO embeddings using GeOKG")
parser.add_argument("input_csv", help="CSV file with GO annotations")
parser.add_argument("--id_col", default="UniProt ID", help="Column with UniProt IDs")
parser.add_argument("--go_col", default="GO Annotation", help="Column with GO term IDs (e.g., GO:0005737)")
parser.add_argument("--mean_pool", action="store_true", help="Mean pool embeddings per UniProt ID")
parser.add_argument("--embedding_path", default="../DATASETS/embeddings/GeOKG/goa_embedding/GeOKG_200dim.npy", help="Path to GeOKG pretrained .npy file")
parser.add_argument("--entity_map", default="../GeOKG/GeOKG/src_data/GO/entities.tsv", help="Path to entities.tsv file from GeOKG")
args = parser.parse_args()

# Load entity2id from entities.tsv
print("Loading entity2id mapping...")
entity2id = {}
df_entities = pd.read_csv(args.entity_map, sep="\t")
for idx, row in df_entities.iterrows():
    entity2id[row["term"]] = idx

# Load pretrained embeddings
print("Loading embeddings from:", args.embedding_path)
embedding_matrix = np.load(args.embedding_path)

# Load your GO annotation dataset
print("Loading annotations from:", args.input_csv)
df = pd.read_csv(args.input_csv)

# Collect GO embeddings per protein
print("Extracting embeddings...")
protein_to_vectors = defaultdict(list)
missing = 0
for _, row in df.iterrows():
    pid = row[args.id_col]
    go = row[args.go_col]
    if go in entity2id:
        emb = embedding_matrix[entity2id[go]]
        protein_to_vectors[pid].append(emb)
    else:
        missing += 1

print(f"Skipped {missing} GO terms not found in entity2id")

# Generate output
print("Generating output vectors...")
if args.mean_pool:
    result = {pid: np.mean(vectors, axis=0) for pid, vectors in protein_to_vectors.items() if vectors}
else:
    result = {pid: vectors for pid, vectors in protein_to_vectors.items() if vectors}

# Save output
output_dir = f"../DATASETS/embeddings/GeOKG/{date.today()}"
os.makedirs(output_dir, exist_ok=True)

if args.mean_pool:
    df_out = pd.DataFrame.from_dict(result, orient="index")
    df_out.index.name = "UniProt_ID"
    output_path = os.path.join(output_dir, "geokg_mean_embeddings.csv")
    df_out.to_csv(output_path)
else:
    # Save as npz with variable-length arrays
    output_path = os.path.join(output_dir, "geokg_raw_embeddings.npz")
    np.savez_compressed(output_path, **result)

print("Saved embeddings to:", output_path)
