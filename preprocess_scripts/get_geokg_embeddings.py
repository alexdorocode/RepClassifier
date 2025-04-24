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
parser.add_argument("--category_col", help="Column with GO categories (e.g., C, F, P)")
parser.add_argument("--categories_to_embed", help="Comma-separated list of categories to include (e.g., C,F)")
parser.add_argument("--one_letter_categories", action="store_true", help="Set if category codes are one-letter (C, F, P)")
parser.add_argument("--two_letter_categories", action="store_true", help="Set if category codes are two-letter (CC, MF, BP)")
parser.add_argument("--mean_pool", action="store_true", help="Mean pool embeddings per UniProt ID")
parser.add_argument("--embedding_folder", default="../DATASETS/embeddings/GeOKG/goa_embedding/", help="Path to GeOKG pretrained .npy file")
parser.add_argument("--embedding_dimentions", default="200", help="Number of the GeOKG pretrained dimentions, could be 50, 100, 200, 500, 1000")
parser.add_argument("--entity_map", default="../GeOKG/GeOKG/src_data/GO/entities.tsv", help="Path to entities.tsv file from GeOKG")
args = parser.parse_args()

# Normalize categories if provided
selected_categories = None
if args.categories_to_embed:
    selected_categories = [cat.strip() for cat in args.categories_to_embed.split(",")]
    if args.two_letter_categories:
        # Convert to one-letter code for filtering
        two_to_one = {"CC": "C", "MF": "F", "BP": "P"}
        selected_categories = [two_to_one.get(cat, cat) for cat in selected_categories]

# Load entity2id
print("Loading entity2id mapping...")
entity2id = {}
df_entities = pd.read_csv(args.entity_map, sep="\t")
for idx, row in df_entities.iterrows():
    entity2id[row["term"]] = idx

# Load embeddings
print("Loading embeddings from:", args.embedding_folder)

if args.embedding_dimentions not in ["50", "100", "200", "500", "1000"]:
    raise ValueError("Invalid embedding dimensions. Choose from 50, 100, 200, 500, or 1000.")

embedding_path = os.path.join(args.embedding_folder, f"GeOKG_{args.embedding_dimentions}dim.npy")
embedding_matrix = np.load(embedding_path)

# Load annotation data
print("Loading annotations from:", args.input_csv)
df = pd.read_csv(args.input_csv)

# Filter if categories are provided
if selected_categories and args.category_col:
    df = df[df[args.category_col].isin(selected_categories)]
    print(f"Filtered to {len(df)} entries matching categories: {selected_categories}")

# Collect embeddings
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

# Set up output path
output_dir = f"../DATASETS/embeddings/GeOKG/{date.today()}"
os.makedirs(output_dir, exist_ok=True)

# Build filename suffix
suffix = ""
if selected_categories:
    suffix = "_" + f"{args.embedding_dimentions}dim" + "_" + "_".join(selected_categories) + "_go_terms"

if args.mean_pool:
    df_out = pd.DataFrame.from_dict(result, orient="index")
    df_out.index.name = "UniProt_ID"
    output_path = os.path.join(output_dir, f"geokg_mean_embeddings{suffix}.csv")
    df_out.to_csv(output_path)
else:
    output_path = os.path.join(output_dir, f"geokg_raw_embeddings{suffix}.npz")
    np.savez_compressed(output_path, **result)

print("Saved embeddings to:", output_path)
