#!/usr/bin/env python3
"""
Usage:
    python scripts/get_raw_aa_seq.py <input_file> --path <output_directory>

Description:
    This script retrieves amino acid sequences for UniProt IDs provided in the input CSV file.
    The input file must contain a column named 'uniprot_id'.
    The retrieved sequences are saved in a CSV file in the specified output directory.
    UniProt IDs that caused errors during retrieval are saved in a separate CSV file.

Arguments:
    <input_file>         Path to the input CSV file containing UniProt IDs.
    --path <output_directory>  Path to the directory where the output file will be saved.

Example:
    python scripts/get_raw_aa_seq.py uniprot_ids.csv --path ../DATASETS/raw_data
"""

import os
import sys
import pandas as pd
from Bio import ExPASy, SwissProt
from urllib.error import HTTPError

# Check for correct usage
if len(sys.argv) != 4 or sys.argv[2] != "--path":
    print(__doc__)
    sys.exit(1)

# Parse command-line arguments
input_file = sys.argv[1]
output_dir = sys.argv[3]

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)
print(f"Output directory created or already exists: {output_dir}")

# Retrieve amino acid sequences
def fetch_sequence(uniprot_id):
    try:
        handle = ExPASy.get_sprot_raw(uniprot_id)
        record = SwissProt.read(handle)
        return record.sequence
    except HTTPError:
        print(f"HTTPError: Could not fetch sequence for UniProt ID: {uniprot_id}")
        return None
    except Exception as e:
        print(f"Error: Could not fetch sequence for UniProt ID: {uniprot_id}. Error: {e}")
        return None

# Load UniProt IDs from the input file
print(f"Loading UniProt IDs from {input_file}...")
accessions_df = pd.read_csv(input_file)
if "uniprot_id" not in accessions_df.columns:
    raise ValueError("The input file must contain a column named 'uniprot_id'.")

uniprot_ids = accessions_df["uniprot_id"].dropna().unique()
print(f"Loaded {len(uniprot_ids)} unique UniProt IDs.")

# Fetch sequences
print("Retrieving amino acid sequences...")
seq_data = []
error_ids = []  # List to store UniProt IDs that caused errors
total_ids = len(uniprot_ids)
for idx, uid in enumerate(uniprot_ids, start=1):
    print(f"Processing {idx}/{total_ids}: UniProt ID {uid}")
    seq = fetch_sequence(uid)
    if seq:
        seq_data.append({"uniprot_id": uid, "aa_seq": seq})
    else:
        error_ids.append(uid)

# Save the sequences to a CSV file
print(f"Retrieved sequences for {len(seq_data)} UniProt IDs.")
seq_df = pd.DataFrame(seq_data)
output_file = os.path.join(output_dir, "aa_sequences.csv")
seq_df.to_csv(output_file, index=False)
print(f"Saved amino acid sequences to '{output_file}'.")

# Save the error IDs to a separate CSV file
if error_ids:
    print(f"Saving {len(error_ids)} UniProt IDs that caused errors to 'error_ids.csv'.")
    error_df = pd.DataFrame(error_ids, columns=["uniprot_id"])
    error_file = os.path.join(output_dir, "error_ids.csv")
    error_df.to_csv(error_file, index=False)
    print(f"Saved error UniProt IDs to '{error_file}'.")
else:
    print("No errors occurred during sequence retrieval.")