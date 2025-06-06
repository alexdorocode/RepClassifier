import os
import pandas as pd
import random
import sys

# Add the project root directory to PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

# Load your aa_sequences.csv
seq_file = "../DATASETS/raw_data/aa_sequences.csv"
df = pd.read_csv(seq_file)

# Dummy 3Di tokens generator
def generate_dummy_3di(length):
    # Simulate 3Di tokens with values between 0–26
    return " ".join(str(random.randint(0, 26)) for _ in range(length))

# Generate dummy 3Di column
df["3di"] = df["aa_seq"].apply(lambda seq: generate_dummy_3di(len(seq)))

# Define output directory and file
output_dir = "../DATASETS/dummy_data"
output_file = os.path.join(output_dir, "dummy_3di.csv")

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)
print(f"Output directory created or already exists: {output_dir}")

# Save to dummy_3di.csv
df[["uniprot_id", "3di"]].to_csv(output_file, index=False)
print(f"Dummy 3Di data saved to '{output_file}'.")