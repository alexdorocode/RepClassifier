import os
import pandas as pd

# Paths
base_path = "../DATASETS"
output_dir = os.path.join(base_path, "row_data")
os.makedirs(output_dir, exist_ok=True)
print(f"Output directory created or already exists: {output_dir}")

# Input files
moonprot_file = os.path.join(base_path, "moonprot3_uniprot_ids_list.csv")
predictor_file = os.path.join(base_path, "predictor_dataset.csv")
print(f"MoonProt file path: {moonprot_file}")
print(f"Predictor file path: {predictor_file}")

# Clean UniProt IDs with debugging
def clean_uniprot_ids(series):
    print("Cleaning UniProt IDs...")
    series = series.astype(str).str.strip()  # Remove leading/trailing whitespace
    series = series[series != ""]  # Remove empty strings

    # Remove invalid characters but keep the core ID
    series = series.str.replace(r"[^A-Za-z0-9-]", "", regex=True)
    print(f"Removed invalid characters. Remaining IDs: {len(series)}")

    # Filter by valid length (6-10 characters)
    invalid_length = series[~series.str.match(r"^[A-Za-z0-9-]{6,10}$")]
    print(f"Discarded {len(invalid_length)} IDs due to invalid length: {invalid_length.tolist()}")
    series = series[series.str.match(r"^[A-Za-z0-9-]{6,10}$")]

    print(f"Cleaned {len(series)} UniProt IDs.")
    return series.drop_duplicates()

# Load and clean IDs
print("Loading and cleaning MoonProt IDs...")
moonprot_ids = clean_uniprot_ids(pd.read_csv(moonprot_file)["UniProt IDs"])
print(f"Loaded {len(moonprot_ids)} unique MoonProt IDs.")

print("Loading and cleaning Predictor dataset...")
predictor_df = pd.read_csv(predictor_file)
predictor_df["UniProt IDs"] = clean_uniprot_ids(predictor_df["UniProt IDs"])
print(f"Loaded {len(predictor_df)} entries from Predictor dataset.")

# Save all unique IDs
print("Saving all unique UniProt IDs...")
all_ids = pd.Series(pd.concat([moonprot_ids, predictor_df["UniProt IDs"]]).unique(), name="uniprot_id")
all_ids.to_csv(os.path.join(output_dir, "uniprot_ids.csv"), index=False)
print(f"Saved {len(all_ids)} unique UniProt IDs to 'uniprot_ids.csv'.")

# Generate labeled dataset
print("Generating labeled dataset...")
predictor_df = predictor_df.dropna(subset=["Class"])
print(f"Removed entries with missing 'Class'. Remaining: {len(predictor_df)}.")
predictor_df = predictor_df.drop_duplicates(subset=["UniProt IDs", "Class"])
print(f"Removed duplicate UniProt ID-Class pairs. Remaining: {len(predictor_df)}.")
duplicated = predictor_df["UniProt IDs"][predictor_df["UniProt IDs"].duplicated(keep=False)]
predictor_df = predictor_df[~predictor_df["UniProt IDs"].isin(duplicated)]
print(f"Removed UniProt IDs with conflicting classes. Remaining: {len(predictor_df)}.")

predictor_df["class"] = False
predictor_df.loc[predictor_df["UniProt IDs"].isin(moonprot_ids), "class"] = True
print("Assigned class labels to Predictor dataset.")

# Add MoonProt-only entries
print("Adding MoonProt-only entries...")
moonprot_only = moonprot_ids[~moonprot_ids.isin(predictor_df["UniProt IDs"])]
moonprot_only_df = pd.DataFrame({"uniprot_id": moonprot_only, "class": True})
print(f"Added {len(moonprot_only_df)} MoonProt-only entries.")

labeled_df = pd.concat([
    predictor_df.rename(columns={"UniProt IDs": "uniprot_id"})[["uniprot_id", "class"]],
    moonprot_only_df
])
labeled_df = labeled_df.drop_duplicates()
labeled_df.to_csv(os.path.join(output_dir, "uniprot_ids_labeled.csv"), index=False)
print(f"Saved labeled dataset with {len(labeled_df)} entries to 'uniprot_ids_labeled.csv'.")