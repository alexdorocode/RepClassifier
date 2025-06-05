import pandas as pd
import requests
from io import StringIO
import os
import sys

def load_uniprot_ids(filepath):
    _, ext = os.path.splitext(filepath)
    if ext in [".csv"]:
        df = pd.read_csv(filepath)
    elif ext in [".tsv", ".txt"]:
        df = pd.read_csv(filepath, sep='\t')
    elif ext in [".xlsx", ".xls"]:
        df = pd.read_excel(filepath)
    else:
        raise ValueError("Unsupported file format. Use CSV, TSV, or Excel.")

    if "UniProt IDs" not in df.columns:
        raise ValueError("Column 'UniProt IDs' not found in the dataset")

    return df["UniProt IDs"].dropna().unique().tolist()

def query_uniprot(ids, batch_size=30):
    results = []

    for i in range(0, len(ids), batch_size):
        batch = ids[i:i+batch_size]
        query = " OR ".join(f"accession:{uid}" for uid in batch)

        url = "https://rest.uniprot.org/uniprotkb/search"
        params = {
            "query": query,
            "fields": "accession,reviewed,cc_structures",
            "format": "tsv",
            "size": batch_size
        }

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            df = pd.read_csv(StringIO(response.text), sep='\t')
            results.append(df)

        except requests.exceptions.HTTPError:
            print(f"[WARNING] Batch failed, retrying IDs individually starting from {batch[0]}")
            for uid in batch:
                try:
                    single_params = {
                        "query": f"accession:{uid}",
                        "fields": "accession,reviewed,cc_structures",
                        "format": "tsv",
                        "size": 1
                    }
                    single_resp = requests.get(url, params=single_params)
                    single_resp.raise_for_status()
                    single_df = pd.read_csv(StringIO(single_resp.text), sep='\t')
                    results.append(single_df)
                except requests.exceptions.RequestException:
                    print(f"[ERROR] Failed to retrieve info for UniProt ID: {uid}")

    if results:
        return pd.concat(results, ignore_index=True)
    else:
        raise RuntimeError("No data could be retrieved from UniProt.")


def get_low_quality_proteins(filepath):
    uniprot_ids = load_uniprot_ids(filepath)
    if not uniprot_ids:
        raise ValueError("No UniProt IDs found in the dataset.")
    df = query_uniprot(uniprot_ids)
    df["has_3d_structure"] = df["Comment[STRUCTURE]"].notna()
    low_quality = df[(df["Reviewed"] == False) & (df["has_3d_structure"] == False)]
    return low_quality["Entry"].tolist()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python filter_uniprot.py <path_to_dataset>")
        sys.exit(1)

    filepath = sys.argv[1]

    try:
        bad_proteins = get_low_quality_proteins(filepath)
        print("Proteins that are neither reviewed nor have 3D structure:")
        for prot in bad_proteins:
            print(prot)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
