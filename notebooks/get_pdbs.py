import pandas as pd
import requests
import os
from itertools import chain

# Load your UniProt ID dataset
df = pd.read_csv('./datasets/predictor_dataset.csv')
uniprot_ids = df['UniProt IDs'].dropna().unique()
uniprot_ids = [uid.split('|')[1] if '|' in uid else uid for uid in uniprot_ids]

# Load the SIFTS mapping file (skip metadata line)
sifts_file = './datasets/uniprot_pdb.tsv'
sifts_df = pd.read_csv(sifts_file, sep='\t', skiprows=1)

# Output directory for downloaded CIFs
output_dir = "./datasets/cif_structures"
os.makedirs(output_dir, exist_ok=True)

# Helper: clean and flatten grouped PDB lists
def split_pdbs(pdb_series):
    return list(set(chain.from_iterable(
        str(p).split(';') for p in pdb_series if pd.notna(p)
    )))

# Helper: validate PDB/CIF ID format (must be 4 alphanumeric characters)
def is_valid_cif_id(cif_id):
    return len(cif_id) == 4 and cif_id.isalnum()

# Group and clean the UniProt → CIF mapping
uniprot_to_cif = {
    uid: [cid for cid in split_pdbs(group['PDB']) if is_valid_cif_id(cid)]
    for uid, group in sifts_df[sifts_df['SP_PRIMARY'].isin(uniprot_ids)].groupby('SP_PRIMARY')
}

# Function to download CIF file only if not already present
def download_cif_if_missing(cif_id, outdir=output_dir):
    filepath = os.path.join(outdir, f"{cif_id}.cif")
    if os.path.exists(filepath):
        print(f"✅ {cif_id}.cif already exists, skipping download.")
        return  # Skip download
    url = f"https://files.rcsb.org/download/{cif_id}.cif"
    r = requests.get(url)
    if r.status_code == 200:
        with open(filepath, 'wb') as f:
            f.write(r.content)
    else:
        print(f"❌ Could not download {cif_id}.cif (status {r.status_code})")

# Start downloading CIFs
all_cifs = set()
uniprot_cif_map = {}
no_cif_uniprots = []

for i, uid in enumerate(uniprot_ids):
    cif_ids = uniprot_to_cif.get(uid, [])
    if not cif_ids:
        print(f"[{i+1}/{len(uniprot_ids)}] No CIFs found for {uid}")
        no_cif_uniprots.append(uid)
        continue

    for cid in cif_ids:
        download_cif_if_missing(cid)
        all_cifs.add(cid)

    uniprot_cif_map[uid] = cif_ids
    print(f"[{i+1}/{len(uniprot_ids)}] Done: {uid} → {len(cif_ids)} CIFs")

# Report summary
print(f"\n✅ Total unique CIFs downloaded: {len(all_cifs)}")
print(f"❌ Total UniProt IDs with no CIFs: {len(no_cif_uniprots)}")
print(f"✅ CIFs saved in: {output_dir}")

# Save the UniProt → CIF mapping
mapping_df = pd.DataFrame({
    'UniProt ID': list(uniprot_cif_map.keys()),
    'CIF IDs': list(uniprot_cif_map.values())
})

mapping_df.to_csv('./datasets/uniprot_to_cif_mapping.csv', index=False)

# Merge mapping info with original dataset
df['UniProt_clean'] = [uid.split('|')[1] if '|' in uid else uid for uid in df['UniProt IDs']]
df_no_cif = df[df['UniProt_clean'].isin(no_cif_uniprots)]
df_with_cif = df[df['UniProt_clean'].isin(uniprot_cif_map)]

# 1. Class distribution among proteins with no CIFs
no_cif_class_counts = df_no_cif['Class'].value_counts()
print("\n🔍 No CIFs found:")
print(no_cif_class_counts)

# 2. Distribution of CIF counts
cif_counts = {uid: len(cids) for uid, cids in uniprot_cif_map.items()}
cif_counts_series = pd.Series(cif_counts)

# Define thresholds
q25 = cif_counts_series.quantile(0.25)
q50 = cif_counts_series.quantile(0.50)
q75 = cif_counts_series.quantile(0.75)

# Count proteins by number of CIFs per threshold
below_25 = cif_counts_series[cif_counts_series <= q25].count()
below_50 = cif_counts_series[cif_counts_series <= q50].count()
below_75 = cif_counts_series[cif_counts_series <= q75].count()

print("\n📊 CIF count thresholds:")
print(f"25th percentile = {q25:.2f}")
print(f"50th percentile = {q50:.2f}")
print(f"75th percentile = {q75:.2f}")


# Load external reference datasets
moonprot_ids = pd.read_csv('./datasets/uniprot_ids_moonprot.csv', header=None).iloc[:, 0].tolist()
moondb_ids = pd.read_csv('./datasets/uniprot_ids_moondb.csv', header=None).iloc[:, 0].tolist()

# Clean format if needed
moonprot_ids = [uid.strip() for uid in moonprot_ids]
moondb_ids = [uid.strip() for uid in moondb_ids]

# Check origin of no-CIF UniProt IDs
no_cif_sources = {
    "MoonProt": [],
    "MoonDB": [],
    "Other": []
}

for uid in no_cif_uniprots:
    if uid in moonprot_ids:
        no_cif_sources["MoonProt"].append(uid)
    elif uid in moondb_ids:
        no_cif_sources["MoonDB"].append(uid)
    else:
        no_cif_sources["Other"].append(uid)

# Report
print("\n📁 Dataset origin for UniProt IDs with no CIFs:")
print(f"MoonProt: {len(no_cif_sources['MoonProt'])}")
print(f"MoonDB:   {len(no_cif_sources['MoonDB'])}")
print(f"Other:    {len(no_cif_sources['Other'])}")

# Print the top 10 UniProt IDs with most CIFs
top_cif_proteins = cif_counts_series.nlargest(10)
print("\n🔝 Top 10 UniProt IDs with most CIFs:" )
for uid, count in top_cif_proteins.items():
    print(f"{uid}: {count} CIFs")
 
