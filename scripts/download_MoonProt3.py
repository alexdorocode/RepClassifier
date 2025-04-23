import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import os

BASE_DETAIL_URL = "http://www.moonlightingproteins.org/protein_detail/?mpid="
UNIPROT_API_URL = "https://rest.uniprot.org/uniprotkb/"

# Output directory
OUTPUT_DIR = "../DATASETS"
os.makedirs(OUTPUT_DIR, exist_ok=True)

sequence_data = []
go_annotation_data = []

def get_uniprot_id(mpid):
    """Extract UniProt ID from MoonProt protein detail page"""
    url = f"{BASE_DETAIL_URL}{mpid}"
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(f"Failed to access {url}")
    soup = BeautifulSoup(response.text, "html.parser")
    uniprot_link = soup.find("a", href=lambda href: href and "uniprot.org" in href)
    
    if not uniprot_link:
        return None

    # Extract only the UniProt ID part (before space or parenthesis)
    full_text = uniprot_link.text.strip()
    clean_id = full_text.split()[0]  # e.g., "Q43155"
    reviwed = "Reviewed" in full_text
    unreviewed = "Unreviewed" in full_text
    return clean_id, reviwed, unreviewed


def fetch_uniprot_json(uniprot_id):
    """Fetch UniProt JSON using UniProt ID"""
    url = f"{UNIPROT_API_URL}{uniprot_id}.json"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        raise ValueError(f"Could not fetch UniProt JSON for {uniprot_id}")

def parse_uniprot_json(uniprot_id, reviewed, data):
    """Extract sequence and GO annotations from UniProt JSON"""
    sequence = data.get("sequence", {}).get("value", "")
    length = data.get("sequence", {}).get("length", 0)

    sequence_data.append({
        "UniProt IDs": uniprot_id,
        "Reviewed": reviewed,
        "Sequence Length": length,
        "Validate Length": length == len(sequence),
        "Amino Acid Sequence": sequence
    })

    # Extract GO annotations
    go_existance = False
    for ref in data.get("uniProtKBCrossReferences", []):
        if ref.get("database") == "GO":
            go_id = ref.get("id")
            properties = ref.get("properties", [])

            go_term = None
            go_evidence = None
            go_category = None

            for prop in properties:
                key = prop.get("key")
                value = prop.get("value")
                if key == "GoTerm":
                    go_term = value
                elif key == "GoEvidenceType":
                    go_evidence = value

            # Infer GO Category from go_term
            if go_term and ":" in go_term:
                aspect_letter = go_term.split(":")[0].strip().upper()
                if aspect_letter == "F":
                    go_category = "MF"
                elif aspect_letter == "P":
                    go_category = "BP"
                elif aspect_letter == "C":
                    go_category = "CC"

            # Save only if we have at least a term or evidence
            if go_id:
                go_existance = True
                go_annotation_data.append({
                    "UniProt IDs": uniprot_id,
                    "GO Annotation": go_id,
                    "GO Evidence": go_evidence.split(":")[0] if go_evidence else None,
                    "GoTerm": go_term,
                    "GO Category": go_category
                })
            else:
                print(f"Missing GO ID for UniProt ID {uniprot_id}")

    return len(sequence) == length, go_existance


def main():
    """Main function to process MPIDs and save data"""
    count_reviewed = 0
    count_unreviewed = 0
    count_non_reviewed_information = 0
    valid_length_count = 0
    without_go_count = 0
    unreviewed_ids = []
    non_reviewed_inf_ids = []

    for mpid in range(2, 516):
        print(f"[{mpid}/515] Processing MPID {mpid}")
        try:
            # Fetch UniProt ID from MoonProt page
            uniprot_id, reviewed, unreviewed = get_uniprot_id(mpid)
            if not uniprot_id:
                print(f"MPID {mpid}: UniProt ID not found.")
                continue

            # Track reviewed/unreviewed counts
            if reviewed:
                count_reviewed += 1
            elif unreviewed:
                count_unreviewed += 1
                unreviewed_ids.append(uniprot_id)
            elif not reviewed and not unreviewed:
                count_non_reviewed_information += 1
                non_reviewed_inf_ids.append(uniprot_id)
            else:
                raise ValueError(f"MPID {mpid}: Invalid reviewed/unreviewed status.")

            # Fetch UniProt JSON
            json_data = fetch_uniprot_json(uniprot_id)
            length_validation, go_existance = parse_uniprot_json(uniprot_id, reviewed, json_data)

            # Track valid sequence lengths and GO annotations
            if length_validation:
                valid_length_count += 1
            if not go_existance:
                without_go_count += 1

        except Exception as e:
            print(f"Error processing MPID {mpid}: {e}")
        time.sleep(1)

    # Display summary
    print("\n\n====================")
    print("\nProcessing Summary:")
    print("====================")
    print(f" - Total MPIDs processed: {len(sequence_data)}")
    print(f" - Total protiens obtained: {count_reviewed + count_unreviewed + count_non_reviewed_information}")
    print(f" - Total valid sequence lengths: {valid_length_count}")
    print(f" - Total proteins without GO annotations: {without_go_count}")
    
    print("\n - Reviewed/Unreviewed counts:")
    print(f" - Total reviewed proteins: {count_reviewed}")
    print(f" - Total unreviewed proteins: {count_unreviewed}")
    print(f" - Total proteins with non-reviewed information: {count_non_reviewed_information}")

    print("\n - Unreviewed/Non-reviewed UniProt IDs:")
    print(f" - Unreviewed UniProt IDs: {', '.join(unreviewed_ids)}")
    print(f" - Non-reviewed information UniProt IDs: {', '.join(non_reviewed_inf_ids)}")

    # Save outputs with headers
    pd.DataFrame(sequence_data).to_csv(
        os.path.join(OUTPUT_DIR, "MoonProt3_sequences.csv"), index=False, header=True
    )
    pd.DataFrame(go_annotation_data).to_csv(
        os.path.join(OUTPUT_DIR, "MoonProt3_go_anotations.csv"), index=False, header=True
    )
    print("\n✅ Saved MoonProt3_sequences.csv and MoonProt3_go_anotations.csv in ../DATASETS/")

if __name__ == "__main__":
    main()
