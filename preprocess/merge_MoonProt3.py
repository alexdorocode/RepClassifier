import pandas as pd

# File paths
PREDICTOR_PATH = "../DATASETS/predictor_dataset.csv"
SEQUENCE_PATH = "../DATASETS/MoonProt3_sequences.csv"
ANNOTATION_PATH = "../DATASETS/MoonProt3_go_anotations.csv"

# Load datasets
predictor_df = pd.read_csv(PREDICTOR_PATH)
moonprot_seq_df = pd.read_csv(SEQUENCE_PATH)
moonprot_go_df = pd.read_csv(ANNOTATION_PATH)

# Clean column names
predictor_df.rename(columns=lambda x: x.strip(), inplace=True)
moonprot_seq_df.rename(columns=lambda x: x.strip(), inplace=True)
moonprot_go_df.rename(columns=lambda x: x.strip(), inplace=True)

# No need to infer GO category anymore — just use it directly
moonprot_go_df = moonprot_go_df[moonprot_go_df["GO Category"].isin(["BP", "CC", "MF"])]

from collections import Counter, defaultdict

def print_mismatch_summary(mismatches, matched):
    print("🔍 Mismatch Summary Report")
    print("="*35)
    print(f"✅ Matched proteins: {len(matched)}")
    print(f"❌ Mismatched proteins: {len(mismatches)}")

    seq_mismatch_count = 0
    go_mismatch_counts = {"GO BP Terms": 0, "GO CC Terms": 0, "GO MF Terms": 0}
    missing_total = {"GO BP Terms": Counter(), "GO CC Terms": Counter(), "GO MF Terms": Counter()}
    extra_total = {"GO BP Terms": Counter(), "GO CC Terms": Counter(), "GO MF Terms": Counter()}
    missing_protein_sets = {"GO BP Terms": set(), "GO CC Terms": set(), "GO MF Terms": set()}
    extra_protein_sets = {"GO BP Terms": set(), "GO CC Terms": set(), "GO MF Terms": set()}

    for m in mismatches:
        uid = m["UniProt IDs"]
        if m.get("Sequence Mismatch"):
            seq_mismatch_count += 1
        for go_type in ["GO BP Terms", "GO CC Terms", "GO MF Terms"]:
            missing = m.get(f"{go_type} - Missing in Predictor", [])
            extra = m.get(f"{go_type} - Extra in Predictor", [])

            if missing or extra:
                go_mismatch_counts[go_type] += 1
            if missing:
                missing_total[go_type].update(missing)
                missing_protein_sets[go_type].add(uid)
            if extra:
                extra_total[go_type].update(extra)
                extra_protein_sets[go_type].add(uid)

    print(f"\n🧬 Sequence mismatches: {seq_mismatch_count}")
    print("\n📊 GO Annotation Mismatches:")
    for go_type in ["GO BP Terms", "GO CC Terms", "GO MF Terms"]:
        print(f"  - {go_type}: {go_mismatch_counts[go_type]} mismatches")

    print("\n🆕 Number of GO terms missing in predictor (MoonProt3 additions):")
    for go_type in missing_total:
        num_missing_terms = len(missing_total[go_type])
        num_proteins = len(missing_protein_sets[go_type])
        print(f"  • {go_type}: {num_missing_terms} from {num_proteins} proteins")

    total_missing_proteins = len(set().union(*missing_protein_sets.values()))
    print(f"  🔢 Total proteins with missing GO terms: {total_missing_proteins}")

    print("\n🗑️ Number of terms extra in predictor (not found in MoonProt3):")
    for go_type in extra_total:
        num_extra_terms = len(extra_total[go_type])
        num_proteins = len(extra_protein_sets[go_type])
        print(f"  • {go_type}: {num_extra_terms} from {num_proteins} proteins")

    total_extra_proteins = len(set().union(*extra_protein_sets.values()))
    print(f"  🔢 Total proteins with extra GO terms: {total_extra_proteins}")

def extract_go_terms_by_type(df):
    """Group GO annotations by UniProt ID and GO category"""
    grouped = df.groupby(["UniProt IDs", "GO Category"])["GO Annotation"].apply(list).unstack(fill_value=[])
    grouped = grouped.rename(columns={"BP": "GO BP Terms", "CC": "GO CC Terms", "MF": "GO MF Terms"})
    return grouped

def check_existing_annotations():
    """Compare sequence and GO terms; report disjoint GO terms per mismatch (new or outdated)."""
    moonprot_full = moonprot_seq_df.merge(extract_go_terms_by_type(moonprot_go_df), how="left",
                                           left_on="UniProt IDs", right_index=True)

    matched = []
    mismatches = []

    for _, row in moonprot_full.iterrows():
        uid = row["UniProt IDs"]
        if uid in predictor_df["UniProt IDs"].values:
            pred_row = predictor_df[predictor_df["UniProt IDs"] == uid].iloc[0]

            mismatch_info = {"UniProt IDs": uid}
            has_any_mismatch = False

            # Sequence comparison
            seq1 = row["Amino Acid Sequence"]
            seq2 = pred_row["Amino Acid Sequence"]
            if seq1 != seq2:
                mismatch_info["Sequence Mismatch"] = True
                mismatch_info["MoonProt3 Sequence"] = seq1
                mismatch_info["Predictor Sequence"] = seq2
                has_any_mismatch = True

            # GO Term comparison with disjoint sets
            for col in ["GO BP Terms", "GO CC Terms", "GO MF Terms"]:
                moon_go = set(row.get(col, []) or [])
                pred_val = pred_row.get(col)
                if isinstance(pred_val, str):
                    pred_go = set(term.strip().split()[0] for term in pred_val.split(";") if term.strip())
                else:
                    pred_go = set()

                missing_in_pred = moon_go - pred_go
                extra_in_pred = pred_go - moon_go

                if missing_in_pred or extra_in_pred:
                    has_any_mismatch = True
                    mismatch_info[f"{col} - Missing in Predictor"] = sorted(missing_in_pred)
                    mismatch_info[f"{col} - Extra in Predictor"] = sorted(extra_in_pred)

            if has_any_mismatch:
                mismatches.append(mismatch_info)
            else:
                matched.append(uid)

    return matched, mismatches


def create_new_predictor_dataset():
    """Create a predictor-style dataset with UniProt ID, sequence, GO CC, and GO MF terms"""
    go_terms = extract_go_terms_by_type(moonprot_go_df)
    new_predictor = moonprot_seq_df.merge(go_terms, how="left", left_on="UniProt ID", right_index=True)

    return new_predictor[["UniProt IDs", "Amino Acid Sequence", "GO CC Terms", "GO MF Terms"]]


# Example usage
if __name__ == "__main__":
    matched, mismatches = check_existing_annotations()
    print_mismatch_summary(mismatches, matched)
    print("🔎 Detailed mismatches saved to ../DATASETS/mismatched_summary.csv")

    new_predictor = create_new_predictor_dataset()
    new_predictor.to_csv("../DATASETS/new_predictor_dataset.csv", index=False)
    print("\n📁 Saved: ../DATASETS/new_predictor_dataset.csv")