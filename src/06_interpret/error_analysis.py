"""
Error analysis: RF vs GNN on the matched scaffold split test set.
Identifies error groups, compares molecular properties, and finds
high-confidence disagreements.
"""
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

# Load matched predictions
preds = pd.read_csv("results/h2h_predictions.csv")
df = pd.read_csv("data/bbb_martins.csv")

y_true = preds["true_label"].values
rf_prob = preds["rf_prob"].values
gnn_prob = preds["gnn_prob"].values
rf_pred = preds["rf_pred"].values
gnn_pred = preds["gnn_pred"].values
smiles_list = preds["smiles"].tolist()

print(f"Total test molecules: {len(preds)}")
print(f"True negatives: {sum(y_true==0)}, True positives: {sum(y_true==1)}")

# Error groups
rf_fp = np.where((y_true == 0) & (rf_pred == 1))[0]
rf_fn = np.where((y_true == 1) & (rf_pred == 0))[0]
gnn_fp = np.where((y_true == 0) & (gnn_pred == 1))[0]
gnn_fn = np.where((y_true == 1) & (gnn_pred == 0))[0]

print(f"\n{'='*70}")
print("ERROR COUNTS")
print(f"{'='*70}")
print(f"RF  False Positives: {len(rf_fp)}  (predicted=1, actual=0)")
print(f"RF  False Negatives: {len(rf_fn)}  (predicted=0, actual=1)")
print(f"GNN False Positives: {len(gnn_fp)} (predicted=1, actual=0)")
print(f"GNN False Negatives: {len(gnn_fn)} (predicted=0, actual=1)")

# Compute molecular properties
def get_props(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {
        "MW": Descriptors.MolWt(mol),
        "logP": Crippen.MolLogP(mol),
        "TPSA": Descriptors.TPSA(mol),
        "HBD": Lipinski.NumHDonors(mol),
        "HBA": Lipinski.NumHAcceptors(mol),
        "nRotB": Lipinski.NumRotatableBonds(mol),
        "nHetero": Lipinski.NumHeteroatoms(mol),
        "nRing": Lipinski.RingCount(mol),
        "FractionCSP3": rdMolDescriptors.CalcFractionCSP3(mol),
    }

all_props = [get_props(smi) for smi in smiles_list]
props_df = pd.DataFrame(all_props)

# High-confidence disagreements
rf_high_pos_gnn_high_neg = np.where((rf_prob > 0.9) & (gnn_prob < 0.1))[0]
gnn_high_pos_rf_high_neg = np.where((gnn_prob > 0.9) & (rf_prob < 0.1))[0]

print(f"\n{'='*70}")
print("HIGH-CONFIDENCE DISAGREEMENTS")
print(f"{'='*70}")
print(f"RF confident positive (>0.9), GNN confident negative (<0.1): {len(rf_high_pos_gnn_high_neg)}")
print(f"GNN confident positive (>0.9), RF confident negative (<0.1): {len(gnn_high_pos_rf_high_neg)}")

# Model disagreement
rf_only_wrong = np.where((rf_pred != y_true) & (gnn_pred == y_true))[0]
gnn_only_wrong = np.where((gnn_pred != y_true) & (rf_pred == y_true))[0]
both_correct = np.where((rf_pred == y_true) & (gnn_pred == y_true))[0]
both_wrong = np.where((rf_pred != y_true) & (gnn_pred != y_true))[0]

print(f"\n{'='*70}")
print("MODEL DISAGREEMENTS")
print(f"{'='*70}")
print(f"Both correct:        {len(both_correct)}")
print(f"RF wrong, GNN right: {len(rf_only_wrong)}")
print(f"GNN wrong, RF right: {len(gnn_only_wrong)}")
print(f"Both wrong:          {len(both_wrong)}")

# Property comparison functions
def summarize_props(indices, label):
    if len(indices) == 0:
        print(f"  {label}: (none)")
        return
    sub = props_df.iloc[indices]
    print(f"  {label} (n={len(indices)}):")
    for col in ["MW","logP","TPSA","HBD","HBA","nRotB","nHetero","nRing","FractionCSP3"]:
        print(f"    {col:15s}: mean={sub[col].mean():.2f}  median={sub[col].median():.2f}  std={sub[col].std():.2f}")

print(f"\n{'='*70}")
print("PROPERTY COMPARISON")
print(f"{'='*70}")

print("\n--- RF Error Groups ---")
summarize_props(rf_fp, "RF False Positives")
summarize_props(rf_fn, "RF False Negatives")

print("\n--- GNN Error Groups ---")
summarize_props(gnn_fp, "GNN False Positives")
summarize_props(gnn_fn, "GNN False Negatives")

print("\n--- Comparison Groups ---")
summarize_props(np.where((y_true == 0) & (rf_pred == 0))[0], "RF True Negatives")
summarize_props(np.where((y_true == 1) & (rf_pred == 1))[0], "RF True Positives")
summarize_props(np.where((y_true == 0) & (gnn_pred == 0))[0], "GNN True Negatives")
summarize_props(np.where((y_true == 1) & (gnn_pred == 1))[0], "GNN True Positives")

# Save error details
error_rows = []
for idx in rf_fp:
    error_rows.append({"index": idx, "model": "RF", "error_type": "FP", "true": y_true[idx],
                       "rf_prob": rf_prob[idx], "gnn_prob": gnn_prob[idx],
                       "smiles": smiles_list[idx], **props_df.iloc[idx]})
for idx in rf_fn:
    error_rows.append({"index": idx, "model": "RF", "error_type": "FN", "true": y_true[idx],
                       "rf_prob": rf_prob[idx], "gnn_prob": gnn_prob[idx],
                       "smiles": smiles_list[idx], **props_df.iloc[idx]})
for idx in gnn_fp:
    error_rows.append({"index": idx, "model": "GNN", "error_type": "FP", "true": y_true[idx],
                       "rf_prob": rf_prob[idx], "gnn_prob": gnn_prob[idx],
                       "smiles": smiles_list[idx], **props_df.iloc[idx]})
for idx in gnn_fn:
    error_rows.append({"index": idx, "model": "GNN", "error_type": "FN", "true": y_true[idx],
                       "rf_prob": rf_prob[idx], "gnn_prob": gnn_prob[idx],
                       "smiles": smiles_list[idx], **props_df.iloc[idx]})

error_df = pd.DataFrame(error_rows)
error_df.to_csv("results/06_interpret_error_analysis.csv", index=False)
print(f"\nError details saved to: results/06_interpret_error_analysis.csv")

# Detailed high-confidence disagreements
print(f"\n{'='*70}")
print("DETAILED HIGH-CONFIDENCE DISAGREEMENTS")
print(f"{'='*70}")

print(f"\nRF confident positive (>0.9), GNN confident negative (<0.1):")
for idx in rf_high_pos_gnn_high_neg:
    p = props_df.iloc[idx]
    print(f"\n  Index {idx}: RF={rf_prob[idx]:.4f}, GNN={gnn_prob[idx]:.4f}, true={y_true[idx]}")
    print(f"    SMILES: {smiles_list[idx]}")
    print(f"    MW={p['MW']:.1f}, logP={p['logP']:.2f}, TPSA={p['TPSA']:.1f}, HBD={p['HBD']}, HBA={p['HBA']}, Hetero={p['nHetero']}, Rings={p['nRing']}")

print(f"\nGNN confident positive (>0.9), RF confident negative (<0.1):")
for idx in gnn_high_pos_rf_high_neg:
    p = props_df.iloc[idx]
    print(f"\n  Index {idx}: RF={rf_prob[idx]:.4f}, GNN={gnn_prob[idx]:.4f}, true={y_true[idx]}")
    print(f"    SMILES: {smiles_list[idx]}")
    print(f"    MW={p['MW']:.1f}, logP={p['logP']:.2f}, TPSA={p['TPSA']:.1f}, HBD={p['HBD']}, HBA={p['HBA']}, Hetero={p['nHetero']}, Rings={p['nRing']}")

print(f"\nDone. Error analysis saved to results/06_interpret_error_analysis.csv")
