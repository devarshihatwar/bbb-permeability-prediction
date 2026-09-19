"""
Head-to-head comparison: RF (classical) vs Chemprop (GNN) on the SAME scaffold split.

Uses Phase 3's scaffold split (fold 0) for both models to ensure a fair,
identical train/val/test partition.
"""
import os, sys, json, pickle, warnings, time

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    f1_score, precision_score, recall_score, confusion_matrix
)
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv("data/bbb_martins.csv")
labels = df["target"].values
smiles = df["drug"].tolist()

# Load Phase 3 scaffold split (fold 0)
with open("data/features/scaffold_splits.pkl", "rb") as f:
    splits = pickle.load(f)
tr_idx, va_idx, te_idx = splits[0]
trainval_idx = tr_idx + va_idx

print(f"Exact Split (Phase 3 Scaffold Fold 0):")
print(f"  Train: {len(tr_idx)} molecules")
print(f"  Val: {len(va_idx)} molecules")
print(f"  Test: {len(te_idx)} molecules")
print(f"  Train+Val (combined): {len(trainval_idx)} molecules")
print(f"  Total: {len(tr_idx) + len(va_idx) + len(te_idx)}")

# Load features
morgan = np.load("data/features/morgan_fps.npy")
descriptors = pd.read_csv("data/features/rdkit_descriptors.csv").values
combined = np.load("data/features/combined_features.npy")

# === RF BASELINE (Combined features) ===
print(f"\n{'='*70}")
print("RF BASELINE (Combined features: Morgan + RDKit descriptors)")
print(f"{'='*70}")

X_trainval = combined[trainval_idx]
y_trainval = labels[trainval_idx]
X_test = combined[te_idx]
y_test = labels[te_idx]

scaler = StandardScaler()
X_trainval_s = scaler.fit_transform(X_trainval)
X_test_s = scaler.transform(X_test)

rf = RandomForestClassifier(
    n_estimators=500, max_depth=None, min_samples_leaf=1,
    class_weight="balanced", random_state=42, n_jobs=-1
)
rf.fit(X_trainval_s, y_trainval)
y_prob_rf = rf.predict_proba(X_test_s)[:, 1]
y_pred_rf = (y_prob_rf >= 0.5).astype(int)

cm = confusion_matrix(y_test, y_pred_rf, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()

rf_metrics = {
    "AUROC": float(roc_auc_score(y_test, y_prob_rf)),
    "AUPRC": float(average_precision_score(y_test, y_prob_rf)),
    "Accuracy": float(accuracy_score(y_test, y_pred_rf)),
    "F1": float(f1_score(y_test, y_pred_rf)),
    "Precision": float(precision_score(y_test, y_pred_rf, zero_division=0)),
    "Recall": float(recall_score(y_test, y_pred_rf, zero_division=0)),
    "Sensitivity": float(tp / (tp + fn) if (tp + fn) > 0 else 0),
    "Specificity": float(tn / (tn + fp) if (tn + fp) > 0 else 0),
    "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
}

print(f"  AUROC: {rf_metrics['AUROC']:.4f}")
print(f"  AUPRC: {rf_metrics['AUPRC']:.4f}")
print(f"  Accuracy: {rf_metrics['Accuracy']:.4f}")
print(f"  F1: {rf_metrics['F1']:.4f}")
print(f"  Precision: {rf_metrics['Precision']:.4f}")
print(f"  Recall: {rf_metrics['Recall']:.4f}")
print(f"  Sensitivity: {rf_metrics['Sensitivity']:.4f}")
print(f"  Specificity: {rf_metrics['Specificity']:.4f}")
print(f"  Confusion Matrix (TN, FP, FN, TP): {tn}, {fp}, {fn}, {tp}")
print(f"  Train+Val: {len(trainval_idx)}, Test: {len(te_idx)}")

# Save RF predictions
rf_results = pd.DataFrame({
    "smiles": [smiles[i] for i in te_idx],
    "true_label": y_test,
    "rf_prob": y_prob_rf,
    "rf_pred": y_pred_rf,
})
rf_results.to_csv("results/h2h_rf_predictions.csv", index=False)

# Write temp data file and splits for Chemprop
df_chemprop = df[["drug", "target"]].copy()
df_chemprop.to_csv("data/chemprop_h2h.csv", index=False)

splits_cp = [{"train": tr_idx, "val": va_idx, "test": te_idx}]
with open("data/chemprop_splits_h2h.json", "w") as f:
    json.dump(splits_cp, f)

print(f"\n{'='*70}")
print("CHEMPPROP GNN (same split)")
print(f"{'='*70}")
print(f"  Train: {len(tr_idx)}, Val: {len(va_idx)}, Test: {len(te_idx)}")
print(f"  Split file: data/chemprop_splits_h2h.json")
print(f"  Data file: data/chemprop_h2h.csv")
print(f"  Ready for Chemprop training.")
PYEOF
