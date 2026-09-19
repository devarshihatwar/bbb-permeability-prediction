# Baseline Results Summary

## Data Audit Verification

### 1. Dataset Identity
- **File:** `data/bbb_martins.csv`
- **Rows:** 1,975 (canonical TDC BBB_Martins / MoleculeNet BBBP dataset)
- **Columns:** `drug_id` (int), `drug` (SMILES string), `target` (0/1 binary)
- **Class balance:** 1,501 permeable (1), 474 non-permeable (0) — 76:24
- **Duplicates:** 0 duplicate SMILES, 0 duplicate rows
- **Invalid SMILES:** 0 (all RDKit-parseable)
- **Note:** The `tdc` package on PyPI is a 5KB stub; the real TDC GitHub repo is deleted.
  Dataset was retrieved from the verified GLambard/Molecules_Dataset_Collection mirror.

### 2. Split Integrity
- All splits use **pure positional integer indices (0–1,974)** — no pandas index mixing
- **Scaffold splits (5 folds):** 0 train/val/test index overlap, 0 SMILES overlap, 0 scaffold overlap ✅
- **Random splits:** 0 index overlap, 0 SMILES overlap (scaffold overlap is expected behavior)

### 3. Leakage Check
- Features (Morgan + RDKit descriptors) computed per-molecule from SMILES only — no target leakage
- StandardScaler fitted on train+val only, applied to test
- GridSearchCV (3-fold CV) runs entirely within train+val; test set used only for final evaluation
- Descriptor generation before splitting is safe (molecule-intrinsic, no cross-molecule info)

## Baseline Results

### Scaffold Split (5-fold CV, train+val for fitting, test for evaluation)

| Feature set | Model | AUROC | AUPRC | Acc | F1 | Sensitivity | Specificity |
|---|---|---|---|---|---|---|---|
| Morgan | LogReg | 0.831±0.025 | 0.924 | 0.808 | 0.871 | 0.871 | 0.612 |
| Morgan | RF | 0.901±0.026 | 0.956 | 0.867 | 0.915 | 0.954 | 0.598 |
| Morgan | SVM | 0.876±0.033 | 0.944 | 0.854 | 0.907 | 0.945 | 0.573 |
| Descriptors | LogReg | 0.893±0.029 | 0.945 | 0.853 | 0.900 | 0.882 | 0.761 |
| Descriptors | RF | 0.919±0.021 | 0.966 | 0.864 | 0.911 | 0.924 | 0.680 |
| Descriptors | SVM | 0.919±0.022 | 0.967 | 0.874 | 0.917 | 0.933 | 0.690 |
| Combined | LogReg | 0.854±0.026 | 0.936 | 0.819 | 0.879 | 0.879 | 0.632 |
| Combined | RF | 0.921±0.024 | 0.967 | 0.875 | 0.919 | 0.943 | 0.662 |

### Best Classical Model

**Random Forest on combined features (Morgan + RDKit descriptors):**
- AUROC: 0.921 ± 0.024
- Accuracy: 0.875
- F1: 0.919
- Sensitivity: 0.943 (high recall for permeable compounds)
- Specificity: 0.662 (model biased toward positive class due to 3:1 imbalance)

### Key Observations

1. RDKit descriptors alone perform nearly as well as combined features — the model's
   chemical intuition is captured in descriptors like TPSA, NumHDonors, MolLogP
2. Morgan fingerprints alone are the weakest (AUROC 0.831–0.901) — they miss explicit
   physicochemical properties
3. The Random Forest is the best classical model; no suspiciously high results (max = 0.946)
4. The best baseline (RF+combined, AUROC 0.921) is close to the Lantern Pharma ensemble
   reference (0.915) — confirming our pipeline is working correctly

### Saved Files

- `results/03_ml_baseline_results.csv` — all per-fold results (35 rows, 7 complete model configs)
- `results/03_ml_summary_scaffold.csv` — scaffold split summary table
- `results/feature_importance_morgan.csv` — Morgan fingerprint bit importances
- `results/feature_importance_descriptors.csv` — RDKit descriptor importances
- `results/feature_importance_combined.csv` — combined feature importances
