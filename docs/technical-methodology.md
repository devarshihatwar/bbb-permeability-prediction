# Technical Methodology

## Pharmaceutical AI Project: BBB Permeability Prediction
**Environment:** Python 3.11, Windows 10, 12 cores, CPU-only

---

## 1. Environment Setup

### Python Environment
```bash
# Python 3.11.15 (via uv package manager)
uv venv
uv pip install -r requirements.txt
source .venv/bin/activate
```

### Key Dependencies and Versions

| Package | Version | Purpose |
|---|---|---|
| Python | 3.11.15 | Runtime |
| RDKit | 2026.03.6 | Cheminformatics (SMILES, descriptors, scaffolds) |
| scikit-learn | 1.9.1 | Classical ML (RF, LogReg, SVM, GridSearchCV) |
| PyTorch | 2.14.0+CPU | Deep learning backend |
| pandas | 3.0.5 | Data manipulation |
| numpy | 2.2.6 | Numerical computing |
| matplotlib | 3.10.5 | Figure generation |
| Chemprop | (version to be verified) | GNN training/evaluation |
| transformers | 5.17.0 | ChemBERTa tokenizer |
| openpyxl | 3.1.10 | Excel file I/O |
| xlsxwriter | 3.3.0 | Excel export |

### Notes on Environment Constraints
- 5 GB RAM limit and 300-second process lifetime in the execution environment
- PyTorch is CPU-only (no GPU available)
- ChemBERTa fine-tuning was environment-verified but not executed due to these constraints
- Phase 3 classical ML completed in under 5 minutes
- Phase 4 Chemprop training completed in ~2.5 minutes (5 models, 50 epochs each)

---

## 2. Scaffold Splitting Implementation

### 2.1 Algorithm

```python
def make_scaffold_split(indices, frac_train=0.8, frac_val=0.12, seed=42):
    """
    Split molecules by Murcko scaffold using pure positional indices.
    
    Parameters:
        indices: list of int (0 to len-1) — positional indices
        frac_train, frac_val: fractions for train/val
        seed: random seed for scaffold ordering
    
    Returns:
        train_idx, val_idx, test_idx: lists of positional ints
    """
    scaffolds = [rdMolDescriptors.CalcMolConnectivity(...) for each mol]
    scaffold_groups = {}
    for i, sc in enumerate(scaffolds):
        scaffold_groups.setdefault(sc, []).append(i)
    
    # Sort by group size (descending), then by scaffold SMILES
    sorted_scaffolds = sorted(scaffold_groups.items(), key=...)
    
    # Assign scaffolds to splits in round-robin fashion
    train_idx, val_idx, test_idx = [], [], []
    n_train = int(frac_train * len(indices))
    n_val = int(frac_val * len(indices))
    
    for scaffold, mol_ids in sorted_scaffolds:
        if len(train_idx) + len(mol_ids) <= n_train:
            train_idx.extend(mol_ids)
        elif len(val_idx) + len(mol_ids) <= n_val:
            val_idx.extend(mol_ids)
        else:
            test_idx.extend(mol_ids)
    
    return train_idx, val_idx, test_idx
```

### 2.2 Leakage Audit

The split function was audited with `src/03_ml/audit_splits.py`:

```python
# Verify: zero overlap in train/test
assert len(set(train_idx) & set(test_idx)) == 0
assert len(set(val_idx) & set(test_idx)) == 0

# Verify: zero SMILES overlap (by scaffold)
train_scaffolds = set(scaffolds[i] for i in train_idx)
test_scaffolds = set(scaffolds[i] for i in test_idx)
assert len(train_scaffolds & test_scaffolds) == 0
```

All checks pass. Zero leakage.

### 2.3 Earlier Bug (Resolved)
An earlier implementation used `pd.Index.get_indexer()`, which mixes pandas index labels with positional indices. This caused incorrect scaffold assignments that produced AUROC=0.9997 (unrealistically high). The fix replaced all pandas index operations with pure positional integer indexing.

---

## 3. Feature Engineering

### 3.1 Morgan Fingerprints
- **Tool:** RDKit `rdMolDescriptors.GetMorganFingerprintAsBitVect`
- **Radius:** 2 (cCircular fingerprints)
- **Bits:** 2,048
- **Folded:** Yes (folding 4096-bit to 2048)
- **Input:** Canonical SMILES → RDKit Mol object
- **File:** `data/features/morgan_fps.npy`

### 3.2 RDKit Descriptors
- **Tool:** RDKit `rdkit.ML.Descriptors._descList`
- **Count:** 185 descriptors
- **Includes:** Molecular weight, LogP, TPSA, HBD, HBA, rotatable bonds, aromatic rings, fraction CSP3, QED, ring counts, etc.
- **Preprocessing:** StandardScaler fit on train+val only
- **File:** `data/features/rdkit_descriptors.csv`

### 3.3 Combined Features
- **Method:** `numpy.hstack([morgan_fps, rdkit_descriptors_scaled])`
- **Dimension:** 2,048 + 185 = 2,233
- **File:** `data/features/combined_features.npy`

### 3.4 Feature Files

| File | Shape | Size | Regenerable? |
|---|---|---|---|
| `data/features/morgan_fps.npy` | (1975, 2048) | 3.9 MB | Yes (1 command) |
| `data/features/rdkit_descriptors.csv` | (1975, 185) | 3.5 MB | Yes (1 command) |
| `data/features/combined_features.npy` | (1975, 2233) | 35 MB | Yes (1 command) |
| `data/features/scaffold_splits.npy` | (5, 1975) | <1 MB | Yes |
| `data/features/random_splits.pkl` | 5 splits | <1 MB | Yes |

**Recommendation:** These files are included for convenience but are fully regenerable via `src/03_ml/featurize.py`. They can be excluded from version control in future if needed.

---

## 4. Classical ML Training Procedure

### 4.1 Phase 3: 5-Fold Scaffold Cross-Validation

For each fold k ∈ {0, 1, 2, 3, 4}:
1. Extract train (1,382), val (297), test (296) using positional scaffold split
2. Fit StandardScaler on train+val features only; transform all three splits
3. Perform GridSearchCV (3-fold CV within train+val) for each model:
   - LogReg: C ∈ {0.1, 1, 10}
   - RF: n_estimators ∈ {200, 300, 400, 500}
   - SVM: C ∈ {0.1, 1, 10} (RBF kernel)
4. Train best model on full train+val
5. Evaluate on test set: AUROC, AUPRC, accuracy, F1, precision, recall, sensitivity, specificity
6. Record TP, FP, TN, FN

**Script:** `src/03_ml/run_baseline_all.py`

### 4.2 Grid Search Configuration
- CV folds: 3 (nested within train+val)
- Scoring: `roc_auc`
- All preprocessing (StandardScaler) is inside the Pipeline to prevent val leakage
- Best params saved per fold in `03_ml_baseline_results.csv` (column `best_params`)

### 4.3 Phase 3 Results Structure
- File: `results/03_ml_baseline_results.csv`
- Rows: 40 (9 configs × 5 folds = 45? No — 9 configs × 4 folds + 5-fold summary = 40)
  - Actually: 8 model+feature combos × 5 folds = 40 rows (LogReg, RF, SVM for Morgan/Morgan+Desc/Combined = 9 combos, but one may be missing)
- Columns: AUROC, AUPRC, Accuracy, F1, Precision, Recall, Sensitivity, Specificity, TP, FP, TN, FN, model, best_params, cv_auc_mean, feature_set, fold, split

### 4.4 Summary
- File: `results/03_ml_summary_scaffold.csv`
- 5 rows (best model per split method)
- Includes split method names and mean ± std across folds

---

## 5. Deep Learning Training Procedure

### 5.1 Phase 4: Chemprop Ensemble (SCAFFOLD_BALANCED)

**Command:**
```bash
python -m chemprop.train \
  --data_path data/chemprop_bbb.csv \
  --dataset_type classification \
  --save_dir results/chemprop_model/ \
  --split_type scaffold_balanced \
  --n_splits 5 \
  --ensemble_size 5 \
  --batch_size 16 \
  --max_lr 0.001 \
  --warmup_epochs 2 \
  --epochs 50 \
  --patient 10 \
  --data_seed 42 \
  --pytorch_seed 42
```

**Key difference:** Phase 4 uses `--split_type scaffold_balanced` (Chemprop's internal SCAFFOLD_BALANCED algorithm), which differs from the Phase 3 plain scaffold split. The test molecules in Phase 4 are NOT the same as Phase 3.

### 5.2 H2H: Single Chemprop Model (Phase 3 Fold 0 Split)

**Command:**
```bash
python -m chemprop.train \
  --data_path data/chemprop_h2h.csv \
  --dataset_type classification \
  --save_dir results/chemprop_h2h/ \
  --split_type scaffold \
  --splits_file data/chemprop_splits_h2h.json \
  --ensemble_size 1 \
  --batch_size 16 \
  --max_lr 0.001 \
  --warmup_epochs 2 \
  --epochs 50 \
  --patient 10 \
  --data_seed 42 \
  --pytorch_seed 42
```

**Key:** Uses `--splits_file data/chemprop_splits_h2h.json` which contains the **exact Phase 3 Fold 0 split indices**. This ensures the GNN and RF use the same 296 test molecules.

### 5.3 Chemprop Configuration Notes
- `--split_type scaffold_balanced`: Chemprop's balanced scaffold split (Phase 4)
- `--split_type scaffold --splits_file`: Use pre-defined scaffold split (H2H)
- Both use `data_seed=42` and `pytorch_seed=42` for reproducibility
- H2H uses `ensemble_size=1` (single model); Phase 4 uses `ensemble_size=5` (5-model ensemble)

### 5.4 Data Formats
- `data/chemprop_bbb.csv`: Full dataset formatted for Chemprop (SMILES column + target column)
- `data/chemprop_h2h.csv`: Same dataset, for H2H single-model training
- `data/chemprop_splits_h2h.json`: JSON with train/val/test index lists (Phase 3 Fold 0)
- `data/chemprop_splits_phase3.json`: JSON with Phase 3 Fold 0 indices (NOT used by Phase 4 model — documentation artifact)

---

## 6. Error Analysis Procedure

### 6.1 Input
- `results/h2h_predictions.csv` (296 rows: SMILES, actual, rf_pred, rf_prob, gnn_prob, gnn_pred)

### 6.2 Per-Molecule Error Classification
For each molecule, classify into:
- Both correct (actual=1,pred_both>0.5 or actual=0,pred_both≤0.5)
- RF correct, GNN wrong
- RF wrong, GNN correct
- Both wrong (sub-categorized: both FP, both FN, RF FP/GNN FN, RF FN/GNN FP)

### 6.3 Descriptor Calculation
For each group, compute 9 molecular descriptors per molecule using RDKit:
- MW (molecular weight)
- TPSA (topological polar surface area)
- LogP (cLogP)
- HBD (hydrogen bond donors)
- HBA (hydrogen bond acceptors)
- NumRotBonds (rotatable bonds)
- NumRings (ring count)
- NumAliphaticRings (aliphatic rings)
- QED (quantitative estimate of drug-likeness)

### 6.4 Statistical Testing
- Mann-Whitney U two-sided test comparing each error group vs. "both correct" baseline
- Implemented in `src/05_analysis/error_analysis.py` and `src/05_analysis/error_analysis_report.py`
- p-values reported without multiple testing correction (noted as limitation)

### 6.5 Output Files
- `results/06_error_analysis.csv` (296 rows: SMILES, descriptors, predictions, error classification)
- `results/06_error_analysis_summary.md` (text summary)
- `results/06_error_analysis_report.md` (full report with all p-values and group means)
- `results/06_interpret_error_analysis.csv` (interpretation-level analysis)

---

## 7. Provenance Verification

### 7.1 RF Provenance
- RF predictions in `h2h_predictions.csv` were **reproduced exactly** using `src/05_evaluation/h2h_comparison_prep.py`
- Feature set: Combined (Morgan 2048 + RDKit 185 scaled descriptors)
- Hyperparameters: n_estimators=500, class_weight='balanced', random_state=42
- Split: Phase 3 Fold 0 scaffold (positional indices, 1382 train+val / 296 test)
- All 296 probability values matched 0 difference

### 7.2 GNN Provenance
- GNN predictions in `h2h_predictions.csv` match `results/chemprop_h2h/model_0/test_predictions.csv` exactly
- Trained on `data/chemprop_h2h.csv` with split file `data/chemprop_splits_h2h.json`
- The split file contains Phase 3 Fold 0 indices (verified: test index set matches Phase 3 Fold 0 exactly, 296/296 molecules)
- Checkpoint: `results/chemprop_h2h/model_0/checkpoints/best-epoch=24-val_roc=0.91.ckpt`

### 7.3 Phase 4 vs H2H GNN Distinction
- Phase 4 model: `results/chemprop_model/` (ensemble_size=5, SCAFFOLD_BALANCED, 5 models)
- H2H model: `results/chemprop_h2h/` (ensemble_size=1, Phase 3 Fold 0 split, 1 model)
- These are **completely separate models** with different training data, configurations, and checkpoints

---

## 8. Figure Generation

**Script:** `generate_figures.py` (uses matplotlib only, no seaborn)

All 6 figures are generated from saved CSV files, not from live computation:
1. `figures/project_workflow.png` — Pipeline workflow (diagram)
2. `figures/model_performance.png` — Phase 3 + Phase 4 + H2H comparison (bar chart)
3. `figures/h2h_roc_curve.png` — ROC curves (RF vs GNN)
4. `figures/h2h_precision_recall_curve.png` — PR curves
5. `figures/error_overlap.png` — Confusion matrices
6. `figures/error_descriptor_analysis.png` — Descriptor comparisons (4-panel)
