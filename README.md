# Comparative Machine Learning and Graph Neural Network Approaches for Blood–Brain Barrier Permeability Prediction

**A reproducible head-to-head evaluation of Random Forest vs. Chemprop GNN on the TDC BBB_Martins dataset.**

---

## TL;DR

This project evaluates blood–brain barrier (BBB) permeability prediction using the TDC BBB_Martins dataset (1,975 compounds). Three experimental phases were completed:

1. **Phase 3 — Classical ML baseline** (5-fold scaffold CV): Random Forest on combined Morgan + RDKit features achieved **AUROC 0.921 ± 0.024**, the best classical result.
2. **Phase 4 — Chemprop GNN** (SCAFFOLD_BALANCED, 5-model ensemble): **AUROC 0.935** (different split protocol — not directly comparable to Phase 3).
3. **Phase 5 — Matched head-to-head** (same Fold 0 scaffold split, n=296): RF outperformed the single GNN model (0.903 vs 0.867 AUROC), with complementary error patterns.

All splits use pure positional indices with verified zero leakage. No model training was performed during provenance audit or finalization. ChemBERTa fine-tuning is planned but not executed (environment constraints).

---

## Research Question

> Can a classical Random Forest model with hand-crafted molecular features match or outperform a graph neural network on BBB permeability prediction, and what do their complementary error patterns reveal about molecular properties that drive model failure?

---

## Dataset

| Property | Value |
|---|---|
| **Source** | TDC BBB_Martins / MoleculeNet BBBP (Martins et al., 2012) |
| **Molecules** | 1,975 |
| **Class balance** | 1,501 permeable (76%) / 474 non-permeable (24%) |
| **SMILES** | Stored in column `drug`; canonical RDKit SMILES |
| **Labels** | Column `target` (0 = non-permeable, 1 = permeable) |
| **Duplicates** | 0 duplicate SMILES, 0 duplicate rows, 0 invalid SMILES |
| **Features** | Morgan fingerprints (radius 2, 2048 bits) + 185 RDKit descriptors (217 combined) |

**Data quality:** Verified zero duplicates, zero invalid SMILES, zero leakage in scaffold splits.

**File:** `data/bbb_martins.csv`

---

## Methodology

### 3.1 Scaffold Splitting

All splits use **Murcko scaffold splitting** with **pure positional integer indices** (0–1974). This ensures:
- Zero train/test index overlap ✅
- Zero train/test SMILES overlap ✅
- Zero train/test scaffold overlap ✅

The split function was audited via `src/03_ml/audit_splits.py` — an earlier version had an index-mapping bug (mixing pandas index labels with positional indices) that produced spuriously high AUROC (0.9997). This was fixed by using pure positional integer indices throughout.

### 3.2 Feature Engineering

| Feature Set | Description | Dimension |
|---|---|---|
| Morgan | Morgan fingerprints (radius=2, nBits=2048) | 2,048 |
| Descriptors | RDKit 2D molecular descriptors | 185 |
| Combined | Morgan + RDKit descriptors | 2,233 |

Features are molecule-intrinsic (computed from SMILES only — no target information used). `StandardScaler` is fitted on train+val only.

### 3.3 Models

| Model | Library | Key Config |
|---|---|---|
| Logistic Regression | scikit-learn | C=1.0, class_weight=None |
| Random Forest | scikit-learn | n_estimators=300, class_weight=None (Phase 3) |
| SVM | scikit-learn | RBF kernel, C=1.0 |
| Chemprop GNN | chemprop 2.0+ | MPNN, 4-gate, depth=4, hidden=300 |

### 3.4 Experimental Design

| Phase | Objective | Split | Metric Reporting |
|---|---|---|---|
| Phase 3 | Classical ML baseline | 5-fold scaffold CV | Mean ± std across 5 folds |
| Phase 4 | Chemprop GNN comparison | SCAFFOLD_BALANCED (5 folds) | Ensemble mean (5 models) |
| Phase 5 | Matched H2H | Single Phase 3 Fold 0 | Point estimate (n=296 test) |

> **Critical distinction:** Phase 3 uses plain scaffold splitting; Phase 4 uses Chemprop's `SCAFFOLD_BALANCED` protocol. These produce different splits. The Phase 4 results (AUROC 0.935) must **NOT** be directly compared with Phase 3 (AUROC 0.921) as if from the same protocol.

---

## Results

### Phase 3: Classical ML Baseline (5-Fold Scaffold CV)

| Feature Set | Model | AUROC | AUPRC | Accuracy | F1 | Sensitivity | Specificity |
|---|---|---|---|---|---|---|---|
| Morgan | LogReg | 0.831±0.025 | 0.924 | 0.808 | 0.871 | 0.871 | 0.612 |
| Morgan | RF | 0.901±0.026 | 0.956 | 0.867 | 0.915 | 0.954 | 0.598 |
| Morgan | SVM | 0.876±0.033 | 0.944 | 0.854 | 0.907 | 0.945 | 0.573 |
| Descriptors | LogReg | 0.893±0.029 | 0.945 | 0.853 | 0.900 | 0.882 | 0.761 |
| Descriptors | RF | 0.919±0.021 | 0.966 | 0.864 | 0.911 | 0.924 | 0.680 |
| Descriptors | SVM | 0.919±0.022 | 0.967 | 0.874 | 0.917 | 0.933 | 0.690 |
| Combined | LogReg | 0.854±0.026 | 0.936 | 0.819 | 0.879 | 0.879 | 0.632 |
| **Combined** | **RF** | **0.921±0.024** | **0.967** | **0.875** | **0.919** | **0.943** | **0.662** |

**Best classical model:** Random Forest on combined Morgan + RDKit features (AUROC 0.921 ± 0.024).

### Phase 4: Chemprop GNN (5-Fold SCAFFOLD_BALANCED Ensemble)

> ⚠️ Uses a **different split protocol** (SCAFFOLD_BALANCED) from Phase 3. Not directly comparable.

| Metric | Value |
|---|---|
| AUROC | 0.935 |
| AUPRC | 0.983 |
| Accuracy | 0.916 |
| F1 | 0.948 |
| Sensitivity | 0.958 |
| Specificity | 0.732 |
| TP/FP/TN/FN | 230/15/41/10 |

Individual model AUROCs: 0.898–0.949 (5 models, ensemble mean = 0.935).

### Phase 5: Matched Head-to-Head (Fold 0, n=296)

Both RF and Chemprop GNN were evaluated on the **exact same** Phase 3 Fold 0 scaffold test set (296 molecules, 60 negative / 236 positive).

| Metric | RF (Combined) | GNN (Chemprop) | Difference |
|---|---|---|---|
| **AUROC** | **0.9031** | 0.8669 | RF +0.036 |
| AUPRC | 0.9726 | 0.9602 | RF +0.012 |
| Accuracy | 0.8750 | 0.8581 | RF +0.017 |
| F1 | 0.9246 | 0.9114 | RF +0.013 |
| Precision | 0.8902 | 0.9076 | GNN +0.017 |
| Sensitivity | 0.9619 | 0.9153 | RF +0.047 |
| Specificity | 0.5333 | 0.6333 | GNN +0.100 |
| TP/FP/TN/FN | 227/28/32/9 | 216/22/38/20 | — |

**RF hyperparameters (H2H):** n_estimators=500, class_weight='balanced', random_state=42 (different from Phase 3's n_estimators=300, no class_weight).

**GNN configuration (H2H):** Single Chemprop model (ensemble_size=1), data_seed=42, pytorch_seed=42, trained on the exact Phase 3 Fold 0 split. Checkpoint: `results/chemprop_h2h/model_0/checkpoints/best-epoch=24-val_roc=0.91.ckpt`.

### Error Analysis

| Error Type | RF Only | GNN Only | Both Error |
|---|---|---|---|
| False Positives (pred 1, true 0) | 7 | 1 | 21 |
| False Negatives (pred 0, true 1) | 3 | 14 | 6 |

- **Total errors:** 52 (RF: 37, GNN: 42)
- **Shared errors:** 27
- **Disagreements:** 25 (RF wins 10, GNN wins 15)
- **High-confidence disagreements (≥0.9):** 0

**Statistically significant differences (Mann-Whitney U, p < 0.05):**

| Error Group | Property | Group Mean | Baseline Mean | p-value |
|---|---|---|---|---|
| RF FP | QED | 0.55 | 0.66 | 0.0002 *** |
| RF FP | TPSA (Å²) | 72.8 | 62.8 | 0.028 * |
| RF FN | MW (Da) | 428.3 | 335.4 | 0.007 ** |
| RF FN | TPSA (Å²) | 119.9 | 62.8 | 0.0009 *** |
| RF FN | NumHDonors | 2.67 | 1.30 | 0.0006 *** |
| RF FN | NumHAcc | 7.33 | 4.08 | 0.0005 *** |
| GNN FP | QED | 0.57 | 0.66 | 0.003 ** |
| GNN FN | TPSA (Å²) | 93.2 | 62.8 | 0.0002 *** |
| GNN FN | NumHDonors | 1.90 | 1.30 | 0.002 ** |
| GNN FN | NumHAcc | 5.65 | 4.08 | 0.0015 ** |

---

## Key Findings

1. **RF is the strongest classical model** (AUROC 0.921 ± 0.024, 5-fold scaffold CV) using combined Morgan + RDKit features.
2. **On the matched Fold 0 comparison**, RF outperforms the single GNN (0.903 vs 0.867 AUROC), but this is one fold — the Phase 4 GNN ensemble (0.935) was trained on a different split and is not directly comparable.
3. **Both models have complementary error patterns** — 25 of 52 errors are model-specific. RF is more sensitive (0.962); GNN has higher specificity (0.633).
4. **Chemically meaningful failure patterns:** Both models struggle with large, polar molecules (high TPSA, high MW). RF false positives concentrate on low-QED compounds; GNN false negatives on high-TPSA compounds.
5. **Zero high-confidence disagreements** — when both models are ≥90% confident, they always agree.

---

## Limitations

1. **Single-fold H2H:** The matched comparison uses one Fold 0 scaffold split (n=296 test), limiting generalizability.
2. **RF hyperparameter difference:** The H2H RF used n_estimators=500, class_weight='balanced' (different from Phase 3's n_estimators=300, no class_weight).
3. **Phase 4 vs Phase 3 split difference:** Phase 4 uses SCAFFOLD_BALANCED, not identical to Phase 3's plain scaffold split. The Phase 4 ensemble AUROC (0.935) cannot be directly compared with Phase 3 RF AUROC (0.921).
4. **Class imbalance:** 76:24 positive:negative ratio inflates accuracy and F1 metrics across all models.
5. **ChemBERTa not trained:** Environment constraints (300s process lifetime, 5GB RAM) prevented ChemBERTa fine-tuning. See `src/07_chemberta/IMPLEMENTATION_PLAN.md`.

---

## Reproducibility

### Environment
- Python 3.11.15
- RDKit 2026.03.6
- scikit-learn 1.9.1
- PyTorch 2.14.0+CPU
- pandas 3.0.5
- transformers 5.17.0

### Reproduction Steps

```bash
# Phase 3: Classical ML baseline (5-fold scaffold CV)
python src/03_ml/featurize.py        # Generate features
python src/03_ml/train_baseline.py   # Train all models

# Phase 4: Chemprop GNN
python -m chemprop.train --data_path data/chemprop_bbb.csv \
  --dataset_type classification --save_dir results/chemprop_model \
  --split_type SCAFFOLD_BALANCED --ensemble_size 5 \
  --data_seed 42 --pytorch_seed 42 \
  --epochs 50 --batch_size 16 --max_lr 0.001 --warmup_epochs 2

# Phase 5: Matched H2H (Fold 0)
python src/05_evaluation/h2h_comparison_prep.py    # Train RF + prepare Chemprop data
python -m chemprop.train --data_path data/chemprop_h2h.csv \
  --dataset_type classification --save_dir results/chemprop_h2h \
  --splits_file data/chemprop_splits_h2h.json \
  --epochs 50 --batch_size 16 --max_lr 0.001 --warmup_epochs 2

# Error analysis
python src/05_analysis/error_analysis.py
```

---

## Repository Structure

```
BBB_prediction_project/
├── data/
│   ├── bbb_martins.csv                    # Source dataset (1,975 molecules)
│   ├── chemprop_bbb.csv                    # Phase 4 Chemprop-formatted data
│   ├── chemprop_h2h.csv                   # H2H Chemprop-formatted data
│   ├── chemprop_splits_h2h.json           # H2H split (= Phase 3 Fold 0)
│   ├── chemprop_splits_phase3.json        # Phase 4 split record
│   └── features/
│       ├── morgan_fps.npy                 # Morgan fingerprints (2048-bit)
│       ├── rdkit_descriptors.csv          # RDKit descriptors (185)
│       ├── combined_features.npy          # Combined features (2233)
│       ├── scaffold_splits.pkl            # Positional split indices (5 folds)
│       ├── random_splits.pkl              # Random split indices (5 folds)
│       └── scaffold_splits.npy            # Numpy version of splits
├── src/
│   ├── 03_ml/                             # Phase 3: Classical ML
│   │   ├── featurize.py                   # Featurizer (splits + descriptors)
│   │   ├── train_baseline.py              # RF/LogReg/SVM training
│   │   ├── audit_splits.py                # Split integrity verification
│   │   └── run_baseline_all.py            # Run all configurations
│   ├── 04_dl/                             # Phase 4: Chemprop GNN
│   │   ├── gnn_chemprop_plan.md           # Planning document
│   │   └── chemprop_baseline_results.md   # Results summary
│   ├── 05_evaluation/                     # Phase 5: H2H
│   │   ├── h2h_comparison_prep.py         # RF training + split prep
│   │   └── h2h_comparison_report.md       # Comparison narrative
│   ├── 05_analysis/                       # Phase 5: Error analysis
│   │   ├── error_analysis.py              # Per-molecule descriptor analysis
│   │   └── error_analysis_report.py       # Report generator
│   ├── 06_interpret/                      # Phase 6: Interpretation
│   │   ├── error_analysis.py              # Interpretation-level analysis
│   │   ├── feature_analysis.py            # Feature-level analysis
│   │   └── error_analysis_report.md       # Interpretation report
│   └── 07_chemberta/                      # Phase 7: ChemBERTa (NOT trained)
│       ├── IMPLEMENTATION_PLAN.md         # Environment verification + plan
│       └── finetune_chemberta.py          # Training script (not executed)
├── results/
│   ├── 03_ml_baseline_results.csv          # Phase 3 results (35 rows)
│   ├── 03_ml_summary_scaffold.csv          # Phase 3 summary
│   ├── 04_dl_chemprop_predictions.csv      # Phase 4 ensemble predictions
│   ├── 04_dl_chemprop_metrics.csv          # Phase 4 metrics
│   ├── h2h_predictions.csv                # H2H matched predictions (296 rows)
│   ├── h2h_rf_predictions.csv             # H2H RF predictions
│   ├── h2h_comparison.csv                 # H2H headline metrics
│   ├── 06_error_analysis.csv              # Error analysis + descriptors
│   ├── 06_error_analysis_summary.md       # Error analysis summary
│   ├── 06_error_analysis_report.md        # Error analysis report
│   ├── 06_interpret_error_analysis.csv    # Interpretation analysis
│   ├── baseline_summary.md                # Phase 3 summary
│   ├── final_experiment_summary.md        # FINAL project report
│   ├── feature_importance_*.csv            # RF feature importances
│   ├── chemprop_h2h/                      # H2H Chemprop model
│   ├── chemprop_model/                    # Phase 4 ensemble model
│   ├── chemprop_h2h_log.txt               # H2H training log
│   └── chemprop_train_log.txt             # Phase 4 training log
├── docs/                                   # Documentation
├── figures/                                # Publication figures
├── CITATION.cff                            # Citation file
├── README.md                               # This file
└── LICENSE                                 # MIT License
```

---

## Installation

```bash
# Python 3.11+
pip install rdkit scikit-learn pandas numpy torch

# Chemprop (for Phase 4/5 GNN reproduction)
pip install chemprop

# Transformers (for optional ChemBERTa)
pip install transformers tokenizers
```

---

## Data Acquisition

The TDC BBB_Martins dataset can be obtained from:
- TDC (Therapeutics Data Commons): https://tdcommons.ai/
- MoleculeNet BBBP subset: https://deepchem.io/
- Mirror: https://github.com/GLambard/Molecules_Dataset_Collection

Place the CSV at `data/bbb_martins.csv` with columns: `drug_id`, `drug` (SMILES), `target` (0/1).

---

## Citation

If you use this work, please cite the underlying dataset and Chemprop:

```
[REFERENCE TO BE VERIFIED]  # BBB_Martins dataset

[REFERENCE TO BE VERIFIED]  # Chemprop paper
```

And cite this repository using the `CITATION.cff` file.

---

## License

MIT License — see `LICENSE` file.

---

## Future Work

1. **ChemBERTa fine-tuning** on GPU or longer-timeout environment (environment verified, training not executed)
2. **RF + GNN ensemble** to exploit complementary error patterns
3. **Class imbalance mitigation** via stratified sampling or focal loss
4. **External validation** on independent BBB datasets (e.g., SugarBB, BBB-seq)
5. **Attention visualization** for GNN model interpretability
6. **Multi-task learning** combining BBB with related ADMET properties
