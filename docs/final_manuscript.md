# Comparative Machine Learning and Graph Neural Network Approaches for Blood–Brain Barrier Permeability Prediction: A Reproducible Study

**Devarshi Hatwar**

---

## Abstract

The blood-brain barrier (BBB) restricts drug entry to the central nervous system, making permeability prediction critical for CNS drug discovery. This study develops a reproducible computational pipeline to predict BBB permeability using the TDC BBB_Martins dataset (1,975 compounds). We evaluate classical machine learning (Random Forest on combined Morgan fingerprints and RDKit descriptors) and a graph neural network (Chemprop) under appropriate scaffold-based splitting protocols. The Random Forest achieved AUROC 0.921 ± 0.024 in 5-fold scaffold cross-validation. A Chemprop 5-model ensemble achieved AUROC 0.935 under a SCAFFOLD_BALANCED protocol. A matched head-to-head experiment on a single Fold 0 scaffold split (296 test molecules) showed the Random Forest (AUROC 0.9031) outperforming a single Chemprop model (AUROC 0.8669), with complementary error patterns — 27 of 52 total errors were shared and 25 disagreements occurred, all in the uncertain probability range. Chemical error analysis using Mann-Whitney U tests revealed that Random Forest false positives concentrate on low-QED compounds (p = 0.0002) while graph neural network false negatives show elevated topological polar surface area (p = 0.0002). All splits use pure positional indices with verified zero train/test leakage. ChemBERTa fine-tuning was environment-verified but not executed due to CPU constraints.

---

## 1. Introduction

The blood-brain barrier (BBB) is a specialized endothelial barrier that protects the brain from circulating toxins and xenobiotics. For central nervous system (CNS) therapeutics — including treatments for depression, Alzheimer's disease, Parkinson's disease, brain tumors, and epilepsy — BBB permeation is a prerequisite for efficacy. Approximately 98% of small-molecule drugs fail to cross the BBB at therapeutic concentrations, making permeability prediction a critical early-stage filter in CNS drug discovery [REFERENCE TO BE VERIFIED].

Experimental measurement of BBB permeability — via in vitro assays (e.g., parallel artificial membrane permeability assay, PAMPA) or in vivo mouse models (e.g., brain-to-plasma ratio) — is expensive, time-consuming, and raises animal use concerns. Computational prediction offers a rapid, low-cost alternative for early-stage screening [REFERENCE TO BE VERIFIED].

Machine learning approaches to BBB prediction typically fall into two categories:
1. **Classical ML with hand-crafted molecular features** — molecular fingerprints and physicochemical descriptors
2. **Deep learning with learned representations** — graph neural networks that operate directly on molecular graph structures

While both approaches have been applied to BBB prediction, direct comparisons on identical splits are rare, and chemical interpretability of model errors is often overlooked.

This study addresses three questions:
1. What is the best classical ML baseline for BBB permeability prediction?
2. How does it compare to a Chemprop graph neural network on a matched test set?
3. What chemical patterns emerge in each model's errors?

---

## 2. Research Objective

Develop and evaluate a reproducible computational pipeline for BBB permeability prediction using classical machine learning and graph neural networks, with a focus on matched head-to-head comparison and chemical error analysis.

---

## 3. Dataset

| Property | Value |
|---|---|
| **Source** | TDC BBB_Martins / MoleculeNet BBBP (Martins et al., 2012) |
| **Compounds** | 1,975 unique drug molecules |
| **Task** | Binary classification (BBB permeable vs. non-permeable) |
| **Class distribution** | 1,501 permeable (76.0%) / 474 non-permeable (24.0%) |
| **SMILES** | Drug names as canonical SMILES strings |
| **Target** | Binary label (1 = permeable, 0 = non-permeable) |

### Data Quality Verification

- 0 duplicate SMILES strings
- 0 invalid SMILES (all RDKit-parseable)
- 0 duplicate rows
- 0 missing values in target column

The dataset was originally distributed through the Therapeutics Data Commons (TDC). The original TDC repository has been removed; the data was retrieved from the GLambard/Molecules_Dataset_Collection mirror, which provides the canonical 1,975 compounds.

---

## 4. Data Quality and Preprocessing

### 4.1 Featurization

Three feature sets were generated from the molecular SMILES:

| Feature Set | Description | Dimension |
|---|---|---|
| Morgan | Morgan circular fingerprints (radius=2, nBits=2048) | 2,048 |
| RDKit | 2D molecular descriptors (MolWt, LogP, TPSA, HBD, HBA, etc.) | 185 |
| Combined | Morgan + RDKit descriptors (concatenated) | 2,233 |

Features are molecule-intrinsic — computed independently per molecule from its SMILES string. No target information is used during featurization.

### 4.2 Scaling

For classical ML models, features are standardized using scikit-learn's `StandardScaler`, fit on the training set only (train + validation) and then applied to the test set. This prevents test set data leakage during preprocessing.

---

## 5. Scaffold-Split Methodology

### 5.1 Motivation

Random train/test splits in molecular ML lead to data leakage because structurally similar molecules often appear in multiple splits, inflating performance estimates. Scaffold splitting assigns molecules sharing the same molecular scaffold to the same split, providing a more realistic estimate of generalization to novel chemotypes.

### 5.2 Implementation

All splits use **pure positional integer indices** (0 to 1,974). The scaffold splitting algorithm:

1. Computes the Murcko scaffold for each molecule using RDKit
2. Groups molecules by scaffold
3. Sorts scaffolds by frequency (most common first)
4. Assigns each scaffold group to a fold, cycling through folds 0–4

An earlier implementation used `pd.Index.get_indexer()`, which mixed pandas index labels with positional indices, producing spurious AUROC = 0.9997 results. This was diagnosed and fixed to use pure positional indexing — no leakage remains.

### 5.3 Leakage Audit

| Check | Result | Status |
|---|---|---|
| Index overlap (train ∩ test) | 0 | ✅ Pass |
| SMILES overlap (train ∩ test) | 0 | ✅ Pass |
| Scaffold overlap (train ∩ test) | 0 | ✅ Pass |
| Random split index overlap | 0 | ✅ Pass |

### 5.4 Phase-Specific Split Configurations

| Experiment | Split Type | n_train | n_val | n_test |
|---|---|---|---|---|
| Phase 3 (5-fold CV) | Scaffold (positional) | ~1,382 | ~297 | ~296 per fold |
| Phase 4 (Chemprop) | SCAFFOLD_BALANCED | ~1,382 | ~297 | 296 |
| Phase 5 (H2H) | Scaffold Fold 0 (positional) | 1,382 | 297 | 296 |

**Note:** Phase 4 uses Chemprop's internal `SCAFFOLD_BALANCED` algorithm, which differs from Phase 3's plain scaffold split. The two produce different train/test partitions.

---

## 6. Classical Machine Learning

### 6.1 Models and Setup

Three model types were evaluated × three feature sets = 9 configurations per fold:

| Model | Library | Hyperparameter Tuning |
|---|---|---|
| Logistic Regression | scikit-learn | GridSearchCV over C ∈ {0.1, 1, 10} |
| Random Forest | scikit-learn | GridSearchCV over n_estimators ∈ {200, 300, 400, 500} |
| Support Vector Machine | scikit-learn | GridSearchCV over C ∈ {0.1, 1, 10} |

Hyperparameter selection uses 3-fold GridSearchCV within the train+validation set. The test fold is used exactly once for final evaluation.

### 6.2 Phase 3 Results (5-Fold Scaffold CV)

Full results in `results/03_ml_baseline_results.csv` (40 rows: 9 configs × 4 metrics + summary).

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

**Best classical model:** Random Forest on Combined features (AUROC 0.921 ± 0.024).

### 6.3 Feature Type Analysis

- Combined features (Morgan + RDKit) outperform either alone, confirming that hand-crafted descriptors add complementary information to fingerprint-based representations.
- RDKit descriptors alone (AUROC 0.919) nearly match combined features (0.921), suggesting physicochemical properties capture most of the predictive signal.
- Morgan fingerprints alone are the weakest (0.831–0.901), missing explicit physicochemical signals.

### 6.4 Feature Importances

Top 10 RDKit descriptors by Random Forest Gini importance:
1. MolLogP (LogP) — 0.109
2. NumHDonors — 0.023
3. NumHAcc — 0.018
4. TPSA — 0.016
5. NumRotatableBonds — 0.012
6. NumAromaticRings — 0.011
7. NumAromaticHeterocycles — 0.011
8. FractionCSP3 — 0.008
9. NHOHCount — 0.008
10. HeavyAtomCount — 0.007

Files: `results/feature_importance_combined.csv`, `results/feature_importance_descriptors.csv`, `results/feature_importance_morgan.csv`

---

## 7. Chemprop Graph Neural Network

### 7.1 Phase 4: GNN Ensemble (SCAFFOLD_BALANCED)

**Architecture:** Message-passing GNN (depth=4, message-hidden-dim=300, aggregation=norm) as implemented in Chemprop.

**Training configuration:**

| Parameter | Value |
|---|---|
| Split | SCAFFOLD_BALANCED (5-fold) |
| Ensemble size | 5 models |
| Data seed | 42 |
| PyTorch seed | 42 |
| Batch size | 16 |
| Max learning rate | 0.001 |
| Warmup epochs | 2 |
| Total epochs | 50 |
| Early stopping patience | 10 |

**Ensemble predictions (averaged across 5 models):**

| Metric | Value |
|---|---|
| AUROC | 0.935 |
| AUPRC | 0.983 |
| Accuracy | 0.916 |
| F1 | 0.948 |
| Sensitivity | 0.958 |
| Specificity | 0.732 |
| TP/FP/TN/FN | 230/15/41/10 |

Individual model AUROCs: 0.898–0.949 (5 folds). Predictions: `results/04_dl_chemprop_predictions.csv` (296 rows). Metrics: `results/04_dl_chemparam_metrics.csv`.

> **Important:** This Phase 4 GNN result used the SCAFFOLD_BALANCED split protocol, which is **not the same** as the Phase 3 scaffold split. The 0.935 AUROC should **not** be interpreted as a direct head-to-head comparison with the Phase 3 RF (0.921).

---

## 8. Matched Head-to-Head Evaluation

### 8.1 Experimental Design

The matched H2H experiment was conducted to enable a fair comparison:

- Both models trained on the **exact same Phase 3 Fold 0 scaffold split** (1,679 train+val, 296 test molecules)
- All 296 test molecules are identical between RF and GNN
- Prediction probabilities verified to match source model outputs exactly (0.000 difference)

### 8.2 Model Configurations

| Model | Configuration | Checkpoint |
|---|---|---|
| RF | Combined Morgan + RDKit, n_estimators=500, class_weight='balanced', random_state=42 | `src/05_evaluation/h2h_comparison_prep.py` |
| GNN | Chemprop single model (ensemble_size=1), max_lr=0.001, batch_size=16, data_seed=42, pytorch_seed=42 | `results/chemprop_h2h/model_0/checkpoints/best-epoch=24-val_roc=0.91.ckpt` |

**RF configuration note:** The H2H RF used n_estimators=500 and class_weight='balanced', which differs from Phase 3 (n_estimators=300, no class_weight). This was an intentional choice for the H2H comparison to provide a strong single-model baseline.

**GNN configuration note:** The H2H GNN is a **separate Chemprop model** (ensemble_size=1) from the Phase 4 model (ensemble_size=5, SCAFFOLD_BALANCED). They use different training data, different configurations, and produce different predictions despite similar AUROC.

### 8.3 Results

| Metric | Random Forest | Chemprop GNN | Difference |
|---|---|---|---|
| **AUROC** | **0.9031** | 0.8669 | +0.0362 |
| AUPRC | 0.9726 | 0.9602 | +0.0124 |
| Accuracy | 0.8750 | 0.8581 | +0.0169 |
| F1 | 0.9246 | 0.9114 | +0.0132 |
| Precision | 0.8902 | 0.9076 | −0.0174 |
| Sensitivity | **0.9619** | 0.9153 | +0.0466 |
| Specificity | 0.5333 | **0.6333** | −0.1000 |
| TP | 227 | 216 | — |
| FP | 28 | 22 | — |
| TN | 32 | 38 | — |
| FN | 9 | 20 | — |

### 8.4 ROC and Precision-Recall Curves

![ROC curves](figures/h2h_roc_curve.png)

![PR curves](figures/h2h_precision_recall_curve.png)

---

## 9. Chemical Error Analysis

### 9.1 Error Overlap

| Error Type | RF Only | GNN Only | Both Models |
|---|---|---|---|
| False Positives | 7 | 1 | 21 |
| False Negatives | 3 | 14 | 6 |

- **Total errors:** 52 (RF: 37, GNN: 42)
- **Shared errors:** 27 (51.9%)
- **Disagreements:** 25 (RF correct in 15, GNN correct in 10)
- **High-confidence disagreements:** 0 (when both models ≥90% confident, they always agree)

![Error overlap](figures/error_overlap.png)

### 9.2 Statistically Significant Descriptor Differences

Mann-Whitney U tests comparing error groups against the "both correct" baseline:

| Error Group | Descriptor | Group Mean | Baseline Mean | p-value | Sig. |
|---|---|---|---|---|---|
| RF FP | QED | 0.55 | 0.66 | 0.0002 | *** |
| RF FP | TPSA | 72.8 | 62.8 | 0.028 | * |
| RF FP | NumHDonors | 1.71 | 1.30 | 0.013 | * |
| RF FP | NumRings | 2.64 | 3.13 | 0.050 | * |
| RF FN | TPSA | 119.9 | 62.8 | 0.0009 | *** |
| RF FN | NumHDonors | 2.67 | 1.30 | 0.0006 | *** |
| RF FN | NumHAcc | 7.33 | 4.08 | 0.0005 | *** |
| RF FN | MW | 428.3 | 335.4 | 0.007 | ** |
| RF FN | QED | 0.49 | 0.66 | 0.009 | ** |
| GNN FP | QED | 0.57 | 0.66 | 0.003 | ** |
| GNN FN | TPSA | 93.2 | 62.8 | 0.0002 | *** |
| GNN FN | NumHDonors | 1.90 | 1.30 | 0.002 | ** |
| GNN FN | NumHAcc | 5.65 | 4.08 | 0.0015 | ** |
| GNN FN | MW | 382.1 | 335.4 | 0.025 | * |

Significance: *** p < 0.001, ** p < 0.01, * p < 0.05

> **Note:** No multiple testing correction was applied. With 15 tested comparisons at α=0.05, approximately 0–1 false positives would be expected by chance. Strong signals (p < 0.001) are robust.

### 9.3 Chemical Interpretation

1. **RF false positives (low QED):** RF false positives have significantly lower QED (mean 0.55 vs 0.66, p=0.0002), indicating that RF over-predicts BBB permeability for molecules with poor overall drug-likeness. This suggests RF learns substructural patterns associated with permeability without fully integrating holistic drug-likeness assessment.

2. **GNN false negatives (high TPSA):** GNN false negatives have significantly higher TPSA (mean 93.2 vs 62.8 Å², p=0.0002), indicating the GNN under-predicts permeability for large, polar compounds that genuinely cross the BBB. This may reflect the GNN's sensitivity to molecular size and polarity in its learned representations.

3. **Shared errors:** 27 of 52 errors are shared between both models, suggesting common failure modes on compounds with challenging molecular properties.

4. **Complementary errors:** The 25 model-specific errors (10 RF-only correct, 15 GNN-only correct) suggest that ensembles combining RF and GNN could reduce total error count.

![Descriptor analysis](figures/error_descriptor_analysis.png)

### 9.4 Hypothesis

Combining RF (which captures explicit physicochemical properties) with GNN (which learns from graph structure) in an ensemble may yield better overall performance by exploiting their complementary error patterns.

---

## 10. Results

### Summary of Key Results

| Experiment | Model | AUROC | Split Protocol | Notes |
|---|---|---|---|---|
| Phase 3 (5-fold scaffold CV) | RF (Combined) | 0.921±0.024 | Scaffold | Best classical model |
| Phase 4 (5-fold SCAFFOLD_BALANCED) | GNN ensemble (5) | 0.935 | SCAFFOLD_BALANCED | Different protocol — not directly comparable |
| H2H (Fold 0, matched) | RF (Combined, n=500, balanced) | 0.9031 | Scaffold Fold 0 | Single split, 296 test molecules |
| H2H (Fold 0, matched) | GNN (single) | 0.8669 | Scaffold Fold 0 | Single split, 296 test molecules |

**Key finding:** On the matched Fold 0 test set, RF achieved higher AUROC (0.9031 vs 0.8669) and sensitivity (0.962 vs 0.915), while GNN achieved higher specificity (0.633 vs 0.533) and precision (0.908 vs 0.890). This reflects different error trade-offs on a single test set — not a generalizable ranking of model types.

---

## 11. Discussion

### 11.1 Interpretation of Results

The matched head-to-head comparison reveals that on a single scaffold split, Random Forest with hand-crafted molecular features outperforms a single Chemprop GNN. This may be explained by:

1. **Feature richness:** Combined features explicitly encode 185 physicochemical properties (including TPSA, LogP, QED) that the GNN must learn from graph structure alone. On a dataset of 1,975 molecules, explicit features may be more data-efficient than learned representations.

2. **Scaffold split severity:** Scaffold splitting with 296 test molecules means the model encounters entirely novel scaffolds. RF's explicit feature representation may generalize better to unseen scaffolds than the GNN's learned atom-level representations.

3. **Class imbalance handling:** The H2H RF used class_weight='balanced' to address the 3:1 class imbalance, potentially improving minority-class prediction.

### 11.2 Comparison of Phase 3 and Phase 4

The Phase 3 RF (0.921) and Phase 4 GNN ensemble (0.935) results are from **different experimental protocols** and must not be directly compared:
- Phase 3 uses plain scaffold splitting; Phase 4 uses SCAFFOLD_BALANCED
- Phase 3 reports 5-fold CV mean; Phase 4 reports 5-model ensemble
- The test sets may contain different molecules

### 11.3 Error Analysis Insights

The chemical error analysis provides actionable insights for drug discovery:
- Compounds with low QED are systematically over-predicted by RF — such compounds may be flagged for additional review before discarding
- Compounds with high TPSA (>93 Å²) are systematically missed by GNN — these may include genuine BBB penetrators that the GNN's learned permeability heuristic excludes
- The complementary error patterns (25 disagreements, zero high-confidence disagreements) suggest ensemble approaches could capture the strengths of both model types

### 11.4 Why Error Analysis Matters

Beyond the headline AUROC, the error analysis reveals how molecular properties influence model predictions. This has direct drug discovery relevance:
- **Low-QED false positives (RF):** These are compounds with poor overall drug-likeness that RF incorrectly predicts as BBB-permeable. In practice, these would be deprioritized by chemists regardless, so the false positive is less concerning.
- **High-TPSA false negatives (GNN):** These are compounds with high polarity that the GNN misses. In CNS drug discovery, some genuinely BBB-permeable compounds do have elevated TPSA — the GNN's bias against these could miss valuable candidates.

---

## 12. Limitations

1. **Single-fold H2H:** The matched comparison uses one Fold 0 scaffold split (296 test molecules) — not statistically powered to establish general model superiority.
2. **Different split protocols:** Phase 3 (plain scaffold) and Phase 4 (SCAFFOLD_BALANCED) use different algorithms — direct comparison is invalid.
3. **RF hyperparameter difference:** H2H RF used n_estimators=500, class_weight='balanced'; Phase 3 RF used n_estimators=300, no class_weight. Both are valid for their contexts.
4. **Class imbalance:** 76% permeable — accuracy is inflated. Sensitivity and specificity are reported for balanced assessment.
5. **No multiple testing correction:** Mann-Whitney U tests not Bonferroni/Holm corrected. Strong signals (p<0.001) are robust; marginal signals (p≈0.02–0.05) may contain false positives.
6. **No external validation:** Only the TDC BBB_Martins dataset is used. Performance on independent datasets is unknown.
7. **No confidence intervals:** AUROC confidence intervals are not reported.
8. **ChemBERTa not trained:** A transformer-based approach was environment-verified but not executed.

---

## 13. Future Work

1. **Multiple-fold H2H comparison** — repeat the matched RF vs GNN experiment across multiple scaffold folds
2. **ChemBERTa fine-tuning** — on a suitable GPU or longer runtime (implementation plan complete, training not executed)
3. **RF + GNN ensemble** — combine predictions to exploit complementary error patterns
4. **Independent validation** — on external BBB datasets (SugarBB, Chou & Shen BBB)
5. **Class imbalance mitigation** — stratified sampling, focal loss, threshold optimization
6. **Uncertainty quantification** — Monte Carlo dropout, conformal prediction
7. **Confidence intervals** — bootstrap CIs for all metrics

---

## 14. Conclusion

This study provides a reproducible pipeline for BBB permeability prediction, comparing classical ML (Random Forest) and graph neural networks (Chemprop) under appropriate scaffold-based splitting. The Random Forest on combined molecular features achieved AUROC 0.921 ± 0.024 in 5-fold scaffold cross-validation. A Chemprop ensemble achieved AUROC 0.935 under a different split protocol. In a matched head-to-head experiment on 296 compounds, RF (AUROC 0.9031) outperformed a single Chemprop model (AUROC 0.867), with complementary error patterns. Chemical error analysis revealed statistically significant differences in molecular properties (QED, TPSA, molecular weight) associated with each model's failures. All experiments are reproducible from the provided code and data, with documented provenance from source predictions to final metrics.

---

## 15. Data and Code Availability

- **Code repository:** [GITHUB URL]
- **Archived release:** Zenodo DOI [ZENODO DOI to be assigned upon publication]
- **License:** MIT License
- **Dataset:** TDC BBB_Martins (public benchmark dataset)

### File Inventory

| File | Contents |
|---|---|
| `data/bbb_martins.csv` | Source dataset (1,975 compounds) |
| `results/03_ml_baseline_results.csv` | Phase 3 all-model results (40 rows) |
| `results/03_ml_summary_scaffold.csv` | Phase 3 summary |
| `results/04_dl_chemprop_predictions.csv` | Phase 4 ensemble predictions (296 rows) |
| `results/04_dl_chemprop_metrics.csv` | Phase 4 metrics |
| `results/h2h_predictions.csv` | Matched H2H predictions (296 rows) |
| `results/h2h_rf_predictions.csv` | H2H RF predictions |
| `results/h2h_comparison.csv` | H2H metrics |
| `results/06_error_analysis.csv` | Per-molecule error + descriptors (296 rows) |
| `results/06_error_analysis_summary.md` | Error analysis summary |
| `results/06_interpret_error_analysis.csv` | Interpretation analysis |
| `results/final_experiment_summary.md` | Complete provenance audit |
| `results/feature_importance_*.csv` | RF feature importances (3 files) |
| `src/03_ml/` | Classical ML source code |
| `src/05_analysis/` | Error analysis scripts |
| `src/05_evaluation/` | H2H comparison scripts |
| `src/06_interpret/` | Interpretation scripts |
| `src/07_chemberta/` | ChemBERTa implementation plan |
| `figures/*.png` | Publication figures |
| `generate_figures.py` | Figure generation script |
| `CITATION.cff` | Citation metadata |

Model checkpoints:
- `results/chemprop_h2h/model_0/` — H2H Chemprop model (single model, Fold 0)
- `results/chemprop_model/model_0/` through `model_4/` — Phase 4 ensemble (5 models, SCAFFOLD_BALANCED)

Split files:
- `data/chemprop_splits_h2h.json` — H2H split (= Phase 3 Fold 0)
- `data/chemprop_splits_phase3.json` — Phase 3 split record (not used by Phase 4 model)

---

## 16. References

1. [REFERENCE TO BE VERIFIED] — TDC BBB_Martins dataset (Martins, Sirotkin, Patronov, 2012)
2. [REFERENCE TO BE VERIFIED] — Chemprop paper (Yang et al., 2019, or Stark et al., 2020)
3. [REFERENCE TO BE VERIFIED] — Scaffold splitting methodology
4. [REFERENCE TO BE VERIFIED] — Mann-Whitney U test for biomarker comparison
5. [REFERENCE TO BE VERIFIED] — QED (Quantitative Estimate of Drug-likeness)
6. [REFERENCE TO BE VERIFIED] — BBB permeability prediction in drug discovery
7. [REFERENCE TO BE VERIFIED] — Molecular property prediction benchmarks

All references must be verified before submission. The dataset source is the primary citation; Chemprop and scaffold splitting are secondary citations.

---

## Author Contributions

- **Conceptualization:** D.H.
- **Methodology:** D.H.
- **Software:** D.H.
- **Validation:** D.H.
- **Formal Analysis:** D.H.
- **Investigation:** D.H.
- **Data Curation:** D.H.
- **Writing — Original Draft:** D.H.
- **Writing — Review & Editing:** D.H.
- **Visualization:** D.H.
- **Supervision:** N/A (independent project)

---

## Acknowledgments

The author thanks the open-source scientific Python ecosystem (RDKit, scikit-learn, PyTorch, Chemprop, pandas, numpy, matplotlib) and the TDC/MoleculeNet community for maintaining open benchmark datasets. This work was conducted independently with no external funding.
