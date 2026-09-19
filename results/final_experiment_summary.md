# Final Experiment Summary

## Dataset
- **Source:** TDC BBB_Martins / MoleculeNet BBBP (1,975 compounds, 76:24 positive:negative)
- **Features:** Morgan fingerprints (radius 2, 2048 bits) + RDKit descriptors (185 computed)
- **Verification:** 0 duplicate SMILES, 0 invalid SMILES, 0 duplicate rows

---

## 1. Phase 3: Classical ML Baseline (5-Fold Scaffold Split)

**Split methodology:** Scaffold split (Murcko scaffold), 5 folds. Train+val for fitting, test for evaluation. Pure positional indices — no leakage.

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

**Best classical model:** RF on combined features (AUROC 0.921±0.024 across 5 folds).

---

## 2. Phase 4: Chemprop GNN (5-Fold SCAFFOLD_BALANCED Ensemble)

**Split methodology:** Chemprop internal SCAFFOLD_BALANCED split (5 folds, 1382 train / 297 val / 296 test per fold). **This is a different split from Phase 3** — the scaffold split algorithm differs (SCAFFOLD_BALANCED vs. plain scaffold), so test-set molecules are NOT the same.

| Metric | Value |
|---|---|
| AUROC | 0.935 |
| AUPRC | 0.983 |
| Accuracy | 0.916 |
| F1 | 0.948 |
| Sensitivity | 0.958 |
| Specificity | 0.732 |
| TP/FP/TN/FN | 230/15/41/10 |

**Note:** The 0.935 AUROC is from a 5-model ensemble (ensemble_size=5, 5 different random seeds). Individual model AUROCs range from 0.898 to 0.949. Checkpoint: `results/chemprop_model/model_*/best-epoch=2X-val_roc=0.8X.ckpt`.

---

## 3. Matched Fold 0 Comparison (RF vs GNN, Same 296 Molecules)

**Split methodology:** Fold 0 from the 5-fold scaffold split used in Phase 3 (positional indices 0–1974). Both models predict on the identical 296 test molecules.

**RF provenance:** Trained by `src/05_evaluation/h2h_comparison_prep.py` using Combined (Morgan + RDKit) features, n_estimators=500, class_weight='balanced', random_state=42, StandardScaler fit on train+val. This uses the same split indices as Phase 3 Fold 0 but DIFFERENT RF hyperparameters (Phase 3 used n_estimators=300, no class_weight).

**GNN provenance:** Chemprop single model (ensemble_size=1), trained with `data_seed=42`, `pytorch_seed=42`, on `chemprop_splits_h2h.json` (identical to Phase 3 Fold 0 split). Checkpoint: `results/chemprop_h2h/model_0/checkpoints/best-epoch=24-val_roc=0.91.ckpt`. This is a SEPARATE model from the Phase 4 Chemprop (ensemble_size=5, SCAFFOLD_BALANCED, different split).

| Model | AUROC | AUPRC | Accuracy | F1 | Sensitivity | Specificity | TP/FP/TN/FN |
|---|---|---|---|---|---|---|---|
| RF (Combined Morgan+RDKit) | 0.9031 | 0.9726 | 0.8750 | 0.9246 | 0.9619 | 0.5333 | 227/28/32/9 |
| GNN (Chemprop) | 0.8669 | 0.9602 | 0.8581 | 0.9114 | 0.9153 | 0.6333 | 216/22/38/20 |

**Key observation:** RF outperforms GNN on this matched Fold 0 test set (AUROC 0.903 vs 0.867). RF has higher sensitivity (0.962 vs 0.915); GNN has higher specificity (0.533 vs 0.633). This is an observed result for one fold, not a general claim about either model's robustness.

---

## 4. Error Analysis (Matched Fold 0 Test Set)

### Error Overlap
| Error Type | RF Only Errors | GNN Only Errors | Both Error |
|---|---|---|---|
| False Positives (pred 1, true 0) | 7 | 1 | 21 |
| False Negatives (pred 0, true 1) | 3 | 14 | 6 |

- **Total disagreements:** 25 (RF wins 10, GNN wins 15)
- **High-confidence disagreements:** 0 (when both models are ≥90% confident, they always agree)

### Statistically Significant Descriptor Differences (Mann-Whitney U vs Both-Correct)

| Error Group | Descriptor | Group Mean | Baseline Mean | p-value | Significance |
|---|---|---|---|---|---|
| RF FP | QED | 0.55 | 0.66 | 0.0002 | *** |
| RF FP | TPSA | 72.8 | 62.8 | 0.028 | * |
| RF FP | NumHDonors | 1.71 | 1.30 | 0.013 | * |
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

### Scientific Interpretation (Observed Patterns)

1. **RF false positives** concentrate on low-QED compounds (p=0.0002) — RF over-predicts BBB permeability for molecules with poor drug-likeness, likely pattern-matching on substructures without holistic assessment.

2. **GNN false negatives** show a strong TPSA signal (mean 93.2 vs 62.8 baseline, p=0.0002) — GNN under-predicts BBB permeability for large, polar molecules that actually cross the BBB, possibly reflecting limited training data for such scaffolds.

3. **Error overlap is partial:** 27 of 52 total errors are shared; the remaining 25 disagreements (where models make different mistakes) suggest complementary strengths that could be exploited via ensembling.

### Files
- `results/06_error_analysis.csv` — per-molecule predictions + descriptors (296 rows)
- `results/06_error_analysis_report.md` — full console output

---

## 5. ChemBERTa (Planned, NOT Trained)

**Status:** Environment verified. Model `seyonec/ChemBERTa-zinc-base-v1` loads correctly in the current Python 3.11 environment (transformers 5.17.0, torch 2.14.0+cpu).

**Why not trained:** The terminal environment has a 5 GB memory limit and a 300-second process lifetime. CPU training of a 44M-parameter RoBERTa model on 1,382 molecules was estimated at ~8.7 seconds per batch (~754s per epoch, ~2261s for 3 epochs). The training process was automatically terminated by the environment after ~370 seconds and ~590 seconds on two separate attempts.

**Scientific value assessment:** Low. The Phase 3 RF baseline (AUROC 0.921±0.024, 5-fold) already provides a strong classical ML baseline. The Phase 4 Chemprop GNN (AUROC 0.935, 5-fold ensemble) provides the deep learning comparison. Adding ChemBERTa would produce a third architecture but on a single Fold 0 scaffold split (n=296 test), where the class imbalance (79.7% positive) already inflates accuracy and F1 metrics. The marginal scientific value does not justify the runtime constraint.

**Recommendation:** ChemBERTa fine-tuning is marked as **optional future work**. It should be attempted in a GPU-enabled or longer-timeout environment.

---

## Final Scientific Conclusion

### What Was Tested
1. **Classical ML (Phase 3):** 7 model configurations (LogReg, RF, SVM) across 3 feature sets (Morgan, RDKit descriptors, Combined) on 5-fold scaffold splits (1,975 molecules, positional indices, zero leakage)
2. **Deep Learning (Phase 4):** Chemprop GNN (ensemble_size=5, 5-fold SCAFFOLD_BALANCED split)
3. **Head-to-Head (Phase 5):** RF (Combined, n=500, class_weight=balanced) vs single Chemprop GNN (ensemble_size=1) on the **exact same Fold 0 scaffold split** (296 test molecules)
4. **Error Analysis (Phase 5):** Matched h2h predictions with molecular descriptor comparison

### What Performed Best

| Evaluation Setting | Best Model | AUROC |
|---|---|---|
| 5-fold aggregate (Phase 3) | RF (Combined) | 0.921±0.024 |
| 5-fold aggregate (Phase 4) | Chemprop GNN (ensemble=5) | 0.935 |
| Matched Fold 0 (Phase 5) | RF (Combined) | 0.9031 |

### What the Results Demonstrate
- RF on combined Morgan+RDKit features is the strongest classical model, achieving 0.921±0.024 AUROC across 5 folds
- The Chemprop GNN ensemble (Phase 4, 0.935 AUROC) outperforms on the aggregate, but this used a different split protocol
- On the **matched Fold 0** comparison (same split, same test set), RF outperforms the single Chemprop model (0.903 vs 0.867), suggesting the ensemble effect in Phase 4 may be a confounding variable
- Both models have complementary error patterns (25 of 52 errors are model-specific), with RF catching more true positives (sensitivity 0.962) and GNN making fewer false positives (specificity 0.633)
- Error analysis reveals chemically meaningful patterns: both models struggle with large, polar molecules (high TPSA), while RF specifically over-predicts permeability for low-QED compounds

### Limitations
1. The matched h2h comparison uses a single fold (n=296 test), limiting statistical power
2. The RF in the matched comparison uses slightly different hyperparameters (n_estimators=500, class_weight=balanced) than Phase 3 (n_estimators=300, no class_weight)
3. The Phase 4 Chemprop ensemble used a different split protocol (SCAFFOLD_BALANCED) than Phase 3, preventing direct cross-phase comparison
4. Class imbalance (79.7% positive) inflates accuracy/F1 metrics across all models
5. No ChemBERTa results due to environment constraints

### Future Work
- ChemBERTa fine-tuning on GPU or longer-timeout environment
- Ensemble of RF + Chemprop GNN to test if complementary errors improve performance
- Stratified sampling or class-weighted training to address imbalance bias
- External validation on an independent BBB dataset
