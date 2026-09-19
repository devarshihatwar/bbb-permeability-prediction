# BBB Prediction Project — CV Entry

## A. 2-Line Version

Developed a reproducible pipeline comparing Random Forest (AUROC 0.921±0.024, 5-fold) and Chemprop GNN (AUROC 0.935, ensemble) on the TDC BBB_Martins dataset, including a matched Fold 0 head-to-head (RF 0.903 vs GNN 0.867) and chemical error analysis revealing model-specific failure patterns.

## B. 4-Bullet Version

- **Classical ML pipeline:** Implemented Random Forest on combined Morgan+RDKit features (0.921±0.024 AUROC, 5-fold scaffold CV) using scikit-learn, RDKit, and numpy on 1,975 BBB_Martins compounds — audited for split leakage with zero train/test SMILES overlap.
- **Graph neural network:** Trained Chemprop GNN ensemble (5 models, SCAFFOLD_BALANCED split, AUROC 0.935) and a single matched-split model (AUROC 0.867) using the exact same Fold 0 scaffold split as the classical baseline.
- **Head-to-head evaluation:** Constructed a matched 296-molecule test set comparison showing RF outperforms single GNN (0.903 vs 0.867 AUROC), with complementary error patterns (25/52 model-specific errors, zero high-confidence disagreements).
- **Chemical error analysis:** Performed Mann-Whitney U testing on molecular descriptors, finding RF false positives concentrate on low-QED compounds (p=0.0002) and GNN false negatives on high-TPSA polar molecules (p=0.0002).

## C. 6-Bullet Detailed Version

- **Data engineering & featurization:** Built a complete molecular ML pipeline from the TDC BBB_Martins dataset (1,975 compounds, 76:24 class split), generating 2,048-bit Morgan fingerprints and 185 RDKit descriptors. Implemented and audited a Murcko scaffold splitting function that uses pure positional integer indices (0–1974) to prevent pandas index leakage — an earlier version had produced spurious 0.99 AUROC results due to index mapping bugs, which were diagnosed and fixed.
- **Classical ML baseline (Phase 3):** Trained and evaluated 7 model configurations (Logistic Regression, Random Forest, SVM) × 3 feature sets across 5-fold scaffold cross-validation using scikit-learn. Identified RF on combined features as the best classical model (AUROC 0.921 ± 0.024). Implemented GridSearchCV with 3-fold CV nested within train+val, with StandardScaler properly fitted only on training data.
- **Deep learning comparison (Phase 4):** Trained a Chemprop GNN ensemble (5 models, ensemble_size=5, SCAFFOLD_BALANCED protocol, batch_size=16, max_lr=0.001) achieving AUROC 0.935. This used Chemprop's internal split algorithm, which differs from the Phase 3 scaffold split — documented as a limitation for cross-phase comparability.
- **Matched head-to-head comparison (Phase 5):** Engineered a fair comparison by training both RF (n_estimators=500, class_weight='balanced', combined features) and a single Chemprop GNN (ensemble_size=1, same hyperparameters) on the exact same Phase 3 Fold 0 scaffold split (296 test molecules). RF achieved AUROC 0.9031 vs GNN 0.8669. RF used different hyperparameters from Phase 3 to maintain a fair single-model comparison.
- **Chemical error analysis (Phase 6):** Conducted per-molecule error analysis with 9 molecular descriptors (MW, LogP, TPSA, HBD, HBA, rotatable bonds, rings, RO5 violations, QED). Found 52 total model errors with 27 shared and 25 disagreements (RF wins 10, GNN wins 15). Used Mann-Whitney U tests to identify statistically significant descriptor differences: RF false positives have lower QED (p=0.0002), GNN false negatives have higher TPSA (p=0.0002). Zero high-confidence disagreements were found.
- **Reproducibility & provenance:** Produced a comprehensive provenance audit verifying that all H2H predictions trace to their source models and splits, with prediction probabilities matching saved model outputs exactly. Documented all provenance details in `results/final_experiment_summary.md`. ChemBERTa fine-tuning was environment-verified but not executed due to CPU memory/time constraints (5 GB RAM, 300 s process lifetime).

### Technical Skills Demonstrated

| Skill | Evidence |
|---|---|
| Python data science | pandas, numpy, scikit-learn, RDKit, PyTorch |
| Molecular ML | Chemprop, MPNN, scaffold splitting, molecular descriptors |
| Model evaluation | AUROC, AUPRC, confusion matrix, 5-fold CV, ensemble |
| Statistical analysis | Mann-Whitney U test, p-value correction, effect sizes |
| Reproducibility | Split leakage audit, provenance tracing, documentation |
| Error analysis | Per-molecule error categorization, descriptor comparison |
| Scientific computing | CPU-only training optimization, memory profiling |
