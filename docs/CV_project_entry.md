# AI-Based Blood–Brain Barrier Permeability Prediction | Python, RDKit, scikit-learn, Chemprop

Comparative evaluation of machine learning and graph neural network approaches for predicting BBB permeability using the TDC BBB_Martins dataset (1,975 compounds). Features include scaffold-based evaluation, matched head-to-head comparison, and chemical error analysis.

---

## About the Project

Blood-brain barrier (BBB) permeability is a critical property in CNS drug discovery — most drugs fail because they cannot cross the BBB. This project evaluates whether machine learning models can predict BBB permeability computationally.

Built during my transition from B.Pharm to pharmaceutical AI/data-driven drug discovery, this project demonstrates a full-stack approach: data understanding, molecular featurization, model evaluation, and chemical interpretability through error analysis.

---

## A. 2-Line Version (Resume/CV)

Developed a reproducible AI pipeline for blood-brain barrier permeability prediction (1,975 compounds), comparing Random Forest (AUROC 0.921±0.024, 5-fold) and Chemprop GNN (0.935) with matched head-to-head evaluation (RF 0.9031 vs GNN 0.8669) and chemical error analysis. Environment: Python, RDKit, scikit-learn, PyTorch.

## B. 4-Bullet Version (Detailed CV)

- **Molecular ML pipeline:** Built and evaluated a complete pipeline for BBB permeability prediction using the TDC BBB_Martins dataset (1,975 compounds). Implemented scaffold-based splitting with pure positional indices, verified zero train/test leakage. Features: Morgan fingerprints (2048-bit) + RDKit descriptors (185 properties).

- **Classical ML (Phase 3):** Trained and evaluated 7 model configurations (Logistic Regression, Random Forest, SVM) × 3 feature sets across 5-fold scaffold cross-validation using scikit-learn and RDKit. Best model: Random Forest on combined features (AUROC 0.921 ± 0.024, AUPRC 0.967, F1 0.919). Used GridSearchCV with proper train/val/test separation (no leakage).

- **Deep learning comparison (Phase 4):** Trained Chemprop graph neural network (5-model ensemble, SCAFFOLD_BALANCED protocol) achieving AUROC 0.935. Architecture: message-passing GNN (depth=4, hidden_dim=300) in PyTorch.

- **Matched H2H evaluation (Phase 5):** Conducted a fair head-to-head comparison on 296 compounds using the identical Fold 0 scaffold split for both RF and GNN. RF achieved AUROC 0.9031; GNN achieved 0.8669. Performed chemical error analysis using Mann-Whitney U tests on 9 molecular descriptors (QED, TPSA, MW, HBD, HBA, etc.) — found statistically significant patterns: RF false positives concentrate on low-QED compounds (p=0.0002), GNN false negatives on high-TPSA molecules (p=0.0002).

## C. 6-Bullet Detailed Version (Portfolio)

### 1. Project Leadership and Scientific Approach
Conceived and executed an independent research project bridging pharmaceutical sciences (B.Pharm background) and AI for drug discovery. Designed the full experimental pipeline from scratch: data acquisition from TDC BBB_Martins, to scaffold splitting, to error analysis — maintaining scientific rigor throughout without external supervision or collaboration.

### 2. Data Engineering and Cheminformatics
Processed 1,975 drug molecules from the TDC BBB_Martins dataset. Generated three molecular feature representations using RDKit: Morgan fingerprints (2,048-bit, radius=2), 185 RDKit 2D descriptors (MW, LogP, TPSA, HBD, HBA, QED, etc.), and their combination. Implemented and audited a Murcko scaffold splitting function using pure positional integer indices (0–1,974), catching and fixing an earlier leakage bug that produced spurious AUROC=0.9997 results.

### 3. Classical Machine Learning
Trained 7 model configurations (Logistic Regression, Random Forest, SVM) × 3 feature sets across 5-fold scaffold cross-validation using scikit-learn. Used GridSearchCV (3-fold, within train+val) for hyperparameter tuning with StandardScaler properly fit on training data only. Best model: Random Forest on combined Morgan + RDKit features (AUROC 0.921 ± 0.024). Verified that combined features outperform either alone, confirming hand-crafted descriptors add complementary information.

### 4. Deep Learning with Chemprop
Trained and evaluated Chemprop graph neural networks — message-passing GNNs that operate directly on molecular graph structures (atoms as nodes, bonds as edges). Phase 4: 5-model SCAFFOLD_BALANCED ensemble (AUROC 0.935). Phase 5 H2H: single model trained on identical Fold 0 split as the RF (AUROC 0.8669). Used PyTorch backend with batch_size=16, max_lr=0.001, warmup=2 epochs, patience=10, 50 max epochs, seeds=42.

### 5. Matched Head-to-Head Evaluation and Provenance Audit
Designed and executed a controlled comparison where both RF and GNN predict on the identical 296 test molecules from Phase 3 Fold 0 scaffold split. Verified prediction provenance end-to-end: RF probabilities reproduced exactly from `h2h_comparison_prep.py`; GNN probabilities match `chemprop_h2h/model_0/test_predictions.csv` (0.000 difference). RF achieved AUROC 0.9031 vs GNN 0.8669 — RF had higher sensitivity (0.962 vs 0.915), GNN had higher specificity (0.633 vs 0.533). This single-fold observation does not establish general superiority.

### 6. Chemical Error Analysis and Scientific Interpretation
Performed per-molecule error analysis on 296 test compounds, categorizing the 52 total errors into shared (27), RF-specific (10), and GNN-specific (15). Used Mann-Whitney U tests to compare 9 molecular descriptors (QED, TPSA, MW, HBD, HBA, rotatable bonds, rings, RO5 violations) between error groups and the correct-prediction baseline. Found statistically significant patterns: RF false positives concentrate on low-QED compounds (mean 0.55 vs 0.66, p=0.0002), GNN false negatives on high-TPSA molecules (mean 93.2 vs 62.8 Å², p=0.0002). Zero high-confidence disagreements were found. This work demonstrates how pharmaceutical knowledge (drug-likeness via QED, polarity via TPSA) enhances ML model interpretation in drug discovery.

### Technical Skills Demonstrated
Python, RDKit, scikit-learn, PyTorch, Chemprop, scaffold splitting, cheminformatics, molecular descriptors, molecular fingerprints, train/test leakage audit, hyperparameter tuning, model evaluation, statistical testing (Mann-Whitney U), error analysis, reproducibility, Git.
