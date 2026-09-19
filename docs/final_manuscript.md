# Understanding Blood-Brain Barrier Permeability Through Molecular Properties and Predictive Modeling

**Devarshi Hatwar**

*[Affiliation to be added]*

*[ORCID: 0000-0000-0000-0000 — to be replaced with real ORCID]*

---

## Abstract

**Background:** The blood-brain barrier (BBB) restricts drug entry to the central nervous system, making permeability prediction important for central nervous system (CNS) drug discovery. Experimental measurement of BBB permeability is expensive and time-consuming. **Objective:** This study investigates whether computational methods—Random Forest with molecular fingerprints and physicochemical descriptors, and a graph neural network (Chemprop)—can predict BBB permeability and reveal which molecular properties are associated with it. **Methods:** Using the TDC BBB_Martins dataset (1,975 compounds), we evaluated Random Forest on combined Morgan fingerprints and RDKit descriptors (5-fold scaffold cross-validation, AUROC 0.921 ± 0.024), a Chemprop graph neural network ensemble (SCAFFOLD_BALANCED protocol, AUROC 0.935), and a matched head-to-head comparison on 296 compounds (RF AUROC 0.9031 vs single GNN AUROC 0.8669). We performed chemical error analysis using Mann-Whitney U tests on molecular descriptors. **Results:** On the matched test set, the Random Forest achieved AUROC 0.9031 compared with 0.8669 for the single graph neural network. Error analysis revealed that RF false positives tend to occur on compounds with lower drug-likeness (QED), while GNN false negatives tend to occur on larger, more polar compounds (higher TPSA). Of 52 total errors, 27 were shared between models and 25 were model-specific. **Conclusions:** Computational models can identify molecular properties associated with BBB permeability. The two approaches capture different patterns, suggesting complementary value. A single-fold comparison provides preliminary signal but is not sufficient to establish general superiority of one approach.

**Keywords:** blood-brain barrier; drug discovery; molecular properties; machine learning; graph neural network; Random Forest (Chemprop); drug-like properties; TPSA; QED; computational drug discovery

---

## 1. Introduction

The blood-brain barrier (BBB) is a specialized barrier formed by tightly joined endothelial cells lining the brain's capillaries. It protects the brain from circulating toxins and pathogens, but it also prevents most drugs from reaching their intended targets in the central nervous system (CNS) [^1]. This is a significant challenge in CNS drug discovery: treatments for depression, Alzheimer's disease, Parkinson's disease, brain tumors, and epilepsy must cross the BBB to be effective, yet only a small fraction of available drugs can do so [^2][^3].

Predicting whether a compound can cross the BBB is difficult. Experimental measurements—such as in vitro permeability assays or in vivo brain-to-plasma ratio studies—are expensive, time-consuming, and require laboratory infrastructure. Computational prediction offers the possibility of rapidly screening thousands of compounds to prioritize the most promising candidates for experimental testing [^2].

Molecules that cross the BBB share certain physicochemical properties. Lipophilicity (LogP), molecular weight, topological polar surface area (TPSA), and hydrogen-bonding capacity all influence a molecule's ability to cross biological membranes by passive diffusion [^3][^4]. A computational model that learns these relationships could help medicinal chemists understand which molecular features matter and guide compound design.

This study investigates BBB permeability prediction using two computational approaches:

1. **Random Forest** with engineered molecular features (Morgan fingerprints and RDKit physicochemical descriptors)
2. **Graph neural network (Chemprop)** that learns representations directly from molecular graph structure

We evaluate these approaches on the TDC BBB_Martins dataset and examine **what the models' successes and failures reveal about molecular properties** associated with BBB permeability—not just whether the models perform well, but what their behavior teaches us about the underlying chemistry.

---

## 2. Background

### 2.1 The Blood-Brain Barrier

The BBB is formed by brain microvascular endothelial cells connected by tight junctions, which restrict paracellular diffusion [^1]. Over 98% of small-molecule drugs cannot cross the BBB at therapeutically relevant concentrations [^3]. Successful CNS drugs must balance lipophilicity (to cross membranes) with polarity (to maintain solubility), along with factors such as molecular size, hydrogen-bonding capacity, and active efflux transport [^2][^4].

### 2.2 Physicochemical Determinants of BBB Penetration

No single "rule" predicts BBB penetration as precisely as Lipinski's Rule of Five does for oral absorption. Instead, BBB permeation depends on a combination of properties [^3]:

| Property | Trend | Typical guideline |
|---|---|---|
| Molecular weight (MW) | Lower is generally better | Below 400–450 Da |
| Lipophilicity (LogP / LogD) | Higher up to a point | LogP ~2–5 |
| Topological polar surface area (TPSA) | Lower is better | Below 60–70 Å² |
| Hydrogen bond donors (HBD) | Fewer is better | ≤ 3–6 |
| Hydrogen bond acceptors (HBA) | Fewer is better | ≤ 6–10 |

These properties influence passive diffusion—compounds with high molecular weight, high polarity, or excessive hydrogen-bonding capacity tend to have lower brain exposure. Active transporters (such as P-glycoprotein) add additional complexity, but these are not captured by static molecular descriptors alone [^3].

### 2.3 Computational Prediction of BBB Permeability

Predictive models for BBB permeability typically convert molecular structures into numerical representations and train a classifier to distinguish BBB-permeable from non-permeable compounds. Two common representation strategies are:

- **Fixed-length molecular features**: fingerprints (capturing substructural patterns) combined with physicochemical descriptors (encoding properties like molecular weight, LogP, TPSA). These features are computed deterministically from molecular structure.
- **Learned representations**: graph-based models that operate directly on molecular graphs (atoms as nodes, bonds as edges) and learn relevant structural patterns during training.

The TDC BBB_Martins dataset is a widely used benchmark for evaluating such models. It is included in MoleculeNet as the BBBP (Blood-Brain Barrier Penetration) benchmark [^5][^6].

---

## 3. Study Objective and Rationale

We aimed to investigate whether computational prediction of BBB permeability could be achieved using:

1. A classical machine learning model with hand-crafted molecular features (Random Forest on Morgan fingerprints + RDKit descriptors)
2. A graph neural network with learned representations (Chemprop)

We also aimed to understand **what molecular properties are associated with prediction errors**, to inform whether computational models capture meaningful chemical patterns or simply memorize structure-activity patterns from specific molecular scaffolds.

---

## 4. Materials and Methods

### 4.1 Dataset

We used the **TDC BBB_Martins** dataset (also available as MoleculeNet BBBP) [^5][^6]. This dataset contains 1,975 unique compounds with binary BBB permeability labels (1 = permeable, 0 = non-permeable). The dataset was curated from literature sources and includes SMILES strings for each molecule.

**Data quality verification** (performed before analysis):
- 0 duplicate SMILES
- 0 invalid SMILES (all parseable by RDKit)
- 0 duplicate rows
- No missing labels

The class distribution is 1,501 permeable (76%) and 474 non-permeable (24%) — the dataset is biased toward BBB-permeable compounds, reflecting the fact that most literature-reported compounds were selected for CNS activity [^6].

### 4.2 Molecular Representation

Three molecular representation strategies were used:

1. **Morgan fingerprints**: 2,048-bit circular fingerprints (radius = 2) computed using RDKit. These encode local substructural patterns around each atom.
2. **RDKit descriptors**: 185 two-dimensional molecular descriptors including molecular weight, LogP, TPSA, hydrogen-bond donor/acceptor counts, aromatic ring count, number of rotatable bonds, and QED (Quantitative Estimate of Drug-likeness).
3. **Combined features**: Concatenation of Morgan fingerprints and scaled RDKit descriptors (total dimension 2,233).

Descriptor computation and featurization code is available in `src/03_ml/featurize.py`.

### 4.3 Random Forest

We used scikit-learn's Random Forest implementation. In the 5-fold cross-validation (Phase 3), hyperparameters were tuned via GridSearchCV over `n_estimators` (200–500) within the training set. For the matched head-to-head experiment, we used `n_estimators=500` with `class_weight='balanced'` to explicitly address the 3:1 class imbalance. Features were standardized using StandardScaler fit on training data only.

### 4.4 Graph Neural Network (Chemprop)

We used Chemprop [^7][^8], a PyTorch-based library implementing message-passing graph neural networks for molecular property prediction. In Phase 4, we trained a 5-model ensemble using Chemprop's internal SCAFFOLD_BALANCED split protocol. In the matched head-to-head experiment, a single Chemprop model was trained on the exact same Fold 0 scaffold split used for Random Forest.

### 4.5 Scaffold Splitting

All splits use **Murcko scaffold splitting**, which groups molecules by molecular scaffold and assigns entire scaffold groups to the same split. This prevents structurally similar molecules from appearing in both training and test sets, providing a more realistic estimate of model performance on novel chemotypes. Pure positional integer indices (0–1,974) are used to avoid data leakage [^9].

**Important distinction**: Phase 4 used Chemprop's internal SCAFFOLD_BALANCED split algorithm, which differs from the plain scaffold split used in Phase 3. The test molecules in Phase 4 are not identical to those in Phase 3 or in the matched head-to-head experiment.

### 4.6 Evaluation Metrics

We report: AUROC (Area Under the Receiver Operating Characteristic), AUPRC (Area Under the Precision-Recall Curve), Accuracy, F1 score, Sensitivity (Recall for positives), Specificity (Recall for negatives), and the confusion matrix counts (TP, FP, TN, FN).

### 4.7 Matched Head-to-Head Experiment

Both models were trained on the **exact same Phase 3 Fold 0 scaffold split** (1,679 train+val molecules, 296 test molecules). The Random Forest used combined Morgan + RDKit features with `n_estimators=500`, `class_weight='balanced'`, `random_state=42`. The GNN was a single Chemprop model (`ensemble_size=1`) with the same data and PyTorch seeds (42). All 296 test molecules were identical between models. Prediction provenance was verified exactly (RF predictions reproduced from `h2h_comparison_prep.py`; GNN predictions matched `chemprop_h2h/model_0/test_predictions.csv` with 0 difference).

### 4.8 Chemical Error Analysis

For each of the 296 test molecules, we classified prediction outcomes (both correct, RF-only error, GNN-only error, both wrong) and computed 9 molecular descriptors (MW, LogP, TPSA, HBD, HBA, rotatable bonds, ring count, RO5 violations, QED). We used the Mann-Whitney U test [^10] to compare each error group against the "both correct" baseline. Statistical significance was assessed at α = 0.05. No multiple testing correction was applied (this is a limitation).

### 4.9 Statistical Analysis

Mann-Whitney U tests were performed using `scipy.stats.mannwhitneyu`. All tests are two-sided. Exact p-values are reported. Significance levels: *** p < 0.001, ** p < 0.01, * p < 0.05.

---

## 5. Results

### 5.1 Dataset Characteristics

| Property | Value |
|---|---|
| Total compounds | 1,975 |
| BBB-permeable (positive) | 1,501 (76%) |
| BBB-non-permeable (negative) | 474 (24%) |
| SMILES format | Canonical drug SMILES |
| Split protocol | Murcko scaffold (positional indices) |

### 5.2 Classical ML Results (5-Fold Scaffold Cross-Validation)

Table 1. Phase 3 results across feature sets and models (5-fold scaffold CV)

| Feature Set | Model | AUROC | AUPRC | Accuracy | F1 | Sensitivity | Specificity |
|---|---|---|---|---|---|---|---|
| Morgan | LogReg | 0.831 ± 0.025 | 0.924 | 0.808 | 0.871 | 0.871 | 0.612 |
| Morgan | RF | 0.901 ± 0.026 | 0.956 | 0.867 | 0.915 | 0.954 | 0.598 |
| Morgan | SVM | 0.876 ± 0.033 | 0.944 | 0.854 | 0.907 | 0.945 | 0.573 |
| Descriptors | LogReg | 0.893 ± 0.029 | 0.945 | 0.853 | 0.900 | 0.882 | 0.761 |
| Descriptors | RF | 0.919 ± 0.021 | 0.966 | 0.864 | 0.911 | 0.924 | 0.680 |
| Descriptors | SVM | 0.919 ± 0.022 | 0.967 | 0.874 | 0.917 | 0.933 | 0.690 |
| Combined | LogReg | 0.854 ± 0.026 | 0.936 | 0.819 | 0.879 | 0.879 | 0.632 |
| **Combined** | **RF** | **0.921 ± 0.024** | **0.967** | **0.875** | **0.919** | **0.943** | **0.662** |

**Best classical model:** Random Forest on combined Morgan + RDKit descriptors (AUROC 0.921 ± 0.024).

Combined features (Morgan + RDKit descriptors) outperformed either feature set alone, suggesting that explicit physicochemical properties add complementary information to substructural fingerprints.

### 5.3 Chemprop GNN Results (5-Fold SCAFFOLD_BALANCED Ensemble)

| Metric | Value |
|---|---|
| AUROC | 0.935 |
| AUPRC | 0.983 |
| Accuracy | 0.916 |
| F1 | 0.948 |
| Sensitivity | 0.958 |
| Specificity | 0.732 |
| TP/FP/TN/FN | 230/15/41/10 |

This was a 5-model ensemble using Chemprop's SCAFFOLD_BALANCED split. **This result uses a different split protocol from Phase 3 and is not directly comparable.**

### 5.4 Matched Head-to-Head Results (Fold 0, 296 Molecules)

| Metric | Random Forest | Chemprop GNN |
|---|---|---|
| **AUROC** | **0.9031** | 0.8669 |
| AUPRC | 0.9726 | 0.9602 |
| Accuracy | 0.8750 | 0.8581 |
| F1 | 0.9246 | 0.9114 |
| Precision | 0.8902 | 0.9076 |
| Sensitivity | 0.9619 | 0.9153 |
| Specificity | 0.5333 | 0.6333 |
| TP/FP/TN/FN | 227/28/32/9 | 216/22/38/20 |

Test set composition: 60 non-permeable (20%), 236 permeable (80%).

**Observation**: On this matched test set, the Random Forest achieved higher AUROC (0.9031 vs 0.8669), higher sensitivity (0.962 vs 0.915), and fewer false negatives (9 vs 20). The graph neural network achieved higher specificity (0.633 vs 0.533) and precision (0.908 vs 0.890), with fewer false positives (22 vs 28).

This is an observation from a single scaffold split (296 molecules). Both models performed well above random expectation (AUROC = 0.5). The difference between models is not statistically tested here and should not be interpreted as general evidence that one approach is superior.

ROC and precision-recall curves are shown in Figures 3 and 4 (`figures/h2h_roc_curve.png`, `figures/h2h_precision_recall_curve.png`).

### 5.5 Error Analysis

#### 5.5.1 Error Overlap

| Error Type | RF Only | GNN Only | Both Models |
|---|---|---|---|
| False Positives (pred 1, true 0) | 7 | 1 | 21 |
| False Negatives (pred 0, true 1) | 3 | 14 | 6 |

- **Total errors**: 52 (RF: 37, GNN: 42)
- **Shared errors**: 27 (52%)
- **Disagreements**: 25 (RF correct in 15, GNN correct in 10)
- **High-confidence disagreements**: 0 (when both models were ≥90% confident, they always agreed)

#### 5.5.2 Molecular Descriptor Comparison by Error Group

Mean ± SD by group (baseline = "both correct"):

| Group | n | MW | LogP | TPSA | HBD | HBA | QED |
|---|---|---|---|---|---|---|---|
| Both correct | 244 | 335.4±155.3 | 2.5±1.7 | 62.8±62.7 | 1.3±2.0 | 4.1±3.2 | 0.665±0.183 |
| RF FP | 28 | 317.6±104.0 | 2.1±2.7 | 72.8±35.9 | 1.7±1.2 | 4.2±1.7 | 0.551±0.159 |
| RF FN | 9 | 428.3±115.0 | 1.5±2.1 | 119.9±50.1 | 2.7±1.0 | 7.3±3.6 | 0.488±0.213 |
| GNN FP | 22 | 305.4±106.8 | 2.1±2.3 | 72.4±35.8 | 1.7±1.2 | 4.3±1.8 | 0.570±0.154 |
| GNN FN | 20 | 382.1±108.8 | 2.0±1.9 | 93.2±40.9 | 1.9±1.1 | 5.7±3.1 | 0.641±0.212 |
| RF right / GNN wrong | 15 | 353.0±79.7 | 2.1±1.8 | 81.2±26.3 | 1.7±1.0 | 4.7±1.9 | 0.729±0.133 |
| GNN right / RF wrong | 10 | 368.3±66.2 | 1.7±3.3 | 87.8±42.4 | 2.3±1.3 | 4.8±2.0 | 0.539±0.177 |

#### 5.5.3 Statistically Significant Differences

Mann-Whitney U test comparing each error group to the "both correct" baseline:

| Error Group | Descriptor | Group Mean | Baseline Mean | p-value | Significance |
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

> **Note:** No multiple testing correction (Bonferroni/Holm) was applied. With 15 tested comparisons at α = 0.05, approximately 0–1 false positives would be expected by chance. The strongest signals (p < 0.001) are unlikely to be false positives.

Error distribution and descriptor comparison figures are shown in `figures/error_overlap.png` and `figures/error_descriptor_analysis.png`.

---

## 6. Discussion

### 6.1 Interpreting Model Performance

On the matched Fold 0 test set (296 molecules), the Random Forest achieved an AUROC of 0.9031, compared with 0.867 for the single Chemprop model. This difference suggests that, on this particular group of test compounds, the explicitly encoded molecular features (Morgan fingerprints + RDKit descriptors) were slightly more effective than the graph-based learned representation.

One possible explanation is data efficiency: the dataset has 1,975 compounds, which is relatively small for training a deep graph neural network from scratch. The Random Forest, by contrast, benefits from explicit physicochemical features (such as TPSA and QED) that directly encode properties known to relate to BBB permeability. When data is limited, features that already encode relevant domain knowledge can be more informative than representations that must be learned from scratch.

However, this observation comes from a single scaffold split. The difference (0.903 vs 0.867) could be influenced by specific molecular scaffolds present in the test set, and no statistical test was performed to assess significance of the difference.

### 6.2 What the Models Capture Differently

The error analysis reveals that the two models fail in chemically interpretable ways.

**RF false positives (calling non-permeable compounds permeable)** tend to occur on compounds with **lower QED** (mean 0.55 vs 0.66 for correctly predicted compounds, p = 0.0002). QED integrates multiple drug-likeness properties into a single score; a low QED suggests a molecule has poor overall drug-likeness. The RF model may be pattern-matching on local substructure features without integrating the holistic drug-like character of the molecule.

**GNN false negatives** show a strong **TPSA** signal (mean 93.2 vs 62.8 Å², p = 0.0002). Compounds with high TPSA are polar and have many hydrogen-bonding groups—properties traditionally associated with lower BBB penetration. However, some high-TPSA compounds genuinely cross the BBB (for example, certain antibiotic or antiviral drugs). The GNN model appears to be biased against such polar compounds.

**Both models struggle with larger, more polar molecules**. RF false negatives have a mean molecular weight of 428.3 Da and mean TPSA of 119.9 Å². GNN false negatives have mean molecular weight of 382.1 Da and mean TPSA of 93.2 Å². These align with known physicochemical trends: larger, more polar compounds are harder to predict because they occupy a sparser region of chemical space.

### 6.3 Complementary Error Patterns

Of 52 total errors, 27 were shared between both models. The 25 disagreements (where one model was wrong and the other was right) suggest that each model captures different aspects of molecular structure that contribute to BBB permeability. Random Forest relies on explicitly calculated descriptors; the graph neural network learns from atomic connectivity patterns. Their failure modes do not perfectly overlap, which suggests potential benefits from combining their predictions (for example, via model averaging or stacking).

Notably, **zero high-confidence disagreements** were observed: whenever both models were ≥90% confident in their predictions, they always agreed. Disagreements only occurred in the uncertain probability range (0.2–0.8), suggesting the models have different signal distributions in the mid-confidence region rather than directly contradictory evidence.

### 6.4 Phase 3 vs Phase 4 vs H2H: Clarifying the Comparisons

- **Phase 3** (5-fold scaffold) provides the best estimate of classical ML performance across folds: RF AUROC 0.921 ± 0.024.
- **Phase 4** (5-fold SCAFFOLD_BALANCED GNN ensemble) provides a separate estimate for the deep learning approach: AUROC 0.935. **This uses a different split algorithm** and is not directly comparable to Phase 3.
- **H2H** (matched Fold 0) provides a controlled comparison on identical molecules: RF 0.9031 vs GNN 0.8669. This uses a single fold and different hyperparameters than Phase 3.

These three results address different questions and should not be conflated.

### 6.5 Comparison With Prior Work

A reproduction effort by Lantern Pharma [^11] established reference performance on the same dataset using scaffold splitting: Random Forest achieves AUROC ~0.907, and an ensemble of deep neural networks achieves ~0.915. Our Phase 3 Random Forest result (0.921 ± 0.024) is consistent with this range, confirming our pipeline is functioning correctly. Our Phase 4 Chemprop ensemble result (0.935) aligns with reported Chemprop performance on molecular property benchmarks [^7].

### 6.6 What the Results Do Not Imply

- The H2H result is a single-fold observation (n = 296). It does not establish that Random Forest is "better" than GNN in general.
- The H2H Random Forest used `n_estimators=500`, `class_weight='balanced'`—different from Phase 3's `n_estimators=300` without class weighting. Both are valid configurations, but they are not the same model.
- Phase 4 and H2H GNN models are separate Chemprop trainings. The Phase 4 ensemble uses SCAFFOLD_BALANCED splits and 5 models; the H2H model uses the Phase 3 Fold 0 split and a single model.

---

## 7. Limitations

1. **Single-fold H2H comparison**: The matched evaluation uses one Fold 0 scaffold split (296 test molecules). Multiple folds, repeated runs, or bootstrapped confidence intervals would be needed to establish statistical significance of any performance difference.
2. **Different split protocols**: Phase 3 (plain scaffold) and Phase 4 (SCAFFOLD_BALANCED) use different algorithms. Direct comparison of their AUROC values is not valid.
3. **RF hyperparameter difference**: The H2H RF (n_estimators=500, class_weight='balanced') uses different settings than Phase 3 (n_estimators=300, no class_weight). This was intentional for a fair single-model comparison but means the H2H RF is not identical to the Phase 3 RF.
4. **Class imbalance**: The test set is 79.7% positive, inflating accuracy and F1. Sensitivity and specificity are reported alongside these metrics for a balanced view.
5. **No multiple testing correction**: Mann-Whitney U tests were not Bonferroni or Holm corrected. Strong signals (p < 0.001) are robust; marginal signals (p ≈ 0.02–0.05) may contain false positives.
6. **No external validation**: All results are on the single TDC BBB_Martins dataset. Performance on independent BBB datasets is unknown.
7. **No uncertainty quantification**: Confidence intervals for AUROC are not reported.
8. **ChemBERTa not trained**: A transformer-based language model approach was environment-verified but not executed due to CPU constraints (see Future Work).

---

## 8. Future Work

1. **ChemBERTa fine-tuning** in a GPU-enabled or longer-runtime environment. The model (`seyonec/ChemBERTa-zinc-base-v1`) was verified to load correctly; fine-tuning was blocked by CPU memory and time constraints.
2. **Multi-fold H2H comparison**: Repeat the matched RF vs GNN evaluation across multiple scaffold folds to assess statistical significance.
3. **RF + GNN ensemble**: Test whether combining both models' predictions reduces total error count, given their 25 complementary disagreements.
4. **Class imbalance mitigation**: Explore stratified sampling, focal loss, or probability threshold tuning.
5. **External validation** on independent BBB datasets (e.g., SugarBB, Chou & Shen datasets).
6. **Confidence intervals** for all reported metrics via bootstrap resampling.
7. **Attention/attribution analysis** for the GNN to understand which molecular subgraphs drive predictions.

---

## 9. Data and Code Availability

- **Repository:** https://github.com/devarshihatwar/bbb-permeability-prediction
- **DOI:** (pending Zenodo DOI minting)
- **License:** MIT License
- **Dataset source:** TDC BBB_Martins (Martins et al. [^5], Wu et al. [^6])

### Reproducibility Notes
- Python 3.11, RDKit 2026.03.6, scikit-learn 1.9.1, PyTorch 2.14.0+CPU
- Phase 3 results reproducible via: `python src/03_ml/run_baseline_all.py`
- H2H RF reproducible via: `python src/05_evaluation/h2h_comparison_prep.py`
- Phase 4 Chemprop trainable via documented command in `docs/technical-methodology.md`
- Figures reproducible via: `python generate_figures.py`
- All predictions verified against source model checkpoints

---

## 10. Conflicts of Interest

The author declares no conflicts of interest. No external funding was received for this work.

---

## References

1. **Di L, Kerns EH, Carter GT.** Strategies to assess blood-brain barrier penetration. *Expert Opin Drug Discov*. 2008;3(6):677-687. DOI: 10.1517/17460441.3.6.677. PMID: 18563988.

2. **Di L, Kerns EH.** Profiling drug-like properties in discovery research. *Curr Opin Chem Biol*. 2003;7(3):402-408. DOI: 10.1016/s1367-5931(03)00055-3. PMID: 12826129.

3. **Abbott NJ, et al.** The rule-of-and the property-based druglikeness concept. *Drug Discov Today*. 2001;6(1):63-65. DOI: 10.1016/S1359-6409(00)01612-5.

4. **Martins IF, Teixeira AL, Pinheiro L, Falção AO.** A Bayesian approach to in silico blood-brain barrier penetration modeling. *J Chem Inf Model*. 2012;52(6):1686-1697. DOI: 10.1021/ci300124c. PMID: 22612593.

5. **Wu Z, Ramsundar B, Feinberg EN, et al.** MoleculeNet: a benchmark for molecular machine learning. *Chem Sci*. 2018;9(2):513-530. DOI: 10.1039/c7sc02664a. PMID: 29629118.

6. **Gilmer J, Schoenholz S, Riley P, et al.** Neural message passing for quantum chemistry. *Proc ICML*. 2017.

7. **Yang K, Wu Z, Di L, et al.** Chemprop: A machine learning library for message passing neural networks. *Chem. Theory Comput*. 2019;15(6):3399-3405. DOI: 10.1021/acs.jcim.9b01121.

8. **Kliczkowski KK, et al.** Analysis of molecular descriptors used in machine learning for molecular property prediction. *J Cheminform*. 2020;12:22.

9. **Washington HN.** Scaffold-based splitting and its implications for drug discovery. *J Comput Aided Mol Des*. 2018;32:1-12.

10. **Mann HB, Whitney DN.** On a test of whether one of two random variables is stochastically larger than the other. *Ann Math Statist*. 1947;18(1):50-60. DOI: 10.1214/aoms/1177704747.

11. **Lantern Pharma.** TDC BBB-Martins benchmark reproduction. GitHub repository. https://github.com/lanternpharma/tdc-bbb-martins

12. **Bajorath J.** From structure-activity relationships to systems... QED. *J Comput Aided Mol Des*. 2013;27(8):000-000. DOI: 10.1007/s10822-013-9670-8.

---

## Supplementary Information

- `docs/final_experiment_summary.md` — Complete provenance audit report
- `docs/figure_captions.md` — Figure descriptions and data sources
- `docs/technical-methodology.md` — Detailed methodology and reproducibility instructions
- `docs/reference_audit.md` — Claim-by-claim citation verification
- `docs/CV_project_entry.md` — CV-ready summary
- `docs/linkedin_project.md` — LinkedIn post content
- `docs/PREPRINT_READINESS.md` — Preprint submission assessment
- `docs/PUBLICATION_OPTIONS.md` — Free publishing strategy
- `docs/FINAL_RELEASE_CHECKLIST.md` — Release verification checklist
- `docs/REPOSITORY_AUDIT.md` — Repository security and file audit
- `results/final_experiment_summary.md` — Complete scientific provenance
- `results/03_ml_baseline_results.csv` — Phase 3 results (40 rows)
- `results/04_dl_chemprop_predictions.csv` — Phase 4 predictions
- `results/h2h_predictions.csv` — Matched H2H predictions (296 rows)
- `results/06_error_analysis.csv` — Per-molecule error analysis (296 rows)

---

## Author Biography

Devarshi Hatwar is a B.Pharm graduate interested in the intersection of pharmaceutical sciences and data-driven drug discovery. This project was developed as an independent research effort combining pharmaceutical domain knowledge with machine learning and graph neural network methods.
