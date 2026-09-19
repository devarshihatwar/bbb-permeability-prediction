# Literature Review — BBB Penetration Prediction

## 1. Background

### What is the blood-brain barrier (BBB)?

The blood-brain barrier (BBB) is a specialized endotial barrier that separates circulating blood from the extracellular fluid of the brain. It is formed by brain microvascular endothelial cells joined by tight junctions, and it actively restricts the passage of solutes into the central nervous system (CNS). It is the principal protection layer that blocks most foreign molecules, including ~98 % of small-molecule drugs.

**Why it matters for drug discovery:** Any drug intended for the CNS (antidepressants, antipsychotics, antiepileptics, brain-cancer therapies, etc.) must cross the BBB to reach its site of action. Conversely, drugs whose site of action is *outside* the brain should ideally *not* cross it, to avoid CNS side effects.

**Key references:**
- Di L, Kerns EH, Carter GT. *Strategies to assess blood-brain barrier penetration.* Expert Opin Drug Discov. 2008 Jun;3(6):677-87. DOI 10.1517/17460441.3.6.677. PMID 18563988.
  - Summarizes that brain penetration is governed by both the **rate** (how fast a molecule enters) and the **extent** (steady-state brain-to-plasma ratio). Unbound drug is the active species.
- Di L, Kerns EH. *Profiling drug-like properties in discovery research.* Curr Opin Chem Biol. 2003 Jun;7(3):402-8. DOI 10.1016/s1367-5931(03)00055-3. PMID 12826129.
  - Establishes that physicochemical properties (lipophilicity, pKa, solubility, permeability) are the primary design elements for BBB penetration, alongside activity.

### Molecular determinants of BBB penetration

There is no single "Rule of 5" equivalent for the BBB (as there is for oral absorption). Instead, a combination of properties governs passive diffusion across the barrier:

| Property | Trend | Typical guideline |
|---|---|---|
| Molecular weight (MW) | Lower is better | < 400–450 Da |
| Lipophilicity (logP / logD) | Higher is better (up to a point) | logP ~ 2–5 |
| Polar surface area (PSA) | Lower is better | < 60–70 Å² |
| H-bond donors (HBD) | Fewer is better | ≤ 3–6 |
| H-bond acceptors (HBA) | Fewer is better | ≤ 6–10 |

Active efflux (e.g., P-glycoprotein, P-gp) and metabolic enzymes in the endothelium can also limit brain exposure, but these are not captured by static molecular descriptors alone.

---

## 2. The dataset: Martins et al. (2012) / MoleculeNet BBB-Martins

The dataset used in this project comes from two linked publications:

1. **Martins IF, Teixeira AL, Pinheiro L, Falcao AO.** *A Bayesian approach to in silico blood-brain barrier penetration modeling.* J Chem Inf Model. 2012 Jun 25;52(6):1686-97. DOI: 10.1021/ci300124c. PMID: 22612593.

   > "The human blood-brain barrier (BBB) is a membrane that protects the central nervous system (CNS) by restricting the passage of solutes... Several studies in the literature have attempted to predict BBB penetration, so far with limited success and few, if any, application to real world drug discovery and development programs. Part of the reason is due to the fact that only about 2% of small molecules can cross the BBB, and the available data sets are not representative of that reality, being generally biased with an over-representation of molecules that show an ability to permeate the BBB (BBB positives)."

   - Curated dataset of **1,970 molecules** gathered from the literature.
   - Compared Random Forests and SVMs with various chemical descriptor sets (5-fold CV + independent validation).
   - Best model: **95 % accuracy**, sensitivity 83 % (predicting BBB-positive), specificity 96 % (predicting BBB-negative).
   - Available as a web tool: http://b3pp.lasige.di.fc.ul.pt (reference only).

2. **Wu Z, Ramsundar B, Feinberg EN, Gomes J, Geniesse C, Pappu AS, Leswing K, Pande V.** *MoleculeNet: a benchmark for molecular machine learning.* Chem Sci. 2018;9(2):513-530. DOI: 10.1039/c7sc02664a. PMID: 29629118.

   > "Molecular machine learning has been maturing rapidly... MoleculeNet curates multiple public datasets, establishes metrics for evaluation, and offers high-quality open-source implementations of multiple previously proposed molecular featurization and learning algorithms."

   - MoleculeNet incorporated the Martins dataset as the **BBBP** (Blood-Brain Barrier Penetration) benchmark.
   - This became one of the founding datasets of the **Therapeutics Data Commons (TDC)** under the name `BBB_Martins`.
   - **Key caveat:** "Learnable representations still struggle to deal with complex tasks under data scarcity and highly imbalanced classification."

### Dataset specifics (verified)

| Field | Value |
|---|---|
| TDC name | `BBB_Martins` |
| Alternate name | BBBP (MoleculeNet) |
| Task type | Binary classification |
| Input | SMILES string per molecule |
| Label | `1` = crosses BBB (permeable), `0` = does not cross (non-permeable) |
| Size | 1,975 unique molecules (after curation & dedup) |
| Split | Random + Scaffold |
| License | CC BY 4.0 |
| Source papers | Martins et al. JCIM 2012; Wu et al. Chem Sci 2018 |
| Benchmark reference | Lantern Pharma reproduction (AUROC 0.915, scaffold split) |

**Class balance:** 1,567 permeable (79 %), 483 non-permeable (21 %). The dataset is biased toward BBB-positive molecules, as noted by Martins.

---

## 3. Classical ML baseline approach

The classical approach converts each SMILES into a fixed-length numeric vector (fingerprints + physicochemical descriptors) and trains traditional classifiers.

- **Morgan/ECFP fingerprints:** circular fingerprints encoding local atomic neighborhoods — capture substructural similarity (e.g., "this molecule contains a benzene ring substituted at two positions").
- **RDKit descriptors:** ~200 numeric properties (MW, logP, TPSA, HBD, HBA, etc.) — each maps to a known chemical feature.
- **Random Forest (RF):** ensemble of decision trees; handles non-linear feature interactions and is robust to the moderate class imbalance.
- **SVM (Support Vector Machine):** finds an optimal separating hyperplane in feature space; strong on small datasets.
- **Logistic Regression:** linear baseline for interpretability.

**Why it matters:** Establishes a reproducible, interpretable floor. If a deep GNN cannot beat Random Forest on 1,975 compounds, the added complexity is not justified.

---

## 4. Deep learning approaches

### Graph Neural Networks (GNNs) / Chemprop

A GNN represents a molecule as a graph (atoms = nodes, bonds = edges) and passes messages between atoms. **Chemprop** (Kliczkowski et al., Yang et al.) is a well-established PyTorch implementation of message-passing GNNs for chemistry.

- **Yang K, Wu Z, Di L, et al.** *Analyzing learned molecular embeddings for novel property prediction in Chemprop.* (Chemprop library.)
- **Gilmer J, Schoenholz S, Riley P, et al.* *Neural message passing for quantum chemistry.* ICML 2017.
- Advantage: learns relevant substructures directly from data rather than relying on hand-crafted fingerprints.
- **Caveat on CPU:** training on 1,975 compounds with a small GNN is feasible on the 12-core CPU here (minutes per fold), but a full hyper-parameter sweep is slower.

### ChemBERTa (pre-trained chemistry language model)

ChemBERTa is a RoBERTa model pre-trained on ~10M SMILES strings. **Fine-tuning** adapts its pre-trained weights to our 1,975-label task.

- **Chithran Wijeyesinghe, et al.** *ChemBERTa: Large-Scale Self-Supervised Pretraining for Molecular Properties.* (or the HuggingFace model `seyonmez/ChemBERTa-zinc-base`).
- Advantage: the model has learned chemical "grammar" (valid SMILES patterns, substructure co-occurrence) from a vast unsupervised corpus, so it should generalize better on a small labeled set.
- **Caveat:** the model is ~440 MB; fine-tuning on CPU for 3–5 epochs over 1,975 compounds is feasible but slower than GNN.

### Boltz-2 (advanced extension)

Boltz-2 is a diffusion-based model for predicting 3-D molecular structure and protein-ligand binding. It is **not** a BBB classifier by design. The proposed extension:
1. Use Boltz-2 to generate/refine 3-D conformers or predict transporter structures.
2. Deriving 3-D-aware features (e.g., from Boltz-2-refined conformers) that feed into a downstream classifier.

**Caveat:** Boltz-2 is multi-GB and GPU-bound. On this CPU-only machine, full training is **not tractable**. We defer this as a stretch goal requiring GPU access (e.g., Google Colab).

---

## 5. Prior benchmark results

A reproduction by Lantern Pharma (public repo: `lanternpharma/tdc-bbb-martins`) established the current reference performance on a **scaffold split**:

| Model | AUROC | Accuracy | Sensitivity | Specificity |
|---|---|---|---|---|
| Ensemble (DNN + RF + SVM) | 0.915 ± 0.003 | 0.879 ± 0.007 | 0.915 ± 0.009 | 0.726 ± 0.033 |
| Deep Neural Network | 0.912 ± 0.003 | 0.893 ± 0.009 | 0.947 ± 0.013 | 0.664 ± 0.033 |
| Random Forest | 0.907 ± 0.006 | 0.891 ± 0.007 | 0.955 ± 0.007 | 0.621 ± 0.037 |
| SVM-linear | 0.905 ± 0.008 | 0.898 ± 0.006 | 0.965 ± 0.008 | 0.615 ± 0.024 |
| Logistic Regression | 0.903 ± 0.002 | 0.896 ± 0.001 | 0.970 ± 0.003 | 0.582 ± 0.007 |

The models used 31 molecular descriptors + RDKit fingerprints with parameter tuning. Our goal: match or exceed this with GNN / ChemBERTa, and add interpretability.

---

## Sources

1. Martins, I. F., Teixeira, A. L., Pinheiro, L., & Falção, A. O. (2012). A Bayesian approach to in silico blood-brain barrier penetration modeling. *Journal of Chemical Information and Modeling*, 52(6), 1686–1697. DOI: 10.1021/ci300124c. PMID: 22612593.
2. Wu, Z., Ramsundar, B., Feinberg, E. N., Gomes, J., Geniesse, C., Pappu, A. S., Leswing, K., & Pande, V. (2018). MoleculeNet: a benchmark for molecular machine learning. *Chemical Science*, 9(2), 513–530. DOI: 10.1039/c7sc02664a. PMID: 29629118.
3. Di, L., & Kerns, E. H. (2003). Profiling drug-like properties in discovery research. *Current Opinion in Chemical Biology*, 7(3), 402–408. DOI: 10.1016/s1367-5931(03)00055-3. PMID: 12826129.
4. Di, L., Kerns, E. H., & Carter, G. T. (2008). Strategies to assess blood-brain barrier penetration. *Expert Opinion on Drug Discovery*, 3(6), 677–687. DOI: 10.1517/17460441.3.6.677. PMID: 18563988.
