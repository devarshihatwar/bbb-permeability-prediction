# LinkedIn Project Entry

## A. LinkedIn Projects Section Description

**Comparative ML vs GNN for BBB Permeability Prediction**

Built a reproducible pipeline comparing Random Forest and Chemprop Graph Neural Network on the TDC BBB_Martins dataset (1,975 compounds). Achieved AUROC 0.921±0.024 (RF, 5-fold scaffold CV) and 0.935 (Chemprop ensemble). Conducted a matched head-to-head on Fold 0 (RF 0.903 vs GNN 0.867) with chemical error analysis revealing complementary failure patterns. All splits audited for leakage (zero train/test SMILES overlap). ChemBERTa fine-tuning was environment-verified but not executed due to CPU constraints.

**Tech stack:** Python, scikit-learn, RDKit, PyTorch, Chemprop, pandas, numpy, matplotlib

**GitHub:** [GITHUB URL]

**DOI:** [ZENODO DOI]

---

## B. LinkedIn Post

I'm excited to share a research project I've been working on: comparing classical machine learning against graph neural networks for predicting blood-brain barrier (BBB) permeability.

**The problem:** BBB permeability is a critical bottleneck in CNS drug development. Getting it wrong wastes months of synthesis and testing. Can we build a reliable in silico predictor — and more importantly, understand *why* each model fails?

**What I built:**
- A reproducible pipeline from the TDC BBB_Martins dataset (1,975 compounds)
- 3 feature sets (Morgan fingerprints + RDKit descriptors, combined)
- 7 model configurations (LogReg, Random Forest, SVM) across 5-fold scaffold splits
- A Chemprop GNN ensemble (5 models) on SCAFFOLD_BALANCED
- A matched head-to-head: both RF and single GNN on the *exact same* 296-molecule test set

**Key finding:** On the matched Fold 0 comparison, RF outperformed the single GNN (AUROC 0.903 vs 0.867), but with different strengths: RF had higher sensitivity (catches more permeable compounds), while GNN had higher specificity (fewer false positives). The two models made largely *disjoint* errors — 25 of 52 failures were model-specific, and zero high-confidence disagreements were found.

**The chemistry behind the errors:**
- RF false positives concentrate on low-QED (poor drug-likeness) compounds (p=0.0002)
- GNN false negatives hit high-TPSA polar molecules that genuinely cross the BBB (p=0.0002)
- Both models struggle with large, polar molecules (high MW, high TPSA)

**Important caveat:** The Phase 4 GNN ensemble (0.935) used a different split protocol and is not directly comparable to Phase 3 RF (0.921). The matched H2H comparison is from a single fold — it's a fair comparison but not generalizable on its own.

**Environment challenge:** I attempted ChemBERTa fine-tuning (seyonec/ChemBERTa-zinc-base-v1) but the CPU environment killed the process after ~5 minutes. The model, tokenizer, and dataset all checked out — just not enough compute time. This is marked as future work.

**Lessons learned:**
1. Provenance matters — I audited every H2H prediction against its source model and split
2. Scaffold splitting is non-negotiable for molecular ML, but the specific algorithm matters
3. Complementary errors suggest ensembling RF + GNN could work
4. Environment limits shape research scope as much as methodology

All code, data, results, figures, and a full manuscript are available at [GITHUB URL]. A Zenodo DOI is pending: [ZENODO DOI].

#MachineLearning #DrugDiscovery #MedicinalChemistry #ComputationalBiology #DeepLearning #RandomForest #GraphNeuralNetworks #OpenSource #ReproducibleResearch #BBB

---

## C. Short Version (Twitter/X style)

Built a reproducible BBB permeability prediction pipeline comparing RF vs Chemprop GNN. RF: 0.921 AUROC (5-fold). GNN ensemble: 0.935 (different split). Matched H2H: RF 0.903 vs GNN 0.867, with complementary error patterns. Error analysis reveals RF over-predicts low-QED compounds (p=0.0002) and GNN misses high-TPSA compounds (p=0.0002). ChemBERTa environment-verified but not trained (CPU limits). Code: [GITHUB URL]

---

## D. Suggested Project Title

"RF vs GNN for BBB Permeability: A Reproducible Head-to-Head with Chemical Error Analysis"

---

## E. Suggested Skills/Tags

- Machine Learning
- Deep Learning
- Drug Discovery
- Medicinal Chemistry
- Computational Biology
- Python
- PyTorch
- scikit-learn
- RDKit
- Graph Neural Networks
- Random Forest
- Molecular Machine Learning
- Reproducible Research
- Scaffold Splitting
- Error Analysis
