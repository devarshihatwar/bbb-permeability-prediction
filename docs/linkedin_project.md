# LinkedIn Project Entry — BBB Permeability Prediction

## A. LinkedIn Projects Section Description

**AI-Based Blood–Brain Barrier Permeability Prediction**

Built a reproducible pipeline to predict whether drug molecules can cross the blood-brain barrier — a critical property for CNS drug development. Using the TDC BBB_Martins dataset (1,975 compounds), I compared Random Forest (Morgan fingerprints + RDKit descriptors) against a Chemprop graph neural network.

Key outcomes:
- Phase 3: RF achieved AUROC 0.921 ± 0.024 (5-fold scaffold CV)
- Phase 4: Chemprop 5-model ensemble achieved AUROC 0.935
- Matched H2H (Fold 0, 296 molecules): RF 0.9031 vs GNN 0.8669
- Error analysis revealed RF false positives on low-QED compounds, GNN false negatives on high-TPSA molecules

ChemBERTa fine-tuning was environment-verified but not executed due to CPU constraints. All code, results, and figures are openly available.

**Tech stack:** Python | RDKit | scikit-learn | PyTorch | Chemprop | pandas | numpy | matplotlib

Repository: [GITHUB URL]
Code DOI: [ZENODO DOI]

---

## B. LinkedIn Post

I wanted to explore how AI could be applied to a problem that's directly relevant to drug discovery.

As someone with a B.Pharm background, I was familiar with the blood-brain barrier (BBB) — the filter that prevents about 98% of small-molecule drugs from reaching the brain. Cross the BBB or fail in CNS development. But predicting permeability experimentally (in vitro assays, mouse models) is slow and expensive. So I set out to build a computational predictor and understand when it works — and when it doesn't.

**The dataset:** 1,975 drug compounds from TDC BBB_Martins, labeled as permeable or not-permeable. About 76% are permeable, which means there's a class imbalance to watch out for.

**Two approaches, two representations:**
- Random Forest with Morgan fingerprints + 185 RDKit descriptors (physicochemical features I choose)
- Chemprop graph neural network (learns molecular features from atom-bond graphs)

Both trained on scaffold splits — not random splits — so the models actually face novel molecular scaffolds in testing.

**Results in brief:**
- RF: AUROC 0.921 ± 0.024 across 5-fold cross-validation
- GNN ensemble (5 models): AUROC 0.935
- Both are solid, single-metric numbers

But I wasn't satisfied with just comparing across different splits. So I ran a **matched head-to-head**: both models on the exact same 296 test molecules from Fold 0.

Result: RF got 0.903, the single GNN got 0.867. But here's what's interesting — they failed on **different** molecules. Of 52 total errors, only 27 were shared. And when both models were highly confident (≥90% probability), they always agreed — disagreements only happened in the uncertain mid-range.

**The chemistry behind the errors** (this is my favorite part):
- RF's false positives — calling non-permeable compounds permeable — tended to be low-QED molecules (poor drug-likeness). It seemed to pattern-match on substructures without checking overall drug-likeness.
- GNN's false negatives — missing compounds that actually cross the BBB — had high TPSA (polar surface area). It was biased against large, polar molecules.
- Both models struggled with larger, more polar compounds.

These aren't just numbers on a benchmark. High-TPSA compounds that genuinely cross the BBB are exactly the kind of challenging cases a drug discovery team would care about. Missing them isn't just a metric drop — it's a candidate compound overlooked.

**Limitations I'm honest about:**
- The H2H comparison is a single fold (296 molecules) — not enough to declare one model "better"
- RF and GNN used different hyperparameters in the H2H experiment
- Phase 3 and Phase 4 used different split protocols, so their AUROC values aren't directly comparable
- ChemBERTa was planned but not trained (CPU ran out of memory/time) — it stays on the to-do list

**What I learned:**
1. Scaffold splitting is non-negotiable in molecular ML — random splits give misleading numbers
2. A single AUROC doesn't tell the full story — error analysis reveals the chemical patterns
3. Explicit physicochemical features (QED, TPSA) can be more data-efficient than learned graph representations on small datasets
4. Model errors are complementary — RF and GNN make different mistakes, suggesting ensemble potential

All code, data, results, and figures are in the repository. The full provenance audit (tracing every prediction back to its source model and split) is in `results/final_experiment_summary.md`.

[Link to repository]
[DOI — pending Zenodo]

#DrugDiscovery #MachineLearning #BloodBrainBarrier #ComputationalDrugDiscovery #ChemicalErrorAnalysis #ReproducibleResearch #Cheminformatics

---

## C. Short Version (for comments/reposts)

Quick project: I built an AI pipeline to predict blood-brain barrier permeability, comparing Random Forest vs a Chemprop graph neural network. On a matched test set (296 compounds), RF got AUROC 0.903 vs GNN 0.867 — but the real insight was in the error analysis: RF over-predicts permeability for low-drug-likeness compounds (p=0.0002), while GNN misses high-TPSA polar molecules (p=0.0002). Both models fail on large, polar molecules. Full reproducible pipeline at [GITHUB URL]

---

## D. Suggested Project Title

"AI-Based Blood–Brain Barrier Permeability Prediction: A Reproducible Comparison of Random Forest and Graph Neural Networks with Chemical Error Analysis"

---

## E. Suggested Skills/Tags

- Machine Learning
- Deep Learning
- Drug Discovery
- Cheminformatics
- Random Forest
- Graph Neural Networks
- Python
- Chemprop
- RDKit
- Reproducible Research
- Molecular Machine Learning
- Medicinal Chemistry
- BBB Permeability
