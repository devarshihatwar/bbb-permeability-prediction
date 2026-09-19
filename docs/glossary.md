# Glossary — BBB Penetration Prediction Project

A plain-language reference for the cheminformatics and ML terms used throughout this project.

## Molecules and representations

| Term | Plain-language definition |
|---|---|
| **SMILES** | A text string encoding a molecule's structure (atoms + bonds). E.g., `"CCO"` = ethanol. Each unique SMILES maps to one molecular structure. |
| **2-D / 3-D structure** | 2-D = connectivity only (what's bonded to what). 3-D = adds atomic coordinates in space, which matters for shape-based binding and some descriptor types. |
| **Isomer** | Molecules with the same atoms and bonds but different 3-D arrangements (stereoisomers). Affects biological activity. |
| **Canonical SMILES** | A standardized SMILES string (same molecule → same string). RDKit produces this by default. |
| **Drug ID (`drug_id`)** | A numeric or text identifier for each compound in the dataset. |

## Molecular descriptors & features

| Term | Plain-language definition |
|---|---|
| **Molecular descriptor** | Any number, category, or label that numerically describes a molecular property. E.g., "molecular weight = 318.4". |
| **Morgan / ECFP fingerprint** | A binary bit-string (e.g., 2,048 bits) where each bit signals the presence of a specific local atomic pattern (e.g., "a carbon bonded to two nitrogens in a ring"). Captures substructural similarity. |
| **RDKit descriptors** | The ~200 built-in property calculations RDKit provides: molecular weight, logP, H-bond donors, TPSA, ring counts, etc. |
| **logP** | A measure of lipophilicity: how much a molecule prefers a fatty (lipid) layer vs. water. High logP → more lipid-soluble → more likely to cross membranes. |
| **logD** | Like logP, but accounts for ionization at a specific pH (usually 7.4). Often more realistic for drug molecules. |
| **Molecular weight (MW)** | The mass of the molecule in Daltons (Da). Lower MW generally helps membrane crossing. |
| **Polar surface area (PSA / TPSA)** | The surface area of polar atoms (N, O) in a molecule. Lower PSA → easier membrane crossing. Guideline: < 60–70 Å² for BBB. |
| **H-bond donors (HBD)** | Atoms (O, N) bonded to hydrogen that can form hydrogen bonds with the barrier. Fewer is better for brain entry. |
| **H-bond acceptors (HBA)** | Atoms (O, N) that can accept a hydrogen bond. Fewer is generally better for brain entry. |
| **Autocorrelation descriptors** | Numbers encoding how atomic properties (mass, charge, electronegativity) are distributed spatially across the molecule. |
| **WHIM descriptors** | "Weighted Holistic Atomic Descriptors" — 3-D shape/charge distribution descriptors derived from atomic positions. Require 3-D coordinates. |
| **Kernel PCA** | A non-linear version of Principal Component Analysis; used here to compress many correlated descriptors into fewer "components" the model can use. |
| **Feature importance** | A score from tree-based models (like Random Forest) saying how much each input feature contributed to correct predictions. |

## Split strategies

| Term | Plain-language definition |
|---|---|
| **Random split** | Shuffle all molecules, then divide into train / validation / test (e.g., 70 / 10 / 20 %). Molecules in different sets may share the same "scaffold" (core ring structure). |
| **Scaffold split** | Group molecules by their core molecular scaffold (ring system), then assign entire scaffolds to train/test, so the model never sees a scaffold it trained on when tested. Tests true generalization. Harder than random split. |
| **5-fold cross-validation** | Divide data into 5 folds; train on 4, test on 1; repeat 5 times (each fold is the test set once). Report the average metric. Gives a robust performance estimate. |

## Machine-learning models

| Term | Plain-language definition |
|---|---|
| **Logistic Regression** | A simple model that draws a straight line (hyperplane) to separate the two classes. Interpretable but limited to linear patterns. |
| **Random Forest (RF)** | An ensemble of many decision trees. Each tree votes on the answer; the majority wins. Handles non-linear patterns and missing data well. |
| **SVM (Support Vector Machine)** | Draws the widest possible "margin" between classes in feature space. With an RBF kernel, it can model non-linear boundaries. |
| **SMOTE** | "Synthetic Minority Over-sampling Technique." Generates artificial examples of the minority class to balance an imbalanced dataset. |
| **AUROC** | Area Under the Receiver Operating Characteristic curve. Ranges 0.5 (random guessing) to 1.0 (perfect). Probability that the model ranks a random "crosses BBB" molecule higher than a random "doesn't cross" one. |
| **AUPRC** | Area Under the Precision-Recall curve. Better than AUROC when classes are imbalanced (our case: 79 % vs 21 %). |
| **Sensitivity (Recall)** | Of all true BBB-permeable molecules, the fraction correctly predicted. (True positives / all actual positives.) |
| **Specificity** | Of all true non-permeable molecules, the fraction correctly predicted. (True negatives / all actual negatives.) |
| **SHAP (SHapley Additive exPlanations)** | A method that explains *why* a model made a specific prediction by attributing importance to each input feature. |

## Deep learning models

| Term | Plain-language definition |
|---|---|
| **Graph Neural Network (GNN)** | A neural network that operates directly on graphs (atoms = nodes, bonds = edges). "Passes messages" between atoms to learn which structural patterns matter. |
| **Message passing** | The core GNN operation: each atom updates its representation based on information from its bonded neighbors, iterated a few times. |
| **Chemprop** | A popular PyTorch implementation of message-passing GNNs specifically for molecular property prediction. Pip-installable. |
| **Pre-trained model** | A large model (e.g., ChemBERTa) trained on a huge corpus (10M+ molecules) that we then adapt ("fine-tune") on our small 1,975-row dataset. |
| **Fine-tuning** | Loading a pre-trained model's weights and continuing training on the target task with a small learning rate. Leverages what the model already learned. |
| **ChemBERTa** | A RoBERTa-style transformer language model pre-trained on SMILES strings. Can be fine-tuned as a classifier. |
| **Transformer** | A neural network architecture (the basis of ChatGPT) that uses self-attention to weigh the importance of different input tokens. Applied to SMILES here. |
| **RoBERTa** | A masked-language-model transformer. ChemBERTa is a RoBERTa model pre-trained on SMILES instead of English text. |
| **Token** | A unit of input. For ChemBERTa, tokens are sub-word pieces of the SMILES string (e.g., "C", "##C", "##(C)C", etc.). |

## Advanced / extension concepts

| Term | Plain-language definition |
|---|---|
| **Boltz-2** | A diffusion model that predicts 3-D molecular structure (and protein–ligand binding). Not a BBB classifier; would need to be adapted (e.g., to generate 3-D conformers → derive 3-D-aware features). Requires GPU; not tractable on this CPU-only machine. |
| **3-D conformer** | A molecule's 3-D arrangement of atoms in space. Different conformers of the same molecule can have different properties. |
| **Diffusion model** | A generative model that incrementally "denoises" a structure; used for molecular 3-D generation and structure prediction. |

## Tools & libraries

| Term | Plain-language definition |
|---|---|
| **RDKit** | The de-facto open-source cheminformatics library (Python). Computes descriptors, fingerprints, draws 2-D structures, validates SMILES. |
| **PyTorch** | A deep-learning framework. The GNN and ChemBERTa are built on PyTorch tensors. |
| **Scikit-learn** | A classic Python ML library (Random Forest, SVM, Logistic Regression, cross-validation). |
| **Pandas** | A Python data-manipulation library (DataFrames). Used for the CSV data. |
| **NumPy** | A Python library for numerical arrays. Underlying all the other libraries. |
| **Streamlit** | A Python library that turns a script into a web UI with almost no frontend code. |
| **PyPI / pip** | The Python package index and installer. `pip install <package>`. |
| **conda** | A cross-platform package/environment manager. Alternative to pip. |
| **GPU** | Graphics Processing Unit. Used to accelerate deep-learning training. This machine has **no GPU** (CPU-only PyTorch). |

## Domain context

| Term | Plain-language definition |
|---|---|
| **BBB** | Blood-Brain Barrier. The barrier that separates blood from brain tissue. |
| **CNS** | Central Nervous System (brain + spinal cord). |
| **Permeable** | Able to cross the BBB. |
| **Physicochemical** | Physical + chemical properties (weight, solubility, charge, etc.) of a molecule. |
| **ADME** | Absorption, Distribution, Metabolism, Excretion — the four "fates" of a drug in the body. BBB penetration is part of "Distribution." |
| **Pharmacovigilance (PV)** | Drug safety monitoring. Understanding which molecules cross the BBB is relevant for predicting CNS adverse effects. |
| **eCTD** | Electronic Common Technical Document — the standard format for regulatory drug submissions. |
| **P-glycoprotein (P-gp)** | An efflux transporter in the BBB that pumps certain drugs *out* of the brain, reducing their brain concentration. Causes false negatives in passive-permeability models. |
