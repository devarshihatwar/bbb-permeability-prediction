# BBB Penetration Prediction — End-to-End Project Plan

A portfolio / research project by two people:
- **Person 1 (P1)**: B.Pharm, Medical-Records Analyst — leads the drug-science, data-understanding, scientific-interpretation, and documentation tracks.
- **Person 2 (P2)**: B.Tech, System Engineer — leads the software-engineering, ML/DL pipelines, evaluation-harness, and application-development tracks.

Goal: predict whether a small molecule crosses the blood–brain barrier (BBB) using the **Therapeutics Data Commons (TDC) `BBB_Martins`** dataset, comparing four computational approaches — classical ML, a GNN/chemprop model, ChemBERTa, and (optionally) a physics-aware extension — then ship a lightweight prediction app.

> ⚠️ **No coding starts yet.** This is a phase-by-phase plan. Each phase lists: what to learn, what to build, why it matters, and the expected output. The plan is designed for a Windows 10, CPU-only machine (no GPU) with Python 3.11, RDKit, PyTorch (CPU), and pip-installable `chemprop`/`transformers`. It is split so P1 and P2 can read only their parts or work in parallel.

---

## Environment snapshot (verified on this machine)

| Item | Value | Notes |
|---|---|---|
| OS | Windows 10 (MSYS/git-bash shell) | Use POSIX syntax; `python` not `python3` |
| Python | 3.11.15 | `python` resolves; `python3` hits a Store stub |
| RDKit | 2026.03.6 | ✅ pre-installed for cheminformatics |
| PyTorch | 2.14.0+cpu | ✅ CPU-only (CUDA not available) |
| CPU | 12 cores | Sufficient for CPU training of small GNNs |
| Disk free | ~93 GB | Plenty for datasets + model caches |
| `chemprop` | installable via pip (2.3.1) | P2 will install |
| `transformers` | installable via pip (5.17.0) | P2 will install |
| `tdc` (PyPI) | ⚠️ **squatted** — 5 KB stub v0.1 | See Phase 2 for the real install path |

---

## Phase 1 — Drug / Scientific Research

**Owner: P1** (P2 assists with tooling).

### What we need to learn
- **What the BBB is and why it matters.** The BBB is a selective endothelial barrier separating circulating blood from brain extracellular fluid. Most drugs (including ~98 % of small-molecule drugs) cannot cross it, which is the central challenge for any CNS drug. Source: the TDC dataset description itself plus a review.
- **The molecular basis of BBB penetration.** There is no single rule (like Lipinski's "Rule of 5" for oral absorption). Key physicochemical drivers: small molecular weight, low H-bond donor count, moderate lipophilicity (often measured as logP or logD), low polar surface area (PSA). Molecules that look "drug-like" by oral-absorption standards often still fail at the BBB.
- **Why this is a hard prediction problem.** BBB penetration depends on subtle structure–property relationships (e.g., a single H-bond donor can flip a molecule from "permeable" to "non-permeable"). The dataset is small (1,975 compounds), so over-fitting to molecular scaffolds is a real risk.
- **The dataset's pedigree.** `BBB_Martins` comes from Martins et al., *J. Chem. Inf. Model.* 2012 (Bayesian modeling of BBB penetration) and is included in MoleculeNet (Wu et al., *Chem. Sci.* 2018). It is curated, deduplicated, and uses a binary label (1 = crosses BBB, 0 = does not).

### What we need to build
- A short **literature primer document** (`docs/literature_review.md`) with 4–6 sourced references covering: (a) BBB biology, (b) BBB as a drug-discovery bottleneck, (c) classical logP/PSA rules, (d) prior ML on `BBB_Martins`, (e) the original Martins Bayesian paper, (f) MoleculeNet benchmark.
- A **glossary** (`docs/glossary.md`) of cheminformatics terms P2 will use: SMILES, fingerprint, scaffold, logP, PSA, HBD, MW, scaffold split.

### Why it matters
Without the scientific framing, the models are just number-crunching. The interpretation phase (Phase 6) will only be meaningful if Person 1 keeps the biology in view. Also, this is the phase that establishes credibility for a portfolio piece — reviewers want to see domain knowledge, not just code.

### Expected output
- ✅ `docs/literature_review.md` with inline citations and a sources list
- ✅ `docs/glossary.md`
- ✅ A one-page **"project framing" note** (P1 → P2 handoff) summarizing the scientific question in plain language.

---

## Phase 2 — Dataset Understanding

**Owner: P1** (drives), **P2** (executes the fetch + verification).

### What we need to learn
The canonical dataset facts (verified from `https://tdcommons.ai/single_pred_tasks/adme/`):

| Property | Detail |
|---|---|
| Name | `BBB_Martins` |
| Source | TDC / MoleculeNet |
| Task | Binary classification |
| Input | Drug SMILES string |
| Label | `0` (does not cross BBB) or `1` (crosses BBB) |
| Raw columns | `drug_id`, `drug` (SMILES), `target` (0/1) |
| Size | **1,975 drugs** |
| Split type | Random Split + Scaffold Split |
| License | CC BY 4.0 |
| Reference | Martins et al., JCIM 2012; Wu et al., Chem. Sci. 2018 |

- **Why there are two split types.** A *random split* shuffles molecules and divides them. A *scaffold split* groups molecules by shared molecular scaffold, so entire substructures appear in only train or test — this tests whether a model learns generalizable chemistry rather than memorizing specific rings. Scaffold split is the harder, more honest benchmark.
- **Class balance.** The label distribution is roughly balanced (roughly half permeable / half non-permeable). We will confirm the exact ratio.

### What we need to build
- **Fetch the raw dataset.** The `tdc` package on PyPI is a 5 KB **squatted stub** (v0.1, wrong dependencies). The real library must be installed from source or conda. Two safe fall-backs that work on this machine:
  1. `pip install tdc` will fail (stub). Avoid it.
  2. Install from the conda-forge channel if available, OR
  3. **Fetch the raw CSV directly.** The canonical TDC data file can be obtained by cloning the real repo or downloading via the TDC website. We will verify by fetching `https://tdcommons.ai` and downloading the `BBB_Martins.csv` (1,975 rows × 3 columns: `drug_id`, `drug`, `target`).
  4. **Verified mirror:** the `lanternpharma/tdc-bbb-martins` GitHub repo contains the dataset and a full reproduction with RDKit descriptors. We will use its TSV as a cross-check, but the primary source is the TDC CSV.
- A **dataset verification script** (`src/02_data/verify_dataset.py`) that: loads the CSV, prints row count, label balance, SMILES-validity rate, duplicate detection, and basic descriptor stats (MW, logP via RDKit) grouped by label.
- A **`data/README.md`** documenting the source, license, column meanings, and the split rationale.

### Why it matters
Garbage-in, garbage-out applies especially here. A 1,975-row dataset is small, so any leak (duplicate scaffold across train/test, mislabeled SMILES) will inflate scores misleadingly. Verifying the raw file and choosing a scaffold split is the single biggest integrity gate.

### Expected output
- ✅ A verified `data/bbb_martins.csv` (1,975 rows, 3 columns)
- ✅ `src/02_data/verify_dataset.py` with printed stats
- ✅ `data/README.md` documenting provenance and license (CC BY 4.0)
- ✅ A **data profile summary** (`docs/data_profile.md`) with label balance, MW/logP/PSA comparison between classes, and a few example SMILES

---

## Phase 3 — Machine Learning (Classical Baseline)

**Owner: P2** (leads), **P1** (reviews features for chemical sense).

### What we need to learn
- **Molecular representations (featurization).** A SMILES string is text — machine learning models can't read it directly. We convert each molecule into a numeric **feature vector**. The three work-horse types:
  - **Morgan / ECFP fingerprints** (circular fingerprints): a binary vector of ~1,024–4,096 bits where each bit encodes the presence of a local atomic neighborhood pattern. Fast, standard, and captures substructural similarity.
  - **Physicochemical descriptors:** ~200 numbers computed by RDKit (molecular weight, logP, TPSA, H-bond donors/acceptors, ring counts, etc.).
  - **Autocorrelations / WHIM descriptors:** physics-inspired numbers encoding 3-D shape and charge distribution.
  The classic approach (used by Lantern Pharma) combines Morgan + descriptors + autocorrelations into a ~3,000–5,000-dimension vector.
- **Why a classical baseline first.** Classical models (Random Forest, SVM, Logistic Regression) with hand-crafted descriptors are fast to train, interpretable (we can see which features matter), and establish a floor that the deep-learning models must beat. If a GNN can't beat a Random Forest on 1,975 compounds, the deep learning isn't earning its complexity.
- **Class-imbalance handling.** If the classes are imbalanced, we will need `class_weight` (sklearn) or SMOTE augmentation.

### What we need to build
- `src/03_ml/featurize.py` — generates three feature sets with RDKit:
  - `morgan_fp` (ECFP4, radius 2, 2048 bits)
  - `rdkit_descriptors` (the full ~200 descriptor set)
  - `combined` (fingerprints + descriptors, the ~3,000-dim stack)
- `src/03_ml/train_baseline.py` — trains three models with scikit-learn:
  - **Logistic Regression** (linear, interpretable baseline)
  - **Random Forest** (ensemble of decision trees, handles non-linear feature interactions)
  - **SVM (RBF kernel)** (margin-based, strong on small datasets)
  Each uses scaffold split, 5-fold cross-validation, and reports AUROC + AUPRC.
- A **hyper-parameter search** using `GridSearchCV` or `RandomizedSearchCV` for each model.
- `src/03_ml/analyze_features.py` — uses the trained Random Forest's feature importances to identify which molecular features correlate most with BBB penetration (e.g., "high logP and low PSA → crosses BBB"). This is where P1's domain knowledge validates whether the model learned sensible chemistry.

### Why it matters
This establishes the baseline that every subsequent phase is benchmarked against. The feature-importance analysis is publishable material and directly feeds Phase 6 (scientific interpretation). It also teaches both of you the data format so the deep-learning code won't be mysterious.

### Expected output
- ✅ Three trained, serialized baseline models (`.joblib`/`.pkl`)
- ✅ A results table (`results/03_ml_baseline_results.xlsx`) with AUROC, AUPRC, accuracy, sensitivity, specificity for each model
- ✅ A feature-importance ranking (`results/03_feature_importance.csv`) cross-checked by P1 against known BBB rules
- ✅ `docs/03_ml_baseline_report.md` documenting methodology, the best baseline, and the comparison to the Lantern Pharma reference (AUROC ~0.90)

---

## Phase 4 — Deep Learning

> **Hardware caveat:** this machine has **no GPU** and `torch` is CPU-only. All deep learning will run on 12 CPU cores. Training is feasible for small models and datasets but requires patience (a GCN on 1,975 compounds trains in minutes to hours; ChemBERTa requires downloading a ~440 MB model and inference dominates).

**Owner: P2** (leads implementation), **P1** (reviews biological plausibility of learned patterns).

### What we need to learn
- **Graph Neural Networks (GNNs).** Instead of fingerprints, a GNN treats the molecule as a graph: atoms = nodes, bonds = edges. The network learns which substructures matter. **Chemprop** (by Ajay et al., Penn) is a battle-tested PyTorch implementation that produces GNN message-passing on molecular graphs. It is pip-installable (`chemprop==2.x`). This is the natural next step after a fingerprint baseline.
- **Self-supervised pre-training + fine-tuning (ChemBERTa).** ChemBERTa (by IBM/MoleculeNet) is a RoBERTa model pre-trained on 10M+ SMILES strings. We **fine-tune** its final classification head on the 1,975 BBB labels. The power: it has already learned chemical grammar, so it should generalize better on a tiny dataset. Install: `pip install transformers datasets` then load `seyonmez/ChemBERTa-zinc-base` or `DeepChem/ChemBERTa`.
- **Boltz-2 (advanced extension — optional, see notes).** Boltz-2 is a diffusion-based model for predicting 3-D molecular structure and binding. It is **not a BBB classifier out of the box**, so "Boltz-2 as an advanced extension" requires creative framing (see Phase 4c).

### What we need to build

#### Phase 4a — GNN with Chemprop (primary target)
- `src/04_dl/chemprop_bbb.py` — uses the `chemprop` CLI or Python API:
  - Reads SMILES + label from CSV
  - Performs scaffold split (matching the ML phase)
  - Trains a 3-layer GNN message-passing network with attention
  - Runs 5-fold cross-validation
- `src/04_dl/evaluate.py` — reports AUROC, AUPRC, accuracy, sensitivity, specificity, and a confusion matrix per fold.

#### Phase 4b — ChemBERTa fine-tuning (primary target)
- `src/04_dl/chemberta_bbb.py` — uses HuggingFace `transformers`:
  - Tokenizerizes SMILES with the ChemBERTa tokenizer (BPE on SMILES characters)
  - Loads the pre-trained RoBERTa checkpoint (`seyonmez/ChemBERTa-zinc-base`)
  - Fine-tunes the `[CLS]` pooled output on the binary BBB task
  - Uses a small learning rate (2e-5), batch size 8–16 (CPU-bound), 3–5 epochs
- Same 5-fold scaffold CV + same metrics.

#### Phase 4c — Boltz-2 as an advanced extension (optional / exploratory)
- **Why it's tricky:** Boltz-2 predicts 3-D structure of protein–ligand complexes, not BBB penetration. To incorporate it we would:
  1. Use Boltz-2 to generate or refine 3-D conformers of our molecules (or predict brain-penetrant transporter structures).
  2. Derive **3-D-aware descriptors** (e.g., from Boltz-2-refined conformations) that feed back into a downstream classifier.
- **Why this is Phase 4c:** Boltz-2 is large (multi-GB model), very compute-heavy (diffusion sampling), and requires GPU memory. On a CPU-only machine it is **not tractable** as a core deliverable. We list it as an optional capstone *if* GPU access becomes available (e.g., Google Colab / a lab server). If attempted locally, budget the download + one inference trial to validate the API call, not a full training run.

### Why it matters
Classical ML with fingerprints is a good baseline, but the deep-learning phase demonstrates the modern standard (GNN + pre-trained chemistry language model). ChemBERTa, in particular, shows transfer learning — a hot topic. Beating or matching the ~0.90 AUROC of the Lantern Pharma ensemble with a GNN would be a strong result.

### Expected output
- ✅ Trained Chemprop GNN model (per CV fold) + predictions
- ✅ Fine-tuned ChemBERTa model (or a clear log of why CPU made it infeasible, with code committed)
- ✅ `results/04_dl_results.xlsx` comparing all deep models against the Phase 3 baseline
- ✅ `docs/04_dl_report.md` documenting architecture choices, hyper-parameters, and training curves
- ✅ A clear decision note on whether Boltz-2 is feasible on available hardware

---

## Phase 5 — Evaluation & Comparison

**Owner: P2** (leads computation), **P1** (leads interpretation of what metrics mean).

### What we need to learn
- **Metrics for binary classification on imbalanced, small datasets.**
  - **AUROC** (Area Under the ROC Curve): probability that the model ranks a random positive above a random negative. Standard but can be misleading on imbalanced data.
  - **AUPRC** (Area Under the Precision-Recall Curve): better than AUROC when positives are rare.
  - **Accuracy, Sensitivity, Specificity**: classic but threshold-dependent — we pick the 0.5 threshold (or a balanced threshold from the ROC curve).
  - **Scaffold-split generalization gap**: compare random-split AUROC vs scaffold-split AUROC. A large gap signals over-fitting to specific scaffolds.
- **Statistical comparison.** With 5-fold CV and 1,975 compounds, each test fold has ~395 samples. The difference in AUROC between two models is "real" only if it survives a paired t-test or a McNemar's test on the same test folds. We will report mean ± std and a paired t-test.
- **The comparison baseline.** Lantern Pharma reported AUROC 0.915 (ensemble) / 0.912 (DNN) / 0.907 (RF) on a scaffold split. Our models should be benchmarked against this.

### What we need to build
- `src/05_eval/compare_models.py` — a single comparison script that:
  - Loads all model predictions (ML baseline + GNN + ChemBERTa) from Phase 3/4
  - Computes all metrics per fold and per split type
  - Runs paired t-tests between best baseline and best deep model
  - Produces plots: ROC curves (all models), precision-recall curves, box-plots of AUROC across folds
- `src/05_eval/ablation.py` — (optional) tests the effect of removing feature types (e.g., descriptors-only vs fingerprints-only vs combined) to see what drives performance.
- A **leaderboard table** saved to `results/05_model_leaderboard.xlsx`.

### Why it matters
This is where we answer the core research question: "do the more complex approaches actually help?" Without a rigorous evaluation, the project is just a collection of models. The ablation (if done) also teaches which molecular information matters most.

### Expected output
- ✅ `results/05_model_leaderboard.xlsx` (all models, all metrics, statistical test p-values)
- ✅ `results/05_roc_curves.png` and `results/05_pr_curves.png` (visual comparison)
- ✅ `docs/05_evaluation_report.md` with the headline result: best model, its AUROC vs the baseline and vs the published reference, and the verdict on whether deep learning helped.

---

## Phase 6 — Scientific Interpretation

**Owner: P1** (leads), **P2** (provides model internals).

### What we need to learn
- **How to read a machine-learning model in chemistry.** A high AUROC is satisfying, but the scientific value is in asking: *what did the model learn?* Did it rediscover the known rules (high logP, low PSA, low MW)? Or did it find a surprising pattern (a specific substructure that enhances BBB crossing)?
- **Model-explanation tools:**
  - For the Random Forest / Logistic Regression baseline: **feature importances** and **SHAP values** (P2 installs `shap`, explains the top features).
  - For the GNN: **attention weights** or **substructure attribution** (which atoms/bonds the GNN "attend to" when predicting permeable).
  - For ChemBERTa: **attention-head visualization** or **token attribution** (which SMILES tokens contributed most).
- **Error analysis.** The cases where the model is *wrong* are often the most informative. Are errors concentrated in certain scaffolds? Certain molecular weights? Compounds with ambiguous SMILES?

### What we need to build
- `src/06_interpret/feature_analysis.py` — loads the Phase 3 feature-importance CSV, groups features by chemical category (lipophilicity, polarity, size, H-bonding), and writes a plain-language summary.
- `src/06_interpret/error_analysis.py` — takes the best model's per-sample predictions, flags the top 10 "most wrong" predictions, and prints the SMILES + known labels + key RDKit descriptors for manual inspection.
- `docs/06_interpretation.md` — P1 writes this, translating model outputs into chemical insights:
  - "The model confirms that logP > 2 and PSA < 70 Å² are strong predictors of BBB penetration, consistent with literature."
  - "The model over-predicts permeability for sulfonamides, a known P-gp substrate class — a reminder that passive-permeability models ignore active efflux."

### Why it matters
This is the phase that makes the project a *research* project rather than a tutorial. Domain insight is what Person 1 contributes uniquely, and it is what makes a portfolio piece stand out to a hiring manager in medical writing or pharmacovigilance. It also surfaces the model's limitations — which is itself a publishable finding.

### Expected output
- ✅ `results/06_feature_analysis.csv`
- ✅ `results/06_error_examples.csv` (10 most-surprising predictions, with SMILES + descriptors)
- ✅ `docs/06_interpretation.md` with chemical insights and error-analysis narrative
- ✅ A **draft results section** ready to drop into a short report (P1 writing)

---

## Phase 7 — Application Development

**Owner: P2** (leads full-stack), **P1** (designs UI/UX and writes copy).

### What we need to learn
- **The simplest path to a deployable app on Windows without paid services.** Options:
  - **Streamlit** (`pip install streamlit`) — a Python script that becomes a web UI. Runs locally with `streamlit run app.py`. Zero frontend experience needed.
  - **Gradio** — similar, optimized for ML model demos.
  - Both can be hosted free on HuggingFace Spaces if a public demo is ever wanted.
- **How to serve a trained model.** We load the best model's weights (e.g., `chemprop` or the RF baseline), expose a single `predict(smiles_string) -> probability` function, and wire it to a text input.

### What we need to build
- `app/app.py` — a Streamlit app:
  - A text box: "Paste a SMILES string"
  - A "Predict" button
  - Output: "Probability of BBB penetration: 0.82" + a colored label (Permeable / Non-permeable) + a 2-D molecule drawing (RDKit `MolDraw2D`)
  - A short explanation: "High logP / low PSA compounds are predicted to cross the BBB."
- `app/model_loader.py` — loads the serialized best model and exposes `predict_bbb(smiles) -> float`.
- `app/requirements.txt` — minimal dependencies (streamlit, rdkit, torch, chemprop).

### Why it matters
A model you can't touch is just an experiment. A working app is portfolio gold — it proves the whole pipeline and gives Person 1 a scientific tool and Person 2 a software-engineering artifact. It also forces robustness (what if a user submits an invalid SMILES?).

### Expected output
- ✅ `app/app.py` — working Streamlit app, testable locally via `streamlit run app/app.py`
- ✅ `app/model_loader.py` — model-loading abstraction
- ✅ `app/requirements.txt`
- ✅ `docs/07_app_user_guide.md` — how to run it, with a screenshot

---

## Phase 8 — Documentation & Delivery

**Co-owned.** P1 writes the scientific narrative; P2 writes the technical narrative.

### What we need to learn
- **README-first structure.** A portfolio project's README is its shop window. It should answer, in order: what is this, why does it matter, what did we build, what did we find, how do I run it.
- **Reproducibility hygiene.** A reviewer should be able to clone and re-run. That means pinned requirements, a clear directory tree, and a single command (or script) that reproduces the headline number.

### What we need to build
- `README.md` — project overview, architecture diagram (P2: draw.io / `docs/architecture.png`), headline result, quick-start instructions, contributor roles (P1 / P2 split).
- `PROJECT_PLAN.md` — this document.
- `requirements.txt` and `environment.yml` (conda alternative) pinned to the versions tested.
- `docs/final_report.pdf` or `.docx` — a 3–4 page mini-paper structure (Introduction / Methods / Results / Discussion / Conclusion) that Person 1 can adapt for a blog post or a scientific-writing portfolio sample.
- `LICENSE` — MIT, since the data is CC BY 4.0 and our code is original.

### Why it matters
The documentation is the deliverable that proves the work is real and reproducible. For a portfolio, it's the artifact a recruiter reads first. For scientific writing (P1's interest), the mini-paper structure is directly transferable to a blog or a journal article.

### Expected output
- ✅ `README.md` (with headline result + quick-start)
- ✅ `PROJECT_PLAN.md` (this file)
- ✅ `requirements.txt` + `environment.yml`
- ✅ `docs/final_report.md` (mini-paper, ready to render to PDF/docx)
- ✅ `docs/architecture.png` (pipeline diagram)
- ✅ A **final summary message** to the user with the headline number and the link to the repo

---

## Cross-cutting conventions

- **Split strategy:** scaffold split (not random) for all phases, to get an honest generalization estimate. We will confirm this matches TDC's recommended split.
- **Version control:** `git init` at project root; commit at the end of each phase.
- **Reproducibility:** every script takes a `--seed` argument; a top-level `run_all.sh` (or `.bat`) reproduces the headline result.
- **No fabrication:** every number reported in a deliverable must come from an executed computation. If a model fails to train, we document the failure honestly.
- **Style (if adapting into a paper):** no em dashes; high sentence-length burstiness; banned vocabulary avoided (delve, tapestry, critical, landscape, robust, leverage).

---

## Role matrix

| Phase | Lead | Supporting | Notes |
|---|---|---|---|
| 1. Drug Research | P1 | P2 (tooling) | P2 fetches sources |
| 2. Dataset | P1 (domain) + P2 (fetch/verify) | — | P2 handles the squatted-`tdc` workaround |
| 3. Classical ML | P2 | P1 (features) | P1 interprets feature importance |
| 4. Deep Learning | P2 | P1 (biology sense-check) | P1 flags implausible learned patterns |
| 5. Evaluation | P2 (computation) | P1 (metrics meaning) | — |
| 6. Interpretation | P1 | P2 (model internals) | P1 writes the narrative |
| 7. Application | P2 | P1 (UX, copy) | — |
| 8. Documentation | Both | — | P1 narrative, P2 technical |

---

## Next step

This plan is the deliverable for the current turn. We hand it to the user for approval, and we **do not begin coding Phase 2 until the user confirms the split strategy (scaffold vs. random) and the go/no-go on the Boltz-2 extension**.
