# Phase 4: GNN/Chemprop Implementation Plan

## What the GNN Will Do

A **Graph Neural Network (GNN)** for BBB penetration prediction treats each molecule
as a graph: atoms are nodes, bonds are edges. It learns to predict whether a molecule
crosses the blood-brain barrier by propagating information along the molecular graph
through multiple rounds of message passing.

Unlike classical ML (where we hand-engineer Morgan fingerprints or RDKit descriptors),
the GNN **learns its own molecular representations** during training. It discovers which
atomic substructures, bond patterns, and spatial relationships correlate with BBB
penetration.

## Why It's Different from Morgan Fingerprints

| Aspect | Morgan Fingerprints (Classical ML) | Graph Neural Network (GNN) |
|---|---|---|
| Representation | Fixed-length binary vector (2,048 bits) | Variable graph structure (atoms + bonds) |
| Information | Local neighborhoods up to radius=2 | Full graph context, learns attention |
| Hand-engineering | We choose the fingerprint params | The model learns what matters |
| Interpretability | Bit → substructure (fixed) | Attention weights → which atoms/bonds mattered |
| Feature type | Pre-computed, molecule-independent | Learned end-to-end, molecule-dependent |

The key difference: Morgan fingerprints are a **static** representation. GNNs produce a
**dynamic** representation that is jointly optimized with the prediction task. This
means the GNN can learn features that are specifically useful for BBB prediction,
whereas Morgan fingerprints are general-purpose.

## What Input It Needs

From the existing `data/bbb_martins.csv`:

1. **SMILES strings** — converted to molecular graphs (atoms = nodes, bonds = edges)
2. **Atom features** — one-hot encoded atom type (C, N, O, S, F, Cl, etc.) + valence + formal charge + aromaticity
3. **Bond features** — bond type (single, double, triple, aromatic) + conjugation + ring membership
4. **Target labels** — 0 (non-permeable) or 1 (permeable)
5. **Split indices** — the same scaffold-split indices already verified in Phase 3
   (`data/features/scaffold_splits.pkl`)

No new data download needed. The SMILES are already in the CSV.

## What Output We Expect

- **Prediction:** For each molecule, a probability ∈ [0, 1] of BBB penetration
- **Metrics:** Same as Phase 3 — AUROC, AUPRC, Accuracy, F1, Sensitivity, Specificity
- **Comparison:** GNN AUROC should be compared against our best baseline (RF+combined:
  AUROC 0.921 ± 0.024 on scaffold split)
- **Expected result:** If the GNN beats 0.921, it demonstrates that learned representations
  outperform hand-engineered features. If it matches or slightly underperforms, the RF
  is already capturing the useful signal.
- **Additional:** Atom-level attention weights for interpretation (Phase 6)

## Simplest Reliable Implementation Plan

### Step 1: Install dependencies

```bash
pip install chemprop  # v2.3.1 (latest stable, works on CPU)
pip install rdkit-pypi torch  # already installed but double-check
```

Chemprop is chosen because:
- Single pip install, no compilation needed
- Battle-tested on TDC datasets (it's the reference GNN implementation)
- CPU-compatible (works on our no-GPU setup)
- Handles SMILES→graph conversion internally

### Step 2: Convert data to Chemprop format

Chemprop expects a TSV/CSV with the SMILES as the first column and targets as
subsequent columns. No need to manually build graphs — Chemprop reads SMILES and
constructs molecular graphs internally.

```python
# data conversion (minimal)
import pandas as pd
df = pd.read_csv("data/bbb_martins.csv")
df[["drug", "target"]].to_csv("data/chemprop_bbb.csv", index=False)
```

### Step 3: Train with scaffold split

```bash
chemprop_train \
  --data_path data/chemprop_bbb.csv \
  --dataset_type classification \
  --save_dir results/chemprop_model \
  --scoring auc \
  --epochs 50 \
  --pytorch_seed 42 \
  --ensemble_size 5 \
  --split_type scaffold \
  --split_sizes 0.7 0.15 0.15
```

This will:
- Use scaffold splitting internally (consistent with our Phase 3 approach)
- Train 5 models (ensemble_size=5) with 5 different random seeds
- Run 50 epochs (early stopping typically kicks in around epoch 15–20)
- Save models + predictions

### Step 4: Evaluate

```bash
chemprop_predict \
  --model_dir results/chemprop_model \
  --test_data data/chemprop_bbb.csv \
  --preds_path results/chemprop_preds.csv \
  --eval_data data/chemprop_bbb.csv
```

Extract: AUROC, AUPRC, Accuracy, F1, Sensitivity, Specificity

### Step 5: Save results

Save GNN results alongside the classical ML results for the Phase 5 comparison.

## Why This Approach

1. **Minimal code:** Chemprop handles all the graph construction, message passing,
   and training internals. We just pass SMILES and labels.
2. **Reproducible:** Single pip install, deterministic with `--pytorch_seed`.
3. **CPU-compatible:** Chemprop runs on CPU (we verified `torch.cuda.is_available() = False`).
4. **Comparable:** Same dataset, same scaffold split convention, same metrics as Phase 3.
5. **Interpretable:** Chemprop can output atom attention weights for Phase 6.

## Risks & Mitigations

| Risk | Mitigation |
|---|---|
| CPU-only training is slow | Use `--epochs 50` with early stopping; 1,975 molecules should finish in <10 min |
| Chemprop API changes | Pin to v2.3.1; it's the stable release |
| Results may not beat RF | That's a valid finding — "RF on handcrafted features matches GNN on learned features" is publishable |
| No SMILES standardization issues | Our Phase 2 verified all 1,975 SMILES are RDKit-clean |

## Expected Timeline

- Step 1 (install): < 2 min
- Step 2 (convert data): < 1 min
- Step 3 (train + ensemble): ~5–10 min on CPU
- Step 4 (evaluate): ~1 min
- Step 5 (save): < 1 min

**Total: ~10–15 minutes** for a complete, comparable GNN baseline.
