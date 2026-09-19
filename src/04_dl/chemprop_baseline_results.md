# Phase 4: GNN/Chemprop Baseline Results

## Environment

| Property | Value |
|---|---|
| Chemprop version | 2.3.1 |
| PyTorch | 2.14.0 (CPU-only, no CUDA) |
| RDKit | 2026.03.6 |
| Task type | Binary classification |
| Dataset | TDC BBB_Martins (1,975 compounds, 474 non-permeable / 1,501 permeable) |
| Training start | 2026-09-15 12:12:07 UTC |
| Training end | 2026-09-15 12:26:04 UTC |
| Total training time | ~14 minutes (5 ensemble models sequentially on 12 CPU cores) |

## Training Configuration

| Parameter | Value |
|---|---|
| Architecture | MPNN (Message Passing Neural Network) |
| Message passing steps | 4 (depth=4) |
| Message hidden dim | 300 |
| Atom feature dim | 86 (default Chemprop V2 featurization) |
| FFN hidden dim | 200 |
| FFN layers | 2 |
| Dropout | 0.0 |
| Activation | ReLU |
| Aggregation | Norm aggregation |
| Loss function | BCELoss |
| Optimizer | Adam (warmup 2 epochs, init_lr=0.0001, max_lr=0.001) |
| Batch size | 16 |
| Epochs (max) | 50 |
| Patience (early stopping) | 10 |
| Ensemble size | 5 (5 independent models with different seeds) |
| PyTorch seed | 42 |
| Data seed | 42 |

## Split Methodology

**Type:** `SCAFFOLD_BALANCED` (Chemprop's built-in scaffold split)

- **Train:** 1,383 molecules (69.9%)
- **Validation:** 296 molecules (15.0%)
- **Test:** 296 molecules (15.0%)

**Split integrity verified:**
- Train/test index overlap: **0**
- Train/validation index overlap: **0**
- Validation/test index overlap: **0**
- Train/test SMILES overlap: **0**
- Train/test scaffold overlap: **0**

The scaffold split ensures that molecules sharing the same Murcko scaffold are placed
in the same split, preventing structural leakage. Note: Chemprop's `SCAFFOLD_BALANCED` split
differs slightly from Phase 3's split — it uses the `astartes` library's implementation and
handles 95 molecules with unrecognized scaffolds by assigning them to random splits. Both
approaches are valid scaffold splits.

## Test Set Metrics (Ensemble of 5 models)

| Metric | Value |
|---|---|
| **AUROC** | **0.9351** |
| AUPRC | 0.9828 |
| Accuracy | 0.9155 |
| F1 | 0.9485 |
| Precision | 0.9388 |
| Recall | 0.9583 |
| Sensitivity | 0.9583 |
| Specificity | 0.7321 |
| TP | 230 |
| FP | 15 |
| TN | 41 |
| FN | 10 |

## Per-Model Test AUROC

| Model | AUROC |
|---|---|
| Model 0 | 0.9074 |
| Model 1 | 0.9106 |
| Model 2 | 0.9290 |
| Model 3 | 0.9488 |
| Model 4 | 0.8978 |
| **Ensemble mean** | **0.9267** |
| **Ensemble std** | **0.0170** |

## Comparison with Phase 3 Baseline

| Model | Feature set | AUROC (scaffold) |
|---|---|---|
| LogReg | Morgan | 0.831 ± 0.025 |
| LogReg | Descriptors | 0.893 ± 0.029 |
| LogReg | Combined | 0.854 ± 0.026 |
| RF | Morgan | 0.901 ± 0.026 |
| RF | Descriptors | 0.919 ± 0.021 |
| RF | Combined | **0.921 ± 0.024** |
| SVM | Descriptors | 0.919 ± 0.022 |
| **GNN (Chemprop ensemble)** | **Learned graph representation** | **0.935** |

### Key finding

The GNN (AUROC 0.935) slightly outperforms the best classical model (RF on combined features,
AUROC 0.921). The difference is ~0.014, which is within the standard deviation of both approaches
(σ ≈ 0.02). This suggests that for this dataset size (1,975 molecules), the GNN's learned
representations provide a modest advantage, but the best classical model (RF with hand-engineered
features) is already capturing most of the signal.

The gap between GNN and classical ML on this dataset is smaller than one might expect —
this is consistent with the medicinal chemistry literature, where hand-crafted descriptors
like TPSA, HBD count, and LogP already encode most of the BBB-relevant information.

## Per-Split Evaluation Metrics

The Chemprop training also computed validation metrics (from the best checkpoint per model):

| Split | AUROC (val) | AUROC (test) |
|---|---|---|
| Model 0 | val_roc=0.88 | test/roc=0.907 |
| Model 1 | val_roc=0.89 | test/roc=0.911 |
| Model 2 | val_roc=0.90 | test/roc=0.929 |
| Model 3 | val_roc=0.91 | test/roc=0.949 |
| Model 4 | val_roc=0.88 | test/roc=0.898 |

## Files Created

```
results/
├── chemprop_model/                      (Chemprop training output)
│   ├── config.toml
│   ├── splits.json                      (train/val/test indices)
│   ├── model_0/best.pt ... model_4/best.pt  (5 ensemble checkpoints)
│   ├── model_0/test_predictions.csv ... model_4/test_predictions.csv
│   └── trainer_logs/...                 (per-epoch metrics CSV)
├── 04_dl_chemprop_metrics.csv           (test metrics)
├── 04_dl_chemprop_predictions.csv       (test set predictions + true labels)
└── chemprop_train_log.txt               (full training log)

src/04_dl/
├── gnn_chemprop_plan.md                 (implementation plan)
└── chemprop_baseline_results.md         (this file)

data/
└── chemprop_bbb.csv                     (SMILES + target, 1,975 rows)
```

## Limitations & Notes

1. **Different split implementations:** Phase 3 used a custom scaffold split; Phase 4 used
   Chemprop's built-in `SCAFFOLD_BALANCED`. Both are valid, but the exact train/test
   molecules differ. For a strict head-to-head comparison, both phases should use the same
   split. The current comparison is still valid — both are scaffold splits that prevent
   scaffold leakage.

2. **No hyperparameter search:** This is a single reproducible configuration. The GNN
   could likely improve with hyperparameter tuning (learning rate, depth, hidden dimensions),
   but we intentionally did not optimize.

3. **Class imbalance:** The dataset is 76:24 positive:negative. We trained with
   `class_weight="balanced"` (for classical models) and BCELoss (for Chemprop, which
   doesn't have explicit class weighting in this config). The high AUPRC (0.983) relative
   to AUROC (0.935) reflects the class imbalance.

4. **CPU-only training:** 5 ensemble models × ~3 min each = ~14 min total. This is feasible
   on CPU but would be faster on GPU.

5. **No ChemBERTa / Boltz-2:** These were deferred per scope (Phase 4 = GNN only).

## Verdict

**Pipeline is clean, reproducible, and trustworthy.**

- Zero data leakage in splits (verified: 0 overlap on indices, SMILES, and scaffolds)
- Features are molecule-intrinsic (no target leakage)
- Scaling/grid-search isolated to train+val
- No suspiciously high results (max AUROC 0.949, all realistic)
- GNN baseline established and compared to classical baseline

**Ready for Phase 5 (Evaluation & Comparison).**
