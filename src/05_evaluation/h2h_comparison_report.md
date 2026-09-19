# Phase 5: Head-to-Head Validation — RF vs Chemprop GNN

## Split Used

Both models were trained and evaluated on the **exact same scaffold split** (Phase 3 Fold 0):

| Split | Count | % |
|---|---|---|
| Train | 1,382 | 69.9% |
| Validation | 297 | 15.0% |
| **Test** | 296 | 15.0% |

- Train+Val (combined for fitting): 1,679 molecules
- Split derived from `data/features/scaffold_splits.pkl` (fold 0)
- **Train/test SMILES overlap: 0** (verified)
- **Train/test scaffold overlap: 0** (verified)
- Passed as pre-defined split file to Chemprop via `--splits-file`

## Results (Single Model Comparison)

| Metric | RF (Combined features) | GNN (Chemprop) | Difference (GNN - RF) |
|---|---|---|---|
| **AUROC** | **0.9031** | 0.8669 | **-0.0362** |
| AUPRC | 0.9726 | 0.9602 | -0.0124 |
| Accuracy | 0.8750 | 0.8581 | -0.0169 |
| F1 | 0.9246 | 0.9114 | -0.0133 |
| Precision | 0.8902 | 0.9076 | +0.0174 |
| Recall | 0.9619 | 0.9153 | -0.0466 |
| Sensitivity | 0.9619 | 0.9153 | -0.0466 |
| Specificity | 0.5333 | 0.6333 | **+0.1000** |

### Confusion Matrix (Test set, 296 molecules: 60 negative, 236 positive)

| | Predicted Negative | Predicted Positive |
|---|---|---|
| **Actual Negative** | 32 (RF) / 38 (GNN) | 28 (RF) / 22 (GNN) |
| **Actual Positive** | 9 (RF) / 20 (GNN) | 227 (RF) / 216 (GNN) |

## Comparison with Phase 3/4 Aggregate Results

Note: Phase 3's reported RF AUROC of 0.921 was a **5-fold average** across 5 different
scaffold splits. This head-to-head comparison uses a **single split** (fold 0 only),
so the numbers differ from the aggregate.

| Comparison | RF (5-fold scaffold) | GNN (5-fold ensemble) | GNN (single model, same split) |
|---|---|---|---|
| AUROC | 0.921 ± 0.024 | 0.935 | 0.867 |

The single-model GNN (0.867) underperforms the 5-model Phase 4 ensemble (0.935), confirming
that ensembling provides a significant boost. On this matched Fold-0 comparison, RF (0.903)
outperforms the single GNN (0.867) across AUROC, AUPRC, accuracy, F1, and sensitivity, while
the GNN has higher precision and specificity. This is an observation from one scaffold split
(n=296 test) and does not establish general superiority of either architecture or feature type.

## Key Takeaways

1. **Both models use the identical split** — the comparison is fair.
2. **RF outperforms single GNN** on this fold (AUROC 0.903 vs 0.867), particularly on
   recall (0.962 vs 0.915) — the RF is more sensitive to positive cases.
3. **GNN has higher specificity** (0.633 vs 0.533) — it's better at identifying true
   negatives, which is important for BBB prediction (false positives waste drug discovery
   resources).
4. **GNN ensemble (0.935) > RF single model (0.903) on this fold** — ensembling the GNN (Phase 4, 5-fold SCAFFOLD_BALANCED) helps significantly versus the single GNN. Note: the 0.935 Phase 4 ensemble used a DIFFERENT split from this Fold 0 comparison, so this is not a direct head-to-head.
5. The GNN model converged at epoch 24 (early stopping), suggesting it may benefit from
   more data or hyperparameter tuning, but we intentionally did not search.

## Files

- `results/h2h_comparison.csv` — side-by-side metrics
- `results/h2h_predictions.csv` — per-molecule predictions for both models
- `results/chemprop_h2h/` — trained Chemprop model on Phase 3 split
- `results/h2h_rf_predictions.csv` — RF predictions on Phase 3 split
- `data/chemprop_h2h.csv` — Chemprop-formatted data
- `data/chemprop_splits_h2h.json` — Phase 3 split in Chemprop format

## Next Steps

Phase 5 is complete. Both baselines (RF and GNN) are established with clean, leak-free
splits. Ready for Phase 6 (interpretation) or Phase 7 (application) upon request.
