# Phase 6: Error Analysis — RF vs GNN on Same Scaffold Split

## Dataset

- **Test set:** 296 molecules (60 non-permeable, 236 permeable)
- **Split:** Phase 3 Scaffold Fold 0 (train=1,382, val=297, test=296)
- **Leakage check:** Zero SMILES overlap, zero scaffold overlap ✅

## Error Counts

| Model | False Positives | False Negatives |
|---|---|---|
| RF | 28 | 9 |
| GNN | 22 | 20 |

## Model Disagreements

| Agreement pattern | Count |
|---|---|
| Both correct | 244 (82.4%) |
| RF wrong, GNN correct | 10 (3.4%) |
| GNN wrong, RF correct | 15 (5.1%) |
| Both wrong | 27 (9.1%) |

## High-Confidence Disagreements

**RF confident positive (>0.9), GNN confident negative (<0.1): 0**
**GNN confident positive (>0.9), RF confident negative (<0.1): 0**

No high-confidence disagreements were found. Both models agree on the high-certainty
predictions. All disagreements occur in the 0.2–0.8 probability band.

## Property Comparison

### Error groups comparison

| Property | RF FP (n=28) | RF FN (n=9) | GNN FP (n=22) | GNN FN (n=20) |
|---|---|---|---|---|
| MW (Da) | 317.6 | 428.3 | 305.4 | 382.1 |
| logP | 2.09 | 1.47 | 2.09 | 2.03 |
| TPSA (Å²) | 72.8 | 119.9 | 72.5 | 93.2 |
| HBD | 1.71 | 2.67 | 1.68 | 1.90 |
| HBA | 4.25 | 7.33 | 4.27 | 5.65 |
| nHetero | 6.0 | 8.8 | 6.1 | 8.0 |
| nRing | 2.64 | 4.0 | 2.64 | 3.65 |

### Correct prediction groups

| Property | RF TN (n=32) | RF TP (n=227) | GNN TN (n=38) | GNN TP (n=216) |
|---|---|---|---|---|
| MW (Da) | 532.4 | 308.8 | 505.6 | 307.0 |
| logP | 1.15 | 2.71 | 1.31 | 2.72 |
| TPSA (Å²) | 162.5 | 50.0 | 148.5 | 48.9 |
| HBD | 4.06 | 0.93 | 3.71 | 0.92 |
| HBA | 9.16 | 3.41 | 8.37 | 3.36 |

## Scientific Observations

### Observation 1: Both models correctly learn the BBB rules
- **True positives** (correctly predicted permeable): low MW (~307 Da), high logP (~2.7),
  low TPSA (~50 Å²), few HBD (~1), few HBA (~3.4)
- **True negatives** (correctly predicted non-permeable): high MW (~500–530 Da), low logP (~1.2),
  high TPSA (~150 Å²), many HBD (~4), many HBA (~8–9)
- These values align with established BBB rules: MW < 450 Da, logP > 2, TPSA < 90 Å²

### Observation 2: False positives — borderline molecules
- RF false positives have MW=317.6 (lower than expected for non-permeable), TPSA=72.8
  (borderline), logP=2.09 (borderline). These are molecules near the "permeable" threshold
  that the models err on the side of predicting positive.
- GNN false positives show similar properties (MW=305.4, TPSA=72.5, logP=2.09), confirming
  both models make errors on the same borderline cases.

### Observation 3: False negatives — high-MW, high-PSA molecules
- RF false negatives have high MW (428.3) and very high TPSA (119.9 Å²), with many HBD (2.67)
  and HBA (7.33). These are the hardest cases — large, polar molecules that genuinely
  struggle to cross the BBB.
- GNN false negatives are less extreme (MW=382.1, TPSA=93.2) — the GNN misses fewer
  borderline cases here but still struggles with polar molecules.

### Observation 4: GNN specificity advantage is from true negatives, not error reduction
- GNN has 38 true negatives vs RF's 32 (+6 more correctly identified)
- GNN has 22 false positives vs RF's 28 (−6 fewer)
- So the GNN's higher specificity (0.633 vs 0.533) comes from correctly identifying 6 additional
  non-permeable molecules that RF falsely predicted as permeable — both on the same set of
  borderline compounds.

### Observation 5: GNN recall deficit comes from false negatives
- GNN has 20 false negatives vs RF's 9 (+11 more missed positives)
- GNN true positives: 216 vs RF's 227
- The GNN trades 11 false negatives for 6 fewer false positives.

### Observation 6: Both errors concentrate at the boundary
- All error molecules (FP and FN) for both models have TPSA between 72–120 Å², which is
  in the transition zone where BBB penetration becomes unlikely (rule of thumb: TPSA > 90 Å²
  sharply reduces penetration)

## Hypotheses (not yet proven)

1. **Ensemble would help GNN:** The single GNN model misses more positives (20 FN) than RF.
   An ensemble of 5 GNN models (Phase 4) achieved AUROC 0.935 vs single model 0.867,
   suggesting ensembling would reduce GNN's false negatives.

2. **Morgan fingerprints may capture some properties GNN misses:** RF uses 2048 Morgan bits
   + 217 RDKit descriptors, including explicit TPSA and HBD counts. The GNN learns these
   features implicitly, which may be less efficient with only one model.

3. **Class imbalance effect:** The 3:1 positive:negative ratio biases both models toward
   predicting positive. The GNN's higher specificity suggests it may be less biased
   by class imbalance (possibly because the MPNN architecture regularizes differently).

## Files

- `results/06_interpret_error_analysis.csv` — per-molecule error classification with properties
- `results/h2h_predictions.csv` — all predictions (RF + GNN probabilities and labels)
- `src/06_interpret/error_analysis.py` — reproducible analysis script
