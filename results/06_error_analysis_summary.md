# Phase 5/6: Error Analysis Report

## Matched Test Set (Fold 0, Scaffold Split)
- **Total molecules:** 296
- **True labels:** {1: 236, 0: 60}
- **Class balance:** 79.7% positive (1), 20.3% negative (0)

## Model Performance (Fold 0)

| Model | AUROC | AUPRC | Accuracy | F1 | Sensitivity | Specificity | TP | FP | TN | FN |
|-------|-------|-------|----------|----|-------------|-------------|----|----|----|----|
| RF (Morgan+RDKit) | 0.9031 | 0.9726 | 0.8750 | 0.9246 | 0.9619 | 0.5333 | 227 | 28 | 32 | 9 |
| GNN (Chemprop) | 0.8669 | 0.9602 | 0.8581 | 0.9114 | 0.9153 | 0.6333 | 216 | 22 | 38 | 20 |

## Error Overlap

| Error Type | RF Only Errors | GNN Only Errors | Both Error |
|---|---|---|---|
| False Positives (pred 1, true 0) | 7 | 1 | 21 |
| False Negatives (pred 0, true 1) | 3 | 14 | 6 |

## Descriptor Comparison by Error Group

| Group | n | MW | LogP | TPSA | HDon | HAcc | RotB | Rings | RO5viol | QED |
|-------|---|-----|------|------|------|------|------|-------|---------|-----|
| All test | 296 | 337.4+/-147.7 | 2.4+/-1.9 | 66.5+/-59.8 | 1.4+/-1.9 | 4.2+/-3.1 | 3.7+/-2.8 | 3.1+/-1.5 | 0.186+/-0.579 | 0.652+/-0.185 |
| RF FP (pred 1, true 0) | 28 | 317.6+/-104.0 | 2.1+/-2.7 | 72.8+/-35.9 | 1.7+/-1.2 | 4.2+/-1.7 | 4.4+/-3.2 | 2.6+/-1.1 | 0.143+/-0.356 | 0.551+/-0.159 |
| RF FN (pred 0, true 1) | 9 | 428.3+/-115.0 | 1.5+/-2.1 | 119.9+/-50.1 | 2.7+/-1.0 | 7.3+/-3.6 | 5.9+/-3.1 | 4.0+/-1.8 | 0.444+/-0.882 | 0.488+/-0.213 |
| GNN FP (pred 1, true 0) | 22 | 305.4+/-106.8 | 2.1+/-2.3 | 72.4+/-35.8 | 1.7+/-1.2 | 4.3+/-1.8 | 4.3+/-3.2 | 2.6+/-1.1 | 0.182+/-0.395 | 0.570+/-0.154 |
| GNN FN (pred 0, true 1) | 20 | 382.1+/-108.8 | 2.0+/-1.9 | 93.2+/-40.9 | 1.9+/-1.1 | 5.7+/-3.1 | 4.4+/-2.8 | 3.6+/-1.4 | 0.250+/-0.639 | 0.641+/-0.212 |
| RF right, GNN wrong (disagree) | 15 | 353.0+/-79.7 | 2.1+/-1.8 | 81.2+/-26.3 | 1.7+/-1.0 | 4.7+/-1.9 | 3.4+/-2.1 | 3.4+/-1.0 | 0.067+/-0.258 | 0.729+/-0.133 |
| GNN right, RF wrong (disagree) | 10 | 368.3+/-66.2 | 1.7+/-3.3 | 87.8+/-42.4 | 2.3+/-1.3 | 4.8+/-2.0 | 4.4+/-3.1 | 3.0+/-1.2 | 0.000+/-0.000 | 0.539+/-0.177 |
| Both correct | 244 | 335.4+/-155.3 | 2.5+/-1.7 | 62.8+/-62.7 | 1.3+/-2.0 | 4.1+/-3.2 | 3.6+/-2.8 | 3.1+/-1.5 | 0.189+/-0.599 | 0.665+/-0.183 |
| Both wrong | 27 | 335.7+/-129.2 | 2.0+/-2.3 | 82.9+/-45.3 | 1.8+/-1.2 | 5.1+/-2.8 | 4.9+/-3.3 | 3.0+/-1.5 | 0.296+/-0.609 | 0.534+/-0.174 |
| Both high-conf disagree | 0 | — | — | — | — | — | — | — | — | — |

## Statistical Tests (Mann-Whitney U, vs Both-Correct)

| Group | Descriptor | Group Mean | Baseline Mean | p-value | Sig |
|-------|-----------|------------|---------------|---------|-----|
| RF FP (pred 1, true 0) | tpsa | 72.81 | 62.84 | 0.0278 | * |
| RF FP (pred 1, true 0) | nhd | 1.71 | 1.30 | 0.0133 | * |
| RF FP (pred 1, true 0) | nring | 2.64 | 3.13 | 0.0500 | * |
| RF FP (pred 1, true 0) | qed | 0.55 | 0.66 | 0.0002 | *** |
| RF FN (pred 0, true 1) | mw | 428.27 | 335.39 | 0.0065 | ** |
| RF FN (pred 0, true 1) | tpsa | 119.85 | 62.84 | 0.0009 | *** |
| RF FN (pred 0, true 1) | nhd | 2.67 | 1.30 | 0.0006 | *** |
| RF FN (pred 0, true 1) | nha | 7.33 | 4.08 | 0.0005 | *** |
| RF FN (pred 0, true 1) | nrot | 5.89 | 3.59 | 0.0222 | * |
| RF FN (pred 0, true 1) | qed | 0.49 | 0.66 | 0.0085 | ** |
| GNN FP (pred 1, true 0) | nhd | 1.68 | 1.30 | 0.0378 | * |
| GNN FP (pred 1, true 0) | qed | 0.57 | 0.66 | 0.0031 | ** |
| GNN FN (pred 0, true 1) | mw | 382.06 | 335.39 | 0.0248 | * |
| GNN FN (pred 0, true 1) | tpsa | 93.19 | 62.84 | 0.0002 | *** |
| GNN FN (pred 0, true 1) | nhd | 1.90 | 1.30 | 0.0021 | ** |
| GNN FN (pred 0, true 1) | nha | 5.65 | 4.08 | 0.0015 | ** |

## Key Observations (Observed Results)

1. **RF outperforms GNN on this scaffold-split test set** (AUROC 0.9031 vs 0.8669). This is an observed result for Fold 0, not a general claim about GNN robustness.
2. **RF has higher sensitivity** (0.962 vs 0.915) — RF catches more true BBB+ compounds.
3. **GNN has higher specificity** (0.533 vs 0.633) — GNN makes fewer false positive calls.
4. **Error overlap is partial**: RF-specific FPs and GNN-specific FNs represent
   distinct failure modes that could benefit from model diversity.
5. **High-confidence disagreements** (both models >90% confident but disagree) count: 0 — no molecules had both models highly confident yet disagreeing.

## Hypotheses (Require Further Investigation)

1. **GNN false negatives** may involve compounds with unusual functional groups or
   stereochemistry not well-represented in the training set scaffolds.
2. **RF false positives** may involve molecules with high LogP or low TPSA that
   trigger the model's permeability pattern matching but lack actual BBB penetration.
3. If descriptor differences are statistically significant, each model fails on
   chemically distinct regions of chemical space — suggesting complementary strengths.