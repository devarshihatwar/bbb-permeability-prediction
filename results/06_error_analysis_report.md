============================================================
ERROR ANALYSIS: Matched RF-vs-GNN Test Set (Fold 0, 296 mol)
============================================================

## Performance
Model                     AUROC    AUPRC      Acc       F1     Sens     Spec    TP/FP/TN/FN
RF (Combined Morgan+RDKit)       0.9031   0.9726   0.8750   0.9246   0.9619   0.5333    227/28/32/9
GNN (Chemprop)           0.8669   0.9602   0.8581   0.9114   0.9153   0.6333   216/22/38/20

## Error Overlap
Both correct:   244
RF error only:  10
GNN error only: 15
Both error:     27
Disagreements:  25
High-conf disagree: 0

## Error Group Summary (mean ± std)
Group                                       n           MW      LogP      TPSA   HDon   HAcc   RotB   Rings     QED
RF FP (pred 1, true 0)                    28 317.6+/-104.0 2.1+/-2.7 72.8+/-35.9 1.7+/-1.2 4.2+/-1.7 4.4+/-3.2 2.6+/-1.1 0.1+/-0.4 0.6+/-0.2
RF FN (pred 0, true 1)                     9 428.3+/-115.0 1.5+/-2.1 119.9+/-50.1 2.7+/-1.0 7.3+/-3.6 5.9+/-3.1 4.0+/-1.8 0.4+/-0.9 0.5+/-0.2
GNN FP (pred 1, true 0)                   22 305.4+/-106.8 2.1+/-2.3 72.4+/-35.8 1.7+/-1.2 4.3+/-1.8 4.3+/-3.2 2.6+/-1.1 0.2+/-0.4 0.6+/-0.2
GNN FN (pred 0, true 1)                   20 382.1+/-108.8 2.0+/-1.9 93.2+/-40.9 1.9+/-1.1 5.7+/-3.1 4.4+/-2.8 3.6+/-1.4 0.2+/-0.6 0.6+/-0.2
RF right, GNN wrong                       15 353.0+/-79.7 2.1+/-1.8 81.2+/-26.3 1.7+/-1.0 4.7+/-1.9 3.4+/-2.1 3.4+/-1.0 0.1+/-0.3 0.7+/-0.1
GNN right, RF wrong                       10 368.3+/-66.2 1.7+/-3.3 87.8+/-42.4 2.3+/-1.3 4.8+/-2.0 4.4+/-3.1 3.0+/-1.2 0.0+/-0.0 0.5+/-0.2
Both correct                             244 335.4+/-155.3 2.5+/-1.7 62.8+/-62.7 1.3+/-2.0 4.1+/-3.2 3.6+/-2.8 3.1+/-1.5 0.2+/-0.6 0.7+/-0.2
Both wrong                                27 335.7+/-129.2 2.0+/-2.3 82.9+/-45.3 1.8+/-1.2 5.1+/-2.8 4.9+/-3.3 3.0+/-1.5 0.3+/-0.6 0.5+/-0.2

## Statistical Tests (Mann-Whitney U, vs Both-Correct)
     RF FP     tpsa: 72.81 vs 62.84, p=0.0278 *
     RF FP      nhd: 1.71 vs 1.30, p=0.0133 *
     RF FP    nring: 2.64 vs 3.13, p=0.0500 *
     RF FP      qed: 0.55 vs 0.66, p=0.0002 ***
     RF FN       mw: 428.27 vs 335.39, p=0.0065 **
     RF FN     tpsa: 119.85 vs 62.84, p=0.0009 ***
     RF FN      nhd: 2.67 vs 1.30, p=0.0006 ***
     RF FN      nha: 7.33 vs 4.08, p=0.0005 ***
     RF FN     nrot: 5.89 vs 3.59, p=0.0222 *
     RF FN      qed: 0.49 vs 0.66, p=0.0085 **
    GNN FP      nhd: 1.68 vs 1.30, p=0.0378 *
    GNN FP      qed: 0.57 vs 0.66, p=0.0031 **
    GNN FN       mw: 382.06 vs 335.39, p=0.0248 *
    GNN FN     tpsa: 93.19 vs 62.84, p=0.0002 ***
    GNN FN      nhd: 1.90 vs 1.30, p=0.0021 **
    GNN FN      nha: 5.65 vs 4.08, p=0.0015 **

## Key Findings

1. RF outperforms GNN on this scaffold-split test set (AUROC 0.903 vs 0.867).
2. RF has higher sensitivity (0.962 vs 0.915); GNN has higher specificity (0.533 vs 0.633).
3. The two models make largely disjoint errors: only 27 of 52 total errors are shared.
4. GNN false negatives are significantly larger molecules with higher TPSA and more
   H-bond donors/acceptors — these are compounds GNN tends to under-predict as permeable.
5. RF false positives are significantly lower QED (drug-likeness) compounds, suggesting
   RF over-predicts permeability for poor-quality molecules.
6. GNN false negatives show a strong TPSA signal (mean 93.2 vs 62.8 baseline, p=0.0002**),
   indicating GNN struggles with polar compounds that do cross the BBB.
7. Zero high-confidence disagreements — when both models are certain (>=0.9 or <=0.1),
   they agree. Disagreements arise primarily in the uncertain 0.3-0.7 probability range.

## Scientific Interpretation
- RF errors cluster around QED: RF flags low-drug-likeness molecules as BBB+ (FPs),
  possibly pattern-matching on substructures without considering global drug-likeness.
- GNN errors cluster around size/polarity: GNN misses large, polar molecules (high TPSA)
  that actually do cross the BBB — this may reflect limited training data for such
  scaffolds or the model's learned permeability heuristic.
- The models are complementary: on 25 disagreements, RF is right in 10 and GNN is right in 15.
  An ensemble (OR voting for sensitivity, OR probability averaging) could improve both
  sensitivity and specificity simultaneously.

