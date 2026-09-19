# Figure Captions

## Figure 1: `figures/project_workflow.png`

Project workflow for the BBB permeability prediction pipeline. The pipeline proceeds from dataset acquisition (1,975 compounds from TDC BBB_Martins) through feature engineering (Morgan fingerprints + RDKit descriptors), Phase 3 classical ML baseline (5-fold scaffold cross-validation), Phase 4 Chemprop GNN ensemble (SCAFFOLD_BALANCED protocol), Phase 5 matched head-to-head comparison on Fold 0 (296 test molecules), and Phase 6 chemical error analysis with molecular descriptors.

## Figure 2: `figures/model_performance.png`

**Left panel:** Phase 3 5-fold scaffold cross-validation results for Random Forest across three feature sets (Morgan, RDKit descriptors, Combined). The best classical model is RF on Combined features (AUROC 0.921 ± 0.024), marked with an orange dashed reference line. **Right panel:** Model performance across all phases. Phase 3 reports 5-fold mean AUROC for RF; Phase 4 reports the 5-model ensemble mean for Chemprop GNN; Phase 5 reports the matched Fold 0 single-split comparison. The Phase 4 result used a different split protocol (SCAFFOLD_BALANCED) and is not directly comparable to Phase 3.

## Figure 3: `figures/h2h_roc_curve.png`

Receiver operating characteristic (ROC) curves for the matched head-to-head comparison on Phase 3 Fold 0 (296 test molecules, 60 negative / 236 positive). RF (blue) achieves AUROC = 0.9031; GNN (Chemprop, magenta) achieves AUROC = 0.8669. The diagonal dashed line represents random performance (AUROC = 0.5). Both models use identical training and test splits.

## Figure 4: `figures/h2h_precision_recall_curve.png`

Precision-recall curves for the matched head-to-head comparison on Phase 3 Fold 0 (n=296). RF achieves AUPRC = 0.9726; GNN achieves AUPRC = 0.9602. The horizontal dashed line represents the positive-class baseline (AUPRC = 0.7895), which is the prevalence of permeable compounds in the test set. PR curves are particularly informative under class imbalance.

## Figure 5: `figures/error_overlap.png`

Confusion matrices for the matched head-to-head comparison on Phase 3 Fold 0 (n=296). Left: RF confusion matrix with TN=32, FP=28, FN=9, TP=227 (Sensitivity=0.962, Specificity=0.533). Right: GNN confusion matrix with TN=38, FP=22, FN=20, TP=216 (Sensitivity=0.915, Specificity=0.633). RF has higher sensitivity but lower specificity; GNN trades sensitivity for specificity.

## Figure 6: `figures/error_descriptor_analysis.png`

**Top left:** TPSA (topological polar surface area) by error group — both models' false negatives have significantly higher TPSA than correct predictions, indicating both struggle with polar compounds. RF false negatives show the highest TPSA (119.9 vs 62.8 baseline). **Top right:** Molecular weight by error group — RF false negatives are significantly larger molecules (428.3 vs 335.4 Da). **Bottom left:** QED (drug-likeness) by error group — RF false positives have significantly lower QED (0.55 vs 0.66). **Bottom right:** Error counts by model — RF makes 37 total errors (28 FP, 9 FN); GNN makes 42 total errors (22 FP, 20 FN). Of these, 27 errors are shared; 25 disagreements exist (RF correct in 15, GNN correct in 10).
