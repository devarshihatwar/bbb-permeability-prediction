"""
Generate all project figures from existing saved results.
DO NOT train any models. Read-only from saved CSV/CSV files.

Figures:
1. figures/project_workflow.png
2. figures/model_performance.png
3. figures/h2h_roc_curve.png
4. figures/h2h_precision_recall_curve.png
5. figures/error_overlap.png
6. figures/error_descriptor_analysis.png
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc

RESULTS = "results"
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

# ─── Color palette ───────────────────────────────────────────────────────
COLOR_RF = '#2E86AB'   # Blue
COLOR_GNN = '#A23B72'  # Magenta
COLOR_PHASE3 = '#F18F01'  # Orange
COLOR_PHASE4 = '#C73E1D'  # Red
COLOR_ACCENT = '#3B1F58'  # Purple

plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

# ─── Load data ───────────────────────────────────────────────────────────
df = pd.read_csv(f"{RESULTS}/03_ml_baseline_results.csv")
h2h = pd.read_csv(f"{RESULTS}/h2h_predictions.csv")
ea = pd.read_csv(f"{RESULTS}/06_error_analysis.csv")
h2h_comp = pd.read_csv(f"{RESULTS}/h2h_comparison.csv")

print(f"Loaded: Phase 3 ({len(df)} rows), H2H ({len(h2h)} rows), EA ({len(ea)} rows)")

# ─── Figure 1: Project Workflow ──────────────────────────────────────────
print("Generating Figure 1: project_workflow.png")
fig, ax = plt.subplots(figsize=(10, 3))
ax.axis('off')

steps = [
    ("1. Dataset\n1,975 compounds", (0.05, 0.5)),
    ("2. Feature Engineering\nMorgan + RDKit", (0.25, 0.5)),
    ("3. Phase 3: 5-Fold\nScaffold CV", (0.45, 0.5)),
    ("4. Phase 4: Chemprop GNN\nSCAFFOLD_BALANCED", (0.65, 0.5)),
    ("5. H2H: Matched Fold 0\nRF vs GNN (n=296)", (0.80, 0.5)),
    ("6. Error Analysis\nDescriptors + Statistics", (0.95, 0.5))
]

for text, (x, y) in steps:
    ax.annotate(
        text, xy=(x, y), fontsize=9, ha='center', va='center',
        bbox=dict(boxstyle='round,pad=0.3', facecolor=COLOR_RF, alpha=0.15, edgecolor=COLOR_RF),
        xycoords='axes fraction'
    )

# Draw arrows between steps
for i in range(len(steps) - 1):
    x1, y1 = steps[i][1]
    x2, y2 = steps[i+1][1]
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5),
                xycoords='axes fraction')

ax.set_title("Project Workflow: BBB Permeability Prediction Pipeline", fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/project_workflow.png", bbox_inches='tight')
plt.close()

# ─── Figure 2: Model Performance Comparison ───────────────────────────────
print("Generating Figure 2: model_performance.png")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left: Phase 3 5-fold results
phase3_rf = df[(df['model'] == 'rf')].sort_values('AUROC')
colors = [COLOR_RF if 'combined' in str(f) else '#666666' for f in phase3_rf['feature_set']]
bars = ax1.barh(
    range(len(phase3_rf)), phase3_rf['AUROC'],
    color=colors, edgecolor='white', capsize=3
)
ax1.set_ylabels = True
ax1.set_yticks(range(len(phase3_rf)))
ax1.set_yticklabels([f"{r['feature_set']} / {r['model']}" for _, r in phase3_rf.iterrows()])
ax1.set_xlabel('AUROC (5-fold mean ± std)')
ax1.set_title('Phase 3: Classical ML (5-Fold Scaffold CV)', fontweight='bold')
ax1.axvline(0.921, color=COLOR_PHASE3, ls='--', alpha=0.7, label=f'RF Combined: 0.921±0.024')
ax1.set_xlim(0.75, 1.0)
ax1.legend(loc='lower right')

# Right: All phases compared
phases = ['Phase 3\n(5-fold mean)', 'Phase 4\n(5-fold ensemble)', 'H2H\n(Fold 0, matched)']
rf_vals = [0.921, 0.921, 0.9031]  # Phase 3 RF, Phase 3 RF (same), H2H RF
gnn_vals = [0.0, 0.935, 0.8669]   # Phase 3 has no GNN, Phase 4 ensemble, H2H single

x = np.arange(len(phases))
width = 0.35

bars1 = ax2.bar(x - width/2, rf_vals, width, label='RF', color=COLOR_RF, edgecolor='white')
bars2 = ax2.bar(x + width/2, gnn_vals, width, label='GNN (Chemprop)', color=COLOR_GNN, edgecolor='white')

ax2.set_ylabel('AUROC')
ax2.set_title('Model Performance Across Phases', fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(phases)
ax2.set_ylim(0.75, 1.0)
ax2.legend()

# Add value labels
for bar in bars1:
    h = bar.get_height()
    if h > 0.01:
        ax2.text(bar.get_x() + bar.get_width()/2., h + 0.005, f'{h:.3f}', ha='center', va='bottom', fontsize=9)
for bar in bars2:
    h = bar.get_height()
    if h > 0.01:
        ax2.text(bar.get_x() + bar.get_width()/2., h + 0.005, f'{h:.3f}', ha='center', va='bottom', fontsize=9)

# Add annotation about different splits
ax2.annotate('Phase 4 uses different\nsplit protocol', xy=(2, 0.77), fontsize=7, color='gray',
             ha='center', style='italic')

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/model_performance.png", bbox_inches='tight')
plt.close()

# ─── Figure 3: H2H ROC Curve ─────────────────────────────────────────────
print("Generating Figure 3: h2h_roc_curve.png")
fig, ax = plt.subplots(figsize=(7, 6))

fpr_rf, tpr_rf, _ = roc_curve(h2h['true_label'], h2h['rf_prob'])
fpr_gnn, tpr_gnn, _ = roc_curve(h2h['true_label'], h2h['gnn_prob'])

auc_rf = auc(fpr_rf, tpr_rf)
auc_gnn = auc(fpr_gnn, tpr_gnn)

ax.plot(fpr_rf, tpr_rf, color=COLOR_RF, linewidth=2.5,
        label=f'RF (AUROC = {auc_rf:.4f})')
ax.plot(fpr_gnn, tpr_gnn, color=COLOR_GNN, linewidth=2.5,
        label=f'GNN (AUROC = {auc_gnn:.4f})')
ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random (AUROC = 0.5)')
ax.set_xlabel('False Positive Rate (1 - Specificity)')
ax.set_ylabel('True Positive Rate (Sensitivity)')
ax.set_title('H2H Comparison: ROC Curves (Fold 0, n=296)', fontweight='bold')
ax.legend(loc='lower right')
ax.set_xlim(-0.01, 1.01)
ax.set_ylim(-0.01, 1.01)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/h2h_roc_curve.png", bbox_inches='tight')
plt.close()

# ─── Figure 4: H2H Precision-Recall Curve ────────────────────────────────
print("Generating Figure 4: h2h_precision_recall_curve.png")
fig, ax = plt.subplots(figsize=(7, 6))

prec_rf, rec_rf, _ = precision_recall_curve(h2h['true_label'], h2h['rf_prob'])
prec_gnn, rec_gnn, _ = precision_recall_curve(h2h['true_label'], h2h['gnn_prob'])

auprc_rf = auc(rec_rf, prec_rf)
auprc_gnn = auc(rec_gnn, prec_gnn)

ax.plot(rec_rf, prec_rf, color=COLOR_RF, linewidth=2.5,
        label=f'RF (AUPRC = {auprc_rf:.4f})')
ax.plot(rec_gnn, prec_gnn, color=COLOR_GNN, linewidth=2.5,
        label=f'GNN (AUPRC = {auprc_gnn:.4f})')

# Baseline (positive rate)
pos_rate = h2h['true_label'].mean()
ax.axhline(y=pos_rate, color='gray', linestyle='--', alpha=0.5,
           label=f'Baseline (AUPRC = {pos_rate:.4f})')

ax.set_xlabel('Recall (Sensitivity)')
ax.set_ylabel('Precision')
ax.set_title('H2H Comparison: Precision-Recall Curves (Fold 0, n=296)', fontweight='bold')
ax.legend(loc='lower left')
ax.set_xlim(-0.01, 1.01)
ax.set_ylim(-0.01, 1.01)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/h2h_precision_recall_curve.png", bbox_inches='tight')
plt.close()

# ─── Figure 5: Error Overlap ─────────────────────────────────────────────
print("Generating Figure 5: error_overlap.png")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left: Confusion matrix
h2h_y = h2h['true_label'].values
rf_pred = h2h['rf_pred'].values
gnn_pred = h2h['gnn_pred'].values

from sklearn.metrics import confusion_matrix
cm_rf = confusion_matrix(h2h_y, rf_pred, labels=[0, 1])
cm_gnn = confusion_matrix(h2h_y, gnn_pred, labels=[0, 1])

im1 = ax1.imshow(cm_rf, cmap='Blues', aspect='auto')
ax1.set_xticks([0, 1])
ax1.set_yticks([0, 1])
ax1.set_xticklabels(['Pred Negative', 'Pred Positive'])
ax1.set_yticklabels(['Actual Negative', 'Actual Positive'])
ax1.set_title('RF Confusion Matrix\n(TN/FP/FN/TP: 32/28/9/227)', fontweight='bold')
for i in range(2):
    for j in range(2):
        ax1.text(j, i, str(cm_rf[i, j]), ha='center', va='center',
                 fontsize=16, fontweight='bold' if cm_rf[i, j] > 20 else 'normal',
                 color='white' if cm_rf[i, j] > 20 else 'black')

im2 = ax2.imshow(cm_gnn, cmap='RdPu', aspect='auto')
ax2.set_xticks([0, 1])
ax2.set_yticks([0, 1])
ax2.set_xticklabels(['Pred Negative', 'Pred Positive'])
ax2.set_yticklabels(['Actual Negative', 'Actual Positive'])
ax2.set_title('GNN Confusion Matrix\n(TN/FP/FN/TP: 38/22/20/216)', fontweight='bold')
for i in range(2):
    for j in range(2):
        ax2.text(j, i, str(cm_gnn[i, j]), ha='center', va='center',
                 fontsize=16, fontweight='bold' if cm_gnn[i, j] > 20 else 'normal',
                 color='white' if cm_gnn[i, j] > 20 else 'black')

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/error_overlap.png", bbox_inches='tight')
plt.close()

# ─── Figure 6: Error Descriptor Analysis ─────────────────────────────────
print("Generating Figure 6: error_descriptor_analysis.png")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Extract error groups from EA
rf_fp = ea[ea['rf_fp'] == True]  # noqa: E712
rf_fn = ea[ea['rf_fn'] == True]  # noqa: E712
gnn_fn = ea[ea['gnn_fn'] == True]  # noqa: E712
both_correct = ea[(ea['rf_error'] == False) & (ea['gnn_error'] == False)]  # noqa: E712

# Plot 1: TPSA comparison
ax = axes[0, 0]
groups = [both_correct, rf_fn, gnn_fn]
labels = ['Both Correct\n(n={})'.format(len(g)) for g in groups]
means = [g['tpsa'].mean() for g in groups]
stds = [g['tpsa'].std() for g in groups]
colors = [COLOR_ACCENT, COLOR_RF, COLOR_GNN]
bars = ax.bar(labels, means, yerr=stds, color=colors, edgecolor='white', capsize=4, alpha=0.8)
ax.set_ylabel('TPSA (Å²)')
ax.set_title('TPSA by Error Group', fontweight='bold')
ax.axhline(both_correct['tpsa'].mean(), color='gray', ls='--', alpha=0.5)
ax.text(0, both_correct['tpsa'].mean() + 3, 'baseline', fontsize=8, color='gray')

# Plot 2: MW comparison
ax = axes[0, 1]
means_mw = [g['mw'].mean() for g in groups]
stds_mw = [g['mw'].std() for g in groups]
bars = ax.bar(labels, means_mw, yerr=stds_mw, color=colors, edgecolor='white', capsize=4, alpha=0.8)
ax.set_ylabel('Molecular Weight (Da)')
ax.set_title('Molecular Weight by Error Group', fontweight='bold')

# Plot 3: QED comparison
ax = axes[1, 0]
groups_qed = [both_correct, rf_fp, gnn_fn]
labels_qed = ['Both Correct\n(n={})'.format(len(g)) for g in groups_qed]
means_qed = [g['qed'].mean() for g in groups_qed]
stds_qed = [g['qed'].std() for g in groups_qed]
colors_qed = [COLOR_ACCENT, COLOR_RF, COLOR_GNN]
bars = ax.bar(labels_qed, means_qed, yerr=stds_qed, color=colors_qed, edgecolor='white', capsize=4, alpha=0.8)
ax.set_ylabel('QED (Drug-likeness)')
ax.set_title('QED by Error Group', fontweight='bold')

# Plot 4: Error count bar chart
ax = axes[1, 1]
error_types = ['RF False\nPositives', 'RF False\nNegatives', 'GNN False\nPositives', 'GNN False\nNegatives']
error_counts = [len(rf_fp), len(rf_fn), len(ea[ea['gnn_fp'] == True]), len(gnn_fn)]  # noqa: E712
bars = ax.bar(error_types, error_counts, color=[COLOR_RF, COLOR_RF, COLOR_GNN, COLOR_GNN],
              edgecolor='white', alpha=0.8)
ax.set_ylabel('Count')
ax.set_title('Error Counts by Model', fontweight='bold')
for bar, count in zip(bars, error_counts):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5, str(count),
            ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.suptitle('Error Group Descriptor Analysis (Fold 0, n=296)', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/error_descriptor_analysis.png", bbox_inches='tight')
plt.close()

print(f"\nAll figures saved to {FIG_DIR}/:")
for f in sorted(os.listdir(FIG_DIR)):
    if f.endswith('.png'):
        size = os.path.getsize(f"{FIG_DIR}/{f}") / 1024
        print(f"  {f} ({size:.1f} KB)")
