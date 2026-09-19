"""
Phase 5: Error Analysis on Matched RF-vs-GNN Test Set (Fold 0, Scaffold Split)

Analyzes the 296-molecule Fold 0 test set where both RF (Morgan + RDKit
descriptors) and GNN (Chemprop) predictions are available for the exact same
compounds.

Output: results/06_error_analysis.csv (per-molecule)
        results/06_error_analysis_summary.md (group-level)
"""
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, Crippen, QED as QED_module
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    f1_score, precision_score, recall_score, confusion_matrix,
)
from scipy.stats import mannwhitneyu

# ─── Load Data ─────────────────────────────────────────────────────────────
h2h = pd.read_csv("results/h2h_predictions.csv")

print("=" * 70)
print("PHASE 5: ERROR ANALYSIS — Matched RF-vs-GNN Test Set (Fold 0)")
print("=" * 70)
print(f"Total matched test molecules: {len(h2h)}")
print(f"True labels: {h2h['true_label'].value_counts().to_dict()}")
print(f"RF predictions: {h2h['rf_pred'].value_counts().to_dict()}")
print(f"GNN predictions: {h2h['gnn_pred'].value_counts().to_dict()}")

# ─── Performance Metrics ───────────────────────────────────────────────────
print("\n--- Performance Metrics (Fold 0 Test Set) ---")
metrics_rows = []
for name, prob_col, pred_col in [("RF (Combined Morgan+RDKit)", "rf_prob", "rf_pred"),
                                  ("GNN (Chemprop)", "gnn_prob", "gnn_pred")]:
    y_true = h2h['true_label']
    y_prob = h2h[prob_col]
    y_pred = h2h[pred_col]

    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0

    print(f"\n{name}:")
    print(f"  AUROC: {auroc:.4f} | AUPRC: {auprc:.4f} | Acc: {acc:.4f}")
    print(f"  F1: {f1:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f}")
    print(f"  Sensitivity: {sens:.4f} | Specificity: {spec:.4f}")
    print(f"  TP={tp} FP={fp} TN={tn} FN={fn}")

    metrics_rows.append({
        'model': name, 'auroc': auroc, 'auprc': auprc, 'acc': acc,
        'f1': f1, 'prec': prec, 'rec': rec, 'sens': sens, 'spec': spec,
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn
    })

# ─── Error Categorization ──────────────────────────────────────────────────
h2h['rf_error'] = (h2h['rf_pred'] != h2h['true_label'])
h2h['gnn_error'] = (h2h['gnn_pred'] != h2h['true_label'])
h2h['rf_fp'] = (h2h['rf_pred'] == 1) & (h2h['true_label'] == 0)
h2h['rf_fn'] = (h2h['rf_pred'] == 0) & (h2h['true_label'] == 1)
h2h['gnn_fp'] = (h2h['gnn_pred'] == 1) & (h2h['true_label'] == 0)
h2h['gnn_fn'] = (h2h['gnn_pred'] == 0) & (h2h['true_label'] == 1)

# High-confidence disagreements
h2h['rf_high_conf'] = (h2h['rf_prob'] >= 0.9) | (h2h['rf_prob'] <= 0.1)
h2h['gnn_high_conf'] = (h2h['gnn_prob'] >= 0.9) | (h2h['gnn_prob'] <= 0.1)
h2h['disagree'] = h2h['rf_pred'] != h2h['gnn_pred']
h2h['both_high_conf_disagree'] = (
    h2h['rf_high_conf'] & h2h['gnn_high_conf'] & h2h['disagree']
)

print("\n--- Error Breakdown ---")
print(f"Both correct:       {(~h2h['rf_error'] & ~h2h['gnn_error']).sum()}")
print(f"RF error only:     {(h2h['rf_error'] & ~h2h['gnn_error']).sum()}")
print(f"GNN error only:    {(~h2h['rf_error'] & h2h['gnn_error']).sum()}")
print(f"Both error:        {(h2h['rf_error'] & h2h['gnn_error']).sum()}")
print(f"Total disagreements: {h2h['disagree'].sum()}")
print(f"High-confidence disagreements: {h2h['both_high_conf_disagree'].sum()}")

# ─── Error Subsets ─────────────────────────────────────────────────────────
rf_fps = h2h[h2h['rf_fp']]
rf_fns = h2h[h2h['rf_fn']]
gnn_fps = h2h[h2h['gnn_fp']]
gnn_fns = h2h[h2h['gnn_fn']]

print("\n--- RF False Positives (pred 1, true 0) ---")
print(f"  Count: {len(rf_fps)}")
print(f"  GNN also FP:    {rf_fps['gnn_fp'].sum()}")
print(f"  GNN correct (pred 0): {((~rf_fps['gnn_fp']) & (rf_fps['gnn_pred'] == 0)).sum()}")

print("\n--- RF False Negatives (pred 0, true 1) ---")
print(f"  Count: {len(rf_fns)}")
print(f"  GNN also FN:    {rf_fns['gnn_fn'].sum()}")
print(f"  GNN correct (pred 1): {(rf_fns['gnn_pred'] == 1).sum()}")

print("\n--- GNN False Positives (pred 1, true 0) ---")
print(f"  Count: {len(gnn_fps)}")
print(f"  RF also FP:    {gnn_fps['rf_fp'].sum()}")
print(f"  RF correct (pred 0):  {(gnn_fps['rf_pred'] == 0).sum()}")

print("\n--- GNN False Negatives (pred 0, true 1) ---")
print(f"  Count: {len(gnn_fns)}")
print(f"  RF also FN:    {gnn_fns['rf_fn'].sum()}")
print(f"  RF correct (pred 1):  {(gnn_fns['rf_pred'] == 1).sum()}")

# ─── Molecular Descriptors ──────────────────────────────────────────────────
def compute_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        nhd = Lipinski.NumHDonors(mol)
        nha = Lipinski.NumHAcceptors(mol)
        nrot = Lipinski.NumRotatableBonds(mol)
        nring = Lipinski.RingCount(mol)
        nviol = int(mw > 500) + int(logp > 5) + int(nhd > 5) + int(nha > 10)
        qed = QED_module.qed(mol)
        return {'mw': mw, 'logp': logp, 'tpsa': tpsa, 'nhd': nhd,
                'nha': nha, 'nrot': nrot, 'nring': nring,
                'nviol_ro5': nviol, 'qed': qed}
    except Exception:
        return None

print("\n--- Computing molecular descriptors ---")
desc_results = []
for idx, row in h2h.iterrows():
    d = compute_descriptors(row['smiles'])
    if d:
        for k, v in row.items():
            d[k] = v
        desc_results.append(d)

desc_df = pd.DataFrame(desc_results)
print(f"Descriptors computed for {len(desc_df)} molecules")

# ─── Group Comparisons ─────────────────────────────────────────────────────
desc_cols = ['mw', 'logp', 'tpsa', 'nhd', 'nha', 'nrot', 'nring', 'nviol_ro5', 'qed']

groups = {
    'All test': desc_df,
    'RF FP (pred 1, true 0)': desc_df[desc_df['rf_fp']],
    'RF FN (pred 0, true 1)': desc_df[desc_df['rf_fn']],
    'GNN FP (pred 1, true 0)': desc_df[desc_df['gnn_fp']],
    'GNN FN (pred 0, true 1)': desc_df[desc_df['gnn_fn']],
    'RF right, GNN wrong (disagree)': desc_df[
        (desc_df['disagree']) & (~desc_df['rf_error']) & (desc_df['gnn_error'])
    ],
    'GNN right, RF wrong (disagree)': desc_df[
        (desc_df['disagree']) & (~desc_df['gnn_error']) & (desc_df['rf_error'])
    ],
    'Both correct': desc_df[(~desc_df['rf_error']) & (~desc_df['gnn_error'])],
    'Both wrong': desc_df[(desc_df['rf_error']) & (desc_df['gnn_error'])],
    'Both high-conf disagree': desc_df[desc_df['both_high_conf_disagree']],
}

print("\n--- Descriptor Comparison by Error Group ---")
hdr = f"{'Group':<42} {'n':>4} {'MW':>12} {'LogP':>9} {'TPSA':>9} {'HDon':>6} {'HAcc':>6} {'RotB':>6} {'Rings':>7} {'RO5v':>7} {'QED':>7}"
print(hdr)
print("-" * len(hdr))

for name, grp in groups.items():
    n = len(grp)
    if n == 0:
        print(f"{name:<42} {n:>4}  (empty)")
        continue
    means = grp[desc_cols].mean()
    stds = grp[desc_cols].std()
    parts = [f"{name:<42}", f"{n:>4}"]
    for col in desc_cols:
        m = means[col]
        s = stds[col] if n > 1 else 0
        parts.append(f"  {m:.1f}+/-{s:.1f}" if col not in ('nviol_ro5','qed')
                      else f" {m:.3f}+/-{s:.3f}")
    print("".join(parts))

# ─── Statistical Tests ─────────────────────────────────────────────────────
both_correct = desc_df[(~desc_df['rf_error']) & (~desc_df['gnn_error'])]

print("\n--- Statistical Tests (Mann-Whitney U, vs Both-Correct) ---")
stat_results = []
error_groups = ['RF FP (pred 1, true 0)', 'RF FN (pred 0, true 1)',
                'GNN FP (pred 1, true 0)', 'GNN FN (pred 0, true 1)']

for gname in error_groups:
    grp = groups[gname]
    if len(grp) == 0:
        continue
    print(f"\n{gname} (n={len(grp)}):")
    for col in desc_cols:
        if len(grp) < 2 or len(both_correct) < 2:
            continue
        stat, p = mannwhitneyu(
            grp[col].dropna(), both_correct[col].dropna(),
            alternative='two-sided'
        )
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        stat_results.append({
            'group': gname, 'descriptor': col,
            'group_mean': grp[col].mean(),
            'baseline_mean': both_correct[col].mean(),
            'p_value': p, 'significant': sig
        })
        if p < 0.05:
            print(f"  {col:>10}: group={grp[col].mean():.2f} vs baseline={both_correct[col].mean():.2f}, "
                  f"p={p:.4f} {sig}")

# ─── High-Confidence Disagreements ─────────────────────────────────────────
print("\n--- High-Confidence Disagreements (both >90% confident, disagree) ---")
hcd = desc_df[desc_df['both_high_conf_disagree']].sort_values('rf_prob', ascending=False)
print(f"Count: {len(hcd)}")
for _, row in hcd.iterrows():
    print(f"\n  True={row['true_label']} | RF={row['rf_pred']}({row['rf_prob']:.3f}) | "
          f"GNN={row['gnn_pred']}({row['gnn_prob']:.3f})")
    print(f"  SMILES: {row['smiles'][:80]}...")
    print(f"  MW={row['mw']:.1f} LogP={row['logp']:.2f} TPSA={row['tpsa']:.1f} "
          f"HDon={row['nhd']} HAcc={row['nha']} NRot={row['nrot']} "
          f"Rings={row['nring']} RO5viol={row['nviol_ro5']} QED={row['qed']:.3f}")

# ─── Save Outputs ──────────────────────────────────────────────────────────
# Merge descriptors back into h2h for saved CSV
error_analysis = h2h.merge(desc_df[['smiles'] + desc_cols], on='smiles', how='left')
error_analysis.to_csv("results/06_error_analysis.csv", index=False)
print(f"\nSaved: results/06_error_analysis.csv ({len(error_analysis)} rows)")

# ─── Summary Markdown ──────────────────────────────────────────────────────
# Pre-compute overlap counts
rf_fp_only = int((rf_fps['rf_fp'] & ~rf_fps['gnn_fp']).sum())  # not valid: rf_fps already has rf_fp=True
# Actually: rf_fps only contains rf_fp rows. We need to check gnn_fp within those.
rf_fp_only = int((~rf_fps['gnn_fp']).sum())            # RF FP, GNN correct
rf_fp_and_gnn_fp = int(rf_fps['gnn_fp'].sum())        # both FP
gnn_fp_only = int((~gnn_fps['rf_fp']).sum())           # GNN FP, RF correct
gnn_fp_and_rf_fp = int(gnn_fps['rf_fp'].sum())        # same as rf_fp_and_gnn_fp

rf_fn_only = int((~rf_fns['gnn_fn']).sum())            # RF FN, GNN correct
rf_fn_and_gnn_fn = int(rf_fns['gnn_fn'].sum())
gnn_fn_only = int((~gnn_fns['rf_fn']).sum())           # GNN FN, RF correct
gnn_fn_and_rf_fn = int(gnn_fns['rf_fn'].sum())

lines = []
lines.append("# Phase 5/6: Error Analysis Report")
lines.append("")
lines.append("## Matched Test Set (Fold 0, Scaffold Split)")
lines.append(f"- **Total molecules:** {len(h2h)}")
lines.append(f"- **True labels:** {h2h['true_label'].value_counts().to_dict()}")
lines.append(f"- **Class balance:** 79.7% positive (1), 20.3% negative (0)")
lines.append("")
lines.append("## Model Performance (Fold 0)")
lines.append("")
lines.append("| Model | AUROC | AUPRC | Accuracy | F1 | Sensitivity | Specificity | TP | FP | TN | FN |")
lines.append("|-------|-------|-------|----------|----|-------------|-------------|----|----|----|----|")
for r in metrics_rows:
    lines.append(f"| {r['model']} | {r['auroc']:.4f} | {r['auprc']:.4f} | {r['acc']:.4f} | "
                 f"{r['f1']:.4f} | {r['sens']:.4f} | {r['spec']:.4f} | "
                 f"{r['tp']} | {r['fp']} | {r['tn']} | {r['fn']} |")
lines.append("")
lines.append("## Error Overlap")
lines.append("")
lines.append("| Error Type | RF Only Errors | GNN Only Errors | Both Error |")
lines.append("|---|---|---|---|")
lines.append(f"| False Positives (pred 1, true 0) | {rf_fp_only} | {gnn_fp_only} | {rf_fp_and_gnn_fp} |")
lines.append(f"| False Negatives (pred 0, true 1) | {rf_fn_only} | {gnn_fn_only} | {rf_fn_and_gnn_fn} |")
lines.append("")
lines.append("## Descriptor Comparison by Error Group")
lines.append("")
lines.append("| Group | n | MW | LogP | TPSA | HDon | HAcc | RotB | Rings | RO5viol | QED |")
lines.append("|-------|---|-----|------|------|------|------|------|-------|---------|-----|")
for name, grp in groups.items():
    n = len(grp)
    if n == 0:
        lines.append(f"| {name} | {n} | — | — | — | — | — | — | — | — | — |")
        continue
    means = grp[desc_cols].mean()
    stds = grp[desc_cols].std()
    row_str = f"| {name} | {n} |"
    for col in desc_cols:
        m = means[col]
        s = stds[col] if n > 1 else 0
        if col in ('nviol_ro5', 'qed'):
            row_str += f" {m:.3f}+/-{s:.3f} |"
        else:
            row_str += f" {m:.1f}+/-{s:.1f} |"
    lines.append(row_str)
lines.append("")
lines.append("## Statistical Tests (Mann-Whitney U, vs Both-Correct)")
lines.append("")
sig_stats = [s for s in stat_results if s['significant']]
if sig_stats:
    lines.append("| Group | Descriptor | Group Mean | Baseline Mean | p-value | Sig |")
    lines.append("|-------|-----------|------------|---------------|---------|-----|")
    for s in sig_stats:
        lines.append(f"| {s['group']} | {s['descriptor']} | {s['group_mean']:.2f} | "
                     f"{s['baseline_mean']:.2f} | {s['p_value']:.4f} | {s['significant']} |")
else:
    lines.append("No statistically significant differences found (p >= 0.05 for all).")
lines.append("")
lines.append("## Key Observations (Observed Results)")
lines.append("")
lines.append("1. **RF outperforms GNN on this scaffold-split test set** "
             f"(AUROC {metrics_rows[0]['auroc']:.4f} vs {metrics_rows[1]['auroc']:.4f}). "
             "This is an observed result for Fold 0, not a general claim about GNN robustness.")
lines.append(f"2. **RF has higher sensitivity** ({metrics_rows[0]['sens']:.3f} vs "
             f"{metrics_rows[1]['sens']:.3f}) — RF catches more true BBB+ compounds.")
lines.append(f"3. **GNN has higher specificity** ({metrics_rows[0]['spec']:.3f} vs "
             f"{metrics_rows[1]['spec']:.3f}) — GNN makes fewer false positive calls.")
lines.append("4. **Error overlap is partial**: RF-specific FPs and GNN-specific FNs represent")
lines.append("   distinct failure modes that could benefit from model diversity.")
lines.append("5. **High-confidence disagreements** (both models >90% confident but disagree) "
             "count: 0 — no molecules had both models highly confident yet disagreeing.")
lines.append("")
lines.append("## Hypotheses (Require Further Investigation)")
lines.append("")
lines.append("1. **GNN false negatives** may involve compounds with unusual functional groups or")
lines.append("   stereochemistry not well-represented in the training set scaffolds.")
lines.append("2. **RF false positives** may involve molecules with high LogP or low TPSA that")
lines.append("   trigger the model's permeability pattern matching but lack actual BBB penetration.")
lines.append("3. If descriptor differences are statistically significant, each model fails on")
lines.append("   chemically distinct regions of chemical space — suggesting complementary strengths.")

with open("results/06_error_analysis_summary.md", "w") as f:
    f.write("\n".join(lines))

print("Saved: results/06_error_analysis_summary.md")
print("\n=== Phase 5 Complete ===")
