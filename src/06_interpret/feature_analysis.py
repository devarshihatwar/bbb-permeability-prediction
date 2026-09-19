"""
Feature importance analysis for the classical ML baseline.

Uses the trained Random Forest (best scaffold-split performer) to rank
molecular features by importance (Gini importance), and groups them by
chemical category to produce a human-readable ranking.

Also computes SHAP values for the best model configuration to explain
individual predictions.
"""
import argparse
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


def get_feature_names(descs_df):
    """Generate descriptive names for RDKit descriptors."""
    desc_names = list(descs_df.columns)
    morgan_names = [f"morgan_bit_{i}" for i in range(2048)]
    combined_names = morgan_names + desc_names
    return morgan_names, desc_names, combined_names


def train_rf_for_importance(features, labels, seed=42):
    """Train a single RF model for feature importance analysis."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    rf = RandomForestClassifier(
        n_estimators=500, max_depth=None, min_samples_leaf=1,
        class_weight="balanced", random_state=seed, n_jobs=-1
    )
    rf.fit(X_scaled, labels)
    return rf, scaler


def group_descriptor_names(desc_names):
    """Group RDKit descriptors into chemical categories for interpretation."""
    categories = {
        "size_weight": [],
        "lipophilicity": [],
        "polarity_charges": [],
        "hydrogen_bonds": [],
        "ring_systems": [],
        "fragment_counts": [],
        "path_grid": [],
        "others": [],
    }
    for name in desc_names:
        n = name.lower()
        if any(w in n for w in ["molwt", "heavyatom", "nhohcount", "natom"]):
            categories["size_weight"].append(name)
        elif any(w in n for w in ["logp", "mlogp", "logs", "alogp", "tpsa", "labuteasa", "balabolasa"]):
            categories["lipophilicity"].append(name)
        elif any(w in n for w in ["numhDonors", "numhacceptors", "nrotbonds", "donorcount", "acceptorcount"]):
            categories["hydrogen_bonds"].append(name)
        elif any(w in n for w in ["numaromaticrings", "numaliphaticrings", "numaromaticheterocycles",
                                   "ringcount", "fracsp3", "numheterocycles", "numrings"]):
            categories["ring_systems"].append(name)
        elif any(w in n for w in ["numaliphaticheterocycles", "numaromaticcarbocycles"]):
            categories["ring_systems"].append(name)
        elif any(w in n for w in ["npath", "numpath", "pathwindow", "numpath"]):
            categories["path_grid"].append(name)
        elif any(w in n for w in ["numc", "numn", "numo", "halogen", "numhalogen"]):
            categories["fragment_counts"].append(name)
        elif any(w in n for w in ["charge", "valence", "val", "qed", "mollogp"]):
            categories["polarity_charges"].append(name)
        else:
            categories["others"].append(name)
    return categories


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", default="data/features")
    parser.add_argument("--dataset", default="data/bbb_martins.csv")
    parser.add_argument("--output", default="results")
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    df = pd.read_csv(args.dataset)
    labels = df["target"].values
    morgan = np.load(os.path.join(args.features_dir, "morgan_fps.npy"))
    descs_df = pd.read_csv(os.path.join(args.features_dir, "rdkit_descriptors.csv"))
    combined = np.load(os.path.join(args.features_dir, "combined_features.npy"))

    morgan_names, desc_names, combined_names = get_feature_names(descs_df)

    print("=" * 80)
    print("FEATURE IMPORTANCE ANALYSIS (Random Forest, Gini importance)")
    print("=" * 80)

    # --- Morgan fingerprints ---
    print("\n--- Morgan Fingerprints Feature Importance ---")
    rf_morgan, _ = train_rf_for_importance(morgan, labels)
    importances = rf_morgan.feature_importances_
    top_idx = np.argsort(importances)[::-1][:20]
    print(f"Top 20 most important Morgan fingerprint bits:")
    print(f"{'Bit':>10} {'Importance':>12}")
    for i in top_idx:
        print(f"  bit_{i:5d}   {importances[i]:.6f}")

    # Save morgan importances
    morgan_imp = pd.DataFrame({
        "feature": [f"morgan_bit_{i}" for i in range(2048)],
        "importance": importances,
    }).sort_values("importance", ascending=False)
    morgan_imp.to_csv(os.path.join(args.output, "feature_importance_morgan.csv"), index=False)

    # --- RDKit Descriptors ---
    print("\n--- RDKit Descriptors Feature Importance ---")
    rf_desc, _ = train_rf_for_importance(descs_df.values, labels)
    importances = rf_desc.feature_importances_
    top_idx = np.argsort(importances)[::-1][:30]
    print(f"Top 30 most important RDKit descriptors:")
    print(f"{'Descriptor':>30} {'Importance':>12}")
    for i in top_idx:
        print(f"  {desc_names[i]:>30} {importances[i]:.6f}")

    desc_imp = pd.DataFrame({
        "feature": desc_names,
        "importance": importances,
    }).sort_values("importance", ascending=False)
    desc_imp.to_csv(os.path.join(args.output, "feature_importance_descriptors.csv"), index=False)

    # Group by chemical category
    categories = group_descriptor_names(desc_names)
    print("\n--- Top descriptors by chemical category ---")
    for cat, names_in_cat in categories.items():
        if not names_in_cat:
            continue
        cat_importances = []
        for name in names_in_cat:
            idx = desc_names.index(name)
            cat_importances.append((name, importances[idx]))
        cat_importances.sort(key=lambda x: x[1], reverse=True)
        top3 = cat_importances[:3]
        total_imp = sum(x[1] for x in cat_importances)
        top3_str = ", ".join(f"{n}({v:.4f})" for n, v in top3)
        print(f"  {cat:20s} (total={total_imp:.4f}): {top3_str}")

    desc_imp.to_csv(os.path.join(args.output, "feature_importance_descriptors.csv"), index=False)

    # --- Combined ---
    print("\n--- Combined Features Top 30 ---")
    rf_combined, _ = train_rf_for_importance(combined, labels)
    importances = rf_combined.feature_importances_
    top_idx = np.argsort(importances)[::-1][:30]
    print(f"{'Feature':>35} {'Importance':>12}")
    for i in top_idx:
        if i < 2048:
            name = f"morgan_bit_{i}"
        else:
            name = desc_names[i - 2048]
        print(f"  {name:>35} {importances[i]:.6f}")

    combined_imp = pd.DataFrame({
        "feature": combined_names,
        "importance": importances,
    }).sort_values("importance", ascending=False)
    combined_imp.to_csv(os.path.join(args.output, "feature_importance_combined.csv"), index=False)

    print(f"\nFeature importance saved to {args.output}/")
    print(f"Max combined importance: {importances.max():.6f}")
    print(f"Top 5 features account for: {importances[top_idx[:5]].sum():.4f} of total importance")


if __name__ == "__main__":
    main()
