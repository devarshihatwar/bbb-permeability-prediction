"""
Classical ML baseline for BBB penetration prediction.

Trains Logistic Regression, Random Forest, and SVM (RBF kernel) on three
feature sets (Morgan fingerprints, RDKit descriptors, combined) using
pre-split train/val/test indices that are pre-audited for zero leakage.

Key design: splits are loaded as positional indices; features are computed
once for the whole dataset (they are molecule-intrinsic, not
target-dependent, so no leakage). Scaling fits ONLY on train+val. Test
set is never touched during model selection.
"""
import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    f1_score, precision_score, recall_score, confusion_matrix,
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

warnings.filterwarnings("ignore")


def compute_metrics(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return {
        "AUROC": float(roc_auc_score(y_true, y_prob)),
        "AUPRC": float(average_precision_score(y_true, y_prob)),
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "F1": float(f1_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "Sensitivity": float(sensitivity),
        "Specificity": float(specificity),
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
    }


def get_model_and_grid(model_name, seed=42):
    if model_name == "logreg":
        return (
            LogisticRegression(max_iter=5000, class_weight="balanced", random_state=seed),
            {"C": [0.1, 1.0, 10.0]},
        )
    elif model_name == "rf":
        return (
            RandomForestClassifier(class_weight="balanced", random_state=seed, n_jobs=-1),
            {"n_estimators": [300], "max_depth": [None, 30],
             "min_samples_leaf": [1, 3]},
        )
    elif model_name == "svm":
        return (
            SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=seed),
            {"C": [1.0, 10.0], "gamma": ["scale"]},
        )
    raise ValueError(f"Unknown model: {model_name}")


def train_and_evaluate(model_name, X_trainval, y_trainval, X_test, y_test, seed=42):
    """Train on train+val, evaluate on test. Scaler fitted ONLY on train+val."""
    scaler = StandardScaler()
    X_trainval_s = scaler.fit_transform(X_trainval)
    X_test_s = scaler.transform(X_test)

    model, param_grid = get_model_and_grid(model_name, seed=seed)
    # n_jobs=1 here: RF already uses n_jobs=-1 internally; nested parallelism
    # causes contention and slowdowns on Windows.
    gs = GridSearchCV(model, param_grid, cv=3, scoring="roc_auc", n_jobs=1)
    gs.fit(X_trainval_s, y_trainval)
    best_model = gs.best_estimator_

    y_prob = best_model.predict_proba(X_test_s)[:, 1]
    metrics = compute_metrics(y_test, y_prob)
    metrics["model"] = model_name
    metrics["best_params"] = str(gs.best_params_)
    metrics["cv_auc_mean"] = float(gs.best_score_)
    return metrics, best_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", default="data/features")
    parser.add_argument("--dataset", default="data/bbb_martins.csv")
    parser.add_argument("--output", default="results")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    df = pd.read_csv(args.dataset)
    labels = df["target"].values
    print(f"Dataset: {len(df)} compounds | {dict(zip(*np.unique(labels, return_counts=True)))}")

    morgan = np.load(os.path.join(args.features_dir, "morgan_fps.npy"))
    descriptors = pd.read_csv(os.path.join(args.features_dir, "rdkit_descriptors.csv")).values
    combined = np.load(os.path.join(args.features_dir, "combined_features.npy"))

    feature_sets = {
        "morgan": morgan,
        "descriptors": descriptors,
        "combined": combined,
    }

    import pickle
    with open(os.path.join(args.features_dir, "scaffold_splits.pkl"), "rb") as f:
        scaffold_splits = pickle.load(f)
    with open(os.path.join(args.features_dir, "random_splits.pkl"), "rb") as f:
        random_splits = pickle.load(f)

    all_results = []

    for split_name, splits in [("scaffold", scaffold_splits), ("random", random_splits)]:
        print(f"\n{'=' * 80}")
        print(f"SPLIT TYPE: {split_name} ({len(splits)} folds)")
        print(f"{'=' * 80}", flush=True)

        for feature_name, features in feature_sets.items():
            print(f"\n--- Feature set: {feature_name} ({features.shape}) ---", flush=True)

            for model_name in ["logreg", "rf", "svm"]:
                fold_metrics = []
                for fold_idx, (tr_idx, va_idx, te_idx) in enumerate(splits):
                    trainval_indices = tr_idx + va_idx
                    X_tv = features[trainval_indices]
                    y_tv = labels[trainval_indices]
                    X_te = features[te_idx]
                    y_te = labels[te_idx]

                    r = train_and_evaluate(model_name, X_tv, y_tv, X_te, y_te, seed=fold_idx)
                    metrics = r[0]
                    metrics["feature_set"] = feature_name
                    metrics["fold"] = fold_idx
                    metrics["split"] = split_name
                    fold_metrics.append(metrics)

                all_results.extend(fold_metrics)

                # Save intermediate results
                results_df = pd.DataFrame(all_results)
                results_df.to_csv(os.path.join(args.output, "03_ml_baseline_results.csv"), index=False)

                aurocs = [m["AUROC"] for m in fold_metrics]
                accs = [m["Accuracy"] for m in fold_metrics]
                f1s = [m["F1"] for m in fold_metrics]
                print(f"  {model_name:10s} | AUROC: {np.mean(aurocs):.4f}\u00b1{np.std(aurocs):.4f} | "
                      f"Acc: {np.mean(accs):.4f}\u00b1{np.std(accs):.4f} | "
                      f"F1: {np.mean(f1s):.4f}\u00b1{np.std(f1s):.4f}", flush=True)

    # Save final results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(args.output, "03_ml_baseline_results.csv"), index=False)
    print(f"\n{'=' * 80}")
    print(f"Results saved. {len(results_df)} total rows.")

    # Summary tables
    for split_name in ["scaffold", "random"]:
        print(f"\n{'=' * 80}")
        print(f"SUMMARY - {split_name.upper()} SPLIT (5-fold CV)")
        print(f"{'=' * 80}")
        sub = results_df[results_df["split"] == split_name]
        summary = (
            sub.groupby(["feature_set", "model"])
            .agg(
                AUROC_mean=("AUROC", "mean"), AUROC_std=("AUROC", "std"),
                AUPRC_mean=("AUPRC", "mean"), AUPRC_std=("AUPRC", "std"),
                Acc_mean=("Accuracy", "mean"), Acc_std=("Accuracy", "std"),
                F1_mean=("F1", "mean"), F1_std=("F1", "std"),
                Sens_mean=("Sensitivity", "mean"), Spec_mean=("Specificity", "mean"),
            )
            .reset_index()
        )
        print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        summary.to_csv(os.path.join(args.output, f"03_ml_summary_{split_name}.csv"), index=False)

    max_auroc = results_df["AUROC"].max()
    if max_auroc > 0.98:
        print(f"\n*** WARNING: AUROC = {max_auroc:.4f} is suspiciously high. Investigate. ***")
    else:
        print(f"\n(Max AUROC across all runs: {max_auroc:.4f} - within plausible range)")


if __name__ == "__main__":
    main()
