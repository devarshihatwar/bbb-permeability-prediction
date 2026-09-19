"""
Quick baseline runner: just the remaining SVM+combined on scaffold split
and the full random split for all combinations.
This supplements the partial results already saved in 03_ml_baseline_results.csv.
"""
import os, sys, warnings, pickle, json

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
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")


def compute_metrics(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return {
        "AUROC": float(roc_auc_score(y_true, y_prob)),
        "AUPRC": float(average_precision_score(y_true, y_prob)),
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "F1": float(f1_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "Sensitivity": float(sens), "Specificity": float(spec),
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
    }


def get_model_and_grid(model_name, seed=42):
    if model_name == "logreg":
        return (LogisticRegression(max_iter=5000, class_weight="balanced", random_state=seed),
                {"C": [0.1, 1.0, 10.0]})
    elif model_name == "rf":
        return (RandomForestClassifier(class_weight="balanced", random_state=seed, n_jobs=-1),
                {"n_estimators": [300], "max_depth": [None, 30], "min_samples_leaf": [1, 3]})
    elif model_name == "svm":
        return (SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=seed),
                {"C": [1.0, 10.0], "gamma": ["scale"]})


def train_and_evaluate(model_name, X_tv, y_tv, X_te, y_te, seed=42):
    scaler = StandardScaler()
    X_tv_s = scaler.fit_transform(X_tv)
    X_te_s = scaler.transform(X_te)
    model, grid = get_model_and_grid(model_name, seed)
    gs = GridSearchCV(model, grid, cv=3, scoring="roc_auc", n_jobs=1)
    gs.fit(X_tv_s, y_tv)
    y_prob = gs.predict_proba(X_te_s)[:, 1]
    metrics = compute_metrics(y_te, y_prob)
    metrics["model"] = model_name
    metrics["best_params"] = str(gs.best_params_)
    metrics["cv_auc_mean"] = float(gs.best_score_)
    return metrics


def main():
    df = pd.read_csv("data/bbb_martins.csv")
    labels = df["target"].values

    morgan = np.load("data/features/morgan_fps.npy")
    descriptors = pd.read_csv("data/features/rdkit_descriptors.csv").values
    combined = np.load("data/features/combined_features.npy")

    feature_sets = {"morgan": morgan, "descriptors": descriptors, "combined": combined}

    with open("data/features/scaffold_splits.pkl", "rb") as f:
        scaffold_splits = pickle.load(f)
    with open("data/features/random_splits.pkl", "rb") as f:
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
                for fold_idx, (tr, va, te) in enumerate(splits):
                    tv_idx = tr + va
                    r = train_and_evaluate(model_name, features[tv_idx], labels[tv_idx],
                                          features[te], labels[te], seed=fold_idx)
                    r["feature_set"] = feature_name
                    r["fold"] = fold_idx
                    r["split"] = split_name
                    fold_metrics.append(r)

                # Load existing results and append
                existing_path = "results/03_ml_baseline_results.csv"
                if os.path.exists(existing_path):
                    existing = pd.read_csv(existing_path)
                    # Check if this exact combination already exists
                    mask = (existing["feature_set"] == feature_name) & \
                           (existing["model"] == model_name) & \
                           (existing["split"] == split_name)
                    if mask.any():
                        # Already exists, skip
                        all_results.extend(fold_metrics)
                    else:
                        existing = pd.concat([existing, pd.DataFrame(fold_metrics)],
                                              ignore_index=True)
                        existing.to_csv(existing_path, index=False)
                        all_results.extend(fold_metrics)
                else:
                    all_results.extend(fold_metrics)

                aurocs = [m["AUROC"] for m in fold_metrics]
                accs = [m["Accuracy"] for m in fold_metrics]
                f1s = [m["F1"] for m in fold_metrics]
                print(f"  {model_name:10s} | AUROC: {np.mean(aurocs):.4f}+/-{np.std(aurocs):.4f} | "
                      f"Acc: {np.mean(accs):.4f}+/-{np.std(accs):.4f} | "
                      f"F1: {np.mean(f1s):.4f}+/-{np.std(f1s):.4f}", flush=True)

    # Save complete results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv("results/03_ml_baseline_results.csv", index=False)
    results_df.to_excel("results/03_ml_baseline_results.xlsx", index=False)

    # Summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY (5-fold CV)")
    print("=" * 80)
    for split_name in ["scaffold", "random"]:
        sub = results_df[results_df["split"] == split_name]
        summary = (sub.groupby(["feature_set", "model"])
                   .agg(AUROC_mean=("AUROC", "mean"), AUROC_std=("AUROC", "std"),
                        AUPRC_mean=("AUPRC", "mean"), Acc_mean=("Accuracy", "mean"),
                        F1_mean=("F1", "mean"))
                   .reset_index())
        print(f"\n--- {split_name.upper()} ---")
        print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        summary.to_csv(f"results/03_ml_summary_{split_name}.csv", index=False)

    max_auroc = results_df["AUROC"].max()
    print(f"\nMax AUROC: {max_auroc:.4f}")
    if max_auroc > 0.98:
        print("*** WARNING: suspiciously high ***")
    else:
        print("All results within plausible range.")


if __name__ == "__main__":
    main()
