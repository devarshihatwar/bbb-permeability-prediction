"""
Split integrity audit script.
Verifies NO overlap between train/val/test in terms of:
- positional indices
- SMILES strings
- molecular scaffolds (for scaffold splits)
"""
import numpy as np
import pandas as pd
import pickle
from rdkit import Chem
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")


def main():
    df = pd.read_csv("data/bbb_martins.csv")
    labels = df["target"].values
    smiles = df["drug"].tolist()
    n = len(df)

    # Compute scaffolds for every molecule (positional)
    scaffolds = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            s = GetScaffoldForMol(mol)
            scaffolds.append(Chem.MolToSmiles(s) if s is not None else "")
        else:
            scaffolds.append("")

    print(f"Total molecules: {n}")
    print(f"Total unique scaffolds: {len(set(scaffolds))}")
    print(f"Total unique SMILES: {len(set(smiles))}")

    with open("data/features/scaffold_splits.pkl", "rb") as f:
        scaff_splits = pickle.load(f)
    with open("data/features/random_splits.pkl", "rb") as f:
        rand_splits = pickle.load(f)

    print(f"\nScaffold splits: {len(scaff_splits)} folds")
    print(f"Random splits: {len(rand_splits)} folds")

    print("\n" + "=" * 80)
    print("SPLIT INTEGRITY AUDIT")
    print("=" * 80)

    def audit_split(train_idx, val_idx, test_idx, smiles, scaffolds, label=""):
        issues = []
        tr_set, va_set, te_set = set(train_idx), set(val_idx), set(test_idx)

        # Index overlap
        if tr_set & va_set:
            issues.append(f"train/val index overlap: {len(tr_set & va_set)}")
        if tr_set & te_set:
            issues.append(f"train/test index overlap: {len(tr_set & te_set)}")
        if va_set & te_set:
            issues.append(f"val/test index overlap: {len(va_set & te_set)}")

        # SMILES overlap
        tr_sm = set(smiles[i] for i in tr_set)
        va_sm = set(smiles[i] for i in va_set)
        te_sm = set(smiles[i] for i in te_set)
        if tr_sm & va_sm:
            issues.append(f"train/val SMILES overlap: {len(tr_sm & va_sm)}")
        if tr_sm & te_sm:
            issues.append(f"train/test SMILES overlap: {len(tr_sm & te_sm)}")
        if va_sm & te_sm:
            issues.append(f"val/test SMILES overlap: {len(va_sm & te_sm)}")

        # Scaffold overlap
        tr_sc = set(scaffolds[i] for i in tr_set)
        va_sc = set(scaffolds[i] for i in va_set)
        te_sc = set(scaffolds[i] for i in te_set)
        if tr_sc & va_sc:
            issues.append(f"train/val scaffold overlap: {len(tr_sc & va_sc)}")
        if tr_sc & te_sc:
            issues.append(f"train/test scaffold overlap: {len(tr_sc & te_sc)}")
        if va_sc & te_sc:
            issues.append(f"val/test scaffold overlap: {len(va_sc & te_sc)}")

        total = len(train_idx) + len(val_idx) + len(test_idx)
        print(f"\n  {label}")
        print(f"    train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}, total={total}")
        print(f"    unique indices: {len(tr_set | va_set | te_set)} (expected {n})")
        if issues:
            print(f"    *** ISSUES: {issues}")
        else:
            print(f"    PASS: No overlap detected")
        return len(issues) == 0

    all_ok = True

    print("\n--- SCAFFOLD SPLITS ---")
    for seed, (tr, va, te) in enumerate(scaff_splits):
        ok = audit_split(tr, va, te, smiles, scaffolds, label=f"Fold {seed}")
        all_ok = all_ok and ok

    print("\n--- RANDOM SPLITS ---")
    for seed, (tr, va, te) in enumerate(rand_splits):
        ok = audit_split(tr, va, te, smiles, scaffolds, label=f"Fold {seed}")
        all_ok = all_ok and ok

    print("\n" + "=" * 80)
    print(f"OVERALL: {'ALL SPLITS CLEAN - NO LEAKAGE' if all_ok else 'LEAKAGE DETECTED'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
