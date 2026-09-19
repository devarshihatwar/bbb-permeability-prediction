"""
Featurization module for the BBB Martins dataset.

Converts SMILES strings into numeric feature vectors using RDKit:
  1. Morgan/ECFP fingerprints (binary, fixed-length)
  2. RDKit molecular descriptors (physicochemical properties)
  3. Combined (fingerprints + descriptors)

Usage:
    python src/03_ml/featurize.py --input data/bbb_martins.csv --output data/features/
"""
import argparse
import os
import warnings

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def load_data(path: str) -> pd.DataFrame:
    """Load the cleaned BBB dataset."""
    df = pd.read_csv(path)
    # Ensure column names match TDC schema
    assert "drug" in df.columns and "target" in df.columns, (
        f"Expected 'drug' and 'target' columns, got: {df.columns.tolist()}"
    )
    # Drop rows with invalid SMILES or missing labels
    df = df.dropna(subset=["drug", "target"]).reset_index(drop=True)
    df["target"] = df["target"].astype(int)
    return df


def smiles_to_mol(smiles: str):
    """Convert SMILES to RDKit Mol, return None if invalid."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol
    except Exception:
        return None


def morgan_fp(mol, n_bits: int = 2048, radius: int = 2) -> np.ndarray:
    """Generate a Morgan (ECFP4) fingerprint as a numpy array."""
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((0,), dtype=np.uint8)
    AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def rdkit_descriptors(mol) -> dict:
    """Compute all available RDKit descriptors."""
    # Use the curated list of descriptor names that RDKit 2026 provides
    desc_names = [d[0] for d in Descriptors._descList]
    props = {}
    for name in desc_names:
        try:
            fn = getattr(Descriptors, name)
            val = fn(mol)
            props[name] = val if val is not None else 0.0
        except Exception:
            props[name] = 0.0
    return props


def featurize(
    smiles_list,
    fp_bits: int = 2048,
    fp_radius: int = 2,
):
    """
    Return (morgan_fps, descriptor_df, combined) for a list of SMILES.
    - morgan_fps: numpy array of shape (n, fp_bits)
    - descriptor_df: pandas DataFrame of RDKit descriptors
    - combined: numpy array of shape (n, fp_bits + n_descriptors)
    """
    fps = []
    desc_list = []
    valid_indices = []

    # Use the original Morgan fingerprint API (works in this RDKit version)
    # and silence the deprecation warnings
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")

    for i, smi in enumerate(smiles_list):
        mol = smiles_to_mol(smi)
        if mol is None:
            fps.append(np.zeros(fp_bits, dtype=np.uint8))
            desc_list.append({})
            continue
        valid_indices.append(i)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, fp_radius, nBits=fp_bits)
        arr = np.zeros((0,), dtype=np.uint8)
        AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
        fps.append(arr)
        desc_list.append(rdkit_descriptors(mol))

    morgan_arr = np.array(fps)

    # Pad descriptor dicts to consistent keys
    desc_df = pd.DataFrame(desc_list).fillna(0.0)
    # Convert to numeric
    for col in desc_df.columns:
        desc_df[col] = pd.to_numeric(desc_df[col], errors="coerce").fillna(0.0)

    combined = np.hstack([morgan_arr, desc_df.values])
    return morgan_arr, desc_df, combined


def scaffold_split(
    smiles_list,
    n_total: int,
    frac_train: float = 0.7,
    frac_val: float = 0.15,
    seed: int = 42,
):
    """
    Split by molecular scaffold (Murcko scaffold).

    Returns three lists of POSITIONAL indices (0-based integers into the
    original smiles_list). No pandas index mixing — pure positional indices.
    """
    from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol

    scaffold_to_positions = {}
    for pos, smi in enumerate(smiles_list):
        mol = smiles_to_mol(smi)
        if mol is None:
            continue
        try:
            scaffold = GetScaffoldForMol(mol)
            scaffold_smiles = Chem.MolToSmiles(scaffold) if scaffold is not None else ""
        except Exception:
            scaffold_smiles = ""
        scaffold_to_positions.setdefault(scaffold_smiles, []).append(pos)

    # Shuffle scaffold groups by seed
    rng = np.random.RandomState(seed)
    scaffold_groups = list(scaffold_to_positions.values())
    rng.shuffle(scaffold_groups)

    train_idx, val_idx, test_idx = [], [], []
    n_train = int(n_total * frac_train)
    n_val = int(n_total * frac_val)

    current = 0
    for group in scaffold_groups:
        if current < n_train:
            train_idx.extend(group)
            current += len(group)
        elif current < n_train + n_val:
            val_idx.extend(group)
            current += len(group)
        else:
            test_idx.extend(group)

    return sorted(train_idx), sorted(val_idx), sorted(test_idx)


def random_split(
    n_total: int,
    frac_train: float = 0.7,
    frac_val: float = 0.15,
    seed: int = 42,
):
    """
    Simple random split.

    Returns three lists of POSITIONAL indices (0-based integers).
    No pandas index mixing — pure positional indices.
    """
    rng = np.random.RandomState(seed)
    all_indices = list(range(n_total))
    rng.shuffle(all_indices)

    n_train = int(n_total * frac_train)
    n_val = int(n_total * frac_val)

    train_idx = sorted(all_indices[:n_train])
    val_idx = sorted(all_indices[n_train : n_train + n_val])
    test_idx = sorted(all_indices[n_train + n_val :])
    return train_idx, val_idx, test_idx


def make_splits(n_total: int, smiles_list, split_type="scaffold", n_seeds=5):
    """Yield (train_pos, val_pos, test_pos) tuples of positional indices."""
    for seed in range(n_seeds):
        if split_type == "scaffold":
            tr, va, te = scaffold_split(smiles_list, n_total, seed=seed)
        else:
            tr, va, te = random_split(n_total, seed=seed)
        yield (tr, va, te)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Featurize the BBB dataset.")
    parser.add_argument("--input", default="data/bbb_martins.csv")
    parser.add_argument("--output", default="data/features")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    df = load_data(args.input)
    print(f"Loaded {len(df)} compounds from {args.input}")

    morgan, descs, combined = featurize(df["drug"].tolist())
    print(f"Morgan fingerprints: {morgan.shape}")
    print(f"RDKit descriptors:   {descs.shape}")
    print(f"Combined features:   {combined.shape}")

    np.save(os.path.join(args.output, "morgan_fps.npy"), morgan)
    descs.to_csv(os.path.join(args.output, "rdkit_descriptors.csv"), index=False)
    np.save(os.path.join(args.output, "combined_features.npy"), combined)

    # Save split indices as positional integer indices
    smiles_list = df["drug"].tolist()
    n = len(df)

    splits = list(make_splits(n, smiles_list, split_type="scaffold", n_seeds=5))
    import pickle
    with open(os.path.join(args.output, "scaffold_splits.pkl"), "wb") as f:
        pickle.dump(splits, f)

    splits_random = list(make_splits(n, smiles_list, split_type="random", n_seeds=5))
    with open(os.path.join(args.output, "random_splits.pkl"), "wb") as f:
        pickle.dump(splits_random, f)

    print(f"Saved features to {args.output}/")
    print(f"Saved 5 scaffold splits + 5 random splits (as .pkl)")
    print(f"Label balance: {df['target'].value_counts().to_dict()}")
