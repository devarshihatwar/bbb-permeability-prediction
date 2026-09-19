"""
Phase 7: ChemBERTa Fine-Tuning (Single Run, Fold 0)

Fine-tunes seyonec/ChemBERTa-zinc-base-v1 on the BBB_Martins dataset
using the exact Phase 3 Fold 0 scaffold split for matched comparison
with RF (Phase 3) and GNN (Phase 4).

No hyperparameter search, no multiple seeds/folds.
"""
import argparse
import json
import os
import time
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

torch.set_num_threads(4)  # Limit threads to avoid CPU contention
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from transformers import AutoTokenizer, RobertaForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    f1_score, precision_score, recall_score, confusion_matrix,
)

warnings.filterwarnings("ignore")
torch.manual_seed(42)
np.random.seed(42)

# ─── Config ────────────────────────────────────────────────────────────────
MODEL_NAME = "seyonec/ChemBERTa-zinc-base-v1"
DATA_PATH = "data/bbb_martins.csv"
SPLITS_PATH = "data/features/scaffold_splits.pkl"
FOLD = 0
OUTPUT_DIR = "results/chemberta_finetune"
BATCH_SIZE = 16
MAX_EPOCHS = 3
LR = 2e-5
MAX_LENGTH = 512
WARMUP_FRACTION = 0.1
SEED = 42


class BBBDataset(Dataset):
    """SMILES → tokenized → (input_ids, attention_mask, label).
    Uses dynamic padding (pad to longest in batch, not fixed 512).
    """
    def __init__(self, smiles_list, labels, tokenizer):
        self.smiles_list = smiles_list
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        # Tokenize without padding — padding handled by collate_fn
        encoding = self.tokenizer(
            self.smiles_list[idx],
            truncation=True,
            max_length=512,
            return_tensors=None,
        )
        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "labels": torch.tensor(self.labels[idx], dtype=torch.float),
        }


def collate_fn(batch):
    """Dynamic padding: pad to longest sequence in the batch, not 512."""
    from torch.nn.utils.rnn import pad_sequence

    input_ids = [item["input_ids"] for item in batch]
    attention_mask = [item["attention_mask"] for item in batch]
    labels = torch.stack([item["labels"] for item in batch])

    input_ids = [torch.tensor(ids) for ids in input_ids]
    attention_mask = [torch.tensor(mask) for mask in attention_mask]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=1)  # pad_token_id=1
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def evaluate(model, dataloader, device):
    """Return AUROC, AUPRC, Accuracy, F1, Sensitivity, Specificity, TP, FP, TN, FN, probs, preds."""
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs: SequenceClassifierOutput = model(
                input_ids=input_ids, attention_mask=attention_mask
            )
            logits = outputs.logits.squeeze(-1)
            probs = torch.sigmoid(logits)

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    y_true = np.array(all_labels).astype(int)
    y_prob = np.array(all_probs)
    y_pred = (y_prob >= 0.5).astype(int)

    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0

    return {
        "auroc": auroc, "auprc": auprc, "accuracy": acc, "f1": f1,
        "precision": prec, "recall": rec, "sensitivity": sens,
        "specificity": spec, "tp": int(tp), "fp": int(fp),
        "tn": int(tn), "fn": int(fn),
        "probs": y_prob, "preds": y_pred, "labels": y_true,
    }


def get_linear_warmup_decay_scheduler(optimizer, num_warmup_steps, num_training_steps):
    """Linear warmup then linear decay to 0."""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 1.0 - progress)
    return LambdaLR(optimizer, lr_lambda)


def main():
    print("=" * 70)
    print("PHASE 7: ChemBERTa Fine-Tuning (Fold 0, Single Run)")
    print("=" * 70)

    # ─── Load Data and Splits ───────────────────────────────────────────────
    df = pd.read_csv(DATA_PATH)
    smiles_all = df["drug"].tolist()
    labels_all = df["target"].tolist()

    with open(SPLITS_PATH, "rb") as f:
        splits = pickle.load(f)

    train_idx, val_idx, test_idx = splits[FOLD]
    train_smiles = [smiles_all[i] for i in train_idx]
    train_labels = [labels_all[i] for i in train_idx]
    val_smiles = [smiles_all[i] for i in val_idx]
    val_labels = [labels_all[i] for i in val_idx]
    test_smiles = [smiles_all[i] for i in test_idx]
    test_labels = [labels_all[i] for i in test_idx]

    print(f"\nFold 0 Split:")
    print(f"  Train: {len(train_smiles)} | Val: {len(val_smiles)} | Test: {len(test_smiles)}")
    print(f"  Train labels: {pd.Series(train_labels).value_counts().to_dict()}")
    print(f"  Val labels:   {pd.Series(val_labels).value_counts().to_dict()}")
    print(f"  Test labels:  {pd.Series(test_labels).value_counts().to_dict()}")

    # ─── Tokenizer & Model ──────────────────────────────────────────────────
    print(f"\nLoading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Load base RoBERTa, then reconfigure for sequence classification
    model = RobertaForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=1,
        problem_type="single_label_classification",
    )

    device = torch.device("cpu")
    model = model.to(device)
    print(f"Model loaded. Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # ─── Datasets & DataLoaders ────────────────────────────────────────────
    train_dataset = BBBDataset(train_smiles, train_labels, tokenizer)
    val_dataset = BBBDataset(val_smiles, val_labels, tokenizer)
    test_dataset = BBBDataset(test_smiles, test_labels, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # ─── Optimizer & Scheduler ─────────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    num_training_steps = MAX_EPOCHS * len(train_loader)
    num_warmup_steps = int(WARMUP_FRACTION * num_training_steps)
    scheduler = get_linear_warmup_decay_scheduler(optimizer, num_warmup_steps, num_training_steps)
    loss_fn = nn.BCEWithLogitsLoss()

    print(f"\nTraining config:")
    print(f"  Epochs: {MAX_EPOCHS} | Batch: {BATCH_SIZE} | LR: {LR}")
    print(f"  Steps: {num_training_steps} | Warmup: {num_warmup_steps}")

    # ─── Training Loop ─────────────────────────────────────────────────────
    best_val_auroc = -1
    best_epoch = 0
    best_model_state = None
    patience_counter = 0
    patience = 1  # Early stopping after 1 epoch without val AUROC improvement

    train_start = time.time()

    for epoch in range(MAX_EPOCHS):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.squeeze(-1)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_train_loss = epoch_loss / num_batches

        # Validation
        val_metrics = evaluate(model, val_loader, device)
        val_auroc = val_metrics["auroc"]

        epoch_time = time.time() - train_start
        print(f"\nEpoch {epoch+1}/{MAX_EPOCHS} [{epoch_time:.1f}s]:")
        print(f"  Train loss: {avg_train_loss:.4f}")
        print(f"  Val AUROC: {val_auroc:.4f} | Val Acc: {val_metrics['accuracy']:.4f}")
        print(f"  Val F1: {val_metrics['f1']:.4f} | Val Sens: {val_metrics['sensitivity']:.4f} | Val Spec: {val_metrics['specificity']:.4f}")

        # Early stopping on val AUROC
        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_epoch = epoch + 1
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping: no improvement for {patience} epoch(s)")
                break

    train_time = time.time() - train_start
    print(f"\nTraining complete. Best epoch: {best_epoch} (val AUROC: {best_val_auroc:.4f})")
    print(f"Total training time: {train_time:.1f}s")

    # ─── Test Evaluation ────────────────────────────────────────────────────
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    test_metrics = evaluate(model, test_loader, device)

    print(f"\n--- TEST SET METRICS (Fold 0) ---")
    print(f"  AUROC:       {test_metrics['auroc']:.4f}")
    print(f"  AUPRC:       {test_metrics['auprc']:.4f}")
    print(f"  Accuracy:    {test_metrics['accuracy']:.4f}")
    print(f"  F1:          {test_metrics['f1']:.4f}")
    print(f"  Sensitivity: {test_metrics['sensitivity']:.4f}")
    print(f"  Specificity: {test_metrics['specificity']:.4f}")
    print(f"  TP={test_metrics['tp']} FP={test_metrics['fp']} TN={test_metrics['tn']} FN={test_metrics['fn']}")

    # ─── Save Outputs ───────────────────────────────────────────────────────
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    fold_dir = out_dir / f"fold_{FOLD}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    # Save test predictions
    test_pred_df = pd.DataFrame({
        "smiles": test_smiles,
        "true_label": test_metrics["labels"],
        "chemberta_prob": test_metrics["probs"],
        "chemberta_pred": test_metrics["preds"],
    })
    test_pred_df.to_csv(fold_dir / "predictions_test.csv", index=False)
    print(f"\nSaved test predictions: {fold_dir / 'predictions_test.csv'} ({len(test_pred_df)} rows)")

    # Save metrics
    metrics_data = {
        "model": "seyonec/ChemBERTa-zinc-base-v1",
        "fold": FOLD,
        "split": "scaffold",
        "epoch": best_epoch,
        "val_auroc": float(best_val_auroc),
        "test_auroc": float(test_metrics["auroc"]),
        "test_auprc": float(test_metrics["auprc"]),
        "test_accuracy": float(test_metrics["accuracy"]),
        "test_f1": float(test_metrics["f1"]),
        "test_precision": float(test_metrics["precision"]),
        "test_recall": float(test_metrics["recall"]),
        "test_sensitivity": float(test_metrics["sensitivity"]),
        "test_specificity": float(test_metrics["specificity"]),
        "tp": test_metrics["tp"],
        "fp": test_metrics["fp"],
        "tn": test_metrics["tn"],
        "fn": test_metrics["fn"],
        "training_time_s": round(train_time, 1),
        "config": {
            "batch_size": BATCH_SIZE,
            "learning_rate": LR,
            "max_epochs": MAX_EPOCHS,
            "max_length": MAX_LENGTH,
            "seed": SEED,
            "optimizer": "AdamW",
            "warmup_fraction": WARMUP_FRACTION,
            "early_stopping_patience": patience,
        },
    }
    with open(fold_dir / "metrics.json", "w") as f:
        json.dump(metrics_data, f, indent=2)
    print(f"Saved metrics: {fold_dir / 'metrics.json'}")

    # Save summary CSV (compatible with Phase 3/4 format)
    summary_df = pd.DataFrame([{
        "model": "ChemBERTa (seyonec/zinc-base-v1)",
        "AUROC": test_metrics["auroc"],
        "AUPRC": test_metrics["auprc"],
        "Accuracy": test_metrics["accuracy"],
        "F1": test_metrics["f1"],
        "Precision": test_metrics["precision"],
        "Recall": test_metrics["recall"],
        "Sensitivity": test_metrics["sensitivity"],
        "Specificity": test_metrics["specificity"],
        "TP": test_metrics["tp"],
        "FP": test_metrics["fp"],
        "TN": test_metrics["tn"],
        "FN": test_metrics["fn"],
        "fold": FOLD,
        "split": "scaffold",
        "training_time_s": round(train_time, 1),
    }])
    summary_df.to_csv(OUTPUT_DIR + "_metrics.csv" if not OUTPUT_DIR.endswith("/") else OUTPUT_DIR[:-1] + "_metrics.csv", index=False)
    # Save to the canonical results location
    summary_df.to_csv("results/07_chemberta_metrics.csv", index=False)
    print(f"Saved summary: results/07_chemberta_metrics.csv")

    print(f"\n=== Phase 7 Complete ===")


if __name__ == "__main__":
    main()
