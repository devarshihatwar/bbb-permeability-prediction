# Phase 7: ChemBERTa Implementation Plan

## Status: Environment Verified — No Training Performed

## 1. Model Selection

**Model:** `seyonec/ChemBERTa-zinc-base-v1`

| Property | Value |
|---|---|
| Architecture | RoBERTa (6 layers, 12 heads, 768 hidden, 3072 intermediate) |
| Vocab size | 767 |
| Max position embeddings | 514 |
| Tokenizer | RobertaTokenizer (Byte-Pair Encoding) |
| Model size | 170.5 MB (pytorch_model.bin) |
| Downloads | 138,000+ on HuggingFace |
| License | Apache 2.0 (MIT permissive) |
| Pretraining | 10M molecules from ZINC, masked language modeling |

**Why this model (not DeepChem/ChemBERTa-77M-MTR):**
- `seyonec/ChemBERTa-zinc-base-v1` is a base RoBERTa MLM — loads cleanly as `RobertaForSequenceClassification`
- `DeepChem/ChemBERTa-77M-MTR` has 200 regression output heads (preconfigured for MTR task) — requires config surgery to repurpose for binary classification
- `seyonasteri/chemberta-base` is gated (401 error — requires authentication)
- `seyonec/ChemBERTa-zinc-base-v1` is the most downloaded public ChemBERTa model

## 2. Environment Verification

| Component | Status |
|---|---|
| Python | 3.11.15 ✅ |
| transformers | 5.17.0 ✅ |
| torch | 2.14.0+cpu ✅ (no CUDA on this machine) |
| rdkit | 2022.x ✅ |
| huggingface_hub | 1.24.0 ✅ |
| tokenizers | 0.23.1 ✅ |
| accelerate | NOT installed (not needed — PyTorch native training) |
| datasets | NOT installed (not needed — manual data loading) |

**Key constraint:** CPU-only training. No GPU available. `torch.cuda.is_available()` returns False.

## 3. Dataset Compatibility

**Source file:** `data/bbb_martins.csv`
- Columns: `drug_id` (int), `drug` (SMILES string), `target` (0/1 binary)
- 1,975 molecules, no NaN, all SMILES parseable by RDKit
- Tokenizer max length: 512
- Tokenized length stats: min=3, max=310, mean=48.6, 95th percentile=88
- **0 molecules exceed 512 tokens** → no truncation issues

## 4. Split Approach

**Reuse existing Fold 0 scaffold split** from `data/features/scaffold_splits.pkl`:
- Train: 1,382 molecules (25% non-permeable)
- Val:   297 molecules (25% non-permeable)
- Test:  296 molecules (20% non-permeable)

This is the **exact same Fold 0** used in Phase 4 (Chemprop) and the error analysis in Phase 5. This ensures the h2h comparison is perfectly matched.

**Split indices are positional (0–1974)** — no pandas index mixing (bug already fixed in Phase 3).

## 5. Minimal Fine-Tuning Configuration

```python
# Single fold, single epoch, no hyperparameter search
Model:       RobertaForSequenceClassification.from_pretrained(
               "seyonec/ChemBERTa-zinc-base-v1", num_labels=1)
Tokenizer:   AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
Max length:  512
Batch size:  16 (CPU memory constrained)
Epochs:      3 (early stopping on val AUROC, patience=1)
LR:          2e-5 (standard for BERT fine-tuning)
Loss:        BCEWithLogitsLoss (binary classification)
Optimizer:   AdamW (default betas, weight_decay=0.01)
Scheduler:   Linear warmup (10% of steps) then linear decay
Seed:        42 (match existing pipeline)
Class weight: None (use raw probabilities; class imbalance is 3:1, manageable)
```

**Training data order:** Train on 1,382 molecules, validate on 297, test on 296 (exact Fold 0 split).

## 6. Expected Runtime

| Phase | CPU Estimation |
|---|---|
| Model download (170 MB) | ~5 seconds |
| Tokenization (1,975 SMILES) | ~3 seconds |
| Epoch 1 (train: 1,382 / batch 16 = ~87 steps) | ~5 minutes |
| Epoch 2 | ~5 minutes |
| Epoch 3 | ~5 minutes |
| Validation (297 molecules) | ~30 seconds |
| Test inference (296 molecules) | ~1 minute |
| **Total (3 epochs)** | **~15–18 minutes** |

*No hyperparameter search — single configuration, 3 epochs max with early stopping.*

## 7. Exact Training Command

```bash
cd C:/Users/vaish/BBB_project
python src/07_chemberta/finetune_chemberta.py \
  --model_name seyonec/ChemBERTa-zinc-base-v1 \
  --data_path data/bbb_martins.csv \
  --splits_path data/features/scaffold_splits.pkl \
  --fold 0 \
  --output_dir results/chemberta_finetune \
  --batch_size 16 \
  --max_epochs 3 \
  --lr 2e-5 \
  --max_length 512 \
  --seed 42
```

## 8. Output Files (Planned)

```
results/chemberta_finetune/
├── fold_0/
│   ├── checkpoint_epoch_1/       # model weights after epoch 1
│   ├── checkpoint_epoch_2/       # model weights after epoch 2
│   ├── checkpoint_best/          # best model (by val AUROC)
│   ├── predictions_test.csv      # test set predictions (SMILES, true, prob, pred)
│   └── metrics.json              # per-epoch metrics
└── chemberta_metrics.csv         # fold-0 summary metrics (matches Phases 3-6 format)
```

The `predictions_test.csv` will follow the same schema as `results/06_error_analysis.csv` for compatibility with the Phase 5 analysis pipeline.

## 9. What This Phase Does NOT Do

- ❌ Does NOT train (environment verification + planning only)
- ❌ Does NOT modify existing Phase 3/4 results
- ❌ Does NOT perform hyperparameter search
- ❌ Does NOT install additional packages (transformers already installed)
- ❌ Does NOT attempt GPU acceleration (no CUDA available)

## 10. Readiness Checklist

- [x] Model `seyonec/ChemBERTa-zinc-base-v1` loads and tokenizes SMILES correctly
- [x] `RobertaForSequenceClassification` can be initialized from checkpoint config
- [x] Dataset `data/bbb_martins.csv` is compatible (SMILES column = `drug`, label = `target`)
- [x] Fold 0 split is reusable (positional indices 0–1974)
- [x] `transformers` and `torch` are installed in the environment
- [x] No molecules exceed 512-token limit
- [ ] Training script `finetune_chemberta.py` not yet created — **pending approval to proceed**
