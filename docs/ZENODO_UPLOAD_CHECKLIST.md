# Zenodo Upload Checklist

## What to Upload

### Core Files (Required)
| File | Purpose |
|---|---|
| `data/bbb_martins.csv` | Source dataset |
| `results/03_ml_baseline_results.csv` | Phase 3 per-fold results |
| `results/03_ml_summary_scaffold.csv` | Phase 3 summary |
| `results/04_dl_chemprop_predictions.csv` | Phase 4 ensemble predictions |
| `results/04_dl_chemprop_metrics.csv` | Phase 4 metrics |
| `results/h2h_predictions.csv` | Matched H2H predictions |
| `results/h2h_rf_predictions.csv` | H2H RF predictions |
| `results/h2h_comparison.csv` | H2H metrics |
| `results/06_error_analysis.csv` | Error analysis with descriptors |
| `results/06_error_analysis_summary.md` | Error analysis summary |
| `results/06_error_analysis_report.md` | Error analysis report |
| `results/06_interpret_error_analysis.csv` | Interpretation analysis |
| `results/baseline_summary.md` | Phase 3 summary |
| `results/final_experiment_summary.md` | FINAL project report |
| `results/feature_importance_*.csv` | RF feature importances (3 files) |
| `src/` (all subdirectories) | All source code |
| `docs/` (all files) | All documentation |
| `figures/` (all PNG files) | Publication figures |
| `README.md` | Project README |
| `LICENSE` | MIT License |
| `CITATION.cff` | Citation file |
| `.gitignore` | Git ignore rules |

### Model Checkpoints (Recommended — Archive or Download)

**H2H model (essential for reproducibility — INCLUDE):**
- `results/chemprop_h2h/model_0/best.pt` (1.3 MB)
- `results/chemprop_h2h/model_0/checkpoints/best-epoch=24-val_roc=0.91.ckpt` (3.8 MB)
- `results/chemprop_h2h/model_0/checkpoints/last.ckpt` (3.8 MB)
- `results/chemprop_h2h/model_0/test_predictions.csv` (296 rows)

**Phase 4 ensemble (optional — LARGE, 60 MB total — consider excluding):**
- `results/chemprop_model/model_0/` through `model_4/` (5 × ~9 MB = ~45 MB)
- Include `config.toml` and `splits.json` but checkpoints can be excluded

**Recommendation:** Include H2H model (10 MB), exclude Phase 4 checkpoints (60 MB). Phase 4 can be regenerated with the documented Chemprop command.

### Log Files (Include for Transparency)
| File | Purpose |
|---|---|
| `results/chemprop_h2h_log.txt` | H2H Chemprop training log |
| `results/chemprop_train_log.txt` | Phase 4 training log |

### Do NOT Upload
| File | Reason |
|---|---|
| `*.xlsx` files | Duplicate of CSV — use CSV as canonical |
| `data/features/scaffold_splits.npy` | Duplicate of `.pkl` |
| `__pycache__/` directories | Python bytecode cache |
| `.venv/`, `venv/` | Virtual environments |
| `*.pyc` files | Compiled Python |
| Temporary scripts (already deleted) | `_*.py` audit scripts |

---

## Recommended Zenodo Deposit Information

### Title
"BBB Prediction Project: Comparative Machine Learning and Graph Neural Network Approaches for Blood-Brain Barrier Permeability Prediction"

### Description
"This project evaluates blood-brain barrier (BBB) permeability prediction using the TDC BBB_Martins dataset (1,975 compounds). Three experimental phases were completed: (1) Phase 3 classical ML baseline using 5-fold scaffold cross-validation (Random Forest on combined Morgan+RDKit features achieved AUROC 0.921±0.024); (2) Phase 4 Chemprop GNN ensemble on SCAFFOLD_BALANCED protocol (AUROC 0.935, 5 models); (3) Phase 5 matched head-to-head comparison on a single Fold 0 scaffold split (296 molecules) where RF (AUROC 0.9031) outperformed a single Chemprop GNN (AUROC 0.867), with complementary error patterns identified via chemical descriptor analysis. All splits use pure positional indices with verified zero leakage. ChemBERTa fine-tuning was environment-verified but not executed due to CPU constraints."

### Keywords
- blood-brain barrier
- permeability prediction
- molecular machine learning
- random forest
- graph neural network
- chemprop
- scaffold splitting
- drug discovery
- reproducible research

### Version
v1.0.0 (final)

### License
MIT

### Author
Devarshi Hatwar (orcid: [REPLACE WITH YOUR ORCID])

### Affiliation
[ADD YOUR AFFILIATION]

---

## Citation Information

The CITATION.cff file is included and will be automatically parsed by Zenodo. GitHub releases will auto-generate a Zenodo DOI with the format: `10.5281/zenodo.XXXXXX`

---

## GitHub Release Connection

1. Tag the final commit: `git tag v1.0.0`
2. Push: `git push origin v1.0.0`
3. Create a GitHub Release titled "v1.0.0 — Final Release"
4. Connect the GitHub repo to Zenodo via https://zenodo.org/account/settings/github/
5. Zenodo will auto-create a new version deposit when the GitHub release is published

---

## File Size Considerations

| Component | Size | Include? |
|---|---|---|
| All CSV/MD/TXT files | ~200 KB total | ✅ Yes |
| Combined features (.npy) | 35 MB | ⚠️ Optional (regenerable) |
| Morgan fingerprints (.npy) | 3.9 MB | ⚠️ Optional (regenerable) |
| RDKit descriptors (.csv) | 3.5 MB | ⚠️ Optional (regenerable) |
| H2H model checkpoint | 3.8 MB | ✅ Yes (needed for reproducibility) |
| Phase 4 model checkpoints | 45 MB × 5 | ❌ No (60 MB total) |
| Figures (6 PNGs) | ~1 MB total | ✅ Yes |
| **Total (core only)** | **~25 MB** | |
| **Total (with all checkpoints)** | **~105 MB** | |

**Zenodo recommends:** Keep the deposit under 50 MB for optimal browser experience. The "core only" package (~25 MB) is well within limits.

---

## Pre-upload Verification Checklist

- [ ] `CITATION.cff` has correct author name and placeholder ORCID
- [ ] `README.md` references are correct
- [ ] All numerical results in `final_experiment_summary.md` match source files
- [ ] No secrets or API keys in any file
- [ ] `LICENSE` file is present
- [ ] `.gitignore` is present
- [ ] Phase 3 results unchanged (35 rows in `03_ml_baseline_results.csv`)
- [ ] H2H predictions unchanged (296 rows in `h2h_predictions.csv`)
- [ ] Error analysis consistent with H2H predictions (296 rows in `06_error_analysis.csv`)
- [ ] All 6 figures are in `figures/`
- [ ] `docs/` directory is complete
- [ ] Source code is complete and runnable
