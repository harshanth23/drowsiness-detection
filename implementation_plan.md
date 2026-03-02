# Cross-Dataset Generalization Experiments

This adds the full cross-dataset generalization pipeline to the drowsiness detection project.
The goal is to train on a **subset** of datasets and evaluate on **held-out** datasets — proving the model generalizes to unseen subjects, lighting conditions, and recording setups. This is a key contribution for a journal paper.

Two experiments are defined:
- **Exp A**: Train on `nthu_ddd + yawdd` → Test on `uta_rldd + nitymed`
- **Exp B**: Train on `uta_rldd + nitymed` → Test on `nthu_ddd + yawdd`

---

## Proposed Changes

### Experiment Configs

#### [NEW] [exp_A.yaml](file:///d:/Class%20Work/Open%20Lab/Drowsiness%20Detection/experiments/exp_A.yaml)
Config derived from [config.yaml](file:///d:/Class%20Work/Open%20Lab/Drowsiness%20Detection/config.yaml), overriding `data.datasets` and adding `cross.train_datasets` / `cross.test_datasets` fields. Sets a unique `paths.checkpoints` and `paths.plots` sub-directory for this experiment.

#### [NEW] [exp_B.yaml](file:///d:/Class%20Work/Open%20Lab/Drowsiness%20Detection/experiments/exp_B.yaml)
Same as above but dataset split is reversed.

---

### New Scripts

#### [NEW] [cross_train.py](file:///d:/Class%20Work/Open%20Lab/Drowsiness%20Detection/src/cross_train.py)
New training entry-point for cross-dataset experiments.

**Key design decisions:**
- Accepts `--config experiments/exp_A.yaml` (or `exp_B.yaml`)
- Reads `cross.train_datasets` and `cross.test_datasets` from YAML
- Uses the *train-split* of the training datasets for [train](file:///d:/Class%20Work/Open%20Lab/Drowsiness%20Detection/src/train.py#223-386) and [val](file:///d:/Class%20Work/Open%20Lab/Drowsiness%20Detection/src/evaluate.py#263-314) loaders
- Uses the *test-split* of the **test datasets** (unseen) for final evaluation after training
- Internally re-uses [build_dataloaders](file:///d:/Class%20Work/Open%20Lab/Drowsiness%20Detection/src/dataset.py#403-456) from [dataset.py](file:///d:/Class%20Work/Open%20Lab/Drowsiness%20Detection/src/dataset.py) — only passes the correct subset of datasets
- Saves best checkpoint to `results/cross_eval/<exp_name>/checkpoints/best_model.pt`
- Logs CSV epoch history to `results/cross_eval/<exp_name>/logs/train_log.csv`

#### [NEW] [cross_evaluate.py](file:///d:/Class%20Work/Open%20Lab/Drowsiness%20Detection/src/cross_evaluate.py)
Post-training evaluation script that tests a cross-trained model on all specified datasets (seen + unseen), producing:
- Per-dataset confusion matrices & ROC curves
- "Seen vs. Unseen" grouped bar chart (F1, Acc, AUC columns)
- Full metrics CSV for paper tables
- All output goes to `results/cross_eval/<exp_name>/plots/`

CLI: `python src/cross_evaluate.py --config experiments/exp_A.yaml --model <ckpt_path>`

#### [NEW] [plot_generalization.py](file:///d:/Class%20Work/Open%20Lab/Drowsiness%20Detection/src/plot_generalization.py)
Standalone plotting utility. After running both experiments, call this to generate:
1. **"Seen vs. Unseen" grouped bar chart** — F1/AUC comparing in-domain vs. out-of-domain per experiment
2. **Generalization drop heatmap** — matrix of (train partition × test partition) F1 scores
3. Saves to `results/cross_eval/comparison/`

CLI: `python src/plot_generalization.py --exp_A results/cross_eval/exp_A/plots/metrics_summary.csv --exp_B results/cross_eval/exp_B/plots/metrics_summary.csv`

---

## Verification Plan

### Smoke Tests (automated, 1 epoch)

Run these to confirm the scripts execute end-to-end without errors before committing to full training runs:

```powershell
# Activate env
conda activate webenv
cd "D:\Class Work\Open Lab\Drowsiness Detection"

# Exp A smoke test (1 epoch, small batch)
python src/cross_train.py --config experiments/exp_A.yaml --epochs 1 --batch_size 8

# Exp B smoke test (1 epoch, small batch)
python src/cross_train.py --config experiments/exp_B.yaml --epochs 1 --batch_size 8
```

Expected: No errors. Checkpoint saved at `results/cross_eval/exp_A/checkpoints/best_model.pt` (and exp_B equivalent).

### Cross-Evaluate Smoke Test (uses existing checkpoint)

```powershell
# Evaluate Exp A model on unseen test sets
python src/cross_evaluate.py --config experiments/exp_A.yaml --model results/cross_eval/exp_A/checkpoints/best_model.pt

# Generate comparison plots (after both exps)
python src/plot_generalization.py \
  --exp_A results/cross_eval/exp_A/plots/metrics_summary.csv \
  --exp_B results/cross_eval/exp_B/plots/metrics_summary.csv
```

Expected: Confusion matrices, ROC curves, and "seen vs. unseen" bar charts appear in `results/cross_eval/`.

### Manual Visual Check
After smoke tests: open `results/cross_eval/exp_A/plots/` and confirm:
- `confusion_matrix_<dataset>.png` exists for each test dataset
- `roc_curve_<dataset>.png` exists
- `seen_vs_unseen_f1.png` exists (bar chart)
