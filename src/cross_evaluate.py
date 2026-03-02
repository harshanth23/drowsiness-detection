# Allow `python src/cross_evaluate.py` to resolve `from src.*` imports
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..")))
"""
src/cross_evaluate.py — Cross-Dataset Generalization Evaluation
----------------------------------------------------------------
Loads a model trained by cross_train.py and evaluates it on BOTH the seen
(training) datasets AND the unseen (test) datasets, generating:

  • Per-dataset confusion matrix & ROC curve
  • "Seen vs. Unseen" grouped bar chart (F1 / Accuracy / AUC)
  • Full metrics CSV for paper tables

Usage:
    conda activate webenv
    python src/cross_evaluate.py --config experiments/exp_A.yaml \\
                                 --model  results/cross_eval/exp_A/checkpoints/best_model.pt

Outputs → paths.plots directory from the config  (e.g. results/cross_eval/exp_A/plots/)
"""

import argparse
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score, roc_curve,
)

from src.dataset import (
    DrowsinessClipDataset, _eval_transform, build_records,
    _DATASET_CONFIGS,
)
from src.model import DrowsinessModel
from src.utils import get_device, load_checkpoint, set_seed

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x   # noqa: E731


_CLASS_NAMES = ["Alert", "Drowsy"]


# ─────────────────────────────────────────────────────────────────────────────
# Inference helper  (with tqdm progress bar)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(
    model:          nn.Module,
    loader,
    device:         torch.device,
    use_amp:        bool = True,
    use_landmarks:  bool = True,
    desc:           str  = "Evaluating",
) -> Tuple[List[int], List[int], List[float]]:
    """Returns (all_labels, all_preds, all_probs_cls1)."""
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    pbar = tqdm(loader, desc=f"  {desc}", unit="batch", leave=False, ncols=100)
    for clips, lms, labels in pbar:
        clips = clips.to(device, non_blocking=True)
        lms   = lms.to(device, non_blocking=True) if use_landmarks else None

        with autocast(enabled=use_amp):
            logits = model(clips, lms)
            probs  = torch.softmax(logits, dim=1)

        preds = probs.argmax(dim=1).cpu().tolist()
        all_labels.extend(labels.tolist())
        all_preds.extend(preds)
        all_probs.extend(probs[:, 1].cpu().tolist())

    pbar.close()
    return all_labels, all_preds, all_probs


# ─────────────────────────────────────────────────────────────────────────────
# Plot helpers
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    labels:    List[int],
    preds:     List[int],
    save_path: str,
    title:     str = "Confusion Matrix",
) -> None:
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    present = sorted(set(labels) | set(preds))
    cm      = confusion_matrix(labels, preds, labels=present)
    names   = [_CLASS_NAMES[i] for i in present]

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set(
        xticks=range(len(names)), yticks=range(len(names)),
        xticklabels=names, yticklabels=names,
        xlabel="Predicted", ylabel="True", title=title,
    )
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  [eval] confusion matrix → {save_path}")


def plot_roc_curve(
    labels:    List[int],
    probs:     List[float],
    save_path: str,
    title:     str = "ROC Curve",
) -> float:
    """Saves ROC curve and returns AUC (or nan if undefined)."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    try:
        auc = roc_auc_score(labels, probs)
        fpr, tpr, _ = roc_curve(labels, probs)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.3f}")
        plt.plot([0, 1], [0, 1], "k--", lw=1)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"  [eval] ROC AUC={auc:.3f} → {save_path}")
        return auc
    except Exception as e:
        print(f"  [eval] ROC skipped: {e}")
        return float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# Per-dataset evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_dataset(
    ds_name:   str,
    split_tag: str,        # "Seen" or "Unseen"
    model:     nn.Module,
    config:    dict,
    device:    torch.device,
    plots_dir: str,
    use_amp:   bool = True,
) -> Dict:
    data_cfg = config["data"]
    use_lm   = data_cfg.get("use_landmarks", True)
    uta_fold = data_cfg.get("uta_fold", 0)

    records = build_records(
        clips_root     = data_cfg["clips_root"],
        landmarks_root = data_cfg.get("landmarks_root", "dataset/landmarks_cropped"),
        splits_dir     = data_cfg.get("splits_dir", "data/splits"),
        split          = "test",
        use_landmarks  = use_lm,
        uta_fold       = uta_fold,
        datasets       = [ds_name],
    )

    if not records:
        print(f"  [{ds_name}] no test records found, skipping.")
        return {}

    dataset = DrowsinessClipDataset(records, _eval_transform(), use_landmarks=use_lm)
    loader  = DataLoader(
        dataset, batch_size=8, shuffle=False,
        num_workers=config["training"].get("num_workers", 0),
        pin_memory=True,
    )

    labels, preds, probs = run_inference(
        model, loader, device, use_amp, use_lm,
        desc=f"[{split_tag}] {ds_name}",
    )

    acc  = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, average="macro", zero_division=0)
    rec  = recall_score   (labels, preds, average="macro", zero_division=0)
    f1   = f1_score       (labels, preds, average="macro", zero_division=0)

    print(f"\n  [{split_tag}] {ds_name}  acc={acc:.3f}  prec={prec:.3f}  rec={rec:.3f}  F1={f1:.3f}")

    unique = sorted(set(labels) | set(preds))
    if len(unique) == 1:
        print(classification_report(labels, preds, labels=unique,
                                    target_names=[_CLASS_NAMES[unique[0]]], zero_division=0))
    else:
        print(classification_report(labels, preds, target_names=_CLASS_NAMES, zero_division=0))

    ds_plots = os.path.join(plots_dir, ds_name)
    plot_confusion_matrix(labels, preds,
                          os.path.join(ds_plots, "confusion_matrix.png"),
                          title=f"Confusion Matrix — {ds_name} [{split_tag}]")
    auc = plot_roc_curve(labels, probs,
                         os.path.join(ds_plots, "roc_curve.png"),
                         title=f"ROC Curve — {ds_name} [{split_tag}]")

    return {
        "dataset": ds_name, "split_tag": split_tag, "n_clips": len(labels),
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Seen vs. Unseen bar chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_seen_vs_unseen(rows: List[Dict], save_path: str, exp_name: str) -> None:
    """Grouped bar chart: one bar per dataset, coloured by Seen / Unseen."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    seen_rows   = [r for r in rows if r["split_tag"] == "Seen"]
    unseen_rows = [r for r in rows if r["split_tag"] == "Unseen"]

    metrics  = ["accuracy", "f1", "auc"]
    m_labels = ["Accuracy", "Macro F1", "ROC AUC"]
    colors   = {"Seen": "#4C72B0", "Unseen": "#DD8452"}

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5), sharey=False)
    fig.suptitle(f"Seen vs. Unseen Generalisation — {exp_name}", fontsize=13, fontweight="bold")

    for ax, metric, mlabel in zip(axes, metrics, m_labels):
        all_rows   = sorted(rows, key=lambda r: r["dataset"])
        ds_names   = [r["dataset"] for r in all_rows]
        seen_vals  = {r["dataset"]: r.get(metric, 0) for r in seen_rows}
        unseen_vals= {r["dataset"]: r.get(metric, 0) for r in unseen_rows}

        x     = np.arange(len(ds_names))
        width = 0.35

        seen_bar  = [seen_vals.get(d,   float("nan")) for d in ds_names]
        unseen_bar= [unseen_vals.get(d, float("nan")) for d in ds_names]

        b1 = ax.bar(x - width/2, seen_bar,   width, label="Seen",   color=colors["Seen"],   alpha=0.85)
        b2 = ax.bar(x + width/2, unseen_bar, width, label="Unseen", color=colors["Unseen"], alpha=0.85)

        # Value labels on bars
        for bar in list(b1) + list(b2):
            h = bar.get_height()
            if not np.isnan(h):
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                        f"{h:.2f}", ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(ds_names, rotation=20, ha="right", fontsize=9)
        ax.set_ylim(0, 1.12)
        ax.set_ylabel(mlabel)
        ax.set_title(mlabel)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  [eval] Seen vs. Unseen chart → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation function
# ─────────────────────────────────────────────────────────────────────────────

def cross_evaluate(config: dict, model_path: Optional[str] = None) -> None:
    device = get_device()
    set_seed(config["training"].get("seed", 42))

    cross_cfg  = config.get("cross", {})
    exp_name   = cross_cfg.get("name", "cross_exp")
    train_ds   = cross_cfg.get("train_datasets", [])
    test_ds    = cross_cfg.get("test_datasets",  [])

    print(f"\n{'='*65}")
    print(f"  Cross-Evaluate : {exp_name}")
    print(f"  Seen  datasets : {train_ds}")
    print(f"  Unseen datasets: {test_ds}")
    print(f"{'='*65}")

    # ── Load model ────────────────────────────────────────────────────────────
    print("\n[eval] Loading model …")
    model    = DrowsinessModel.from_config(config).to(device)
    ckpt_path= model_path or config["paths"].get("best_model", "best_model.pt")
    load_checkpoint(ckpt_path, model, device=str(device))
    use_amp  = config["training"].get("use_amp", True) and device.type == "cuda"

    plots_dir = config["paths"].get("plots", f"results/cross_eval/{exp_name}/plots")
    all_rows: List[Dict] = []

    # ── Seen datasets ─────────────────────────────────────────────────────────
    total = len(train_ds) + len(test_ds)
    idx   = 0
    for ds_name in train_ds:
        idx += 1
        print(f"\n[eval] Progress: {idx}/{total} — {ds_name}  [SEEN]")
        row = evaluate_dataset(ds_name, "Seen", model, config, device, plots_dir, use_amp)
        if row:
            all_rows.append(row)

    # ── Unseen datasets ───────────────────────────────────────────────────────
    for ds_name in test_ds:
        idx += 1
        print(f"\n[eval] Progress: {idx}/{total} — {ds_name}  [UNSEEN ← held-out]")
        row = evaluate_dataset(ds_name, "Unseen", model, config, device, plots_dir, use_amp)
        if row:
            all_rows.append(row)

    # ── Metrics CSV ───────────────────────────────────────────────────────────
    if all_rows:
        import csv
        csv_path = os.path.join(plots_dir, "metrics_summary.csv")
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\n  [eval] metrics summary → {csv_path}")

        # Console table
        print("\n" + "=" * 75)
        print(f"{'Dataset':<14} {'Tag':<8} {'N':>6} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'AUC':>6}")
        print("-" * 75)
        for r in all_rows:
            tag = "✓ Seen  " if r["split_tag"] == "Seen" else "✗ Unseen"
            print(f"{r['dataset']:<14} {tag:<8} {r['n_clips']:>6} "
                  f"{r['accuracy']:>6.3f} {r['precision']:>6.3f} "
                  f"{r['recall']:>6.3f} {r['f1']:>6.3f} "
                  f"{r.get('auc', float('nan')):>6.3f}")
        print("=" * 75)

        # ── Seen vs. Unseen bar chart ─────────────────────────────────────────
        chart_path = os.path.join(plots_dir, "seen_vs_unseen.png")
        plot_seen_vs_unseen(all_rows, chart_path, exp_name)

    print("\n✅ Cross-evaluation complete.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Cross-Dataset Generalization Evaluation")
    p.add_argument("--config", type=str, required=True,
                   help="Experiment YAML (e.g. experiments/exp_A.yaml)")
    p.add_argument("--model",  type=str, default=None,
                   help="Path to model checkpoint (overrides config)")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    cross_evaluate(cfg, args.model)
