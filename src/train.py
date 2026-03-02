# Allow `python src/train.py` to resolve `from src.*` imports
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..")))
"""
src/train.py — Training Loop
------------------------------
Usage:
    conda activate webenv
    python src/train.py                          # uses config.yaml
    python src/train.py --config config.yaml --epochs 30 --batch_size 32

Key features per plan.md:
  • Adam + cosine LR decay
  • Mixed-precision (torch.cuda.amp)
  • Weighted cross-entropy or Focal loss to handle class imbalance
  • Per-epoch validation; checkpoint saved on best val-F1
  • Early stopping (patience configurable)
  • Gradient clipping
  • TensorBoard + CSV logging
"""

import argparse
import csv
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
import yaml
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm

from src.dataset import build_dataloaders
from src.model import DrowsinessModel
from src.utils import (
    TBWriter, count_parameters, get_device, load_checkpoint,
    plot_loss_curves, plot_metric_curves, save_checkpoint, set_seed,
)


# ─────────────────────────────────────────────────────────────────────────────
# Focal Loss
# ─────────────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """Multi-class Focal Loss (Lin et al. 2017)."""

    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma  = gamma
        self.weight = weight   # per-class weight tensor

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = nn.functional.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Class-weight helper (fast — no image I/O, reads labels from clip records)
# ─────────────────────────────────────────────────────────────────────────────

def _fast_class_weights(dataset, num_classes: int, device: torch.device) -> torch.Tensor:
    """
    Count labels directly from the dataset's ClipRecord list (no image I/O).
    Each record is (clip_dir, label_int, lm_dir).
    """
    counts = torch.zeros(num_classes)
    for _, label, _ in dataset.records:
        counts[label] += 1
    weights = counts.sum() / (num_classes * counts.clamp(min=1))
    print(f"  [loss] samples per class : {counts.long().tolist()}")
    print(f"  [loss] class weights     : {[f'{w:.3f}' for w in weights.tolist()]}")
    return weights.to(device)


# ─────────────────────────────────────────────────────────────────────────────
# One training epoch
# ─────────────────────────────────────────────────────────────────────────────

def _train_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler:    GradScaler,
    device:    torch.device,
    use_amp:   bool,
    grad_clip: float,
    use_landmarks: bool,
    epoch: int = 0,
    total_epochs: int = 0,
) -> Tuple[float, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    pbar = tqdm(
        loader,
        desc=f"  Train [{epoch}/{total_epochs}]",
        unit="batch",
        leave=True,
        ncols=100,
    )

    for clips, lms, labels in pbar:
        clips  = clips.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        lms    = lms.to(device, non_blocking=True) if use_landmarks else None

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=use_amp):
            logits = model(clips, lms)
            loss   = criterion(logits, labels)

        scaler.scale(loss).backward()

        if grad_clip > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()

        batch_loss  = loss.item()
        total_loss += batch_loss * labels.size(0)
        preds       = logits.argmax(dim=1)
        batch_corr  = (preds == labels).sum().item()
        correct    += batch_corr
        total      += labels.size(0)

        pbar.set_postfix(
            loss=f"{batch_loss:.4f}",
            acc=f"{correct / total:.3f}",
            gpu=f"{torch.cuda.memory_reserved() / 1e9:.1f}GB" if device.type == 'cuda' else "",
        )

    pbar.close()
    return total_loss / total, correct / total


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation on one split
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _eval_epoch(
    model:         nn.Module,
    loader,
    criterion:     nn.Module,
    device:        torch.device,
    use_amp:       bool,
    use_landmarks: bool,
    split:         str = "Val",
) -> Tuple[float, float, float, float, float]:
    """Returns (loss, acc, precision, recall, f1)."""
    model.eval()
    total_loss, total = 0.0, 0
    all_preds, all_labels = [], []

    pbar = tqdm(
        loader,
        desc=f"  {split:>5}             ",
        unit="batch",
        leave=True,
        ncols=100,
    )

    for clips, lms, labels in pbar:
        clips  = clips.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        lms    = lms.to(device, non_blocking=True) if use_landmarks else None

        with autocast(enabled=use_amp):
            logits = model(clips, lms)
            loss   = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        total      += labels.size(0)
        all_preds.extend(logits.argmax(dim=1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    pbar.close()
    acc  = sum(p == l for p, l in zip(all_preds, all_labels)) / total
    prec = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    rec  = recall_score   (all_labels, all_preds, average="macro", zero_division=0)
    f1   = f1_score       (all_labels, all_preds, average="macro", zero_division=0)
    return total_loss / total, acc, prec, rec, f1


# ─────────────────────────────────────────────────────────────────────────────
# CSV logger
# ─────────────────────────────────────────────────────────────────────────────

class CSVLogger:
    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        self._file = open(path, "w", newline="")
        self._writer = None

    def log(self, row: dict):
        if self._writer is None:
            self._writer = csv.DictWriter(self._file, fieldnames=row.keys())
            self._writer.writeheader()
        self._writer.writerow(row)
        self._file.flush()

    def close(self):
        self._file.close()


# ─────────────────────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────────────────────

def train(config: dict) -> None:
    device = get_device()

    # ── Reproducibility ────────────────────────────────────────────────────────
    seed = config["training"].get("seed", 42)
    set_seed(seed)

    # ── Data ──────────────────────────────────────────────────────────────────
    print("\n[data] Building DataLoaders …")
    data_cfg = config["data"]
    train_cfg = config["training"]

    loader_cfg = {
        **data_cfg,
        "batch_size":  train_cfg["batch_size"],
        "num_workers": train_cfg["num_workers"],
    }
    loaders = build_dataloaders(loader_cfg)
    use_lm  = data_cfg.get("use_landmarks", True)

    # ── Model ─────────────────────────────────────────────────────────────────
    print("\n[model] Building model …")
    model = DrowsinessModel.from_config(config).to(device)
    print(f"  Trainable parameters: {count_parameters(model):,}")

    # ── Loss ──────────────────────────────────────────────────────────────────
    num_classes = config["model"]["num_classes"]
    use_focal   = train_cfg.get("use_focal_loss", False)

    print("\n[loss] Computing class weights (fast scan) …")
    class_weights = _fast_class_weights(loaders["train"].dataset, num_classes, device)

    if use_focal:
        criterion = FocalLoss(gamma=train_cfg.get("focal_gamma", 2.0), weight=class_weights)
        print("  Using Focal loss")
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print("  Using Weighted CrossEntropy loss")

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg.get("weight_decay", 1e-4),
    )

    # ── Scheduler ─────────────────────────────────────────────────────────────
    sched_cfg = config.get("scheduler", {})
    sched_type = sched_cfg.get("type", "cosine").lower()
    if sched_type == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max  = sched_cfg.get("T_max", train_cfg["epochs"]),
            eta_min= sched_cfg.get("eta_min", 1e-6),
        )
    elif sched_type == "step":
        scheduler = StepLR(optimizer, step_size=sched_cfg.get("step_size", 10), gamma=0.5)
    else:
        scheduler = None   # constant LR

    # ── AMP ───────────────────────────────────────────────────────────────────
    use_amp    = train_cfg.get("use_amp", True) and device.type == "cuda"
    scaler     = GradScaler(enabled=use_amp)
    grad_clip  = train_cfg.get("grad_clip", 1.0)

    # ── Logging ───────────────────────────────────────────────────────────────
    paths = config.get("paths", {})
    tb    = TBWriter(paths.get("logs", "results/logs"))
    csv_l = CSVLogger(os.path.join(paths.get("logs", "results/logs"), "train_log.csv"))
    Path(paths.get("plots", "results/plots")).mkdir(parents=True, exist_ok=True)
    Path(paths.get("checkpoints", "results/checkpoints")).mkdir(parents=True, exist_ok=True)

    # ── Training loop ─────────────────────────────────────────────────────────
    epochs            = train_cfg["epochs"]
    patience          = train_cfg.get("early_stop_patience", 10)
    best_val_f1       = 0.0
    no_improve_count  = 0
    train_losses: List[float] = []
    val_losses:   List[float] = []
    val_f1s:      List[float] = []

    print(f"\n[train] Starting training for {epochs} epochs …\n")

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()

        tr_loss, tr_acc = _train_epoch(
            model, loaders["train"], criterion, optimizer, scaler,
            device, use_amp, grad_clip, use_lm,
            epoch=epoch, total_epochs=epochs,
        )
        val_loss, val_acc, val_prec, val_rec, val_f1 = _eval_epoch(
            model, loaders["val"], criterion, device, use_amp, use_lm,
            split="Val",
        )

        if scheduler is not None:
            scheduler.step()

        elapsed = time.perf_counter() - t0
        lr_now  = optimizer.param_groups[0]["lr"]

        tqdm.write(
            f"Epoch {epoch}/{epochs}  "
            f"train_acc={tr_acc:.3f}  train_loss={tr_loss:.4f}  "
            f"val_acc={val_acc:.3f}  val_loss={val_loss:.4f}  val_f1={val_f1:.3f}  "
            f"lr={lr_now:.6f}  [{elapsed:.1f}s]"
        )

        # TensorBoard
        tb.add_scalars("loss", {"train": tr_loss, "val": val_loss}, epoch)
        tb.add_scalars("acc",  {"train": tr_acc, "val": val_acc},   epoch)
        tb.add_scalar("val/f1",        val_f1,   epoch)
        tb.add_scalar("val/precision", val_prec, epoch)
        tb.add_scalar("val/recall",    val_rec,  epoch)
        tb.add_scalar("lr",            lr_now,   epoch)

        # CSV
        csv_l.log({
            "epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
            "val_loss": val_loss, "val_acc": val_acc,
            "val_f1": val_f1, "val_prec": val_prec, "val_rec": val_rec,
            "lr": lr_now,
        })

        train_losses.append(tr_loss)
        val_losses.append(val_loss)
        val_f1s.append(val_f1)

        # Checkpoint on improvement
        if val_f1 > best_val_f1:
            best_val_f1      = val_f1
            no_improve_count = 0
            save_checkpoint(
                model, optimizer, epoch, val_f1,
                paths.get("best_model", "results/checkpoints/best_model.pt"),
                scheduler=scheduler, scaler=scaler,
            )
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                tqdm.write(f"\n[early stop] No val-F1 improvement for {patience} epochs. Stopping.")
                break

        # Save latest checkpoint every 5 epochs
        if epoch % 5 == 0:
            save_checkpoint(
                model, optimizer, epoch, val_f1,
                os.path.join(paths.get("checkpoints", "results/checkpoints"),
                             f"epoch_{epoch:03d}.pt"),
            )

    # ── Post-training plots ───────────────────────────────────────────────────
    plots_dir = paths.get("plots", "results/plots")
    plot_loss_curves(train_losses, val_losses,
                     os.path.join(plots_dir, "loss_curve.png"))
    plot_metric_curves({"val_f1": val_f1s},
                       os.path.join(plots_dir, "val_f1_curve.png"),
                       title="Validation F1 over Epochs")

    tb.close()
    csv_l.close()
    print(f"\n Training complete. Best val F1 = {best_val_f1:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Train Drowsiness Detection Model")
    p.add_argument("--config",      type=str, default="config.yaml")
    p.add_argument("--epochs",      type=int, default=None)
    p.add_argument("--batch_size",  type=int, default=None)
    p.add_argument("--lr",          type=float, default=None)
    p.add_argument("--uta_fold",    type=int, default=None)
    p.add_argument("--resume",      type=str, default=None,
                   help="Path to checkpoint to resume from")
    p.add_argument("--no_lm",       action="store_true",
                   help="Disable landmark branch")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # CLI overrides
    if args.epochs     is not None: cfg["training"]["epochs"]         = args.epochs
    if args.batch_size is not None: cfg["training"]["batch_size"]     = args.batch_size
    if args.lr         is not None: cfg["training"]["learning_rate"]  = args.lr
    if args.uta_fold   is not None: cfg["data"]["uta_fold"]           = args.uta_fold
    if args.no_lm:
        cfg["data"]["use_landmarks"]  = False
        cfg["model"]["use_landmarks"] = False

    if args.resume:
        # Just set path; train() will reload weights
        cfg["resume"] = args.resume

    train(cfg)
