"""
src/utils.py — Shared Utilities
---------------------------------
  save_checkpoint / load_checkpoint
  plot_loss_curves
  set_seed
  TensorBoard SummaryWriter wrapper
  EAR / MOR helpers (importable from here too)
"""

import os
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")   # headless
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    """Fix seeds for reproducibility across Python / NumPy / PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ─────────────────────────────────────────────────────────────────────────────
# Checkpointing
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    score: float,
    path: str,
    scheduler=None,
    scaler=None,
) -> None:
    """Save model + optimizer (+ optional scheduler / GradScaler) to *path*."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch":      epoch,
        "score":      score,
        "model":      model.state_dict(),
        "optimizer":  optimizer.state_dict(),
    }
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    if scaler is not None:
        state["scaler"] = scaler.state_dict()
    torch.save(state, path)
    print(f"  [ckpt] saved → {path}  (epoch={epoch}, score={score:.4f})")


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    scaler=None,
    device: str = "cpu",
) -> Dict:
    """Load checkpoint; returns the state dict for introspection."""
    state = torch.load(path, map_location=device)
    model.load_state_dict(state["model"])
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and "scheduler" in state:
        scheduler.load_state_dict(state["scheduler"])
    if scaler is not None and "scaler" in state:
        scaler.load_state_dict(state["scaler"])
    print(f"  [ckpt] loaded <- {path}  (epoch={state.get('epoch')}, score={state.get('score', 'n/a'):.4f})")
    return state


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_loss_curves(
    train_losses: List[float],
    val_losses:   List[float],
    save_path:    str,
    title:        str = "Loss Curves",
) -> None:
    """Save a train/val loss curve PNG."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="Train", linewidth=2)
    plt.plot(epochs, val_losses,   label="Val",   linewidth=2, linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  [plot] loss curve → {save_path}")


def plot_metric_curves(
    metrics:    Dict[str, List[float]],
    save_path:  str,
    title:      str = "Metrics",
) -> None:
    """Generic multi-metric plot (e.g. accuracy, F1 over epochs)."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(next(iter(metrics.values()))) + 1)
    plt.figure(figsize=(8, 5))
    for name, vals in metrics.items():
        plt.plot(epochs, vals, label=name, linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# TensorBoard wrapper
# ─────────────────────────────────────────────────────────────────────────────

class TBWriter:
    """Thin wrapper around TensorBoard SummaryWriter with graceful fallback."""

    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        try:
            from torch.utils.tensorboard import SummaryWriter
            self._writer = SummaryWriter(log_dir=log_dir)
            print(f"  [tb] logging to {log_dir}")
        except ImportError:
            self._writer = None
            print("  [tb] TensorBoard not installed; skipping.")

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        if self._writer:
            self._writer.add_scalar(tag, value, step)

    def add_scalars(self, main_tag: str, tag_scalar_dict: dict, step: int) -> None:
        if self._writer:
            self._writer.add_scalars(main_tag, tag_scalar_dict, step)

    def close(self) -> None:
        if self._writer:
            self._writer.close()


# ─────────────────────────────────────────────────────────────────────────────
# EAR / MOR (also importable from here for inference)
# ─────────────────────────────────────────────────────────────────────────────

_LEFT_EAR_IDX  = [362, 385, 387, 263, 373, 380]
_RIGHT_EAR_IDX = [33,  160, 158, 133, 153, 144]
_LIPS_IDX      = [13,  14,  61,  291]


def _lm_xy(lm_flat: np.ndarray, idx: int) -> np.ndarray:
    return lm_flat[idx * 3: idx * 3 + 2]


def compute_ear(lm_flat: np.ndarray) -> float:
    """Eye Aspect Ratio from flat 478*3 MediaPipe landmark array."""
    def _one(indices):
        p1, p2, p3, p4, p5, p6 = [_lm_xy(lm_flat, i) for i in indices]
        return float((np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5))
                     / (2.0 * np.linalg.norm(p1 - p4) + 1e-6))
    return (_one(_LEFT_EAR_IDX) + _one(_RIGHT_EAR_IDX)) / 2.0


def compute_mor(lm_flat: np.ndarray) -> float:
    """Mouth Opening Ratio from flat 478*3 MediaPipe landmark array."""
    top, bot, left, right = [_lm_xy(lm_flat, i) for i in _LIPS_IDX]
    return float(np.linalg.norm(top - bot) / (np.linalg.norm(left - right) + 1e-6))


# ─────────────────────────────────────────────────────────────────────────────
# Misc
# ─────────────────────────────────────────────────────────────────────────────

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        dev = torch.device("cuda")
        print(f"  [device] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        dev = torch.device("cpu")
        print("  [device] Using CPU")
    return dev
