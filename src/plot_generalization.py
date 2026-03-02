# Allow `python src/plot_generalization.py` to resolve imports
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..")))
"""
src/plot_generalization.py — Paper-Quality Generalization Comparison Plots
---------------------------------------------------------------------------
Reads the metrics_summary.csv files produced by cross_evaluate.py for
Experiment A and Experiment B, then generates:

  1. Seen vs. Unseen grouped bar chart (F1 / AUC side by side across both exps)
  2. Generalization Drop heatmap — matrix of F1 scores indexed by
     (train partition × test partition), immediately usable in a paper

Usage (after running both cross_evaluate.py runs):
    python src/plot_generalization.py \\
      --exp_A results/cross_eval/exp_A/plots/metrics_summary.csv \\
      --exp_B results/cross_eval/exp_B/plots/metrics_summary.csv \\
      --out   results/cross_eval/comparison

Outputs → out/ directory:
  • comparison_bar.png       — Grouped bar chart (Exp A vs. Exp B, Seen vs. Unseen)
  • generalization_heatmap.png — F1 heatmap across training/test splits
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# ─────────────────────────────────────────────────────────────────────────────
# CSV loader
# ─────────────────────────────────────────────────────────────────────────────

def load_metrics(csv_path: str) -> List[Dict]:
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Cast numeric columns
            for key in ("n_clips", "accuracy", "precision", "recall", "f1", "auc"):
                try:
                    row[key] = float(row[key])
                except (ValueError, KeyError):
                    row[key] = float("nan")
            rows.append(row)
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1 — Comparison bar chart  (F1 and AUC side-by-side across both exps)
# ─────────────────────────────────────────────────────────────────────────────

def plot_comparison_bar(
    rows_A: List[Dict],
    rows_B: List[Dict],
    out_path: str,
) -> None:
    """
    Two-panel figure:
      Left:  F1 per dataset, grouped by Seen/Unseen within each experiment
      Right: ROC AUC per dataset, same grouping
    """
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    # Collect all unique datasets in display order
    all_ds = []
    for r in rows_A + rows_B:
        if r["dataset"] not in all_ds:
            all_ds.append(r["dataset"])

    def ds_metric(rows, tag, metric):
        return {r["dataset"]: r.get(metric, float("nan"))
                for r in rows if r.get("split_tag") == tag}

    metrics  = ["f1", "auc"]
    m_labels = ["Macro F1", "ROC AUC"]

    # colour scheme
    palette = {
        ("A", "Seen"):   "#4C72B0",
        ("A", "Unseen"): "#A8C8F0",
        ("B", "Seen"):   "#DD8452",
        ("B", "Unseen"): "#F5C6A0",
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Cross-Dataset Generalisation: Exp A vs. Exp B", fontsize=13,
                 fontweight="bold")

    for ax, metric, mlabel in zip(axes, metrics, m_labels):
        x      = np.arange(len(all_ds))
        n_bars = 4   # A-Seen, A-Unseen, B-Seen, B-Unseen
        w      = 0.18
        offsets= [-1.5*w, -0.5*w, 0.5*w, 1.5*w]

        configs = [
            ("A", "Seen",   rows_A),
            ("A", "Unseen", rows_A),
            ("B", "Seen",   rows_B),
            ("B", "Unseen", rows_B),
        ]

        for offset, (exp, tag, rows) in zip(offsets, configs):
            vals = ds_metric(rows, tag, metric)
            bar_vals = [vals.get(d, float("nan")) for d in all_ds]
            label    = f"Exp {exp} — {tag}"
            color    = palette[(exp, tag)]

            bars = ax.bar(x + offset, bar_vals, w, label=label, color=color, alpha=0.88)
            for bar in bars:
                h = bar.get_height()
                if not np.isnan(h):
                    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.012,
                            f"{h:.2f}", ha="center", va="bottom", fontsize=7, rotation=90)

        ax.set_xticks(x)
        ax.set_xticklabels(all_ds, rotation=25, ha="right", fontsize=9)
        ax.set_ylim(0, 1.22)
        ax.set_ylabel(mlabel)
        ax.set_title(mlabel)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [plot] Comparison bar chart → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2 — Generalization Drop Heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_generalization_heatmap(
    rows_A: List[Dict],
    rows_B: List[Dict],
    out_path: str,
) -> None:
    """
    Row = Training partition (what the model was trained on)
    Col = Test dataset       (what it was evaluated on)
    Cell = macro F1
    Cells from the same "seen" partition are highlighted differently.
    """
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    # Collect datasets and their seen/unseen labels from both exps
    # Exp A: train=['nthu_ddd','yawdd'], test=['uta_rldd','nitymed']
    # Exp B: train=['uta_rldd','nitymed'], test=['nthu_ddd','yawdd']

    def summarise(rows):
        return {r["dataset"]: {"f1": r.get("f1", float("nan")),
                               "tag": r.get("split_tag", "?")} for r in rows}

    map_A = summarise(rows_A)
    map_B = summarise(rows_B)
    all_ds = sorted(set(map_A) | set(map_B))

    # Build F1 matrix:  2 rows (Exp A, Exp B)  x  N columns (datasets)
    train_labels = ["NTHU+YawDD (Exp A)", "UTA+NITYMED (Exp B)"]
    cols  = all_ds
    data  = []
    masks = []   # True if "Seen" cell (diagonal split)

    for row_idx, map_ in enumerate([map_A, map_B]):
        row_vals = []
        row_mask = []
        for ds in cols:
            entry = map_.get(ds, {})
            row_vals.append(entry.get("f1", float("nan")))
            row_mask.append(entry.get("tag", "?") == "Seen")
        data.append(row_vals)
        masks.append(row_mask)

    data  = np.array(data, dtype=float)
    masks = np.array(masks, dtype=bool)

    fig, ax = plt.subplots(figsize=(max(8, len(cols) * 1.6), 4))
    im = ax.imshow(data, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Macro F1", fontsize=10)

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=30, ha="right", fontsize=10)
    ax.set_yticks(range(len(train_labels)))
    ax.set_yticklabels(train_labels, fontsize=10)
    ax.set_title("Generalization Heatmap — Macro F1 (green=high, red=low)\n"
                 "Bold border = dataset seen during training",
                 fontsize=10)

    # Annotate cells
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            txt = f"{val:.3f}" if not np.isnan(val) else "N/A"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=11, fontweight="bold" if masks[i, j] else "normal",
                    color="black")
            if masks[i, j]:
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                     linewidth=3, edgecolor="navy",
                                     facecolor="none")
                ax.add_patch(rect)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [plot] Generalization heatmap → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(exp_A_csv: str, exp_B_csv: str, out_dir: str) -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    print(f"\n[plot] Loading Exp A metrics: {exp_A_csv}")
    rows_A = load_metrics(exp_A_csv)
    print(f"[plot] Loading Exp B metrics: {exp_B_csv}")
    rows_B = load_metrics(exp_B_csv)

    bar_path      = os.path.join(out_dir, "comparison_bar.png")
    heatmap_path  = os.path.join(out_dir, "generalization_heatmap.png")

    print("\n[plot] Generating comparison bar chart …")
    plot_comparison_bar(rows_A, rows_B, bar_path)

    print("[plot] Generating generalization heatmap …")
    plot_generalization_heatmap(rows_A, rows_B, heatmap_path)

    print("\n✅ Comparison plots saved to:", out_dir)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os  # noqa — needed by main()

    p = argparse.ArgumentParser(description="Generate cross-experiment generalisation plots")
    p.add_argument("--exp_A", required=True,
                   help="Path to Exp A metrics_summary.csv")
    p.add_argument("--exp_B", required=True,
                   help="Path to Exp B metrics_summary.csv")
    p.add_argument("--out",   default="results/cross_eval/comparison",
                   help="Output directory for plots")
    args = p.parse_args()

    main(args.exp_A, args.exp_B, args.out)
