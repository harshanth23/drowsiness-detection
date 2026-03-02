# Allow `python src/find_gradcam_candidates.py` to resolve `from src.*` imports
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..")))
"""
src/find_gradcam_candidates.py — Auto-find Best Grad-CAM Clips
---------------------------------------------------------------
Runs inference on a dataset split while tracking every clip path, then:
  1. Identifies high-confidence misclassifications (best FN + FP candidates)
  2. Identifies high-confidence correct predictions (for comparison)
  3. Prints a ranked candidate table
  4. Optionally runs Grad-CAM on every selected clip automatically

Why high-confidence errors?  A model that is *very sure* but *very wrong*
is the most visually informative Grad-CAM subject — the heatmap reveals
exactly which features led it astray (background, lighting, head pose, …).

Usage:
    conda activate webenv
    cd "D:\\Class Work\\Open Lab\\Drowsiness Detection"

    # Full model — NTHU-DDD test split (find candidates only, no Grad-CAM yet)
    python src/find_gradcam_candidates.py \\
        --config config.yaml \\
        --model  results/checkpoints/best_model.pt \\
        --dataset nthu_ddd --split test

    # Cross-model Exp A — run Grad-CAM automatically on best picks
    python src/find_gradcam_candidates.py \\
        --config experiments/exp_A.yaml \\
        --model  results/cross_eval/exp_A/checkpoints/best_model.pt \\
        --dataset uta_rldd --split test \\
        --run_gradcam \\
        --out_dir results/plots/gradcam/exp_A_unseen

    # Cross-model Exp A — seen dataset comparison
    python src/find_gradcam_candidates.py \\
        --config experiments/exp_A.yaml \\
        --model  results/cross_eval/exp_A/checkpoints/best_model.pt \\
        --dataset nthu_ddd --split test \\
        --run_gradcam \\
        --out_dir results/plots/gradcam/exp_A_seen

Outputs:
  • Console table of ranked candidates (sorted by drowsy probability)
  • [optional] Grad-CAM grid PNGs for every selected clip
"""

import argparse
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, Dataset
import yaml
from tqdm.auto import tqdm

from src.dataset import (
    DrowsinessClipDataset, _eval_transform, build_records,
)
from src.model import DrowsinessModel
from src.utils import get_device, load_checkpoint, set_seed
from src.gradcam import run_gradcam


_CLASS_NAMES = ["Alert", "Drowsy"]


# ─────────────────────────────────────────────────────────────────────────────
# Inference pass — keeps clip paths in sync with predictions
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _infer_with_paths(
    model:      nn.Module,
    records,                   # list of (clip_dir, label, lm_dir)
    device:     torch.device,
    use_amp:    bool,
    use_lm:     bool,
    batch_size: int = 8,
) -> List[Dict]:
    """
    Returns a list of dicts, one per clip:
        clip_dir    : str   — absolute path to the clip directory
        true_label  : int   — ground-truth class (0=Alert, 1=Drowsy)
        pred_label  : int   — predicted class
        prob_alert  : float — softmax probability for Alert
        prob_drowsy : float — softmax probability for Drowsy
        correct     : bool
        error_type  : str   — "TP","TN","FP","FN"
    """
    dataset = DrowsinessClipDataset(records, _eval_transform(), use_landmarks=use_lm)
    # num_workers=0 is mandatory here — on Windows, multiprocessing workers write
    # to stdout between tqdm's cursor-up escape codes, causing the bar to print
    # on a new line every update instead of overwriting in place.
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                         num_workers=0, pin_memory=False)

    model.eval()
    results: List[Dict] = []
    sample_idx = 0

    # dynamic_ncols=True — let tqdm measure the real terminal width each update.
    # Do NOT combine with a fixed ncols= value; they conflict and cause jumpy output.
    pbar = tqdm(loader, desc="  Scanning clips", unit="batch",
                dynamic_ncols=True, leave=True)
    
    for clips, lms, labels in pbar:
        clips = clips.to(device, non_blocking=True)
        lms   = lms.to(device, non_blocking=True) if use_lm else None

        with autocast(enabled=use_amp):
            logits = model(clips, lms)
            probs  = torch.softmax(logits, dim=1)

        preds       = probs.argmax(dim=1).cpu().tolist()
        prob_alert  = probs[:, 0].cpu().tolist()
        prob_drowsy = probs[:, 1].cpu().tolist()
        true_labels = labels.tolist()

        for i in range(len(true_labels)):
            tl  = true_labels[i]
            pl  = preds[i]
            pa  = prob_alert[i]
            pd  = prob_drowsy[i]
            ok  = (tl == pl)

            if ok:
                err = "TP" if tl == 1 else "TN"
            else:
                err = "FN" if tl == 1 else "FP"   # FN: drowsy predicted Alert; FP: alert predicted Drowsy

            clip_dir, _, _ = records[sample_idx]
            results.append({
                "clip_dir":   str(clip_dir),
                "true_label": tl,
                "pred_label": pl,
                "prob_alert":  pa,
                "prob_drowsy": pd,
                "confidence": max(pa, pd),
                "correct":    ok,
                "error_type": err,
            })
            sample_idx += 1

    pbar.close()
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Candidate selection
# ─────────────────────────────────────────────────────────────────────────────

def select_candidates(
    results:    List[Dict],
    n_fp:       int = 2,    # Alert→predicted Drowsy (false alarm)
    n_fn:       int = 2,    # Drowsy→predicted Alert (missed detection)
    n_correct:  int = 2,    # high-confidence correct predictions (1 alert + 1 drowsy)
) -> Dict[str, List[Dict]]:
    """
    Returns a dict with keys:
        'FP'      — top n_fp False Positives by confidence
        'FN'      — top n_fn False Negatives by confidence
        'correct' — top n_correct correct predictions
    """
    fp_pool = sorted([r for r in results if r["error_type"] == "FP"],
                     key=lambda r: r["confidence"], reverse=True)
    fn_pool = sorted([r for r in results if r["error_type"] == "FN"],
                     key=lambda r: r["confidence"], reverse=True)

    # For correct clips, pick highest-confidence Alert AND Drowsy separately
    tn_pool = sorted([r for r in results if r["error_type"] == "TN"],
                     key=lambda r: r["confidence"], reverse=True)
    tp_pool = sorted([r for r in results if r["error_type"] == "TP"],
                     key=lambda r: r["confidence"], reverse=True)
    n_per   = max(1, n_correct // 2)
    correct_pool = tp_pool[:n_per] + tn_pool[:n_per]

    return {
        "FP":      fp_pool[:n_fp],
        "FN":      fn_pool[:n_fn],
        "correct": correct_pool,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Pretty-print candidate table
# ─────────────────────────────────────────────────────────────────────────────

def print_candidates(candidates: Dict[str, List[Dict]], dataset: str) -> None:
    _err_labels = {
        "FP": "FP  Alert→Drowsy (false alarm)",
        "FN": "FN  Drowsy→Alert (missed)",
        "TP": "TP  Drowsy (correct)",
        "TN": "TN  Alert  (correct)",
    }
    print(f"\n{'='*80}")
    print(f"  Grad-CAM Candidates — {dataset}")
    print(f"{'='*80}")
    print(f"  {'Type':<30} {'Conf':>6} {'P(Alert)':>9} {'P(Drowsy)':>10}  Clip")
    print(f"  {'-'*78}")
    for group_key in ("FP", "FN", "correct"):
        for r in candidates[group_key]:
            label = _err_labels.get(r["error_type"], r["error_type"])
            clip_name = Path(r["clip_dir"]).name
            print(f"  {label:<30} {r['confidence']:>6.3f} {r['prob_alert']:>9.3f}"
                  f" {r['prob_drowsy']:>10.3f}  {clip_name}")
        if candidates[group_key]:
            print()
    print(f"{'='*80}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def find_candidates(
    config:       dict,
    model_path:   str,
    dataset:      str,
    split:        str        = "test",
    n_fp:         int        = 2,
    n_fn:         int        = 2,
    n_correct:    int        = 2,
    run_gc:       bool       = False,
    out_dir:      str        = "results/plots/gradcam",
    batch_size:   int        = 32,
    alpha:        float      = 0.35,
    max_subjects: int        = 0,    # 0 = use all subjects; N = randomly sample N subjects
) -> None:
    device = get_device()
    set_seed(config["training"].get("seed", 42))

    # Cross-experiment configs have a 'cross' block; check for train_datasets
    cross_cfg   = config.get("cross", {})
    train_ds    = cross_cfg.get("train_datasets", None)
    test_ds     = cross_cfg.get("test_datasets",  None)
    seen_tag    = ""
    if train_ds and test_ds:
        if dataset in train_ds:
            seen_tag = "[SEEN]"
        elif dataset in test_ds:
            seen_tag = "[UNSEEN ← key for paper!]"

    print(f"\n[candidates] Dataset : {dataset} {seen_tag}")
    print(f"[candidates] Split   : {split}")
    print(f"[candidates] Model   : {model_path}")

    data_cfg = config["data"]
    use_lm   = data_cfg.get("use_landmarks", True)
    uta_fold = data_cfg.get("uta_fold", 0)

    records = build_records(
        clips_root     = data_cfg["clips_root"],
        landmarks_root = data_cfg.get("landmarks_root", "dataset/landmarks_cropped"),
        splits_dir     = data_cfg.get("splits_dir", "data/splits"),
        split          = split,
        use_landmarks  = use_lm,
        uta_fold       = uta_fold,
        datasets       = [dataset],
    )
    if not records:
        print(f"[WARN] No records found for {dataset}/{split}. Exiting.")
        return

    # ── Subject subsample (for speed) ────────────────────────────────────────
    if max_subjects > 0:
        # Extract subject id from clip path: .../clips/<ds>/<subject>/<label>/...
        # Path index -3 = <subject> (e.g. subject001)
        from pathlib import Path as _Path
        import random as _random
        all_subjects = sorted({_Path(r[0]).parts[-3] for r in records})
        if max_subjects < len(all_subjects):
            rng = _random.Random(config["training"].get("seed", 42))
            chosen = set(rng.sample(all_subjects, max_subjects))
            records = [r for r in records if _Path(r[0]).parts[-3] in chosen]
            print(f"  [subsample] {max_subjects}/{len(all_subjects)} subjects "
                  f"→ {len(records)} clips  (subjects: {sorted(chosen)})")
        else:
            print(f"  [subsample] max_subjects={max_subjects} >= total subjects "
                  f"({len(all_subjects)}), using all.")

    # Load model
    print("\n[candidates] Loading model …")
    model   = DrowsinessModel.from_config(config).to(device)
    load_checkpoint(model_path, model, device=str(device))
    use_amp = config["training"].get("use_amp", True) and device.type == "cuda"

    # Run inference (always single-worker to keep tqdm bar stable on Windows)
    print(f"[candidates] Running inference on {len(records)} clips …")
    results = _infer_with_paths(model, records, device, use_amp, use_lm,
                                batch_size=batch_size)

    # Stats summary
    n_correct_total = sum(1 for r in results if r["correct"])
    n_fp_total = sum(1 for r in results if r["error_type"] == "FP")
    n_fn_total = sum(1 for r in results if r["error_type"] == "FN")
    print(f"\n  Total clips : {len(results)}")
    print(f"  Correct     : {n_correct_total} ({100*n_correct_total/len(results):.1f}%)")
    print(f"  FP (Alert→Drowsy): {n_fp_total}")
    print(f"  FN (Drowsy→Alert): {n_fn_total}")

    # Select candidates
    candidates = select_candidates(results, n_fp=n_fp, n_fn=n_fn, n_correct=n_correct)
    print_candidates(candidates, dataset)

    if not run_gc:
        print("  [tip] Add --run_gradcam to automatically generate Grad-CAM for all candidates above.\n")
        return

    # ── Run Grad-CAM on each candidate ────────────────────────────────────────
    all_picks = (
        [("FP",      r) for r in candidates["FP"]] +
        [("FN",      r) for r in candidates["FN"]] +
        [("correct", r) for r in candidates["correct"]]
    )

    total = len(all_picks)
    print(f"[gradcam] Running Grad-CAM on {total} candidate clips …\n")

    for idx, (group, r) in enumerate(all_picks, 1):
        clip_dir    = r["clip_dir"]
        true_name   = _CLASS_NAMES[r["true_label"]]
        pred_name   = _CLASS_NAMES[r["pred_label"]]
        err_type    = r["error_type"]
        clip_name   = Path(clip_dir).name

        # Visualise from the PREDICTED class perspective for misclassifications
        # (shows what the model *thought* it was seeing), and true class for correct ones.
        if r["correct"]:
            vis_class_idx = r["true_label"]
            vis_label     = f"correct_{true_name.lower()}"
        else:
            vis_class_idx = r["pred_label"]
            vis_label     = f"{err_type}_pred{pred_name}_true{true_name}"

        # run_gradcam() appends clip_name/class_label internally, so pass the
        # parent directory only — avoids double-nesting: vis_label/clip_name/clip_name/
        clip_out_dir = str(Path(out_dir) / dataset / vis_label)

        print(f"  [{idx}/{total}] {err_type}  true={true_name}  pred={pred_name}"
              f"  conf={r['confidence']:.3f}  → {clip_name}")

        try:
            run_gradcam(
                clip_dir   = clip_dir,
                model_path = model_path,
                config     = config,
                out_dir    = clip_out_dir,
                class_idx  = vis_class_idx,
                alpha      = alpha,
            )
        except Exception as e:
            print(f"    [WARN] Grad-CAM failed for {clip_name}: {e}")

    print(f"\n✅ Grad-CAM complete. Output → {out_dir}/")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Auto-find misclassified clips and run Grad-CAM for paper figures"
    )
    p.add_argument("--config",      required=True,
                   help="config.yaml or experiments/exp_X.yaml")
    p.add_argument("--model",       required=True,
                   help="Path to model checkpoint (.pt)")
    p.add_argument("--dataset",     required=True,
                   help="Dataset name: nthu_ddd | yawdd | uta_rldd | nitymed")
    p.add_argument("--split",       default="test",
                   help="Data split: train | val | test  (default: test)")
    p.add_argument("--n_fp",        type=int, default=2,
                   help="Number of False Positives to pick (default: 2)")
    p.add_argument("--n_fn",        type=int, default=2,
                   help="Number of False Negatives to pick (default: 2)")
    p.add_argument("--n_correct",   type=int, default=2,
                   help="Number of correct clips to pick (default: 2, 1 TP + 1 TN)")
    p.add_argument("--run_gradcam", action="store_true",
                   help="Run Grad-CAM automatically on all selected candidates")
    p.add_argument("--out_dir",      default="results/plots/gradcam",
                   help="Root output directory for Grad-CAM images")
    p.add_argument("--batch_size",   type=int, default=32,
                   help="Inference batch size (default 32, larger = faster GPU utilisation)")
    p.add_argument("--alpha",        type=float, default=0.35,
                   help="Heatmap blend strength 0.0-1.0 (default 0.35 = face clearly visible)")
    p.add_argument("--max_subjects", type=int, default=0,
                   help="Randomly sample N subjects before inference (0 = all). "
                        "Use 5 for a fast ~7-min run instead of ~36 min.")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    find_candidates(
        config       = cfg,
        model_path   = args.model,
        dataset      = args.dataset,
        split        = args.split,
        n_fp         = args.n_fp,
        n_fn         = args.n_fn,
        n_correct    = args.n_correct,
        run_gc       = args.run_gradcam,
        out_dir      = args.out_dir,
        batch_size   = args.batch_size,
        alpha        = args.alpha,
        max_subjects = args.max_subjects,
    )
