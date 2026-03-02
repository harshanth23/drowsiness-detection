"""
subject_split.py — Subject-wise train/val/test split
------------------------------------------------------
Uses sklearn GroupShuffleSplit to ensure no subject appears in more
than one partition (prevents identity leakage).

Split ratios (per plan.md):
  Train : 80%
  Val   : 10%   (held out from train fold for early stopping)
  Test  : 20%

UTA-RLDD uses the official 5-fold CV scheme instead.

Run from project root:
  conda activate webenv
  python preprocessing/subject_split.py
"""

import os
import json
import random
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ── Per-dataset subject directories ─────────────────────────────────────────
# Datasets where top-level dirs are subjects (nthu_ddd, nitymed)
DATASET_SUBJECT_ROOTS = {
    "nthu_ddd":  "dataset/processed_cropped/nthu_ddd",
    "nitymed":   "dataset/processed_cropped/nitymed",
    # UTA-RLDD uses official 5-fold split (generated separately below)
    # YawDD has no subject dirs — handled separately via yawdd_subject_split()
}

UTA_CROPPED_ROOT = "dataset/processed_cropped/uta_rldd"
OUTPUT_DIR       = "data/splits"


def _safe_inner_split(subjects_tv: list, seed: int) -> tuple[list, list]:
    """
    Split train+val subjects into train / val.
    Uses GroupShuffleSplit(10%) when possible; falls back to a
    simple last-one-out if there are too few subjects (≤ 2).
    """
    n = len(subjects_tv)
    if n <= 1:
        return subjects_tv, []          # only 1 subject — no val possible
    if n == 2:
        return [subjects_tv[0]], [subjects_tv[1]]  # simple fallback

    X_tv      = np.arange(n)
    groups_tv = np.arange(n)
    gss_inner = GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=seed)
    train_rel, val_rel = next(gss_inner.split(X_tv, groups=groups_tv))
    return [subjects_tv[i] for i in train_rel], [subjects_tv[i] for i in val_rel]


def simple_subject_split(dataset_name: str, dataset_path: str, save_path: str,
                         subjects: list | None = None) -> dict:
    """
    80 / 20 subject-wise split using GroupShuffleSplit.
    The 80% train portion is further split ~90/10 to produce a val set.
    Pass *subjects* explicitly for datasets whose top-level dirs are not subjects.
    """
    if not os.path.isdir(dataset_path):
        print(f"[WARN] Path not found, skipping {dataset_name}: {dataset_path}")
        return {}

    if subjects is None:
        subjects = sorted([
            d for d in os.listdir(dataset_path)
            if os.path.isdir(os.path.join(dataset_path, d))
        ])
    if len(subjects) == 0:
        print(f"[WARN] No subjects found in {dataset_path}")
        return {}

    # Dummy arrays — GroupShuffleSplit only needs group labels
    X      = np.arange(len(subjects))
    groups = np.arange(len(subjects))   # one sample per subject

    # ── Train+Val vs Test (80 / 20) ─────────────────────────────────────────
    gss_outer = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
    train_val_idx, test_idx = next(gss_outer.split(X, groups=groups))

    subjects_tv   = [subjects[i] for i in train_val_idx]
    test_subjects = [subjects[i] for i in test_idx]

    # ── Train vs Val (safe inner split) ─────────────────────────────────────
    train_subjects, val_subjects = _safe_inner_split(subjects_tv, SEED)

    split_dict = {
        "train": sorted(train_subjects),
        "val":   sorted(val_subjects),
        "test":  sorted(test_subjects),
    }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(split_dict, f, indent=4)

    print(
        f"  [{dataset_name}] {len(train_subjects)} train | "
        f"{len(val_subjects)} val | {len(test_subjects)} test subjects  → {save_path}"
    )
    return split_dict


def yawdd_subject_split(dataset_path: str, save_path: str) -> dict:
    """
    YawDD has no explicit subject IDs; the video names are used as subject proxies.
    Collect all video-folder names from inside each label subdir,
    then do the standard 80/20 split across those video names.
    """
    if not os.path.isdir(dataset_path):
        print(f"[WARN] YawDD path not found: {dataset_path}")
        return {}

    video_subjects: list[str] = []
    for label in sorted(os.listdir(dataset_path)):
        label_path = os.path.join(dataset_path, label)
        if not os.path.isdir(label_path):
            continue
        for vid in sorted(os.listdir(label_path)):
            if os.path.isdir(os.path.join(label_path, vid)):
                video_subjects.append(f"{label}/{vid}")  # unique composite key

    print(f"  [yawdd] {len(video_subjects)} video-level subjects found")
    return simple_subject_split("yawdd", dataset_path, save_path, subjects=video_subjects)


def uta_five_fold_split(uta_path: str, save_path: str) -> dict:
    """
    UTA-RLDD official 5-fold cross-validation (12 subjects per fold).
    Subjects are sorted and assigned round-robin to folds 0–4.
    """
    if not os.path.isdir(uta_path):
        print(f"[WARN] UTA-RLDD path not found, skipping: {uta_path}")
        return {}

    subjects = sorted([
        d for d in os.listdir(uta_path)
        if os.path.isdir(os.path.join(uta_path, d))
    ])

    folds: dict[str, list] = {str(i): [] for i in range(5)}
    for i, subj in enumerate(subjects):
        folds[str(i % 5)].append(subj)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(folds, f, indent=4)

    sizes = {k: len(v) for k, v in folds.items()}
    print(f"  [uta_rldd] 5-fold CV: {sizes}  → {save_path}")
    return folds


def main():
    print("\n" + "=" * 70)
    print("Subject-wise Split Generator (seed=42, plan.md compliant)")
    print("=" * 70 + "\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for dataset_name, dataset_path in DATASET_SUBJECT_ROOTS.items():
        save_path = os.path.join(OUTPUT_DIR, f"{dataset_name}_split.json")
        simple_subject_split(dataset_name, dataset_path, save_path)

    # YawDD — video-level proxies for subjects
    yawdd_path = "dataset/processed_cropped/yawdd"
    yawdd_save = os.path.join(OUTPUT_DIR, "yawdd_split.json")
    yawdd_subject_split(yawdd_path, yawdd_save)

    # UTA-RLDD — official 5-fold CV
    uta_save = os.path.join(OUTPUT_DIR, "uta_rldd_5fold.json")
    uta_five_fold_split(UTA_CROPPED_ROOT, uta_save)

    print(f"\nAll splits saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
