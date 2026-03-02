"""
src/dataset.py — DrowsinessClipDataset
---------------------------------------
Loads face-cropped clip images + optional MediaPipe landmark .npy files and
returns (clip_tensor, landmark_tensor, label) tuples for training/evaluation.

Supports all four dataset layouts rooted in dataset/clips/:
  standard  (nthu_ddd, nitymed):  subject / label / video_clip / frames
  yawdd:                          label / video_clip / frames
  uta:                            subject / state_clip / frames   [no separate clip level]
  testing:                        video_clip / frames  (no label)

Split JSON format  (nthu_ddd, nitymed, yawdd):
  {"train": [...subjects...], "val": [...], "test": [...]}

Split JSON format  (uta_rldd — 5-fold):
  {"0": [...subjects...], "1": [...], ..., "4": [...]}

Usage:
  from src.dataset import build_dataloaders
  loaders = build_dataloaders(config)
  for clips, lms, labels in loaders['train']:
      ...
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# ── ImageNet stats ────────────────────────────────────────────────────────────
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

T_FRAMES = 16   # clip length (must match clip_builder.py)
FRAME_SIZE = 112

# ─────────────────────────────────────────────────────────────────────────────
# Augmentation pipelines
# ─────────────────────────────────────────────────────────────────────────────

def _train_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((FRAME_SIZE, FRAME_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomCrop(FRAME_SIZE, padding=8),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])


def _eval_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((FRAME_SIZE, FRAME_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Label helpers
# ─────────────────────────────────────────────────────────────────────────────

# Unified binary label map (alert=0, drowsy=1)
# ─────────────────────────────────────────────────────────────────
# Dataset-specific label notes:
#   nthu_ddd : folders named "alert" / "drowsy"
#   nitymed  : folders named "microsleep" / "yawning"
#              — BOTH map to drowsy=1 (nitymed has no alert class)
#   yawdd    : folders named "normal" / "yawning"
#   uta_rldd : clip prefix "alert" / "drowsy" / "low_vigilance"
#              — low_vigilance treated as drowsy
# ─────────────────────────────────────────────────────────────────
_LABEL_MAP: Dict[str, int] = {
    # nthu_ddd
    "alert":              0,
    "drowsy":             1,
    # nitymed  — drowsy-only dataset (no alert class folder)
    "microsleep":         1,   # NITYMED: microsleep → drowsy
    # "yawning" shared with yawdd below (also → drowsy)
    # yawdd
    "normal":             0,
    "yawning":            1,
    # uta_rldd
    "low_vigilance":      1,   # treat low-vigilance as drowsy
}


def _parse_label(label_str: str) -> int:
    return _LABEL_MAP.get(label_str.lower(), -1)


# ─────────────────────────────────────────────────────────────────────────────
# Clip discovery helpers
# ─────────────────────────────────────────────────────────────────────────────

ClipRecord = Tuple[Path, int, Optional[Path]]   # (clip_dir, label, lm_dir | None)


def _standard_clips(
    ds_clip_root: Path, ds_lm_root: Optional[Path],
    subjects: List[str], layout: str,
) -> List[ClipRecord]:
    """
    Discover clip directories for standard / yawdd layouts.

    standard  → subject / label / video_clip
    yawdd     → label / video_clip   (subject_key = "label/video")
    """
    records: List[ClipRecord] = []

    if layout == "standard":
        for subj in subjects:
            subj_path = ds_clip_root / subj
            if not subj_path.is_dir():
                continue
            for label_dir in subj_path.iterdir():
                if not label_dir.is_dir():
                    continue
                label_int = _parse_label(label_dir.name)
                if label_int < 0:
                    continue
                for clip_dir in label_dir.iterdir():
                    if not clip_dir.is_dir():
                        continue
                    lm_dir = (ds_lm_root / subj / label_dir.name / clip_dir.name
                               if ds_lm_root else None)
                    records.append((clip_dir, label_int, lm_dir))

    elif layout == "yawdd":
        # subjects are composite keys "label/video"
        subject_set = set(subjects)
        for label_dir in ds_clip_root.iterdir():
            if not label_dir.is_dir():
                continue
            label_int = _parse_label(label_dir.name)
            if label_int < 0:
                continue
            for video_clip_dir in label_dir.iterdir():
                if not video_clip_dir.is_dir():
                    continue
                # Extract "video_name" by stripping trailing _clipNNNN
                parts = video_clip_dir.name.rsplit("_clip", 1)
                composite = f"{label_dir.name}/{parts[0]}"
                if composite not in subject_set:
                    continue
                lm_dir = (ds_lm_root / label_dir.name / video_clip_dir.name
                           if ds_lm_root else None)
                records.append((video_clip_dir, label_int, lm_dir))

    return records


def _uta_clips(
    ds_clip_root: Path, ds_lm_root: Optional[Path],
    subjects: List[str],
) -> List[ClipRecord]:
    """
    UTA-RLDD layout:  subject / state_clipNNNN/frames
    The clip dirs are named "<state>_clip<NNNN>" directly under the subject dir.
    """
    subject_set = set(subjects)
    records: List[ClipRecord] = []

    for subj_dir in ds_clip_root.iterdir():
        if not subj_dir.is_dir() or subj_dir.name not in subject_set:
            continue
        for clip_dir in subj_dir.iterdir():
            if not clip_dir.is_dir():
                continue
            # Name pattern: <state>_clipNNNN  e.g. alert_clip0012
            state_part = clip_dir.name.rsplit("_clip", 1)[0]
            label_int = _parse_label(state_part)
            if label_int < 0:
                continue
            lm_dir = ds_lm_root / subj_dir.name / clip_dir.name if ds_lm_root else None
            records.append((clip_dir, label_int, lm_dir))

    return records


# ─────────────────────────────────────────────────────────────────────────────
# Core Dataset
# ─────────────────────────────────────────────────────────────────────────────

class DrowsinessClipDataset(Dataset):
    """
    Returns:
        clip   : FloatTensor  (3, T, H, W)  — channel-first, normalized
        lms    : FloatTensor  (T, 2)         — [EAR, MOR] per frame (zeros if unavailable)
        label  : LongTensor   scalar
    """

    def __init__(
        self,
        records: List[ClipRecord],
        transform,
        use_landmarks: bool = True,
        lm_feature_dim: int = 478 * 3,
    ):
        self.records = records
        self.transform = transform
        self.use_landmarks = use_landmarks
        self.lm_feature_dim = lm_feature_dim

        # ── Pre-cache file paths as strings (pickle-safe for Windows workers) ─
        self._frame_cache: List[List[str]] = []
        self._lm_cache:    List[List[str]] = []

        for clip_dir, _, lm_dir in records:
            # Frame files
            frames = sorted(clip_dir.glob("*.jpg"))
            if not frames:
                frames = sorted(clip_dir.glob("*.png"))
            if len(frames) < T_FRAMES:
                frames = frames + [frames[-1]] * (T_FRAMES - len(frames)) if frames else []
            self._frame_cache.append([str(f) for f in frames[:T_FRAMES]])

            # Landmark npy files
            if use_landmarks and lm_dir is not None and lm_dir.is_dir():
                npys = sorted(lm_dir.glob("*.npy"))
                if len(npys) < T_FRAMES:
                    npys = npys + [npys[-1]] * (T_FRAMES - len(npys)) if npys else []
                self._lm_cache.append([str(f) for f in npys[:T_FRAMES]])
            else:
                self._lm_cache.append([])

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        _, label, _ = self.records[idx]

        # Use pre-cached paths (strings); fall back to glob if cache missing
        if hasattr(self, '_frame_cache'):
            frame_files = self._frame_cache[idx]        # list[str]
            lm_files    = self._lm_cache[idx]
        else:
            clip_dir, _, lm_dir = self.records[idx]
            frame_files_raw = [str(f) for f in sorted(clip_dir.glob("*.jpg"))]
            if not frame_files_raw:
                frame_files_raw = [str(f) for f in sorted(clip_dir.glob("*.png"))]
            if len(frame_files_raw) < T_FRAMES:
                frame_files_raw = frame_files_raw + [frame_files_raw[-1]] * (T_FRAMES - len(frame_files_raw)) if frame_files_raw else []
            frame_files = frame_files_raw[:T_FRAMES]
            lm_files = []
            if self.use_landmarks and lm_dir is not None and lm_dir.is_dir():
                lm_files_raw = [str(f) for f in sorted(lm_dir.glob("*.npy"))]
                if len(lm_files_raw) < T_FRAMES:
                    lm_files_raw = lm_files_raw + [lm_files_raw[-1]] * (T_FRAMES - len(lm_files_raw)) if lm_files_raw else []
                lm_files = lm_files_raw[:T_FRAMES]

        # ── Load frames ───────────────────────────────────────────────────────
        frames = []
        for fp in frame_files:
            img = Image.open(fp).convert("RGB")
            frames.append(self.transform(img))   # (3, H, W)

        clip_tensor = torch.stack(frames, dim=1)  # (3, T, H, W)

        # ── Load landmarks → EAR / MOR ────────────────────────────────────────
        lm_tensor = torch.zeros(T_FRAMES, 2)
        for t, nf in enumerate(lm_files):
            try:
                lm  = np.load(nf)
                ear = _compute_ear(lm)
                mor = _compute_mor(lm)
                lm_tensor[t] = torch.tensor([ear, mor], dtype=torch.float32)
            except Exception:
                pass  # keep zeros

        return clip_tensor, lm_tensor, torch.tensor(label, dtype=torch.long)


# ─────────────────────────────────────────────────────────────────────────────
# EAR / MOR from flat 478-point MediaPipe landmarks
# ─────────────────────────────────────────────────────────────────────────────
# MediaPipe FaceMesh landmark indices (left/right eye, lips)
# Reference: https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png

_LEFT_EYE  = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
_RIGHT_EYE = [33,  7,   163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
# Simplified 6-point EAR indices (vertical/horizontal pairs)
_LEFT_EAR_IDX  = [362, 385, 387, 263, 373, 380]   # p1..p6
_RIGHT_EAR_IDX = [33,  160, 158, 133, 153, 144]
# Lips: top-center, bottom-center, left-corner, right-corner
_LIPS_IDX = [13, 14, 61, 291]


def _lm_xy(lm_flat: np.ndarray, idx: int) -> np.ndarray:
    """Extract (x, y) for landmark index from flat 478*3 array."""
    return lm_flat[idx * 3: idx * 3 + 2]


def _eye_aspect_ratio(lm_flat: np.ndarray, indices: List[int]) -> float:
    p1, p2, p3, p4, p5, p6 = [_lm_xy(lm_flat, i) for i in indices]
    numerator   = np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)
    denominator = 2.0 * np.linalg.norm(p1 - p4) + 1e-6
    return float(numerator / denominator)


def _compute_ear(lm_flat: np.ndarray) -> float:
    if lm_flat.sum() == 0:
        return 0.0
    left  = _eye_aspect_ratio(lm_flat, _LEFT_EAR_IDX)
    right = _eye_aspect_ratio(lm_flat, _RIGHT_EAR_IDX)
    return (left + right) / 2.0


def _compute_mor(lm_flat: np.ndarray) -> float:
    """Mouth Opening Ratio: vertical / horizontal lip distance."""
    if lm_flat.sum() == 0:
        return 0.0
    top, bot, left, right = [_lm_xy(lm_flat, i) for i in _LIPS_IDX]
    vertical   = np.linalg.norm(top - bot)
    horizontal = np.linalg.norm(left - right) + 1e-6
    return float(vertical / horizontal)


# ─────────────────────────────────────────────────────────────────────────────
# Record builders per dataset
# ─────────────────────────────────────────────────────────────────────────────

_DATASET_CONFIGS = {
    "nthu_ddd":  {"layout": "standard", "split_type": "subject"},
    "nitymed":   {"layout": "standard", "split_type": "subject"},
    "yawdd":     {"layout": "yawdd",    "split_type": "subject"},
    "uta_rldd":  {"layout": "uta",      "split_type": "fold"},
}


def _load_split(split_path: str, split: str, fold: int = 0) -> List[str]:
    """Load subject list for train/val/test or a UTA fold."""
    with open(split_path) as f:
        data = json.load(f)
    if "train" in data:
        return data.get(split, [])
    # 5-fold format: {"0": [...], ...}
    folds = list(data.values())
    if split == "test":
        return folds[fold]
    # Use the remaining 4 folds as train+val
    train_subjects = []
    for i, fold_subjects in enumerate(folds):
        if i != fold:
            train_subjects.extend(fold_subjects)
    if split == "train":
        return train_subjects[: int(len(train_subjects) * 0.9)]
    return train_subjects[int(len(train_subjects) * 0.9):]


def build_records(
    clips_root: str,
    landmarks_root: str,
    splits_dir: str,
    split: str,          # "train" | "val" | "test"
    use_landmarks: bool = True,
    uta_fold: int = 0,
    datasets: Optional[List[str]] = None,
) -> List[ClipRecord]:
    """
    Aggregate clip records from all (or a subset of) datasets for a given split.
    """
    clips_root_path = Path(clips_root)
    lm_root_path    = Path(landmarks_root) if use_landmarks else None
    splits_dir_path = Path(splits_dir)

    if datasets is None:
        datasets = list(_DATASET_CONFIGS.keys())

    all_records: List[ClipRecord] = []

    for ds_name in datasets:
        cfg = _DATASET_CONFIGS.get(ds_name)
        if cfg is None:
            continue

        ds_clip_dir = clips_root_path / ds_name
        ds_lm_dir   = (lm_root_path / ds_name) if lm_root_path else None

        if not ds_clip_dir.is_dir():
            print(f"[WARN] clips dir missing for {ds_name}, skipping.")
            continue

        # Determine split file
        if cfg["split_type"] == "fold":
            split_file = splits_dir_path / "uta_rldd_5fold.json"
            subjects = _load_split(str(split_file), split, fold=uta_fold)
            records = _uta_clips(ds_clip_dir, ds_lm_dir, subjects)
        else:
            split_file = splits_dir_path / f"{ds_name}_split.json"
            if not split_file.exists():
                print(f"[WARN] split file missing: {split_file}, skipping.")
                continue
            subjects = _load_split(str(split_file), split)
            records = _standard_clips(ds_clip_dir, ds_lm_dir, subjects, cfg["layout"])

        print(f"  [{ds_name}] {split}: {len(records)} clips")
        all_records.extend(records)

    return all_records


# ─────────────────────────────────────────────────────────────────────────────
# High-level DataLoader factory
# ─────────────────────────────────────────────────────────────────────────────

def build_dataloaders(config: dict) -> Dict[str, DataLoader]:
    """
    Build train / val / test DataLoaders from a config dict.

    Expected config keys:
        clips_root       : str  — e.g. "dataset/clips"
        landmarks_root   : str  — e.g. "dataset/landmarks_cropped"
        splits_dir       : str  — e.g. "data/splits"
        use_landmarks    : bool
        uta_fold         : int  (0–4)
        batch_size       : int
        num_workers      : int
        datasets         : list[str] | None
    """
    clips_root     = config["clips_root"]
    landmarks_root = config.get("landmarks_root", "dataset/landmarks_cropped")
    splits_dir     = config.get("splits_dir", "data/splits")
    use_landmarks  = config.get("use_landmarks", True)
    uta_fold       = config.get("uta_fold", 0)
    batch_size     = config.get("batch_size", 8)
    num_workers    = config.get("num_workers", 4)
    ds_list        = config.get("datasets", None)

    loaders: Dict[str, DataLoader] = {}

    for split in ("train", "val", "test"):
        records = build_records(
            clips_root=clips_root,
            landmarks_root=landmarks_root,
            splits_dir=splits_dir,
            split=split,
            use_landmarks=use_landmarks,
            uta_fold=uta_fold,
            datasets=ds_list,
        )
        transform = _train_transform() if split == "train" else _eval_transform()
        dataset   = DrowsinessClipDataset(records, transform, use_landmarks=use_landmarks)
        shuffle   = (split == "train")

        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(split == "train"),
            persistent_workers=(num_workers > 0),   # avoid worker respawn each epoch
            prefetch_factor=4 if num_workers > 0 else None,  # deeper pipeline
        )
        print(f"  [{split}] {len(dataset)} clips — {len(loaders[split])} batches "
              f"(batch={batch_size})")

    return loaders


# ─────────────────────────────────────────────────────────────────────────────
# Quick sanity-check (run as standalone script)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = {
        "clips_root":     "dataset/clips",
        "landmarks_root": "dataset/landmarks_cropped",
        "splits_dir":     "data/splits",
        "use_landmarks":  True,
        "uta_fold":       0,
        "batch_size":     4,
        "num_workers":    0,    # 0 for Windows debug
    }

    print("Building DataLoaders …")
    loaders = build_dataloaders(cfg)

    for split, loader in loaders.items():
        clips, lms, labels = next(iter(loader))
        print(f"\n[{split}]")
        print(f"  clip tensor   : {clips.shape}   dtype={clips.dtype}")
        print(f"  landmark tensor: {lms.shape}   dtype={lms.dtype}")
        print(f"  labels        : {labels}")
        assert clips.shape == (cfg['batch_size'], 3, T_FRAMES, FRAME_SIZE, FRAME_SIZE), \
            f"Unexpected clip shape: {clips.shape}"
        assert lms.shape == (cfg['batch_size'], T_FRAMES, 2), \
            f"Unexpected landmark shape: {lms.shape}"

    print("\n✅ DrowsinessClipDataset sanity check passed.")
