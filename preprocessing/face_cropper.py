"""
face_cropper.py — Batch InsightFace face cropper for all training datasets
--------------------------------------------------------------------------
Reads raw extracted frames from dataset/processed/<dataset>/... and writes
112x112 face-cropped images to dataset/processed_cropped/<dataset>/...

Dataset layout modes:
  standard  → <subject>/<label>/<video_clip>/frames   (NTHU-DDD, NITYMED)
  yawdd     → <label>/<video_clip>/frames
  uta       → <subject>/<state>/frames                (no video subfolder)
  testing   → <video_name>/frames                     (flat, no label)

Run from project root:
  conda activate webenv
  python preprocessing/face_cropper.py
"""

import os
import cv2
from glob import glob
from tqdm import tqdm
from insightface.app import FaceAnalysis

# ── Configurables ─────────────────────────────────────────────────────────────
IMG_SIZE      = 112         # Output face crop resolution (px)
FACE_DET_SIZE = (320, 320)  # RetinaFace internal resolution
SKIP_IF_DONE  = True        # Skip frame if output already exists

# (input_root, output_root, layout)
DATASETS = [
    ("dataset/processed/nthu_ddd",     "dataset/processed_cropped/nthu_ddd",     "standard"),
    ("dataset/processed/nitymed",      "dataset/processed_cropped/nitymed",      "standard"),
    ("dataset/processed/yawdd",        "dataset/processed_cropped/yawdd",        "yawdd"),
    ("dataset/processed/uta_rldd",     "dataset/processed_cropped/uta_rldd",     "uta"),
    ("dataset/processed/testing_nthu", "dataset/processed_cropped/testing_nthu", "testing"),
]
# ─────────────────────────────────────────────────────────────────────────────

# Initialize InsightFace RetinaFace detector (GPU)
face_app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection"])
face_app.prepare(ctx_id=0, det_size=FACE_DET_SIZE)


def crop_and_save_face(img_path: str, out_path: str) -> bool:
    """Detect face in img_path, crop, resize to IMG_SIZE, save to out_path."""
    if SKIP_IF_DONE and os.path.exists(out_path):
        return True
    img = cv2.imread(img_path)
    if img is None:
        return False
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = face_app.get(rgb)
    if len(faces) == 0:
        return False
    bbox = faces[0].bbox.astype(int)
    x1, y1, x2, y2 = bbox
    h, w = rgb.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    face = rgb[y1:y2, x1:x2]
    if face.size == 0:
        return False
    face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
    face_bgr = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, face_bgr)
    return True


def process_frames_in_dir(in_dir: str, out_dir: str) -> tuple[int, int]:
    """Crop all .jpg frames in in_dir → out_dir. Returns (saved, missed)."""
    frames = sorted(glob(os.path.join(in_dir, "*.jpg")))
    saved, missed = 0, 0
    for fp in frames:
        fname = os.path.basename(fp)
        ok = crop_and_save_face(fp, os.path.join(out_dir, fname))
        if ok:
            saved += 1
        else:
            missed += 1
    return saved, missed


# ── Layout handlers ───────────────────────────────────────────────────────────

def handle_standard(in_root: str, out_root: str) -> tuple[int, int]:
    """NTHU-DDD, NITYMED: <subject>/<label>/<video>/frames"""
    s, m = 0, 0
    for subject in tqdm(sorted(os.listdir(in_root)), desc="subjects"):
        subj_path = os.path.join(in_root, subject)
        if not os.path.isdir(subj_path):
            continue
        for label in sorted(os.listdir(subj_path)):
            label_path = os.path.join(subj_path, label)
            if not os.path.isdir(label_path):
                continue
            for video in sorted(os.listdir(label_path)):
                video_path = os.path.join(label_path, video)
                if not os.path.isdir(video_path):
                    continue
                out_dir = os.path.join(out_root, subject, label, video)
                ns, nm = process_frames_in_dir(video_path, out_dir)
                s += ns; m += nm
    return s, m


def handle_yawdd(in_root: str, out_root: str) -> tuple[int, int]:
    """YawDD: <label>/<video>/frames"""
    s, m = 0, 0
    for label in tqdm(sorted(os.listdir(in_root)), desc="labels"):
        label_path = os.path.join(in_root, label)
        if not os.path.isdir(label_path):
            continue
        for video in sorted(os.listdir(label_path)):
            video_path = os.path.join(label_path, video)
            if not os.path.isdir(video_path):
                continue
            out_dir = os.path.join(out_root, label, video)
            ns, nm = process_frames_in_dir(video_path, out_dir)
            s += ns; m += nm
    return s, m


def handle_uta(in_root: str, out_root: str) -> tuple[int, int]:
    """UTA-RLDD: <subject>/<state>/frames (no video subfolder)"""
    s, m = 0, 0
    for subject in tqdm(sorted(os.listdir(in_root)), desc="subjects"):
        subj_path = os.path.join(in_root, subject)
        if not os.path.isdir(subj_path):
            continue
        for state in sorted(os.listdir(subj_path)):
            state_path = os.path.join(subj_path, state)
            if not os.path.isdir(state_path):
                continue
            out_dir = os.path.join(out_root, subject, state)
            ns, nm = process_frames_in_dir(state_path, out_dir)
            s += ns; m += nm
    return s, m


def handle_testing(in_root: str, out_root: str) -> tuple[int, int]:
    """testing_nthu: <video_name>/frames (flat, no label)"""
    s, m = 0, 0
    for video in tqdm(sorted(os.listdir(in_root)), desc="videos"):
        video_path = os.path.join(in_root, video)
        if not os.path.isdir(video_path):
            continue
        out_dir = os.path.join(out_root, video)
        ns, nm = process_frames_in_dir(video_path, out_dir)
        s += ns; m += nm
    return s, m


HANDLERS = {
    "standard": handle_standard,
    "yawdd":    handle_yawdd,
    "uta":      handle_uta,
    "testing":  handle_testing,
}


def main():
    print("\n" + "=" * 70)
    print("Face Cropper — InsightFace RetinaFace (buffalo_l, 112×112)")
    print("=" * 70 + "\n")

    grand_saved, grand_missed = 0, 0

    for in_root, out_root, layout in DATASETS:
        if not os.path.isdir(in_root):
            print(f"[SKIP] Input not found: {in_root}")
            continue

        handler = HANDLERS.get(layout)
        if handler is None:
            print(f"[WARN] Unknown layout '{layout}' for {in_root}")
            continue

        print(f"\n[INFO] {os.path.basename(in_root)}  (layout={layout})")
        os.makedirs(out_root, exist_ok=True)
        saved, missed = handler(in_root, out_root)
        print(f"  → {saved} faces cropped | {missed} frames skipped (no face)")
        grand_saved += saved
        grand_missed += missed

    print(f"\n{'=' * 70}")
    print(f"Complete: {grand_saved} faces saved | {grand_missed} missed")
    print(f"Output: dataset/processed_cropped/<dataset>/...")


if __name__ == "__main__":
    main()
