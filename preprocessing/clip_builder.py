"""
clip_builder.py — Sliding-window clip builder
----------------------------------------------
Scans dataset/processed_cropped/<dataset>/<subject>/<label>/<video>/
and groups consecutive frames into fixed-length clips of T frames,
saving each clip as a subfolder with symlinked (or copied) frame files.

Output layout:
  dataset/clips/<dataset>/<subject>/<label>/<video>_clip<N>/
      frame_000001.jpg
      frame_000002.jpg
      ...
      frame_000016.jpg

Run from project root:
  conda activate webenv
  python preprocessing/clip_builder.py
"""

import os
import shutil
from glob import glob
from tqdm import tqdm

# ── Configurables ────────────────────────────────────────────────────────────
CROPPED_ROOT  = "dataset/processed_cropped"   # input: cropped face frames
CLIPS_ROOT    = "dataset/clips"               # output: clip subfolders
CLIP_LEN      = 16                            # frames per clip  (T)
STRIDE        = 8                             # sliding-window stride (T//2)
SKIP_IF_DONE  = True                          # skip video if clips already built
MIN_FRAMES    = CLIP_LEN                      # skip video with fewer frames
# ─────────────────────────────────────────────────────────────────────────────


def get_sorted_frames(folder: str) -> list[str]:
    """Return sorted list of .jpg frame paths inside *folder*."""
    frames = sorted(glob(os.path.join(folder, "*.jpg")))
    return frames


def build_clips_for_video(frame_paths: list[str], out_base: str) -> int:
    """
    Slice *frame_paths* into overlapping clips of CLIP_LEN with STRIDE.
    Each clip is written to out_base + f'_clip{idx:04d}/'.
    Returns number of clips created.
    """
    n = len(frame_paths)
    if n < MIN_FRAMES:
        return 0

    clip_idx = 0
    start = 0
    while start + CLIP_LEN <= n:
        clip_dir = f"{out_base}_clip{clip_idx:04d}"
        if SKIP_IF_DONE and os.path.isdir(clip_dir) and len(os.listdir(clip_dir)) == CLIP_LEN:
            clip_idx += 1
            start += STRIDE
            continue

        os.makedirs(clip_dir, exist_ok=True)
        for frame_i, src in enumerate(frame_paths[start: start + CLIP_LEN]):
            frame_name = f"frame_{frame_i:06d}.jpg"
            dst = os.path.join(clip_dir, frame_name)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)   # copy; use os.symlink for large datasets

        clip_idx += 1
        start += STRIDE

    return clip_idx


def process_dataset_standard(dataset_path: str, out_dataset: str) -> int:
    """
    Handle datasets with structure:
      <dataset>/<subject>/<label>/<video_folder>/frames…
    e.g. nthu_ddd, nitymed
    """
    total_clips = 0
    for subject in sorted(os.listdir(dataset_path)):
        subject_path = os.path.join(dataset_path, subject)
        if not os.path.isdir(subject_path):
            continue
        for label in sorted(os.listdir(subject_path)):
            label_path = os.path.join(subject_path, label)
            if not os.path.isdir(label_path):
                continue
            for video_folder in tqdm(
                sorted(os.listdir(label_path)),
                desc=f"{os.path.basename(dataset_path)}/{subject}/{label}",
                leave=False,
            ):
                video_path = os.path.join(label_path, video_folder)
                if not os.path.isdir(video_path):
                    continue
                frames = get_sorted_frames(video_path)
                out_base = os.path.join(out_dataset, subject, label, video_folder)
                n = build_clips_for_video(frames, out_base)
                total_clips += n
    return total_clips


def process_dataset_yawdd(dataset_path: str, out_dataset: str) -> int:
    """
    Handle YawDD structure:
      yawdd/<label>/<video_folder>/frames…
    """
    total_clips = 0
    for label in sorted(os.listdir(dataset_path)):
        label_path = os.path.join(dataset_path, label)
        if not os.path.isdir(label_path):
            continue
        for video_folder in tqdm(
            sorted(os.listdir(label_path)),
            desc=f"yawdd/{label}",
            leave=False,
        ):
            video_path = os.path.join(label_path, video_folder)
            if not os.path.isdir(video_path):
                continue
            frames = get_sorted_frames(video_path)
            out_base = os.path.join(out_dataset, label, video_folder)
            n = build_clips_for_video(frames, out_base)
            total_clips += n
    return total_clips


def process_dataset_uta(dataset_path: str, out_dataset: str) -> int:
    """
    Handle UTA-RLDD structure:
      uta_rldd/<subject>/<state>/frames…
    (state is a single video worth of frames, no clip subfolders)
    """
    total_clips = 0
    for subject in sorted(os.listdir(dataset_path)):
        subject_path = os.path.join(dataset_path, subject)
        if not os.path.isdir(subject_path):
            continue
        for state in tqdm(
            sorted(os.listdir(subject_path)),
            desc=f"uta_rldd/{subject}",
            leave=False,
        ):
            state_path = os.path.join(subject_path, state)
            if not os.path.isdir(state_path):
                continue
            frames = get_sorted_frames(state_path)
            out_base = os.path.join(out_dataset, subject, state)
            n = build_clips_for_video(frames, out_base)
            total_clips += n
    return total_clips


# ── Dataset dispatch ─────────────────────────────────────────────────────────
DATASET_HANDLERS = {
    "nthu_ddd":  process_dataset_standard,
    "nitymed":   process_dataset_standard,
    "yawdd":     process_dataset_yawdd,
    "uta_rldd":  process_dataset_uta,
}


def main():
    print("\n" + "=" * 70)
    print("Clip Builder — Sliding-window clip generation")
    print(f"  Clip length : {CLIP_LEN} frames")
    print(f"  Stride      : {STRIDE} frames")
    print(f"  Input root  : {CROPPED_ROOT}")
    print(f"  Output root : {CLIPS_ROOT}")
    print("=" * 70 + "\n")

    os.makedirs(CLIPS_ROOT, exist_ok=True)
    grand_total = 0

    for dataset_name in sorted(os.listdir(CROPPED_ROOT)):
        dataset_path = os.path.join(CROPPED_ROOT, dataset_name)
        if not os.path.isdir(dataset_path):
            continue

        out_dataset = os.path.join(CLIPS_ROOT, dataset_name)
        os.makedirs(out_dataset, exist_ok=True)

        handler = DATASET_HANDLERS.get(dataset_name.lower())
        if handler is None:
            print(f"[WARN] No handler for dataset '{dataset_name}', skipping.")
            continue

        print(f"[INFO] Processing dataset: {dataset_name}")
        n_clips = handler(dataset_path, out_dataset)
        print(f"  → {n_clips} clips created in {out_dataset}")
        grand_total += n_clips

    print(f"\n{'=' * 70}")
    print(f"Clip building complete. Total clips: {grand_total}")
    print(f"Output: {CLIPS_ROOT}/<dataset>/<subject>/<label>/<video>_clip<N>/")


if __name__ == "__main__":
    main()
