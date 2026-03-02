import os
import cv2
import csv
from tqdm import tqdm

# CONFIGURABLES


# Set these to the parent folder containing all datasets
METADATA_CSV = "data/metadata.csv"
OUTPUT_FRAME_ROOT = "dataset/processed"
FPS = 15
IMG_SIZE = 224
SAVE_PNG = False
SKIP_IF_DONE = True
UTA_RLDD_SUBJECT_LIMIT = 15


# Helper to ensure output directory exists
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def normalize_rel_path(path):
    return path.replace("\\", "/")


def extract_subject_from_path(video_path):
    parts = normalize_rel_path(video_path).split("/")
    return parts[1] if len(parts) > 2 else None


def extract_frames_from_video(video_path, out_dir, target_fps=FPS):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Failed to open video: {video_path}")
        return 0
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps == 0:
        video_fps = 30
    step = max(1, int(video_fps / target_fps))
    idx = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            frame_name = f"frame_{idx:06d}{'.png' if SAVE_PNG else '.jpg'}"
            out_path = os.path.join(out_dir, frame_name)
            try:
                if SAVE_PNG:
                    cv2.imwrite(out_path, frame, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
                else:
                    cv2.imwrite(out_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                saved += 1
            except Exception as e:
                print(f"[ERROR] Could not save frame {frame_name}: {e}")
        idx += 1
    cap.release()
    return saved


def get_output_dir_from_metadata(row):
    dataset_raw = row["dataset"]
    dataset = dataset_raw.strip().lower()
    video_path = normalize_rel_path(row["video_path"])
    video_base = os.path.splitext(os.path.basename(video_path))[0]
    parts = video_path.split("/")

    if dataset == "nthu_ddd":
        subject = parts[1]
        state = parts[2]
        return os.path.join(OUTPUT_FRAME_ROOT, "nthu_ddd", subject, state, video_base)
    elif dataset == "testing_dataset_nthu":
        # Match processed structure: processed/testing_nthu/<subject_condition_mix>
        return os.path.join(OUTPUT_FRAME_ROOT, "testing_nthu", video_base)
    elif dataset == "uta_rldd":
        subject = parts[1]
        state = os.path.splitext(parts[2])[0]
        # Match processed structure: processed/uta_rldd/<subject>/<state>
        return os.path.join(OUTPUT_FRAME_ROOT, "uta_rldd", subject, state)
    elif dataset == "nitymed":
        subject = parts[1]
        state = parts[2]
        return os.path.join(OUTPUT_FRAME_ROOT, "nitymed", subject, state, video_base)
    elif dataset == "yawdd":
        # Match processed structure: processed/yawdd/<state>/<participant_clip>
        state = parts[1]
        return os.path.join(OUTPUT_FRAME_ROOT, "yawdd", state, video_base)
    else:
        return os.path.join(OUTPUT_FRAME_ROOT, dataset_raw, video_base)


def main():
    print("\n" + "=" * 70)
    print("WebDrowseNet - Unified Frame Extraction Pipeline (metadata-driven)")
    print("=" * 70)

    frames_root = OUTPUT_FRAME_ROOT
    if not os.path.exists(frames_root):
        os.makedirs(frames_root)

    total_videos = 0
    total_frames = 0
    selected_uta_subjects = set()

    with open(METADATA_CSV, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            dataset = row["dataset"].strip().lower()
            if dataset == "uta_rldd":
                subject = extract_subject_from_path(row["video_path"])
                if subject is None:
                    print(f"[WARN] Invalid UTA path format, skipping: {row['video_path']}")
                    continue
                if len(selected_uta_subjects) >= UTA_RLDD_SUBJECT_LIMIT and subject not in selected_uta_subjects:
                    continue
                selected_uta_subjects.add(subject)

            abs_video_path = os.path.join("dataset", row["video_path"])
            out_dir = get_output_dir_from_metadata(row)
            if SKIP_IF_DONE and os.path.isdir(out_dir) and len(os.listdir(out_dir)) > 0:
                print(f"  [OK] Frames already exist for {out_dir}, skipping extraction")
                continue
            os.makedirs(out_dir, exist_ok=True)
            frames = extract_frames_from_video(abs_video_path, out_dir, target_fps=FPS)
            if frames == 0:
                continue
            total_videos += 1
            total_frames += frames
            print(f"  [OK] Saved {frames} frames for {out_dir}")

    print(f"\nExtraction complete: {total_videos} videos, {total_frames} frames saved.")
    print(
        f"UTA-RLDD subjects processed (max {UTA_RLDD_SUBJECT_LIMIT}): "
        f"{', '.join(sorted(selected_uta_subjects)) if selected_uta_subjects else 'none'}"
    )
    print(f"Location: {frames_root}/<dataset>/... (see PROCESSED_STRUCTURE.md)")


if __name__ == "__main__":
    main()
