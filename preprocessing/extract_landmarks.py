"""
extract_landmarks.py — MediaPipe FaceMesh landmark extraction for all datasets
-------------------------------------------------------------------------------
Reads face-cropped frames from dataset/processed_cropped/<dataset>/...
and saves per-frame 478-point MediaPipe landmarks as .npy files in
dataset/landmarks_cropped/<dataset>/...

Handles each dataset's actual directory structure:
  standard  (nthu_ddd, nitymed): subject / label / video_clip / frames
  yawdd:                         label / video_clip / frames
  uta:                           subject / state / frames  (no clip subfolder)

Run from project root:
  conda activate webenv
  python preprocessing/extract_landmarks.py
"""

import os
import numpy as np
from PIL import Image
import mediapipe as mp
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────
DATASET_ROOT  = 'dataset/processed_cropped'
LANDMARKS_ROOT = 'dataset/landmarks_cropped'

# Dataset layout:  'standard' | 'yawdd' | 'uta'
DATASET_LAYOUTS = {
    'nthu_ddd':      'standard',   # subject/label/clip/frames
    'nitymed':       'standard',   # subject/label/clip/frames
    'yawdd':         'yawdd',      # label/video/frames
    'uta_rldd':      'uta',        # subject/state/frames (no clip subfolder)
    'testing_nthu':  'testing',    # video_name/frames (flat, no label)
}
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(LANDMARKS_ROOT, exist_ok=True)

mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True, max_num_faces=1, refine_landmarks=True
)


def extract_and_save(frame_path: str, out_path: str):
    """Run MediaPipe FaceMesh on one frame, save 478×3 array as .npy."""
    if os.path.exists(out_path):
        return
    image = Image.open(frame_path).convert('RGB')
    img_np = np.array(image)
    results = mp_face_mesh.process(img_np)
    if results.multi_face_landmarks:
        lms = results.multi_face_landmarks[0].landmark
        arr = np.array([[lm.x, lm.y, lm.z] for lm in lms]).flatten().astype(np.float32)
    else:
        arr = np.zeros(478 * 3, dtype=np.float32)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, arr)


def process_frames_in_dir(in_dir: str, out_dir: str):
    """Extract landmarks for all .jpg frames in in_dir → out_dir."""
    frames = [f for f in os.listdir(in_dir) if f.endswith('.jpg')]
    for frame_file in tqdm(frames, desc=os.path.basename(in_dir), leave=False):
        frame_path = os.path.join(in_dir, frame_file)
        out_path   = os.path.join(out_dir, frame_file.replace('.jpg', '.npy'))
        extract_and_save(frame_path, out_path)


# ── Layout handlers ───────────────────────────────────────────────────────────

def handle_standard(ds_in: str, ds_out: str):
    """nthu_ddd, nitymed: subject / label / video_clip / frames"""
    for subject in sorted(os.listdir(ds_in)):
        sp = os.path.join(ds_in, subject)
        if not os.path.isdir(sp): continue
        for label in sorted(os.listdir(sp)):
            lp = os.path.join(sp, label)
            if not os.path.isdir(lp): continue
            for clip in tqdm(sorted(os.listdir(lp)), desc=f'{subject}/{label}', leave=False):
                cp = os.path.join(lp, clip)
                if not os.path.isdir(cp): continue
                out_dir = os.path.join(ds_out, subject, label, clip)
                process_frames_in_dir(cp, out_dir)


def handle_yawdd(ds_in: str, ds_out: str):
    """yawdd: label / video_clip / frames"""
    for label in sorted(os.listdir(ds_in)):
        lp = os.path.join(ds_in, label)
        if not os.path.isdir(lp): continue
        for video in tqdm(sorted(os.listdir(lp)), desc=f'yawdd/{label}', leave=False):
            vp = os.path.join(lp, video)
            if not os.path.isdir(vp): continue
            out_dir = os.path.join(ds_out, label, video)
            process_frames_in_dir(vp, out_dir)


def handle_uta(ds_in: str, ds_out: str):
    """uta_rldd: subject / state / frames (frames sit directly in state folder)"""
    for subject in sorted(os.listdir(ds_in)):
        sp = os.path.join(ds_in, subject)
        if not os.path.isdir(sp): continue
        for state in tqdm(sorted(os.listdir(sp)), desc=f'uta/{subject}', leave=False):
            stp = os.path.join(sp, state)
            if not os.path.isdir(stp): continue
            out_dir = os.path.join(ds_out, subject, state)
            process_frames_in_dir(stp, out_dir)


def handle_testing(ds_in: str, ds_out: str):
    """testing_nthu: video_name / frames (flat)"""
    for video in tqdm(sorted(os.listdir(ds_in)), desc='testing_nthu', leave=False):
        vp = os.path.join(ds_in, video)
        if not os.path.isdir(vp): continue
        out_dir = os.path.join(ds_out, video)
        process_frames_in_dir(vp, out_dir)


HANDLERS = {
    'standard': handle_standard,
    'yawdd':    handle_yawdd,
    'uta':      handle_uta,
    'testing':  handle_testing,
}


def main():
    print('\n' + '=' * 70)
    print('Landmark Extraction — MediaPipe FaceMesh 478-point (all datasets)')
    print('=' * 70 + '\n')

    for dataset in sorted(os.listdir(DATASET_ROOT)):
        ds_in  = os.path.join(DATASET_ROOT,  dataset)
        ds_out = os.path.join(LANDMARKS_ROOT, dataset)
        if not os.path.isdir(ds_in):
            continue

        layout  = DATASET_LAYOUTS.get(dataset.lower())
        handler = HANDLERS.get(layout) if layout else None

        if handler is None:
            print(f'[WARN] No layout defined for dataset "{dataset}", skipping.')
            continue

        print(f'[INFO] {dataset}  (layout={layout})')
        os.makedirs(ds_out, exist_ok=True)
        handler(ds_in, ds_out)
        print(f'  → done: {ds_out}')

    print('\nLandmark extraction complete.')


if __name__ == '__main__':
    main()
