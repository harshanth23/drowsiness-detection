"""
face_detection.py — Standalone InsightFace face detection utility
-----------------------------------------------------------------
Provides reusable helpers for:
  - Single-frame face detection (bounding box + landmarks)
  - Batch detection over a folder of images
  - EAR (Eye Aspect Ratio) and MOR (Mouth Opening Ratio) from 5-point landmarks

This module is imported by inference_realtime.py and can also be run
stand-alone to test detection on a single image or a folder.

InsightFace 5-point landmark order (buffalo_l model):
  0 = left eye    1 = right eye    2 = nose tip
  3 = left mouth corner    4 = right mouth corner

Run from project root:
  conda activate webenv
  python preprocessing/face_detection.py --image <path>
  python preprocessing/face_detection.py --folder <path>
"""

import argparse
import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis

# ── Configurables ─────────────────────────────────────────────────────────────
FACE_DET_SIZE = (320, 320)   # RetinaFace internal resolution
IMG_SIZE      = 112          # Default crop output size
# ─────────────────────────────────────────────────────────────────────────────


class FaceDetector:
    """Thin wrapper around InsightFace RetinaFace for reuse across scripts."""

    def __init__(self, det_size: tuple[int, int] = FACE_DET_SIZE, ctx_id: int = 0):
        self.app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection"])
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)

    def detect(self, bgr_frame: np.ndarray) -> list:
        """
        Detect faces in *bgr_frame* (BGR uint8 numpy array).
        Returns a list of InsightFace Face objects sorted by bbox area (largest first).
        Each face has:
            .bbox       → [x1, y1, x2, y2] float32
            .kps        → (5, 2) float32  landmark points
            .det_score  → float confidence
        """
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        faces = self.app.get(rgb)
        # Sort largest face first
        faces = sorted(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
        return faces

    def crop_face(
        self,
        bgr_frame: np.ndarray,
        face,
        out_size: int = IMG_SIZE,
        padding: float = 0.0,
    ) -> np.ndarray | None:
        """
        Crop the largest face from *bgr_frame* using *face*.bbox, with optional
        fractional *padding* around the bbox. Returns BGR image or None.
        """
        x1, y1, x2, y2 = face.bbox.astype(int)
        h, w = bgr_frame.shape[:2]
        if padding > 0:
            pad_x = int((x2 - x1) * padding)
            pad_y = int((y2 - y1) * padding)
            x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
            x2, y2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
        crop = bgr_frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        return cv2.resize(crop, (out_size, out_size))

    def detect_and_crop(
        self,
        bgr_frame: np.ndarray,
        out_size: int = IMG_SIZE,
        padding: float = 0.0,
    ) -> tuple[np.ndarray | None, list]:
        """
        Convenience method: detect + crop largest face.
        Returns (cropped_bgr_or_None, face_list).
        """
        faces = self.detect(bgr_frame)
        if not faces:
            return None, []
        crop = self.crop_face(bgr_frame, faces[0], out_size=out_size, padding=padding)
        return crop, faces


# ── EAR / MOR helpers using 5-point InsightFace landmarks ────────────────────

def ear_from_kps(kps: np.ndarray) -> float:
    """
    Approximate Eye Aspect Ratio from 5-point landmarks.
    kps shape: (5, 2)  [left_eye, right_eye, nose, left_mouth, right_mouth]

    EAR = average inter-eye distance proxy (simplified; for full EAR use
    68-point Dlib landmarks or MediaPipe 478-point mesh).
    Returns a normalised value in [0, 1] relative to inter-eye distance.
    """
    left_eye  = kps[0]
    right_eye = kps[1]
    # With only 2 eye points we approximate: EAR is the ratio of
    # vertical eye opening to horizontal distance.  Because InsightFace
    # 5-pt gives centre points only, we return the normalised inter-eye
    # distance as a rough proxy.  For accurate EAR use MediaPipe mesh.
    inter_eye_dist = float(np.linalg.norm(right_eye - left_eye))
    return inter_eye_dist   # caller should normalise by face width


def mor_from_kps(kps: np.ndarray) -> float:
    """
    Mouth Opening Ratio from 5-point landmarks.
    kps[3] = left mouth corner, kps[4] = right mouth corner.
    MOR = mouth width / inter-eye distance  (no vertical mouth openness
    available without full mesh — use MediaPipe for that).
    """
    mouth_width   = float(np.linalg.norm(kps[4] - kps[3]))
    inter_eye_dist = float(np.linalg.norm(kps[1] - kps[0]))
    if inter_eye_dist == 0:
        return 0.0
    return mouth_width / inter_eye_dist


# ── CLI entry point ───────────────────────────────────────────────────────────

def _demo_image(detector: FaceDetector, img_path: str):
    """Run detection on a single image and display results."""
    frame = cv2.imread(img_path)
    if frame is None:
        print(f"[ERROR] Cannot read image: {img_path}")
        return

    faces = detector.detect(frame)
    print(f"Detected {len(faces)} face(s) in {img_path}")
    for i, face in enumerate(faces):
        x1, y1, x2, y2 = face.bbox.astype(int)
        score = float(face.det_score)
        print(f"  Face {i}: bbox=({x1},{y1},{x2},{y2})  conf={score:.3f}")
        if face.kps is not None:
            ear = ear_from_kps(face.kps)
            mor = mor_from_kps(face.kps)
            print(f"           EAR-proxy={ear:.1f}px  MOR={mor:.3f}")
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{score:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    out_path = img_path.replace(".", "_detected.", 1)
    cv2.imwrite(out_path, frame)
    print(f"Saved annotated image: {out_path}")


def _demo_folder(detector: FaceDetector, folder: str):
    """Detect faces in all .jpg images in *folder* and print a summary."""
    images = sorted([
        os.path.join(folder, f) for f in os.listdir(folder)
        if f.lower().endswith(".jpg")
    ])
    print(f"Processing {len(images)} images in {folder} ...")
    total_detected = 0
    for img_path in images:
        frame = cv2.imread(img_path)
        if frame is None:
            continue
        faces = detector.detect(frame)
        total_detected += len(faces)
    print(f"Done. Total faces detected: {total_detected} across {len(images)} images.")


def main():
    parser = argparse.ArgumentParser(description="InsightFace face detection utility")
    parser.add_argument("--image",  type=str, default=None, help="Path to a single image")
    parser.add_argument("--folder", type=str, default=None, help="Path to a folder of images")
    args = parser.parse_args()

    detector = FaceDetector()

    if args.image:
        _demo_image(detector, args.image)
    elif args.folder:
        _demo_folder(detector, args.folder)
    else:
        print("Usage:")
        print("  python preprocessing/face_detection.py --image <path>")
        print("  python preprocessing/face_detection.py --folder <path>")


if __name__ == "__main__":
    main()
