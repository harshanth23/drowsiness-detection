# Allow `python src/inference_realtime.py` to resolve `from src.*` imports
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..")))
"""
src/inference_realtime.py — Live Webcam Drowsiness Demo
---------------------------------------------------------
Pipeline (per frame, as per plan.md):
  1. Capture frame via OpenCV webcam
  2. Face detection — InsightFace RetinaFace
  3. Crop & resize face ROI to FRAME_SIZE × FRAME_SIZE
  4. MediaPipe landmark extraction → compute EAR and MOR
  5. Push frame + [EAR, MOR] into rolling buffers (len = T)
  6. Every INFERENCE_STRIDE frames: stack → 3D-CNN+LSTM → drowsiness score
  7. Overlay prediction label ("Alert 🟢" / "Drowsy 🔴") + EAR/MOR on frame
  8. Display via cv2.imshow

Usage:
    conda activate webenv
    python src/inference_realtime.py --model results/checkpoints/best_model.pt
    python src/inference_realtime.py --model results/checkpoints/best_model.pt --camera 0
"""

import argparse
import collections
import time

import cv2
import mediapipe as mp
import numpy as np
import torch
import insightface
from insightface.app import FaceAnalysis
from torchvision import transforms
import yaml

from src.dataset import FRAME_SIZE, T_FRAMES
from src.model import DrowsinessModel
from src.utils import compute_ear, compute_mor, get_device, load_checkpoint


# ─────────────────────────────────────────────────────────────────────────────
# Transforms (eval mode, no augmentation)
# ─────────────────────────────────────────────────────────────────────────────

_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

_FRAME_TFM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])

# ─────────────────────────────────────────────────────────────────────────────
# Display helpers
# ─────────────────────────────────────────────────────────────────────────────

_ALERT_COLOR  = (0, 220, 80)    # BGR green
_DROWSY_COLOR = (0, 50, 220)    # BGR red
_TEXT_COLOR   = (255, 255, 255)
_FONT         = cv2.FONT_HERSHEY_DUPLEX
_FONT_SCALE   = 0.7
_THICKNESS    = 2


def _draw_overlay(
    frame:      np.ndarray,
    label:      str,
    score:      float,
    ear:        float,
    mor:        float,
    fps:        float,
    bbox:       tuple | None,
) -> np.ndarray:
    out = frame.copy()

    # Face bounding box
    if bbox is not None:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        color = _DROWSY_COLOR if label == "Drowsy" else _ALERT_COLOR
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

    # Status bar (top-left)
    bg_color = _DROWSY_COLOR if label == "Drowsy" else _ALERT_COLOR
    cv2.rectangle(out, (0, 0), (320, 38), bg_color, -1)
    status_txt = f"{label}  {score * 100:.1f}%"
    cv2.putText(out, status_txt, (8, 26), _FONT, _FONT_SCALE, _TEXT_COLOR, _THICKNESS)

    # EAR / MOR / FPS (bottom-left)
    h, w = out.shape[:2]
    cv2.rectangle(out, (0, h - 70), (220, h), (20, 20, 20), -1)
    cv2.putText(out, f"EAR: {ear:.3f}", (8, h - 48), _FONT, 0.55, _TEXT_COLOR, 1)
    cv2.putText(out, f"MOR: {mor:.3f}", (8, h - 24), _FONT, 0.55, _TEXT_COLOR, 1)
    cv2.putText(out, f"FPS: {fps:.1f}", (130, h - 24), _FONT, 0.55, (180, 255, 180), 1)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main inference loop
# ─────────────────────────────────────────────────────────────────────────────

def run_realtime(config: dict) -> None:
    inf_cfg  = config.get("inference", {})
    model_path     = inf_cfg.get("model_path", "results/checkpoints/best_model.pt")
    camera_idx     = inf_cfg.get("camera_index", 0)
    threshold      = inf_cfg.get("alert_threshold", 0.5)
    T              = inf_cfg.get("buffer_size", T_FRAMES)
    stride         = inf_cfg.get("inference_stride", 2)
    show_ear_mor   = inf_cfg.get("display_ear_mor", True)
    use_lm         = config["data"].get("use_landmarks", True)

    device = get_device()

    # ── Load model ────────────────────────────────────────────────────────────
    print("[demo] Loading model …")
    model = DrowsinessModel.from_config(config).to(device)
    load_checkpoint(model_path, model, device=str(device))
    model.eval()

    # ── InsightFace face detector ─────────────────────────────────────────────
    print("[demo] Initialising InsightFace …")
    app = FaceAnalysis(name="buffalo_sc", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(320, 320))

    # ── MediaPipe FaceMesh for landmarks ─────────────────────────────────────
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh    = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # ── Rolling buffers ───────────────────────────────────────────────────────
    frame_buf: collections.deque = collections.deque(maxlen=T)
    lm_buf:    collections.deque = collections.deque(maxlen=T)

    # ── Webcam ────────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {camera_idx}")
        return

    print(f"[demo] Running on camera {camera_idx}. Press 'q' to quit.\n")

    label, score, ear_val, mor_val = "Initialising …", 0.0, 0.0, 0.0
    frame_count, fps_ema = 0, 0.0
    t_last = time.perf_counter()
    last_bbox = None

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            print("[ERROR] Frame capture failed.")
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_count += 1

        # ── Face detection ────────────────────────────────────────────────────
        faces = app.get(frame_bgr)
        face_crop = None
        bbox      = None

        if faces:
            face  = faces[0]
            bbox  = face.bbox   # [x1, y1, x2, y2]
            x1, y1, x2, y2 = [int(v) for v in bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame_bgr.shape[1], x2), min(frame_bgr.shape[0], y2)
            crop_rgb  = frame_rgb[y1:y2, x1:x2]
            if crop_rgb.size > 0:
                face_crop = cv2.resize(crop_rgb, (FRAME_SIZE, FRAME_SIZE))
                last_bbox = bbox

        # Fallback to centre crop if no face detected
        if face_crop is None:
            H, W = frame_rgb.shape[:2]
            s    = min(H, W)
            y0   = (H - s) // 2;  x0 = (W - s) // 2
            face_crop = cv2.resize(frame_rgb[y0:y0+s, x0:x0+s], (FRAME_SIZE, FRAME_SIZE))

        # ── Landmark extraction ───────────────────────────────────────────────
        if use_lm:
            lm_result = face_mesh.process(face_crop)
            if lm_result.multi_face_landmarks:
                lm = lm_result.multi_face_landmarks[0].landmark
                lm_flat = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32).flatten()
                ear_val = compute_ear(lm_flat)
                mor_val = compute_mor(lm_flat)
                lm_buf.append([ear_val, mor_val])
            else:
                lm_buf.append([ear_val, mor_val])  # keep last known
        else:
            lm_buf.append([0.0, 0.0])

        # ── Frame buffer ──────────────────────────────────────────────────────
        face_tensor = _FRAME_TFM(face_crop)   # (3, H, W)
        frame_buf.append(face_tensor)

        # ── Model inference ───────────────────────────────────────────────────
        if len(frame_buf) == T and frame_count % stride == 0:
            clip_t  = torch.stack(list(frame_buf), dim=1).unsqueeze(0).to(device)  # (1,3,T,H,W)
            lm_t    = torch.tensor(list(lm_buf), dtype=torch.float32).unsqueeze(0).to(device)  # (1,T,2)

            with torch.no_grad():
                logits = model(clip_t, lm_t if use_lm else None)
                probs  = torch.softmax(logits, dim=1)[0]

            drowsy_prob = probs[1].item()
            score       = drowsy_prob
            label       = "Drowsy" if drowsy_prob >= threshold else "Alert"

        # ── FPS ───────────────────────────────────────────────────────────────
        t_now   = time.perf_counter()
        elapsed = t_now - t_last
        t_last  = t_now
        fps_ema = 0.9 * fps_ema + 0.1 * (1.0 / (elapsed + 1e-8))

        # ── Display ───────────────────────────────────────────────────────────
        out_frame = _draw_overlay(
            frame_bgr if len(frame_bgr.shape) == 3 else frame_rgb,
            label, score, ear_val, mor_val, fps_ema,
            last_bbox,
        )
        cv2.imshow("Drowsiness Detection — press Q to quit", out_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()
    print("[demo] Done.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Real-time Drowsiness Detection Demo")
    p.add_argument("--model",   type=str, default=None,
                   help="Path to checkpoint (overrides config)")
    p.add_argument("--config",  type=str, default="config.yaml")
    p.add_argument("--camera",  type=int, default=None)
    p.add_argument("--no_lm",   action="store_true", help="Disable landmarks")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.model:
        cfg["inference"]["model_path"] = args.model
    if args.camera is not None:
        cfg["inference"]["camera_index"] = args.camera
    if args.no_lm:
        cfg["data"]["use_landmarks"]  = False
        cfg["model"]["use_landmarks"] = False

    run_realtime(cfg)
