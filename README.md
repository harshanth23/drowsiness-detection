# Driver Drowsiness Detection — 3D-CNN + LSTM Multi-Dataset System

A deep learning system for real-time driver drowsiness detection using a **3D-CNN + LSTM** hybrid architecture with optional **facial landmark** features. The project trains on multiple public face-video datasets and includes **cross-dataset generalization** experiments to evaluate robustness across different recording conditions, cameras, and subjects.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Datasets](#datasets)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Data Preprocessing Pipeline](#data-preprocessing-pipeline)
- [Training](#training)
- [Evaluation](#evaluation)
- [Cross-Dataset Generalization](#cross-dataset-generalization)
- [Grad-CAM Visualization](#grad-cam-visualization)
- [Real-Time Inference](#real-time-inference)
- [Configuration](#configuration)
- [Results & Outputs](#results--outputs)

---

## Overview

The system performs binary classification (**Alert** vs. **Drowsy**) on short video clips of a driver's face. Key highlights:

- **3D-CNN backbone** processes 16-frame clips at 112×112 resolution for spatiotemporal feature learning
- **LSTM temporal head** captures sequential drowsiness patterns across frames
- **Parallel landmark branch** fuses Eye Aspect Ratio (EAR) and Mouth Opening Ratio (MOR) signals extracted via MediaPipe FaceMesh
- **Cross-dataset generalization** — train on one dataset pair, evaluate on completely unseen datasets
- **Grad-CAM visual explanations** — heatmaps showing which facial regions the model attends to
- **Real-time webcam demo** with live drowsiness prediction overlay

---

## Architecture

### Model: 3D-CNN + LSTM Hybrid

```
Input: (B, 3, 16, 112, 112) — 16-frame RGB clip
                │
    ┌───────────┴───────────┐
    │     3D-CNN Backbone    │
    │  4× Conv3D Blocks      │
    │  (3→16→32→64→128 ch)   │
    │  + BN + ReLU + Pool    │
    └───────────┬───────────┘
                │ (B, 128, 2, 7, 7)
         Spatial Flatten
                │ (B, 2, 6272)
         Linear → 512-dim
                │
    ┌───────────┴───────────┐
    │   Temporal LSTM        │
    │   256 hidden, 1-2 layers│
    └───────────┬───────────┘
                │ (B, 256)
                │                    ┌──────────────────┐
                │                    │ Landmark LSTM     │
                │                    │ Input: EAR + MOR  │
                │                    │ (B, 16, 2) → 64  │
                │                    └────────┬─────────┘
                │                             │
                └──────────┬──────────────────┘
                     Concatenate (B, 320)
                           │
                   Classification Head
                   FC → ReLU → Dropout → FC
                           │
                    Output: (B, 2)
                  [P(Alert), P(Drowsy)]
```

**Key design choices:**
- Temporal pooling is deferred until Block 4 to preserve temporal information for the LSTM
- The landmark branch is independent of CNN features, capturing behavioral cues (blink rate, yawning) orthogonally
- Weighted cross-entropy (or optional Focal Loss) handles class imbalance

---

## Datasets

Four public drowsiness/yawning datasets with unified binary labeling:

| Dataset | Subjects | Format | Original Labels | Unified Label |
|---------|----------|--------|-----------------|--------------|
| **NTHU-DDD** | 18 | .avi | Alert, Drowsy | Alert (0) / Drowsy (1) |
| **UTA-RLDD** | 36 | .mov | Alert, Drowsy, Low Vigilance | Alert (0) / Drowsy (1) |
| **NITYMED** | 7 | .mp4 | Microsleep, Yawning | Alert (0) / Drowsy (1) |
| **YawDD** | Multiple | .avi | Normal, Yawning | Alert (0) / Drowsy (1) |
| **Testing_NTHU** | 14 | .mp4 | Mixed | Held-out test |

**Split strategy:**
- **Subject-wise** 80% train / 10% val / 10% test (prevents identity leakage)
- **UTA-RLDD** uses official 5-fold cross-validation (configurable via `uta_fold`)
- **YawDD** uses video-wise splits

---

## Project Structure

```
Drowsiness Detection/
├── config.yaml                    # Main training configuration
├── requirements.txt               # Python dependencies
├── _smoke_train.py                # Quick 2-epoch GPU validation test
│
├── data/
│   ├── metadata.csv               # Master clip-level metadata
│   └── splits/                    # Train/val/test split JSONs per dataset
│       ├── nthu_ddd_split.json
│       ├── uta_rldd_5fold.json
│       ├── yawdd_split.json
│       ├── nitymed_split.json
│       └── testing_nthu_split.json
│
├── dataset/
│   ├── README_dataset.md
│   ├── nthu_ddd/                  # Raw NTHU-DDD videos
│   ├── uta_rldd/                  # Raw UTA-RLDD videos
│   ├── nitymed/                   # Raw NITYMED videos
│   ├── yawdd/                     # Raw YawDD videos
│   ├── Testing_Dataset_NTHU/      # Held-out NTHU test set
│   ├── processed/                 # Extracted frames (224×224)
│   ├── processed_cropped/         # Face-cropped frames (112×112)
│   ├── clips/                     # 16-frame clip folders (model input)
│   └── landmarks_cropped/         # MediaPipe landmark .npy files
│
├── experiments/
│   ├── exp_A.yaml                 # Cross-eval: Train NTHU+YawDD → Test UTA+NITYMED
│   └── exp_B.yaml                 # Cross-eval: Train UTA+NITYMED → Test NTHU+YawDD
│
├── preprocessing/                 # Data pipeline scripts (run once)
│   ├── frame_extraction.py        # Videos → frames at 15 FPS
│   ├── face_cropper.py            # Frames → face-cropped 112×112 (InsightFace)
│   ├── face_detection.py          # Standalone face detection utility
│   ├── extract_landmarks.py       # Cropped faces → 478-point landmarks (MediaPipe)
│   ├── clip_builder.py            # Frames → 16-frame overlapping clips (stride 8)
│   ├── subject_split.py           # Generate train/val/test split JSONs
│   └── create_metadata.py         # Build metadata.csv from clip directories
│
├── src/                           # Core ML pipeline
│   ├── model.py                   # DrowsinessModel (3D-CNN + LSTM + landmarks)
│   ├── dataset.py                 # DataLoader, transforms, record building
│   ├── train.py                   # Single-config training loop
│   ├── evaluate.py                # Metrics, confusion matrices, ROC curves
│   ├── cross_train.py             # Cross-dataset training (Exp A/B)
│   ├── cross_evaluate.py          # Cross-dataset evaluation
│   ├── gradcam.py                 # Grad-CAM heatmap generation
│   ├── find_gradcam_candidates.py # Auto-select best clips for Grad-CAM
│   ├── plot_generalization.py     # Seen vs. unseen comparison plots
│   ├── inference_realtime.py      # Live webcam drowsiness detection
│   └── utils.py                   # Seeds, checkpoints, metrics, plotting helpers
│
└── results/
    ├── checkpoints/               # Model weights (best_model.pt, epoch snapshots)
    ├── logs/                      # TensorBoard events + CSV training logs
    ├── plots/                     # Training curves, confusion matrices, ROC, Grad-CAM
    └── cross_eval/                # Cross-experiment outputs (exp_A/, exp_B/)
```

---

## Installation

### Prerequisites

- Python 3.9+
- CUDA-compatible GPU (recommended; CPU fallback supported)
- Conda (recommended for environment management)

### Setup

```bash
# Create and activate environment
conda create -n webenv python=3.10
conda activate webenv

# Install PyTorch (adjust for your CUDA version — see https://pytorch.org)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install remaining dependencies
pip install opencv-python mediapipe insightface numpy scikit-learn matplotlib tqdm pyyaml tensorboard pillow
```

### Dependencies

| Package | Purpose |
|---------|---------|
| `torch`, `torchvision` | Deep learning framework |
| `opencv-python` (cv2) | Frame extraction, image I/O |
| `mediapipe` | FaceMesh landmark extraction |
| `insightface` | RetinaFace face detection & cropping |
| `numpy` | Numerical operations |
| `scikit-learn` | Metrics (F1, AUC, confusion matrix) |
| `matplotlib` | Plot generation |
| `tqdm` | Progress bars |
| `pyyaml` | Configuration loading |
| `tensorboard` | Training visualization |
| `Pillow` (PIL) | Image processing |

---

## Data Preprocessing Pipeline

Run these scripts **once** in order to prepare raw videos for training:

```bash
cd "D:\Class Work\Open Lab\Drowsiness Detection"
conda activate webenv
```

### Step 1 — Extract Frames

Reads raw videos and saves individual frames at 15 FPS, resized to 224×224.

```bash
python preprocessing/frame_extraction.py
```
Output: `dataset/processed/<dataset>/<subject>/<label>/<video>/frame_XXXXXX.jpg`

### Step 2 — Crop Faces

Detects the largest face in each frame using InsightFace RetinaFace and crops to 112×112.

```bash
python preprocessing/face_cropper.py
```
Output: `dataset/processed_cropped/<dataset>/<subject>/<label>/<video>/frame_XXXXXX.jpg`

### Step 3 — Extract Landmarks

Runs MediaPipe FaceMesh on cropped faces to extract 478-point 3D facial landmark arrays.

```bash
python preprocessing/extract_landmarks.py
```
Output: `dataset/landmarks_cropped/<dataset>/<subject>/<label>/<video>/frame_XXXXXX.npy`

### Step 4 — Build Clips

Groups consecutive frames into overlapping 16-frame clips with stride 8.

```bash
python preprocessing/clip_builder.py
```
Output: `dataset/clips/<dataset>/<subject>/<label>/<video>_clipNNNN/frame_XXXXXX.jpg`

### Step 5 — Generate Splits

Creates subject-wise train/val/test split JSON files.

```bash
python preprocessing/subject_split.py
```
Output: `data/splits/<dataset>_split.json`

### Step 6 — Build Metadata (optional)

Generates a master CSV mapping clips to labels and subjects.

```bash
python preprocessing/create_metadata.py
```
Output: `data/metadata.csv`

---

## Training

### Standard Training (all datasets)

```bash
python src/train.py --config config.yaml
```

### Key Training Features

| Feature | Details |
|---------|---------|
| Optimizer | Adam (lr=1e-4, weight_decay=1e-4) |
| Scheduler | Cosine Annealing (η_min=1e-6) |
| Loss | Weighted CrossEntropy (or Focal Loss with γ=2.0) |
| Mixed Precision | Enabled by default on CUDA |
| Gradient Clipping | max_norm=1.0 |
| Early Stopping | Patience=12, monitors macro val-F1 |
| Checkpointing | Best model + periodic snapshots every 5 epochs |

### Smoke Test

Quick 2-epoch sanity check that the GPU pipeline works:

```bash
python _smoke_train.py
```

---

## Evaluation

Run evaluation on test splits to generate metrics and visualizations:

```bash
python src/evaluate.py --config config.yaml --model results/checkpoints/best_model.pt
```

### Metrics Computed

- **Accuracy**, **Precision**, **Recall**, **Macro F1-Score**
- **ROC-AUC** with per-dataset ROC curves
- **Confusion matrices** (per dataset + combined)
- **Inference speed** (FPS benchmark)
- Per-class classification report

### Outputs

- `results/plots/<dataset>/confusion_matrix.png`
- `results/plots/<dataset>/roc_curve.png`
- `results/plots/metrics_summary.csv`

---

## Cross-Dataset Generalization

The project includes experiments to test whether models generalize to **completely unseen** datasets — a critical requirement for real-world deployment.

### Experiment A — Train: NTHU-DDD + YawDD → Test: UTA-RLDD + NITYMED

```bash
# Train
python src/cross_train.py --config experiments/exp_A.yaml

# Evaluate on seen + unseen datasets
python src/cross_evaluate.py \
    --config experiments/exp_A.yaml \
    --model results/cross_eval/exp_A/checkpoints/best_model.pt
```

### Experiment B — Train: UTA-RLDD + NITYMED → Test: NTHU-DDD + YawDD

```bash
# Train
python src/cross_train.py --config experiments/exp_B.yaml

# Evaluate
python src/cross_evaluate.py \
    --config experiments/exp_B.yaml \
    --model results/cross_eval/exp_B/checkpoints/best_model.pt
```

### Generalization Comparison Plots

```bash
python src/plot_generalization.py \
    --exp_A results/cross_eval/exp_A/plots/metrics_summary.csv \
    --exp_B results/cross_eval/exp_B/plots/metrics_summary.csv
```

Generates:
- Side-by-side bar charts (F1, AUC by dataset, colored by Seen/Unseen)
- Generalization heatmap (training set × test dataset F1 matrix)

---

## Grad-CAM Visualization

Grad-CAM produces class-discriminative heatmaps showing which face regions the model focuses on for its predictions. This is especially useful for understanding **why** the model makes errors.

### Direct Usage

```bash
python src/gradcam.py \
    --clip dataset/clips/nthu_ddd/subject001/alert/clip0000 \
    --model results/checkpoints/best_model.pt \
    --config config.yaml
```

### Auto-Find Best Candidates

Automatically identifies the most informative clips (high-confidence misclassifications + correct predictions) and generates Grad-CAM for all of them:

```bash
# Find candidates only (no Grad-CAM yet)
python src/find_gradcam_candidates.py \
    --config config.yaml \
    --model results/checkpoints/best_model.pt \
    --dataset nthu_ddd --split test

# Find candidates + auto-run Grad-CAM
python src/find_gradcam_candidates.py \
    --config experiments/exp_A.yaml \
    --model results/cross_eval/exp_A/checkpoints/best_model.pt \
    --dataset uta_rldd --split test \
    --run_gradcam \
    --out_dir results/plots/gradcam/exp_A_unseen
```

Outputs per-frame heatmap overlay grids (PNG) organized by error type (FP, FN, TP, TN).

---

## Real-Time Inference

Live webcam-based drowsiness detection with on-screen prediction overlay:

```bash
python src/inference_realtime.py \
    --model results/checkpoints/best_model.pt \
    --config config.yaml \
    --camera 0
```

### Features

- **Live face detection** and bounding box overlay
- **Rolling 16-frame buffer** for continuous prediction
- **Alert 🟢 / Drowsy 🔴** prediction display
- **EAR & MOR** metrics shown on-screen
- **FPS counter** for performance monitoring
- Configurable drowsiness probability threshold (default: 0.5)

---

## Configuration

All hyperparameters are centralized in YAML config files.

### Main Config (`config.yaml`)

```yaml
data:
  clips_root:     "dataset/clips"
  landmarks_root: "dataset/landmarks_cropped"
  use_landmarks:  true          # Enable parallel EAR/MOR branch
  uta_fold:       0             # UTA-RLDD fold (0-4)
  datasets:       null          # null = all datasets; or ["nthu_ddd", "yawdd"]

model:
  num_classes:     2             # Binary: Alert / Drowsy
  lstm_layers:     1
  lstm_hidden:     256
  lstm_input_size: 512
  use_landmarks:   true
  lm_hidden:       64
  dropout:         0.3

training:
  epochs:              30
  batch_size:          32
  learning_rate:       0.0001
  use_amp:             true       # Mixed precision
  use_focal_loss:      false      # Weighted CE by default
  early_stop_patience: 12
  grad_clip:           1.0

scheduler:
  type:    "cosine"
  T_max:   30
  eta_min: 0.000001
```

### Cross-Experiment Configs (`experiments/exp_A.yaml`, `exp_B.yaml`)

Same structure as the main config with an additional `cross` block:

```yaml
cross:
  name:           "exp_A"
  train_datasets: ["nthu_ddd", "yawdd"]
  test_datasets:  ["uta_rldd", "nitymed"]
```

---

## Results & Outputs

```
results/
├── checkpoints/
│   ├── best_model.pt              # Best validation F1 checkpoint
│   ├── epoch_005.pt               # Periodic snapshots
│   ├── epoch_010.pt
│   └── epoch_015.pt
├── logs/
│   └── train_log.csv              # Per-epoch metrics (loss, F1, acc, etc.)
├── plots/
│   ├── loss_curve.png             # Train vs. val loss
│   ├── val_f1_curve.png           # Validation F1 over epochs
│   ├── <dataset>/
│   │   ├── confusion_matrix.png
│   │   └── roc_curve.png
│   ├── metrics_summary.csv
│   └── gradcam/                   # Grad-CAM heatmap grids
└── cross_eval/
    ├── exp_A/
    │   ├── checkpoints/
    │   ├── logs/
    │   └── plots/
    │       ├── metrics_summary.csv
    │       └── seen_vs_unseen.png
    └── exp_B/
        └── ...
```

---

## Quick Start

```bash
# 1. Setup environment
conda create -n webenv python=3.10 && conda activate webenv
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install opencv-python mediapipe insightface numpy scikit-learn matplotlib tqdm pyyaml tensorboard pillow

# 2. Preprocess data (run once, in order)
python preprocessing/frame_extraction.py
python preprocessing/face_cropper.py
python preprocessing/extract_landmarks.py
python preprocessing/clip_builder.py
python preprocessing/subject_split.py

# 3. Train
python src/train.py --config config.yaml

# 4. Evaluate
python src/evaluate.py --config config.yaml --model results/checkpoints/best_model.pt

# 5. Real-time demo
python src/inference_realtime.py --model results/checkpoints/best_model.pt --camera 0
```
