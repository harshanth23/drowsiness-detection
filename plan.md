# 3D CNN+LSTM Drowsiness Detection System — Implementation Plan

## Overview

A hybrid spatiotemporal deep learning system for real-time driver drowsiness detection. The architecture combines a 3D-CNN backbone for per-clip feature extraction with an LSTM for long-range temporal modelling. Optional fusion of facial landmark features (EAR/MOR) is supported. The system is trained on four public benchmarks (NTHU-DDD, YawDD, UTA-RLDD, NITYMED) using subject-wise splits and deployed as a live webcam demo via InsightFace.

---

## Directory Structure

```
Drowsiness Detection/
├── dataset/
│   ├── processed/                     # Raw extracted frames, per dataset
│   │   ├── NTHU-DDD/
│   │   ├── YawDD/
│   │   ├── UTA-RLDD/
│   │   └── NITYMED/
│   ├── processed_cropped/             # Cropped face clips, per dataset
│   │   ├── NTHU-DDD/
│   │   ├── YawDD/
│   │   ├── UTA-RLDD/
│   │   └── NITYMED/
│   └── landmarks_cropped/             # Per-frame landmark .npy files
├── src/
│   ├── dataset.py                     # PyTorch Dataset classes
│   ├── model.py                       # 3D-CNN + LSTM architecture
│   ├── train.py                       # Training loop
│   ├── evaluate.py                    # Metrics, confusion matrix, ROC
│   ├── inference_realtime.py          # Live webcam demo
│   ├── gradcam.py                     # Grad-CAM visualisation utilities
│   └── utils.py                       # Logging, checkpointing, plotting
├── results/
│   ├── checkpoints/                   # Best model weights (.pt)
│   ├── logs/                          # TensorBoard / CSV logs
│   └── plots/                         # Loss curves, confusion matrices, Grad-CAM
├── config.yaml                        # Centralised hyperparameter config
├── requirements.txt                   # Python dependencies
└── README.md                          # Setup and usage instructions
```

---

## Model Architecture

### 1. Input
- Clip tensor: `(B, C, T, H, W)` — e.g. `(B, 3, 16, 112, 112)`
- Optional landmark tensor: `(B, T, 2)` — EAR and MOR per frame

### 2. 3D-CNN Backbone
Four convolutional blocks, each consisting of:

```
Conv3D(kernel=3×3×3, padding=same) → BatchNorm3D → ReLU → MaxPool3D → Dropout(p=0.2)
```

| Block | Filters | Output Shape (T×H×W) |
|-------|---------|----------------------|
| 1     | 16      | 16×56×56             |
| 2     | 32      | 8×28×28              |
| 3     | 64      | 4×14×14              |
| 4     | 128     | 2×7×7                |

After block 4, the spatial dimensions are flattened → temporal sequence fed to LSTM.

### 3. LSTM Temporal Modelling
- 1–2 stacked LSTM layers (hidden size 256)
- Input: sequence of per-timestep CNN feature vectors
- Output: final hidden state `h_T`

### 4. Landmark Fusion (optional)
Two strategies (ablation study):
- **Feature-level**: concatenate time-averaged EAR/MOR to LSTM output.
- **Parallel branch**: separate small LSTM on EAR/MOR sequence; merge outputs before classifier.

### 5. Classifier Head
```
Linear(256 [+2]) → ReLU → Dropout(0.3) → Linear(2) → Softmax
```
Binary output: **Alert** / **Drowsy** (or multi-class if UTA-RLDD 3-class mode is enabled).

---

## Datasets

| Dataset   | Subjects | Hours | Labels           | Split Strategy          |
|-----------|----------|-------|------------------|-------------------------|
| NTHU-DDD  | 18       | 9.5   | Binary           | 80/20 subject-wise      |
| YawDD     | 119      | 2.6   | Binary           | 80/20 subject-wise      |
| UTA-RLDD  | 60       | 30    | 3-class → binary | Official 5-fold CV      |
| NITYMED   | 21       | 4     | Binary           | 80/20 subject-wise      |

- Splits use `sklearn.model_selection.GroupShuffleSplit(groups=subject_id)` to prevent identity leakage.
- All four datasets are combined for training; each dataset's held-out subjects are evaluated independently.
- Class imbalance handled via weighted cross-entropy or oversampling.

---

## Data Loading (`src/dataset.py`)

- **`DrowsinessClipDataset`**: loads a list of clip paths + labels; returns `(clip_tensor, landmark_tensor, label)`.
- Clip sampling: stride-based sliding window (stride = T//2) over each video.
- On-the-fly augmentations: random horizontal flip, colour jitter (brightness/contrast), random crop.
- Normalisation: ImageNet mean/std per frame.
- `DataLoader` with `num_workers=4`, `pin_memory=True`.

---

## Training (`src/train.py`)

| Hyperparameter    | Value / Range        |
|-------------------|----------------------|
| Optimiser         | Adam (β₁=0.9, β₂=0.999) |
| Learning rate     | 1e-4 (cosine decay)  |
| Batch size        | 8–16 (GPU-tuned)     |
| Epochs            | 30–50                |
| Loss              | Binary cross-entropy / Focal loss |
| Mixed precision   | `torch.cuda.amp` (AMP) |
| Random seed       | Fixed (42)           |

- Validation after every epoch; save checkpoint when validation F1 improves.
- Early stopping with patience = 10 epochs.
- All hyperparameters configurable via `config.yaml` or CLI (`argparse`).

---

## Evaluation (`src/evaluate.py`)

Metrics computed per dataset on held-out test subjects:

- Accuracy, Precision, Recall, F1-score
- Confusion matrix (saved as PNG)
- ROC curve + AUC (saved as PNG)
- Inference speed benchmark (FPS on RTX 3050)

All outputs saved to `results/plots/`.

---

## Real-Time Webcam Demo (`src/inference_realtime.py`)

**Pipeline (per frame):**

1. Capture frame from webcam (OpenCV).
2. **Face detection** — InsightFace RetinaFace (PyTorch).
3. Crop & resize face ROI to model input size.
4. **Landmark extraction** — InsightFace / Dlib → compute EAR and MOR.
5. Push frame + EAR/MOR into a circular buffer (length T=16).
6. When buffer is full, stack into clip tensor → run 3D-CNN+LSTM → get drowsiness score.
7. Overlay prediction label ("Alert 🟢" / "Drowsy 🔴") and score on frame.
8. Display via `cv2.imshow`; inference every 2–3 frames to maintain ≥10 FPS.

---

## Grad-CAM Visualisation (`src/gradcam.py`)

- Backpropagate class score to the last Conv3D layer using **TorchCAM** or **Captum**.
- Produce per-frame saliency maps (averaged over time dimension).
- Overlay heatmaps on face frames using `cv2.applyColorMap`.
- Expected behaviour: eyes highlighted during blink sequences, mouth during yawns.
- Sample heatmaps saved to `results/plots/gradcam/`.

---

## Utilities (`src/utils.py`)

- `save_checkpoint(model, optimizer, epoch, path)`
- `load_checkpoint(path) → model, optimizer, epoch`
- `plot_loss_curves(train_losses, val_losses, save_path)`
- `compute_ear(landmarks) → float`
- `compute_mor(landmarks) → float`
- `set_seed(seed)` — fixes `torch`, `numpy`, `random` seeds
- TensorBoard `SummaryWriter` wrapper

---
Use the Conda webenv 
use the command to activate the conda environment: `conda activate webenv`

## Dependencies (`requirements.txt`)

```
torch==2.3.1+cu121
torchvision==0.18.1+cu121
torchaudio==2.3.1+cu121
insightface>=0.7.3
onnxruntime-gpu
opencv-python>=4.9
numpy
scikit-learn
matplotlib
torchcam          # Grad-CAM (or captum)
tensorboard
tqdm
pyyaml
```

Install: `pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121`

---

## Verification & Testing Plan

| Step | Method |
|------|--------|
| Unit tests | Test `DrowsinessClipDataset` shape/dtype, EAR/MOR computation |
| Training smoke test | 2-epoch run on small subset; check loss decreases |
| Full training | 30–50 epochs on combined datasets; log to TensorBoard |
| Evaluation | Per-dataset metrics; compare with baselines from literature |
| Inference speed | Benchmark FPS with `time.perf_counter`; target ≥10 FPS |
| Grad-CAM | Visual inspection: eyes/mouth regions highlighted |
| Webcam demo | End-to-end live demo on RTX 3050 |

---

## References

1. 3D-CNN + LSTM for spatiotemporal video understanding
2. Lightweight 3D-CNN (4 conv layers, 4–8 filters) for drowsiness detection
3. EAR (Eye Aspect Ratio) for blink/drowsiness detection
4. MOR (Mouth Opening Ratio) for yawn detection
5. High accuracy with 60×60 face crops
6. NTHU-DDD dataset (18 subjects, 9.5 hours)
7. UTA-RLDD dataset (60 subjects, 3 states, 5-fold split)
8. YawDD dataset (119 subjects, 2.6 hours)
9. NITYMED dataset (21 subjects, 4 hours)
10. InsightFace RetinaFace for real-time face detection
11. Grad-CAM for CNN interpretability
12. Lightweight 3D-CNN baseline architecture
