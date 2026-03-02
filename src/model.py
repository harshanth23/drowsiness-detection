"""
src/model.py — 3D-CNN + LSTM Drowsiness Detection Model
---------------------------------------------------------
Architecture (from plan.md):
  Input:  (B, 3, T=16, H=112, W=112)   clip tensor
          (B, T, 2)                      landmark tensor [EAR, MOR] per frame

  Backbone (4 Conv3D blocks):
    Block 1: Conv3D → BN → ReLU → MaxPool3D → Dropout  →  (B, 16,  16, 56,  56)
    Block 2: Conv3D → BN → ReLU → MaxPool3D → Dropout  →  (B, 32,   8, 28,  28)
    Block 3: Conv3D → BN → ReLU → MaxPool3D → Dropout  →  (B, 64,   4, 14,  14)
    Block 4: Conv3D → BN → ReLU → MaxPool3D → Dropout  →  (B, 128,  2,  7,   7)

  Temporal sequence extraction:
    Flatten spatial dims per timestep → (B, T', 128*7*7)
    Project to lstm_input_size        → (B, T', 512)

  LSTM: 1–2 stacked layers, hidden=256 → h_T  →  (B, 256)

  Landmark branch (optional, parallel):
    Small LSTM on (B, T, 2) → (B, 64)

  Classifier head:
    Linear(256 [+64]) → ReLU → Dropout(0.3) → Linear(num_classes)
"""

import torch
import torch.nn as nn
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Conv3D Block
# ─────────────────────────────────────────────────────────────────────────────

class ConvBlock3D(nn.Module):
    """Conv3D → BatchNorm3D → ReLU → MaxPool3D → Dropout"""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int = 3,
        pool_kernel=(1, 2, 2),
        pool_stride=(1, 2, 2),
        dropout: float = 0.2,
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=kernel, padding=kernel // 2, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=pool_kernel, stride=pool_stride),
            nn.Dropout3d(p=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ─────────────────────────────────────────────────────────────────────────────
# 3D-CNN Backbone
# ─────────────────────────────────────────────────────────────────────────────

class Backbone3DCNN(nn.Module):
    """
    Four Conv3D blocks that convert (B, 3, T, 112, 112) →
    a sequence of per-timestep feature vectors (B, T', D).

    Pool strategy keeps the temporal dimension alive through blocks 1–3,
    halving it only in block 4, giving T' = 2 for T=16.
    """

    def __init__(self, lstm_input_size: int = 512, dropout: float = 0.2):
        super().__init__()
        # B, 3,  16, 112, 112
        self.block1 = ConvBlock3D(3,   16, pool_kernel=(1,2,2), pool_stride=(1,2,2), dropout=dropout)
        # B, 16, 16, 56,  56
        self.block2 = ConvBlock3D(16,  32, pool_kernel=(2,2,2), pool_stride=(2,2,2), dropout=dropout)
        # B, 32, 8,  28,  28
        self.block3 = ConvBlock3D(32,  64, pool_kernel=(2,2,2), pool_stride=(2,2,2), dropout=dropout)
        # B, 64, 4,  14,  14
        self.block4 = ConvBlock3D(64, 128, pool_kernel=(2,2,2), pool_stride=(2,2,2), dropout=dropout)
        # B, 128, 2, 7,  7   → spatial flat = 128*7*7 = 6272

        spatial_flat = 128 * 7 * 7
        self.proj = nn.Linear(spatial_flat, lstm_input_size)
        self.proj_bn = nn.LayerNorm(lstm_input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T, H, W)
        Returns:
            seq: (B, T', lstm_input_size)
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        # x: (B, 128, T', 7, 7)
        B, C, T, H, W = x.shape
        # Reshape to (B, T', C*H*W)
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(B, T, C * H * W)
        x = self.proj(x)
        x = self.proj_bn(x)
        return x   # (B, T', lstm_input_size)


# ─────────────────────────────────────────────────────────────────────────────
# Full Model
# ─────────────────────────────────────────────────────────────────────────────

class DrowsinessModel(nn.Module):
    """
    3D-CNN + LSTM with optional parallel landmark LSTM branch.

    Args:
        num_classes:        2 (alert/drowsy) or 3 (uta 3-class mode)
        lstm_layers:        number of stacked LSTM layers (1 or 2)
        lstm_hidden:        LSTM hidden size (256)
        lstm_input_size:    projection size from CNN spatial features (512)
        use_landmarks:      enable parallel landmark LSTM branch
        lm_hidden:          hidden size for landmark LSTM (64)
        dropout:            shared dropout rate
    """

    def __init__(
        self,
        num_classes: int = 2,
        lstm_layers: int = 1,
        lstm_hidden: int = 256,
        lstm_input_size: int = 512,
        use_landmarks: bool = True,
        lm_hidden: int = 64,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.use_landmarks = use_landmarks

        # 3D-CNN backbone
        self.backbone = Backbone3DCNN(lstm_input_size=lstm_input_size, dropout=0.2)

        # Main LSTM
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        # Landmark branch (parallel small LSTM)
        self.lm_hidden = lm_hidden
        if use_landmarks:
            self.lm_lstm = nn.LSTM(
                input_size=2,
                hidden_size=lm_hidden,
                num_layers=1,
                batch_first=True,
            )
            classifier_in = lstm_hidden + lm_hidden
        else:
            classifier_in = lstm_hidden

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(classifier_in, classifier_in // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(classifier_in // 2, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        clip: torch.Tensor,            # (B, 3, T, H, W)
        landmarks: Optional[torch.Tensor] = None,   # (B, T, 2)
    ) -> torch.Tensor:
        """
        Returns:
            logits: (B, num_classes)
        """
        # CNN backbone → sequence
        feat_seq = self.backbone(clip)   # (B, T', lstm_input_size)

        # Main LSTM — take last hidden state
        _, (h_n, _) = self.lstm(feat_seq)
        h_cnn = h_n[-1]                 # (B, lstm_hidden)

        if self.use_landmarks and landmarks is not None:
            _, (h_lm, _) = self.lm_lstm(landmarks)
            h_lm = h_lm[-1]             # (B, lm_hidden)
            h = torch.cat([h_cnn, h_lm], dim=1)   # (B, lstm_hidden + lm_hidden)
        else:
            h = h_cnn

        return self.classifier(h)       # (B, num_classes)

    @staticmethod
    def from_config(config: dict) -> "DrowsinessModel":
        """Instantiate from a dict (e.g. parsed from config.yaml)."""
        m = config.get("model", config)
        return DrowsinessModel(
            num_classes     = m.get("num_classes",     2),
            lstm_layers     = m.get("lstm_layers",     1),
            lstm_hidden     = m.get("lstm_hidden",     256),
            lstm_input_size = m.get("lstm_input_size", 512),
            use_landmarks   = m.get("use_landmarks",   True),
            lm_hidden       = m.get("lm_hidden",       64),
            dropout         = m.get("dropout",         0.3),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Quick architecture sanity-check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    B, T, H, W = 2, 16, 112, 112

    model = DrowsinessModel(use_landmarks=True)
    model.eval()

    clip = torch.randn(B, 3, T, H, W)
    lms  = torch.randn(B, T, 2)

    with torch.no_grad():
        out = model(clip, lms)

    print(f"Input  clip : {clip.shape}")
    print(f"Input  lms  : {lms.shape}")
    print(f"Output logits: {out.shape}")
    assert out.shape == (B, 2), f"Unexpected output shape: {out.shape}"

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")
    print("✅ Model sanity check passed.")
