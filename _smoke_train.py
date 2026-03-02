"""
_smoke_train.py — 2-epoch GPU smoke test (run from project root)
    conda activate webenv
    python _smoke_train.py
"""
import sys, os
sys.path.insert(0, os.path.abspath('.'))

def main():
    import torch
    import yaml

    # ── Load config ───────────────────────────────────────────────────────────
    with open('config.yaml') as f:
        cfg = yaml.safe_load(f)

    # ── Overrides for smoke test ──────────────────────────────────────────────
    cfg['training']['epochs']      = 2
    cfg['training']['batch_size']  = 32
    cfg['training']['num_workers'] = 4
    cfg['training']['use_amp']     = True

    # Verify GPU
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n[smoke] Device : {device_str}")
    if device_str == 'cuda':
        print(f"[smoke] GPU    : {torch.cuda.get_device_name(0)}")
        print(f"[smoke] VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("[smoke] WARNING: CUDA not available, running on CPU (will be slow)")

    from src.train import train
    train(cfg)

# Windows multiprocessing requires the guard
if __name__ == '__main__':
    main()
