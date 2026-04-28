"""
utils/helpers.py
----------------
Shared utility functions used across training, evaluation, and inference.
"""

import os
import random
import time
import json
import numpy as np
import torch


# ── Reproducibility ────────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ── Device ─────────────────────────────────────────────────────────────────────

def get_device() -> str:
    """Return 'cuda' if a GPU is available, else 'cpu'."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[Device] GPU detected: {gpu_name} ({vram_gb:.1f} GB VRAM)")
    else:
        print("[Device] No GPU found — using CPU (training will be slow).")
    return device


# ── Checkpointing ──────────────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, epoch: int, val_acc: float,
                    path: str, model_name: str = "resnet50",
                    class_map: dict = None):
    """
    Save model state + metadata to a .pth file.

    Saved keys:
        - model_state_dict
        - optimizer_state_dict
        - epoch
        - val_acc
        - model_name
        - class_map
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "epoch":               epoch,
        "val_acc":             val_acc,
        "model_name":          model_name,
        "class_map":           class_map or {},
        "model_state_dict":    model.state_dict(),
        "optimizer_state_dict":optimizer.state_dict(),
    }, path)


def load_checkpoint(model, path: str, device: str = "cpu"):
    """
    Load model weights from a checkpoint file.

    Returns:
        model, optimizer_state_dict, epoch, val_acc
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Checkpoint not found: {path}\n"
            "Run train.py first to generate a checkpoint."
        )

    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)

    return model, ckpt.get("optimizer_state_dict"), \
           ckpt.get("epoch", 0), ckpt.get("val_acc", 0.0)


# ── Running Average ─────────────────────────────────────────────────────────────

class AverageMeter:
    """Tracks running average of a metric (e.g. loss, accuracy)."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val   = 0.0
        self.avg   = 0.0
        self.sum   = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count


# ── Formatting ─────────────────────────────────────────────────────────────────

def format_time(seconds: float) -> str:
    """Convert elapsed seconds to a human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins}m {secs}s"


# ── Filesystem ─────────────────────────────────────────────────────────────────

def ensure_dirs(*paths: str):
    """Create directories if they don't exist."""
    for p in paths:
        os.makedirs(p, exist_ok=True)


def count_images(img_dir: str) -> int:
    """Count .jpg/.png images in a directory."""
    exts = {".jpg", ".jpeg", ".png"}
    return sum(
        1 for f in os.listdir(img_dir)
        if os.path.splitext(f)[1].lower() in exts
    ) if os.path.isdir(img_dir) else 0
