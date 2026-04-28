"""
train.py
--------
Full training pipeline for the Skin Cancer Detection model.

Usage:
    python train.py --model resnet50 --epochs 30 --batch_size 32 --lr 1e-4
"""

import os
import argparse
import time
import json
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless backend
import matplotlib.pyplot as plt

from dataset import load_dataset
from model   import build_model
from utils.helpers import (
    set_seed, get_device, save_checkpoint, AverageMeter, format_time
)


# ── CLI Arguments ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Skin Cancer ResNet Training")
    p.add_argument("--csv",         default="data/raw/HAM10000_metadata.csv")
    p.add_argument("--img_dir",     default="data/raw/images")
    p.add_argument("--model",       default="resnet50", choices=["resnet18", "resnet50"])
    p.add_argument("--epochs",      type=int,   default=10)
    p.add_argument("--batch_size",  type=int,   default=16)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--val_size",    type=float, default=0.2)
    p.add_argument("--workers",     type=int,   default=4)
    p.add_argument("--patience",    type=int,   default=7,
                   help="Early stopping patience (epochs)")
    p.add_argument("--unfreeze_at", type=int,   default=10,
                   help="Epoch to unfreeze all layers for fine-tuning")
    p.add_argument("--output_dir",  default="models")
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--resume",      type=str,   default=None,
                   help="Path to checkpoint to resume from")
    return p.parse_args()


# ── One Epoch Training ─────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    loss_meter = AverageMeter()
    correct    = 0
    total      = 0
    t0         = time.time()

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()

        # Gradient clipping for stability
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        loss_meter.update(loss.item(), images.size(0))
        preds    = outputs.argmax(dim=1)
        correct += preds.eq(labels).sum().item()
        total   += labels.size(0)
        
        # Clear GPU cache to prevent memory buildup
        del images, labels, outputs, loss
        if device == "cuda":
            torch.cuda.empty_cache()

        if (batch_idx + 1) % 20 == 0:
            elapsed = format_time(time.time() - t0)
            print(f"  [Epoch {epoch}] Batch {batch_idx+1}/{len(loader)} | "
                  f"Loss: {loss_meter.avg:.4f} | "
                  f"Acc: {100.*correct/total:.2f}% | Time: {elapsed}")

    acc = 100.0 * correct / total
    return loss_meter.avg, acc


# ── Validation ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    loss_meter = AverageMeter()
    correct    = 0
    total      = 0

    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        outputs = model(images)
        loss    = criterion(outputs, labels)

        loss_meter.update(loss.item(), images.size(0))
        preds    = outputs.argmax(dim=1)
        correct += preds.eq(labels).sum().item()
        total   += labels.size(0)

    acc = 100.0 * correct / total
    return loss_meter.avg, acc


# ── Plot Training Curves ───────────────────────────────────────────────────────

def plot_training_curves(history: dict, output_dir: str):
    """Save loss and accuracy plots to output_dir."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training History", fontsize=16, fontweight="bold")

    # Loss
    axes[0].plot(epochs, history["train_loss"], "b-o", label="Train Loss",   markersize=4)
    axes[0].plot(epochs, history["val_loss"],   "r-o", label="Val Loss",     markersize=4)
    axes[0].set_title("Loss per Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, history["train_acc"], "b-o", label="Train Acc", markersize=4)
    axes[1].plot(epochs, history["val_acc"],   "r-o", label="Val Acc",   markersize=4)
    axes[1].set_title("Accuracy per Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Training curves saved → {path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = get_device()
    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)

    print("=" * 60)
    print("  SKIN CANCER DETECTION — TRAINING")
    print("=" * 60)
    print(f"  Model      : {args.model}")
    print(f"  Device     : {device}")
    print(f"  Epochs     : {args.epochs}")
    print(f"  Batch size : {args.batch_size}")
    print(f"  LR         : {args.lr}")
    print("=" * 60)

    # ── Data ────────────────────────────────────────────────────────────
    train_loader, val_loader, le, class_weights = load_dataset(
        csv_path   = args.csv,
        img_dir    = args.img_dir,
        val_size   = args.val_size,
        batch_size = args.batch_size,
        num_workers= args.workers,
        seed       = args.seed,
    )

    num_classes = len(le.classes_)

    # Save class mapping for inference
    class_map = {int(i): cls for i, cls in enumerate(le.classes_)}
    with open(os.path.join(args.output_dir, "class_map.json"), "w") as f:
        json.dump(class_map, f, indent=2)

    # ── Model ────────────────────────────────────────────────────────────
    model = build_model(
        num_classes = num_classes,
        model_name  = args.model,
        pretrained  = True,
        device      = device,
    )

    # Weighted loss to handle class imbalance
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()),
                     lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # ── Resume from checkpoint ───────────────────────────────────────────
    start_epoch = 1
    history      = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    patience_ctr = 0
    
    if args.resume:
        print(f"[Resume] Loading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_val_acc = checkpoint.get('val_acc', 0.0)
        print(f"[Resume] Resuming from epoch {start_epoch}, best val acc: {best_val_acc:.2f}%")

    # ── Training Loop ────────────────────────────────────────────────────

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()

        # Unfreeze all layers mid-training for fine-tuning
        if epoch == args.unfreeze_at:
            print(f"\n[Epoch {epoch}] Unfreezing all layers for full fine-tuning…")
            model.unfreeze_all()
            optimizer = Adam(model.parameters(), lr=args.lr / 10, weight_decay=1e-4)
            scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - epoch, eta_min=1e-7)

        print(f"\n{'-'*60}")
        print(f"  EPOCH {epoch}/{args.epochs}  |  LR: {scheduler.get_last_lr()[0]:.2e}")
        print(f"{'-'*60}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_acc     = validate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        epoch_time = format_time(time.time() - epoch_start)
        print(f"\n  ▶ Train  — Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
        print(f"  ▶ Val    — Loss: {val_loss:.4f}  | Acc: {val_acc:.2f}%")
        print(f"  ▶ Time   — {epoch_time}")

        # ── Checkpoint ──────────────────────────────────────────────────
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            patience_ctr = 0
            save_checkpoint(
                model, optimizer, epoch, val_acc,
                path=os.path.join(args.output_dir, "best_model.pth"),
                model_name=args.model, class_map=class_map,
            )
            print(f"  ✅ New best model saved (Val Acc: {best_val_acc:.2f}%)")
        else:
            patience_ctr += 1
            # Save periodic checkpoint every 5 epochs
            if epoch % 5 == 0:
                save_checkpoint(
                    model, optimizer, epoch, val_acc,
                    path=os.path.join(args.output_dir, "checkpoints", f"ckpt_epoch{epoch}.pth"),
                    model_name=args.model, class_map=class_map,
                )

        # ── Early Stopping ───────────────────────────────────────────────
        if patience_ctr >= args.patience:
            print(f"\n[EarlyStopping] No improvement for {args.patience} epochs. Stopping.")
            break

    # ── Post-training ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE")
    print(f"  Best Val Accuracy: {best_val_acc:.2f}%")
    print(f"{'='*60}")

    plot_training_curves(history, args.output_dir)

    # Save history for later analysis
    with open(os.path.join(args.output_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"[Done] History saved → {args.output_dir}/history.json")


if __name__ == "__main__":
    main()
