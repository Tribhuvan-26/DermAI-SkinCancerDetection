"""
evaluate.py
-----------
Evaluation script: generates confusion matrix, classification report,
and per-class metrics from a trained model checkpoint.

Usage:
    python evaluate.py --checkpoint models/best_model.pth \
                       --csv data/raw/HAM10000_metadata.csv \
                       --img_dir data/raw/images
"""

import os
import argparse
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_auc_score, roc_curve, auc
)
from sklearn.preprocessing import label_binarize

from dataset import load_dataset, CLASS_NAMES
from model   import build_model
from utils.helpers import load_checkpoint, get_device


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate Skin Cancer Model")
    p.add_argument("--checkpoint", default="models/best_model.pth")
    p.add_argument("--csv",        default="data/raw/HAM10000_metadata.csv")
    p.add_argument("--img_dir",    default="data/raw/images")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--output_dir", default="models")
    return p.parse_args()


@torch.no_grad()
def get_predictions(model, loader, device):
    """Run full validation set through model, return labels, preds, and probabilities."""
    model.eval()
    all_labels = []
    all_preds  = []
    all_probs  = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        outputs = model(images)
        probs   = torch.softmax(outputs, dim=1)
        preds   = outputs.argmax(dim=1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    return (np.array(all_labels),
            np.array(all_preds),
            np.array(all_probs))


def plot_confusion_matrix(cm, class_names, output_dir):
    """Save a heatmap of the confusion matrix."""
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        linewidths=0.5, linecolor="gray",
    )
    ax.set_title("Confusion Matrix", fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Confusion matrix saved → {path}")


def plot_roc_curves(y_true, y_probs, class_names, output_dir):
    """Save one-vs-rest ROC curves for all classes."""
    n_classes = len(class_names)
    y_bin     = label_binarize(y_true, classes=range(n_classes))

    fig, ax = plt.subplots(figsize=(10, 7))
    colors  = plt.cm.tab10(np.linspace(0, 1, n_classes))

    macro_auc = 0.0
    for i, (name, color) in enumerate(zip(class_names, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
        roc_auc_val = auc(fpr, tpr)
        macro_auc  += roc_auc_val
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{name} (AUC={roc_auc_val:.3f})")

    macro_auc /= n_classes
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random (AUC=0.500)")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_title(f"ROC Curves — Macro AUC: {macro_auc:.4f}", fontsize=15, fontweight="bold")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "roc_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] ROC curves saved → {path}")
    return macro_auc


def plot_per_class_accuracy(y_true, y_pred, class_names, output_dir):
    """Bar chart of per-class accuracy."""
    per_class_acc = []
    for i in range(len(class_names)):
        mask = y_true == i
        if mask.sum() > 0:
            per_class_acc.append(100.0 * (y_pred[mask] == i).mean())
        else:
            per_class_acc.append(0.0)

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(class_names, per_class_acc,
                  color=plt.cm.RdYlGn(np.array(per_class_acc) / 100),
                  edgecolor="black", linewidth=0.7)
    ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=10)
    ax.set_title("Per-Class Accuracy", fontsize=15, fontweight="bold")
    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_ylim(0, 115)
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    path = os.path.join(output_dir, "per_class_accuracy.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Per-class accuracy saved → {path}")


def main():
    args   = parse_args()
    device = get_device()
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load class map from checkpoint ──────────────────────────────────
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    class_map  = ckpt.get("class_map", {})
    model_name = ckpt.get("model_name", "resnet50")
    num_classes = len(class_map)

    class_names = [class_map[str(i)] for i in range(num_classes)]
    full_names  = [CLASS_NAMES.get(c, c) for c in class_names]

    print(f"[Eval] Model      : {model_name}")
    print(f"[Eval] Classes    : {class_names}")
    print(f"[Eval] Device     : {device}")

    # ── Build and load model ─────────────────────────────────────────────
    model = build_model(num_classes=num_classes, model_name=model_name, device=device)
    model, _, epoch, best_acc = load_checkpoint(model, args.checkpoint, device)
    print(f"[Eval] Loaded checkpoint from epoch {epoch} (Val Acc: {best_acc:.2f}%)")

    # ── Data ─────────────────────────────────────────────────────────────
    _, val_loader, le, _ = load_dataset(
        csv_path   = args.csv,
        img_dir    = args.img_dir,
        batch_size = args.batch_size,
        num_workers= 4,
    )

    # ── Predictions ───────────────────────────────────────────────────────
    print("[Eval] Running inference on validation set…")
    y_true, y_pred, y_probs = get_predictions(model, val_loader, device)

    overall_acc = 100.0 * (y_true == y_pred).mean()
    print(f"\n[Eval] Overall Accuracy: {overall_acc:.2f}%")

    # ── Classification Report ─────────────────────────────────────────────
    report = classification_report(y_true, y_pred, target_names=full_names, digits=4)
    print("\n" + "="*60)
    print("  CLASSIFICATION REPORT")
    print("="*60)
    print(report)

    # Save report to file
    report_path = os.path.join(args.output_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Overall Accuracy: {overall_acc:.4f}%\n\n")
        f.write(report)
    print(f"[Eval] Report saved → {report_path}")

    # ── Plots ─────────────────────────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, full_names, args.output_dir)
    macro_auc = plot_roc_curves(y_true, y_probs, full_names, args.output_dir)
    plot_per_class_accuracy(y_true, y_pred, full_names, args.output_dir)

    # ── Save metrics JSON ─────────────────────────────────────────────────
    metrics = {
        "overall_accuracy": round(float(overall_acc), 4),
        "macro_auc":        round(float(macro_auc),  4),
        "checkpoint_epoch": int(epoch),
        "checkpoint_val_acc": float(best_acc),
    }
    with open(os.path.join(args.output_dir, "eval_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n[Done] All evaluation artifacts saved to → {args.output_dir}/")


if __name__ == "__main__":
    main()
