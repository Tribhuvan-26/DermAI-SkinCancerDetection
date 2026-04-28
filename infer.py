"""
infer.py
--------
Single-image inference with Grad-CAM visualization.

Usage:
    python infer.py --image path/to/image.jpg --checkpoint models/best_model.pth
"""

import os
import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from model  import build_model
from utils.preprocess  import get_inference_transform
from utils.helpers     import load_checkpoint, get_device
from dataset import CLASS_NAMES, CLASS_DESCRIPTIONS, RISK_LEVEL


def parse_args():
    p = argparse.ArgumentParser(description="Skin Cancer Single-Image Inference")
    p.add_argument("--image",      required=True, help="Path to input image (.jpg/.png)")
    p.add_argument("--checkpoint", default="models/best_model.pth")
    p.add_argument("--output",     default="inference_result.png",
                   help="Path to save Grad-CAM visualization")
    p.add_argument("--topk",       type=int, default=3, help="Show top-k predictions")
    return p.parse_args()


def load_model_from_checkpoint(checkpoint_path: str, device: str):
    """Load model and metadata from a saved checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    class_map  = ckpt.get("class_map", {})
    model_name = ckpt.get("model_name", "resnet50")
    num_classes = len(class_map)

    model = build_model(num_classes=num_classes, model_name=model_name, device=device)
    model, _, epoch, best_acc = load_checkpoint(model, checkpoint_path, device)
    model.eval()

    return model, class_map, model_name


def preprocess_image(image_path: str, device: str):
    """Load and preprocess a single image for inference."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    transform = get_inference_transform()
    image_pil = Image.open(image_path).convert("RGB")
    tensor    = transform(image_pil).unsqueeze(0).to(device)
    return tensor, image_pil


def predict(model, tensor: torch.Tensor, class_map: dict, topk: int = 3):
    """
    Run forward pass and return top-k predictions.

    Returns:
        List of dicts: [{"class_code", "class_name", "confidence", "risk", "description"}]
    """
    with torch.no_grad():
        logits = model(tensor)
        probs  = F.softmax(logits, dim=1).squeeze(0)

    topk_probs, topk_idx = probs.topk(topk)

    results = []
    for prob, idx in zip(topk_probs.cpu().numpy(), topk_idx.cpu().numpy()):
        code = class_map.get(str(idx), str(idx))
        results.append({
            "class_code":   code,
            "class_name":   CLASS_NAMES.get(code, code),
            "confidence":   float(prob),
            "risk":         RISK_LEVEL.get(code, "unknown"),
            "description":  CLASS_DESCRIPTIONS.get(code, ""),
        })

    return results


def generate_gradcam_overlay(model, tensor: torch.Tensor,
                              image_pil: Image.Image,
                              class_idx: int) -> np.ndarray:
    """
    Generate Grad-CAM heatmap and blend with original image.

    Returns:
        Overlaid RGB image as np.ndarray (H×W×3 uint8)
    """
    # Need grad tracking
    model.eval()
    t = tensor.clone().requires_grad_(True)

    cam_np, _ = model.get_gradcam(t, class_idx=class_idx)

    # Resize CAM to original image size
    orig_w, orig_h = image_pil.size
    cam_resized = cv2.resize(cam_np, (orig_w, orig_h))

    # Apply colormap
    heatmap     = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    orig_np     = np.array(image_pil.resize((orig_w, orig_h)))
    overlay     = cv2.addWeighted(orig_np, 0.55, heatmap_rgb, 0.45, 0)

    return overlay, cam_resized


def save_visualization(image_pil: Image.Image,
                        overlay: np.ndarray,
                        cam: np.ndarray,
                        results: list,
                        output_path: str):
    """Create and save a 4-panel visualization."""
    fig = plt.figure(figsize=(16, 8))
    fig.patch.set_facecolor("#0f1117")

    # Panel 1 — Original image
    ax1 = fig.add_subplot(1, 4, 1)
    ax1.imshow(image_pil)
    ax1.set_title("Input Image", color="white", fontsize=12, fontweight="bold")
    ax1.axis("off")

    # Panel 2 — Grad-CAM heatmap
    ax2 = fig.add_subplot(1, 4, 2)
    ax2.imshow(cam, cmap="jet")
    ax2.set_title("Grad-CAM Heatmap", color="white", fontsize=12, fontweight="bold")
    ax2.axis("off")

    # Panel 3 — Overlay
    ax3 = fig.add_subplot(1, 4, 3)
    ax3.imshow(overlay)
    ax3.set_title("CAM Overlay", color="white", fontsize=12, fontweight="bold")
    ax3.axis("off")

    # Panel 4 — Prediction bar chart
    ax4 = fig.add_subplot(1, 4, 4)
    ax4.set_facecolor("#1a1d27")
    names  = [r["class_code"] for r in results]
    confs  = [r["confidence"] * 100 for r in results]
    colors = ["#ef4444" if r["risk"] == "high" else
              "#f59e0b" if r["risk"] == "medium" else "#22c55e"
              for r in results]
    bars = ax4.barh(names, confs, color=colors, edgecolor="none", height=0.55)
    for bar, c in zip(bars, confs):
        ax4.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                 f"{c:.1f}%", va="center", color="white", fontsize=11, fontweight="bold")
    ax4.set_xlim(0, 115)
    ax4.set_title("Top Predictions", color="white", fontsize=12, fontweight="bold")
    ax4.tick_params(colors="white")
    ax4.spines[:].set_color("#333")
    ax4.set_xlabel("Confidence (%)", color="white", fontsize=10)
    for label in ax4.get_yticklabels():
        label.set_color("white")

    plt.suptitle(
        f"Prediction: {results[0]['class_name']} ({results[0]['confidence']*100:.1f}%)",
        color="white", fontsize=14, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Output] Visualization saved → {output_path}")


def run_inference(image_path: str, checkpoint_path: str,
                  output_path: str = "inference_result.png",
                  topk: int = 3) -> dict:
    """
    Full inference pipeline — callable from other modules (e.g. API).

    Returns:
        dict with 'predictions' list and 'output_path'
    """
    device = get_device()

    model, class_map, model_name = load_model_from_checkpoint(checkpoint_path, device)
    tensor, image_pil            = preprocess_image(image_path, device)
    results                      = predict(model, tensor, class_map, topk=topk)

    top_idx = list(class_map.values()).index(results[0]["class_code"])
    overlay, cam = generate_gradcam_overlay(model, tensor, image_pil, class_idx=top_idx)
    save_visualization(image_pil, overlay, cam, results, output_path)

    return {"predictions": results, "output_path": output_path}


def main():
    args = parse_args()

    print("=" * 60)
    print("  SKIN CANCER DETECTION — INFERENCE")
    print("=" * 60)
    print(f"  Image      : {args.image}")
    print(f"  Checkpoint : {args.checkpoint}")
    print("=" * 60)

    result = run_inference(
        image_path      = args.image,
        checkpoint_path = args.checkpoint,
        output_path     = args.output,
        topk            = args.topk,
    )

    print("\n  TOP PREDICTIONS:")
    print("  " + "-"*50)
    for i, r in enumerate(result["predictions"]):
        risk_icon = "🔴" if r["risk"] == "high" else "🟡" if r["risk"] == "medium" else "🟢"
        print(f"  #{i+1}  {risk_icon}  {r['class_name']} ({r['class_code']})")
        print(f"       Confidence : {r['confidence']*100:.2f}%")
        print(f"       Risk Level : {r['risk'].upper()}")
        print(f"       Info       : {r['description']}")
        print()


if __name__ == "__main__":
    main()
