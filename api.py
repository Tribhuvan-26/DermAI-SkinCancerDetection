"""
api.py
------
FastAPI backend for the Skin Cancer Detection React UI.
Exposes REST endpoints for image upload, prediction, and health checks.

Usage:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import io
import json
import time
import base64
import tempfile
import traceback
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import cv2
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as colormap

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from model  import build_model
from utils.preprocess  import get_inference_transform
from utils.helpers     import load_checkpoint, get_device
from dataset import CLASS_NAMES, CLASS_DESCRIPTIONS, RISK_LEVEL


# ── App Setup ──────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "Skin Cancer Detection API",
    description = "ResNet-based skin lesion classification with Grad-CAM explainability",
    version     = "1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],   # In production, restrict to your frontend origin
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ── Model State (loaded once at startup) ──────────────────────────────────────

MODEL_STATE = {
    "model":      None,
    "class_map":  {},
    "device":     "cpu",
    "loaded":     False,
    "model_name": "resnet50",
}

CHECKPOINT_PATH = os.environ.get("MODEL_CHECKPOINT", "models/best_model.pth")


@app.on_event("startup")
async def load_model():
    """Load model checkpoint when the API starts."""
    device = get_device()
    MODEL_STATE["device"] = device

    if not os.path.exists(CHECKPOINT_PATH):
        print(f"[API] ⚠️  Checkpoint not found at {CHECKPOINT_PATH}. "
              "Predictions will fail until a trained model is provided.")
        return

    try:
        ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
        class_map  = ckpt.get("class_map", {})
        model_name = ckpt.get("model_name", "resnet50")
        num_classes = len(class_map)

        model = build_model(num_classes=num_classes, model_name=model_name, device=device)
        model, _, epoch, best_acc = load_checkpoint(model, CHECKPOINT_PATH, device)
        model.eval()

        MODEL_STATE["model"]      = model
        MODEL_STATE["class_map"]  = class_map
        MODEL_STATE["loaded"]     = True
        MODEL_STATE["model_name"] = model_name

        print(f"[API] ✅ Model loaded: {model_name} | Epoch {epoch} | Val Acc: {best_acc:.2f}%")
    except Exception as e:
        print(f"[API] ❌ Failed to load model: {e}")


# ── Response Models ───────────────────────────────────────────────────────────

class PredictionItem(BaseModel):
    class_code:  str
    class_name:  str
    confidence:  float
    risk:        str
    description: str

class PredictionResponse(BaseModel):
    predictions:    List[PredictionItem]
    top_prediction: PredictionItem
    gradcam_base64: Optional[str] = None
    inference_ms:   float
    model_name:     str


# ── Helper Functions ──────────────────────────────────────────────────────────

def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert [1, C, H, W] tensor back to PIL Image."""
    MEAN = [0.485, 0.456, 0.406]
    STD  = [0.229, 0.224, 0.225]
    t = tensor.squeeze(0).cpu().clone()
    for c, (m, s) in enumerate(zip(MEAN, STD)):
        t[c] = t[c] * s + m
    t = t.clamp(0, 1).permute(1, 2, 0).numpy()
    return Image.fromarray((t * 255).astype(np.uint8))


def generate_gradcam_b64(model, tensor: torch.Tensor,
                          image_pil: Image.Image, class_idx: int) -> str:
    """
    Generate a Grad-CAM overlay and return it as a base64-encoded PNG string.
    """
    try:
        t = tensor.clone().requires_grad_(True)
        cam_np, _ = model.get_gradcam(t, class_idx=class_idx)

        # Resize to original image dimensions
        w, h = image_pil.size
        cam_resized = cv2.resize(cam_np, (w, h))

        # Apply jet colormap
        heatmap     = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        orig_np     = np.array(image_pil.convert("RGB"))
        overlay     = cv2.addWeighted(orig_np, 0.55, heatmap_rgb, 0.45, 0)

        # Encode to PNG → base64
        overlay_pil = Image.fromarray(overlay)
        buf = io.BytesIO()
        overlay_pil.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"[API] Grad-CAM failed: {e}")
        return ""


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status":       "ok",
        "model_loaded": MODEL_STATE["loaded"],
        "model_name":   MODEL_STATE["model_name"],
        "device":       MODEL_STATE["device"],
    }


@app.get("/classes")
async def get_classes():
    """Return all supported class codes and their full names."""
    class_map = MODEL_STATE["class_map"]
    return {
        "classes": [
            {
                "index":       int(k),
                "code":        v,
                "name":        CLASS_NAMES.get(v, v),
                "risk":        RISK_LEVEL.get(v, "unknown"),
                "description": CLASS_DESCRIPTIONS.get(v, ""),
            }
            for k, v in class_map.items()
        ]
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file:     UploadFile = File(...),
    topk:     int        = Query(default=3, ge=1, le=7),
    gradcam:  bool       = Query(default=True),
):
    """
    Accept an uploaded image and return top-k skin lesion predictions.

    Args:
        file    : JPEG/PNG image file
        topk    : Number of top predictions to return (1–7)
        gradcam : Whether to include Grad-CAM overlay (base64)
    """
    if not MODEL_STATE["loaded"]:
        raise HTTPException(
            status_code = 503,
            detail      = "Model not loaded. Train the model first and restart the API."
        )

    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image (JPEG/PNG).")

    t0 = time.time()

    # Read and process image
    contents  = await file.read()
    image_pil = Image.open(io.BytesIO(contents)).convert("RGB")

    transform = get_inference_transform()
    tensor    = transform(image_pil).unsqueeze(0).to(MODEL_STATE["device"])

    model     = MODEL_STATE["model"]
    class_map = MODEL_STATE["class_map"]

    # Forward pass
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1).squeeze(0)

    topk_probs, topk_indices = probs.topk(topk)

    predictions = []
    for prob, idx in zip(topk_probs.cpu().numpy(), topk_indices.cpu().numpy()):
        code = class_map.get(str(idx), str(idx))
        predictions.append(PredictionItem(
            class_code  = code,
            class_name  = CLASS_NAMES.get(code, code),
            confidence  = float(prob),
            risk        = RISK_LEVEL.get(code, "unknown"),
            description = CLASS_DESCRIPTIONS.get(code, ""),
        ))

    # Grad-CAM (optional)
    gradcam_b64 = ""
    if gradcam and predictions:
        top_code    = predictions[0].class_code
        top_idx     = next((int(k) for k, v in class_map.items() if v == top_code), 0)
        gradcam_b64 = generate_gradcam_b64(model, tensor, image_pil, top_idx)

    inference_ms = (time.time() - t0) * 1000

    return PredictionResponse(
        predictions    = predictions,
        top_prediction = predictions[0],
        gradcam_base64 = gradcam_b64,
        inference_ms   = round(inference_ms, 1),
        model_name     = MODEL_STATE["model_name"],
    )


@app.get("/model-info")
async def model_info():
    """Return basic model metadata."""
    model = MODEL_STATE.get("model")
    if not model:
        return {"loaded": False}

    return {
        "loaded":         True,
        "model_name":     MODEL_STATE["model_name"],
        "num_classes":    len(MODEL_STATE["class_map"]),
        "total_params":   model.total_params(),
        "trainable_params": model.trainable_params(),
        "device":         MODEL_STATE["device"],
    }


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
