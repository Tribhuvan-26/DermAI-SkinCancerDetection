# 🔬 DermAI — Skin Cancer Detection with ResNet CNN

> AI-powered multi-class skin lesion classification using PyTorch transfer learning,
> with a production-grade React UI and FastAPI backend.

---

## 📋 Project Overview

DermAI classifies dermoscopic skin lesion images into **7 HAM10000 categories** using a
pretrained ResNet50 (or ResNet18) backbone fine-tuned with transfer learning. It includes:

- **Full training pipeline** with class-imbalance handling and early stopping
- **Grad-CAM explainability** to highlight diagnostically relevant skin regions
- **FastAPI REST backend** serving model predictions
- **Professional React UI** with drag-and-drop upload, confidence charts, and Grad-CAM overlay

### Supported Classes

| Code    | Name                   | Risk     |
|---------|------------------------|----------|
| `nv`    | Melanocytic Nevi       | 🟢 Low    |
| `mel`   | Melanoma               | 🔴 High   |
| `bkl`   | Benign Keratosis       | 🟢 Low    |
| `bcc`   | Basal Cell Carcinoma   | 🟡 Medium |
| `akiec` | Actinic Keratoses      | 🟡 Medium |
| `vasc`  | Vascular Lesions       | 🟢 Low    |
| `df`    | Dermatofibroma         | 🟢 Low    |

---

## 📁 Project Structure

```
skin_cancer_project/
├── data/
│   ├── raw/
│   │   ├── images/              ← HAM10000 .jpg images go here
│   │   └── HAM10000_metadata.csv
│   └── processed/               ← (auto-generated splits, if needed)
├── models/
│   ├── best_model.pth           ← Saved after training
│   ├── class_map.json           ← Class index → code mapping
│   ├── history.json             ← Training loss/acc history
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   ├── roc_curves.png
│   └── checkpoints/             ← Epoch checkpoints
├── utils/
│   ├── preprocess.py            ← Image transforms
│   └── helpers.py               ← Shared utilities
├── ui/                          ← React frontend
│   ├── src/
│   │   ├── App.jsx
│   │   ├── main.jsx
│   │   ├── index.css
│   │   └── components/
│   │       ├── Header.jsx
│   │       ├── UploadZone.jsx
│   │       ├── ResultPanel.jsx
│   │       ├── ModelStatus.jsx
│   │       ├── ClassesGrid.jsx
│   │       ├── ParticleBackground.jsx
│   │       └── Footer.jsx
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   └── tailwind.config.js
├── dataset.py                   ← PyTorch Dataset + DataLoaders
├── model.py                     ← ResNet model + Grad-CAM
├── train.py                     ← Training pipeline
├── evaluate.py                  ← Metrics + plots
├── infer.py                     ← Single-image CLI inference
├── api.py                       ← FastAPI backend
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

### 1. Python Environment

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

# Install Python dependencies
pip install -r requirements.txt
```

### 2. Dataset Setup

Download the **HAM10000** dataset from Kaggle:
👉 https://www.kaggle.com/datasets/kmader/skin-lesion-analysis-toward-melanoma-detection

Place files as follows:
```
data/raw/
├── images/
│   ├── ISIC_0024306.jpg
│   ├── ISIC_0024307.jpg
│   └── ...                   (10,015 images total)
└── HAM10000_metadata.csv
```

> The CSV must have columns: `lesion_id`, `image_id`, `dx` (diagnosis code)

### 3. React Frontend Setup

```bash
cd ui
npm install
```

---

## 🏋️ Training

```bash
# Basic training (ResNet50, 30 epochs, GPU auto-detected)
python train.py

# Custom training with all options
python train.py \
  --model resnet50 \
  --epochs 50 \
  --batch_size 32 \
  --lr 1e-4 \
  --patience 8 \
  --unfreeze_at 15 \
  --val_size 0.2 \
  --output_dir models

# Use ResNet18 for faster training (less accurate)
python train.py --model resnet18 --epochs 25
```

**Training flags:**

| Flag           | Default                            | Description                              |
|----------------|------------------------------------|------------------------------------------|
| `--model`      | `resnet50`                         | `resnet18` or `resnet50`                 |
| `--epochs`     | `30`                               | Total training epochs                    |
| `--batch_size` | `32`                               | Images per batch                         |
| `--lr`         | `1e-4`                             | Initial learning rate (Adam)             |
| `--patience`   | `7`                                | Early stopping patience                  |
| `--unfreeze_at`| `10`                               | Epoch to unfreeze all layers             |
| `--val_size`   | `0.2`                              | Fraction of data for validation          |
| `--output_dir` | `models`                           | Directory to save checkpoints and plots  |

**Expected training output:**
```
============================================================
  SKIN CANCER DETECTION — TRAINING
============================================================
  Model      : resnet50
  Device     : cuda (NVIDIA RTX 3090)
  Epochs     : 30
  Batch size : 32
  LR         : 0.0001
============================================================
[Dataset] Total unique samples: 9,148
[Dataset] Class distribution:
  nv       6,705
  mel      1,113
  bkl       902
  ...

EPOCH 1/30  |  LR: 1.00e-04
──────────────────────────────
  [Epoch 1] Batch 20/285 | Loss: 1.2341 | Acc: 48.23% | Time: 12.4s
  ...
  ▶ Train  — Loss: 0.8921 | Acc: 71.44%
  ▶ Val    — Loss: 0.7233 | Acc: 76.82%
  ✅ New best model saved (Val Acc: 76.82%)
```

---

## 📊 Evaluation

```bash
python evaluate.py \
  --checkpoint models/best_model.pth \
  --csv data/raw/HAM10000_metadata.csv \
  --img_dir data/raw/images
```

**Generates:**
- `models/confusion_matrix.png` — Per-class confusion heatmap
- `models/roc_curves.png` — One-vs-rest ROC curves (Macro AUC)
- `models/per_class_accuracy.png` — Bar chart of per-class accuracy
- `models/classification_report.txt` — Precision, Recall, F1 per class

---

## 🔍 CLI Inference (Single Image)

```bash
python infer.py \
  --image path/to/lesion.jpg \
  --checkpoint models/best_model.pth \
  --output result.png \
  --topk 3
```

**Output:**
```
TOP PREDICTIONS:
  ──────────────────────────────────────────────────
  #1  🔴  Melanoma (mel)
       Confidence : 87.43%
       Risk Level : HIGH
       Info       : Dangerous skin cancer. Requires immediate medical attention.

  #2  🟢  Melanocytic Nevi (nv)
       Confidence : 8.21%
       Risk Level : LOW
       ...
```

---

## 🌐 Running the Full Stack

### Step 1 — Start FastAPI Backend

```bash
# Ensure training is complete and models/best_model.pth exists
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

API will be available at: `http://localhost:8000`
- **Swagger docs:** `http://localhost:8000/docs`
- **Health check:** `http://localhost:8000/health`

### Step 2 — Start React Frontend

```bash
cd ui
npm run dev
```

UI will open at: **`http://localhost:3000`**

---

## 🎨 React UI Features

| Feature                 | Description                                              |
|-------------------------|----------------------------------------------------------|
| **Drag & Drop Upload**  | Drop JPEG/PNG images directly onto the upload zone       |
| **Live Scan Animation** | Visual scanning effect while analyzing                   |
| **Primary Diagnosis**   | Top prediction with confidence ring + risk level badge   |
| **Grad-CAM Overlay**    | Side-by-side original vs. heatmap explainability view    |
| **Confidence Bars**     | Color-coded animated bars for all 7 classes              |
| **Bar + Radar Charts**  | Visual probability distribution using Recharts           |
| **Disease Guide**       | Collapsible reference for all 7 HAM10000 classes         |
| **API Status**          | Real-time model health indicator in header               |
| **Particle Canvas**     | Animated neural-network background                       |

---

## 🧠 Model Architecture

```
Input (224×224×3)
       ↓
ResNet50 Backbone (pretrained ImageNet)
  ├── Conv1 + BN + ReLU + MaxPool
  ├── Layer1 (3× Bottleneck)   ← Frozen initially
  ├── Layer2 (4× Bottleneck)   ← Frozen initially
  ├── Layer3 (6× Bottleneck)   ← Trainable
  └── Layer4 (3× Bottleneck)   ← Trainable + Grad-CAM hook
       ↓
  AdaptiveAvgPool2d → Flatten [2048]
       ↓
  Custom Head:
    Dropout(0.4) → Linear(2048→512) → ReLU → BatchNorm → Dropout(0.2) → Linear(512→7)
       ↓
  Softmax → 7-class probabilities
```

**Key design decisions:**
- **Weighted CrossEntropyLoss** — Handles the severe class imbalance (nv: 67% vs vasc: 1%)
- **CosineAnnealingLR** — Smooth learning rate decay
- **Layer unfreezing at epoch N** — Starts with frozen early layers for stable feature extraction, then unfreezes for fine-tuning
- **Grad-CAM** — Forward/backward hooks on `layer4` for visual explanations

---

## 📈 Expected Results

| Metric              | ResNet18    | ResNet50    |
|---------------------|-------------|-------------|
| Val Accuracy        | ~78–82%     | ~82–87%     |
| Macro AUC           | ~0.88–0.91  | ~0.91–0.95  |
| Melanoma Recall     | ~70–75%     | ~75–82%     |
| Training Time (GPU) | ~25 min     | ~45 min     |

> Results vary based on GPU, batch size, augmentation, and random seed.

---

## 🔬 Grad-CAM Explainability

The model uses **Gradient-weighted Class Activation Mapping (Grad-CAM)** to generate
visual explanations of predictions. This highlights which regions of the skin image
were most influential in making the classification decision.

- **Red** regions = high model attention
- **Blue** regions = low model attention

---

## ⚠️ Medical Disclaimer

> **This software is strictly for research and educational purposes.**
> It is NOT a substitute for professional medical advice, diagnosis, or treatment.
> Always consult a qualified dermatologist for any skin concerns.

---

## 📜 License

MIT License — see LICENSE for details.

---

## 🙏 Acknowledgements

- **HAM10000 Dataset** — Tschandl, P., Rosendahl, C., & Kittler, H. (2018)
- **PyTorch** & **torchvision** teams
- **Recharts** for React chart components
