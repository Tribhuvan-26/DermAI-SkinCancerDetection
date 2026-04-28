"""
model.py
--------
ResNet-based CNN for multi-class skin lesion classification.
Supports ResNet18 and ResNet50 with configurable layer freezing.
Includes Grad-CAM hooks for explainability.
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional, Tuple
import numpy as np


class SkinCancerResNet(nn.Module):
    """
    Transfer-learning model using a pretrained ResNet backbone.

    Architecture:
        ResNet18 / ResNet50  →  Custom FC head  →  num_classes outputs

    Args:
        num_classes  (int): Number of output classes (7 for HAM10000).
        model_name   (str): 'resnet18' or 'resnet50'.
        pretrained   (bool): Load ImageNet weights.
        freeze_layers(int): Number of initial ResNet layers to freeze (0–4).
        dropout_rate (float): Dropout before final FC layer.
    """

    def __init__(
        self,
        num_classes: int  = 7,
        model_name: str   = "resnet50",
        pretrained: bool  = True,
        freeze_layers: int = 2,
        dropout_rate: float = 0.4,
    ):
        super(SkinCancerResNet, self).__init__()

        self.model_name  = model_name
        self.num_classes = num_classes

        # ── Load backbone ──────────────────────────────────────────────
        if model_name == "resnet18":
            weights   = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            backbone  = models.resnet18(weights=weights)
            fc_in_features = 512
        elif model_name == "resnet50":
            weights   = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            backbone  = models.resnet50(weights=weights)
            fc_in_features = 2048
        else:
            raise ValueError(f"model_name must be 'resnet18' or 'resnet50', got '{model_name}'")

        # ── Extract layers (for layer-level freezing) ──────────────────
        self.conv1   = backbone.conv1
        self.bn1     = backbone.bn1
        self.relu    = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1  = backbone.layer1
        self.layer2  = backbone.layer2
        self.layer3  = backbone.layer3
        self.layer4  = backbone.layer4
        self.avgpool = backbone.avgpool

        # ── Freeze early layers ────────────────────────────────────────
        frozen = [self.conv1, self.bn1, self.layer1, self.layer2][:freeze_layers]
        for layer in frozen:
            for param in layer.parameters():
                param.requires_grad = False

        # ── Custom classification head ─────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(fc_in_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout_rate / 2),
            nn.Linear(512, num_classes),
        )

        # ── Grad-CAM hook storage ──────────────────────────────────────
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        self._register_hooks()

    def _register_hooks(self):
        """Register forward/backward hooks on layer4 for Grad-CAM."""
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.layer4.register_forward_hook(forward_hook)
        self.layer4.register_full_backward_hook(backward_hook)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def get_gradcam(self, x: torch.Tensor, class_idx: Optional[int] = None
                    ) -> Tuple[np.ndarray, int]:
        """
        Generate Grad-CAM heatmap for the given input image tensor.

        Args:
            x         : Input tensor [1, C, H, W]
            class_idx : Target class index (None → argmax of prediction)

        Returns:
            heatmap (np.ndarray, H×W float), predicted class index
        """
        self.eval()
        x.requires_grad_(True)

        logits = self.forward(x)
        pred_class = logits.argmax(dim=1).item() if class_idx is None else class_idx

        # Backprop only through the target class score
        self.zero_grad()
        logits[0, pred_class].backward()

        # Pool gradients across spatial dimensions
        grads = self.gradients           # [1, C, h, w]
        acts  = self.activations         # [1, C, h, w]

        weights = grads.mean(dim=[2, 3], keepdim=True)   # [1, C, 1, 1]
        cam     = (weights * acts).sum(dim=1).squeeze()   # [h, w]
        cam     = torch.clamp(cam, min=0)

        # Normalize to [0, 1]
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()

        return cam.cpu().numpy(), pred_class

    def unfreeze_all(self):
        """Unfreeze all parameters for full fine-tuning."""
        for param in self.parameters():
            param.requires_grad = True

    def trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_model(num_classes: int = 7, model_name: str = "resnet50",
                pretrained: bool = True, device: str = "cpu") -> SkinCancerResNet:
    """
    Factory function to build and move model to device.

    Args:
        num_classes (int): Number of output classes.
        model_name  (str): 'resnet18' or 'resnet50'.
        pretrained  (bool): Use ImageNet pretrained weights.
        device      (str): 'cuda' or 'cpu'.

    Returns:
        SkinCancerResNet instance on the specified device.
    """
    model = SkinCancerResNet(num_classes=num_classes, model_name=model_name,
                              pretrained=pretrained)
    model = model.to(device)

    print(f"[Model] Architecture : {model_name}")
    print(f"[Model] Total params : {model.total_params():,}")
    print(f"[Model] Trainable    : {model.trainable_params():,}")
    print(f"[Model] Device       : {device}")
    return model


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = build_model(num_classes=7, model_name="resnet50", device=device)

    dummy = torch.randn(2, 3, 224, 224).to(device)
    out   = model(dummy)
    print(f"[Test] Output shape: {out.shape}")   # Expected: [2, 7]
