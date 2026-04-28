"""
utils/preprocess.py
--------------------
Image transformations for HAM10000 skin lesion images.
Provides separate augmentation pipelines for training and inference.
"""

from torchvision import transforms
from typing import Tuple


# ImageNet normalization stats (used for pretrained ResNet)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Standard input size for ResNet
INPUT_SIZE = 224


def get_transforms(input_size: int = INPUT_SIZE) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Returns (train_transform, val_transform).

    Training augmentations:
        - Random resized crop (scale 0.7–1.0)
        - Random horizontal + vertical flip
        - Random rotation ±30°
        - Color jitter (brightness, contrast, saturation)
        - Gaussian blur (occasional)
        - Normalize with ImageNet stats

    Validation transforms:
        - Resize to 256 → Center crop 224
        - Normalize only
    """

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(
            input_size,
            scale=(0.7, 1.0),
            ratio=(0.9, 1.1),
            interpolation=transforms.InterpolationMode.BILINEAR,
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(
            brightness=0.25,
            contrast=0.25,
            saturation=0.25,
            hue=0.05,
        ),
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))],
            p=0.2
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(int(input_size * 1.14)),          # 256 for 224 target
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return train_transform, val_transform


def get_inference_transform(input_size: int = INPUT_SIZE) -> transforms.Compose:
    """
    Single transform for inference (same as validation, no augmentation).
    """
    return transforms.Compose([
        transforms.Resize(int(input_size * 1.14)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def denormalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """
    Undo ImageNet normalization for visualization purposes.

    Args:
        tensor: Normalized image tensor [C, H, W]

    Returns:
        Denormalized tensor [C, H, W] clipped to [0, 1]
    """
    import torch
    t = tensor.clone()
    for c, (m, s) in enumerate(zip(mean, std)):
        t[c] = t[c] * s + m
    return t.clamp(0, 1)
