"""
dataset.py
----------
Custom PyTorch Dataset class for the HAM10000 skin lesion dataset.
Handles CSV loading, label encoding, train/val splitting, and augmentation.
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset, DataLoader
from utils.preprocess import get_transforms


# HAM10000 class label mapping
CLASS_NAMES = {
    "nv":   "Melanocytic Nevi",
    "mel":  "Melanoma",
    "bkl":  "Benign Keratosis-like Lesions",
    "bcc":  "Basal Cell Carcinoma",
    "akiec":"Actinic Keratoses",
    "vasc": "Vascular Lesions",
    "df":   "Dermatofibroma",
}

CLASS_DESCRIPTIONS = {
    "nv":   "Common benign mole. Usually harmless but monitor for changes.",
    "mel":  "Dangerous skin cancer. Requires immediate medical attention.",
    "bkl":  "Non-cancerous growths. Generally harmless.",
    "bcc":  "Common skin cancer. Usually treatable when caught early.",
    "akiec":"Pre-cancerous lesion. Treatment recommended.",
    "vasc": "Blood vessel lesion. Usually benign.",
    "df":   "Benign fibrous skin nodule. Usually harmless.",
}

RISK_LEVEL = {
    "nv":   "low",
    "mel":  "high",
    "bkl":  "low",
    "bcc":  "medium",
    "akiec":"medium",
    "vasc": "low",
    "df":   "low",
}


class SkinLesionDataset(Dataset):
    """
    PyTorch Dataset for HAM10000 skin lesion images.

    Args:
        dataframe  (pd.DataFrame): Filtered dataframe with 'image_id' and 'dx' columns.
        img_dir    (str): Path to folder containing .jpg images.
        transform  (callable, optional): Torchvision transforms to apply.
        label_encoder (LabelEncoder): Fitted sklearn LabelEncoder instance.
    """

    def __init__(self, dataframe: pd.DataFrame, img_dir: str,
                 transform=None, label_encoder: LabelEncoder = None):
        self.df           = dataframe.reset_index(drop=True)
        self.img_dir      = img_dir
        self.transform    = transform
        self.label_encoder = label_encoder

        # Encode string labels → integer indices
        if label_encoder is not None:
            self.labels = label_encoder.transform(self.df["dx"].values)
        else:
            self.labels = self.df["dx"].values

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        img_name = self.df.loc[idx, "image_id"] + ".jpg"
        img_path = os.path.join(self.img_dir, img_name)

        if not os.path.exists(img_path):
            raise FileNotFoundError(
                f"Image not found: {img_path}\n"
                "Please ensure the HAM10000 images are in the data/raw/images/ directory."
            )

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label


def load_dataset(csv_path: str, img_dir: str,
                 val_size: float = 0.2,
                 batch_size: int = 32,
                 num_workers: int = 4,
                 seed: int = 42):
    """
    Full pipeline: reads CSV → encodes labels → splits → builds DataLoaders.

    Args:
        csv_path   (str): Path to HAM10000_metadata.csv
        img_dir    (str): Path to image folder
        val_size   (float): Fraction of data for validation
        batch_size (int): DataLoader batch size
        num_workers(int): Number of parallel workers
        seed       (int): Random seed for reproducibility

    Returns:
        train_loader, val_loader, label_encoder, class_weights
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"CSV not found: {csv_path}\n"
            "Download HAM10000_metadata.csv from Kaggle and place it in data/raw/"
        )

    df = pd.read_csv(csv_path)

    # Drop duplicates (HAM10000 has duplicated lesion images)
    df = df.drop_duplicates(subset=["lesion_id"]).reset_index(drop=True)

    print(f"[Dataset] Total unique samples: {len(df)}")
    print(f"[Dataset] Class distribution:\n{df['dx'].value_counts()}\n")

    # Fit label encoder
    le = LabelEncoder()
    le.fit(df["dx"].values)
    num_classes = len(le.classes_)
    print(f"[Dataset] Classes ({num_classes}): {list(le.classes_)}")

    # Compute class weights to handle imbalance
    class_counts = df["dx"].value_counts().reindex(le.classes_).values
    total        = class_counts.sum()
    weights      = total / (num_classes * class_counts)
    class_weights = torch.tensor(weights, dtype=torch.float32)

    # Stratified train / val split
    train_df, val_df = train_test_split(
        df, test_size=val_size, random_state=seed,
        stratify=df["dx"]
    )

    print(f"[Dataset] Train: {len(train_df)} | Val: {len(val_df)}")

    train_transforms, val_transforms = get_transforms()

    train_dataset = SkinLesionDataset(train_df, img_dir, train_transforms, le)
    val_dataset   = SkinLesionDataset(val_df,   img_dir, val_transforms,   le)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, le, class_weights


if __name__ == "__main__":
    # Quick sanity check
    train_loader, val_loader, le, cw = load_dataset(
        csv_path="data/raw/HAM10000_metadata.csv",
        img_dir="data/raw/images",
        batch_size=4
    )
    imgs, lbls = next(iter(train_loader))
    print(f"[Test] Batch shape: {imgs.shape}, Labels: {lbls}")
