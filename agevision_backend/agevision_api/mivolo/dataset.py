"""
Indian Face Dataset for MiVOLO v2 Fine-tuning
===============================================
Loads face images with age/gender labels from a CSV file.
Uses HuggingFace AutoImageProcessor for preprocessing to match
the pretrained MiVOLO v2 input format.

Returns:
  - face_pixel_values: [3, 224, 224] preprocessed face tensor
  - body_pixel_values: [3, 224, 224] black placeholder (face-only training)
  - age: float (regression target)
  - gender: int (0=male, 1=female)
"""

import csv
import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T


class IndianFaceMiVOLODataset(Dataset):
    """
    Dataset for MiVOLO v2 fine-tuning on Indian face images.

    CSV format:
        filename,age,gender,source
        utk_25_0_3_201701.jpg,25,Male,utkface
        ifad_actor1_img3.jpg,42,Unknown,ifad
    """

    GENDER_MAP = {"male": 0, "man": 0, "m": 0,
                  "female": 1, "woman": 1, "f": 1}

    def __init__(self, data_root: str, labels_csv: str,
                 processor=None, augment: bool = False):
        """
        Parameters
        ----------
        data_root : str
            Directory containing face images.
        labels_csv : str
            Path to CSV with columns: filename, age, gender, source.
        processor : AutoImageProcessor
            HuggingFace image processor for MiVOLO v2.
        augment : bool
            Apply data augmentation (training only).
        """
        self.data_root = data_root
        self.processor = processor
        self.augment = augment
        self.samples = []

        # Augmentation transforms (applied before processor)
        self.aug_transform = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
            T.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        ]) if augment else None

        with open(labels_csv, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                path = os.path.join(data_root, row["filename"])
                if not os.path.isfile(path):
                    continue

                age = float(row["age"])
                age = max(0.0, min(100.0, age))

                gender_str = row.get("gender", "unknown").strip().lower()
                gender = self.GENDER_MAP.get(gender_str, -1)

                self.samples.append({
                    "path": path,
                    "age": age,
                    "gender": gender,
                })

        if not self.samples:
            raise ValueError(f"No valid samples found in {labels_csv} "
                             f"with images in {data_root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = Image.open(sample["path"]).convert("RGB")

        # Apply augmentation
        if self.aug_transform is not None:
            img = self.aug_transform(img)

        # MiVOLO processor expects numpy arrays, not PIL images
        img_np = np.array(img)
        face_inputs = self.processor(images=[img_np])["pixel_values"]
        face_tensor = face_inputs[0]  # [3, H, W]

        # Body placeholder: black image (no body context for face-only training)
        body_placeholder = torch.zeros_like(face_tensor)

        return {
            "face_pixel_values": face_tensor,
            "body_pixel_values": body_placeholder,
            "age": torch.tensor(sample["age"], dtype=torch.float32),
            "gender": torch.tensor(sample["gender"], dtype=torch.long),
        }


def create_data_loaders(config, processor):
    """
    Create train and validation data loaders.

    Parameters
    ----------
    config : MiVOLOTrainConfig
        Training configuration.
    processor : AutoImageProcessor
        HuggingFace image processor for MiVOLO v2.

    Returns
    -------
    train_loader, val_loader
    """
    # Full dataset without augmentation for splitting
    full_dataset = IndianFaceMiVOLODataset(
        data_root=config.data_root,
        labels_csv=config.labels_csv,
        processor=processor,
        augment=False,
    )

    total = len(full_dataset)
    val_size = max(1, int(total * config.val_split))
    train_size = total - val_size

    train_indices, val_indices = random_split(
        range(total), [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    # Create separate datasets for train (with augment) and val (without)
    train_dataset = IndianFaceMiVOLODataset(
        data_root=config.data_root,
        labels_csv=config.labels_csv,
        processor=processor,
        augment=True,
    )
    val_dataset = IndianFaceMiVOLODataset(
        data_root=config.data_root,
        labels_csv=config.labels_csv,
        processor=processor,
        augment=False,
    )

    # Subset using split indices
    train_subset = torch.utils.data.Subset(train_dataset, train_indices.indices)
    val_subset = torch.utils.data.Subset(val_dataset, val_indices.indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader
