"""
Indian Face Dataset for SAM Fine-tuning
========================================
Expects a folder of face images and a CSV file with columns: filename, age
Optionally: gender column.

The dataset returns:
  - input: [4, 256, 256] tensor (RGB + age channel via AgeTransformer)
  - target: [3, 256, 256] tensor (original image, for self-reconstruction)
  - real_age: int (0-100)
  - target_age: int (age used for the age channel)
"""

import csv
import os
import random

import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from .datasets.augmentations import AgeTransformer

IMG_SIZE = 256

# Standard preprocessing matching SAM inference
TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # → [-1, 1]
])

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


class IndianFaceDataset(Dataset):
    """
    Dataset for Indian face images with age labels.

    CSV format:
        filename,age[,gender]
        img001.jpg,25,Male
        img002.jpg,42,Female
    """

    def __init__(self, data_root: str, age_csv: str, transform=None,
                 mode: str = 'self_reconstruct'):
        """
        Parameters
        ----------
        data_root : str
            Directory containing face images.
        age_csv : str
            Path to CSV file with columns: filename, age.
        transform : callable, optional
            Image transform pipeline. Defaults to TRAIN_TRANSFORM.
        mode : str
            'self_reconstruct' - target_age equals real_age (learns identity)
            'random_age' - target_age is random (learns age transformation)
            'both' - 50/50 mix of self_reconstruct and random_age
        """
        self.data_root = data_root
        self.transform = transform or TRAIN_TRANSFORM
        self.mode = mode
        self.samples = []

        with open(age_csv, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                path = os.path.join(data_root, row['filename'])
                if os.path.isfile(path):
                    age = int(float(row['age']))
                    age = max(0, min(100, age))
                    self.samples.append({
                        'path': path,
                        'age': age,
                        'gender': row.get('gender', 'Unknown'),
                    })

        if not self.samples:
            raise ValueError(f"No valid samples found in {age_csv} "
                             f"with images in {data_root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = Image.open(sample['path']).convert('RGB')
        real_age = sample['age']

        # Apply image transform (resize, flip, normalize)
        img_tensor = self.transform(img)

        # Determine target age based on mode
        if self.mode == 'self_reconstruct':
            target_age = real_age
        elif self.mode == 'random_age':
            target_age = random.randint(0, 100)
        else:  # 'both'
            target_age = real_age if random.random() < 0.5 else random.randint(0, 100)

        # Add age channel (4th channel) via AgeTransformer
        age_transformer = AgeTransformer(target_age=target_age)
        input_tensor = age_transformer(img_tensor)  # [4, H, W]

        return {
            'input': input_tensor,           # [4, 256, 256]
            'target': img_tensor,            # [3, 256, 256] ground truth
            'real_age': real_age,
            'target_age': target_age,
        }


def create_data_loaders(config):
    """
    Create train and validation data loaders from config.

    Parameters
    ----------
    config : SAMTrainConfig
        Training configuration with data_root, age_csv, batch_size, etc.

    Returns
    -------
    train_loader, val_loader (or None if no validation split)
    """
    from torch.utils.data import DataLoader, random_split

    dataset = IndianFaceDataset(
        data_root=config.data_root,
        age_csv=config.age_csv,
        transform=TRAIN_TRANSFORM,
        mode='both',
    )

    # 90/10 train/val split
    total = len(dataset)
    val_size = max(1, int(total * 0.1))
    train_size = total - val_size

    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader
