"""
HRFAE Training Script
======================
Train the HRFAE model from scratch on the UTKFace dataset.

UTKFace contains ~23K face images labelled by age, gender, and ethnicity.
Download from: https://susanqq.github.io/UTKFace/

Usage:
    # 1. Download UTKFace and extract to a folder:
    #    agevision_backend/datasets/UTKFace/
    #
    # 2. Run training:
    python -m agevision_api.hrfae.train \\
        --data_dir datasets/UTKFace \\
        --epochs 100 \\
        --batch_size 8 \\
        --lr 0.0002

    # 3. Trained weights are saved to:
    #    agevision_backend/checkpoints/hrfae_best.pth

Typical training times:
    GPU (RTX 3060+) : ~2-4 hours for 100 epochs
    CPU             : ~20-30 hours for 100 epochs
"""

import os
import sys
import time
import glob
import random
import argparse

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)))

from agevision_api.hrfae.model import HRFAE, Dis_PatchGAN


# ═══════════════════════════════════════════════════════════════
#  UTKFace Dataset
# ═══════════════════════════════════════════════════════════════

class UTKFaceDataset(Dataset):
    """
    UTKFace dataset loader.
    Files are named: [age]_[gender]_[race]_[date&time].jpg

    Returns images in [0, 1] range and integer ages (0–100).
    """

    def __init__(self, root_dir: str, transform=None, max_age: int = 100):
        self.transform = transform
        self.max_age = max_age
        self.samples = []

        patterns = ['*.jpg', '*.png', '*.jpeg', '*.JPG', '*.chip.jpg']
        all_files = []
        for pat in patterns:
            all_files.extend(glob.glob(os.path.join(root_dir, pat)))

        for fpath in sorted(all_files):
            fname = os.path.basename(fpath)
            parts = fname.split('_')
            if len(parts) >= 3:
                try:
                    age = int(parts[0])
                    if 0 <= age <= max_age:
                        self.samples.append((fpath, age))
                except ValueError:
                    continue

        print(f"[UTKFace] Loaded {len(self.samples)} samples from {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fpath, age = self.samples[idx]
        img = cv2.imread(fpath)
        if img is None:
            img = np.zeros((256, 256, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        from PIL import Image as PILImage
        img_pil = PILImage.fromarray(img)

        if self.transform:
            img_tensor = self.transform(img_pil)
        else:
            img_tensor = transforms.ToTensor()(img_pil)

        # Current age as integer
        age_int = torch.tensor(age, dtype=torch.long)

        # Random target age for cross-age training
        target_age = random.randint(0, self.max_age)
        target_int = torch.tensor(target_age, dtype=torch.long)

        return {
            'image': img_tensor,        # [3, 256, 256] in [0, 1]
            'age': age_int,              # integer 0–100
            'target_age': target_int,    # integer 0–100
        }


# ═══════════════════════════════════════════════════════════════
#  Training Loop
# ═══════════════════════════════════════════════════════════════

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Train] Device: {device}")

    # ── Transforms (no mean/std normalisation – HRFAE uses [0, 1]) ──
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),   # → [0, 1]
    ])

    # ── Dataset ──
    dataset = UTKFaceDataset(args.data_dir, transform=train_transform)
    if len(dataset) == 0:
        print("ERROR: No valid images found in the data directory.")
        print("Make sure UTKFace images are in:", args.data_dir)
        print("Files should be named like: 25_0_0_20170119202748439.jpg")
        return

    n_val = max(1, len(dataset) // 10)
    n_train = len(dataset) - n_val
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size,
                            shuffle=False, num_workers=0)

    print(f"[Train] Train: {n_train} | Val: {n_val}")

    # ── Models ──
    model = HRFAE().to(device)
    disc = Dis_PatchGAN().to(device)

    # ── Optimizers ──
    opt_g = optim.Adam(
        list(model.encoder.parameters()) +
        list(model.decoder.parameters()) +
        list(model.mod_net.parameters()),
        lr=args.lr, betas=(0.5, 0.999),
    )
    opt_d = optim.Adam(disc.parameters(), lr=args.lr, betas=(0.5, 0.999))

    # ── LR schedulers ──
    sched_g = optim.lr_scheduler.StepLR(opt_g, step_size=30, gamma=0.5)
    sched_d = optim.lr_scheduler.StepLR(opt_d, step_size=30, gamma=0.5)

    # ── Losses ──
    l1_loss = nn.L1Loss()
    mse_loss = nn.MSELoss()   # LSGAN

    # ── Loss weights ──
    w_recon = 10.0    # Self-reconstruction
    w_adv_g = 1.0     # Generator adversarial
    w_ident = 5.0     # Identity preservation

    # ── Checkpoint dir ──
    os.makedirs(args.ckpt_dir, exist_ok=True)
    best_val_loss = float('inf')

    print(f"[Train] Starting training for {args.epochs} epochs...")
    print("-" * 70)

    for epoch in range(1, args.epochs + 1):
        model.train()
        disc.train()
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        t_epoch = time.time()

        for batch_idx, batch in enumerate(train_loader):
            imgs = batch['image'].to(device)            # [B, 3, 256, 256]
            ages = batch['age'].to(device)              # [B] int
            target_ages = batch['target_age'].to(device)  # [B] int
            bs = imgs.size(0)

            # ── Self-reconstruction (same age) ──
            recon = model(imgs, ages)
            loss_recon = l1_loss(recon, imgs) * w_recon

            # ── Cross-age transformation ──
            aged = model(imgs, target_ages)

            # ── Discriminator update ──
            real_out = disc(imgs)
            fake_out = disc(aged.detach())

            real_label = torch.ones_like(real_out)
            fake_label = torch.zeros_like(fake_out)

            loss_d = (mse_loss(real_out, real_label) +
                      mse_loss(fake_out, fake_label)) * 0.5

            opt_d.zero_grad()
            loss_d.backward()
            opt_d.step()

            # ── Generator update ──
            fake_out_g = disc(aged)
            loss_adv_g = mse_loss(fake_out_g, real_label) * w_adv_g

            # Identity preservation via encoder features
            with torch.no_grad():
                feat_orig = model.get_content(imgs)
            feat_aged = model.get_content(aged)
            loss_id = l1_loss(feat_aged, feat_orig) * w_ident

            loss_g = loss_recon + loss_adv_g + loss_id

            opt_g.zero_grad()
            loss_g.backward()
            opt_g.step()

            epoch_g_loss += loss_g.item()
            epoch_d_loss += loss_d.item()

            if (batch_idx + 1) % 50 == 0:
                print(f"  Epoch {epoch}/{args.epochs} | "
                      f"Batch {batch_idx + 1}/{len(train_loader)} | "
                      f"G: {loss_g.item():.4f} | D: {loss_d.item():.4f}")

        sched_g.step()
        sched_d.step()

        # ── Validation ──
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch['image'].to(device)
                ages = batch['age'].to(device)
                recon = model(imgs, ages)
                val_loss += l1_loss(recon, imgs).item()
        val_loss /= max(len(val_loader), 1)

        avg_g = epoch_g_loss / max(len(train_loader), 1)
        avg_d = epoch_d_loss / max(len(train_loader), 1)
        elapsed = time.time() - t_epoch

        print(f"Epoch {epoch}/{args.epochs} | "
              f"G: {avg_g:.4f} | D: {avg_d:.4f} | "
              f"Val: {val_loss:.4f} | "
              f"LR: {sched_g.get_last_lr()[0]:.6f} | "
              f"Time: {elapsed:.0f}s")

        # ── Save best ──
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = os.path.join(args.ckpt_dir, 'hrfae_best.pth')
            torch.save({
                'epoch': epoch,
                'enc': model.encoder.state_dict(),
                'dec': model.decoder.state_dict(),
                'mod': model.mod_net.state_dict(),
                'dis': disc.state_dict(),
                'opt_g': opt_g.state_dict(),
                'opt_d': opt_d.state_dict(),
                'val_loss': val_loss,
            }, ckpt_path)
            print(f"  >>> Saved best model (val={val_loss:.4f}) -> {ckpt_path}")

        # ── Periodic checkpoint ──
        if epoch % 10 == 0:
            ckpt_path = os.path.join(args.ckpt_dir, f'hrfae_epoch_{epoch:04d}.pth')
            torch.save({
                'epoch': epoch,
                'enc': model.encoder.state_dict(),
                'dec': model.decoder.state_dict(),
                'mod': model.mod_net.state_dict(),
                'val_loss': val_loss,
            }, ckpt_path)
            print(f"  >>> Checkpoint -> {ckpt_path}")

    print("=" * 70)
    print(f"Training complete. Best val loss: {best_val_loss:.4f}")
    print(f"Best model saved at: {os.path.join(args.ckpt_dir, 'hrfae_best.pth')}")


# ═══════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train HRFAE on UTKFace')
    parser.add_argument('--data_dir', type=str, default='datasets/UTKFace',
                        help='Path to UTKFace images directory')
    parser.add_argument('--ckpt_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='Learning rate')
    args = parser.parse_args()

    train(args)
