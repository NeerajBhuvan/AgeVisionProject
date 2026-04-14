"""
SAM Fine-tuning Configuration
==============================
Dataclass holding all hyperparameters for fine-tuning the SAM model
(pSp encoder + StyleGAN2 decoder) on Indian face data.
"""

from dataclasses import dataclass


@dataclass
class SAMTrainConfig:
    # ── Paths ─────────────────────────────────────────────────────────
    checkpoint_path: str = ''           # Source checkpoint (sam_ffhq_aging.pt)
    output_dir: str = 'checkpoints/sam_finetune'
    data_root: str = ''                 # Folder containing face images
    age_csv: str = ''                   # CSV with columns: filename, age

    # ── Architecture (must match source checkpoint) ───────────────────
    input_nc: int = 4                   # 3 RGB + 1 age channel
    output_size: int = 1024             # StyleGAN2 output resolution
    start_from_latent_avg: bool = False
    start_from_encoded_w_plus: bool = True

    # ── Training ──────────────────────────────────────────────────────
    batch_size: int = 4
    num_epochs: int = 50
    lr_encoder: float = 1e-4
    lr_decoder: float = 1e-5            # Lower LR for decoder (transfer learning)
    lr_discriminator: float = 1e-4
    weight_decay: float = 1e-5
    save_every: int = 5                 # Save checkpoint every N epochs
    log_every: int = 50                 # Log metrics every N batches

    # ── Loss weights ──────────────────────────────────────────────────
    lambda_l1: float = 1.0              # L1 pixel reconstruction
    lambda_identity: float = 0.5        # ArcFace cosine identity preservation
    lambda_age: float = 0.1             # Age classification cross-entropy
    lambda_adv: float = 0.01            # Adversarial (discriminator) loss
    lambda_lpips: float = 0.8           # Perceptual loss (LPIPS)

    # ── Data ──────────────────────────────────────────────────────────
    img_size: int = 256
    num_workers: int = 2                # Windows-safe default
    pin_memory: bool = True
    age_bins: int = 101                 # 0-100 inclusive for cross-entropy

    # ── Memory optimization ─────────────────────────────────────────
    freeze_decoder: bool = True         # Freeze decoder (encoder-only fine-tuning, saves ~8GB VRAM)
    gradient_accumulation_steps: int = 1  # Accumulate gradients over N steps
    gradient_checkpointing: bool = True  # Checkpoint decoder to save ~10GB VRAM (trades compute for memory)

    # ── Device ────────────────────────────────────────────────────────
    device: str = 'cuda'
    mixed_precision: bool = True        # Use torch.amp for memory savings
