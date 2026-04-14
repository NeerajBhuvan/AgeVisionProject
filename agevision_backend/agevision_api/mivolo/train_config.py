"""
MiVOLO v2 Fine-tuning Configuration
=====================================
Dataclass holding all hyperparameters for fine-tuning MiVOLO v2
(Vision Transformer) on Indian face data (IFAD + UTKFace-Indian).
"""

from dataclasses import dataclass


@dataclass
class MiVOLOTrainConfig:
    # ── Paths ─────────────────────────────────────────────────────────
    model_name: str = "iitolstykh/mivolo_v2"   # HuggingFace model ID
    output_dir: str = "checkpoints/mivolo_indian"
    data_root: str = ""                         # Folder containing face images
    labels_csv: str = ""                        # CSV: filename, age, gender, source

    # ── Training ──────────────────────────────────────────────────────
    batch_size: int = 16
    num_epochs: int = 30
    learning_rate: float = 2e-5                 # Low LR for fine-tuning ViT
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1                   # 10% warmup steps
    lr_scheduler: str = "cosine"                # cosine | linear | constant
    save_every: int = 5                         # Save checkpoint every N epochs
    log_every: int = 25                         # Log metrics every N batches
    early_stopping_patience: int = 7            # Stop after N epochs without improvement

    # ── Loss ──────────────────────────────────────────────────────────
    age_loss: str = "l1"                        # l1 | mse | huber
    gender_loss_weight: float = 0.3             # Weight for gender CE loss
    age_loss_weight: float = 1.0                # Weight for age regression loss

    # ── Data ──────────────────────────────────────────────────────────
    img_size: int = 224                         # MiVOLO v2 input size
    val_split: float = 0.1                      # 10% validation
    num_workers: int = 2                        # Windows-safe default
    pin_memory: bool = True

    # ── Freezing strategy ─────────────────────────────────────────────
    freeze_backbone_epochs: int = 5             # Freeze backbone for first N epochs
    freeze_body_encoder: bool = True            # Freeze body encoder (not needed for face-only)
    unfreeze_last_n_blocks: int = 4             # Unfreeze last N transformer blocks

    # ── Device & Memory ───────────────────────────────────────────────
    device: str = "cuda"
    mixed_precision: bool = True
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0                  # Gradient clipping
