"""
MiVOLO v2 Fine-tuning on Indian Faces
=======================================
Fine-tunes the pretrained MiVOLO v2 (iitolstykh/mivolo_v2) on a combined
IFAD + UTKFace-Indian dataset for improved age estimation on Indian faces.

Works on both GPU (CUDA) and CPU. CPU mode automatically:
  - Disables mixed precision
  - Uses smaller batch size
  - Skips GradScaler

Strategy:
  1. Load pretrained MiVOLO v2 from HuggingFace Hub
  2. Freeze backbone for N warmup epochs (train only head)
  3. Gradually unfreeze last N transformer blocks
  4. Train with L1/Huber age loss + optional gender CE loss
  5. Save best checkpoint based on validation MAE

Usage (CPU):
    cd agevision_backend
    python -m agevision_api.mivolo.train \
        --data-root datasets/indian_faces_mivolo/images \
        --labels-csv datasets/indian_faces_mivolo/labels.csv \
        --num-epochs 15 --batch-size 4 --device cpu

Usage (GPU):
    python -m agevision_api.mivolo.train \
        --data-root datasets/indian_faces_mivolo/images \
        --labels-csv datasets/indian_faces_mivolo/labels.csv \
        --num-epochs 30 --batch-size 16 --lr 2e-5
"""

import argparse
import logging
import os
import time

import torch
import torch.nn as nn

logger = logging.getLogger("agevision.mivolo.train")


def train(config):
    """Main training loop. Works on both GPU and CPU."""
    from transformers import (
        AutoModelForImageClassification,
        AutoConfig,
        AutoImageProcessor,
    )
    from .dataset import create_data_loaders

    # ── Device selection ──────────────────────────────────────────────
    if config.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        config.device = "cpu"

    device = torch.device(config.device)
    use_amp = config.mixed_precision and device.type == "cuda"

    if device.type == "cpu":
        logger.info("Running on CPU — this will be slower but works fine")
        logger.info("Tip: reduce --num-epochs and --batch-size for faster iteration")

    logger.info("Device: %s | Mixed precision: %s", device, use_amp)

    # ── Load model ────────────────────────────────────────────────────
    logger.info("Loading MiVOLO v2 from %s ...", config.model_name)
    model_config = AutoConfig.from_pretrained(
        config.model_name, trust_remote_code=True
    )
    model = AutoModelForImageClassification.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        dtype=torch.float32,
    )
    model = model.to(device)

    processor = AutoImageProcessor.from_pretrained(
        config.model_name, trust_remote_code=True
    )

    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Model loaded: %s parameters", f"{total_params:,}")

    # ── Dataset ───────────────────────────────────────────────────────
    logger.info("Loading dataset from %s ...", config.labels_csv)
    train_loader, val_loader = create_data_loaders(config, processor)
    logger.info("Train: %d batches, Val: %d batches",
                len(train_loader), len(val_loader))

    # ── Freeze strategy ───────────────────────────────────────────────
    _apply_freeze(model, config, phase="warmup")

    # ── Loss functions ────────────────────────────────────────────────
    if config.age_loss == "l1":
        age_criterion = nn.L1Loss()
    elif config.age_loss == "huber":
        age_criterion = nn.SmoothL1Loss(beta=2.0)
    else:
        age_criterion = nn.MSELoss()

    gender_criterion = nn.CrossEntropyLoss(ignore_index=-1)

    # ── Optimizer & scheduler ─────────────────────────────────────────
    optimizer = _build_optimizer(model, config)
    total_steps = len(train_loader) * config.num_epochs
    scheduler = _build_scheduler(optimizer, config, total_steps)

    # GradScaler only for CUDA + mixed precision
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp) if use_amp else None

    # ── Training state ────────────────────────────────────────────────
    os.makedirs(config.output_dir, exist_ok=True)
    best_val_mae = float("inf")
    patience_counter = 0

    logger.info("=" * 60)
    logger.info("  Starting training: %d epochs, batch=%d, lr=%s",
                config.num_epochs, config.batch_size, config.learning_rate)
    logger.info("  Freeze backbone: %d epochs, then unfreeze last %d blocks",
                config.freeze_backbone_epochs, config.unfreeze_last_n_blocks)
    logger.info("=" * 60)

    for epoch in range(1, config.num_epochs + 1):
        epoch_start = time.time()

        # Unfreeze backbone after warmup
        if epoch == config.freeze_backbone_epochs + 1:
            logger.info("Epoch %d: Unfreezing last %d transformer blocks",
                        epoch, config.unfreeze_last_n_blocks)
            _apply_freeze(model, config, phase="finetune")
            optimizer = _build_optimizer(model, config)
            remaining_steps = len(train_loader) * (config.num_epochs - epoch + 1)
            scheduler = _build_scheduler(optimizer, config, remaining_steps)

        # Train
        train_metrics = _train_epoch(
            model, train_loader, age_criterion, gender_criterion,
            optimizer, scheduler, scaler, config, device, epoch, use_amp,
        )

        # Validate
        val_metrics = _validate(
            model, val_loader, age_criterion, gender_criterion,
            config, device,
        )

        epoch_time = time.time() - epoch_start

        # Estimate remaining time
        remaining = (config.num_epochs - epoch) * epoch_time
        eta_min = remaining / 60

        improved = ""
        if val_metrics["mae"] < best_val_mae:
            best_val_mae = val_metrics["mae"]
            patience_counter = 0
            _save_checkpoint(model, processor, model_config, config,
                             epoch, val_metrics, is_best=True)
            improved = " * BEST (saved)"
        else:
            patience_counter += 1

        logger.info(
            "Epoch %d/%d [%.0fs, ETA %.0fm] | Train MAE: %.2f, Loss: %.4f | "
            "Val MAE: %.2f, Loss: %.4f%s",
            epoch, config.num_epochs, epoch_time, eta_min,
            train_metrics["mae"], train_metrics["loss"],
            val_metrics["mae"], val_metrics["loss"],
            improved,
        )

        # Periodic save
        if epoch % config.save_every == 0:
            _save_checkpoint(model, processor, model_config, config,
                             epoch, val_metrics, is_best=False)

        # Early stopping
        if (config.early_stopping_patience > 0 and
                patience_counter >= config.early_stopping_patience):
            logger.info("Early stopping at epoch %d (patience=%d)",
                        epoch, config.early_stopping_patience)
            break

    logger.info("=" * 60)
    logger.info("  Training complete. Best Val MAE: %.2f", best_val_mae)
    logger.info("  Checkpoint: %s",
                os.path.join(config.output_dir, "mivolo_indian_best.pt"))
    logger.info("=" * 60)
    return best_val_mae


# ══════════════════════════════════════════════════════════════════════════
#  Train / Validate one epoch
# ══════════════════════════════════════════════════════════════════════════

def _train_epoch(model, loader, age_criterion, gender_criterion,
                 optimizer, scheduler, scaler, config, device, epoch, use_amp):
    model.train()
    total_loss = 0.0
    total_mae = 0.0
    num_samples = 0
    accum_steps = config.gradient_accumulation_steps
    num_batches = len(loader)

    optimizer.zero_grad()

    for step, batch in enumerate(loader, 1):
        face_input = batch["face_pixel_values"].to(device)
        body_input = batch["body_pixel_values"].to(device)
        age_target = batch["age"].to(device)
        gender_target = batch["gender"].to(device)

        # Forward pass (with or without AMP)
        if use_amp:
            with torch.amp.autocast('cuda'):
                output = model(faces_input=face_input, body_input=body_input)
                age_pred = output.age_output.squeeze(-1)
                loss = _compute_loss(age_pred, age_target, output, gender_target,
                                     age_criterion, gender_criterion, config)
                loss = loss / accum_steps
            scaler.scale(loss).backward()
        else:
            output = model(faces_input=face_input, body_input=body_input)
            age_pred = output.age_output.squeeze(-1)
            loss = _compute_loss(age_pred, age_target, output, gender_target,
                                 age_criterion, gender_criterion, config)
            loss = loss / accum_steps
            loss.backward()

        # Optimizer step
        if step % accum_steps == 0:
            if use_amp:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()

        bs = age_target.size(0)
        total_loss += loss.item() * accum_steps * bs
        total_mae += (age_pred.detach() - age_target).abs().sum().item()
        num_samples += bs

        if step % config.log_every == 0:
            running_mae = total_mae / num_samples
            running_loss = total_loss / num_samples
            pct = step / num_batches * 100
            logger.info("  [Epoch %d, %d/%d (%.0f%%)] Loss: %.4f, MAE: %.2f",
                        epoch, step, num_batches, pct, running_loss, running_mae)

    return {
        "loss": total_loss / max(num_samples, 1),
        "mae": total_mae / max(num_samples, 1),
    }


def _compute_loss(age_pred, age_target, output, gender_target,
                  age_criterion, gender_criterion, config):
    """Compute combined age + gender loss."""
    loss_age = age_criterion(age_pred, age_target) * config.age_loss_weight

    loss_gender = torch.tensor(0.0, device=age_pred.device)
    if config.gender_loss_weight > 0 and hasattr(output, "gender_output"):
        valid_gender = gender_target != -1
        if valid_gender.any():
            loss_gender = gender_criterion(
                output.gender_output[valid_gender],
                gender_target[valid_gender],
            ) * config.gender_loss_weight

    return loss_age + loss_gender


@torch.no_grad()
def _validate(model, loader, age_criterion, gender_criterion, config, device):
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    num_samples = 0

    for batch in loader:
        face_input = batch["face_pixel_values"].to(device)
        body_input = batch["body_pixel_values"].to(device)
        age_target = batch["age"].to(device)
        gender_target = batch["gender"].to(device)

        output = model(faces_input=face_input, body_input=body_input)
        age_pred = output.age_output.squeeze(-1)

        loss = _compute_loss(age_pred, age_target, output, gender_target,
                             age_criterion, gender_criterion, config)

        bs = age_target.size(0)
        total_loss += loss.item() * bs
        total_mae += (age_pred - age_target).abs().sum().item()
        num_samples += bs

    return {
        "loss": total_loss / max(num_samples, 1),
        "mae": total_mae / max(num_samples, 1),
    }


# ══════════════════════════════════════════════════════════════════════════
#  Freeze / Unfreeze
# ══════════════════════════════════════════════════════════════════════════

def _apply_freeze(model, config, phase: str):
    for param in model.parameters():
        param.requires_grad = False

    if phase == "warmup":
        _unfreeze_heads(model)
    else:
        _unfreeze_heads(model)
        _unfreeze_last_blocks(model, config.unfreeze_last_n_blocks)
        if config.freeze_body_encoder:
            _freeze_body_encoder(model)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info("  Trainable params: %s / %s (%.1f%%)",
                f"{trainable:,}", f"{total:,}", 100 * trainable / max(total, 1))


def _unfreeze_heads(model):
    for name, param in model.named_parameters():
        if any(k in name.lower() for k in ("head", "classifier", "age", "gender",
                                             "fc", "output")):
            param.requires_grad = True


def _unfreeze_last_blocks(model, n_blocks: int):
    block_params = []
    for name, param in model.named_parameters():
        if "body" in name.lower():
            continue
        for pattern in ("blocks.", "layers.", "encoder.layer."):
            if pattern in name:
                try:
                    idx_str = name.split(pattern)[1].split(".")[0]
                    block_idx = int(idx_str)
                    block_params.append((block_idx, name, param))
                except (ValueError, IndexError):
                    pass
                break

    if not block_params:
        logger.warning("Could not identify transformer blocks. "
                       "Unfreezing all face encoder parameters.")
        for name, param in model.named_parameters():
            if "body" not in name.lower():
                param.requires_grad = True
        return

    max_idx = max(bp[0] for bp in block_params)
    threshold = max_idx - n_blocks + 1
    for block_idx, name, param in block_params:
        if block_idx >= threshold:
            param.requires_grad = True


def _freeze_body_encoder(model):
    for name, param in model.named_parameters():
        if "body" in name.lower():
            param.requires_grad = False


# ══════════════════════════════════════════════════════════════════════════
#  Optimizer & Scheduler
# ══════════════════════════════════════════════════════════════════════════

def _build_optimizer(model, config):
    head_params = []
    backbone_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(k in name.lower() for k in ("head", "classifier", "age", "gender",
                                             "fc", "output")):
            head_params.append(param)
        else:
            backbone_params.append(param)

    param_groups = []
    if head_params:
        param_groups.append({"params": head_params, "lr": config.learning_rate * 5})
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": config.learning_rate})
    if not param_groups:
        param_groups = [{"params": model.parameters(), "lr": config.learning_rate}]

    return torch.optim.AdamW(param_groups, weight_decay=config.weight_decay)


def _build_scheduler(optimizer, config, total_steps: int):
    warmup_steps = int(total_steps * config.warmup_ratio)
    if config.lr_scheduler == "cosine":
        from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR
        warmup = LinearLR(optimizer, start_factor=0.1, total_iters=max(warmup_steps, 1))
        cosine = CosineAnnealingLR(optimizer, T_max=max(total_steps - warmup_steps, 1))
        return SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_steps])
    elif config.lr_scheduler == "linear":
        from torch.optim.lr_scheduler import LinearLR
        return LinearLR(optimizer, start_factor=1.0, end_factor=0.01,
                        total_iters=max(total_steps, 1))
    return None


# ══════════════════════════════════════════════════════════════════════════
#  Checkpoint
# ══════════════════════════════════════════════════════════════════════════

def _save_checkpoint(model, processor, model_config, config,
                     epoch, metrics, is_best=False):
    save_dict = {
        "model_state_dict": model.state_dict(),
        "config": {
            "model_name": config.model_name,
            "output_dir": config.output_dir,
            "age_loss": config.age_loss,
            "img_size": config.img_size,
        },
        "epoch": epoch,
        "val_mae": metrics["mae"],
        "val_loss": metrics["loss"],
    }
    if is_best:
        path = os.path.join(config.output_dir, "mivolo_indian_best.pt")
    else:
        path = os.path.join(config.output_dir, f"mivolo_indian_epoch{epoch}.pt")
    torch.save(save_dict, path)
    logger.info("Checkpoint saved: %s (MAE: %.2f)", path, metrics["mae"])


# ══════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════

def main():
    from .train_config import MiVOLOTrainConfig

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Fine-tune MiVOLO v2 on Indian face dataset"
    )
    parser.add_argument("--data-root", required=True,
                        help="Path to images directory")
    parser.add_argument("--labels-csv", required=True,
                        help="Path to labels CSV file")
    parser.add_argument("--output-dir", default="checkpoints/mivolo_indian",
                        help="Output checkpoint directory")
    parser.add_argument("--num-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Base learning rate")
    parser.add_argument("--age-loss", choices=["l1", "mse", "huber"], default="l1")
    parser.add_argument("--freeze-epochs", type=int, default=5,
                        help="Epochs to freeze backbone (head-only warmup)")
    parser.add_argument("--unfreeze-blocks", type=int, default=4,
                        help="Number of last transformer blocks to unfreeze")
    parser.add_argument("--device", default="auto",
                        help="Device: cuda, cpu, or auto (default)")
    parser.add_argument("--patience", type=int, default=7,
                        help="Early stopping patience (0 to disable)")

    args = parser.parse_args()

    # Auto-detect device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # CPU-optimized defaults
    if device == "cpu":
        batch_size = min(args.batch_size, 4)
        mixed_precision = False
        num_epochs = min(args.num_epochs, 15)
        num_workers = 0        # Windows multiprocessing safety
        pin_memory = False     # No CUDA = no point pinning
        logger.info("CPU mode: batch_size=%d, epochs=%d, no mixed precision",
                     batch_size, num_epochs)
    else:
        batch_size = args.batch_size
        mixed_precision = True
        num_epochs = args.num_epochs
        num_workers = 2
        pin_memory = True

    config = MiVOLOTrainConfig(
        data_root=args.data_root,
        labels_csv=args.labels_csv,
        output_dir=args.output_dir,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=args.lr,
        age_loss=args.age_loss,
        freeze_backbone_epochs=args.freeze_epochs,
        unfreeze_last_n_blocks=args.unfreeze_blocks,
        mixed_precision=mixed_precision,
        device=device,
        early_stopping_patience=args.patience,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    train(config)


if __name__ == "__main__":
    main()
