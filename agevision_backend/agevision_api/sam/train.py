"""
SAM Fine-tuning Script
======================
Fine-tunes SAM (pSp encoder + StyleGAN2 decoder) on Indian face data.

The script loads a pretrained checkpoint (e.g. sam_ffhq_aging.pt) and
fine-tunes it with separate learning rates for encoder and decoder.

Usage:
    cd agevision_backend
    python -m agevision_api.sam.train \\
        --data_root /path/to/indian_faces \\
        --age_csv /path/to/ages.csv \\
        --checkpoint checkpoints/sam_ffhq_aging.pt \\
        --epochs 50 \\
        --batch_size 4

CSV format:
    filename,age
    img001.jpg,25
    img002.jpg,42
"""

import argparse
import logging
import os
import sys
import time
from argparse import Namespace

import torch
from torch.amp import GradScaler, autocast
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from .train_config import SAMTrainConfig
from .dataset import create_data_loaders
from .losses import SAMTrainingLoss
from .models.psp import pSp
from .models.stylegan2.model import Discriminator
from .configs.paths_config import model_paths

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger('sam.train')


def build_opts(ckpt_path: str, config: SAMTrainConfig) -> Namespace:
    """Build opts Namespace from checkpoint + training config."""
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    # Extract opts from checkpoint
    if 'opts' in ckpt:
        opts_dict = ckpt['opts']
        if isinstance(opts_dict, Namespace):
            opts_dict = vars(opts_dict)
    else:
        opts_dict = {}

    # Override with training config
    opts_dict['checkpoint_path'] = ckpt_path
    opts_dict['device'] = config.device
    opts_dict['input_nc'] = config.input_nc
    opts_dict['output_size'] = config.output_size
    opts_dict['start_from_latent_avg'] = config.start_from_latent_avg
    opts_dict['start_from_encoded_w_plus'] = config.start_from_encoded_w_plus

    return Namespace(**opts_dict)


def forward_with_checkpointing(net: pSp, input_4ch: torch.Tensor,
                               use_checkpointing: bool = True) -> torch.Tensor:
    """Forward through SAM with optional gradient checkpointing on decoder.

    Gradient checkpointing discards decoder intermediate activations during
    forward and recomputes them during backward. This saves ~10 GB VRAM at the
    cost of ~1.3x compute time — makes T4 (15 GB) training possible.
    """
    # 1. Encoder → latent codes
    codes = net.encoder(input_4ch)

    # 2. Add latent offset
    if getattr(net.opts, 'start_from_latent_avg', False):
        codes = codes + net.latent_avg
    elif getattr(net.opts, 'start_from_encoded_w_plus', False):
        with torch.no_grad():
            encoded_latents = net.pretrained_encoder(input_4ch[:, :-1, :, :])
            encoded_latents = encoded_latents + net.latent_avg
        codes = codes + encoded_latents

    # 3. Decoder (optionally checkpointed)
    def _decoder_forward(codes_inner):
        images, _ = net.decoder([codes_inner],
                                input_is_latent=True,
                                randomize_noise=False,
                                return_latents=False)
        return images

    if use_checkpointing:
        images = grad_checkpoint(_decoder_forward, codes, use_reentrant=False)
    else:
        images = _decoder_forward(codes)

    # 4. Downscale 1024 → 256
    images = net.face_pool(images)
    return images


def save_checkpoint(net: pSp, opts: Namespace, discriminator: Discriminator,
                    epoch: int, config: SAMTrainConfig):
    """Save checkpoint in SAM-compatible format."""
    os.makedirs(config.output_dir, exist_ok=True)

    save_dict = {
        'state_dict': net.state_dict(),
        'opts': vars(opts),
        'discriminator': discriminator.state_dict(),
        'epoch': epoch,
    }
    if hasattr(net, 'latent_avg') and net.latent_avg is not None:
        save_dict['latent_avg'] = net.latent_avg

    # Save only the latest checkpoint (no epoch-specific file to save storage)
    best_path = os.path.join(config.output_dir, 'sam_indian_finetuned.pt')
    torch.save(save_dict, best_path)
    logger.info("Checkpoint saved (epoch %d): %s", epoch, best_path)


def train(config: SAMTrainConfig):
    """Main training loop."""
    import gc

    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    logger.info("Device: %s", device)
    logger.info("Config: batch_size=%d, epochs=%d, lr_enc=%.1e, lr_dec=%.1e",
                config.batch_size, config.num_epochs, config.lr_encoder, config.lr_decoder)

    # ── 1. Load pretrained SAM ────────────────────────────────────────
    # Prefer Indian checkpoint for continued fine-tuning, fall back to FFHQ
    ckpt_path = config.checkpoint_path or ''
    if not ckpt_path or not os.path.isfile(ckpt_path):
        indian_path = model_paths.get('sam_indian', '')
        ffhq_path = model_paths.get('sam_ffhq_aging', '')
        if os.path.isfile(indian_path):
            ckpt_path = indian_path
            logger.info("Using Indian checkpoint as base: %s", ckpt_path)
        elif os.path.isfile(ffhq_path):
            ckpt_path = ffhq_path
            logger.info("Using FFHQ checkpoint as base: %s", ckpt_path)
    if not os.path.isfile(ckpt_path):
        logger.error("Checkpoint not found: %s", ckpt_path)
        sys.exit(1)

    opts = build_opts(ckpt_path, config)
    opts.device = str(device)
    net = pSp(opts).to(device)
    net.train()

    # Freeze pretrained encoder (identity reference, not trainable)
    if hasattr(net, 'pretrained_encoder'):
        for p in net.pretrained_encoder.parameters():
            p.requires_grad = False
        net.pretrained_encoder.eval()
        logger.info("Pretrained encoder frozen (identity reference)")

    # Freeze decoder to save VRAM — encoder-only fine-tuning
    # (standard approach for SAM: the decoder already generates good faces,
    #  we just need the encoder to produce better latent codes for Indian faces)
    freeze_decoder = getattr(config, 'freeze_decoder', True)
    if freeze_decoder:
        for p in net.decoder.parameters():
            p.requires_grad = False
        net.decoder.eval()
        logger.info("Decoder frozen (encoder-only fine-tuning to save VRAM)")
        trainable_params = list(net.encoder.parameters())
    else:
        trainable_params = list(net.encoder.parameters()) + list(net.decoder.parameters())

    # ── 2. Discriminator ──────────────────────────────────────────────
    discriminator = Discriminator(256).to(device)  # 256x256 after face_pool
    logger.info("Discriminator initialized (input 256x256)")

    # ── 3. Optimizers ─────────────────────────────────────────────────
    if freeze_decoder:
        opt_gen = torch.optim.Adam(
            trainable_params, lr=config.lr_encoder,
            weight_decay=config.weight_decay,
        )
    else:
        encoder_params = list(net.encoder.parameters())
        decoder_params = list(net.decoder.parameters())
        opt_gen = torch.optim.Adam([
            {'params': encoder_params, 'lr': config.lr_encoder},
            {'params': decoder_params, 'lr': config.lr_decoder},
        ], weight_decay=config.weight_decay)

    opt_disc = torch.optim.Adam(
        discriminator.parameters(),
        lr=config.lr_discriminator,
        weight_decay=config.weight_decay,
    )

    # ── 4. Data ───────────────────────────────────────────────────────
    logger.info("Loading dataset from %s", config.data_root)
    train_loader, val_loader = create_data_loaders(config)
    logger.info("Train: %d batches, Val: %d batches",
                len(train_loader), len(val_loader))

    # ── 5. Losses ─────────────────────────────────────────────────────
    criterion = SAMTrainingLoss(config).to(device)

    # Set identity encoder if available
    if hasattr(net, 'pretrained_encoder'):
        criterion.set_identity_encoder(net.pretrained_encoder)
        logger.info("Identity loss: using pretrained pSp encoder")
    else:
        logger.info("Identity loss: using L1 fallback (no pretrained encoder)")

    # ── 6. Mixed precision ────────────────────────────────────────────
    use_amp = config.mixed_precision and device.type == 'cuda'
    scaler = GradScaler(device=device.type, enabled=use_amp)

    # Gradient accumulation for effective larger batch on limited VRAM
    accum_steps = getattr(config, 'gradient_accumulation_steps', 1)
    use_ckpt = getattr(config, 'gradient_checkpointing', True)
    logger.info("Mixed precision: %s, Gradient accumulation: %d steps, Grad checkpointing: %s",
                use_amp, accum_steps, use_ckpt)

    # Memory info
    if device.type == 'cuda':
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info("GPU memory: %.1f GB", total_mem)

    # ── 7. Training loop ──────────────────────────────────────────────
    logger.info("Starting training...")
    for epoch in range(1, config.num_epochs + 1):
        net.encoder.train()
        if not freeze_decoder:
            net.decoder.train()
        discriminator.train()
        epoch_losses = {'gen_total': 0, 'disc': 0, 'l1': 0, 'identity': 0, 'adv': 0}
        batch_count = 0

        for batch_idx, batch in enumerate(train_loader):
            input_4ch = batch['input'].to(device)     # [B, 4, 256, 256]
            target = batch['target'].to(device)        # [B, 3, 256, 256]
            real_age = batch['real_age'].to(device)    # [B]

            # ──── Generator step ──────────────────────────────────────
            with autocast(device_type=device.type, enabled=use_amp):
                generated = forward_with_checkpointing(
                    net, input_4ch, use_checkpointing=use_ckpt)
                disc_fake_for_gen = discriminator(generated)
                gen_loss, loss_dict = criterion(
                    generated, target,
                    target_ages=real_age,
                    disc_fake=disc_fake_for_gen,
                )
                gen_loss = gen_loss / accum_steps

            scaler.scale(gen_loss).backward()

            if (batch_idx + 1) % accum_steps == 0:
                scaler.step(opt_gen)
                opt_gen.zero_grad(set_to_none=True)

            # ──── Discriminator step ──────────────────────────────────
            with autocast(device_type=device.type, enabled=use_amp):
                disc_real = discriminator(target)
                disc_fake = discriminator(generated.detach())
                disc_loss = criterion.discriminator_loss(disc_real, disc_fake)
                disc_loss = disc_loss / accum_steps

            scaler.scale(disc_loss).backward()

            if (batch_idx + 1) % accum_steps == 0:
                scaler.step(opt_disc)
                opt_disc.zero_grad(set_to_none=True)
                scaler.update()

            # Free intermediate tensors
            del generated, disc_fake_for_gen, disc_real, disc_fake
            if device.type == 'cuda':
                torch.cuda.empty_cache()

            # Accumulate losses
            epoch_losses['gen_total'] += loss_dict.get('total', 0)
            epoch_losses['disc'] += disc_loss.item()
            epoch_losses['l1'] += loss_dict.get('l1', 0)
            epoch_losses['identity'] += loss_dict.get('identity', 0)
            epoch_losses['adv'] += loss_dict.get('adv', 0)
            batch_count += 1

            # Per-batch logging
            if (batch_idx + 1) % config.log_every == 0:
                logger.info(
                    "Epoch %d/%d [%d/%d] | Gen: %.4f | Disc: %.4f | L1: %.4f | ID: %.4f",
                    epoch, config.num_epochs, batch_idx + 1, len(train_loader),
                    loss_dict.get('total', 0), disc_loss.item(),
                    loss_dict.get('l1', 0), loss_dict.get('identity', 0),
                )

        # Epoch summary
        if batch_count > 0:
            avg = {k: v / batch_count for k, v in epoch_losses.items()}
            logger.info(
                "Epoch %d/%d | Gen: %.4f | Disc: %.4f | L1: %.4f | ID: %.4f | Adv: %.4f",
                epoch, config.num_epochs,
                avg['gen_total'], avg['disc'], avg['l1'],
                avg['identity'], avg['adv'],
            )

        # Validation
        if val_loader and epoch % config.save_every == 0:
            val_loss = validate(net, val_loader, criterion, device, config)
            logger.info("Epoch %d | Val Loss: %.4f", epoch, val_loss)

        # Save checkpoint
        if epoch % config.save_every == 0 or epoch == config.num_epochs:
            save_checkpoint(net, opts, discriminator, epoch, config)

    logger.info("Training complete!")


def validate(net: pSp, val_loader, criterion: SAMTrainingLoss,
             device: torch.device, config: SAMTrainConfig) -> float:
    """Run validation and return average loss."""
    net.eval()
    total_loss = 0
    count = 0

    with torch.no_grad():
        for batch in val_loader:
            input_4ch = batch['input'].to(device)
            target = batch['target'].to(device)
            generated = forward_with_checkpointing(
                net, input_4ch, use_checkpointing=False)
            loss, _ = criterion(generated, target)
            total_loss += loss.item()
            count += 1

    net.train()
    return total_loss / max(count, 1)


def main():
    parser = argparse.ArgumentParser(description='Fine-tune SAM on Indian face data')
    parser.add_argument('--data_root', required=True, help='Directory with face images')
    parser.add_argument('--age_csv', required=True, help='CSV with filename,age columns')
    parser.add_argument('--checkpoint', default=None, help='Source checkpoint path')
    parser.add_argument('--output_dir', default='checkpoints/sam_finetune', help='Output directory')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr_encoder', type=float, default=1e-4, help='Encoder learning rate')
    parser.add_argument('--lr_decoder', type=float, default=1e-5, help='Decoder learning rate')
    parser.add_argument('--device', default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--save_every', type=int, default=5, help='Save checkpoint every N epochs')
    parser.add_argument('--no_mixed_precision', action='store_true', help='Disable mixed precision')
    args = parser.parse_args()

    config = SAMTrainConfig(
        data_root=args.data_root,
        age_csv=args.age_csv,
        checkpoint_path=args.checkpoint or '',
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr_encoder=args.lr_encoder,
        lr_decoder=args.lr_decoder,
        device=args.device,
        save_every=args.save_every,
        mixed_precision=not args.no_mixed_precision,
    )

    train(config)


if __name__ == '__main__':
    main()
