"""
SAM Evaluation Script
=====================
Evaluates SAM checkpoints by measuring:
  - L1 reconstruction error (self-reconstruction)
  - Identity preservation score (cosine similarity)
  - Visual comparison (saves sample grids)

Can compare two checkpoints side-by-side (e.g. FFHQ vs fine-tuned).

Usage:
    cd agevision_backend
    python -m agevision_api.sam.evaluate \\
        --checkpoint checkpoints/sam_indian_finetuned.pt \\
        --data_root /path/to/test_images \\
        --age_csv /path/to/test_ages.csv

    # Compare before/after:
    python -m agevision_api.sam.evaluate \\
        --checkpoint checkpoints/sam_ffhq_aging.pt \\
        --checkpoint_compare checkpoints/sam_indian_finetuned.pt \\
        --data_root /path/to/test_images \\
        --age_csv /path/to/test_ages.csv
"""

import argparse
import logging
import os

import cv2
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms

from .inference import SAMInference
from .dataset import IndianFaceDataset, VAL_TRANSFORM
from .datasets.augmentations import AgeTransformer

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('sam.evaluate')

TARGET_AGES = [10, 20, 30, 40, 50, 60, 70, 80]


def evaluate_checkpoint(checkpoint_path: str, data_root: str, age_csv: str,
                        num_samples: int = 20, output_dir: str = 'eval_output') -> dict:
    """
    Evaluate a SAM checkpoint on a test set.

    Returns dict with metrics:
        l1_mean, l1_std, identity_mean, identity_std, num_samples
    """
    logger.info("Evaluating: %s", checkpoint_path)

    sam = SAMInference(checkpoint_path=checkpoint_path)
    if not sam.load():
        logger.error("Failed to load checkpoint: %s", checkpoint_path)
        return {}

    dataset = IndianFaceDataset(data_root, age_csv, transform=VAL_TRANSFORM,
                                mode='self_reconstruct')

    # Limit samples
    indices = list(range(min(num_samples, len(dataset))))

    l1_errors = []
    identity_scores = []

    os.makedirs(output_dir, exist_ok=True)

    for idx in indices:
        sample = dataset[idx]
        input_tensor = sample['input'].unsqueeze(0)  # [1, 4, 256, 256]
        target_tensor = sample['target'].unsqueeze(0)  # [1, 3, 256, 256]
        real_age = sample['real_age']

        # Self-reconstruction: generate at same age
        with torch.no_grad():
            if sam.model is not None:
                device = next(sam.model.parameters()).device
                output = sam.model(input_tensor.to(device),
                                   randomize_noise=False, resize=True)
                output_cpu = output.cpu()

                # L1 error
                l1 = torch.nn.functional.l1_loss(output_cpu, target_tensor).item()
                l1_errors.append(l1)

                # Identity score (cosine similarity on flattened features)
                out_flat = output_cpu.view(1, -1)
                tgt_flat = target_tensor.view(1, -1)
                cos_sim = torch.nn.functional.cosine_similarity(out_flat, tgt_flat).item()
                identity_scores.append(cos_sim)

    metrics = {
        'checkpoint': os.path.basename(checkpoint_path),
        'num_samples': len(indices),
        'l1_mean': float(np.mean(l1_errors)) if l1_errors else 0,
        'l1_std': float(np.std(l1_errors)) if l1_errors else 0,
        'identity_mean': float(np.mean(identity_scores)) if identity_scores else 0,
        'identity_std': float(np.std(identity_scores)) if identity_scores else 0,
    }

    logger.info("Results for %s:", os.path.basename(checkpoint_path))
    logger.info("  L1 Error:   %.4f +/- %.4f", metrics['l1_mean'], metrics['l1_std'])
    logger.info("  Identity:   %.4f +/- %.4f", metrics['identity_mean'], metrics['identity_std'])

    return metrics


def generate_comparison_grid(checkpoint_path: str, image_path: str,
                             target_ages: list = None,
                             output_path: str = 'comparison_grid.jpg'):
    """
    Generate a visual comparison grid showing age progression results.

    Creates a single image with:
      Row 0: Original | Age 10 | Age 20 | ... | Age 80
    """
    if target_ages is None:
        target_ages = TARGET_AGES

    sam = SAMInference(checkpoint_path=checkpoint_path)
    if not sam.load():
        logger.error("Failed to load checkpoint")
        return

    # Load and process original image
    original_bgr = cv2.imread(image_path)
    if original_bgr is None:
        logger.error("Cannot read image: %s", image_path)
        return

    original_resized = cv2.resize(original_bgr, (256, 256))
    images = [original_resized]

    # Generate aged versions
    for age in target_ages:
        aged = sam.transform_face(original_bgr, target_age=age)
        if aged is not None:
            aged_resized = cv2.resize(aged, (256, 256))
            images.append(aged_resized)
        else:
            # Placeholder for failed generation
            placeholder = np.zeros((256, 256, 3), dtype=np.uint8)
            cv2.putText(placeholder, f'Age {age}', (70, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 128, 128), 2)
            images.append(placeholder)

    # Create grid
    grid = np.hstack(images)

    # Add labels
    labels = ['Original'] + [f'Age {a}' for a in target_ages]
    for i, label in enumerate(labels):
        x = i * 256 + 10
        cv2.putText(grid, label, (x, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imwrite(output_path, grid, [cv2.IMWRITE_JPEG_QUALITY, 95])
    logger.info("Comparison grid saved: %s", output_path)


def compare_checkpoints(ckpt_before: str, ckpt_after: str,
                        data_root: str, age_csv: str,
                        num_samples: int = 20) -> dict:
    """
    Compare two checkpoints side by side.

    Returns dict with 'before' and 'after' metrics.
    """
    logger.info("=" * 60)
    logger.info("COMPARING CHECKPOINTS")
    logger.info("=" * 60)

    before = evaluate_checkpoint(ckpt_before, data_root, age_csv,
                                 num_samples=num_samples,
                                 output_dir='eval_output/before')
    after = evaluate_checkpoint(ckpt_after, data_root, age_csv,
                                num_samples=num_samples,
                                output_dir='eval_output/after')

    logger.info("")
    logger.info("%-25s %-15s %-15s", "Metric", "Before", "After")
    logger.info("-" * 55)
    logger.info("%-25s %-15.4f %-15.4f", "L1 Error (mean)",
                before.get('l1_mean', 0), after.get('l1_mean', 0))
    logger.info("%-25s %-15.4f %-15.4f", "Identity Score (mean)",
                before.get('identity_mean', 0), after.get('identity_mean', 0))

    # Determine improvement
    l1_improved = after.get('l1_mean', 0) < before.get('l1_mean', 0)
    id_improved = after.get('identity_mean', 0) > before.get('identity_mean', 0)
    logger.info("")
    logger.info("L1 Error:   %s", "IMPROVED" if l1_improved else "DEGRADED")
    logger.info("Identity:   %s", "IMPROVED" if id_improved else "DEGRADED")

    return {'before': before, 'after': after}


def main():
    parser = argparse.ArgumentParser(description='Evaluate SAM checkpoints')
    parser.add_argument('--checkpoint', required=True, help='Checkpoint to evaluate')
    parser.add_argument('--checkpoint_compare', default=None,
                        help='Second checkpoint for comparison')
    parser.add_argument('--data_root', required=True, help='Test images directory')
    parser.add_argument('--age_csv', required=True, help='Test ages CSV')
    parser.add_argument('--num_samples', type=int, default=20, help='Number of samples')
    parser.add_argument('--grid_image', default=None,
                        help='Single image for visual comparison grid')
    args = parser.parse_args()

    if args.checkpoint_compare:
        compare_checkpoints(
            args.checkpoint, args.checkpoint_compare,
            args.data_root, args.age_csv,
            num_samples=args.num_samples,
        )
    else:
        evaluate_checkpoint(
            args.checkpoint, args.data_root, args.age_csv,
            num_samples=args.num_samples,
        )

    if args.grid_image:
        generate_comparison_grid(args.checkpoint, args.grid_image)
        if args.checkpoint_compare:
            generate_comparison_grid(
                args.checkpoint_compare, args.grid_image,
                output_path='comparison_grid_finetuned.jpg',
            )


if __name__ == '__main__':
    main()
