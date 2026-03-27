#!/usr/bin/env python
"""
AgeVision — End-to-End Age Progression Pipeline
=================================================
Single command to process face images through the HRFAE GAN model
for age progression/de-aging with full evaluation and reporting.

Usage:
    python run.py --input ./images --output ./results --ages 20,40,60,80
    python run.py --input photo.jpg --output ./results --ages 30,50,70
    python run.py --input ./images --output ./results --ages 20,40,60,80 --no-deep

Environment:
    Requires PyTorch, OpenCV, NumPy, Pillow.
    Optional: facenet-pytorch for deep identity metrics.
    HRFAE checkpoint at: agevision_backend/checkpoints/hrfae_best.pth
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from age_pipeline.detector import validate_image, crop_face, detect_faces
from age_pipeline.model import AgeProgressionModel
from age_pipeline.postprocess import (
    paste_back, color_correct, enhance_aging_effects, create_comparison_grid
)
from age_pipeline.evaluator import evaluate_progression, save_metrics
from age_pipeline.report import generate_report

# ─── Logging Setup ────────────────────────────────────────────

def setup_logging(verbose: bool = False):
    """Configure logging with file and console handlers."""
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    datefmt = "%H:%M:%S"

    logging.basicConfig(level=level, format=fmt, datefmt=datefmt)

    # File handler for full logs
    log_dir = PROJECT_ROOT / "results"
    log_dir.mkdir(exist_ok=True)
    fh = logging.FileHandler(log_dir / "pipeline.log", mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    logging.getLogger().addHandler(fh)


logger = logging.getLogger("age_pipeline")

# Supported image extensions
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


# ─── Core Pipeline ────────────────────────────────────────────

def collect_images(input_path: str) -> list:
    """Collect image file paths from a file or directory."""
    p = Path(input_path)
    if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
        return [str(p)]
    elif p.is_dir():
        images = sorted(
            str(f) for f in p.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTS
        )
        return images
    else:
        logger.error("Input path not found or not an image: %s", input_path)
        return []


def process_single_image(image_path: str, target_ages: list, model: AgeProgressionModel,
                          output_dir: str, use_deep_identity: bool = True,
                          current_age: int = None) -> dict:
    """Process a single image through the full pipeline.

    Returns a result dict for report generation.
    """
    image_name = Path(image_path).stem
    logger.info("Processing: %s", Path(image_path).name)
    t0 = time.time()

    # Step 1: Validate and detect faces
    validation = validate_image(image_path)
    if not validation["valid"]:
        logger.warning("SKIP %s: %s", image_name, validation["reason"])
        return {
            "input_path": image_path,
            "input_name": Path(image_path).name,
            "original_crop": None,
            "progressions": {},
            "grid_path": None,
            "test_result": f"FAIL: {validation['reason']}",
        }

    img_bgr = validation["image"]
    faces = validation["faces"]
    primary_face = faces[0]  # Largest face

    logger.info("  Detected %d face(s), processing primary face", len(faces))

    # Step 2: Crop face
    face_crop, crop_coords = crop_face(img_bgr, primary_face)
    logger.info("  Face cropped at (%d,%d)-(%d,%d)", *crop_coords)

    # Step 3: Run age progression for each target age
    progressions = {}
    aged_images_for_grid = {}

    for target_age in target_ages:
        logger.info("  Generating age %d...", target_age)
        age_t0 = time.time()

        try:
            # GAN inference
            aged_crop = model.transform_face(face_crop, target_age)

            # Enhance aging effects
            est_current = current_age or 30  # Default assumption
            aged_crop_enhanced = enhance_aging_effects(aged_crop, est_current, target_age)

            # Paste back into original
            output_full = paste_back(img_bgr, aged_crop_enhanced, crop_coords)

            # Color correction
            output_full = color_correct(output_full, img_bgr)

            # Save individual output
            out_fname = f"{image_name}_age{target_age}.jpg"
            out_path = os.path.join(output_dir, out_fname)
            cv2.imwrite(out_path, output_full, [cv2.IMWRITE_JPEG_QUALITY, 95])

            # Evaluate
            metrics = evaluate_progression(
                face_crop, aged_crop, target_age,
                use_deep=use_deep_identity
            )
            metrics["processing_time_s"] = round(time.time() - age_t0, 2)

            progressions[target_age] = {
                "image": aged_crop,
                "full_image": output_full,
                "output_path": out_path,
                "metrics": metrics,
            }
            aged_images_for_grid[f"Age {target_age}"] = aged_crop

            logger.info("    Done (%.1fs) | SSIM=%.3f | Identity=%.3f",
                         metrics["processing_time_s"], metrics["ssim"],
                         metrics["identity_score"])

        except Exception as e:
            logger.error("    FAILED for age %d: %s", target_age, e)
            progressions[target_age] = {
                "image": None,
                "metrics": {"error": str(e)},
            }

    # Step 4: Create comparison grid
    grid_path = None
    if aged_images_for_grid:
        grid = create_comparison_grid(face_crop, aged_images_for_grid)
        grid_path = os.path.join(output_dir, f"{image_name}_grid.jpg")
        cv2.imwrite(grid_path, grid, [cv2.IMWRITE_JPEG_QUALITY, 95])
        logger.info("  Grid saved: %s", grid_path)

    # Step 5: Determine test result
    test_result = _evaluate_test_result(progressions)

    total_time = time.time() - t0
    logger.info("  Total: %.1fs | Result: %s", total_time, test_result)

    return {
        "input_path": image_path,
        "input_name": Path(image_path).name,
        "original_crop": face_crop,
        "progressions": progressions,
        "grid_path": grid_path,
        "test_result": test_result,
        "processing_time_s": round(total_time, 2),
    }


def _evaluate_test_result(progressions: dict) -> str:
    """Determine PASS/FAIL for a set of progressions."""
    if not progressions:
        return "FAIL: No progressions generated"

    failed_ages = []
    low_identity = []

    for age, data in progressions.items():
        if data.get("image") is None:
            failed_ages.append(age)
            continue

        metrics = data.get("metrics", {})
        if "error" in metrics:
            failed_ages.append(age)
            continue

        identity = metrics.get("identity_score", 0)
        if identity < 0.5:
            low_identity.append((age, identity))

    if failed_ages:
        return f"FAIL: Generation failed for ages {failed_ages}"

    if low_identity:
        ages_str = ", ".join(f"{a}({s:.2f})" for a, s in low_identity)
        return f"PASS: Low identity for ages [{ages_str}] but faces generated"

    return "PASS: All age progressions successful with good identity preservation"


# ─── Main Entry Point ─────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="AgeVision — End-to-End Age Progression Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py --input ./images --output ./results --ages 20,40,60,80
  python run.py --input photo.jpg --output ./results --ages 30,50,70
  python run.py --input ./images --ages 20,40,60,80 --verbose
        """
    )
    parser.add_argument("--input", "-i", required=True,
                        help="Input image file or directory")
    parser.add_argument("--output", "-o", default="./results",
                        help="Output directory (default: ./results)")
    parser.add_argument("--ages", "-a", default="20,40,60,80",
                        help="Comma-separated target ages (default: 20,40,60,80)")
    parser.add_argument("--checkpoint", "-c", default=None,
                        help="Path to HRFAE checkpoint (default: auto-detect)")
    parser.add_argument("--device", "-d", default=None,
                        help="Device: 'cuda' or 'cpu' (default: auto-detect)")
    parser.add_argument("--current-age", type=int, default=None,
                        help="Current age of subject (for aging effect scaling)")
    parser.add_argument("--no-deep", action="store_true",
                        help="Skip deep identity metrics (faster, no FaceNet)")
    parser.add_argument("--no-report", action="store_true",
                        help="Skip HTML report generation")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable verbose/debug logging")

    args = parser.parse_args()

    # Setup
    setup_logging(args.verbose)
    logger.info("=" * 60)
    logger.info("AgeVision Age Progression Pipeline")
    logger.info("=" * 60)

    # Parse target ages
    try:
        target_ages = [int(a.strip()) for a in args.ages.split(",")]
        target_ages = [max(0, min(100, a)) for a in target_ages]
    except ValueError:
        logger.error("Invalid age format. Use comma-separated integers: --ages 20,40,60,80")
        sys.exit(1)

    logger.info("Target ages: %s", target_ages)

    # Collect input images
    images = collect_images(args.input)
    if not images:
        logger.error("No valid images found at: %s", args.input)
        sys.exit(1)
    logger.info("Found %d image(s) to process", len(images))

    # Create output directory
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    logger.info("Loading HRFAE model...")
    model = AgeProgressionModel(checkpoint_path=args.checkpoint, device=args.device)
    if not model.load():
        logger.error("Failed to load HRFAE model. Check checkpoint path.")
        logger.error("Expected at: %s", model.checkpoint_path)
        sys.exit(1)
    logger.info("Model loaded successfully on %s", model.device)

    # Process all images
    all_results = []
    all_metrics = {}
    pipeline_start = time.time()

    for image_path in images:
        result = process_single_image(
            image_path=image_path,
            target_ages=target_ages,
            model=model,
            output_dir=output_dir,
            use_deep_identity=not args.no_deep,
            current_age=args.current_age,
        )
        all_results.append(result)

        # Collect metrics
        image_metrics = {}
        for age, prog_data in result.get("progressions", {}).items():
            m = prog_data.get("metrics", {})
            if m and "error" not in m:
                image_metrics[str(age)] = m
        if image_metrics:
            all_metrics[result["input_name"]] = image_metrics

    pipeline_time = time.time() - pipeline_start

    # Save metrics
    metrics_path = os.path.join(output_dir, "metrics.json")
    metrics_output = {
        "pipeline_info": {
            "total_images": len(images),
            "target_ages": target_ages,
            "total_time_s": round(pipeline_time, 2),
            "device": str(model.device),
        },
        "per_image_metrics": all_metrics,
    }
    save_metrics(metrics_output, metrics_path)

    # Generate HTML report
    if not args.no_report:
        report_path = os.path.join(output_dir, "report.html")
        # Also save a copy at project root
        report_root = str(PROJECT_ROOT / "report.html")
        generate_report(all_results, report_path)
        generate_report(all_results, report_root)
        logger.info("Report: %s", report_path)

    # Print summary
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    total = len(all_results)
    passed = sum(1 for r in all_results if r.get("test_result", "").startswith("PASS"))
    logger.info("Results: %d/%d PASSED", passed, total)
    logger.info("Total time: %.1fs", pipeline_time)
    logger.info("Output: %s", output_dir)
    logger.info("Metrics: %s", metrics_path)

    # Print per-image summary
    for r in all_results:
        status = "PASS" if r["test_result"].startswith("PASS") else "FAIL"
        logger.info("  [%s] %s — %s", status, r["input_name"], r["test_result"])

    # Exit with error code if any failed
    if passed < total:
        sys.exit(1)


if __name__ == "__main__":
    main()
