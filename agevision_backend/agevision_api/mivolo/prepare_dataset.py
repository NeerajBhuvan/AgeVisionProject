"""
Dataset Preparation for MiVOLO v2 Fine-tuning
===============================================
Downloads and prepares Indian face datasets into a unified format:
  1. IFAD  (Indian Face Age Database) — 3,296 images, 55 Indian celebrities
  2. UTKFace Indian subset (race=3) — thousands of Indian faces, 0-116 years

Output: a single folder of face images + a CSV (filename, age, gender, source).
"""

import csv
import logging
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger("agevision.mivolo.prepare_dataset")

# ── Defaults ──────────────────────────────────────────────────────────────
DEFAULT_OUTPUT_DIR = os.path.join("datasets", "indian_faces_mivolo")
IFAD_REPO_URL = "https://github.com/IndianAgeDatabase/IFAD.git"
UTKFACE_KAGGLE = "jangedoo/utkface-new"
MIN_FACE_SIZE = 64   # Skip images smaller than this


def prepare_all(output_dir: str = DEFAULT_OUTPUT_DIR,
                include_ifad: bool = True,
                include_utkface: bool = True,
                utkface_path: str = None,
                upscale_ifad: bool = True) -> str:
    """
    Download/prepare datasets and produce a unified CSV + image folder.

    Parameters
    ----------
    output_dir : str
        Root output directory. Will contain images/ and labels.csv.
    include_ifad : bool
        Download and include IFAD dataset.
    include_utkface : bool
        Include UTKFace Indian subset (race=3).
    utkface_path : str, optional
        Path to existing UTKFace folder. If None, attempts Kaggle download.
    upscale_ifad : bool
        Upscale IFAD 128x128 images to 224x224 using bicubic interpolation.

    Returns
    -------
    str : Path to the generated labels.csv.
    """
    output_dir = os.path.abspath(output_dir)
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    all_samples = []

    if include_ifad:
        logger.info("Preparing IFAD dataset...")
        ifad_samples = _prepare_ifad(output_dir, images_dir, upscale=upscale_ifad)
        all_samples.extend(ifad_samples)
        logger.info("IFAD: %d samples", len(ifad_samples))

    if include_utkface:
        logger.info("Preparing UTKFace Indian subset...")
        utk_samples = _prepare_utkface_indian(output_dir, images_dir,
                                               existing_path=utkface_path)
        all_samples.extend(utk_samples)
        logger.info("UTKFace-Indian: %d samples", len(utk_samples))

    if not all_samples:
        raise RuntimeError("No samples collected. Check dataset paths and network.")

    # Write unified CSV
    csv_path = os.path.join(output_dir, "labels.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "age", "gender", "source"])
        writer.writeheader()
        writer.writerows(all_samples)

    logger.info("Dataset ready: %d total samples -> %s", len(all_samples), csv_path)
    _print_stats(all_samples)
    return csv_path


# ══════════════════════════════════════════════════════════════════════════
#  IFAD
# ══════════════════════════════════════════════════════════════════════════

def _prepare_ifad(output_dir: str, images_dir: str,
                  upscale: bool = True) -> list[dict]:
    """Clone IFAD repo and extract images with age labels."""
    ifad_dir = os.path.join(output_dir, "_raw", "IFAD")

    if not os.path.isdir(ifad_dir):
        logger.info("Cloning IFAD repository...")
        os.makedirs(os.path.dirname(ifad_dir), exist_ok=True)
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", IFAD_REPO_URL, ifad_dir],
                check=True, capture_output=True, text=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.error("Failed to clone IFAD repo: %s", e)
            logger.info("Please manually clone %s to %s", IFAD_REPO_URL, ifad_dir)
            return []

    samples = []
    # IFAD structure: Subject folders with images sorted by age
    # Annotations are in MATLAB .mat files or folder/filename patterns
    # We parse age from folder structure and filename patterns
    for subject_dir in sorted(Path(ifad_dir).iterdir()):
        if not subject_dir.is_dir() or subject_dir.name.startswith((".", "_")):
            continue

        # Collect images from this subject
        image_files = sorted(
            [f for f in subject_dir.iterdir()
             if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")],
            key=lambda f: f.name,
        )

        if not image_files:
            continue

        for img_path in image_files:
            age = _parse_ifad_age(img_path)
            if age is None:
                continue

            # Copy/upscale image to output
            dest_name = f"ifad_{subject_dir.name}_{img_path.stem}.jpg"
            dest_path = os.path.join(images_dir, dest_name)

            img = cv2.imread(str(img_path))
            if img is None:
                continue

            h, w = img.shape[:2]
            if min(h, w) < MIN_FACE_SIZE:
                continue

            if upscale and max(h, w) < 224:
                img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)

            cv2.imwrite(dest_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])

            samples.append({
                "filename": dest_name,
                "age": age,
                "gender": "Unknown",
                "source": "ifad",
            })

    return samples


def _parse_ifad_age(img_path: Path) -> int | None:
    """
    Extract age from IFAD filename/path.

    IFAD images are named with patterns like:
      - 'age_25.jpg', '25.jpg'
      - Or numbered sequentially within age-sorted folders.
    We try numeric extraction from the filename.
    """
    stem = img_path.stem.lower()

    # Pattern: explicit age in filename (e.g., "age_25", "25_face", "img_25")
    match = re.search(r'(?:age[_\-]?)(\d{1,3})', stem)
    if match:
        age = int(match.group(1))
        if 0 <= age <= 100:
            return age

    # Pattern: filename is just a number (e.g., "25.jpg")
    match = re.fullmatch(r'(\d{1,3})', stem)
    if match:
        age = int(match.group(1))
        if 0 <= age <= 100:
            return age

    # Pattern: number at start or end of filename
    match = re.search(r'(\d{1,3})', stem)
    if match:
        age = int(match.group(1))
        if 1 <= age <= 100:
            return age

    return None


# ══════════════════════════════════════════════════════════════════════════
#  UTKFace Indian subset
# ══════════════════════════════════════════════════════════════════════════

def _prepare_utkface_indian(output_dir: str, images_dir: str,
                             existing_path: str = None) -> list[dict]:
    """
    Extract Indian faces (race=3) from UTKFace dataset.

    UTKFace filename format: [age]_[gender]_[race]_[date&time].jpg
    Gender: 0=Male, 1=Female
    Race: 0=White, 1=Black, 2=Asian, 3=Indian, 4=Others
    """
    utk_dir = existing_path

    if utk_dir is None or not os.path.isdir(utk_dir):
        utk_dir = os.path.join(output_dir, "_raw", "UTKFace")
        if not os.path.isdir(utk_dir):
            utk_dir = _download_utkface(utk_dir)

    if utk_dir is None or not os.path.isdir(utk_dir):
        logger.warning("UTKFace not available. Skipping.")
        return []

    samples = []
    gender_map = {"0": "Male", "1": "Female"}

    for img_file in Path(utk_dir).rglob("*.jpg"):
        parts = img_file.name.split("_")
        if len(parts) < 4:
            continue

        try:
            age = int(parts[0])
            gender_code = parts[1]
            race = int(parts[2])
        except (ValueError, IndexError):
            continue

        # Only keep Indian faces (race=3)
        if race != 3:
            continue

        if age < 0 or age > 100:
            continue

        # Copy to output
        dest_name = f"utk_{img_file.name}"
        dest_path = os.path.join(images_dir, dest_name)

        img = cv2.imread(str(img_file))
        if img is None:
            continue

        h, w = img.shape[:2]
        if min(h, w) < MIN_FACE_SIZE:
            continue

        # UTKFace images are 200x200, resize to 224x224
        if h != 224 or w != 224:
            img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)

        cv2.imwrite(dest_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])

        samples.append({
            "filename": dest_name,
            "age": age,
            "gender": gender_map.get(gender_code, "Unknown"),
            "source": "utkface",
        })

    return samples


def _download_utkface(dest_dir: str) -> str | None:
    """Attempt to download UTKFace via kagglehub or kaggle CLI."""
    os.makedirs(dest_dir, exist_ok=True)

    # Try kagglehub first
    try:
        import kagglehub
        path = kagglehub.dataset_download("jangedoo/utkface-new")
        logger.info("UTKFace downloaded via kagglehub to: %s", path)
        # Copy contents to dest_dir
        for f in Path(path).rglob("*.jpg"):
            shutil.copy2(str(f), dest_dir)
        return dest_dir
    except ImportError:
        pass
    except Exception as e:
        logger.warning("kagglehub download failed: %s", e)

    # Try kaggle CLI
    try:
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", UTKFACE_KAGGLE,
             "-p", dest_dir, "--unzip"],
            check=True, capture_output=True, text=True,
        )
        logger.info("UTKFace downloaded via Kaggle CLI to: %s", dest_dir)
        return dest_dir
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    logger.warning(
        "Could not auto-download UTKFace. Please download manually:\n"
        "  1. From Kaggle: https://www.kaggle.com/datasets/jangedoo/utkface-new\n"
        "  2. Extract to: %s\n"
        "  Or: pip install kagglehub && python -c "
        "\"import kagglehub; kagglehub.dataset_download('jangedoo/utkface-new')\"",
        dest_dir,
    )
    return None


# ══════════════════════════════════════════════════════════════════════════
#  Utilities
# ══════════════════════════════════════════════════════════════════════════

def _print_stats(samples: list[dict]):
    """Print dataset statistics."""
    import collections

    ages = [s["age"] for s in samples]
    sources = collections.Counter(s["source"] for s in samples)
    genders = collections.Counter(s["gender"] for s in samples)

    print(f"\n{'='*50}")
    print(f"  Dataset Summary")
    print(f"{'='*50}")
    print(f"  Total samples : {len(samples)}")
    print(f"  Age range     : {min(ages)} - {max(ages)}")
    print(f"  Mean age      : {sum(ages)/len(ages):.1f}")
    print(f"  Sources       : {dict(sources)}")
    print(f"  Gender        : {dict(genders)}")

    # Age distribution buckets
    buckets = {"0-10": 0, "11-20": 0, "21-30": 0, "31-40": 0,
               "41-50": 0, "51-60": 0, "61-70": 0, "71+": 0}
    for a in ages:
        if a <= 10: buckets["0-10"] += 1
        elif a <= 20: buckets["11-20"] += 1
        elif a <= 30: buckets["21-30"] += 1
        elif a <= 40: buckets["31-40"] += 1
        elif a <= 50: buckets["41-50"] += 1
        elif a <= 60: buckets["51-60"] += 1
        elif a <= 70: buckets["61-70"] += 1
        else: buckets["71+"] += 1

    print(f"  Age buckets   :")
    for bucket, count in buckets.items():
        bar = "#" * (count // max(1, len(samples) // 40))
        print(f"    {bucket:>5s}: {count:>5d}  {bar}")
    print(f"{'='*50}\n")


# ── CLI entry point ───────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Prepare Indian face dataset for MiVOLO fine-tuning")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                        help="Output directory (default: %(default)s)")
    parser.add_argument("--utkface-path", default=None,
                        help="Path to existing UTKFace folder (skips download)")
    parser.add_argument("--no-ifad", action="store_true",
                        help="Skip IFAD dataset")
    parser.add_argument("--no-utkface", action="store_true",
                        help="Skip UTKFace Indian subset")
    parser.add_argument("--no-upscale", action="store_true",
                        help="Don't upscale small IFAD images")

    args = parser.parse_args()
    prepare_all(
        output_dir=args.output_dir,
        include_ifad=not args.no_ifad,
        include_utkface=not args.no_utkface,
        utkface_path=args.utkface_path,
        upscale_ifad=not args.no_upscale,
    )
