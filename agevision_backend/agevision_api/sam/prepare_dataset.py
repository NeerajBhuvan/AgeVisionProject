"""
Indian Face Dataset Preparation
================================
Downloads and prepares Indian face images with age labels for SAM fine-tuning.

Supported sources:
  1. UTKFace - ethnicity label 3 = Indian (ages 0-116, filename format: age_gender_race_date.jpg)
  2. FairFace - 'Indian' category with age ranges (10K+ images)
  3. Manual directory - user-provided images with ages.csv

Usage:
    cd agevision_backend
    python -m agevision_api.sam.prepare_dataset --source utkface --output ./indian_faces
    python -m agevision_api.sam.prepare_dataset --source fairface --output ./indian_faces
    python -m agevision_api.sam.prepare_dataset --source manual --input_dir ./my_images --output ./indian_faces
"""

import argparse
import csv
import logging
import os
import re
import shutil
import sys
import zipfile
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger('sam.prepare_dataset')

# Age range we care about for the missing-person use case
MIN_AGE = 10
MAX_AGE = 80


def download_file(url: str, dest: str) -> str:
    """Download a file with progress display."""
    import requests
    from tqdm import tqdm

    logger.info("Downloading %s ...", url)
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    total = int(resp.headers.get('content-length', 0))

    with open(dest, 'wb') as f, tqdm(total=total, unit='B', unit_scale=True) as pbar:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))

    return dest


def try_gdown(file_id: str, dest: str) -> str:
    """Download from Google Drive using gdown."""
    import gdown
    url = f"https://drive.google.com/uc?id={file_id}"
    logger.info("Downloading from Google Drive (ID: %s) ...", file_id)
    gdown.download(url, dest, quiet=False)
    return dest


# ─────────────────────────────────────────────────────────────
#  UTKFace Source
# ─────────────────────────────────────────────────────────────

# UTKFace Google Drive file IDs (3 parts)
UTKFACE_GDRIVE_IDS = [
    '0BxYys69jI14kYVM3aVhKS1VhRUk',   # part 1
    '0BxYys69jI14kU0I1YUQyY1ZDRUE',   # part 2
    '0BxYys69jI14kSVdWWllDMWhnN2c',   # part 3 (aligned & cropped)
]

# Alternative: Kaggle UTKFace
UTKFACE_KAGGLE = "jangedoo/utkface-new"


def prepare_utkface(output_dir: str, min_age: int = MIN_AGE, max_age: int = MAX_AGE):
    """
    Download and prepare UTKFace Indian subset.

    UTKFace filename format: [age]_[gender]_[race]_[date].jpg
    Race labels: 0=White, 1=Black, 2=Asian, 3=Indian, 4=Others

    We filter for race=3 (Indian) and age range [min_age, max_age].
    """
    images_dir = os.path.join(output_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)

    # Step 1: Try to download via gdown
    temp_dir = os.path.join(output_dir, '_temp')
    os.makedirs(temp_dir, exist_ok=True)

    utkface_dir = None

    # Try gdown for the aligned & cropped set (part 3)
    try:
        zip_path = os.path.join(temp_dir, 'utkface_aligned.tar.gz')
        try_gdown(UTKFACE_GDRIVE_IDS[2], zip_path)

        import tarfile
        if zip_path.endswith('.tar.gz'):
            with tarfile.open(zip_path, 'r:gz') as tar:
                tar.extractall(temp_dir)
        utkface_dir = temp_dir
        logger.info("UTKFace downloaded and extracted")

    except Exception as e:
        logger.warning("gdown download failed: %s", e)
        logger.info("Trying alternative download method...")

        # Try Kaggle if available
        try:
            import kaggle
            kaggle.api.dataset_download_files(
                UTKFACE_KAGGLE, path=temp_dir, unzip=True)
            utkface_dir = temp_dir
            logger.info("UTKFace downloaded from Kaggle")
        except Exception as e2:
            logger.warning("Kaggle download also failed: %s", e2)
            logger.info(
                "\n╔══════════════════════════════════════════════════════════╗\n"
                "║  MANUAL DOWNLOAD REQUIRED                               ║\n"
                "║                                                          ║\n"
                "║  Download UTKFace dataset manually:                      ║\n"
                "║  1. Kaggle: kaggle.com/jangedoo/utkface-new             ║\n"
                "║  2. Place images in: %s                                  ║\n"
                "║  3. Re-run this script with --source manual             ║\n"
                "╚══════════════════════════════════════════════════════════╝",
                temp_dir,
            )
            return None

    # Step 2: Filter for Indian faces (race=3) in age range
    if utkface_dir is None:
        return None

    # Find all UTKFace images (search recursively)
    all_images = []
    for root, dirs, files in os.walk(utkface_dir):
        for fname in files:
            if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.chip.jpg')):
                all_images.append(os.path.join(root, fname))

    logger.info("Found %d total UTKFace images", len(all_images))

    # Parse filename and filter
    indian_samples = []
    pattern = re.compile(r'^(\d+)_(\d)_(\d)_(\d+)')

    for img_path in all_images:
        fname = os.path.basename(img_path)
        match = pattern.match(fname)
        if not match:
            continue

        age = int(match.group(1))
        gender = int(match.group(2))   # 0=Male, 1=Female
        race = int(match.group(3))     # 3=Indian

        if race == 3 and min_age <= age <= max_age:
            indian_samples.append({
                'src_path': img_path,
                'age': age,
                'gender': 'Male' if gender == 0 else 'Female',
                'filename': fname,
            })

    logger.info("Found %d Indian faces (age %d-%d)", len(indian_samples), min_age, max_age)

    if not indian_samples:
        logger.warning("No Indian faces found! Check if UTKFace was extracted correctly.")
        return None

    # Step 3: Copy images and create CSV
    csv_path = os.path.join(output_dir, 'ages.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['filename', 'age', 'gender'])
        writer.writeheader()

        for sample in indian_samples:
            dest = os.path.join(images_dir, sample['filename'])
            if not os.path.exists(dest):
                shutil.copy2(sample['src_path'], dest)
            writer.writerow({
                'filename': sample['filename'],
                'age': sample['age'],
                'gender': sample['gender'],
            })

    # Cleanup temp
    shutil.rmtree(temp_dir, ignore_errors=True)

    # Print age distribution
    age_dist = {}
    for s in indian_samples:
        decade = (s['age'] // 10) * 10
        age_dist[decade] = age_dist.get(decade, 0) + 1

    logger.info("\nAge distribution of Indian faces:")
    for decade in sorted(age_dist):
        bar = '█' * (age_dist[decade] // 5) or '▏'
        logger.info("  %2d-%2d: %4d  %s", decade, decade + 9, age_dist[decade], bar)

    logger.info("\nDataset ready:")
    logger.info("  Images: %s", images_dir)
    logger.info("  CSV:    %s", csv_path)
    logger.info("  Total:  %d images", len(indian_samples))

    return {'images_dir': images_dir, 'csv_path': csv_path, 'count': len(indian_samples)}


# ─────────────────────────────────────────────────────────────
#  FairFace Source
# ─────────────────────────────────────────────────────────────

FAIRFACE_LABELS_URL = "https://raw.githubusercontent.com/joojs/fairface/master/fairface_label_train.csv"
FAIRFACE_VAL_LABELS_URL = "https://raw.githubusercontent.com/joojs/fairface/master/fairface_label_val.csv"


def prepare_fairface(output_dir: str, min_age: int = MIN_AGE, max_age: int = MAX_AGE):
    """
    Download and prepare FairFace Indian subset.

    FairFace has age ranges: 0-2, 3-9, 10-19, 20-29, 30-39, 40-49, 50-59, 60-69, 70+
    Race includes: 'Indian' category
    """
    import requests
    import pandas as pd

    images_dir = os.path.join(output_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)

    # Download labels
    logger.info("Downloading FairFace labels...")
    try:
        train_labels = pd.read_csv(FAIRFACE_LABELS_URL)
        val_labels = pd.read_csv(FAIRFACE_VAL_LABELS_URL)
        labels = pd.concat([train_labels, val_labels], ignore_index=True)
    except Exception as e:
        logger.error("Failed to download FairFace labels: %s", e)
        logger.info(
            "\n╔══════════════════════════════════════════════════════════╗\n"
            "║  Download FairFace dataset from:                         ║\n"
            "║  github.com/joojs/fairface                               ║\n"
            "║  Place images in the output directory and re-run.        ║\n"
            "╚══════════════════════════════════════════════════════════╝"
        )
        return None

    # Filter for Indian faces
    indian = labels[labels['race'] == 'Indian'].copy()
    logger.info("Found %d Indian faces in FairFace labels", len(indian))

    # Map age ranges to midpoint ages for training
    age_range_map = {
        '0-2': 1, '3-9': 6, '10-19': 15, '20-29': 25,
        '30-39': 35, '40-49': 45, '50-59': 55, '60-69': 65,
        'more than 70': 75,
    }

    indian['age_midpoint'] = indian['age'].map(age_range_map)
    indian = indian[
        (indian['age_midpoint'] >= min_age) & (indian['age_midpoint'] <= max_age)
    ]

    logger.info("After age filter (%d-%d): %d images", min_age, max_age, len(indian))

    if indian.empty:
        logger.warning("No matching images found")
        return None

    # FairFace images need to be downloaded separately
    # The labels CSV has 'file' column with paths like 'train/1.jpg'
    logger.info(
        "\n╔══════════════════════════════════════════════════════════════╗\n"
        "║  FairFace images must be downloaded separately.              ║\n"
        "║                                                              ║\n"
        "║  Steps:                                                      ║\n"
        "║  1. Download from: github.com/joojs/fairface                ║\n"
        "║     (padding=0.25 version recommended)                       ║\n"
        "║  2. Extract to a folder (e.g., fairface_images/)            ║\n"
        "║  3. Re-run with:                                             ║\n"
        "║     --source fairface --fairface_images <path>              ║\n"
        "╚══════════════════════════════════════════════════════════════╝"
    )

    # Save the filtered labels for later use
    filtered_csv = os.path.join(output_dir, 'fairface_indian_labels.csv')
    indian.to_csv(filtered_csv, index=False)
    logger.info("Saved filtered labels to: %s (%d entries)", filtered_csv, len(indian))

    return {'csv_path': filtered_csv, 'count': len(indian)}


def copy_fairface_images(output_dir: str, fairface_images_dir: str,
                         min_age: int = MIN_AGE, max_age: int = MAX_AGE):
    """Copy FairFace Indian images once the image archive is available."""
    import pandas as pd

    filtered_csv = os.path.join(output_dir, 'fairface_indian_labels.csv')
    if not os.path.isfile(filtered_csv):
        logger.error("Run prepare_fairface first to get filtered labels")
        return None

    indian = pd.read_csv(filtered_csv)
    images_dir = os.path.join(output_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)

    age_range_map = {
        '0-2': 1, '3-9': 6, '10-19': 15, '20-29': 25,
        '30-39': 35, '40-49': 45, '50-59': 55, '60-69': 65,
        'more than 70': 75,
    }

    csv_path = os.path.join(output_dir, 'ages.csv')
    copied = 0

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['filename', 'age', 'gender'])
        writer.writeheader()

        for _, row in indian.iterrows():
            src = os.path.join(fairface_images_dir, row['file'])
            if not os.path.isfile(src):
                continue

            dest_name = f"ff_{os.path.basename(row['file'])}"
            dest = os.path.join(images_dir, dest_name)
            shutil.copy2(src, dest)

            age = age_range_map.get(row['age'], 30)
            gender = row.get('gender', 'Unknown')

            writer.writerow({
                'filename': dest_name,
                'age': age,
                'gender': gender,
            })
            copied += 1

    logger.info("Copied %d FairFace Indian images", copied)
    return {'images_dir': images_dir, 'csv_path': csv_path, 'count': copied}


# ─────────────────────────────────────────────────────────────
#  Manual Source
# ─────────────────────────────────────────────────────────────

def prepare_manual(output_dir: str, input_dir: str,
                   min_age: int = MIN_AGE, max_age: int = MAX_AGE):
    """
    Prepare dataset from a manual directory of images.

    Supports two formats:
    1. UTKFace-style filenames: age_gender_race_id.jpg
    2. Existing ages.csv in the input directory
    """
    images_dir = os.path.join(output_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)

    existing_csv = os.path.join(input_dir, 'ages.csv')
    samples = []

    if os.path.isfile(existing_csv):
        # Use existing CSV
        logger.info("Found existing ages.csv in %s", input_dir)
        with open(existing_csv, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                age = int(float(row['age']))
                if min_age <= age <= max_age:
                    src = os.path.join(input_dir, row['filename'])
                    if os.path.isfile(src):
                        samples.append({
                            'src_path': src,
                            'filename': row['filename'],
                            'age': age,
                            'gender': row.get('gender', 'Unknown'),
                        })
    else:
        # Try to parse from filenames
        pattern = re.compile(r'^(\d+)_(\d)_(\d)_(\d+)')
        for fname in os.listdir(input_dir):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            match = pattern.match(fname)
            if match:
                age = int(match.group(1))
                gender = 'Male' if int(match.group(2)) == 0 else 'Female'
            else:
                # Try simple "age_name.jpg" format
                age_match = re.match(r'^(\d+)[_\-]', fname)
                if age_match:
                    age = int(age_match.group(1))
                    gender = 'Unknown'
                else:
                    continue

            if min_age <= age <= max_age:
                samples.append({
                    'src_path': os.path.join(input_dir, fname),
                    'filename': fname,
                    'age': age,
                    'gender': gender,
                })

    if not samples:
        logger.error("No valid samples found in %s", input_dir)
        logger.info("Expected either:\n"
                     "  - ages.csv with columns: filename,age,gender\n"
                     "  - Images named as: age_gender_race_id.jpg (UTKFace format)")
        return None

    # Copy and create CSV
    csv_path = os.path.join(output_dir, 'ages.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['filename', 'age', 'gender'])
        writer.writeheader()

        for sample in samples:
            dest = os.path.join(images_dir, sample['filename'])
            if not os.path.exists(dest):
                shutil.copy2(sample['src_path'], dest)
            writer.writerow({
                'filename': sample['filename'],
                'age': sample['age'],
                'gender': sample['gender'],
            })

    logger.info("Prepared %d images from manual source", len(samples))
    return {'images_dir': images_dir, 'csv_path': csv_path, 'count': len(samples)}


# ─────────────────────────────────────────────────────────────
#  Merge Multiple Sources
# ─────────────────────────────────────────────────────────────

def merge_datasets(output_dir: str, *source_dirs):
    """Merge multiple prepared dataset directories into one."""
    merged_images = os.path.join(output_dir, 'images')
    os.makedirs(merged_images, exist_ok=True)

    all_samples = []

    for src_dir in source_dirs:
        csv_path = os.path.join(src_dir, 'ages.csv')
        images_dir = os.path.join(src_dir, 'images')

        if not os.path.isfile(csv_path):
            logger.warning("No ages.csv in %s, skipping", src_dir)
            continue

        with open(csv_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                src_img = os.path.join(images_dir, row['filename'])
                if os.path.isfile(src_img):
                    dest_img = os.path.join(merged_images, row['filename'])
                    if not os.path.exists(dest_img):
                        shutil.copy2(src_img, dest_img)
                    all_samples.append(row)

    # Write merged CSV
    merged_csv = os.path.join(output_dir, 'ages.csv')
    with open(merged_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['filename', 'age', 'gender'])
        writer.writeheader()
        writer.writerows(all_samples)

    logger.info("Merged dataset: %d total images in %s", len(all_samples), output_dir)
    return {'images_dir': merged_images, 'csv_path': merged_csv, 'count': len(all_samples)}


# ─────────────────────────────────────────────────────────────
#  Dataset Statistics
# ─────────────────────────────────────────────────────────────

def print_dataset_stats(csv_path: str):
    """Print detailed statistics about the prepared dataset."""
    samples = []
    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append(row)

    if not samples:
        logger.warning("Empty dataset!")
        return

    ages = [int(float(s['age'])) for s in samples]
    genders = [s.get('gender', 'Unknown') for s in samples]

    logger.info("\n" + "=" * 50)
    logger.info("Dataset Statistics")
    logger.info("=" * 50)
    logger.info("Total images: %d", len(samples))
    logger.info("Age range: %d - %d", min(ages), max(ages))
    logger.info("Mean age: %.1f", sum(ages) / len(ages))

    # Age distribution by decade
    logger.info("\nAge Distribution:")
    for decade in range(0, 90, 10):
        count = sum(1 for a in ages if decade <= a < decade + 10)
        bar = '█' * (count // 3) or ('▏' if count > 0 else '')
        logger.info("  %2d-%2d: %4d  %s", decade, decade + 9, count, bar)

    # Gender distribution
    logger.info("\nGender Distribution:")
    for g in set(genders):
        count = sum(1 for x in genders if x == g)
        logger.info("  %-10s: %4d (%.1f%%)", g, count, 100 * count / len(genders))

    logger.info("=" * 50)


# ─────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Prepare Indian face dataset for SAM fine-tuning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download UTKFace Indian subset (recommended)
  python -m agevision_api.sam.prepare_dataset --source utkface --output ./indian_faces

  # Download FairFace labels (images need separate download)
  python -m agevision_api.sam.prepare_dataset --source fairface --output ./indian_faces

  # Copy FairFace images after manual download
  python -m agevision_api.sam.prepare_dataset --source fairface --output ./indian_faces \\
      --fairface_images /path/to/fairface_images

  # Use your own images (with ages.csv or UTKFace-named files)
  python -m agevision_api.sam.prepare_dataset --source manual --input_dir ./my_images --output ./indian_faces

  # Merge multiple sources
  python -m agevision_api.sam.prepare_dataset --source merge --output ./indian_faces_merged \\
      --merge_dirs ./utkface_indian ./fairface_indian
        """,
    )
    parser.add_argument('--source', required=True,
                        choices=['utkface', 'fairface', 'manual', 'merge'],
                        help='Data source to prepare')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--input_dir', help='Input directory (for manual source)')
    parser.add_argument('--fairface_images', help='FairFace images directory')
    parser.add_argument('--merge_dirs', nargs='+', help='Directories to merge')
    parser.add_argument('--min_age', type=int, default=MIN_AGE, help='Minimum age filter')
    parser.add_argument('--max_age', type=int, default=MAX_AGE, help='Maximum age filter')

    args = parser.parse_args()

    if args.source == 'utkface':
        result = prepare_utkface(args.output, args.min_age, args.max_age)

    elif args.source == 'fairface':
        if args.fairface_images:
            result = copy_fairface_images(
                args.output, args.fairface_images, args.min_age, args.max_age)
        else:
            result = prepare_fairface(args.output, args.min_age, args.max_age)

    elif args.source == 'manual':
        if not args.input_dir:
            parser.error("--input_dir required for manual source")
        result = prepare_manual(args.output, args.input_dir, args.min_age, args.max_age)

    elif args.source == 'merge':
        if not args.merge_dirs:
            parser.error("--merge_dirs required for merge source")
        result = merge_datasets(args.output, *args.merge_dirs)

    if result:
        print_dataset_stats(result['csv_path'])
        logger.info("\nReady for training! Run:")
        logger.info("  python -m agevision_api.sam.train \\")
        logger.info("    --data_root %s/images \\", args.output)
        logger.info("    --age_csv %s/ages.csv \\", args.output)
        logger.info("    --checkpoint checkpoints/sam_indian_best.pt \\")
        logger.info("    --epochs 50 --batch_size 4")


if __name__ == '__main__':
    main()
