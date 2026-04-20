"""
AgeVision — Model Checkpoint Downloader
========================================
Downloads all ML model checkpoints from Google Drive.

Usage:
    python download_checkpoints.py              # Download all models
    python download_checkpoints.py --model sam  # Download specific model

After uploading your files to Google Drive:
  1. Right-click the file → Share → "Anyone with the link" → Viewer
  2. Copy the link. The file ID is the string between /d/ and /view
     e.g. https://drive.google.com/file/d/THIS_IS_THE_ID/view
  3. Paste the ID into the MODELS dict below.
"""

import argparse
import os
import sys
import zipfile

try:
    import gdown
except ImportError:
    print("ERROR: gdown is required. Run: pip install gdown")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────
#  FILL IN YOUR GOOGLE DRIVE FILE IDs BELOW
#  After uploading each file to Google Drive, replace
#  "YOUR_GDRIVE_FILE_ID" with the actual ID from the share link.
# ─────────────────────────────────────────────────────────────

CHECKPOINTS_DIR = os.path.join(
    os.path.dirname(__file__), "agevision_backend", "checkpoints"
)

MODELS = {
    # ── SAM: Style-based Age Manipulation ────────────────────
    # Upload: agevision_backend/checkpoints/sam_indian_best.pt  (2.2 GB)
    "sam_indian": {
        "gdrive_id":   "YOUR_GDRIVE_FILE_ID",
        "output_path": os.path.join(CHECKPOINTS_DIR, "sam_indian_best.pt"),
        "size":        "2.2 GB",
        "description": "SAM fine-tuned on Indian face dataset",
        "required":    True,
    },
    # Upload: agevision_backend/checkpoints/sam_ffhq_aging.pt  (2.2 GB)
    "sam_ffhq": {
        "gdrive_id":   "YOUR_GDRIVE_FILE_ID",
        "output_path": os.path.join(CHECKPOINTS_DIR, "sam_ffhq_aging.pt"),
        "size":        "2.2 GB",
        "description": "SAM pretrained on FFHQ dataset",
        "required":    True,
    },

    # ── HRFAE: High-Resolution Face Age Editing GAN ──────────
    # Upload: agevision_backend/checkpoints/hrfae_best.pth  (50 MB)
    "hrfae": {
        "gdrive_id":   "YOUR_GDRIVE_FILE_ID",
        "output_path": os.path.join(CHECKPOINTS_DIR, "hrfae_best.pth"),
        "size":        "50 MB",
        "description": "HRFAE GAN weights (used by CLI pipeline)",
        "required":    True,
    },

    # ── Fast-AgingGAN ─────────────────────────────────────────
    # Upload: agevision_backend/checkpoints/fast_aging_gan.pth  (11 MB)
    "fast_aging": {
        "gdrive_id":   "YOUR_GDRIVE_FILE_ID",
        "output_path": os.path.join(CHECKPOINTS_DIR, "fast_aging_gan.pth"),
        "size":        "11 MB",
        "description": "Fast CycleGAN-based aging model",
        "required":    False,
    },

    # ── MiVOLO: Age + Gender Predictor ───────────────────────
    # Upload: agevision_backend/checkpoints/mivolo_indian/mivolo_indian_best.pt  (110 MB)
    # Upload the file directly (not the folder). The script creates the folder.
    "mivolo": {
        "gdrive_id":   "YOUR_GDRIVE_FILE_ID",
        "output_path": os.path.join(CHECKPOINTS_DIR, "mivolo_indian",
                                    "mivolo_indian_best.pt"),
        "size":        "110 MB",
        "description": "MiVOLO age/gender predictor fine-tuned on Indian faces",
        "required":    True,
    },

    # ── dlib Face Landmarks ───────────────────────────────────
    # Upload: agevision_backend/checkpoints/shape_predictor_68_face_landmarks.dat  (96 MB)
    "dlib_landmarks": {
        "gdrive_id":   "YOUR_GDRIVE_FILE_ID",
        "output_path": os.path.join(CHECKPOINTS_DIR,
                                    "shape_predictor_68_face_landmarks.dat"),
        "size":        "96 MB",
        "description": "dlib 68-point facial landmark predictor",
        "required":    True,
    },

    # ── FADING: Diffusion-based Aging Model ──────────────────
    # This is a FOLDER — zip the entire directory first:
    #   Zip:    agevision_backend/checkpoints/fading/finetune_double_prompt_150_random/
    #   Rename: fading_model.zip
    #   Upload: fading_model.zip to Google Drive  (~5.2 GB)
    # The script downloads and extracts it automatically.
    "fading": {
        "gdrive_id":   "YOUR_GDRIVE_FILE_ID",
        "output_path": os.path.join(CHECKPOINTS_DIR, "fading", "fading_model.zip"),
        "extract_to":  os.path.join(CHECKPOINTS_DIR, "fading"),
        "size":        "5.2 GB",
        "description": "FADING Stable Diffusion model (requires ~6-8 GB VRAM)",
        "required":    False,
        "is_zip":      True,
    },
}


def _already_exists(model_key: str, info: dict) -> bool:
    if info.get("is_zip"):
        # Check for extracted folder marker
        marker = os.path.join(info["extract_to"],
                              "finetune_double_prompt_150_random",
                              "model_index.json")
        return os.path.isfile(marker)
    return os.path.isfile(info["output_path"])


def download_model(model_key: str, info: dict) -> bool:
    if info["gdrive_id"] == "YOUR_GDRIVE_FILE_ID":
        print(f"  [SKIP] {model_key}: Google Drive ID not set in download_checkpoints.py")
        return False

    if _already_exists(model_key, info):
        print(f"  [OK]   {model_key}: already present, skipping.")
        return True

    os.makedirs(os.path.dirname(info["output_path"]), exist_ok=True)

    print(f"\n  Downloading {model_key} ({info['size']}) — {info['description']}")
    url = f"https://drive.google.com/uc?id={info['gdrive_id']}"

    try:
        gdown.download(url, info["output_path"], quiet=False)
    except Exception as e:
        print(f"  [ERROR] Failed to download {model_key}: {e}")
        return False

    if not os.path.isfile(info["output_path"]):
        print(f"  [ERROR] Download completed but file not found: {info['output_path']}")
        return False

    if info.get("is_zip"):
        print(f"  Extracting {model_key}...")
        try:
            with zipfile.ZipFile(info["output_path"], "r") as zf:
                zf.extractall(info["extract_to"])
            os.remove(info["output_path"])
            print(f"  [OK]   {model_key}: extracted to {info['extract_to']}")
        except Exception as e:
            print(f"  [ERROR] Failed to extract {model_key}: {e}")
            return False
    else:
        print(f"  [OK]   {model_key}: saved to {info['output_path']}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Download AgeVision model checkpoints from Google Drive"
    )
    parser.add_argument(
        "--model", "-m",
        choices=list(MODELS.keys()) + ["all"],
        default="all",
        help="Which model to download (default: all)"
    )
    parser.add_argument(
        "--required-only",
        action="store_true",
        help="Only download models marked as required=True"
    )
    args = parser.parse_args()

    targets = MODELS if args.model == "all" else {args.model: MODELS[args.model]}
    if args.required_only:
        targets = {k: v for k, v in targets.items() if v.get("required")}

    print("\nAgeVision Checkpoint Downloader")
    print("=" * 50)
    print(f"Destination: {CHECKPOINTS_DIR}\n")

    results = {}
    for key, info in targets.items():
        results[key] = download_model(key, info)

    print("\n" + "=" * 50)
    print("Summary:")
    for key, ok in results.items():
        status = "[OK]  " if ok else "[FAIL]"
        print(f"  {status} {key}")

    failed = [k for k, ok in results.items() if not ok]
    if failed:
        print(f"\nFailed or skipped: {', '.join(failed)}")
        print("Check that Google Drive IDs are set in download_checkpoints.py")
        sys.exit(1)
    else:
        print("\nAll checkpoints ready.")


if __name__ == "__main__":
    main()
