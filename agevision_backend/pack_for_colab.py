"""
Pack everything needed for SAM fine-tuning on Google Colab.

Creates a single zip file: colab_sam_training.zip
Upload this to Google Drive, then open the notebook in Colab.

Usage:
    cd agevision_backend
    python pack_for_colab.py
"""

import os
import shutil
import zipfile

BASE = os.path.dirname(os.path.abspath(__file__))
SAM_DIR = os.path.join(BASE, 'agevision_api', 'sam')
CHECKPOINTS_DIR = os.path.join(BASE, 'checkpoints')
OUTPUT_ZIP = os.path.join(BASE, 'colab_sam_training.zip')

# Files/folders to include from sam/
SAM_ITEMS = [
    '__init__.py',
    'train.py',
    'train_config.py',
    'train_colab.ipynb',
    'train_kaggle.ipynb',
    'dataset.py',
    'losses.py',
    'evaluate.py',
    'inference.py',
    'prepare_dataset.py',
    'configs',
    'datasets',
    'models',
    'scripts',
    'utils',
]

# Checkpoints to include
CHECKPOINTS = ['sam_ffhq_aging.pt']


def pack():
    print("Packing SAM training files for Google Colab...")
    print(f"Source: {SAM_DIR}")
    print(f"Output: {OUTPUT_ZIP}")
    print()

    with zipfile.ZipFile(OUTPUT_ZIP, 'w', zipfile.ZIP_DEFLATED) as zf:

        # 1. Pack SAM code
        for item in SAM_ITEMS:
            full_path = os.path.join(SAM_DIR, item)
            if os.path.isfile(full_path):
                arcname = f"sam/{item}"
                zf.write(full_path, arcname)
                print(f"  + {arcname}")
            elif os.path.isdir(full_path):
                for root, dirs, files in os.walk(full_path):
                    # Skip __pycache__
                    dirs[:] = [d for d in dirs if d != '__pycache__']
                    for fname in files:
                        if fname.endswith('.pyc'):
                            continue
                        fpath = os.path.join(root, fname)
                        arcname = os.path.join(
                            'sam',
                            os.path.relpath(fpath, SAM_DIR)
                        ).replace('\\', '/')
                        zf.write(fpath, arcname)
                        print(f"  + {arcname}")

        # 2. Pack agevision_api __init__.py
        init_path = os.path.join(BASE, 'agevision_api', '__init__.py')
        if os.path.isfile(init_path):
            zf.write(init_path, 'agevision_api/__init__.py')
            print(f"  + agevision_api/__init__.py")

        # 3. Pack checkpoints (large files — this will take a moment)
        for ckpt_name in CHECKPOINTS:
            ckpt_path = os.path.join(CHECKPOINTS_DIR, ckpt_name)
            if os.path.isfile(ckpt_path):
                size_gb = os.path.getsize(ckpt_path) / 1e9
                print(f"\n  Packing checkpoint: {ckpt_name} ({size_gb:.2f} GB)")
                print(f"  This may take a few minutes...")
                zf.write(ckpt_path, f"checkpoints/{ckpt_name}")
                print(f"  + checkpoints/{ckpt_name}")
            else:
                print(f"\n  WARNING: Checkpoint not found: {ckpt_path}")

    zip_size = os.path.getsize(OUTPUT_ZIP) / 1e9
    print(f"\n{'='*55}")
    print(f"Done! Created: {OUTPUT_ZIP}")
    print(f"Size: {zip_size:.2f} GB")
    print(f"\nNext steps:")
    print(f"  1. Upload colab_sam_training.zip to Google Drive")
    print(f"     (drag & drop to drive.google.com)")
    print(f"  2. Open Google Colab: colab.research.google.com")
    print(f"  3. File → Upload Notebook → choose train_colab.ipynb")
    print(f"     (or upload the notebook from the zip)")
    print(f"  4. Runtime → Change runtime type → GPU (T4)")
    print(f"  5. Run all cells!")
    print(f"{'='*55}")


if __name__ == '__main__':
    pack()
