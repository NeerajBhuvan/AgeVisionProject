#!/usr/bin/env python
"""
Download High-Quality Diverse Test Images
===========================================
Curates diverse real face images from the LFW dataset + web sources.
Includes Indian, Asian, African, European, Latin American, Middle Eastern faces.

LFW Source: http://vis-www.cs.umass.edu/lfw/
"""

import logging
import os
import sys
import cv2
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("download_samples")

IMAGES_DIR = Path(__file__).resolve().parent / "images"

# ─── Diverse selection: (filename, LFW_person, image_index, ethnicity, gender) ───
DIVERSE_FACES = [
    # Indian / South Asian
    ("indian_male_1", "Atal_Bihari_Vajpayee", 0, "Indian", "Male"),
    ("indian_female_1", "Sonia_Gandhi", 0, "Indian", "Female"),
    ("indian_male_2", "Sachin_Tendulkar", 0, "Indian", "Male"),
    ("indian_female_2", "Aishwarya_Rai", 0, "Indian", "Female"),
    # Pakistani / South Asian
    ("pakistani_male", "Pervez_Musharraf", 0, "Pakistani", "Male"),
    ("pakistani_female", "Benazir_Bhutto", 0, "Pakistani", "Female"),
    # East Asian
    ("east_asian_male_1", "Junichiro_Koizumi", 5, "Japanese", "Male"),
    ("east_asian_male_2", "Hu_Jintao", 0, "Chinese", "Male"),
    ("east_asian_female", "Lucy_Liu", 0, "Chinese-American", "Female"),
    ("korean_male", "Roh_Moo-hyun", 0, "Korean", "Male"),
    # African / African American
    ("african_male", "Kofi_Annan", 2, "Ghanaian", "Male"),
    ("african_american_male", "Colin_Powell", 10, "African-American", "Male"),
    ("african_american_female_1", "Serena_Williams", 3, "African-American", "Female"),
    ("african_american_female_2", "Condoleezza_Rice", 0, "African-American", "Female"),
    ("african_american_female_3", "Halle_Berry", 0, "African-American", "Female"),
    # European
    ("european_male", "Tony_Blair", 5, "British", "Male"),
    ("european_male_2", "Jacques_Chirac", 3, "French", "Male"),
    ("european_female", "Gerhard_Schroeder", 2, "German", "Male"),
    # Latin American
    ("latin_american_male_1", "Hugo_Chavez", 3, "Venezuelan", "Male"),
    ("latin_american_male_2", "Luiz_Inacio_Lula_da_Silva", 0, "Brazilian", "Male"),
    ("latin_american_male_3", "Vicente_Fox", 0, "Mexican", "Male"),
    # Middle Eastern / Central Asian
    ("middle_eastern_male", "Abdullah_Gul", 0, "Turkish", "Male"),
    ("afghan_male", "Hamid_Karzai", 0, "Afghan", "Male"),
    # Southeast Asian
    ("southeast_asian_female", "Gloria_Macapagal_Arroyo", 2, "Filipino", "Female"),
    ("southeast_asian_male", "Mahathir_Mohamad", 0, "Malaysian", "Male"),
]


def get_lfw_dir() -> Path:
    """Locate the LFW dataset directory from scikit-learn's data cache."""
    try:
        from sklearn.datasets import get_data_home
        lfw_dir = Path(get_data_home()) / "lfw_home" / "lfw_funneled"
        if lfw_dir.exists():
            return lfw_dir
    except ImportError:
        pass
    return None


def download_lfw_if_needed() -> Path:
    """Download LFW dataset via scikit-learn if not already cached."""
    lfw_dir = get_lfw_dir()
    if lfw_dir and lfw_dir.exists():
        logger.info("LFW dataset cached at: %s", lfw_dir)
        return lfw_dir

    logger.info("Downloading LFW dataset (~200MB)...")
    try:
        from sklearn.datasets import fetch_lfw_people
        fetch_lfw_people(min_faces_per_person=10, resize=1.0)
        return get_lfw_dir()
    except Exception as e:
        logger.error("Failed to download LFW: %s", e)
        return None


def upscale_face(img: np.ndarray, target_size: int = 512) -> np.ndarray:
    """Upscale face image with quality enhancement.

    Pipeline: Lanczos upscale -> sharpen -> bilateral denoise -> CLAHE.
    """
    # Lanczos upscale
    upscaled = cv2.resize(img, (target_size, target_size),
                           interpolation=cv2.INTER_LANCZOS4)

    # Mild sharpening
    kernel = np.array([[0, -0.5, 0], [-0.5, 3.0, -0.5], [0, -0.5, 0]])
    sharpened = cv2.filter2D(upscaled, -1, kernel)
    result = cv2.addWeighted(sharpened, 0.55, upscaled, 0.45, 0)

    # Bilateral denoise (edge-preserving)
    result = cv2.bilateralFilter(result, 5, 40, 40)

    # CLAHE on luminance channel
    lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    result = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    return result


def download_high_res_web_images() -> dict:
    """Try to download higher resolution face images from the web.

    Returns dict of {filename: image_bgr} for successfully downloaded images.
    Falls back gracefully if network is unavailable.
    """
    downloaded = {}
    try:
        import requests
    except ImportError:
        return downloaded

    # Pexels free stock photos (direct image URLs, no auth needed)
    web_sources = {
        "indian_male_hq": "https://images.pexels.com/photos/2379004/pexels-photo-2379004.jpeg?auto=compress&cs=tinysrgb&w=600",
        "indian_female_hq": "https://images.pexels.com/photos/3586798/pexels-photo-3586798.jpeg?auto=compress&cs=tinysrgb&w=600",
        "african_female_hq": "https://images.pexels.com/photos/1239291/pexels-photo-1239291.jpeg?auto=compress&cs=tinysrgb&w=600",
        "asian_female_hq": "https://images.pexels.com/photos/1587009/pexels-photo-1587009.jpeg?auto=compress&cs=tinysrgb&w=600",
        "european_male_hq": "https://images.pexels.com/photos/220453/pexels-photo-220453.jpeg?auto=compress&cs=tinysrgb&w=600",
        "latin_female_hq": "https://images.pexels.com/photos/774909/pexels-photo-774909.jpeg?auto=compress&cs=tinysrgb&w=600",
        "middle_eastern_hq": "https://images.pexels.com/photos/2182970/pexels-photo-2182970.jpeg?auto=compress&cs=tinysrgb&w=600",
    }

    for name, url in web_sources.items():
        try:
            resp = requests.get(url, timeout=15, headers={
                "User-Agent": "Mozilla/5.0 (AgeVision Research)"
            })
            if resp.status_code == 200 and len(resp.content) > 5000:
                arr = np.frombuffer(resp.content, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is not None and img.shape[0] >= 200 and img.shape[1] >= 200:
                    downloaded[name] = img
                    logger.info("  Web OK: %-25s %dx%d", name, img.shape[1], img.shape[0])
                else:
                    logger.warning("  Web SKIP: %s (invalid image)", name)
            else:
                logger.warning("  Web FAIL: %s (HTTP %d)", name, resp.status_code)
        except Exception as e:
            logger.warning("  Web FAIL: %s (%s)", name, e)

    return downloaded


def crop_face_from_image(img: np.ndarray, target_size: int = 512) -> np.ndarray:
    """Detect and crop the primary face, then resize to target_size."""
    proto = str(Path(__file__).parent / "agevision_backend" / "age_progression" /
                "trained_models" / "deploy.prototxt")
    model_path = str(Path(__file__).parent / "agevision_backend" / "age_progression" /
                     "trained_models" / "res10_300x300_ssd_iter_140000.caffemodel")

    face_rect = None
    h, w = img.shape[:2]

    if os.path.exists(proto) and os.path.exists(model_path):
        net = cv2.dnn.readNetFromCaffe(proto, model_path)
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,
                                      (300, 300), (104, 177, 123))
        net.setInput(blob)
        dets = net.forward()
        for i in range(dets.shape[2]):
            if dets[0, 0, i, 2] > 0.5:
                box = dets[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype(int)
                face_rect = (x1, y1, x2 - x1, y2 - y1)
                break

    if face_rect is None:
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))
        if len(faces) > 0:
            face_rect = tuple(sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0])

    if face_rect is None:
        s = min(h, w)
        x0, y0 = (w - s) // 2, (h - s) // 2
        crop = img[y0:y0+s, x0:x0+s]
    else:
        x, y, fw, fh = face_rect
        pad = int(max(fw, fh) * 0.5)
        x1 = max(0, x - pad)
        y1 = max(0, y - int(pad * 1.3))
        x2 = min(w, x + fw + pad)
        y2 = min(h, y + fh + int(pad * 0.6))
        cw, ch = x2 - x1, y2 - y1
        side = max(cw, ch)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        x1 = max(0, cx - side // 2)
        y1 = max(0, cy - side // 2)
        x2 = min(w, x1 + side)
        y2 = min(h, y1 + side)
        crop = img[y1:y2, x1:x2]

    resized = cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)

    # CLAHE enhancement
    lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def create_test_images():
    """Create high-quality diverse test images."""
    lfw_dir = download_lfw_if_needed()
    if lfw_dir is None:
        logger.error("Cannot access LFW dataset")
        return []

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    created = []

    # ── Part 1: LFW images (upscaled) ──
    logger.info("")
    logger.info("── LFW Dataset Images ──")
    for filename, person, idx, ethnicity, gender in DIVERSE_FACES:
        out_path = IMAGES_DIR / f"{filename}.jpg"
        person_dir = lfw_dir / person
        if not person_dir.exists():
            logger.warning("  NOT FOUND: %s", person)
            continue

        images = sorted(person_dir.glob("*.jpg"))
        if not images:
            continue

        src = images[min(idx, len(images) - 1)]
        img = cv2.imread(str(src))
        if img is None:
            continue

        img_up = upscale_face(img, 512)
        cv2.imwrite(str(out_path), img_up, [cv2.IMWRITE_JPEG_QUALITY, 95])
        logger.info("  %-32s %-25s %-15s %s", out_path.name, person, ethnicity, gender)
        created.append(str(out_path))

    # ── Part 2: Web high-resolution images ──
    logger.info("")
    logger.info("── Web High-Resolution Images ──")
    web_images = download_high_res_web_images()
    for name, img in web_images.items():
        out_path = IMAGES_DIR / f"{name}.jpg"
        face_img = crop_face_from_image(img, 512)
        cv2.imwrite(str(out_path), face_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        logger.info("  %-32s HIGH-RES web download", out_path.name)
        created.append(str(out_path))

    return created


def verify_all():
    """Verify all images: readable, face-detectable, proper size."""
    if not IMAGES_DIR.exists():
        return

    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    logger.info("")
    logger.info("── Verification ──")
    total, face_ok = 0, 0
    for img_path in sorted(IMAGES_DIR.glob("*.jpg")):
        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning("  FAIL %-35s UNREADABLE", img_path.name)
            continue

        total += 1
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
        if len(faces) > 0:
            face_ok += 1
            tag = f"OK ({len(faces)} face)"
        else:
            tag = "NO FACE (fallback will handle)"
        logger.info("  %-35s %4dx%-4d %s", img_path.name, w, h, tag)

    logger.info("")
    logger.info("Total: %d images | Faces detected: %d/%d (%.0f%%)",
                 total, face_ok, total, face_ok / max(total, 1) * 100)


if __name__ == "__main__":
    logger.info("=" * 65)
    logger.info("  High-Quality Diverse Face Images Setup")
    logger.info("=" * 65)

    # Clear old images
    if IMAGES_DIR.exists():
        for old in IMAGES_DIR.glob("*.jpg"):
            old.unlink()

    created = create_test_images()
    verify_all()

    logger.info("")
    logger.info("Images ready in: %s", IMAGES_DIR)
    logger.info("Total: %d images", len(created))
