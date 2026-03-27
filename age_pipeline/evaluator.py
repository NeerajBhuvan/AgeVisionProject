"""
Evaluation & Metrics Pipeline
===============================
Computes identity preservation, image quality metrics,
and age estimation accuracy for age progression outputs.
"""

import logging
import json
import cv2
import numpy as np
from pathlib import Path

logger = logging.getLogger("age_pipeline.evaluator")


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute Structural Similarity Index between two images.

    Uses the Wang et al. (2004) formula with default constants.
    Both images must be the same size.
    """
    # Convert to grayscale float
    if len(img1.shape) == 3:
        g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float64)
    else:
        g1 = img1.astype(np.float64)
    if len(img2.shape) == 3:
        g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(np.float64)
    else:
        g2 = img2.astype(np.float64)

    # Resize to match if needed
    if g1.shape != g2.shape:
        g2 = cv2.resize(g2, (g1.shape[1], g1.shape[0]))

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    mu1 = cv2.GaussianBlur(g1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(g2, (11, 11), 1.5)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(g1 ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(g2 ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(g1 * g2, (11, 11), 1.5) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return float(np.mean(ssim_map))


def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute Peak Signal-to-Noise Ratio between two images."""
    if len(img1.shape) == 3:
        g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float64)
    else:
        g1 = img1.astype(np.float64)
    if len(img2.shape) == 3:
        g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(np.float64)
    else:
        g2 = img2.astype(np.float64)

    if g1.shape != g2.shape:
        g2 = cv2.resize(g2, (g1.shape[1], g1.shape[0]))

    mse = np.mean((g1 - g2) ** 2)
    if mse == 0:
        return float("inf")
    return float(10 * np.log10(255.0 ** 2 / mse))


def compute_identity_score(face1: np.ndarray, face2: np.ndarray) -> float:
    """Compute identity preservation score using feature-based comparison.

    Uses ORB feature matching + histogram correlation as a lightweight
    identity similarity metric (no deep model required).
    Higher is better (0.0 to 1.0 scale).
    """
    # Resize both to 256x256
    size = (256, 256)
    f1 = cv2.resize(face1, size)
    f2 = cv2.resize(face2, size)

    # Method 1: Histogram correlation (color distribution similarity)
    hist_score = 0.0
    for ch in range(3):
        h1 = cv2.calcHist([f1], [ch], None, [64], [0, 256])
        h2 = cv2.calcHist([f2], [ch], None, [64], [0, 256])
        cv2.normalize(h1, h1)
        cv2.normalize(h2, h2)
        hist_score += cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)
    hist_score /= 3.0

    # Method 2: ORB feature matching
    orb = cv2.ORB_create(nfeatures=500)
    g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)
    kp1, des1 = orb.detectAndCompute(g1, None)
    kp2, des2 = orb.detectAndCompute(g2, None)

    feature_score = 0.5  # Default if matching fails
    if des1 is not None and des2 is not None and len(des1) > 10 and len(des2) > 10:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        if matches:
            matches = sorted(matches, key=lambda m: m.distance)
            # Fraction of good matches (distance < 50)
            good = sum(1 for m in matches if m.distance < 50)
            feature_score = min(good / max(len(matches), 1), 1.0)

    # Method 3: Structural similarity on face region
    ssim_score = compute_ssim(f1, f2)

    # Weighted combination
    identity_score = (hist_score * 0.3 + feature_score * 0.3 +
                      ssim_score * 0.4)
    return max(0.0, min(1.0, identity_score))


def compute_identity_score_deep(face1: np.ndarray, face2: np.ndarray) -> float:
    """Compute identity score using FaceNet (InceptionResnetV1) cosine similarity.

    Falls back to feature-based method if facenet-pytorch is unavailable.
    """
    try:
        import torch
        from facenet_pytorch import InceptionResnetV1
        from torchvision import transforms

        model = InceptionResnetV1(pretrained="vggface2").eval()
        preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

        f1_rgb = cv2.cvtColor(cv2.resize(face1, (160, 160)), cv2.COLOR_BGR2RGB)
        f2_rgb = cv2.cvtColor(cv2.resize(face2, (160, 160)), cv2.COLOR_BGR2RGB)

        with torch.no_grad():
            emb1 = model(preprocess(f1_rgb).unsqueeze(0))
            emb2 = model(preprocess(f2_rgb).unsqueeze(0))

        cos_sim = torch.nn.functional.cosine_similarity(emb1, emb2)
        return float(cos_sim.item())
    except ImportError:
        logger.info("facenet-pytorch not available, using feature-based identity score")
        return compute_identity_score(face1, face2)
    except Exception as e:
        logger.warning("FaceNet identity score failed: %s, using fallback", e)
        return compute_identity_score(face1, face2)


def estimate_age_dex(face_bgr: np.ndarray) -> float:
    """Estimate age using a simple heuristic based on image features.

    Uses texture complexity (Laplacian variance) and color statistics
    as age proxies. This is a lightweight estimator — not a deep model.
    """
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (256, 256))

    # Texture complexity (wrinkle proxy)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Skin brightness (older skin tends to be less saturated)
    hsv = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1].mean()

    # Heuristic age estimation
    # High texture + low saturation → older
    age = 20 + (lap_var / 15.0) * 0.5 + (1.0 - saturation / 255.0) * 30
    return max(5.0, min(95.0, age))


def evaluate_progression(original_crop: np.ndarray, aged_crop: np.ndarray,
                          target_age: int, use_deep: bool = True) -> dict:
    """Run full evaluation on a single age progression result.

    Args:
        original_crop: Original face crop (BGR).
        aged_crop: Aged face crop (BGR).
        target_age: The target age used for progression.
        use_deep: Whether to try FaceNet for identity score.

    Returns:
        Dict with all metrics.
    """
    ssim = compute_ssim(original_crop, aged_crop)
    psnr = compute_psnr(original_crop, aged_crop)

    if use_deep:
        identity = compute_identity_score_deep(original_crop, aged_crop)
    else:
        identity = compute_identity_score(original_crop, aged_crop)

    estimated_age = estimate_age_dex(aged_crop)
    age_error = abs(estimated_age - target_age)

    return {
        "ssim": round(ssim, 4),
        "psnr": round(psnr, 2),
        "identity_score": round(identity, 4),
        "estimated_age": round(estimated_age, 1),
        "target_age": target_age,
        "age_error": round(age_error, 1),
        "age_accuracy": round(max(0, 100 - age_error * 2), 1),
    }


def save_metrics(metrics: dict, output_path: str):
    """Save metrics dict to JSON file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info("Metrics saved to %s", output_path)
