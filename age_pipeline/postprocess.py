"""
Post-Processing Pipeline
=========================
Blends aged face back into original image, applies color correction,
skin tone preservation, and optional enhancement effects.
"""

import logging
import cv2
import numpy as np

logger = logging.getLogger("age_pipeline.postprocess")


def paste_back(original_bgr: np.ndarray, aged_crop: np.ndarray,
               crop_coords: tuple) -> np.ndarray:
    """Paste an aged face crop back into the original image with soft blending.

    Args:
        original_bgr: Original full image (BGR).
        aged_crop: Aged face crop (256x256 BGR).
        crop_coords: (x1, y1, x2, y2) from the crop operation.

    Returns:
        Blended output image (BGR).
    """
    x1, y1, x2, y2 = crop_coords
    crop_w, crop_h = x2 - x1, y2 - y1

    result = original_bgr.copy()
    aged_resized = cv2.resize(aged_crop, (crop_w, crop_h),
                               interpolation=cv2.INTER_LANCZOS4)

    # Soft elliptical mask for seamless blending
    mask = np.zeros((crop_h, crop_w), dtype=np.float32)
    cv2.ellipse(mask,
                (crop_w // 2, crop_h // 2),
                (int(crop_w * 0.42), int(crop_h * 0.42)),
                0, 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (31, 31), 10)
    mask3 = np.stack([mask] * 3, axis=-1)

    # Blend
    roi = result[y1:y2, x1:x2].astype(np.float32)
    blended = aged_resized.astype(np.float32) * mask3 + roi * (1 - mask3)
    result[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)

    return result


def color_correct(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Color transfer in LAB space to match reference image tones."""
    src_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB).astype(np.float32)

    for ch in range(3):
        src_mean = src_lab[:, :, ch].mean()
        src_std = src_lab[:, :, ch].std() + 1e-6
        ref_mean = ref_lab[:, :, ch].mean()
        ref_std = ref_lab[:, :, ch].std() + 1e-6
        src_lab[:, :, ch] = ((src_lab[:, :, ch] - src_mean) *
                              (ref_std / src_std) + ref_mean)

    src_lab = np.clip(src_lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(src_lab, cv2.COLOR_LAB2BGR)


def enhance_aging_effects(image: np.ndarray, current_age: int,
                           target_age: int) -> np.ndarray:
    """Apply additional visual aging cues on top of GAN output.

    Adds wrinkle enhancement, skin texture changes, hair graying,
    and contrast adjustment scaled by age difference.
    """
    age_diff = target_age - current_age
    if age_diff <= 5:
        return image

    intensity = min(abs(age_diff) / 40.0, 1.0)
    result = image.copy()

    # Wrinkle enhancement via Laplacian edge darkening
    if intensity > 0.15 and age_diff > 0:
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        lap = np.abs(cv2.Laplacian(gray, cv2.CV_64F))
        if lap.max() > 0:
            lap = (lap / lap.max() * 255).astype(np.uint8)
        else:
            lap = lap.astype(np.uint8)
        wrinkle_mask = cv2.GaussianBlur(lap, (3, 3), 0)
        wrinkle_layer = np.stack([wrinkle_mask] * 3, axis=-1).astype(np.float32)
        result = result.astype(np.float32) - wrinkle_layer * intensity * 0.4
        result = np.clip(result, 0, 255).astype(np.uint8)

    # Skin desaturation for aging
    if intensity > 0.2 and age_diff > 0:
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] *= max(1.0 - intensity * 0.18, 0.65)
        hsv = np.clip(hsv, 0, [179, 255, 255]).astype(np.uint8)
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Hair graying (dark region desaturation)
    if intensity > 0.3 and age_diff > 0:
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
        dark_mask = (hsv[:, :, 2] < 120).astype(np.float32)
        dark_mask = cv2.GaussianBlur(dark_mask, (11, 11), 3)
        hsv[:, :, 1] *= (1.0 - dark_mask * intensity * 0.6)
        hsv[:, :, 2] += dark_mask * intensity * 18
        hsv = np.clip(hsv, 0, [179, 255, 255]).astype(np.uint8)
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Slight contrast reduction for elderly look
    if intensity > 0.25 and age_diff > 0:
        mean_val = result.mean()
        alpha = max(1.0 - intensity * 0.07, 0.86)
        result = cv2.convertScaleAbs(result, alpha=alpha,
                                      beta=mean_val * (1 - alpha))

    # De-aging: increase saturation and contrast slightly
    if age_diff < 0:
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] *= min(1.0 + intensity * 0.15, 1.3)
        hsv = np.clip(hsv, 0, [179, 255, 255]).astype(np.uint8)
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return result


def create_comparison_grid(original: np.ndarray, aged_images: dict,
                            cell_size: int = 256) -> np.ndarray:
    """Create a side-by-side comparison grid.

    Args:
        original: Original image (BGR).
        aged_images: Dict of {age_label: aged_image_bgr}.
        cell_size: Size of each cell in the grid.

    Returns:
        Grid image (BGR).
    """
    n_images = 1 + len(aged_images)
    # Layout: single row
    grid_w = cell_size * n_images
    grid_h = cell_size + 40  # Extra space for labels

    grid = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 255

    # Draw original
    orig_resized = cv2.resize(original, (cell_size, cell_size),
                               interpolation=cv2.INTER_LANCZOS4)
    grid[0:cell_size, 0:cell_size] = orig_resized
    cv2.putText(grid, "Original", (10, cell_size + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Draw aged images
    for i, (label, img) in enumerate(aged_images.items(), start=1):
        x_offset = i * cell_size
        resized = cv2.resize(img, (cell_size, cell_size),
                              interpolation=cv2.INTER_LANCZOS4)
        grid[0:cell_size, x_offset:x_offset + cell_size] = resized
        cv2.putText(grid, str(label), (x_offset + 10, cell_size + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    return grid
