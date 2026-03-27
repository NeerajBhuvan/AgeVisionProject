"""
Image Processor
===============
Helper utilities for image pre/post-processing, comparison views,
and texture helpers.
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import uuid
import logging

from django.conf import settings

logger = logging.getLogger(__name__)

# Default output size
FACE_SIZE = (256, 256)


class ImageProcessor:
    """Static helper methods for image processing tasks."""

    # ------------------------------------------------------------------
    #  Pre-processing
    # ------------------------------------------------------------------

    @staticmethod
    def preprocess_for_model(image: np.ndarray, size: tuple = FACE_SIZE) -> np.ndarray:
        """
        Resize and normalize an image for model input.

        Returns
        -------
        np.ndarray  – resized image in BGR uint8 format (256×256 by default).
        """
        resized = cv2.resize(image, size, interpolation=cv2.INTER_LANCZOS4)
        return resized

    @staticmethod
    def normalize_tensor(image: np.ndarray) -> np.ndarray:
        """Normalize to [0, 1] float32 (for model consumption)."""
        return image.astype(np.float32) / 255.0

    @staticmethod
    def denormalize_tensor(tensor: np.ndarray) -> np.ndarray:
        """Convert [0,1] float32 back to uint8."""
        return np.clip(tensor * 255.0, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------
    #  Comparison / output views
    # ------------------------------------------------------------------

    @staticmethod
    def create_comparison_view(
        original: np.ndarray,
        progressed: np.ndarray,
        current_age: int,
        target_age: int,
        padding: int = 20,
        label_height: int = 40,
        bg_colour: tuple = (30, 30, 30),
    ) -> np.ndarray:
        """
        Create a side-by-side comparison image with labels.

        Parameters
        ----------
        original, progressed : np.ndarray  – BGR images.
        current_age, target_age : int
        padding : int – pixel gap between and around images.
        label_height : int – space above each image for text labels.
        bg_colour : tuple – BGR background fill colour.

        Returns
        -------
        np.ndarray – the composite BGR image.
        """
        # Ensure both images are the same height
        h1, w1 = original.shape[:2]
        h2, w2 = progressed.shape[:2]
        target_h = max(h1, h2, 256)

        if h1 != target_h:
            scale = target_h / h1
            original = cv2.resize(original, (int(w1 * scale), target_h))
            w1 = original.shape[1]
        if h2 != target_h:
            scale = target_h / h2
            progressed = cv2.resize(progressed, (int(w2 * scale), target_h))
            w2 = progressed.shape[1]

        canvas_w = padding + w1 + padding + w2 + padding
        canvas_h = padding + label_height + target_h + padding

        canvas = np.full((canvas_h, canvas_w, 3), bg_colour, dtype=np.uint8)

        # Place images
        y_off = padding + label_height
        x1_off = padding
        x2_off = padding + w1 + padding

        canvas[y_off:y_off + target_h, x1_off:x1_off + w1] = original
        canvas[y_off:y_off + target_h, x2_off:x2_off + w2] = progressed

        # Add labels using OpenCV (no PIL font dependency)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        white = (255, 255, 255)

        label_orig = f'Original (Age {current_age})'
        label_prog = f'Aged (Age {target_age})'

        cv2.putText(
            canvas, label_orig,
            (x1_off, padding + label_height - 12),
            font, font_scale, white, thickness,
        )
        cv2.putText(
            canvas, label_prog,
            (x2_off, padding + label_height - 12),
            font, font_scale, white, thickness,
        )

        # Draw thin border around each image
        cv2.rectangle(
            canvas,
            (x1_off - 1, y_off - 1),
            (x1_off + w1, y_off + target_h),
            (100, 100, 100), 1,
        )
        cv2.rectangle(
            canvas,
            (x2_off - 1, y_off - 1),
            (x2_off + w2, y_off + target_h),
            (100, 100, 100), 1,
        )

        return canvas

    # ------------------------------------------------------------------
    #  Save helpers
    # ------------------------------------------------------------------

    @staticmethod
    def save_image(image: np.ndarray, subfolder: str, prefix: str = '') -> dict:
        """
        Save a BGR image to ``MEDIA_ROOT/<subfolder>/`` with a unique name.

        Returns
        -------
        dict with keys:
            absolute_path : str
            relative_path : str  (relative to MEDIA_ROOT, for URL generation)
            filename      : str
        """
        folder = os.path.join(settings.MEDIA_ROOT, subfolder)
        os.makedirs(folder, exist_ok=True)

        filename = f'{prefix}{uuid.uuid4().hex[:12]}.jpg'
        abs_path = os.path.join(folder, filename)
        rel_path = os.path.join(subfolder, filename).replace('\\', '/')

        cv2.imwrite(abs_path, image, [cv2.IMWRITE_JPEG_QUALITY, 92])
        return {
            'absolute_path': abs_path,
            'relative_path': rel_path,
            'filename': filename,
        }

    # ------------------------------------------------------------------
    #  Texture helpers
    # ------------------------------------------------------------------

    @staticmethod
    def add_aging_texture(image: np.ndarray, strength: float = 0.3) -> np.ndarray:
        """
        Generic wrinkle/texture overlay helper.

        Uses difference-of-Gaussians to extract skin texture and
        amplify it, producing a subtle aged-skin effect.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        g1 = cv2.GaussianBlur(gray, (3, 3), 0)
        g2 = cv2.GaussianBlur(gray, (9, 9), 0)
        detail = cv2.subtract(g1, g2)

        detail_3ch = cv2.merge([detail] * 3).astype(np.float32) / 255.0
        result = image.astype(np.float32) - detail_3ch * strength * 180
        return np.clip(result, 0, 255).astype(np.uint8)

    @staticmethod
    def read_image(path: str) -> np.ndarray:
        """Read an image from disk, raising ValueError on failure."""
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f'Could not read image: {path}')
        return img
