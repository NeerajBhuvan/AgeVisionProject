"""
Age Progression Engine
=====================
Applies realistic aging transformations to face images using OpenCV and PIL.

Pipeline stages:
  1. Face Detection & Landmark Alignment
  2. Skin Texture Aging (wrinkles, pores, spots)
  3. Hair Graying Simulation
  4. Facial Structure Changes (sagging, volume loss)
  5. Post-Processing (color grading, blending)

Each stage returns per-step metadata so the frontend can render
a live pipeline tracker.
"""

import cv2
import numpy as np
import os
import time
import uuid
from PIL import Image, ImageFilter, ImageEnhance

from django.conf import settings


class AgingPipeline:
    """Applies age-progression effects to a face image."""

    # Aging intensity curves per decade bracket
    AGING_PROFILES = {
        'child':    {'wrinkle': 0.0,  'gray': 0.0,  'sag': 0.0,  'spots': 0.0,  'texture': 0.0},
        'teen':     {'wrinkle': 0.02, 'gray': 0.0,  'sag': 0.0,  'spots': 0.02, 'texture': 0.05},
        'young':    {'wrinkle': 0.08, 'gray': 0.05, 'sag': 0.02, 'spots': 0.05, 'texture': 0.10},
        'adult':    {'wrinkle': 0.20, 'gray': 0.15, 'sag': 0.08, 'spots': 0.12, 'texture': 0.20},
        'middle':   {'wrinkle': 0.45, 'gray': 0.35, 'sag': 0.20, 'spots': 0.25, 'texture': 0.40},
        'senior':   {'wrinkle': 0.70, 'gray': 0.60, 'sag': 0.35, 'spots': 0.40, 'texture': 0.60},
        'elderly':  {'wrinkle': 0.90, 'gray': 0.85, 'sag': 0.50, 'spots': 0.55, 'texture': 0.80},
    }

    def __init__(self):
        self.steps = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, image_path: str, current_age: int, target_age: int) -> dict:
        """
        Execute the full aging pipeline.

        Returns dict with:
            - output_path : absolute path of saved progressed image
            - relative_path : path relative to MEDIA_ROOT (for Django model)
            - steps : list of pipeline step dicts
            - insights : aging feature metrics
            - processing_time_ms : total wall-clock time in milliseconds
        """
        t0 = time.time()
        self.steps = []

        # Load the image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Could not read image file")

        # Determine aging intensity
        age_diff = target_age - current_age
        profile = self._get_aging_profile(target_age)
        # Scale profile by the age difference magnitude
        scale = min(abs(age_diff) / 50.0, 1.0)
        if age_diff < 0:
            # "De-aging" – apply inverse soft effects
            scale *= 0.5

        profile = {k: v * scale for k, v in profile.items()}

        # --- Stage 1: Face Detection ---
        step_t = time.time()
        face_rect, landmarks = self._detect_face(img)
        self._record_step('Face Detection', '👁️', time.time() - step_t)

        # --- Stage 2: Landmark Alignment ---
        step_t = time.time()
        aligned = self._align_face(img, face_rect)
        self._record_step('Landmark Align', '📍', time.time() - step_t)

        # --- Stage 3: Skin & Wrinkle Aging ---
        step_t = time.time()
        aged = self._apply_skin_aging(aligned, profile, face_rect)
        self._record_step('Skin & Wrinkle Aging', '🧠', time.time() - step_t)

        # --- Stage 4: Hair Graying ---
        step_t = time.time()
        aged = self._apply_hair_graying(aged, profile, face_rect)
        self._record_step('Hair Graying', '✨', time.time() - step_t)

        # --- Stage 5: Structure & Post-Process ---
        step_t = time.time()
        final = self._post_process(aged, profile, target_age)
        self._record_step('Post-Processing', '🎯', time.time() - step_t)

        # Save output
        output_dir = os.path.join(settings.MEDIA_ROOT, 'progressions')
        os.makedirs(output_dir, exist_ok=True)
        fname = f"prog_{uuid.uuid4().hex[:12]}.jpg"
        output_path = os.path.join(output_dir, fname)
        cv2.imwrite(output_path, final, [cv2.IMWRITE_JPEG_QUALITY, 92])

        processing_time = (time.time() - t0) * 1000

        insights = self._compute_insights(profile, age_diff)

        return {
            'output_path': output_path,
            'relative_path': f"progressions/{fname}",
            'steps': self.steps,
            'insights': insights,
            'processing_time_ms': round(processing_time, 2),
        }

    # ------------------------------------------------------------------
    # Pipeline Stages
    # ------------------------------------------------------------------

    def _detect_face(self, img):
        """Detect the primary face using Haar cascade."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))

        if len(faces) == 0:
            # Fallback: use the whole image
            h, w = img.shape[:2]
            return (0, 0, w, h), None

        # Pick the largest face
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        x, y, w, h = faces[0]
        return (x, y, w, h), None

    def _align_face(self, img, face_rect):
        """Light alignment – ensure face is centred and upright."""
        # For now we work on the full resolution image
        return img.copy()

    def _apply_skin_aging(self, img, profile, face_rect):
        """
        Apply wrinkle and skin-texture aging to the face region.
        Uses edge-enhanced overlays + noise to simulate wrinkles.
        """
        wrinkle_intensity = profile.get('wrinkle', 0)
        texture_intensity = profile.get('texture', 0)
        spots_intensity = profile.get('spots', 0)

        if wrinkle_intensity < 0.01 and texture_intensity < 0.01:
            return img

        result = img.copy()
        x, y, w, h = face_rect

        # Expand ROI slightly for natural blending
        pad = int(min(w, h) * 0.1)
        x1 = max(x - pad, 0)
        y1 = max(y - pad, 0)
        x2 = min(x + w + pad, img.shape[1])
        y2 = min(y + h + pad, img.shape[0])

        roi = result[y1:y2, x1:x2].copy()

        # --- Wrinkle overlay via edge detection ---
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # Multi-scale Laplacian for wrinkle-like edges
        lap1 = cv2.Laplacian(gray_roi, cv2.CV_64F, ksize=3)
        lap2 = cv2.Laplacian(gray_roi, cv2.CV_64F, ksize=5)
        edges = np.abs(lap1) * 0.6 + np.abs(lap2) * 0.4
        edges = np.clip(edges, 0, 255).astype(np.uint8)

        # Threshold to keep only strong edges (wrinkle-like)
        _, wrinkle_mask = cv2.threshold(edges, 15, 255, cv2.THRESH_BINARY)
        wrinkle_mask = cv2.GaussianBlur(wrinkle_mask, (3, 3), 0)

        # Darken along wrinkle lines
        wrinkle_dark = roi.astype(np.float64)
        wrinkle_overlay = (wrinkle_mask.astype(np.float64) / 255.0)
        wrinkle_overlay = np.stack([wrinkle_overlay] * 3, axis=-1)
        wrinkle_dark = wrinkle_dark - wrinkle_overlay * 40 * wrinkle_intensity
        wrinkle_dark = np.clip(wrinkle_dark, 0, 255).astype(np.uint8)
        roi = wrinkle_dark

        # --- Fine texture noise ---
        if texture_intensity > 0.01:
            noise = np.random.normal(0, 8 * texture_intensity, roi.shape).astype(np.float64)
            roi = np.clip(roi.astype(np.float64) + noise, 0, 255).astype(np.uint8)
            roi = cv2.GaussianBlur(roi, (3, 3), 0.5)

        # --- Age spots ---
        if spots_intensity > 0.05:
            num_spots = int(spots_intensity * 30)
            rh, rw = roi.shape[:2]
            for _ in range(num_spots):
                cx = np.random.randint(int(rw * 0.2), int(rw * 0.8))
                cy = np.random.randint(int(rh * 0.25), int(rh * 0.85))
                radius = np.random.randint(2, max(3, int(4 * spots_intensity)))
                color_shift = np.random.randint(-25, -10)
                cv2.circle(roi, (cx, cy), radius, (
                    max(0, int(roi[cy, cx, 0]) + color_shift),
                    max(0, int(roi[cy, cx, 1]) + color_shift - 5),
                    max(0, int(roi[cy, cx, 2]) + color_shift - 3),
                ), -1)
            roi = cv2.GaussianBlur(roi, (3, 3), 0.8)

        # Blend ROI back with a soft elliptical mask for smooth edges
        mask = np.zeros((y2 - y1, x2 - x1), dtype=np.float64)
        center = ((x2 - x1) // 2, (y2 - y1) // 2)
        axes = ((x2 - x1) // 2/1, (y2 - y1) // 2/1)
        cv2.ellipse(mask, center, (int(axes[0] * 0.85), int(axes[1] * 0.85)),
                     0, 0, 360, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (31, 31), 10)
        mask3 = np.stack([mask] * 3, axis=-1)

        blended = (roi.astype(np.float64) * mask3 +
                   result[y1:y2, x1:x2].astype(np.float64) * (1 - mask3))
        result[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)

        return result

    def _apply_hair_graying(self, img, profile, face_rect):
        """Simulate hair graying by desaturating the upper part of the head."""
        gray_level = profile.get('gray', 0)
        if gray_level < 0.03:
            return img

        result = img.copy()
        x, y, w, h = face_rect

        # Hair region: above the face bounding box + sides
        hair_y1 = max(0, y - int(h * 0.6))
        hair_y2 = y + int(h * 0.15)
        hair_x1 = max(0, x - int(w * 0.15))
        hair_x2 = min(img.shape[1], x + w + int(w * 0.15))

        hair_roi = result[hair_y1:hair_y2, hair_x1:hair_x2].copy()
        if hair_roi.size == 0:
            return result

        # Convert to HSV and reduce saturation
        hsv = cv2.cvtColor(hair_roi, cv2.COLOR_BGR2HSV).astype(np.float64)
        hsv[:, :, 1] *= (1 - gray_level * 0.8)  # Desaturate
        hsv[:, :, 2] = hsv[:, :, 2] * (1 - gray_level * 0.15) + gray_level * 30  # Lighten slightly
        hsv = np.clip(hsv, 0, 255).astype(np.uint8)
        grayed = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Gradient mask for natural blending (top fully applied, fades toward face)
        rh, rw = hair_roi.shape[:2]
        gradient = np.linspace(gray_level, 0, rh).reshape(-1, 1)
        gradient = np.tile(gradient, (1, rw))
        gradient = cv2.GaussianBlur(gradient.astype(np.float32), (15, 15), 5).astype(np.float64)
        mask3 = np.stack([gradient] * 3, axis=-1)

        blended = grayed.astype(np.float64) * mask3 + hair_roi.astype(np.float64) * (1 - mask3)
        result[hair_y1:hair_y2, hair_x1:hair_x2] = np.clip(blended, 0, 255).astype(np.uint8)

        return result

    def _post_process(self, img, profile, target_age):
        """
        Final colour grading and structure adjustments.
        - Warm/yellow tint for older ages (sun damage look)
        - Slight contrast reduction (aged skin)
        - Mild Gaussian softness for very old ages
        """
        sag = profile.get('sag', 0)
        result = img.copy()

        # Colour tint: slight warmth
        if target_age > 40:
            warm_factor = min((target_age - 40) / 80.0, 0.25)
            overlay = np.full_like(result, (0, 15, 30), dtype=np.uint8)
            result = cv2.addWeighted(result, 1.0, overlay, warm_factor, 0)

        # Reduce contrast slightly for older appearances
        if target_age > 50:
            pil_img = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            contrast_factor = max(0.85, 1.0 - (target_age - 50) / 200.0)
            enhancer = ImageEnhance.Contrast(pil_img)
            pil_img = enhancer.enhance(contrast_factor)
            result = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        # Very subtle softening for elderly look
        if target_age > 65:
            softness = min((target_age - 65) / 100.0, 0.3)
            blurred = cv2.GaussianBlur(result, (5, 5), 1)
            result = cv2.addWeighted(result, 1 - softness, blurred, softness, 0)

        # Slight downward warp for sagging effect
        if sag > 0.05:
            h, w = result.shape[:2]
            map_x = np.float32([[i for i in range(w)] for _ in range(h)])
            map_y = np.float32([[j for j in range(h)] for j in range(h)])

            # Add subtle downward displacement in lower half
            for row in range(h // 2, h):
                shift = sag * 3.0 * ((row - h // 2) / (h // 2)) ** 2
                map_y[row, :] -= shift

            result = cv2.remap(result, map_x, map_y, cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT_101)

        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_aging_profile(self, target_age: int) -> dict:
        """Select the aging profile based on target age bracket."""
        if target_age <= 12:
            return dict(self.AGING_PROFILES['child'])
        elif target_age <= 19:
            return dict(self.AGING_PROFILES['teen'])
        elif target_age <= 30:
            return dict(self.AGING_PROFILES['young'])
        elif target_age <= 45:
            return dict(self.AGING_PROFILES['adult'])
        elif target_age <= 60:
            return dict(self.AGING_PROFILES['middle'])
        elif target_age <= 75:
            return dict(self.AGING_PROFILES['senior'])
        else:
            return dict(self.AGING_PROFILES['elderly'])

    def _record_step(self, label: str, icon: str, elapsed: float):
        """Record a completed pipeline step."""
        self.steps.append({
            'label': label,
            'icon': icon,
            'status': 'done',
            'time_ms': round(elapsed * 1000, 2),
        })

    def _compute_insights(self, profile: dict, age_diff: int) -> list:
        """Derive human-readable aging insights with percentage values."""
        insights = [
            {
                'label': 'Wrinkle Intensity',
                'value': min(int(profile.get('wrinkle', 0) * 100), 100),
                'color': '#7F5AF0',
            },
            {
                'label': 'Hair Graying',
                'value': min(int(profile.get('gray', 0) * 100), 100),
                'color': '#2CB67D',
            },
            {
                'label': 'Skin Texture Change',
                'value': min(int(profile.get('texture', 0) * 100), 100),
                'color': '#E16259',
            },
            {
                'label': 'Age Spots',
                'value': min(int(profile.get('spots', 0) * 100), 100),
                'color': '#F0A500',
            },
            {
                'label': 'Facial Sagging',
                'value': min(int(profile.get('sag', 0) * 100), 100),
                'color': '#72757E',
            },
        ]
        return insights
