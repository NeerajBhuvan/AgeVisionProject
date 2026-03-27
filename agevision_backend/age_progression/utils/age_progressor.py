"""
Age Progressor  –  Hybrid GAN + Image-Processing Aging
======================================================
Applies realistic aging (or de-aging) effects to a face crop.

Strategy
--------
1. **PRIMARY** – Try to load a pre-trained GAN model (HRFAE, StarGAN, etc.)
   from ``age_progression/trained_models/``.
2. **FALLBACK** – If no GAN is available, use an enhanced OpenCV/PIL pipeline
   that produces convincing aging artefacts (wrinkles, spots, graying,
   skin-tone shift, structural sagging).

The fallback is designed to work **out-of-the-box** with zero external
downloads and still produce a clearly visible aging transformation.
"""

import os
import cv2
import numpy as np
import logging
from PIL import Image, ImageFilter, ImageEnhance

logger = logging.getLogger(__name__)

_GAN_MODEL_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'trained_models'
)


class AgeProgressor:
    """
    Hybrid age-progression engine.

    * ``progress_age(face, current_age, target_age)`` → aged BGR image.
    * Automatically selects GAN or image-processing path.
    """

    _gan_model = None
    _gan_checked = False

    # ── Age bracket profiles ─────────────────────────────────────────
    # Each value is a **maximum intensity** for that effect at a given
    # life-stage.  The actual intensity is interpolated from the
    # current-age profile to the target-age profile.

    _PROFILES = {
        (0, 12):   {'wrinkle': 0.00, 'spots': 0.00, 'gray': 0.00, 'sag': 0.00, 'texture': 0.00, 'tone_shift': 0.00},
        (13, 19):  {'wrinkle': 0.02, 'spots': 0.01, 'gray': 0.00, 'sag': 0.00, 'texture': 0.03, 'tone_shift': 0.01},
        (20, 30):  {'wrinkle': 0.06, 'spots': 0.03, 'gray': 0.03, 'sag': 0.02, 'texture': 0.08, 'tone_shift': 0.02},
        (31, 40):  {'wrinkle': 0.18, 'spots': 0.08, 'gray': 0.10, 'sag': 0.06, 'texture': 0.18, 'tone_shift': 0.05},
        (41, 50):  {'wrinkle': 0.35, 'spots': 0.15, 'gray': 0.25, 'sag': 0.14, 'texture': 0.30, 'tone_shift': 0.08},
        (51, 60):  {'wrinkle': 0.55, 'spots': 0.28, 'gray': 0.45, 'sag': 0.25, 'texture': 0.48, 'tone_shift': 0.12},
        (61, 70):  {'wrinkle': 0.72, 'spots': 0.40, 'gray': 0.65, 'sag': 0.38, 'texture': 0.62, 'tone_shift': 0.16},
        (71, 80):  {'wrinkle': 0.85, 'spots': 0.52, 'gray': 0.80, 'sag': 0.50, 'texture': 0.75, 'tone_shift': 0.20},
        (81, 100): {'wrinkle': 0.95, 'spots': 0.62, 'gray': 0.92, 'sag': 0.60, 'texture': 0.88, 'tone_shift': 0.25},
    }

    # ------------------------------------------------------------------
    #  GAN loader (optional)
    # ------------------------------------------------------------------

    @classmethod
    def _try_load_gan(cls):
        """Attempt to load a pre-trained GAN.  Called once."""
        if cls._gan_checked:
            return cls._gan_model
        cls._gan_checked = True

        # Check for HRFAE checkpoint
        try:
            from agevision_api.hrfae.inference import HRFAEInference
            hrfae = HRFAEInference()
            if hrfae.is_ready:
                cls._gan_model = hrfae
                logger.info('HRFAE GAN model loaded successfully')
                return cls._gan_model
        except Exception as exc:
            logger.info('HRFAE not available (%s) – will use image-processing fallback', exc)

        # Could add StarGAN / Age-cGAN checks here in the future
        cls._gan_model = None
        return None

    @classmethod
    def is_gan_available(cls) -> bool:
        cls._try_load_gan()
        return cls._gan_model is not None

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    def progress_age(
        self,
        face_image: np.ndarray,
        current_age: int,
        target_age: int,
    ) -> dict:
        """
        Age (or de-age) a face image.

        Parameters
        ----------
        face_image : np.ndarray   BGR face crop.
        current_age : int
        target_age : int

        Returns
        -------
        dict with:
            image        : np.ndarray  – the transformed BGR face image
            method_used  : str         – 'gan' | 'deepface_hybrid' | 'image_processing'
            aging_params : dict        – the intensity parameters applied
        """
        gan = self._try_load_gan()

        if gan is not None:
            try:
                result_img = self._progress_with_gan(face_image, current_age, target_age, gan)
                # Enhance GAN output with subtle image-processing for visible cues
                profile = self._interpolated_profile(current_age, target_age)
                # Apply lighter image-processing on top (50 % intensity)
                light_profile = {k: v * 0.5 for k, v in profile.items()}
                result_img = self._apply_aging_pipeline(result_img, light_profile, target_age)
                return {
                    'image': result_img,
                    'method_used': 'deepface_hybrid',
                    'aging_params': profile,
                }
            except Exception as exc:
                logger.warning('GAN inference failed (%s) – falling back to image processing', exc)

        # Fallback: pure image-processing
        profile = self._interpolated_profile(current_age, target_age)
        result_img = self._apply_aging_pipeline(face_image.copy(), profile, target_age)
        return {
            'image': result_img,
            'method_used': 'image_processing',
            'aging_params': profile,
        }

    # ------------------------------------------------------------------
    #  GAN path
    # ------------------------------------------------------------------

    @staticmethod
    def _progress_with_gan(face_image, current_age, target_age, gan):
        """Run HRFAE (or compatible) GAN inference."""
        result = gan.infer(face_image, current_age, target_age)
        if isinstance(result, dict):
            return result.get('output', face_image)
        return result  # numpy array expected

    # ------------------------------------------------------------------
    #  Profile interpolation
    # ------------------------------------------------------------------

    def _interpolated_profile(self, current_age: int, target_age: int) -> dict:
        """Compute per-effect intensity by interpolating between brackets."""
        p_curr = self._profile_for_age(current_age)
        p_tgt = self._profile_for_age(target_age)

        profile = {}
        for key in p_tgt:
            diff = p_tgt[key] - p_curr[key]
            profile[key] = max(diff, 0.0)  # only additive aging

        # Scale by magnitude of age jump
        age_diff = abs(target_age - current_age)
        magnitude = min(age_diff / 40.0, 1.0)
        profile = {k: v * magnitude for k, v in profile.items()}

        return profile

    def _profile_for_age(self, age: int) -> dict:
        for (lo, hi), prof in self._PROFILES.items():
            if lo <= age <= hi:
                return dict(prof)
        # Clamp to elderly
        return dict(list(self._PROFILES.values())[-1])

    # ------------------------------------------------------------------
    #  Image-processing aging pipeline
    # ------------------------------------------------------------------

    def _apply_aging_pipeline(
        self,
        image: np.ndarray,
        profile: dict,
        target_age: int,
    ) -> np.ndarray:
        """
        Apply a multi-stage aging transformation using OpenCV + PIL.

        Stages:
          1. Wrinkle texture overlay
          2. Age spot simulation
          3. Hair / temple graying
          4. Skin tone shift (warm → sallow)
          5. Facial sag simulation
          6. Overall texture roughening
          7. Final colour grading
        """
        img = image.copy()

        # 1. Wrinkles
        if profile.get('wrinkle', 0) > 0.01:
            img = self._add_wrinkles(img, profile['wrinkle'])

        # 2. Age spots
        if profile.get('spots', 0) > 0.01:
            img = self._add_age_spots(img, profile['spots'])

        # 3. Hair graying
        if profile.get('gray', 0) > 0.01:
            img = self._add_hair_graying(img, profile['gray'])

        # 4. Skin tone shift
        if profile.get('tone_shift', 0) > 0.01:
            img = self._shift_skin_tone(img, profile['tone_shift'])

        # 5. Sag
        if profile.get('sag', 0) > 0.01:
            img = self._add_sag(img, profile['sag'])

        # 6. Texture roughening
        if profile.get('texture', 0) > 0.01:
            img = self._roughen_texture(img, profile['texture'])

        # 7. Final colour grading
        img = self._colour_grade(img, target_age)

        return img

    # ── Stage implementations ────────────────────────────────────────

    def _add_wrinkles(self, img: np.ndarray, intensity: float) -> np.ndarray:
        """
        Simulate wrinkles by extracting high-frequency detail,
        amplifying it, and blending back.
        """
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # High-pass via difference-of-Gaussian
        blur1 = cv2.GaussianBlur(gray, (0, 0), 1)
        blur2 = cv2.GaussianBlur(gray, (0, 0), 3)
        detail = cv2.subtract(blur1, blur2)

        # Amplify dark lines (wrinkle-like)
        detail = cv2.normalize(detail, None, 0, 255, cv2.NORM_MINMAX)
        _, wrinkle_mask = cv2.threshold(detail, 30, 255, cv2.THRESH_BINARY)

        # Directional emphasis for forehead / nasolabial lines
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1))
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
        wrinkles_h = cv2.morphologyEx(wrinkle_mask, cv2.MORPH_CLOSE, kernel_h)
        wrinkles_v = cv2.morphologyEx(wrinkle_mask, cv2.MORPH_CLOSE, kernel_v)
        wrinkle_lines = cv2.addWeighted(wrinkles_h, 0.6, wrinkles_v, 0.4, 0)

        # Soften
        wrinkle_lines = cv2.GaussianBlur(wrinkle_lines, (3, 3), 0)

        # Darken where wrinkles are
        wrinkle_overlay = cv2.merge([wrinkle_lines] * 3).astype(np.float32) / 255.0
        alpha = intensity * 0.45
        result = img.astype(np.float32)
        result = result * (1.0 - wrinkle_overlay * alpha)
        return np.clip(result, 0, 255).astype(np.uint8)

    def _add_age_spots(self, img: np.ndarray, intensity: float) -> np.ndarray:
        """Scatter small brownish blotches over the face region."""
        h, w = img.shape[:2]
        result = img.copy()
        np.random.seed(42)  # reproducible

        num_spots = int(intensity * 60)
        for _ in range(num_spots):
            cx = np.random.randint(int(w * 0.15), int(w * 0.85))
            cy = np.random.randint(int(h * 0.10), int(h * 0.85))
            radius = np.random.randint(2, max(3, int(6 * intensity)))
            # Brown-ish colour
            colour = (
                int(40 + np.random.randint(-10, 10)),
                int(70 + np.random.randint(-10, 10)),
                int(110 + np.random.randint(-15, 15)),
            )
            overlay = result.copy()
            cv2.circle(overlay, (cx, cy), radius, colour, -1)
            alpha = 0.15 + intensity * 0.25
            cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0, result)

        return result

    def _add_hair_graying(self, img: np.ndarray, intensity: float) -> np.ndarray:
        """
        Desaturate the upper ~30 % of the image (hair region)
        to simulate greying.
        """
        h, w = img.shape[:2]
        hair_region_h = int(h * 0.35)

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        # Reduce saturation in hair region
        sat_reduction = intensity * 0.85
        hsv[:hair_region_h, :, 1] *= (1.0 - sat_reduction)
        # Lighten slightly
        hsv[:hair_region_h, :, 2] = np.clip(
            hsv[:hair_region_h, :, 2] + intensity * 25, 0, 255
        )

        # Also affect temple regions (sides, top 40 %)
        temple_h = int(h * 0.40)
        temple_w = int(w * 0.20)
        hsv[:temple_h, :temple_w, 1] *= (1.0 - sat_reduction * 0.7)
        hsv[:temple_h, w - temple_w:, 1] *= (1.0 - sat_reduction * 0.7)

        hsv = np.clip(hsv, 0, 255).astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def _shift_skin_tone(self, img: np.ndarray, intensity: float) -> np.ndarray:
        """
        Shift skin tones towards a slightly more yellow/sallow hue
        (characteristic of aged skin losing redness).
        """
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)

        # Slightly decrease saturation globally
        hsv[:, :, 1] *= (1.0 - intensity * 0.3)

        # Slight warmth shift (increase hue towards yellow)
        hsv[:, :, 0] = np.clip(hsv[:, :, 0] + intensity * 4, 0, 179)

        # Reduce brightness slightly (aged skin is less luminous)
        hsv[:, :, 2] *= (1.0 - intensity * 0.12)

        hsv = np.clip(hsv, 0, 255).astype(np.uint8)
        hsv[:, :, 0] = np.clip(hsv[:, :, 0], 0, 179)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def _add_sag(self, img: np.ndarray, intensity: float) -> np.ndarray:
        """
        Simulate facial sagging by warping the lower third of the face
        slightly downward using an affine/perspective transform.
        """
        h, w = img.shape[:2]
        result = img.copy()

        # Define source points (lower face quadrilateral)
        y_start = int(h * 0.55)
        src_pts = np.float32([
            [0, y_start],
            [w, y_start],
            [0, h - 1],
            [w, h - 1],
        ])

        # Shift bottom corners downward slightly
        shift = int(h * intensity * 0.04)
        dst_pts = np.float32([
            [0, y_start],
            [w, y_start],
            [int(w * 0.02 * intensity), h - 1],
            [int(w * (1 - 0.02 * intensity)), h - 1],
        ])

        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        lower_warped = cv2.warpPerspective(
            result[y_start:, :], M,
            (w, h - y_start),
            borderMode=cv2.BORDER_REFLECT_101,
        )
        # Recompute M for the sub-image region
        sub_h = h - y_start
        src_sub = np.float32([
            [0, 0], [w, 0],
            [0, sub_h - 1], [w, sub_h - 1],
        ])
        dst_sub = np.float32([
            [0, 0], [w, 0],
            [int(w * 0.02 * intensity), sub_h - 1],
            [int(w * (1 - 0.02 * intensity)), sub_h - 1],
        ])
        M_sub = cv2.getPerspectiveTransform(src_sub, dst_sub)
        lower_warped = cv2.warpPerspective(
            result[y_start:, :], M_sub,
            (w, sub_h),
            borderMode=cv2.BORDER_REFLECT_101,
        )

        # Blend warped lower face back
        alpha = min(intensity * 0.7, 0.6)
        blended = cv2.addWeighted(lower_warped, alpha, result[y_start:, :], 1 - alpha, 0)
        result[y_start:, :] = blended

        return result

    def _roughen_texture(self, img: np.ndarray, intensity: float) -> np.ndarray:
        """
        Increase perceived skin texture (pores, roughness) by sharpening
        mid-frequency detail.
        """
        # Un-sharp mask
        blurred = cv2.GaussianBlur(img, (0, 0), 2.0)
        sharpened = cv2.addWeighted(img, 1.0 + intensity * 0.8, blurred, -intensity * 0.8, 0)

        # Add subtle noise for skin texture
        noise = np.random.normal(0, intensity * 6, img.shape).astype(np.float32)
        result = sharpened.astype(np.float32) + noise
        return np.clip(result, 0, 255).astype(np.uint8)

    def _colour_grade(self, img: np.ndarray, target_age: int) -> np.ndarray:
        """
        Apply a subtle warm colour grade for older targets (like a
        vintage/warm feel that connotes age).
        """
        if target_age < 40:
            return img

        # Convert to PIL for easy contrast/brightness
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        # Slightly reduce contrast (aged photos feel softer)
        factor = max(0.88, 1.0 - (target_age - 40) * 0.002)
        pil_img = ImageEnhance.Contrast(pil_img).enhance(factor)

        # Warm colour cast
        warmth = min((target_age - 40) * 0.003, 0.12)
        arr = np.array(pil_img, dtype=np.float32)
        arr[:, :, 0] = np.clip(arr[:, :, 0] + warmth * 20, 0, 255)  # R
        arr[:, :, 2] = np.clip(arr[:, :, 2] - warmth * 10, 0, 255)  # B

        result = arr.astype(np.uint8)
        return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
