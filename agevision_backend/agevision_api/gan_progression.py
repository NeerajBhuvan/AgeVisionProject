"""
Age Progression Pipeline
=========================
Supports two GAN models:
  - SAM: Style-based Age Manipulation (pSp + StyleGAN2). Target-age control (0-100).
  - Fast-AgingGAN: CycleGAN trained on UTKFace (includes Indian faces). Young→old only.

Falls back to rule-based OpenCV pipeline if no GAN model is available.
"""

import logging
import os
import sys
import time
import uuid

import cv2
import numpy as np

from django.conf import settings

# Add the project root to path for age_pipeline imports
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

logger = logging.getLogger("agevision.gan_progression")


class GANProgressionPipeline:
    """
    Age progression using SAM or Fast-AgingGAN.
    Falls back to rule-based approach if no GAN model is available.
    """

    _sam_model = None
    _sam_model_tried = False
    _fast_aging_model = None
    _fast_aging_tried = False

    def __init__(self):
        self.steps = []

    # ──────────────────────────────────────────────────────────
    #  Model Management (Singletons)
    # ──────────────────────────────────────────────────────────

    @classmethod
    def _get_sam_model(cls):
        """Lazy-load the SAM model (singleton across requests)."""
        if cls._sam_model is not None:
            return cls._sam_model
        if cls._sam_model_tried:
            return None

        cls._sam_model_tried = True
        try:
            from .sam.inference import SAMInference
            sam = SAMInference()
            if sam.load():
                cls._sam_model = sam
                logger.info("SAM model loaded successfully on %s", sam.device)
                return cls._sam_model
            else:
                logger.warning("SAM model checkpoint not found")
                return None
        except Exception as e:
            logger.error("Failed to load SAM model: %s", e)
            return None

    @classmethod
    def _get_fast_aging_model(cls):
        """Lazy-load the Fast-AgingGAN model (singleton across requests)."""
        if cls._fast_aging_model is not None:
            return cls._fast_aging_model
        if cls._fast_aging_tried:
            return None

        cls._fast_aging_tried = True
        try:
            from .fast_aging.inference import FastAgingInference
            model = FastAgingInference()
            if model.load():
                cls._fast_aging_model = model
                logger.info("Fast-AgingGAN loaded on %s", model.device)
                return cls._fast_aging_model
            else:
                logger.warning("Fast-AgingGAN checkpoint not found")
                return None
        except Exception as e:
            logger.error("Failed to load Fast-AgingGAN: %s", e)
            return None

    @classmethod
    def is_gan_available(cls) -> bool:
        """Check if any GAN model is ready for inference."""
        sam = cls._get_sam_model()
        if sam is not None and sam.is_ready:
            return True
        fast = cls._get_fast_aging_model()
        return fast is not None and fast.is_ready

    # ──────────────────────────────────────────────────────────
    #  Main Entry Point
    # ──────────────────────────────────────────────────────────

    def run(self, image_path: str, current_age: int, target_age: int,
            gan_model: str = 'sam') -> dict:
        """
        Run age progression on a single image.

        Args:
            gan_model: 'sam' for SAM (target-age control) or
                       'fast_aging' for Fast-AgingGAN (young→old, Indian-inclusive).

        Returns dict with:
            - output_path, relative_path, steps, insights,
              processing_time_ms, model_type
        """
        t0 = time.time()
        self.steps = []

        if gan_model == 'fast_aging':
            fast = self._get_fast_aging_model()
            if fast is not None and fast.is_ready:
                result = self._run_fast_aging_pipeline(
                    image_path, current_age, target_age, fast)
            else:
                logger.warning("Fast-AgingGAN not available, trying SAM")
                sam = self._get_sam_model()
                if sam is not None and sam.is_ready:
                    result = self._run_sam_pipeline(
                        image_path, current_age, target_age, sam)
                else:
                    result = self._run_fallback(image_path, current_age, target_age)
        else:
            sam = self._get_sam_model()
            if sam is not None and sam.is_ready:
                result = self._run_sam_pipeline(
                    image_path, current_age, target_age, sam)
            else:
                logger.warning("SAM not available, trying Fast-AgingGAN")
                fast = self._get_fast_aging_model()
                if fast is not None and fast.is_ready:
                    result = self._run_fast_aging_pipeline(
                        image_path, current_age, target_age, fast)
                else:
                    result = self._run_fallback(image_path, current_age, target_age)

        result['processing_time_ms'] = round((time.time() - t0) * 1000, 2)
        result['steps'] = self.steps
        return result

    def run_multi_age(self, image_path: str, current_age: int,
                       target_ages: list, gan_model: str = 'sam') -> dict:
        """
        Run age progression for multiple target ages on a single image.

        Returns dict with:
            - progressions: dict of {age: {output_path, relative_path, insights}}
            - grid_path: path to comparison grid image
            - steps: pipeline steps
            - processing_time_ms: total time
            - model_type: model used
        """
        t0 = time.time()
        self.steps = []

        sam_model = self._get_sam_model()
        use_sam = sam_model is not None and sam_model.is_ready

        # Detect and crop face once
        step_t = time.time()
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise ValueError(f"Cannot read image: {image_path}")

        # FFHQ alignment or fallback crop
        if use_sam:
            aligned_pil = sam_model.align_face(image_path)
            if aligned_pil is not None:
                aligned_np = np.array(aligned_pil)
                face_crop = cv2.cvtColor(aligned_np, cv2.COLOR_RGB2BGR)
                face_crop = cv2.resize(face_crop, (256, 256))
                self._record_step('Face Alignment (FFHQ/dlib)', '\U0001f4cd',
                                  time.time() - step_t)
            else:
                face_crop, _ = self._simple_face_crop(img_bgr)
                self._record_step('Face Detection (fallback)', '\U0001f441\ufe0f',
                                  time.time() - step_t)
        else:
            face_crop, _ = self._simple_face_crop(img_bgr)
            self._record_step('Face Detection (basic)', '\U0001f441\ufe0f',
                              time.time() - step_t)

        progressions = {}
        aged_images = {}
        model_type = 'OpenCV-RuleBased'

        for target_age in target_ages:
            age_t0 = time.time()

            if use_sam:
                try:
                    # Direct SAM output — no paste-back or extra processing
                    aged_crop = sam_model.transform_face(face_crop, target_age)
                    model_type = 'SAM-GAN'
                except Exception as e:
                    logger.error("SAM failed for age %d: %s, using fallback", target_age, e)
                    aged_crop = cv2.resize(
                        self._opencv_age(img_bgr, current_age, target_age), (256, 256)
                    )
                    model_type = 'OpenCV-RuleBased'
            else:
                aged_crop = cv2.resize(
                    self._opencv_age(img_bgr, current_age, target_age), (256, 256)
                )
                model_type = 'OpenCV-RuleBased'

            # Save individual output
            output_dir = os.path.join(settings.MEDIA_ROOT, 'progressions')
            os.makedirs(output_dir, exist_ok=True)
            fname = f"sam_prog_{uuid.uuid4().hex[:12]}_age{target_age}.jpg"
            output_path = os.path.join(output_dir, fname)
            cv2.imwrite(output_path, aged_crop, [cv2.IMWRITE_JPEG_QUALITY, 95])

            # Compute insights
            insights = self._compute_insights(current_age, target_age, aged_crop, face_crop)

            progressions[target_age] = {
                'output_path': output_path,
                'relative_path': f"progressions/{fname}",
                'insights': insights,
                'processing_time_ms': round((time.time() - age_t0) * 1000, 2),
            }
            aged_images[f"Age {target_age}"] = aged_crop

            self._record_step(f'Age {target_age} Transform', '\U0001f9ec',
                              time.time() - age_t0)

        # Create comparison grid
        grid_path = None
        grid_fname = None
        try:
            from age_pipeline.postprocess import create_comparison_grid
            grid = create_comparison_grid(face_crop, aged_images)
            grid_fname = f"grid_{uuid.uuid4().hex[:12]}.jpg"
            grid_path = os.path.join(
                settings.MEDIA_ROOT, 'progressions', grid_fname
            )
            cv2.imwrite(grid_path, grid, [cv2.IMWRITE_JPEG_QUALITY, 95])
        except Exception as e:
            logger.warning("Grid creation failed: %s", e)

        return {
            'progressions': progressions,
            'grid_path': grid_path,
            'grid_relative_path': f"progressions/{grid_fname}" if grid_path else None,
            'steps': self.steps,
            'processing_time_ms': round((time.time() - t0) * 1000, 2),
            'model_type': model_type,
        }

    # ──────────────────────────────────────────────────────────
    #  SAM Pipeline
    # ──────────────────────────────────────────────────────────

    def _run_sam_pipeline(self, image_path: str, current_age: int,
                          target_age: int, sam_model) -> dict:
        """Run the SAM-based age transformation pipeline.

        Matches the original SAM repo: align face -> encode+decode -> output.
        No paste-back, no color correction, no extra aging enhancement.
        """

        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise ValueError(f"Cannot read image: {image_path}")

        # Step 1: Face Alignment (FFHQ-style via dlib, or fallback crop)
        step_t = time.time()
        aligned_pil = sam_model.align_face(image_path)

        if aligned_pil is not None:
            aligned_np = np.array(aligned_pil)
            face_crop = cv2.cvtColor(aligned_np, cv2.COLOR_RGB2BGR)
            face_crop = cv2.resize(face_crop, (256, 256))
            self._record_step('Face Alignment (FFHQ/dlib)', '\U0001f4cd',
                              time.time() - step_t)
        else:
            face_crop, _ = self._simple_face_crop(img_bgr)
            self._record_step('Face Detection (fallback)', '\U0001f441\ufe0f',
                              time.time() - step_t)

        # Step 2: SAM GAN Inference — direct output, no post-processing
        step_t = time.time()
        try:
            output_image = sam_model.transform_face(face_crop, target_age)
        except Exception as e:
            logger.error("SAM inference failed: %s, using fallback", e)
            return self._run_fallback(image_path, current_age, target_age)
        self._record_step('SAM Age Transform', '\U0001f9ec', time.time() - step_t)

        # Step 3: Quality Assessment
        step_t = time.time()
        insights = self._compute_insights(current_age, target_age, output_image, face_crop)
        self._record_step('Quality Assessment', '\U0001f4ca', time.time() - step_t)

        # Save output
        output_dir = os.path.join(settings.MEDIA_ROOT, 'progressions')
        os.makedirs(output_dir, exist_ok=True)
        fname = f"sam_prog_{uuid.uuid4().hex[:12]}.jpg"
        output_path = os.path.join(output_dir, fname)
        cv2.imwrite(output_path, output_image, [cv2.IMWRITE_JPEG_QUALITY, 95])

        return {
            'output_path': output_path,
            'relative_path': f"progressions/{fname}",
            'insights': insights,
            'model_type': 'SAM-GAN',
        }

    # ──────────────────────────────────────────────────────────
    #  Fast-AgingGAN Pipeline
    # ──────────────────────────────────────────────────────────

    def _run_fast_aging_pipeline(self, image_path: str, current_age: int,
                                  target_age: int, fast_model) -> dict:
        """Run Fast-AgingGAN age transformation (young→old).

        No face alignment needed — model accepts full face images.
        Trained on UTKFace (Indian/South Asian inclusive).
        """

        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise ValueError(f"Cannot read image: {image_path}")

        # Step 1: Face Detection & Crop
        step_t = time.time()
        face_crop, _ = self._simple_face_crop(img_bgr)
        # Resize to 512x512 for Fast-AgingGAN
        face_crop_512 = cv2.resize(face_crop, (512, 512))
        self._record_step('Face Detection & Crop', '\U0001f441\ufe0f',
                          time.time() - step_t)

        # Step 2: Fast-AgingGAN Inference
        step_t = time.time()
        try:
            output_image = fast_model.transform_face(face_crop_512, target_age)
        except Exception as e:
            logger.error("Fast-AgingGAN inference failed: %s, using fallback", e)
            return self._run_fallback(image_path, current_age, target_age)
        self._record_step('Fast-AgingGAN Transform', '\U0001f9ec', time.time() - step_t)

        # Step 3: Quality Assessment
        step_t = time.time()
        insights = self._compute_insights(current_age, target_age, output_image, face_crop_512)
        self._record_step('Quality Assessment', '\U0001f4ca', time.time() - step_t)

        # Save output
        output_dir = os.path.join(settings.MEDIA_ROOT, 'progressions')
        os.makedirs(output_dir, exist_ok=True)
        fname = f"fast_aging_{uuid.uuid4().hex[:12]}.jpg"
        output_path = os.path.join(output_dir, fname)
        cv2.imwrite(output_path, output_image, [cv2.IMWRITE_JPEG_QUALITY, 95])

        return {
            'output_path': output_path,
            'relative_path': f"progressions/{fname}",
            'insights': insights,
            'model_type': 'Fast-AgingGAN',
        }

    # ──────────────────────────────────────────────────────────
    #  Fallback Path (OpenCV rule-based)
    # ──────────────────────────────────────────────────────────

    def _run_fallback(self, image_path: str, current_age: int,
                      target_age: int) -> dict:
        """Fall back to the rule-based OpenCV pipeline."""
        from .age_progression import AgingPipeline

        pipeline = AgingPipeline()
        result = pipeline.run(image_path, current_age, target_age)

        self.steps = result['steps']
        return {
            'output_path': result['output_path'],
            'relative_path': result['relative_path'],
            'insights': result['insights'],
            'model_type': 'OpenCV-RuleBased',
        }

    def _opencv_age(self, img_bgr: np.ndarray, current_age: int,
                     target_age: int) -> np.ndarray:
        """Apply OpenCV-only aging to an in-memory image."""
        result = self._enhance_aging(img_bgr.copy(), current_age, target_age)
        return self._color_match(result, img_bgr)

    # ──────────────────────────────────────────────────────────
    #  Utilities
    # ──────────────────────────────────────────────────────────

    def _record_step(self, label: str, icon: str, elapsed):
        """Record a completed pipeline step."""
        time_ms = elapsed * 1000 if isinstance(elapsed, float) and elapsed < 100 else elapsed
        self.steps.append({
            'label': label,
            'icon': icon,
            'status': 'done',
            'time_ms': round(time_ms, 2),
        })

    @staticmethod
    def _simple_face_crop(img_bgr: np.ndarray) -> tuple:
        """Simple face crop using Haar cascade (no age_pipeline dependency)."""
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))

        h, w = img_bgr.shape[:2]
        if len(faces) == 0:
            s = min(h, w)
            x0, y0 = (w - s) // 2, (h - s) // 2
            crop = img_bgr[y0:y0+s, x0:x0+s]
            return cv2.resize(crop, (256, 256)), (x0, y0, x0+s, y0+s)

        x, y, fw, fh = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
        pad = int(max(fw, fh) * 0.4)
        x1, y1 = max(0, x - pad), max(0, y - int(pad * 1.2))
        x2, y2 = min(w, x + fw + pad), min(h, y + fh + int(pad * 0.5))
        side = max(x2 - x1, y2 - y1)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        x1, y1 = max(0, cx - side // 2), max(0, cy - side // 2)
        x2, y2 = min(w, x1 + side), min(h, y1 + side)
        crop = img_bgr[y1:y2, x1:x2]
        return cv2.resize(crop, (256, 256)), (x1, y1, x2, y2)

    @staticmethod
    def _old_paste_back(original, aged_crop, crop_coords):
        """Paste back using old-style coords (x1, y1, x2, y2)."""
        x1, y1, x2, y2 = crop_coords
        cw, ch = x2 - x1, y2 - y1
        result = original.copy()
        aged_resized = cv2.resize(aged_crop, (cw, ch))

        mask = np.zeros((ch, cw), dtype=np.float32)
        cv2.ellipse(mask, (cw//2, ch//2), (int(cw*0.42), int(ch*0.42)),
                    0, 0, 360, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (31, 31), 10)
        mask3 = np.stack([mask]*3, axis=-1)

        roi = result[y1:y2, x1:x2].astype(np.float32)
        blended = aged_resized.astype(np.float32) * mask3 + roi * (1 - mask3)
        result[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)
        return result

    @staticmethod
    def _enhance_aging(image: np.ndarray, current_age: int,
                       target_age: int) -> np.ndarray:
        """Apply OpenCV-based aging enhancements."""
        age_diff = target_age - current_age
        intensity = min(abs(age_diff) / 40.0, 1.0)
        result = image.copy()

        if age_diff <= 0:
            return result

        # Wrinkle Enhancement
        if intensity > 0.15:
            gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
            lap = np.abs(cv2.Laplacian(gray, cv2.CV_64F))
            lap = (lap / max(lap.max(), 1) * 255).astype(np.uint8)
            wrinkle_mask = cv2.GaussianBlur(lap, (3, 3), 0)
            wrinkle_layer = np.stack([wrinkle_mask] * 3, axis=-1).astype(np.float32)
            result = np.clip(
                result.astype(np.float32) - wrinkle_layer * intensity * 0.45,
                0, 255
            ).astype(np.uint8)

        # Desaturation
        if intensity > 0.2:
            hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 1] *= max(1.0 - intensity * 0.20, 0.65)
            hsv = np.clip(hsv, 0, [179, 255, 255]).astype(np.uint8)
            result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Hair graying
        if intensity > 0.3:
            hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
            dark_mask = (hsv[:, :, 2] < 120).astype(np.float32)
            dark_mask = cv2.GaussianBlur(dark_mask, (11, 11), 3)
            hsv[:, :, 1] *= (1.0 - dark_mask * intensity * 0.65)
            hsv[:, :, 2] += dark_mask * intensity * 20
            hsv = np.clip(hsv, 0, [179, 255, 255]).astype(np.uint8)
            result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Contrast reduction
        if intensity > 0.25:
            mean_val = result.mean()
            alpha = max(1.0 - intensity * 0.08, 0.85)
            result = cv2.convertScaleAbs(result, alpha=alpha,
                                          beta=mean_val * (1 - alpha))

        return result

    @staticmethod
    def _color_match(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """Colour transfer in LAB space."""
        src_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
        ref_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB).astype(np.float32)
        for ch in range(3):
            s_mean, s_std = src_lab[:,:,ch].mean(), src_lab[:,:,ch].std() + 1e-6
            r_mean, r_std = ref_lab[:,:,ch].mean(), ref_lab[:,:,ch].std() + 1e-6
            src_lab[:,:,ch] = (src_lab[:,:,ch] - s_mean) * (r_std / s_std) + r_mean
        return cv2.cvtColor(np.clip(src_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)

    @staticmethod
    def _compute_insights(current_age: int, target_age: int,
                          aged_crop=None, original_crop=None) -> list:
        """Compute aging insight metrics using real image analysis."""
        age_diff = abs(target_age - current_age)
        age_ratio = min(age_diff / 60.0, 1.0)

        # Default heuristic values (overridden below when images available)
        age_transform_val = min(int(age_ratio * 100), 100)
        identity_val = max(100 - int(age_ratio * 30), 60)
        wrinkle_val = min(int(max(target_age - 30, 0) / 50 * 100), 100)
        hair_val = min(int(max(target_age - 35, 0) / 45 * 100), 100)
        structural_val = min(int(max(target_age - 40, 0) / 40 * 100), 100)

        if aged_crop is not None and original_crop is not None:
            try:
                size = (256, 256)
                orig = cv2.resize(original_crop, size)
                aged = cv2.resize(aged_crop, size)

                # 1. Age Transformation: measure actual pixel change
                orig_gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY).astype(np.float64)
                aged_gray = cv2.cvtColor(aged, cv2.COLOR_BGR2GRAY).astype(np.float64)
                mse = np.mean((orig_gray - aged_gray) ** 2)
                if mse > 0:
                    age_transform_val = min(int(mse / 15.0), 100)
                    age_transform_val = max(age_transform_val, 5)

                # 2. Identity Preservation: SSIM + histogram correlation
                try:
                    from age_pipeline.evaluator import compute_identity_score
                    identity_score = compute_identity_score(orig, aged)
                    identity_val = min(int(identity_score * 100), 100)
                except ImportError:
                    pass

                # 3. Wrinkle Generation: Laplacian variance comparison
                lap_orig = cv2.Laplacian(orig_gray, cv2.CV_64F).var()
                lap_aged = cv2.Laplacian(aged_gray, cv2.CV_64F).var()
                if lap_orig > 0:
                    texture_ratio = lap_aged / lap_orig
                    if target_age > current_age:
                        wrinkle_val = min(int((texture_ratio - 0.5) * 100), 100)
                    else:
                        wrinkle_val = min(int((1.0 / max(texture_ratio, 0.1) - 0.5) * 50), 100)
                    wrinkle_val = max(wrinkle_val, 0)

                # 4. Hair Color Change: saturation difference in dark regions
                orig_hsv = cv2.cvtColor(orig, cv2.COLOR_BGR2HSV).astype(np.float64)
                aged_hsv = cv2.cvtColor(aged, cv2.COLOR_BGR2HSV).astype(np.float64)
                h_cut = int(size[1] * 0.4)
                orig_hair_sat = orig_hsv[:h_cut, :, 1].mean()
                aged_hair_sat = aged_hsv[:h_cut, :, 1].mean()
                orig_hair_val = orig_hsv[:h_cut, :, 2].mean()
                aged_hair_val = aged_hsv[:h_cut, :, 2].mean()
                sat_drop = max(orig_hair_sat - aged_hair_sat, 0)
                val_rise = max(aged_hair_val - orig_hair_val, 0)
                hair_change = (sat_drop / max(orig_hair_sat, 1) * 100 +
                               val_rise / max(orig_hair_val, 1) * 50)
                hair_val = min(int(hair_change), 100)
                hair_val = max(hair_val, 0)

                # 5. Structural Change: pixel displacement via optical flow
                flow = cv2.calcOpticalFlowFarneback(
                    orig_gray.astype(np.uint8), aged_gray.astype(np.uint8),
                    None, 0.5, 3, 15, 3, 5, 1.2, 0
                )
                magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
                avg_displacement = magnitude.mean()
                structural_val = min(int(avg_displacement * 5), 100)
                structural_val = max(structural_val, 0)

            except Exception as e:
                logger.warning("Image-based insights computation failed: %s", e)

        insights = [
            {'label': 'Age Transformation', 'value': age_transform_val, 'color': '#7F5AF0'},
            {'label': 'Identity Preservation', 'value': identity_val, 'color': '#2CB67D'},
            {'label': 'Wrinkle Generation', 'value': wrinkle_val, 'color': '#E16259'},
            {'label': 'Hair Color Change', 'value': hair_val, 'color': '#F0A500'},
            {'label': 'Structural Change', 'value': structural_val, 'color': '#72757E'},
        ]
        return insights
