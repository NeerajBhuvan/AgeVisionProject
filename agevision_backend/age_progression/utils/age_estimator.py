"""
Age Estimator
=============
Uses DeepFace (0.0.98) to estimate the age of a face in an image.
Caches the underlying model after first load for fast subsequent calls.
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)

# Module-level flag so DeepFace builds its internal model only once
_deepface_model_loaded = False


class AgeEstimator:
    """Estimate apparent age from a face image using DeepFace."""

    @staticmethod
    def estimate_age(image_or_path, enforce_detection: bool = False) -> dict:
        """
        Estimate the age (and optionally gender) of the primary face.

        Parameters
        ----------
        image_or_path : str | np.ndarray
            File path **or** BGR numpy array.
        enforce_detection : bool
            If True, raise when no face is found; otherwise fall back
            to analysing the full image.

        Returns
        -------
        dict  with keys:
            age      : int
            gender   : str   ('Man' | 'Woman' | 'Unknown')
            raw      : dict  (full DeepFace result for the primary face)
        """
        global _deepface_model_loaded

        try:
            from deepface import DeepFace

            # Warm-up: DeepFace downloads/compiles TF models on first call.
            if not _deepface_model_loaded:
                logger.info('Loading DeepFace age model (first call) …')

            analysis = DeepFace.analyze(
                img_path=image_or_path,
                actions=['age', 'gender'],
                enforce_detection=enforce_detection,
                silent=True,
            )

            _deepface_model_loaded = True

            if isinstance(analysis, list):
                primary = analysis[0]
            else:
                primary = analysis

            age = int(primary.get('age', 25))
            gender = primary.get('dominant_gender', 'Unknown')

            return {
                'age': age,
                'gender': gender,
                'raw': primary,
            }

        except Exception as exc:
            logger.warning('DeepFace age estimation failed: %s', exc)
            # Graceful fallback – return a safe default
            return {
                'age': 25,
                'gender': 'Unknown',
                'raw': {},
                'error': str(exc),
            }
