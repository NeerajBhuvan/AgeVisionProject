"""
Emotion Detector
================
Lightweight facial emotion classification using a ViT model
(trpakov/vit-face-expression) fine-tuned on FER2013.

Outputs one of 7 emotions: happy, sad, angry, surprise, fear, disgust, neutral.
"""

import logging
import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

_pipeline = None
_init_failed = False

VALID_EMOTIONS = {'happy', 'sad', 'angry', 'surprise', 'fear', 'disgust', 'neutral'}


def _get_pipeline():
    """Lazy-load the emotion classification pipeline."""
    global _pipeline, _init_failed

    if _init_failed:
        return None
    if _pipeline is not None:
        return _pipeline

    try:
        import torch
        from transformers import pipeline as hf_pipeline

        device = 0 if torch.cuda.is_available() else -1
        _pipeline = hf_pipeline(
            "image-classification",
            model="trpakov/vit-face-expression",
            device=device,
            top_k=1,
        )
        logger.info("Emotion detector loaded (trpakov/vit-face-expression)")
        return _pipeline

    except Exception as e:
        logger.error("Failed to load emotion detector: %s", e)
        _init_failed = True
        return None


def _bgr_to_pil(face_crop: np.ndarray) -> Image.Image:
    """Convert BGR numpy array to RGB PIL Image."""
    rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def detect_emotion(face_crop: np.ndarray) -> str:
    """
    Classify emotion for a single face crop (BGR numpy array).
    Returns one of: happy, sad, angry, surprise, fear, disgust, neutral.
    """
    pipe = _get_pipeline()
    if pipe is None:
        return 'neutral'

    try:
        pil_img = _bgr_to_pil(face_crop)
        result = pipe(pil_img)
        label = result[0]['label'].lower()
        return label if label in VALID_EMOTIONS else 'neutral'
    except Exception as e:
        logger.warning("Emotion detection failed: %s", e)
        return 'neutral'


def detect_emotions_batch(face_crops: list[np.ndarray]) -> list[str]:
    """
    Classify emotions for multiple face crops in a single batch.
    Returns list of emotion strings, one per crop.
    """
    if not face_crops:
        return []

    pipe = _get_pipeline()
    if pipe is None:
        return ['neutral'] * len(face_crops)

    try:
        pil_images = [_bgr_to_pil(crop) for crop in face_crops]
        results = pipe(pil_images)
        emotions = []
        for r in results:
            label = r[0]['label'].lower() if isinstance(r, list) else r['label'].lower()
            emotions.append(label if label in VALID_EMOTIONS else 'neutral')
        return emotions
    except Exception as e:
        logger.warning("Batch emotion detection failed: %s", e)
        return ['neutral'] * len(face_crops)
