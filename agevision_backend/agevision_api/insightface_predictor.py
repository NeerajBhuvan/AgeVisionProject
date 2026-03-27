"""
InsightFace Age Predictor
=========================
Uses InsightFace's buffalo_l model (ArcFace + age/gender model)
for high-accuracy age prediction.

InsightFace achieves ~3.5 MAE on major benchmarks (UTKFace, MORPH-II),
significantly outperforming DeepFace's ~5-8 MAE.

The buffalo_l model includes:
- det_10g: RetinaFace-based face detection (high accuracy)
- genderage: Age and gender estimation model
- recognition: ArcFace face recognition (optional)
"""

import logging
import os
import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Singleton model instance
_app = None
_init_failed = False


def _get_model():
    """Lazy-load InsightFace model (singleton). Downloads on first use (~300MB)."""
    global _app, _init_failed

    if _init_failed:
        return None

    if _app is not None:
        return _app

    try:
        from insightface.app import FaceAnalysis

        _app = FaceAnalysis(
            name='buffalo_l',
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
        )
        # det_size: detection input size. 640x640 is the default for good accuracy.
        _app.prepare(ctx_id=0, det_size=(640, 640))
        logger.info("InsightFace buffalo_l model loaded successfully")
        return _app

    except ImportError:
        logger.warning("insightface not installed. Run: pip install insightface onnxruntime")
        _init_failed = True
        return None
    except Exception as e:
        logger.error("Failed to load InsightFace model: %s", e)
        _init_failed = True
        return None


def is_available() -> bool:
    """Check if InsightFace is available and loadable."""
    return _get_model() is not None


def predict_single(image_path: str) -> dict | None:
    """
    Predict age, gender for the primary (largest) face in the image.

    Returns dict with:
        age: int
        gender: str ('Man' / 'Woman')
        confidence: float (0-1, based on detection score)
        bbox: dict with x, y, w, h
        face_count: int
    Or None if no face detected or model unavailable.
    """
    app = _get_model()
    if app is None:
        return None

    img = cv2.imread(image_path)
    if img is None:
        logger.warning("Could not read image: %s", image_path)
        return None

    # Resize large images to prevent memory issues
    h, w = img.shape[:2]
    max_dim = 1280
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        scale = 1.0

    try:
        faces = app.get(img)
    except Exception as e:
        logger.error("InsightFace analysis failed: %s", e)
        return None

    if not faces:
        return None

    # Sort by bounding box area (largest face = primary)
    faces = sorted(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)

    primary = faces[0]
    age = int(primary.age)
    # InsightFace gender: 0=Female, 1=Male
    gender = 'Man' if primary.sex == 'M' else 'Woman'
    det_score = float(primary.det_score) if hasattr(primary, 'det_score') else 0.85

    bbox = primary.bbox  # [x1, y1, x2, y2]
    x1, y1, x2, y2 = [int(v / scale) for v in bbox]

    return {
        'age': age,
        'gender': gender,
        'confidence': round(min(1.0, det_score), 3),
        'bbox': {'x': x1, 'y': y1, 'w': x2 - x1, 'h': y2 - y1},
        'face_count': len(faces),
        'detector': 'insightface',
    }


def predict_all_faces(image_path: str) -> list[dict]:
    """
    Predict age, gender for ALL faces in the image.

    Returns list of dicts, each with:
        age, gender, confidence, bbox, face_id, face_region
    Sorted by face size (largest first).
    """
    app = _get_model()
    if app is None:
        return []

    img = cv2.imread(image_path)
    if img is None:
        return []

    return _analyze_faces(app, img)


def predict_frame(frame: np.ndarray) -> list[dict]:
    """
    Predict age, gender for ALL faces in a raw BGR numpy frame.
    Used for real-time camera prediction.

    Parameters
    ----------
    frame : np.ndarray
        BGR image as numpy array.

    Returns list of dicts, each with:
        age, gender, confidence, bbox, face_id, face_region
    Sorted by face size (largest first).
    """
    app = _get_model()
    if app is None:
        return []

    if frame is None or frame.size == 0:
        return []

    return _analyze_faces(app, frame)


def _analyze_faces(app, img: np.ndarray) -> list[dict]:
    """
    Core face analysis on a BGR numpy array.
    Shared by file-based and frame-based prediction paths.
    """
    orig_h, orig_w = img.shape[:2]
    max_dim = 1280
    if max(orig_h, orig_w) > max_dim:
        scale = max_dim / max(orig_h, orig_w)
        img = cv2.resize(img, (int(orig_w * scale), int(orig_h * scale)), interpolation=cv2.INTER_AREA)
    else:
        scale = 1.0

    try:
        faces = app.get(img)
    except Exception as e:
        logger.error("InsightFace analysis failed: %s", e)
        return []

    if not faces:
        return []

    # Sort by area descending
    faces = sorted(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)

    results = []
    face_crops = []
    for i, face in enumerate(faces):
        age = int(face.age)
        gender = 'Man' if face.sex == 'M' else 'Woman'
        det_score = float(face.det_score) if hasattr(face, 'det_score') else 0.85

        bbox = face.bbox
        x1, y1, x2, y2 = [int(v / scale) for v in bbox]
        w, h = x2 - x1, y2 - y1

        # Crop face from (possibly resized) image for emotion detection
        rh, rw = img.shape[:2]
        crop_x1 = max(0, int(face.bbox[0]))
        crop_y1 = max(0, int(face.bbox[1]))
        crop_x2 = min(rw, int(face.bbox[2]))
        crop_y2 = min(rh, int(face.bbox[3]))
        face_crop = img[crop_y1:crop_y2, crop_x1:crop_x2]
        if face_crop.size > 0:
            face_crops.append(face_crop)

        results.append({
            'face_id': i + 1,
            'age': age,
            'gender': gender,
            'confidence': round(min(1.0, det_score), 3),
            'bbox': {'x': x1, 'y': y1, 'w': w, 'h': h},
            'face_region': {
                'x_pct': round(x1 / orig_w * 100, 2),
                'y_pct': round(y1 / orig_h * 100, 2),
                'w_pct': round(w / orig_w * 100, 2),
                'h_pct': round(h / orig_h * 100, 2),
            },
        })

    # Batch emotion detection
    from . import emotion_detector
    emotions = emotion_detector.detect_emotions_batch(face_crops)
    for j, result_dict in enumerate(results):
        result_dict['emotion'] = emotions[j] if j < len(emotions) else 'neutral'

    return results
