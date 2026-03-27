"""
Age Predictor
=============
Age prediction with MiVOLO v2 (primary) and InsightFace buffalo_l (fallback).

MiVOLO v2 (~3.65-4.24 MAE): Vision Transformer + YOLOv8 face/body detection.
InsightFace buffalo_l (~8.5 MAE): RetinaFace + ArcFace (fallback if MiVOLO unavailable).
"""

import logging
import numpy as np
import cv2

from . import mivolo_predictor
from . import insightface_predictor

logger = logging.getLogger(__name__)


def _get_predictor():
    """Return the best available predictor module (MiVOLO first, InsightFace fallback)."""
    try:
        if mivolo_predictor.is_available():
            return mivolo_predictor, 'mivolo_v2'
    except Exception as e:
        logger.warning("MiVOLO v2 unavailable, falling back to InsightFace: %s", e)

    return insightface_predictor, 'insightface'


def predict_age(image_path: str) -> dict:
    """
    Predict age for the primary (largest) face in an image file.

    Returns dict with:
        predicted_age, confidence, gender, face_count,
        detector_used, faces (list of per-face results)
    """
    # Use full group prediction and return primary face
    return predict_group_faces(image_path)


def predict_group_faces(image_path: str, **kwargs) -> dict:
    """
    Detect ALL faces in an image and return per-face age predictions.
    Works for single selfies, group photos, and any multi-face image.

    Returns dict with 'faces' list of per-face results plus primary
    prediction fields for backward compatibility.
    """
    predictor, detector_name = _get_predictor()
    all_faces = predictor.predict_all_faces(image_path)

    if not all_faces:
        return _empty_result()

    faces = []
    for face_data in all_faces:
        age = max(1, min(100, face_data['age']))
        faces.append({
            'face_id': face_data['face_id'],
            'predicted_age': age,
            'confidence': face_data['confidence'],
            'gender': face_data['gender'],
            'emotion': face_data.get('emotion', 'neutral'),
            'race': 'Unknown',
            'face_region': face_data['face_region'],
        })

    primary = faces[0]
    all_ages = [f['predicted_age'] for f in faces]

    return {
        'predicted_age': primary['predicted_age'],
        'confidence': primary['confidence'],
        'gender': primary['gender'],
        'emotion': primary['emotion'],
        'race': primary.get('race', 'Unknown'),
        'face_count': len(faces),
        'detector_used': detector_name,
        'ensemble_ages': all_ages,
        'age_std': round(float(np.std(all_ages)), 2) if len(all_ages) > 1 else 0.0,
        'faces': faces,
    }


def predict_frame(frame: np.ndarray) -> dict:
    """
    Predict ages for all faces in a raw BGR numpy frame (for real-time camera).

    Parameters
    ----------
    frame : np.ndarray
        BGR image as numpy array (from cv2.VideoCapture or decoded base64).

    Returns dict with 'faces' list and primary prediction fields.
    """
    predictor, detector_name = _get_predictor()
    all_faces = predictor.predict_frame(frame)

    if not all_faces:
        return _empty_result()

    faces = []
    for face_data in all_faces:
        age = max(1, min(100, face_data['age']))
        faces.append({
            'face_id': face_data['face_id'],
            'predicted_age': age,
            'confidence': face_data['confidence'],
            'gender': face_data['gender'],
            'emotion': face_data.get('emotion', 'neutral'),
            'race': 'Unknown',
            'face_region': face_data['face_region'],
        })

    primary = faces[0]
    all_ages = [f['predicted_age'] for f in faces]

    return {
        'predicted_age': primary['predicted_age'],
        'confidence': primary['confidence'],
        'gender': primary['gender'],
        'emotion': primary['emotion'],
        'race': 'Unknown',
        'face_count': len(faces),
        'detector_used': detector_name,
        'ensemble_ages': all_ages,
        'age_std': round(float(np.std(all_ages)), 2) if len(all_ages) > 1 else 0.0,
        'faces': faces,
    }


# Keep old name as alias for backward compatibility
predict_age_ensemble = predict_group_faces


def _bbox_to_region(bbox: dict, image_path: str) -> dict:
    """Convert absolute bbox to percentage-based face_region."""
    img = cv2.imread(image_path)
    if img is None:
        return {'x_pct': 0, 'y_pct': 0, 'w_pct': 0, 'h_pct': 0}
    h, w = img.shape[:2]
    return {
        'x_pct': round(bbox['x'] / w * 100, 2),
        'y_pct': round(bbox['y'] / h * 100, 2),
        'w_pct': round(bbox['w'] / w * 100, 2),
        'h_pct': round(bbox['h'] / h * 100, 2),
    }


def _empty_result() -> dict:
    return {
        'predicted_age': 0,
        'confidence': 0.0,
        'gender': 'Unknown',
        'emotion': 'Unknown',
        'race': 'Unknown',
        'face_count': 0,
        'detector_used': 'none',
        'ensemble_ages': [],
        'age_std': 0.0,
        'faces': [],
    }
