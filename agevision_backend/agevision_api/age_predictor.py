"""
Age Predictor (local CPU inference)
===================================
Thin wrapper around `mivolo_predictor`, which runs MiVOLO v2 + YOLOv8 +
emotion classification locally (CPU on this machine; CUDA if ever available).

Prediction used to be offloaded to a Modal GPU endpoint, but that added a
60-180s cold-start that made the predict API "hang" with no response. Running
the models locally removes the network round-trip entirely: the first request
after a server start loads the weights (~10-30s, one-time), and every request
after that returns in a few seconds — no external dependency, no cold start.

External interface (predict_group_faces / predict_frame / predict_age) is
unchanged, so views and serializers don't need to be touched.
"""

import logging

import numpy as np

from . import mivolo_predictor

logger = logging.getLogger(__name__)


def _from_local_faces(faces: list) -> dict:
    """Translate mivolo_predictor's per-face list into the legacy result shape."""
    if not faces:
        return _empty_result()

    out_faces = []
    for face_data in faces:
        age = max(1, min(100, int(face_data.get("age", 0))))
        out_faces.append({
            "face_id": face_data.get("face_id", len(out_faces) + 1),
            "predicted_age": age,
            "confidence": float(face_data.get("confidence", 0.0)),
            "gender": face_data.get("gender", "Unknown"),
            "emotion": face_data.get("emotion", "neutral"),
            "race": "Unknown",
            "face_region": face_data.get("face_region", {
                "x_pct": 0, "y_pct": 0, "w_pct": 0, "h_pct": 0,
            }),
        })

    primary = out_faces[0]
    all_ages = [f["predicted_age"] for f in out_faces]

    return {
        "predicted_age": primary["predicted_age"],
        "confidence": primary["confidence"],
        "gender": primary["gender"],
        "emotion": primary["emotion"],
        "race": primary.get("race", "Unknown"),
        "face_count": len(out_faces),
        "detector_used": "mivolo_v2",
        "ensemble_ages": all_ages,
        "age_std": round(float(np.std(all_ages)), 2) if len(all_ages) > 1 else 0.0,
        "faces": out_faces,
    }


def predict_age(image_path: str) -> dict:
    """Predict age for the primary (largest) face in an image file."""
    return predict_group_faces(image_path)


def predict_group_faces(image_path: str, **kwargs) -> dict:
    """Detect ALL faces in an image file and return per-face predictions."""
    try:
        faces = mivolo_predictor.predict_all_faces(image_path)
    except Exception as e:
        logger.error("Local age prediction failed for %s: %s", image_path, e)
        return _empty_result()
    return _from_local_faces(faces)


def predict_frame(frame: np.ndarray) -> dict:
    """Predict ages for all faces in a raw BGR numpy frame (real-time camera)."""
    if frame is None or frame.size == 0:
        return _empty_result()
    try:
        faces = mivolo_predictor.predict_frame(frame)
    except Exception as e:
        logger.error("Local age prediction (frame) failed: %s", e)
        return _empty_result()
    return _from_local_faces(faces)


predict_age_ensemble = predict_group_faces


def _empty_result(detector_used: str = "none") -> dict:
    return {
        "predicted_age": 0,
        "confidence": 0.0,
        "gender": "Unknown",
        "emotion": "Unknown",
        "race": "Unknown",
        "face_count": 0,
        "detector_used": detector_used,
        "ensemble_ages": [],
        "age_std": 0.0,
        "faces": [],
    }
