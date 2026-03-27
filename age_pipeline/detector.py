"""
Face Detection & Preprocessing
================================
Detects faces using OpenCV DNN (Caffe SSD) with Haar cascade fallback.
Handles alignment, cropping, and quality validation.
"""

import logging
import cv2
import numpy as np
from pathlib import Path

logger = logging.getLogger("age_pipeline.detector")

# Minimum acceptable face size (pixels)
MIN_FACE_SIZE = 64
# Minimum input image resolution
MIN_IMAGE_RES = 128

# Path to Caffe SSD model (ships with the project)
_CAFFE_DIR = Path(__file__).resolve().parent.parent / "agevision_backend" / "age_progression" / "trained_models"
_CAFFE_PROTO = _CAFFE_DIR / "deploy.prototxt"
_CAFFE_MODEL = _CAFFE_DIR / "res10_300x300_ssd_iter_140000.caffemodel"

_dnn_net = None
_haar_cascade = None


def _get_dnn_detector():
    """Load the Caffe SSD face detector (singleton)."""
    global _dnn_net
    if _dnn_net is None:
        if _CAFFE_PROTO.exists() and _CAFFE_MODEL.exists():
            try:
                _dnn_net = cv2.dnn.readNetFromCaffe(
                    str(_CAFFE_PROTO), str(_CAFFE_MODEL)
                )
                logger.info("Loaded Caffe SSD face detector")
            except Exception as e:
                logger.warning("Failed to load Caffe SSD: %s", e)
                _dnn_net = False
        else:
            logger.info("Caffe SSD model files not found, using Haar cascade")
            _dnn_net = False
    return _dnn_net if _dnn_net is not False else None


def _get_haar_cascade():
    """Load Haar cascade face detector (fallback)."""
    global _haar_cascade
    if _haar_cascade is None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _haar_cascade = cv2.CascadeClassifier(cascade_path)
        logger.info("Loaded Haar cascade face detector")
    return _haar_cascade


def detect_faces_dnn(img_bgr: np.ndarray, confidence_threshold: float = 0.5):
    """Detect faces using OpenCV DNN (Caffe SSD).

    Returns list of (x, y, w, h, confidence) tuples.
    """
    net = _get_dnn_detector()
    if net is None:
        return []

    h, w = img_bgr.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(img_bgr, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0)
    )
    net.setInput(blob)
    detections = net.forward()

    faces = []
    for i in range(detections.shape[2]):
        conf = detections[0, 0, i, 2]
        if conf < confidence_threshold:
            continue
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        x1, y1, x2, y2 = box.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        fw, fh = x2 - x1, y2 - y1
        if fw >= MIN_FACE_SIZE and fh >= MIN_FACE_SIZE:
            faces.append((x1, y1, fw, fh, float(conf)))

    return faces


def detect_faces_haar(img_bgr: np.ndarray):
    """Detect faces using Haar cascade (fallback).

    Returns list of (x, y, w, h, confidence) tuples.
    """
    cascade = _get_haar_cascade()
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE))
    # Haar doesn't provide confidence; use 0.9 as placeholder
    return [(x, y, w, h, 0.9) for (x, y, w, h) in faces]


def detect_faces(img_bgr: np.ndarray):
    """Detect faces with DNN primary and Haar fallback.

    Returns list of (x, y, w, h, confidence) sorted by area descending.
    """
    faces = detect_faces_dnn(img_bgr)
    if not faces:
        faces = detect_faces_haar(img_bgr)
    # Sort by face area (largest first)
    faces.sort(key=lambda f: f[2] * f[3], reverse=True)
    return faces


def crop_face(img_bgr: np.ndarray, face_rect: tuple, target_size: int = 256,
              padding_ratio: float = 0.4) -> tuple:
    """Crop and align face region with padding.

    Args:
        img_bgr: Input BGR image.
        face_rect: (x, y, w, h, ...) face bounding box.
        target_size: Output crop size (square).
        padding_ratio: Padding around face as fraction of face size.

    Returns:
        (cropped_face, crop_coords) where crop_coords is (x1, y1, x2, y2).
    """
    x, y, w, h = face_rect[:4]
    img_h, img_w = img_bgr.shape[:2]

    # Expand ROI with padding
    pad_w = int(w * padding_ratio)
    pad_h = int(h * padding_ratio * 1.2)  # Extra top padding for forehead
    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(img_w, x + w + pad_w)
    y2 = min(img_h, y + h + int(pad_h * 0.6))

    # Make it square
    crop_w, crop_h = x2 - x1, y2 - y1
    side = max(crop_w, crop_h)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    x1 = max(0, cx - side // 2)
    y1 = max(0, cy - side // 2)
    x2 = min(img_w, x1 + side)
    y2 = min(img_h, y1 + side)

    crop = img_bgr[y1:y2, x1:x2]
    crop_resized = cv2.resize(crop, (target_size, target_size),
                               interpolation=cv2.INTER_LANCZOS4)
    return crop_resized, (x1, y1, x2, y2)


def validate_image(image_path: str) -> dict:
    """Validate an image for processing.

    Returns dict with 'valid', 'reason', 'image', 'faces'.
    """
    result = {"valid": False, "reason": "", "image": None, "faces": []}

    img = cv2.imread(image_path)
    if img is None:
        result["reason"] = f"Cannot read image: {image_path}"
        logger.warning(result["reason"])
        return result

    h, w = img.shape[:2]
    if h < MIN_IMAGE_RES or w < MIN_IMAGE_RES:
        result["reason"] = f"Image too small ({w}x{h}), minimum {MIN_IMAGE_RES}px"
        logger.warning(result["reason"])
        return result

    # Limit image size to prevent OOM
    max_dim = 2048
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)),
                          interpolation=cv2.INTER_AREA)
        logger.info("Resized image from %dx%d to %dx%d", w, h,
                     img.shape[1], img.shape[0])

    faces = detect_faces(img)
    if not faces:
        # Fallback: assume the image IS a face (center crop)
        logger.info("No face detected, using center-crop fallback")
        h, w = img.shape[:2]
        s = min(h, w)
        x0, y0 = (w - s) // 2, (h - s) // 2
        faces = [(x0, y0, s, s, 0.5)]

    result["valid"] = True
    result["image"] = img
    result["faces"] = faces
    result["reason"] = f"Valid: {len(faces)} face(s) detected"
    return result
