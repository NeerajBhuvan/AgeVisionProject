"""
Face Detector
=============
Detects faces in images using OpenCV's Haar Cascade classifier (primary)
with DNN-based fallback.

Returns list of (face_image, bounding_box) tuples.
"""

import os
import cv2
import numpy as np
import logging
import urllib.request

logger = logging.getLogger(__name__)

# Paths for optional DNN fallback model files
_DNN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'trained_models')
_PROTO_PATH = os.path.join(_DNN_DIR, 'deploy.prototxt')
_MODEL_PATH = os.path.join(_DNN_DIR, 'res10_300x300_ssd_iter_140000.caffemodel')

_PROTO_URL = (
    'https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt'
)
_MODEL_URL = (
    'https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/'
    'res10_300x300_ssd_iter_140000.caffemodel'
)


class FaceDetector:
    """
    Detects faces using OpenCV Haar Cascade (primary) with optional
    DNN SSD fallback.  Works out-of-the-box with no downloads.
    """

    _haar_cascade = None
    _dnn_net = None

    # ------------------------------------------------------------------
    #  Lazy-loaded classifiers
    # ------------------------------------------------------------------

    @classmethod
    def _get_haar_cascade(cls):
        if cls._haar_cascade is None:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            cls._haar_cascade = cv2.CascadeClassifier(cascade_path)
            if cls._haar_cascade.empty():
                logger.warning('Haar cascade file not found at %s', cascade_path)
                cls._haar_cascade = None
        return cls._haar_cascade

    @classmethod
    def _get_dnn_net(cls):
        """Load Caffe DNN face detector (downloads weights if missing)."""
        if cls._dnn_net is None:
            if not os.path.isfile(_PROTO_PATH) or not os.path.isfile(_MODEL_PATH):
                cls._download_dnn_models()
            if os.path.isfile(_PROTO_PATH) and os.path.isfile(_MODEL_PATH):
                try:
                    cls._dnn_net = cv2.dnn.readNetFromCaffe(_PROTO_PATH, _MODEL_PATH)
                    logger.info('DNN face detector loaded')
                except Exception as exc:
                    logger.warning('Failed to load DNN model: %s', exc)
                    cls._dnn_net = False  # mark as unavailable
            else:
                cls._dnn_net = False
        return cls._dnn_net if cls._dnn_net is not False else None

    @classmethod
    def _download_dnn_models(cls):
        """Download the Caffe SSD face detector if not present."""
        os.makedirs(_DNN_DIR, exist_ok=True)
        for url, path in [(_PROTO_URL, _PROTO_PATH), (_MODEL_URL, _MODEL_PATH)]:
            if not os.path.isfile(path):
                try:
                    logger.info('Downloading %s …', os.path.basename(path))
                    urllib.request.urlretrieve(url, path)
                except Exception as exc:
                    logger.warning('Download failed for %s: %s', url, exc)

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    @classmethod
    def detect_faces(cls, image: np.ndarray, padding: float = 0.15):
        """
        Detect faces in an image.

        Parameters
        ----------
        image : np.ndarray
            BGR image (as read by cv2.imread).
        padding : float
            Fractional padding around the detected face box (0.15 = 15 %).

        Returns
        -------
        list[tuple[np.ndarray, tuple[int, int, int, int]]]
            Each element is (face_crop, (x, y, w, h)).
        """
        if image is None:
            return []

        faces = cls._detect_haar(image)
        if not faces:
            faces = cls._detect_dnn(image)
        if not faces:
            # Ultimate fallback: treat entire image as the face
            h, w = image.shape[:2]
            faces = [(0, 0, w, h)]
            logger.info('No detector succeeded – using full image as face region')

        results = []
        h_img, w_img = image.shape[:2]
        for (x, y, w, h) in faces:
            # Apply padding
            pad_x = int(w * padding)
            pad_y = int(h * padding)
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(w_img, x + w + pad_x)
            y2 = min(h_img, y + h + pad_y)
            face_crop = image[y1:y2, x1:x2].copy()
            results.append((face_crop, (x1, y1, x2 - x1, y2 - y1)))

        return results

    # ------------------------------------------------------------------
    #  Detection back-ends
    # ------------------------------------------------------------------

    @classmethod
    def _detect_haar(cls, image: np.ndarray):
        cascade = cls._get_haar_cascade()
        if cascade is None:
            return []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        if len(rects) == 0:
            return []
        return [tuple(r) for r in rects]

    @classmethod
    def _detect_dnn(cls, image: np.ndarray, confidence_threshold: float = 0.5):
        net = cls._get_dnn_net()
        if net is None:
            return []
        h, w = image.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)),
            1.0, (300, 300),
            (104.0, 177.0, 123.0),
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
            faces.append((x1, y1, x2 - x1, y2 - y1))
        return faces
