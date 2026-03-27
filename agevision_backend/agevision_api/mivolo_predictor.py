"""
MiVOLO v2 Age Predictor
========================
Uses MiVOLO v2 (Vision Transformer) with YOLOv8 face+body detection
for high-accuracy age and gender estimation.

MiVOLO v2 achieves ~3.65 MAE on LAGENDA, ~4.24 MAE on IMDB-Clean,
significantly outperforming InsightFace's ~8.5 MAE on diverse populations.

Pipeline:
  1. YOLOv8 detects faces (class 1) and persons (class 0)
  2. Faces are paired with their closest body bbox via IoU
  3. Face + body crops are fed to MiVOLO v2 transformer
  4. Model outputs age (regression) and gender (classification)
"""

import logging
import os
import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)

# Singleton instances
_detector = None
_mivolo_model = None
_mivolo_config = None
_image_processor = None
_init_failed = False
_device = None


def _get_device():
    global _device
    if _device is None:
        _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return _device


def _get_detector():
    """Lazy-load YOLOv8 face/person detector."""
    global _detector, _init_failed

    if _init_failed:
        return None
    if _detector is not None:
        return _detector

    try:
        from ultralytics import YOLO
        from huggingface_hub import hf_hub_download
        import torch

        weights_path = hf_hub_download(
            'iitolstykh/YOLO-Face-Person-Detector',
            'yolov8x_person_face.pt'
        )
        # PyTorch 2.6+ requires weights_only=False for YOLO checkpoints
        original_load = torch.load
        def _patched_load(*args, **kwargs):
            kwargs.setdefault('weights_only', False)
            return original_load(*args, **kwargs)
        torch.load = _patched_load
        try:
            _detector = YOLO(weights_path)
        finally:
            torch.load = original_load
        logger.info("YOLOv8 face/person detector loaded: %s", weights_path)
        return _detector

    except Exception as e:
        logger.error("Failed to load YOLOv8 detector: %s", e)
        _init_failed = True
        return None


def _get_mivolo():
    """Lazy-load MiVOLO v2 model, config, and image processor."""
    global _mivolo_model, _mivolo_config, _image_processor, _init_failed

    if _init_failed:
        return None, None, None
    if _mivolo_model is not None:
        return _mivolo_model, _mivolo_config, _image_processor

    try:
        from transformers import (
            AutoModelForImageClassification,
            AutoConfig,
            AutoImageProcessor,
        )

        model_name = "iitolstykh/mivolo_v2"
        device = _get_device()

        _mivolo_config = AutoConfig.from_pretrained(
            model_name, trust_remote_code=True
        )
        dtype = torch.float16 if device.type == 'cuda' else torch.float32
        _mivolo_model = AutoModelForImageClassification.from_pretrained(
            model_name,
            trust_remote_code=True,
            dtype=dtype,
        )
        _mivolo_model = _mivolo_model.to(device).eval()

        _image_processor = AutoImageProcessor.from_pretrained(
            model_name, trust_remote_code=True
        )

        logger.info("MiVOLO v2 model loaded on %s", device)
        return _mivolo_model, _mivolo_config, _image_processor

    except Exception as e:
        logger.error("Failed to load MiVOLO v2: %s", e)
        _init_failed = True
        return None, None, None


def is_available() -> bool:
    """Check if MiVOLO v2 is available and loadable."""
    detector = _get_detector()
    model, config, processor = _get_mivolo()
    return detector is not None and model is not None


def _detect_faces_and_persons(img: np.ndarray):
    """
    Run YOLOv8 on image to get face and person bounding boxes.

    Returns:
        face_boxes: list of [x1, y1, x2, y2, conf] for faces
        person_boxes: list of [x1, y1, x2, y2, conf] for persons
    """
    detector = _get_detector()
    if detector is None:
        return [], []

    results = detector(img, verbose=False, conf=0.3)
    if not results or len(results) == 0:
        return [], []

    face_boxes = []
    person_boxes = []

    for r in results:
        boxes = r.boxes
        if boxes is None:
            continue
        for i in range(len(boxes)):
            cls = int(boxes.cls[i].item())
            conf = float(boxes.conf[i].item())
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
            bbox = [float(x1), float(y1), float(x2), float(y2), conf]
            if cls == 1:  # face
                face_boxes.append(bbox)
            elif cls == 0:  # person
                person_boxes.append(bbox)

    # Sort faces by area (largest first)
    face_boxes.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)

    return face_boxes, person_boxes


def _iou(box_a, box_b):
    """Compute IoU between two boxes [x1, y1, x2, y2]."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0


def _face_inside_person(face_box, person_box):
    """Check if face center is inside person bbox."""
    face_cx = (face_box[0] + face_box[2]) / 2
    face_cy = (face_box[1] + face_box[3]) / 2
    return (person_box[0] <= face_cx <= person_box[2] and
            person_box[1] <= face_cy <= person_box[3])


def _pair_faces_with_bodies(face_boxes, person_boxes):
    """
    Match each face to its closest person body using containment + IoU.

    Returns list of (face_box, person_box_or_None) tuples.
    """
    pairs = []
    used_persons = set()

    for face in face_boxes:
        best_person = None
        best_score = 0

        for j, person in enumerate(person_boxes):
            if j in used_persons:
                continue
            # Check if face is inside person bbox
            if _face_inside_person(face, person):
                score = _iou(face[:4], person[:4])
                # Containment is more important than IoU for pairing
                score += 1.0  # bonus for containment
                if score > best_score:
                    best_score = score
                    best_person = (j, person)

        if best_person is not None:
            used_persons.add(best_person[0])
            pairs.append((face, best_person[1]))
        else:
            pairs.append((face, None))

    return pairs


def _safe_crop(img, x1, y1, x2, y2):
    """Crop image with boundary clamping."""
    h, w = img.shape[:2]
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(w, int(x2))
    y2 = min(h, int(y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2]


def _run_mivolo_inference(img: np.ndarray, face_boxes, person_boxes):
    """
    Run MiVOLO v2 inference on detected face+body pairs.

    Returns list of dicts with age, gender, confidence, bbox, face_region.
    """
    model, config, processor = _get_mivolo()
    if model is None:
        return []

    pairs = _pair_faces_with_bodies(face_boxes, person_boxes)
    if not pairs:
        return []

    orig_h, orig_w = img.shape[:2]
    device = _get_device()
    results = []
    face_crops_for_emotion = []

    # Process each face individually (handles None body gracefully)
    for i, (face_box, person_box) in enumerate(pairs):
        face_crop = _safe_crop(img, face_box[0], face_box[1], face_box[2], face_box[3])
        if face_crop is None:
            continue

        # Prepare face input
        face_input = processor(images=[face_crop])["pixel_values"]
        face_input = face_input.to(dtype=model.dtype, device=device)

        # Prepare body input — use actual body crop, or a black placeholder
        # MiVOLO v2 requires a tensor for body_input (cannot be None)
        body_crop = None
        if person_box is not None:
            body_crop = _safe_crop(img, person_box[0], person_box[1], person_box[2], person_box[3])
        if body_crop is None:
            # No body detected — use black placeholder so model still runs
            body_crop = np.zeros((224, 224, 3), dtype=np.uint8)
        body_input = processor(images=[body_crop])["pixel_values"]
        body_input = body_input.to(dtype=model.dtype, device=device)

        try:
            with torch.no_grad():
                output = model(faces_input=face_input, body_input=body_input)

            age = output.age_output[0].item()
            gender_idx = output.gender_class_idx[0].item()
            gender_label = config.gender_id2label[gender_idx]
            gender_conf = output.gender_probs[0].item()

            # Map gender to display format
            gender_display = 'Man' if gender_label == 'male' else 'Woman'

            # Face bbox in original image coordinates
            fx1, fy1, fx2, fy2 = face_box[:4]
            fw = fx2 - fx1
            fh = fy2 - fy1

            face_crops_for_emotion.append(face_crop)
            results.append({
                'face_id': i + 1,
                'age': round(age),
                'gender': gender_display,
                'confidence': round(min(1.0, face_box[4]), 3),
                'gender_confidence': round(gender_conf, 3),
                'bbox': {
                    'x': int(fx1), 'y': int(fy1),
                    'w': int(fw), 'h': int(fh),
                },
                'face_region': {
                    'x_pct': round(fx1 / orig_w * 100, 2),
                    'y_pct': round(fy1 / orig_h * 100, 2),
                    'w_pct': round(fw / orig_w * 100, 2),
                    'h_pct': round(fh / orig_h * 100, 2),
                },
            })

        except Exception as e:
            logger.error("MiVOLO inference failed for face %d: %s", i + 1, e)
            continue

    # Batch emotion detection on collected face crops
    from . import emotion_detector
    emotions = emotion_detector.detect_emotions_batch(face_crops_for_emotion)
    for j, result_dict in enumerate(results):
        result_dict['emotion'] = emotions[j] if j < len(emotions) else 'neutral'

    return results


def predict_all_faces(image_path: str) -> list[dict]:
    """
    Predict age, gender for ALL faces in an image file.

    Returns list of dicts, each with:
        face_id, age, gender, confidence, bbox, face_region
    Sorted by face size (largest first).
    """
    img = cv2.imread(image_path)
    if img is None:
        logger.warning("Could not read image: %s", image_path)
        return []

    return _analyze_faces(img)


def predict_frame(frame: np.ndarray) -> list[dict]:
    """
    Predict age, gender for ALL faces in a raw BGR numpy frame.
    Used for real-time camera prediction.
    """
    if frame is None or frame.size == 0:
        return []

    return _analyze_faces(frame)


def _analyze_faces(img: np.ndarray) -> list[dict]:
    """Core face analysis on a BGR numpy array."""
    # Resize large images to prevent OOM
    orig_h, orig_w = img.shape[:2]
    max_dim = 1280
    if max(orig_h, orig_w) > max_dim:
        scale = max_dim / max(orig_h, orig_w)
        resized = cv2.resize(img, (int(orig_w * scale), int(orig_h * scale)),
                             interpolation=cv2.INTER_AREA)
    else:
        scale = 1.0
        resized = img

    # Step 1: Detect faces and persons
    face_boxes, person_boxes = _detect_faces_and_persons(resized)

    if not face_boxes:
        return []

    # Step 2: Scale boxes back to original image size if resized
    if scale != 1.0:
        face_boxes = [[v / scale for v in b[:4]] + [b[4]] for b in face_boxes]
        person_boxes = [[v / scale for v in b[:4]] + [b[4]] for b in person_boxes]

    # Step 3: Run MiVOLO on original image with scaled boxes
    return _run_mivolo_inference(img, face_boxes, person_boxes)
