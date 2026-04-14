import os
import time
import uuid
import base64

import cv2
import numpy as np
from django.conf import settings

from rest_framework import status
from rest_framework.decorators import api_view, permission_classes, parser_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser

from ..mongodb import MongoPredictionManager, MongoBatchPredictionManager
from ..serializers import PredictionSerializer
from ..age_predictor import predict_group_faces, predict_frame

# Per-image cap for batch uploads. Larger files are rejected with an
# error entry rather than killing the whole batch.
BATCH_MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

# Hard cap on number of images in a single batch to keep request memory
# bounded and prevent abuse. Frontend enforces the same limit.
BATCH_MAX_IMAGES = 20


def _to_native(val):
    """Convert numpy scalar to Python native type."""
    try:
        if isinstance(val, (np.integer,)):
            return int(val)
        if isinstance(val, (np.floating,)):
            return float(val)
        if isinstance(val, np.ndarray):
            return val.tolist()
    except Exception:
        pass
    return val


@api_view(['POST'])
@permission_classes([IsAuthenticated])
@parser_classes([MultiPartParser, FormParser])
def predict_view(request):
    """Upload image and return predicted age, gender for all detected faces.
    Works with single selfies and group photos."""
    image = request.FILES.get('image')
    if not image:
        return Response(
            {'error': 'No image provided'},
            status=status.HTTP_400_BAD_REQUEST
        )

    start_time = time.time()

    # Save image temporarily
    temp_dir = os.path.join(settings.MEDIA_ROOT, 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    ext = os.path.splitext(image.name)[1] or '.jpg'
    temp_name = f"{uuid.uuid4().hex}{ext}"
    temp_path = os.path.join(temp_dir, temp_name)

    with open(temp_path, 'wb+') as f:
        for chunk in image.chunks():
            f.write(chunk)

    try:
        result = predict_group_faces(temp_path)
        faces_data = result.pop('faces', [])
        processing_time = (time.time() - start_time) * 1000

        # Save to predictions directory
        predictions_dir = os.path.join(settings.MEDIA_ROOT, 'predictions')
        os.makedirs(predictions_dir, exist_ok=True)
        save_name = f"{uuid.uuid4().hex}{ext}"
        save_path = os.path.join(predictions_dir, save_name)

        image.seek(0)
        with open(save_path, 'wb+') as f:
            for chunk in image.chunks():
                f.write(chunk)

        image_relative_path = f"predictions/{save_name}"

        ensemble_ages_raw = result.get('ensemble_ages', [])
        ensemble_ages_clean = [_to_native(a) for a in ensemble_ages_raw] if ensemble_ages_raw else []

        record = MongoPredictionManager.create(
            user_id=request.user.id,
            image_path=image_relative_path,
            predicted_age=int(result['predicted_age']),
            confidence=float(_to_native(result['confidence'])),
            gender=str(result['gender']),
            emotion=str(result['emotion']),
            race=str(result.get('race', 'Unknown')),
            face_count=int(result['face_count']),
            processing_time_ms=round(processing_time, 2),
            detector_used=str(result.get('detector_used', 'insightface')),
            ensemble_ages=ensemble_ages_clean,
            age_std=float(_to_native(result.get('age_std', 0.0))),
        )

        serializer = PredictionSerializer(record, context={'request': request})

        return Response({
            'message': 'Age prediction successful',
            'prediction': serializer.data,
            'faces': faces_data,
        }, status=status.HTTP_200_OK)

    except Exception as e:
        return Response(
            {'error': f'Prediction failed: {str(e)}'},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
@parser_classes([JSONParser])
def predict_camera_view(request):
    """
    Real-time camera age prediction.
    Accepts a base64-encoded JPEG/PNG frame and returns per-face predictions.
    Does NOT save to history (designed for continuous streaming).

    Request body:
        { "frame": "<base64 encoded image>" }

    Response:
        { "faces": [...], "face_count": N, "processing_time_ms": X }
    """
    frame_b64 = request.data.get('frame')
    if not frame_b64:
        return Response(
            {'error': 'No frame provided'},
            status=status.HTTP_400_BAD_REQUEST
        )

    start_time = time.time()

    try:
        # Strip data URL prefix if present (e.g., "data:image/jpeg;base64,...")
        if ',' in frame_b64:
            frame_b64 = frame_b64.split(',', 1)[1]

        # Decode base64 to numpy array
        img_bytes = base64.b64decode(frame_b64)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if frame is None:
            return Response(
                {'error': 'Could not decode frame'},
                status=status.HTTP_400_BAD_REQUEST
            )

        result = predict_frame(frame)
        processing_time = (time.time() - start_time) * 1000

        return Response({
            'faces': result.get('faces', []),
            'face_count': result.get('face_count', 0),
            'processing_time_ms': round(processing_time, 2),
        }, status=status.HTTP_200_OK)

    except Exception as e:
        return Response(
            {'error': f'Camera prediction failed: {str(e)}'},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['POST'])
@permission_classes([IsAuthenticated])
@parser_classes([MultiPartParser, FormParser])
def batch_predict_view(request):
    """
    Upload multiple images at once and predict ages for all detected faces.

    Each file is processed independently — a failure on one image does NOT
    abort the batch. Per-file errors are returned as `error` entries in
    the `results` array so the client can render a partial-success table.

    Request (multipart/form-data):
        images: file[]   (one or more image files, max BATCH_MAX_IMAGES)

    Response 200:
        {
          "message": "...",
          "batch_id": "<mongo _id>",
          "total_images": int,
          "total_faces": int,
          "processing_time_ms": float,
          "results": [
            {
              "file_index": int,
              "filename": str,
              "face_count": int,
              "image_url": str,
              "faces": [
                { "face_id", "predicted_age", "confidence",
                  "gender", "emotion", "race", "face_region" },
                ...
              ]
            },
            // OR on per-file failure:
            { "file_index": int, "filename": str, "error": str }
          ]
        }
    """
    images = request.FILES.getlist('images')
    if not images:
        return Response(
            {'error': 'No images provided. Send one or more files in the "images" field.'},
            status=status.HTTP_400_BAD_REQUEST
        )

    if len(images) > BATCH_MAX_IMAGES:
        return Response(
            {'error': f'Too many images. Maximum {BATCH_MAX_IMAGES} per batch.'},
            status=status.HTTP_400_BAD_REQUEST
        )

    batch_start = time.time()

    temp_dir = os.path.join(settings.MEDIA_ROOT, 'temp')
    predictions_dir = os.path.join(settings.MEDIA_ROOT, 'predictions')
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(predictions_dir, exist_ok=True)

    results = []
    total_faces = 0

    for idx, image in enumerate(images):
        filename = image.name or f'image_{idx}.jpg'

        # ── 1. Type / size guard ───────────────────────────────────
        content_type = (image.content_type or '').lower()
        if not content_type.startswith('image/'):
            results.append({
                'file_index': idx,
                'filename': filename,
                'error': 'Not an image file',
            })
            continue

        if image.size and image.size > BATCH_MAX_FILE_SIZE:
            results.append({
                'file_index': idx,
                'filename': filename,
                'error': f'File too large (max {BATCH_MAX_FILE_SIZE // (1024 * 1024)} MB)',
            })
            continue

        ext = os.path.splitext(filename)[1] or '.jpg'
        temp_name = f"{uuid.uuid4().hex}{ext}"
        temp_path = os.path.join(temp_dir, temp_name)

        try:
            # ── 2. Persist temp file ───────────────────────────────
            with open(temp_path, 'wb+') as f:
                for chunk in image.chunks():
                    f.write(chunk)

            # ── 3. Run the existing predict pipeline ───────────────
            result = predict_group_faces(temp_path)
            faces_data = result.get('faces', []) or []

            if not faces_data:
                results.append({
                    'file_index': idx,
                    'filename': filename,
                    'error': 'No faces detected',
                })
                continue

            # ── 4. Persist a copy under media/predictions ──────────
            save_name = f"{uuid.uuid4().hex}{ext}"
            save_path = os.path.join(predictions_dir, save_name)
            image.seek(0)
            with open(save_path, 'wb+') as f:
                for chunk in image.chunks():
                    f.write(chunk)
            image_relative_path = f"predictions/{save_name}"
            image_url = request.build_absolute_uri(
                f'{settings.MEDIA_URL}{image_relative_path}'
            )

            # ── 5. Sanitize numpy types in face data ───────────────
            cleaned_faces = []
            for face in faces_data:
                cleaned_faces.append({
                    'face_id': int(_to_native(face.get('face_id', 0))),
                    'predicted_age': int(_to_native(face.get('predicted_age', 0))),
                    'confidence': round(float(_to_native(face.get('confidence', 0.0))), 4),
                    'gender': str(face.get('gender', 'Unknown')),
                    'emotion': str(face.get('emotion', 'Unknown')),
                    'race': str(face.get('race', 'Unknown')),
                    'face_region': {
                        k: float(_to_native(v))
                        for k, v in (face.get('face_region') or {}).items()
                    },
                })

            face_count = len(cleaned_faces)
            total_faces += face_count

            # ── 6. Persist a per-image prediction record so the
            #       single-image history page still surfaces these.
            primary = cleaned_faces[0]
            ensemble_ages_raw = result.get('ensemble_ages', []) or []
            ensemble_ages_clean = [_to_native(a) for a in ensemble_ages_raw]
            try:
                MongoPredictionManager.create(
                    user_id=request.user.id,
                    image_path=image_relative_path,
                    predicted_age=primary['predicted_age'],
                    confidence=primary['confidence'],
                    gender=primary['gender'],
                    emotion=primary['emotion'],
                    race=primary.get('race', 'Unknown'),
                    face_count=face_count,
                    processing_time_ms=0.0,  # batch-level only
                    detector_used=str(result.get('detector_used', 'insightface')),
                    ensemble_ages=ensemble_ages_clean,
                    age_std=float(_to_native(result.get('age_std', 0.0))),
                )
            except Exception:
                # History persistence failure must not break the batch
                pass

            results.append({
                'file_index': idx,
                'filename': filename,
                'face_count': face_count,
                'image_url': image_url,
                'image_path': image_relative_path,
                'faces': cleaned_faces,
            })

        except Exception as e:
            results.append({
                'file_index': idx,
                'filename': filename,
                'error': f'Processing failed: {str(e)}',
            })
        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass

    processing_time_ms = (time.time() - batch_start) * 1000

    # ── 7. Persist the batch record ───────────────────────────────
    try:
        batch_record = MongoBatchPredictionManager.create(
            user_id=request.user.id,
            total_images=len(images),
            total_faces=total_faces,
            results=results,
            processing_time_ms=processing_time_ms,
        )
        batch_id = batch_record['id']
    except Exception:
        batch_id = None

    return Response({
        'message': f'Batch processed: {len(images)} image(s), {total_faces} face(s) detected',
        'batch_id': batch_id,
        'total_images': len(images),
        'total_faces': total_faces,
        'processing_time_ms': round(processing_time_ms, 2),
        'results': results,
    }, status=status.HTTP_200_OK)
