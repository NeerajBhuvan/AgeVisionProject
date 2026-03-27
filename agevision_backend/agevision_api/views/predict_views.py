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

from ..mongodb import MongoPredictionManager
from ..serializers import PredictionSerializer
from ..age_predictor import predict_group_faces, predict_frame


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
