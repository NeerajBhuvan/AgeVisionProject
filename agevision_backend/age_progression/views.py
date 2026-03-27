"""
Age Progression REST API Views
===============================
Endpoints:
  POST   /api/age-progression/progress/      – Run age progression
  GET    /api/age-progression/history/        – List user's records
  GET    /api/age-progression/history/<id>/   – Single record detail
  DELETE /api/age-progression/history/<id>/   – Delete a record
  GET    /api/age-progression/stats/          – Aggregated statistics
  GET    /api/age-progression/health/         – Health / capability check
"""

import os
import time
import logging

from django.conf import settings

from rest_framework import status
from rest_framework.decorators import (
    api_view,
    permission_classes,
    parser_classes,
)
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response

from .models import ProgressionDB
from .serializers import (
    AgeProgressionInputSerializer,
    ProgressionRecordSerializer,
)
from .utils.face_detector import FaceDetector
from .utils.age_estimator import AgeEstimator
from .utils.age_progressor import AgeProgressor
from .utils.image_processor import ImageProcessor

logger = logging.getLogger(__name__)


# ======================================================================
#  POST  /api/age-progression/progress/
# ======================================================================

@api_view(['POST'])
@permission_classes([IsAuthenticated])
@parser_classes([MultiPartParser, FormParser])
def progress_age_view(request):
    """
    Upload an image + target_age → detect face → estimate current age
    → apply aging → return URLs for original, progressed, and comparison.
    """
    # ── Validate input ────────────────────────────────────────────
    serializer = AgeProgressionInputSerializer(data=request.data)
    if not serializer.is_valid():
        return Response(
            {'success': False, 'errors': serializer.errors},
            status=status.HTTP_400_BAD_REQUEST,
        )

    image_file = request.FILES['image']
    target_age = serializer.validated_data['target_age']

    # ── Save uploaded image to temp ───────────────────────────────
    temp_dir = os.path.join(settings.MEDIA_ROOT, 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, image_file.name)

    with open(temp_path, 'wb+') as f:
        for chunk in image_file.chunks():
            f.write(chunk)

    start_time = time.time()

    try:
        # ── 1. Read image ─────────────────────────────────────────
        img = ImageProcessor.read_image(temp_path)

        # ── 2. Detect face ────────────────────────────────────────
        faces = FaceDetector.detect_faces(img)
        if not faces:
            return Response(
                {'success': False, 'error': 'No face found in image'},
                status=status.HTTP_400_BAD_REQUEST,
            )

        face_crop, bbox = faces[0]  # primary face

        # ── 3. Estimate current age ───────────────────────────────
        estimation = AgeEstimator.estimate_age(temp_path)
        current_age = estimation['age']
        gender = estimation.get('gender', 'Unknown')

        # ── 4. Validate target_age ────────────────────────────────
        if target_age <= current_age:
            return Response(
                {
                    'success': False,
                    'error': (
                        f'Target age ({target_age}) must be greater than '
                        f'estimated current age ({current_age})'
                    ),
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        # ── 5. Resize face for processing ─────────────────────────
        face_preprocessed = ImageProcessor.preprocess_for_model(face_crop)

        # ── 6. Run age progression ────────────────────────────────
        progressor = AgeProgressor()
        result = progressor.progress_age(face_preprocessed, current_age, target_age)

        progressed_face = result['image']
        method_used = result['method_used']

        # ── 7. Save output images ─────────────────────────────────
        orig_info = ImageProcessor.save_image(
            face_preprocessed, 'age_progression/originals', prefix='orig_',
        )
        prog_info = ImageProcessor.save_image(
            progressed_face, 'age_progression/progressions', prefix='prog_',
        )

        comparison = ImageProcessor.create_comparison_view(
            face_preprocessed, progressed_face, current_age, target_age,
        )
        comp_info = ImageProcessor.save_image(
            comparison, 'age_progression/comparisons', prefix='comp_',
        )

        processing_time = round((time.time() - start_time) * 1000, 2)  # ms

        # ── 8. Store in MongoDB ───────────────────────────────────
        try:
            record_id = ProgressionDB.insert_record(
                user_id=str(request.user.id),
                original_image_path=orig_info['relative_path'],
                progressed_image_path=prog_info['relative_path'],
                comparison_image_path=comp_info['relative_path'],
                current_age=current_age,
                target_age=target_age,
                processing_time=processing_time,
                method_used=method_used,
                extra={
                    'gender': gender,
                    'aging_params': result.get('aging_params', {}),
                },
            )
        except Exception as db_exc:
            logger.warning('MongoDB insert failed: %s – continuing without DB', db_exc)
            record_id = 'unavailable'

        # ── 9. Build response ─────────────────────────────────────
        base_url = request.build_absolute_uri(settings.MEDIA_URL)

        return Response(
            {
                'success': True,
                'record_id': record_id,
                'current_age': current_age,
                'target_age': target_age,
                'gender': gender,
                'method_used': method_used,
                'processing_time_ms': processing_time,
                'original_image_url': f"{base_url}{orig_info['relative_path']}",
                'progressed_image_url': f"{base_url}{prog_info['relative_path']}",
                'comparison_image_url': f"{base_url}{comp_info['relative_path']}",
            },
            status=status.HTTP_200_OK,
        )

    except ValueError as ve:
        return Response(
            {'success': False, 'error': str(ve)},
            status=status.HTTP_400_BAD_REQUEST,
        )

    except Exception as exc:
        logger.exception('Age progression failed')
        return Response(
            {'success': False, 'error': f'Processing failed: {str(exc)}'},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


# ======================================================================
#  GET  /api/age-progression/history/
# ======================================================================

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def history_list_view(request):
    """Return the authenticated user's progression history."""
    try:
        records = ProgressionDB.get_user_records(str(request.user.id))
        serializer = ProgressionRecordSerializer(
            records, many=True, context={'request': request},
        )
        return Response({'success': True, 'records': serializer.data})
    except Exception as exc:
        logger.warning('History fetch failed: %s', exc)
        return Response(
            {'success': True, 'records': []},
            status=status.HTTP_200_OK,
        )


# ======================================================================
#  GET / DELETE  /api/age-progression/history/<id>/
# ======================================================================

@api_view(['GET', 'DELETE'])
@permission_classes([IsAuthenticated])
def history_detail_view(request, record_id: str):
    """Retrieve or delete a single progression record."""
    if request.method == 'GET':
        record = ProgressionDB.get_record(record_id)
        if not record:
            return Response(
                {'success': False, 'error': 'Record not found'},
                status=status.HTTP_404_NOT_FOUND,
            )
        # Ownership check
        if record.get('user_id') != str(request.user.id):
            return Response(
                {'success': False, 'error': 'Not authorised'},
                status=status.HTTP_403_FORBIDDEN,
            )
        serializer = ProgressionRecordSerializer(
            record, context={'request': request},
        )
        return Response({'success': True, 'record': serializer.data})

    # DELETE
    record = ProgressionDB.get_record(record_id)
    if not record or record.get('user_id') != str(request.user.id):
        return Response(
            {'success': False, 'error': 'Record not found or not authorised'},
            status=status.HTTP_404_NOT_FOUND,
        )
    ProgressionDB.delete_record(record_id)
    return Response({'success': True, 'message': 'Record deleted'})


# ======================================================================
#  GET  /api/age-progression/stats/
# ======================================================================

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def stats_view(request):
    """Return aggregated progression statistics for the user."""
    try:
        stats = ProgressionDB.get_stats(user_id=str(request.user.id))
        return Response({'success': True, 'stats': stats})
    except Exception as exc:
        logger.warning('Stats fetch failed: %s', exc)
        return Response({'success': True, 'stats': {'total': 0}})


# ======================================================================
#  GET  /api/age-progression/health/
# ======================================================================

@api_view(['GET'])
@permission_classes([AllowAny])
def health_view(request):
    """
    Health-check endpoint.  Reports which capabilities are available
    (DeepFace, GAN model, MongoDB).
    """
    capabilities = {
        'deepface': False,
        'gan_model': False,
        'mongodb': False,
        'opencv': False,
    }

    # DeepFace
    try:
        from deepface import DeepFace  # noqa: F401
        capabilities['deepface'] = True
    except ImportError:
        pass

    # OpenCV
    try:
        import cv2  # noqa: F401
        capabilities['opencv'] = True
    except ImportError:
        pass

    # GAN
    capabilities['gan_model'] = AgeProgressor.is_gan_available()

    # MongoDB
    try:
        ProgressionDB.collection().find_one()
        capabilities['mongodb'] = True
    except Exception:
        pass

    return Response({
        'success': True,
        'status': 'healthy',
        'capabilities': capabilities,
    })
