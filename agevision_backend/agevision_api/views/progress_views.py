import json
import os
import queue
import threading
import time
import uuid
import shutil

from django.conf import settings
from django.http import StreamingHttpResponse

from rest_framework import status
from rest_framework.decorators import api_view, permission_classes, parser_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser

from ..mongodb import MongoProgressionManager
from ..serializers import ProgressionSerializer
from ..gan_progression import GANProgressionPipeline
from ..age_predictor import predict_group_faces


# Map frontend model IDs to display names
GAN_MODEL_NAMES = {
    'sam': 'SAM-GAN',
    'fast_aging': 'Fast-AgingGAN',
    'diffusion': 'FADING-Diffusion',
}


@api_view(['POST'])
@permission_classes([IsAuthenticated])
@parser_classes([MultiPartParser, FormParser])
def progress_view(request):
    """
    Upload image + target_age -> run the aging pipeline -> return
    original image, progressed image, pipeline steps and aging insights.
    """
    image = request.FILES.get('image')
    target_age = request.data.get('target_age')
    gan_model = request.data.get('gan_model', 'sam')

    if not image:
        return Response(
            {'error': 'No image provided'},
            status=status.HTTP_400_BAD_REQUEST
        )

    if not target_age:
        return Response(
            {'error': 'Target age is required'},
            status=status.HTTP_400_BAD_REQUEST
        )

    try:
        target_age = int(target_age)
    except (ValueError, TypeError):
        return Response(
            {'error': 'Target age must be a valid integer'},
            status=status.HTTP_400_BAD_REQUEST
        )

    if target_age < 1 or target_age > 100:
        return Response(
            {'error': 'Target age must be between 1 and 100'},
            status=status.HTTP_400_BAD_REQUEST
        )

    start_time = time.time()

    # Save uploaded image to a temp location for processing
    temp_dir = os.path.join(settings.MEDIA_ROOT, 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    ext = os.path.splitext(image.name)[1] or '.jpg'
    temp_name = f"{uuid.uuid4().hex}{ext}"
    temp_path = os.path.join(temp_dir, temp_name)

    with open(temp_path, 'wb+') as f:
        for chunk in image.chunks():
            f.write(chunk)

    try:
        # -- Step 1: Detect current age using MiVOLO (same model as predict page) --
        age_result = predict_group_faces(temp_path)
        current_age = int(age_result['predicted_age']) or 25
        gender = age_result.get('gender', 'Unknown')

        # -- Step 2: Run the aging pipeline (GAN if available, else OpenCV) --
        pipeline = GANProgressionPipeline()
        result = pipeline.run(temp_path, current_age, target_age,
                              gan_model=gan_model, gender=gender)

        total_time = (time.time() - start_time) * 1000  # ms
        model_type = result.get('model_type', 'Unknown')

        # -- Step 3: Save original image to originals dir --
        originals_dir = os.path.join(settings.MEDIA_ROOT, 'originals')
        os.makedirs(originals_dir, exist_ok=True)
        orig_save_name = f"{uuid.uuid4().hex}{ext}"
        orig_save_path = os.path.join(originals_dir, orig_save_name)

        image.seek(0)
        with open(orig_save_path, 'wb+') as f:
            for chunk in image.chunks():
                f.write(chunk)

        original_relative_path = f"originals/{orig_save_name}"

        # Copy progressed image to progressions dir
        progressions_dir = os.path.join(settings.MEDIA_ROOT, 'progressions')
        os.makedirs(progressions_dir, exist_ok=True)
        progressed_path = result['output_path']
        prog_ext = os.path.splitext(progressed_path)[1] or '.jpg'
        prog_save_name = f"{uuid.uuid4().hex}{prog_ext}"
        prog_save_path = os.path.join(progressions_dir, prog_save_name)

        shutil.copy2(progressed_path, prog_save_path)
        progressed_relative_path = f"progressions/{prog_save_name}"

        # -- Step 4: Save the DB record in MongoDB --
        record = MongoProgressionManager.create(
            user_id=request.user.id,
            original_image_path=original_relative_path,
            progressed_image_path=progressed_relative_path,
            current_age=current_age,
            target_age=target_age,
            model_used=model_type,
            processing_time_ms=round(total_time, 2),
            gender=gender,
            pipeline_steps=result['steps'],
            aging_insights=result['insights'],
        )

        serializer = ProgressionSerializer(record, context={'request': request})

        return Response({
            'message': 'Age progression completed successfully',
            'progression': serializer.data,
            'steps': result['steps'],
            'insights': result['insights'],
        }, status=status.HTTP_200_OK)

    except Exception as e:
        import traceback, io
        try:
            traceback.print_exc()
        except OSError:
            # Windows stderr may not support tqdm escape sequences
            buf = io.StringIO()
            traceback.print_exc(file=buf)
            print(buf.getvalue())
        return Response(
            {'error': f'Progression failed: {str(e)}'},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
@parser_classes([MultiPartParser, FormParser])
def progress_stream_view(request):
    """
    SSE streaming version of the progress endpoint.
    Streams pipeline step updates in real-time as they happen.

    Events:
      - event: step   → a pipeline step started or completed
      - event: age    → age detection result
      - event: result → final result with images + insights
      - event: error  → something went wrong
    """
    image = request.FILES.get('image')
    target_age = request.data.get('target_age')
    gan_model = request.data.get('gan_model', 'sam')

    if not image:
        return Response({'error': 'No image provided'},
                        status=status.HTTP_400_BAD_REQUEST)
    if not target_age:
        return Response({'error': 'Target age is required'},
                        status=status.HTTP_400_BAD_REQUEST)
    try:
        target_age = int(target_age)
    except (ValueError, TypeError):
        return Response({'error': 'Target age must be a valid integer'},
                        status=status.HTTP_400_BAD_REQUEST)
    if target_age < 1 or target_age > 100:
        return Response({'error': 'Target age must be between 1 and 100'},
                        status=status.HTTP_400_BAD_REQUEST)

    # Save uploaded image to temp
    temp_dir = os.path.join(settings.MEDIA_ROOT, 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    ext = os.path.splitext(image.name)[1] or '.jpg'
    temp_name = f"{uuid.uuid4().hex}{ext}"
    temp_path = os.path.join(temp_dir, temp_name)

    with open(temp_path, 'wb+') as f:
        for chunk in image.chunks():
            f.write(chunk)

    # Read image bytes for later saving
    image.seek(0)
    image_bytes = image.read()

    # Queue for SSE events between pipeline thread and response stream
    event_queue = queue.Queue()
    user_id = request.user.id

    def step_callback(step_data):
        """Called by GANProgressionPipeline as each step starts/completes."""
        event_queue.put(('step', step_data))

    def run_pipeline():
        """Execute the full pipeline in a background thread."""
        start_time = time.time()
        try:
            # Step 0: Age Detection
            event_queue.put(('step', {
                'label': 'Age Detection (MiVOLO)',
                'icon': '\U0001f9d1',
                'status': 'running',
            }))
            age_t0 = time.time()
            age_result = predict_group_faces(temp_path)
            current_age = int(age_result['predicted_age']) or 25
            gender = age_result.get('gender', 'Unknown')
            age_time = round((time.time() - age_t0) * 1000, 2)

            event_queue.put(('step', {
                'label': 'Age Detection (MiVOLO)',
                'icon': '\U0001f9d1',
                'status': 'done',
                'time_ms': age_time,
            }))
            event_queue.put(('age', {
                'current_age': current_age,
                'gender': gender,
            }))

            # Run aging pipeline with step callback
            pipeline = GANProgressionPipeline(step_callback=step_callback)
            result = pipeline.run(temp_path, current_age, target_age,
                                  gan_model=gan_model, gender=gender)

            total_time = (time.time() - start_time) * 1000
            model_type = result.get('model_type', 'Unknown')

            # Save images
            originals_dir = os.path.join(settings.MEDIA_ROOT, 'originals')
            os.makedirs(originals_dir, exist_ok=True)
            orig_save_name = f"{uuid.uuid4().hex}{ext}"
            orig_save_path = os.path.join(originals_dir, orig_save_name)
            with open(orig_save_path, 'wb') as f:
                f.write(image_bytes)
            original_relative_path = f"originals/{orig_save_name}"

            progressions_dir = os.path.join(settings.MEDIA_ROOT, 'progressions')
            os.makedirs(progressions_dir, exist_ok=True)
            progressed_path = result['output_path']
            prog_ext = os.path.splitext(progressed_path)[1] or '.jpg'
            prog_save_name = f"{uuid.uuid4().hex}{prog_ext}"
            prog_save_path = os.path.join(progressions_dir, prog_save_name)
            shutil.copy2(progressed_path, prog_save_path)
            progressed_relative_path = f"progressions/{prog_save_name}"

            # Build all pipeline steps (age detection + pipeline steps)
            all_steps = [{
                'label': 'Age Detection (MiVOLO)',
                'icon': '\U0001f9d1',
                'status': 'done',
                'time_ms': age_time,
            }] + result['steps']

            # Save to MongoDB
            record = MongoProgressionManager.create(
                user_id=user_id,
                original_image_path=original_relative_path,
                progressed_image_path=progressed_relative_path,
                current_age=current_age,
                target_age=target_age,
                model_used=model_type,
                processing_time_ms=round(total_time, 2),
                gender=gender,
                pipeline_steps=all_steps,
                aging_insights=result['insights'],
            )

            # Build image URLs
            base_url = f"http://localhost:8000{settings.MEDIA_URL}"
            event_queue.put(('result', {
                'progression': {
                    'id': str(record.get('_id', '')),
                    'original_image_url': f"{base_url}{original_relative_path}",
                    'progressed_image_url': f"{base_url}{progressed_relative_path}",
                    'current_age': current_age,
                    'target_age': target_age,
                    'model_used': model_type,
                    'processing_time_ms': round(total_time, 2),
                    'gender': gender,
                },
                'steps': all_steps,
                'insights': result['insights'],
            }))

        except Exception as e:
            import traceback
            traceback.print_exc()
            event_queue.put(('error', {'error': str(e)}))
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            event_queue.put(('done', None))

    def event_stream():
        """Generator that yields SSE formatted events."""
        # Start pipeline in background thread
        thread = threading.Thread(target=run_pipeline, daemon=True)
        thread.start()

        while True:
            try:
                event_type, data = event_queue.get(timeout=600)
            except queue.Empty:
                yield "event: error\ndata: {\"error\": \"Pipeline timeout\"}\n\n"
                break

            if event_type == 'done':
                break

            yield f"event: {event_type}\ndata: {json.dumps(data)}\n\n"

    response = StreamingHttpResponse(
        event_stream(),
        content_type='text/event-stream',
    )
    response['Cache-Control'] = 'no-cache'
    response['X-Accel-Buffering'] = 'no'
    return response
