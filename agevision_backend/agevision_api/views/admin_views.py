"""Admin Panel endpoints — superuser-only platform monitoring and user management."""

import os
from pathlib import Path

from django.conf import settings
from django.contrib.auth.models import User

from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response

from ..mongodb import (
    MongoDB,
    MongoUserManager,
    MongoPredictionManager,
    MongoProgressionManager,
)
from ..permissions import IsSuperUser


# ── Dashboard ───────────────────────────────────────────────────────

@api_view(['GET'])
@permission_classes([IsSuperUser])
def admin_dashboard_view(request):
    """Platform-wide stats aggregating data across all users."""
    total_users = User.objects.count()
    active_users = User.objects.filter(is_active=True).count()
    superusers = User.objects.filter(is_superuser=True).count()

    pred_stats = MongoPredictionManager.platform_stats()
    prog_stats = MongoProgressionManager.platform_stats()

    return Response({
        'total_users': total_users,
        'active_users': active_users,
        'superusers': superusers,
        'total_predictions': pred_stats['total'],
        'total_progressions': prog_stats['total'],
        'average_predicted_age': pred_stats['avg_age'],
        'average_confidence': pred_stats['avg_confidence'],
        'avg_prediction_time_ms': pred_stats['avg_processing_time_ms'],
        'avg_progression_time_ms': prog_stats['avg_processing_time_ms'],
        'detector_breakdown': MongoPredictionManager.platform_detector_breakdown(),
        'gan_model_breakdown': MongoProgressionManager.platform_model_breakdown(),
        'gender_distribution': MongoPredictionManager.platform_gender_distribution(),
        'emotion_distribution': MongoPredictionManager.platform_emotion_distribution(),
        'prediction_daily_counts': MongoPredictionManager.platform_daily_counts(days=14),
        'progression_daily_counts': MongoProgressionManager.platform_daily_counts(days=14),
    })


# ── Users ───────────────────────────────────────────────────────────

def _safe_int(value, default):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


@api_view(['GET'])
@permission_classes([IsSuperUser])
def admin_users_view(request):
    """Paginated list of all users, merged Django + MongoDB profile + per-user counts."""
    page = max(_safe_int(request.GET.get('page'), 1), 1)
    limit = min(max(_safe_int(request.GET.get('limit'), 50), 1), 200)
    search = (request.GET.get('search') or '').strip()

    # Primary source of truth is Django User table (includes superusers)
    qs = User.objects.all().order_by('-date_joined')
    if search:
        from django.db.models import Q
        qs = qs.filter(
            Q(username__icontains=search)
            | Q(email__icontains=search)
            | Q(first_name__icontains=search)
            | Q(last_name__icontains=search)
        )

    total = qs.count()
    start = (page - 1) * limit
    end = start + limit
    page_users = list(qs[start:end])

    users_payload = []
    for u in page_users:
        try:
            mongo_profile = MongoUserManager.get_by_django_id(u.id)
        except Exception:
            mongo_profile = None
        try:
            prediction_count = MongoPredictionManager.count(u.id)
        except Exception:
            prediction_count = 0
        try:
            progression_count = MongoProgressionManager.count(u.id)
        except Exception:
            progression_count = 0

        users_payload.append({
            'id': u.id,
            'username': u.username,
            'email': u.email,
            'first_name': u.first_name,
            'last_name': u.last_name,
            'is_active': u.is_active,
            'is_superuser': u.is_superuser,
            'is_staff': u.is_staff,
            'date_joined': u.date_joined.isoformat() if u.date_joined else None,
            'last_login': u.last_login.isoformat() if u.last_login else None,
            'plan': (mongo_profile or {}).get('plan', 'free'),
            'avatar_url': (mongo_profile or {}).get('avatar_url', ''),
            'prediction_count': prediction_count,
            'progression_count': progression_count,
        })

    return Response({
        'users': users_payload,
        'total': total,
        'page': page,
        'limit': limit,
        'pages': (total + limit - 1) // limit if limit else 1,
    })


@api_view(['GET'])
@permission_classes([IsSuperUser])
def admin_user_detail_view(request, user_id):
    """Full profile plus recent 10 predictions and progressions for a user."""
    try:
        u = User.objects.get(pk=user_id)
    except User.DoesNotExist:
        return Response({'error': 'User not found'}, status=status.HTTP_404_NOT_FOUND)

    try:
        mongo_profile = MongoUserManager.get_by_django_id(u.id)
    except Exception:
        mongo_profile = None

    try:
        recent_predictions = MongoPredictionManager.get_by_user(u.id, limit=10)
    except Exception:
        recent_predictions = []
    try:
        recent_progressions = MongoProgressionManager.get_by_user(u.id, limit=10)
    except Exception:
        recent_progressions = []
    try:
        prediction_count = MongoPredictionManager.count(u.id)
        progression_count = MongoProgressionManager.count(u.id)
    except Exception:
        prediction_count = progression_count = 0

    return Response({
        'user': {
            'id': u.id,
            'username': u.username,
            'email': u.email,
            'first_name': u.first_name,
            'last_name': u.last_name,
            'is_active': u.is_active,
            'is_superuser': u.is_superuser,
            'is_staff': u.is_staff,
            'date_joined': u.date_joined.isoformat() if u.date_joined else None,
            'last_login': u.last_login.isoformat() if u.last_login else None,
            'plan': (mongo_profile or {}).get('plan', 'free'),
            'avatar_url': (mongo_profile or {}).get('avatar_url', ''),
        },
        'prediction_count': prediction_count,
        'progression_count': progression_count,
        'recent_predictions': recent_predictions,
        'recent_progressions': recent_progressions,
    })


def _set_user_active(request, user_id, new_state: bool, action_label: str):
    try:
        target = User.objects.get(pk=user_id)
    except User.DoesNotExist:
        return Response({'error': 'User not found'}, status=status.HTTP_404_NOT_FOUND)

    # Guard against operating on self or other superusers
    if target.id == request.user.id:
        return Response(
            {'error': f'You cannot {action_label} your own account.'},
            status=status.HTTP_400_BAD_REQUEST,
        )
    if target.is_superuser:
        return Response(
            {'error': f'Cannot {action_label} another superuser.'},
            status=status.HTTP_400_BAD_REQUEST,
        )

    target.is_active = new_state
    target.save(update_fields=['is_active'])

    try:
        MongoUserManager.set_active(target.id, new_state)
    except Exception:
        pass

    return Response({
        'message': f'User {action_label}d successfully.',
        'user_id': target.id,
        'is_active': target.is_active,
    })


@api_view(['POST'])
@permission_classes([IsSuperUser])
def admin_user_suspend_view(request, user_id):
    """Suspend a user (sets is_active=False). Existing refresh tokens are not
    explicitly blacklisted — Django's is_active check blocks new logins and
    SimpleJWT rejects tokens for inactive users on next request."""
    return _set_user_active(request, user_id, False, 'suspen')


@api_view(['POST'])
@permission_classes([IsSuperUser])
def admin_user_reinstate_view(request, user_id):
    """Reinstate a previously suspended user."""
    return _set_user_active(request, user_id, True, 'reinstate')


# ── System health ───────────────────────────────────────────────────

def _dir_size_bytes(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += (Path(root) / name).stat().st_size
            except OSError:
                pass
    return total


@api_view(['GET'])
@permission_classes([IsSuperUser])
def admin_system_health_view(request):
    """MongoDB status, checkpoints/uploads disk usage, GPU memory."""
    # MongoDB health
    mongodb_status = 'connected'
    mongodb_error = None
    try:
        MongoDB.get_client().admin.command('ping')
    except Exception as exc:  # pragma: no cover — depends on runtime
        mongodb_status = 'error'
        mongodb_error = str(exc)

    # Model checkpoints directory (size per file)
    checkpoints_dir = Path(settings.BASE_DIR) / 'checkpoints'
    checkpoints_info = []
    total_checkpoints_bytes = 0
    if checkpoints_dir.exists():
        for entry in sorted(checkpoints_dir.iterdir()):
            try:
                if entry.is_file():
                    size = entry.stat().st_size
                    checkpoints_info.append({'name': entry.name, 'size_bytes': size})
                    total_checkpoints_bytes += size
                elif entry.is_dir():
                    size = _dir_size_bytes(entry)
                    checkpoints_info.append({'name': entry.name + '/', 'size_bytes': size})
                    total_checkpoints_bytes += size
            except OSError:
                continue

    # Uploads directory size
    uploads_bytes = _dir_size_bytes(Path(settings.MEDIA_ROOT))

    # GPU info via torch (optional)
    gpu_info = {
        'available': False,
        'name': None,
        'memory_total_bytes': 0,
        'memory_allocated_bytes': 0,
    }
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            gpu_info = {
                'available': True,
                'name': props.name,
                'memory_total_bytes': int(props.total_memory),
                'memory_allocated_bytes': int(torch.cuda.memory_allocated(0)),
            }
    except Exception:
        pass

    return Response({
        'mongodb': {
            'status': mongodb_status,
            'error': mongodb_error,
        },
        'checkpoints': {
            'total_bytes': total_checkpoints_bytes,
            'files': checkpoints_info,
        },
        'uploads': {
            'total_bytes': uploads_bytes,
            'path': str(settings.MEDIA_ROOT),
        },
        'gpu': gpu_info,
    })
