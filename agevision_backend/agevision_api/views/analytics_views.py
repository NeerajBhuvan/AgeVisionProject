from datetime import datetime, timedelta, timezone

from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from ..mongodb import MongoPredictionManager, MongoProgressionManager


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def analytics_view(request):
    """Return weekly stats and totals for the logged-in user from MongoDB."""
    user_id = request.user.id
    now = datetime.now(timezone.utc)
    week_ago = now - timedelta(days=7)

    # Total counts
    total_predictions = MongoPredictionManager.count(user_id)
    total_progressions = MongoProgressionManager.count(user_id)

    # This week's counts
    week_predictions = MongoPredictionManager.count_since(user_id, week_ago)
    week_progressions = MongoProgressionManager.count_since(user_id, week_ago)

    # Aggregate stats (avg age, avg confidence)
    stats = MongoPredictionManager.aggregate_stats(user_id)
    avg_age = stats['avg_age']
    avg_confidence = stats['avg_confidence']

    # Gender distribution
    gender_stats = MongoPredictionManager.gender_distribution(user_id)

    # Emotion distribution
    emotion_stats = MongoPredictionManager.emotion_distribution(user_id)

    # Daily counts for the past week
    daily_counts = MongoPredictionManager.daily_counts(user_id, days=7)

    return Response({
        'total_predictions': total_predictions,
        'total_progressions': total_progressions,
        'week_predictions': week_predictions,
        'week_progressions': week_progressions,
        'average_predicted_age': avg_age,
        'average_confidence': avg_confidence,
        'gender_distribution': gender_stats,
        'emotion_distribution': emotion_stats,
        'daily_counts': daily_counts,
    })
