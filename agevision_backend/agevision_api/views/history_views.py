from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from ..mongodb import MongoPredictionManager, MongoProgressionManager
from ..serializers import PredictionSerializer, ProgressionSerializer


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def history_view(request):
    """Get all prediction and progression records for logged-in user from MongoDB."""
    user_id = request.user.id

    predictions = MongoPredictionManager.get_by_user(user_id)
    progressions = MongoProgressionManager.get_by_user(user_id)

    prediction_data = PredictionSerializer(
        predictions, many=True, context={'request': request}
    ).data
    progression_data = ProgressionSerializer(
        progressions, many=True, context={'request': request}
    ).data

    return Response({
        'predictions': prediction_data,
        'progressions': progression_data,
        'total_predictions': len(predictions),
        'total_progressions': len(progressions),
    })


@api_view(['DELETE'])
@permission_classes([IsAuthenticated])
def history_delete_view(request, pk):
    """Delete a prediction or progression record by ID (MongoDB ObjectId string)."""
    user_id = request.user.id

    # Try deleting from predictions first
    if MongoPredictionManager.delete(pk, user_id):
        return Response(
            {'message': 'Record deleted successfully'},
            status=status.HTTP_204_NO_CONTENT
        )

    # Try progression records
    if MongoProgressionManager.delete(pk, user_id):
        return Response(
            {'message': 'Record deleted successfully'},
            status=status.HTTP_204_NO_CONTENT
        )

    return Response(
        {'error': 'Record not found'},
        status=status.HTTP_404_NOT_FOUND
    )
