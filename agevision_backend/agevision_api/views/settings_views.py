from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from ..mongodb import MongoUserSettingsManager
from ..serializers import UserSettingsSerializer


@api_view(['GET', 'PUT'])
@permission_classes([IsAuthenticated])
def settings_view(request):
    """Get or update user settings from MongoDB."""
    user_id = request.user.id

    if request.method == 'GET':
        user_settings = MongoUserSettingsManager.get_or_create(user_id)
        serializer = UserSettingsSerializer(user_settings)
        return Response(serializer.data)

    elif request.method == 'PUT':
        serializer = UserSettingsSerializer(data=request.data, partial=True)
        if serializer.is_valid():
            updated = MongoUserSettingsManager.update(user_id, **serializer.validated_data)
            return Response(UserSettingsSerializer(updated).data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
