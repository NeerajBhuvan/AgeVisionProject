"""
Serializers for the age_progression app.
=========================================
Since we use MongoDB (not Django ORM models), these are plain
``rest_framework.serializers.Serializer`` classes rather than
``ModelSerializer``.
"""

from rest_framework import serializers


class AgeProgressionInputSerializer(serializers.Serializer):
    """Validate the incoming POST data for the progression endpoint."""

    image = serializers.ImageField(
        required=True,
        help_text='Face image to age-progress (JPEG/PNG, max 10 MB).',
    )
    target_age = serializers.IntegerField(
        required=True,
        min_value=1,
        max_value=100,
        help_text='Desired target age (1–100).',
    )


class AgeProgressionOutputSerializer(serializers.Serializer):
    """Shape of the successful response body."""

    success = serializers.BooleanField()
    record_id = serializers.CharField()
    current_age = serializers.IntegerField()
    target_age = serializers.IntegerField()
    gender = serializers.CharField()
    method_used = serializers.CharField()
    processing_time_ms = serializers.FloatField()
    original_image_url = serializers.URLField()
    progressed_image_url = serializers.URLField()
    comparison_image_url = serializers.URLField()


class ProgressionRecordSerializer(serializers.Serializer):
    """Serialise a MongoDB progression document for list / detail views."""

    id = serializers.CharField(source='_id')
    user_id = serializers.CharField()
    current_age = serializers.IntegerField()
    target_age = serializers.IntegerField()
    processing_time = serializers.FloatField()
    method_used = serializers.CharField()
    status = serializers.CharField()
    created_at = serializers.DateTimeField()
    original_image_url = serializers.SerializerMethodField()
    progressed_image_url = serializers.SerializerMethodField()
    comparison_image_url = serializers.SerializerMethodField()

    def _build_url(self, path: str) -> str:
        request = self.context.get('request')
        if request and path:
            from django.conf import settings
            media_url = settings.MEDIA_URL
            # Ensure path is relative
            if path.startswith(str(settings.MEDIA_ROOT)):
                import os
                path = os.path.relpath(path, settings.MEDIA_ROOT).replace('\\', '/')
            return request.build_absolute_uri(f'{media_url}{path}')
        return ''

    def get_original_image_url(self, obj):
        return self._build_url(obj.get('original_image_path', ''))

    def get_progressed_image_url(self, obj):
        return self._build_url(obj.get('progressed_image_path', ''))

    def get_comparison_image_url(self, obj):
        return self._build_url(obj.get('comparison_image_path', ''))
