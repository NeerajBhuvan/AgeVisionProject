from rest_framework import serializers
from django.contrib.auth.models import User

from .mongodb import MongoUserSettingsManager


class RegisterSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True, min_length=8)

    class Meta:
        model = User
        fields = ['username', 'email', 'password', 'first_name', 'last_name']

    def create(self, validated_data):
        user = User.objects.create_user(
            username=validated_data['username'],
            email=validated_data['email'],
            password=validated_data['password'],
            first_name=validated_data.get('first_name', ''),
            last_name=validated_data.get('last_name', ''),
        )
        # Create default settings in MongoDB
        try:
            MongoUserSettingsManager.get_or_create(user.id)
        except Exception:
            pass
        return user


class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['id', 'username', 'email', 'first_name', 'last_name']


class PredictionSerializer(serializers.Serializer):
    """Serializer for MongoDB prediction documents (dict-based)."""
    id = serializers.CharField(read_only=True)
    user_id = serializers.IntegerField(read_only=True)
    image_path = serializers.CharField(read_only=True)
    image_url = serializers.SerializerMethodField()
    predicted_age = serializers.IntegerField(read_only=True)
    confidence = serializers.FloatField(read_only=True)
    gender = serializers.CharField(read_only=True)
    emotion = serializers.CharField(read_only=True)
    race = serializers.CharField(read_only=True)
    face_count = serializers.IntegerField(read_only=True)
    processing_time_ms = serializers.FloatField(read_only=True)
    detector_used = serializers.CharField(read_only=True)
    ensemble_ages = serializers.ListField(read_only=True)
    age_std = serializers.FloatField(read_only=True)
    created_at = serializers.CharField(read_only=True)

    def get_image_url(self, obj):
        request = self.context.get('request')
        image_path = obj.get('image_path', '')
        if image_path and request:
            # Build URL from relative media path
            from django.conf import settings as django_settings
            media_url = django_settings.MEDIA_URL
            return request.build_absolute_uri(f'{media_url}{image_path}')
        return None


class ProgressionSerializer(serializers.Serializer):
    """Serializer for MongoDB progression documents (dict-based)."""
    id = serializers.CharField(read_only=True)
    user_id = serializers.IntegerField(read_only=True)
    original_image_path = serializers.CharField(read_only=True)
    progressed_image_path = serializers.CharField(read_only=True)
    original_image_url = serializers.SerializerMethodField()
    progressed_image_url = serializers.SerializerMethodField()
    current_age = serializers.IntegerField(read_only=True)
    target_age = serializers.IntegerField(read_only=True)
    model_used = serializers.CharField(read_only=True)
    processing_time_ms = serializers.FloatField(read_only=True)
    gender = serializers.CharField(read_only=True)
    pipeline_steps = serializers.ListField(read_only=True)
    aging_insights = serializers.ListField(read_only=True)
    created_at = serializers.CharField(read_only=True)

    def get_original_image_url(self, obj):
        request = self.context.get('request')
        image_path = obj.get('original_image_path', '')
        if image_path and request:
            from django.conf import settings as django_settings
            media_url = django_settings.MEDIA_URL
            return request.build_absolute_uri(f'{media_url}{image_path}')
        return None

    def get_progressed_image_url(self, obj):
        request = self.context.get('request')
        image_path = obj.get('progressed_image_path', '')
        if image_path and request:
            from django.conf import settings as django_settings
            media_url = django_settings.MEDIA_URL
            return request.build_absolute_uri(f'{media_url}{image_path}')
        return None


class UserSettingsSerializer(serializers.Serializer):
    """Serializer for MongoDB user settings documents (dict-based)."""
    id = serializers.CharField(read_only=True)
    user_id = serializers.IntegerField(read_only=True)
    theme = serializers.CharField(required=False)
    default_model = serializers.CharField(required=False)
    notifications_enabled = serializers.BooleanField(required=False)
    auto_detect_faces = serializers.BooleanField(required=False)
    save_to_history = serializers.BooleanField(required=False)
    high_accuracy_mode = serializers.BooleanField(required=False)
    show_confidence = serializers.BooleanField(required=False)
    language = serializers.CharField(required=False)
    timezone = serializers.CharField(required=False)
    updated_at = serializers.CharField(read_only=True)
