from django.db import models
from django.contrib.auth.models import User


class PredictionRecord(models.Model):
    user = models.ForeignKey(
        User, on_delete=models.CASCADE, null=True, blank=True
    )
    image = models.ImageField(upload_to='predictions/')
    predicted_age = models.IntegerField()
    confidence = models.FloatField(default=0.0)
    gender = models.CharField(max_length=20, default='Unknown')
    emotion = models.CharField(max_length=30, default='Unknown')
    race = models.CharField(max_length=30, default='Unknown')
    face_count = models.IntegerField(default=1)
    processing_time_ms = models.FloatField(default=0.0)
    detector_used = models.CharField(max_length=30, default='opencv')
    ensemble_ages = models.JSONField(default=list, blank=True)
    age_std = models.FloatField(default=0.0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"Prediction - Age: {self.predicted_age} - {self.created_at}"


class ProgressionRecord(models.Model):
    user = models.ForeignKey(
        User, on_delete=models.CASCADE, null=True, blank=True
    )
    original_image = models.ImageField(upload_to='originals/')
    progressed_image = models.ImageField(
        upload_to='progressions/', null=True, blank=True
    )
    current_age = models.IntegerField(default=0)
    target_age = models.IntegerField()
    model_used = models.CharField(max_length=50, default='StyleGAN3')
    processing_time_ms = models.FloatField(default=0.0)
    gender = models.CharField(max_length=20, default='Unknown')
    pipeline_steps = models.TextField(default='[]')     # JSON list of step dicts
    aging_insights = models.TextField(default='[]')     # JSON list of insight dicts
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"Progression - {self.current_age} → {self.target_age}"


class UserSettings(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    theme = models.CharField(max_length=10, default='dark')
    default_model = models.CharField(
        max_length=50, default='DeepFace v3'
    )
    notifications_enabled = models.BooleanField(default=True)
    auto_detect_faces = models.BooleanField(default=True)
    save_to_history = models.BooleanField(default=True)
    high_accuracy_mode = models.BooleanField(default=False)
    show_confidence = models.BooleanField(default=True)
    language = models.CharField(max_length=20, default='English')
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Settings for {self.user.username}"
