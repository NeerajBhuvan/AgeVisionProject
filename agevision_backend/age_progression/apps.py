from django.apps import AppConfig


class AgeProgressionConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'age_progression'
    verbose_name = 'Age Progression Module'

    def ready(self):
        """Bootstrap MongoDB indexes on startup."""
        try:
            from .models import ProgressionDB
            ProgressionDB.ensure_indexes()
        except Exception:
            pass  # MongoDB may not be running during migrations
