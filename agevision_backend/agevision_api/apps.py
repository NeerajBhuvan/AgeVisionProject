import logging

from django.apps import AppConfig

logger = logging.getLogger(__name__)


class AgevisionApiConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'agevision_api'

    def ready(self):
        # Predict (MiVOLO v2 + YOLOv8 + emotion) runs locally on CPU and is
        # loaded lazily on the first /api/predict/ request, so there's nothing
        # to warm up at boot. GAN aging models (SAM/Fast-Aging/HRFAE) are also
        # loaded lazily on the first /api/progress/ request that selects them,
        # so the worker boots fast and idle memory stays low.
        return
