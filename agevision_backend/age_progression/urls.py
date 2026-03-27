"""
URL configuration for the age_progression app.
"""

from django.urls import path
from . import views

app_name = 'age_progression'

urlpatterns = [
    # Core progression endpoint
    path('progress/', views.progress_age_view, name='progress'),

    # History
    path('history/', views.history_list_view, name='history-list'),
    path('history/<str:record_id>/', views.history_detail_view, name='history-detail'),

    # Statistics
    path('stats/', views.stats_view, name='stats'),

    # Health check
    path('health/', views.health_view, name='health'),
]
