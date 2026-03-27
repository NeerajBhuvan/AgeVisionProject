from django.urls import path
from rest_framework_simplejwt.views import TokenRefreshView
from .views import (
    register_view,
    login_view,
    profile_view,
    profile_update_view,
    forgot_password_view,
    recover_password_view,
    reset_password_view,
    change_password_view,
    predict_view,
    predict_camera_view,
    progress_view,
    history_view,
    history_delete_view,
    analytics_view,
    settings_view,
)

urlpatterns = [
    # Auth
    path('auth/register/', register_view, name='register'),
    path('auth/login/', login_view, name='login'),
    path('auth/profile/', profile_view, name='profile'),
    path('auth/profile/update/', profile_update_view, name='profile-update'),
    path('auth/token/refresh/', TokenRefreshView.as_view(), name='token-refresh'),
    path('auth/forgot-password/', forgot_password_view, name='forgot-password'),
    path('auth/recover-password/', recover_password_view, name='recover-password'),
    path('auth/reset-password/', reset_password_view, name='reset-password'),
    path('auth/change-password/', change_password_view, name='change-password'),

    # Core Features
    path('predict/', predict_view, name='predict'),
    path('predict/camera/', predict_camera_view, name='predict-camera'),
    path('progress/', progress_view, name='progress'),

    # History
    path('history/', history_view, name='history'),
    path('history/<str:pk>/', history_delete_view, name='history-delete'),

    # Analytics
    path('analytics/', analytics_view, name='analytics'),

    # Settings
    path('settings/', settings_view, name='settings'),
]
