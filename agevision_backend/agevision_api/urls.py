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
    batch_predict_view,
    progress_view,
    progress_stream_view,
    history_view,
    history_delete_view,
    analytics_view,
    settings_view,
    admin_dashboard_view,
    admin_users_view,
    admin_user_detail_view,
    admin_user_suspend_view,
    admin_user_reinstate_view,
    admin_system_health_view,
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
    path('predict/batch/', batch_predict_view, name='predict-batch'),
    path('progress/', progress_view, name='progress'),
    path('progress/stream/', progress_stream_view, name='progress-stream'),

    # History
    path('history/', history_view, name='history'),
    path('history/<str:pk>/', history_delete_view, name='history-delete'),

    # Analytics
    path('analytics/', analytics_view, name='analytics'),

    # Settings
    path('settings/', settings_view, name='settings'),

    # Admin Panel (superuser only)
    path('admin/dashboard/', admin_dashboard_view, name='admin-dashboard'),
    path('admin/users/', admin_users_view, name='admin-users'),
    path('admin/users/<int:user_id>/', admin_user_detail_view, name='admin-user-detail'),
    path('admin/users/<int:user_id>/suspend/', admin_user_suspend_view, name='admin-user-suspend'),
    path('admin/users/<int:user_id>/reinstate/', admin_user_reinstate_view, name='admin-user-reinstate'),
    path('admin/system/health/', admin_system_health_view, name='admin-system-health'),
]
