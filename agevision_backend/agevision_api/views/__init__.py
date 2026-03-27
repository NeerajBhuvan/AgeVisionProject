from .auth_views import (
    register_view, login_view, profile_view, profile_update_view,
    forgot_password_view, recover_password_view, reset_password_view,
    change_password_view,
)
from .predict_views import predict_view, predict_camera_view
from .progress_views import progress_view
from .history_views import history_view, history_delete_view
from .analytics_views import analytics_view
from .settings_views import settings_view

__all__ = [
    'register_view',
    'login_view',
    'profile_view',
    'profile_update_view',
    'forgot_password_view',
    'recover_password_view',
    'reset_password_view',
    'change_password_view',
    'predict_view',
    'predict_camera_view',
    'progress_view',
    'history_view',
    'history_delete_view',
    'analytics_view',
    'settings_view',
]
