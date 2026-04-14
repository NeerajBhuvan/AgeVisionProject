from .auth_views import (
    register_view, login_view, profile_view, profile_update_view,
    forgot_password_view, recover_password_view, reset_password_view,
    change_password_view,
)
from .predict_views import predict_view, predict_camera_view, batch_predict_view
from .progress_views import progress_view, progress_stream_view
from .history_views import history_view, history_delete_view
from .analytics_views import analytics_view
from .settings_views import settings_view
from .admin_views import (
    admin_dashboard_view,
    admin_users_view,
    admin_user_detail_view,
    admin_user_suspend_view,
    admin_user_reinstate_view,
    admin_system_health_view,
)
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
    'batch_predict_view',
    'progress_view',
    'progress_stream_view',
    'history_view',
    'history_delete_view',
    'analytics_view',
    'settings_view',
    'admin_dashboard_view',
    'admin_users_view',
    'admin_user_detail_view',
    'admin_user_suspend_view',
    'admin_user_reinstate_view',
    'admin_system_health_view',
]
