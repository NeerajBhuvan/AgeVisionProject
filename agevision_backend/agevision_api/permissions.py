"""Custom DRF permission classes for AgeVision."""

from rest_framework.permissions import BasePermission


class IsSuperUser(BasePermission):
    """Allow access only to authenticated Django superusers."""

    message = 'Superuser privileges required.'

    def has_permission(self, request, view):
        user = getattr(request, 'user', None)
        return bool(user and user.is_authenticated and user.is_superuser)
