from django.contrib.auth.models import User

from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from rest_framework_simplejwt.tokens import RefreshToken

from ..serializers import RegisterSerializer, UserSerializer
from ..mongodb import MongoUserManager, MongoDB, MongoPasswordResetManager


@api_view(['POST'])
@permission_classes([AllowAny])
def register_view(request):
    """Register a new user (Django + MongoDB)."""
    serializer = RegisterSerializer(data=request.data)
    if serializer.is_valid():
        user = serializer.save()
        raw_password = request.data.get('password', '')

        # Sync to MongoDB (with encrypted password)
        try:
            MongoDB.ensure_indexes()
            mongo_user = MongoUserManager.create_user(
                django_user_id=user.id,
                username=user.username,
                email=user.email,
                first_name=user.first_name,
                last_name=user.last_name,
                raw_password=raw_password,
            )
        except Exception:
            mongo_user = {
                'id': user.id,
                'username': user.username,
                'email': user.email,
                'first_name': user.first_name,
                'last_name': user.last_name,
                'plan': 'free',
            }

        if isinstance(mongo_user, dict):
            mongo_user['is_admin'] = bool(user.is_superuser)

        refresh = RefreshToken.for_user(user)
        return Response({
            'message': 'Registration successful',
            'user': mongo_user,
            'tokens': {
                'access': str(refresh.access_token),
                'refresh': str(refresh),
            }
        }, status=status.HTTP_201_CREATED)
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


@api_view(['POST'])
@permission_classes([AllowAny])
def login_view(request):
    """Login with username/email and return JWT tokens + MongoDB profile."""
    username = request.data.get('username')
    password = request.data.get('password')

    if not username or not password:
        return Response(
            {'error': 'Username and password are required'},
            status=status.HTTP_400_BAD_REQUEST,
        )

    # Support login with email OR username
    try:
        if '@' in username:
            user = User.objects.get(email=username)
        else:
            user = User.objects.get(username=username)
    except User.DoesNotExist:
        return Response(
            {'error': 'Invalid credentials'},
            status=status.HTTP_401_UNAUTHORIZED,
        )

    if not user.check_password(password):
        return Response(
            {'error': 'Invalid credentials'},
            status=status.HTTP_401_UNAUTHORIZED,
        )

    # Block suspended accounts with a distinct, descriptive response
    if not user.is_active:
        return Response(
            {
                'error': 'Your account has been suspended. Please contact an administrator to restore access.',
                'account_suspended': True,
            },
            status=status.HTTP_403_FORBIDDEN,
        )

    # Get user profile from MongoDB (fall back to Django data)
    try:
        mongo_user = MongoUserManager.get_by_django_id(user.id)
    except Exception:
        mongo_user = None
    if mongo_user is None:
        # First-time login or MongoDB was cleared — re-sync
        try:
            MongoDB.ensure_indexes()
            mongo_user = MongoUserManager.create_user(
                django_user_id=user.id,
                username=user.username,
                email=user.email,
                first_name=user.first_name,
                last_name=user.last_name,
                raw_password=password,
            )
        except Exception:
            mongo_user = UserSerializer(user).data
    else:
        # Update encrypted password if it was missing (legacy users)
        try:
            MongoUserManager.store_encrypted_password(user.id, password, user.username)
        except Exception:
            pass

    # Record last login timestamp in MongoDB
    try:
        MongoUserManager.update_last_login(user.id)
    except Exception:
        pass

    if isinstance(mongo_user, dict):
        mongo_user['is_admin'] = bool(user.is_superuser)

    refresh = RefreshToken.for_user(user)
    return Response({
        'message': 'Login successful',
        'user': mongo_user,
        'tokens': {
            'access': str(refresh.access_token),
            'refresh': str(refresh),
        }
    })


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def profile_view(request):
    """Return user profile from MongoDB, falling back to Django."""
    try:
        mongo_user = MongoUserManager.get_by_django_id(request.user.id)
    except Exception:
        mongo_user = None
    if mongo_user:
        mongo_user['is_admin'] = bool(request.user.is_superuser)
        return Response(mongo_user)
    data = UserSerializer(request.user).data
    if isinstance(data, dict):
        data['is_admin'] = bool(request.user.is_superuser)
    return Response(data)


@api_view(['PUT', 'PATCH'])
@permission_classes([IsAuthenticated])
def profile_update_view(request):
    """Update user profile fields in MongoDB."""
    allowed = {'first_name', 'last_name', 'email', 'avatar_url'}
    updates = {k: v for k, v in request.data.items() if k in allowed}
    if not updates:
        return Response(
            {'error': 'No valid fields to update'},
            status=status.HTTP_400_BAD_REQUEST,
        )

    # Also update Django User for fields it owns
    django_fields = {'first_name', 'last_name', 'email'}
    django_updates = {k: v for k, v in updates.items() if k in django_fields}
    if django_updates:
        User.objects.filter(pk=request.user.id).update(**django_updates)

    try:
        mongo_user = MongoUserManager.update_user(request.user.id, **updates)
    except Exception:
        mongo_user = None
    if mongo_user:
        return Response(mongo_user)
    return Response(UserSerializer(request.user).data)


# ── Password Recovery (MongoDB-based, no email) ─────────────────────

@api_view(['POST'])
@permission_classes([AllowAny])
def forgot_password_view(request):
    """
    Step 1 — Look up the user by email, generate a reset token stored
    in MongoDB, and return a password hint + the token to the client.
    Everything stays in-app; no email is sent.
    """
    email = request.data.get('email', '').strip()
    if not email:
        return Response(
            {'error': 'Email is required'},
            status=status.HTTP_400_BAD_REQUEST,
        )

    try:
        django_user = User.objects.get(email=email)
    except User.DoesNotExist:
        # Don't reveal whether the email exists
        return Response({
            'message': 'If an account with that email exists, recovery info has been sent.',
        })

    # Recover password from MongoDB encrypted store to build a hint
    recovered = MongoUserManager.recover_password(django_user.id)
    if recovered is None:
        return Response({
            'message': 'If an account with that email exists, recovery info has been sent.',
            'recovery_available': False,
        })

    # Build a masked hint: show first and last char, mask the rest
    if len(recovered) <= 2:
        hint = '*' * len(recovered)
    else:
        hint = recovered[0] + '*' * (len(recovered) - 2) + recovered[-1]

    # Generate a reset token and persist it in MongoDB
    recovery_token = MongoPasswordResetManager.create_reset_token(
        django_user_id=django_user.id,
        email=django_user.email,
        username=django_user.username,
    )

    return Response({
        'message': 'Password recovery info retrieved.',
        'recovery_available': True,
        'password_hint': hint,
        'password_length': len(recovered),
        'recovery_token': recovery_token,
        'username': django_user.username,
    })


@api_view(['POST'])
@permission_classes([AllowAny])
def recover_password_view(request):
    """
    Step 2 (optional) — Reveal the full decrypted password.
    The user must supply both the MongoDB-stored recovery token and their
    email to prevent token theft.
    """
    token = request.data.get('recovery_token', '')
    email = request.data.get('email', '').strip()

    if not token or not email:
        return Response(
            {'error': 'Recovery token and email are required'},
            status=status.HTTP_400_BAD_REQUEST,
        )

    # Verify the token in MongoDB
    token_doc = MongoPasswordResetManager.verify_token(token)
    if token_doc is None:
        return Response(
            {'error': 'Invalid or expired recovery token'},
            status=status.HTTP_401_UNAUTHORIZED,
        )

    # Verify the email matches the token
    if token_doc.get('email', '').lower() != email.lower():
        return Response(
            {'error': 'Email does not match the recovery token'},
            status=status.HTTP_400_BAD_REQUEST,
        )

    user_id = token_doc['django_user_id']

    # Verify the Django user exists
    try:
        django_user = User.objects.get(pk=user_id)
    except User.DoesNotExist:
        return Response(
            {'error': 'User account not found'},
            status=status.HTTP_404_NOT_FOUND,
        )

    # Decrypt from MongoDB
    recovered = MongoUserManager.recover_password(django_user.id)
    if recovered is None:
        return Response(
            {'error': 'Password recovery data not available for this account'},
            status=status.HTTP_404_NOT_FOUND,
        )

    return Response({
        'message': 'Password recovered successfully',
        'password': recovered,
        'username': django_user.username,
    })


@api_view(['POST'])
@permission_classes([AllowAny])
def reset_password_view(request):
    """
    Step 3 — Reset the password using the MongoDB-stored recovery token.
    Updates both Django User and MongoDB encrypted store, then
    invalidates the token so it cannot be reused.
    """
    token = request.data.get('recovery_token', '')
    email = request.data.get('email', '').strip()
    new_password = request.data.get('new_password', '')

    if not token or not email or not new_password:
        return Response(
            {'error': 'Recovery token, email, and new password are required'},
            status=status.HTTP_400_BAD_REQUEST,
        )

    if len(new_password) < 8:
        return Response(
            {'error': 'Password must be at least 8 characters'},
            status=status.HTTP_400_BAD_REQUEST,
        )

    # Verify the token in MongoDB
    token_doc = MongoPasswordResetManager.verify_token(token)
    if token_doc is None:
        return Response(
            {'error': 'Invalid or expired recovery token'},
            status=status.HTTP_401_UNAUTHORIZED,
        )

    # Verify the email matches
    if token_doc.get('email', '').lower() != email.lower():
        return Response(
            {'error': 'Email does not match the recovery token'},
            status=status.HTTP_400_BAD_REQUEST,
        )

    user_id = token_doc['django_user_id']

    try:
        django_user = User.objects.get(pk=user_id)
    except User.DoesNotExist:
        return Response(
            {'error': 'User account not found'},
            status=status.HTTP_404_NOT_FOUND,
        )

    # Update Django user password (hashed)
    django_user.set_password(new_password)
    django_user.save()

    # Update MongoDB encrypted password
    try:
        MongoUserManager.store_encrypted_password(
            django_user.id, new_password, django_user.username
        )
    except Exception:
        pass

    # Invalidate the token in MongoDB so it can't be reused
    MongoPasswordResetManager.invalidate_token(token)

    return Response({
        'message': 'Password reset successfully. You can now log in with your new password.',
    })


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def change_password_view(request):
    """Change password for authenticated user (requires current password)."""
    current_password = request.data.get('current_password', '')
    new_password = request.data.get('new_password', '')

    if not current_password or not new_password:
        return Response(
            {'error': 'Current password and new password are required'},
            status=status.HTTP_400_BAD_REQUEST,
        )

    if len(new_password) < 8:
        return Response(
            {'error': 'New password must be at least 8 characters'},
            status=status.HTTP_400_BAD_REQUEST,
        )

    if not request.user.check_password(current_password):
        return Response(
            {'error': 'Current password is incorrect'},
            status=status.HTTP_400_BAD_REQUEST,
        )

    request.user.set_password(new_password)
    request.user.save()

    # Update MongoDB encrypted password
    try:
        MongoUserManager.store_encrypted_password(
            request.user.id, new_password, request.user.username
        )
    except Exception:
        pass

    return Response({'message': 'Password changed successfully'})
