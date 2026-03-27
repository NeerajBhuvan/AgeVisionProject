"""
End-to-end test for auth views + MongoDB storage.
Run from agevision_backend folder with the venv activated:
    python test_auth.py
"""
import django, os, json, sys

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'agevision_backend.settings')
django.setup()

from django.test import RequestFactory
from agevision_api.views.auth_views import register_view, login_view, profile_view
from agevision_api.mongodb import MongoUserManager, MongoDB
from django.contrib.auth.models import User

factory = RequestFactory()
PASS = 0
FAIL = 0


def check(label, condition):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {label}")
    else:
        FAIL += 1
        print(f"  [FAIL] {label}")


# ── TEST 1: Register ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("TEST 1: Register a new user")
print("=" * 60)
req = factory.post(
    '/api/auth/register/',
    data=json.dumps({
        'username': 'testuser1',
        'email': 'test1@agevision.com',
        'password': 'TestPass123!',
        'first_name': 'Test',
        'last_name': 'User',
    }),
    content_type='application/json',
)
resp = register_view(req)
check("Status 201", resp.status_code == 201)
check("Message present", resp.data.get('message') == 'Registration successful')
check("User object has username", resp.data['user']['username'] == 'testuser1')
check("Tokens present", 'tokens' in resp.data)
check("Access token present", len(resp.data['tokens']['access']) > 20)
check("Refresh token present", len(resp.data['tokens']['refresh']) > 20)
print(f"  User data: {json.dumps(resp.data['user'], default=str)}")

# ── TEST 2: Login with username ───────────────────────────────────
print("\n" + "=" * 60)
print("TEST 2: Login with username")
print("=" * 60)
req = factory.post(
    '/api/auth/login/',
    data=json.dumps({'username': 'testuser1', 'password': 'TestPass123!'}),
    content_type='application/json',
)
resp = login_view(req)
check("Status 200", resp.status_code == 200)
check("Message = Login successful", resp.data.get('message') == 'Login successful')
check("User has plan field", 'plan' in resp.data['user'])

# ── TEST 3: Login with email ─────────────────────────────────────
print("\n" + "=" * 60)
print("TEST 3: Login with email")
print("=" * 60)
req = factory.post(
    '/api/auth/login/',
    data=json.dumps({'username': 'test1@agevision.com', 'password': 'TestPass123!'}),
    content_type='application/json',
)
resp = login_view(req)
check("Status 200", resp.status_code == 200)
check("Correct user returned", resp.data['user']['email'] == 'test1@agevision.com')

# ── TEST 4: Wrong password ───────────────────────────────────────
print("\n" + "=" * 60)
print("TEST 4: Login with wrong password")
print("=" * 60)
req = factory.post(
    '/api/auth/login/',
    data=json.dumps({'username': 'testuser1', 'password': 'WrongPassword!'}),
    content_type='application/json',
)
resp = login_view(req)
check("Status 401", resp.status_code == 401)
check("Error message", resp.data.get('error') == 'Invalid credentials')

# ── TEST 5: Duplicate registration ───────────────────────────────
print("\n" + "=" * 60)
print("TEST 5: Duplicate registration")
print("=" * 60)
req = factory.post(
    '/api/auth/register/',
    data=json.dumps({
        'username': 'testuser1',
        'email': 'test1@agevision.com',
        'password': 'TestPass123!',
        'first_name': 'Test',
        'last_name': 'User',
    }),
    content_type='application/json',
)
resp = register_view(req)
check("Status 400", resp.status_code == 400)
check("Username error present", 'username' in resp.data)

# ── TEST 6: Missing fields ──────────────────────────────────────
print("\n" + "=" * 60)
print("TEST 6: Login with missing fields")
print("=" * 60)
req = factory.post(
    '/api/auth/login/',
    data=json.dumps({'username': ''}),
    content_type='application/json',
)
resp = login_view(req)
check("Status 400", resp.status_code == 400)

# ── TEST 7: MongoDB storage verification ────────────────────────
print("\n" + "=" * 60)
print("TEST 7: MongoDB storage verification")
print("=" * 60)
django_user = User.objects.get(username='testuser1')
mongo_user = MongoUserManager.get_by_django_id(django_user.id)
check("MongoDB doc exists", mongo_user is not None)
check("username matches", mongo_user['username'] == 'testuser1')
check("email matches", mongo_user['email'] == 'test1@agevision.com')
check("first_name matches", mongo_user['first_name'] == 'Test')
check("last_name matches", mongo_user['last_name'] == 'User')
check("plan = free", mongo_user['plan'] == 'free')
check("created_at present", mongo_user['created_at'] is not None)
check("last_login present", mongo_user['last_login'] is not None)

# Raw check
raw_doc = MongoDB.get_db()['users'].find_one({'username': 'testuser1'})
check("Raw doc has _id", raw_doc is not None and '_id' in raw_doc)
print(f"  MongoDB fields: {list(raw_doc.keys())}")

# ── TEST 8: Profile endpoint ────────────────────────────────────
print("\n" + "=" * 60)
print("TEST 8: Profile endpoint (authenticated)")
print("=" * 60)
from rest_framework_simplejwt.tokens import RefreshToken
refresh = RefreshToken.for_user(django_user)
from rest_framework.test import APIRequestFactory
api_factory = APIRequestFactory()
req = api_factory.get('/api/auth/profile/')
from rest_framework_simplejwt.authentication import JWTAuthentication
req.META['HTTP_AUTHORIZATION'] = f'Bearer {refresh.access_token}'
# Force authentication for testing
from django.contrib.auth.models import AnonymousUser
req.user = django_user
resp = profile_view(req)
check("Status 200", resp.status_code == 200)
check("Profile has username", resp.data.get('username') == 'testuser1')
check("Profile has plan", 'plan' in resp.data)

# ── CLEANUP ──────────────────────────────────────────────────────
print("\n" + "-" * 60)
print("Cleaning up test data...")
MongoUserManager.delete_user(django_user.id)
django_user.delete()
check("Django user deleted", not User.objects.filter(username='testuser1').exists())
check("MongoDB user deleted", MongoUserManager.get_by_django_id(django_user.id) is None)

# ── SUMMARY ──────────────────────────────────────────────────────
print("\n" + "=" * 60)
total = PASS + FAIL
print(f"RESULTS: {PASS}/{total} passed, {FAIL} failed")
print("=" * 60)
if FAIL > 0:
    sys.exit(1)
