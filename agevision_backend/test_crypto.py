"""
Test the AgeVision custom encryption + password storage/recovery.
Run from agevision_backend folder:
    python test_crypto.py
"""
import django, os, json, sys

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'agevision_backend.settings')
django.setup()

from agevision_api.crypto import agevision_encrypt, agevision_decrypt
from agevision_api.mongodb import MongoUserManager, MongoDB
from django.contrib.auth.models import User
from django.test import RequestFactory
from agevision_api.views.auth_views import (
    register_view, login_view, forgot_password_view,
    recover_password_view, reset_password_view,
)

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


factory = RequestFactory()

# ══════════════════════════════════════════════════════════════════
# TEST 1: Encryption / Decryption basics
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TEST 1: AgeVision Encrypt / Decrypt")
print("=" * 60)

pw = "MySecretP@ss123!"
ctx = "testuser1"

encrypted = agevision_encrypt(pw, context=ctx)
print(f"  Plaintext:  {pw}")
print(f"  Encrypted:  {encrypted[:60]}...")
print(f"  Enc length: {len(encrypted)} chars")

decrypted = agevision_decrypt(encrypted, context=ctx)
check("Decrypt matches original", decrypted == pw)

# Different context produces different ciphertext
enc2 = agevision_encrypt(pw, context="otheruser")
check("Different context → different ciphertext", enc2 != encrypted)

# Wrong context fails to produce correct plaintext
wrong_ctx = agevision_decrypt(encrypted, context="wronguser")
check("Wrong context → wrong plaintext", wrong_ctx != pw)

# Tampered ciphertext returns None
check("Tampered ciphertext → None", agevision_decrypt("tampered_data!", context=ctx) is None)

# ══════════════════════════════════════════════════════════════════
# TEST 2: Register stores encrypted password in MongoDB
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TEST 2: Registration stores encrypted password")
print("=" * 60)

req = factory.post(
    '/api/auth/register/',
    data=json.dumps({
        'username': 'cryptotest',
        'email': 'crypto@agevision.com',
        'password': 'CryptoP@ss99!',
        'first_name': 'Crypto',
        'last_name': 'Test',
    }),
    content_type='application/json',
)
resp = register_view(req)
check("Registration status 201", resp.status_code == 201)

# Check raw MongoDB document
raw = MongoDB.get_db()['users'].find_one({'username': 'cryptotest'})
check("encrypted_password field exists", 'encrypted_password' in raw)
check("encrypted_password is not plaintext", raw['encrypted_password'] != 'CryptoP@ss99!')
print(f"  Encrypted in DB: {raw['encrypted_password'][:50]}...")

# Recover it
recovered = MongoUserManager.recover_password(raw['django_user_id'])
check("Recovered password matches original", recovered == 'CryptoP@ss99!')

# ══════════════════════════════════════════════════════════════════
# TEST 3: Login updates encrypted password for legacy users
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TEST 3: Login syncs encrypted password")
print("=" * 60)

req = factory.post(
    '/api/auth/login/',
    data=json.dumps({'username': 'cryptotest', 'password': 'CryptoP@ss99!'}),
    content_type='application/json',
)
resp = login_view(req)
check("Login status 200", resp.status_code == 200)

# Raw doc still has encrypted_password
raw2 = MongoDB.get_db()['users'].find_one({'username': 'cryptotest'})
check("encrypted_password still present after login", 'encrypted_password' in raw2)

# ══════════════════════════════════════════════════════════════════
# TEST 4: Forgot password endpoint
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TEST 4: Forgot password")
print("=" * 60)

req = factory.post(
    '/api/auth/forgot-password/',
    data=json.dumps({'email': 'crypto@agevision.com'}),
    content_type='application/json',
)
resp = forgot_password_view(req)
check("Forgot-password status 200", resp.status_code == 200)
check("recovery_available = True", resp.data.get('recovery_available') is True)
check("password_hint present", len(resp.data.get('password_hint', '')) > 0)
check("password_length = 13", resp.data.get('password_length') == 13)
check("recovery_token present", len(resp.data.get('recovery_token', '')) > 20)
check("username = cryptotest", resp.data.get('username') == 'cryptotest')

hint = resp.data['password_hint']
print(f"  Password hint: {hint}")
print(f"  Length: {resp.data['password_length']}")

recovery_token = resp.data['recovery_token']

# Non-existent email
req = factory.post(
    '/api/auth/forgot-password/',
    data=json.dumps({'email': 'nobody@fake.com'}),
    content_type='application/json',
)
resp = forgot_password_view(req)
check("Non-existent email → no leak", 'recovery_available' not in resp.data or resp.data.get('recovery_available') is False)

# ══════════════════════════════════════════════════════════════════
# TEST 5: Recover password (full decryption via API)
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TEST 5: Recover password (reveal)")
print("=" * 60)

req = factory.post(
    '/api/auth/recover-password/',
    data=json.dumps({
        'email': 'crypto@agevision.com',
        'recovery_token': recovery_token,
    }),
    content_type='application/json',
)
resp = recover_password_view(req)
check("Recover status 200", resp.status_code == 200)
check("Recovered password = CryptoP@ss99!", resp.data.get('password') == 'CryptoP@ss99!')
print(f"  Recovered: {resp.data.get('password')}")

# Wrong email with valid token
req = factory.post(
    '/api/auth/recover-password/',
    data=json.dumps({
        'email': 'wrong@email.com',
        'recovery_token': recovery_token,
    }),
    content_type='application/json',
)
resp = recover_password_view(req)
check("Wrong email → rejected", resp.status_code == 400)

# ══════════════════════════════════════════════════════════════════
# TEST 6: Reset password
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TEST 6: Reset password")
print("=" * 60)

# Get a fresh token for reset
req = factory.post(
    '/api/auth/forgot-password/',
    data=json.dumps({'email': 'crypto@agevision.com'}),
    content_type='application/json',
)
resp = forgot_password_view(req)
reset_token = resp.data['recovery_token']

req = factory.post(
    '/api/auth/reset-password/',
    data=json.dumps({
        'email': 'crypto@agevision.com',
        'recovery_token': reset_token,
        'new_password': 'NewSecure@2026',
    }),
    content_type='application/json',
)
resp = reset_password_view(req)
check("Reset status 200", resp.status_code == 200)
check("Success message", 'reset successfully' in resp.data.get('message', '').lower())

# Verify new password works in Django
user = User.objects.get(username='cryptotest')
check("Django password updated", user.check_password('NewSecure@2026'))
check("Old password no longer works", not user.check_password('CryptoP@ss99!'))

# Verify MongoDB has updated encrypted password
new_recovered = MongoUserManager.recover_password(user.id)
check("MongoDB encrypted password updated", new_recovered == 'NewSecure@2026')
print(f"  New recovered password: {new_recovered}")

# Login with new password
req = factory.post(
    '/api/auth/login/',
    data=json.dumps({'username': 'cryptotest', 'password': 'NewSecure@2026'}),
    content_type='application/json',
)
resp = login_view(req)
check("Login with new password works", resp.status_code == 200)

# ══════════════════════════════════════════════════════════════════
# CLEANUP
# ══════════════════════════════════════════════════════════════════
print("\n" + "-" * 60)
print("Cleaning up test data...")
MongoUserManager.delete_user(user.id)
user.delete()
check("Test user cleaned up", not User.objects.filter(username='cryptotest').exists())

# ══════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
total = PASS + FAIL
print(f"RESULTS: {PASS}/{total} passed, {FAIL} failed")
print("=" * 60)
if FAIL > 0:
    sys.exit(1)
