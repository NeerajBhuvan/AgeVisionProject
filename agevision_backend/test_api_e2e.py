"""Quick end-to-end API test."""
import requests
import json
import sys

BASE = 'http://127.0.0.1:8000/api'

# Register
print("[1] Register...", end=" ")
r = requests.post(f'{BASE}/auth/register/', json={
    'username': 'e2etest99',
    'email': 'e2e99@test.com',
    'password': 'E2ePass123!',
    'password2': 'E2ePass123!',
    }, timeout=30)
if r.status_code == 201:
    print(f"OK ({r.status_code})")
elif r.status_code == 400 and 'username' in r.text:
    print(f"User exists ({r.status_code}) - OK")
else:
    print(f"UNEXPECTED: {r.status_code} {r.text[:200]}")

# Login
print("[2] Login...", end=" ")
r = requests.post(f'{BASE}/auth/login/', json={
    'username': 'e2etest99',
    'password': 'E2ePass123!',
}, timeout=30)
if r.status_code == 200:
    data = r.json()
    token = data.get('tokens', {}).get('access', '') or data.get('access', '')
    print(f"OK (token: {token[:30]}...)")
else:
    print(f"FAIL: {r.status_code} {r.text[:200]}")
    # Try registering fresh
    r2 = requests.post(f'{BASE}/auth/register/', json={
        'username': 'e2etest99',
        'email': 'e2e99@test.com',
        'password': 'E2ePass123!',
        'password2': 'E2ePass123!',
    }, timeout=10)
    print(f"  Fresh register: {r2.status_code}")
    if r2.status_code == 201:
        data = r2.json()
        token = data.get('tokens', {}).get('access', '') or data.get('access', '')
        if not token:
            # Try login
            r3 = requests.post(f'{BASE}/auth/login/', json={
                'username': 'e2etest99',
                'password': 'E2ePass123!',
            }, timeout=10)
            print(f"  Login: {r3.status_code} {r3.text[:200]}")
            token = r3.json().get('access', '') if r3.status_code == 200 else ''
        if token:
            print(f"  Got token: {token[:30]}...")
    if not token:
        print("  Cannot get token, stopping.")
        sys.exit(1)

# Predict
print("[3] Predict (Kohli)...", end=" ", flush=True)
headers = {'Authorization': f'Bearer {token}'}
with open('media/test_images/virat_kohli.jpg', 'rb') as f:
    r = requests.post(f'{BASE}/predict/', files={'image': f}, headers=headers, timeout=300)
if r.status_code == 200:
    d = r.json()
    print(f"OK - Age:{d.get('predicted_age')} Gender:{d.get('gender')} "
          f"Emotion:{d.get('emotion')} Confidence:{d.get('confidence')}")
    print(f"    Full response keys: {list(d.keys())}")
    if d.get('predicted_age') is None:
        print(f"    Full response: {str(d)[:500]}")
else:
    print(f"FAIL: {r.status_code} {r.text[:300]}")

print("\nAll API checks done.")
