#!/usr/bin/env python
"""
Comprehensive end-to-end API test for MongoDB migration.
=========================================================
Tests all modules: auth, predict, progress, history, analytics, settings.
All application data should now be stored in MongoDB.

Usage:
    python test_mongodb_e2e.py
"""
import os
import sys
import time
import requests
import json

BASE = 'http://127.0.0.1:8000/api'
PASS = 0
FAIL = 0


def check(label, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {label}")
    else:
        FAIL += 1
        print(f"  [FAIL] {label} -- {detail}")


def main():
    global PASS, FAIL

    print("=" * 65)
    print("  AgeVision MongoDB E2E Test Suite")
    print("=" * 65)

    # -- 1. Registration --
    print("\n[1] AUTH - Register")
    ts = int(time.time())
    username = f"mongotest_{ts}"
    email = f"mongotest_{ts}@test.com"
    password = "MongoTest123!"

    r = requests.post(f'{BASE}/auth/register/', json={
        'username': username,
        'email': email,
        'password': password,
    }, timeout=30)

    check("Register new user", r.status_code == 201, f"status={r.status_code} body={r.text[:200]}")

    # -- 2. Login --
    print("\n[2] AUTH - Login")
    r = requests.post(f'{BASE}/auth/login/', json={
        'username': username,
        'password': password,
    }, timeout=30)

    check("Login returns 200", r.status_code == 200, f"status={r.status_code} body={r.text[:200]}")

    token = ""
    if r.status_code == 200:
        data = r.json()
        token = data.get('tokens', {}).get('access', '') or data.get('access', '')
        check("Login returns JWT token", bool(token), "No token in response")
        mongo_user = data.get('user', {})
        check("Login returns MongoDB user profile", bool(mongo_user.get('username')),
              f"user={mongo_user}")

    if not token:
        print("\n  Cannot continue without auth token. Exiting.")
        sys.exit(1)

    headers = {'Authorization': f'Bearer {token}'}

    # -- 3. Profile --
    print("\n[3] AUTH - Profile")
    r = requests.get(f'{BASE}/auth/profile/', headers=headers, timeout=10)
    check("Profile returns 200", r.status_code == 200, f"status={r.status_code}")
    if r.status_code == 200:
        profile = r.json()
        check("Profile has username", profile.get('user', {}).get('username') == username or
              profile.get('username') == username,
              f"profile={json.dumps(profile)[:200]}")

    # -- 4. Settings (GET) --
    print("\n[4] SETTINGS - Get defaults")
    r = requests.get(f'{BASE}/settings/', headers=headers, timeout=10)
    check("Settings GET returns 200", r.status_code == 200, f"status={r.status_code}")
    if r.status_code == 200:
        settings_data = r.json()
        check("Settings has 'theme' field", 'theme' in settings_data, f"keys={list(settings_data.keys())}")
        check("Default theme is 'dark'", settings_data.get('theme') == 'dark',
              f"theme={settings_data.get('theme')}")
        check("Settings has MongoDB 'id' (ObjectId)", 'id' in settings_data,
              f"keys={list(settings_data.keys())}")

    # -- 5. Settings (PUT) --
    print("\n[5] SETTINGS - Update")
    r = requests.put(f'{BASE}/settings/', headers=headers, json={
        'theme': 'light',
        'language': 'Hindi',
        'high_accuracy_mode': True,
    }, timeout=10)
    check("Settings PUT returns 200", r.status_code == 200, f"status={r.status_code} body={r.text[:200]}")
    if r.status_code == 200:
        updated = r.json()
        check("Theme updated to 'light'", updated.get('theme') == 'light',
              f"theme={updated.get('theme')}")
        check("Language updated to 'Hindi'", updated.get('language') == 'Hindi',
              f"language={updated.get('language')}")
        check("High accuracy mode enabled", updated.get('high_accuracy_mode') is True,
              f"high_accuracy_mode={updated.get('high_accuracy_mode')}")

    # -- 6. Predict (requires test image) --
    print("\n[6] PREDICT - Age prediction")
    test_image = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'media', 'test_images', 'virat_kohli.jpg')

    prediction_id = None
    if os.path.exists(test_image):
        with open(test_image, 'rb') as f:
            r = requests.post(f'{BASE}/predict/', files={'image': f},
                              headers=headers, timeout=300)

        check("Predict returns 200", r.status_code == 200,
              f"status={r.status_code} body={r.text[:300]}")

        if r.status_code == 200:
            d = r.json()
            pred = d.get('prediction', {})
            check("Prediction has MongoDB 'id'", bool(pred.get('id')),
                  f"keys={list(pred.keys())}")
            check("Prediction has predicted_age", pred.get('predicted_age') is not None,
                  f"pred={json.dumps(pred)[:200]}")
            check("Prediction has confidence", pred.get('confidence') is not None, "")
            check("Prediction has image_url", bool(pred.get('image_url')),
                  f"image_url={pred.get('image_url')}")
            prediction_id = pred.get('id')
            print(f"       -> Age: {pred.get('predicted_age')}, "
                  f"Gender: {pred.get('gender')}, "
                  f"Confidence: {pred.get('confidence')}")
    else:
        print(f"  [SKIP] Test image not found: {test_image}")

    # -- 7. Progress (requires test image) --
    print("\n[7] PROGRESS - Age progression")
    progression_id = None
    if os.path.exists(test_image):
        with open(test_image, 'rb') as f:
            r = requests.post(f'{BASE}/progress/',
                              files={'image': f},
                              data={'target_age': '60'},
                              headers=headers, timeout=300)

        check("Progress returns 200", r.status_code == 200,
              f"status={r.status_code} body={r.text[:300]}")

        if r.status_code == 200:
            d = r.json()
            prog = d.get('progression', {})
            check("Progression has MongoDB 'id'", bool(prog.get('id')),
                  f"keys={list(prog.keys())}")
            check("Progression has original_image_url", bool(prog.get('original_image_url')), "")
            check("Progression has progressed_image_url", bool(prog.get('progressed_image_url')), "")
            check("Progression has pipeline_steps (list)", isinstance(prog.get('pipeline_steps'), list), "")
            check("Progression has aging_insights (list)", isinstance(prog.get('aging_insights'), list), "")
            progression_id = prog.get('id')
            print(f"       -> Current: {prog.get('current_age')}, "
                  f"Target: {prog.get('target_age')}, "
                  f"Model: {prog.get('model_used')}")
    else:
        print(f"  [SKIP] Test image not found: {test_image}")

    # -- 8. History --
    print("\n[8] HISTORY - List records")
    r = requests.get(f'{BASE}/history/', headers=headers, timeout=30)
    check("History returns 200", r.status_code == 200, f"status={r.status_code}")
    if r.status_code == 200:
        hist = r.json()
        check("History has 'predictions' list", isinstance(hist.get('predictions'), list),
              f"keys={list(hist.keys())}")
        check("History has 'progressions' list", isinstance(hist.get('progressions'), list), "")
        check("History has total_predictions count",
              hist.get('total_predictions') is not None, "")
        check("History has total_progressions count",
              hist.get('total_progressions') is not None, "")

        # Verify predictions have MongoDB ObjectId style IDs
        preds = hist.get('predictions', [])
        if preds:
            check("Prediction records have MongoDB 'id' field",
                  bool(preds[0].get('id')), f"first_pred_keys={list(preds[0].keys())}")
            check("Prediction id is a 24-char hex string (ObjectId)",
                  len(str(preds[0].get('id', ''))) == 24,
                  f"id={preds[0].get('id')}")

        progs = hist.get('progressions', [])
        if progs:
            check("Progression records have MongoDB 'id' field",
                  bool(progs[0].get('id')), f"first_prog_keys={list(progs[0].keys())}")

        print(f"       -> {hist.get('total_predictions')} predictions, "
              f"{hist.get('total_progressions')} progressions")

    # -- 9. Analytics --
    print("\n[9] ANALYTICS - Dashboard stats")
    r = requests.get(f'{BASE}/analytics/', headers=headers, timeout=30)
    check("Analytics returns 200", r.status_code == 200, f"status={r.status_code}")
    if r.status_code == 200:
        analytics = r.json()
        check("Analytics has total_predictions", 'total_predictions' in analytics, "")
        check("Analytics has total_progressions", 'total_progressions' in analytics, "")
        check("Analytics has average_predicted_age", 'average_predicted_age' in analytics, "")
        check("Analytics has gender_distribution", 'gender_distribution' in analytics, "")
        check("Analytics has emotion_distribution", 'emotion_distribution' in analytics, "")
        check("Analytics has daily_counts", 'daily_counts' in analytics, "")
        check("Analytics daily_counts is a list", isinstance(analytics.get('daily_counts'), list), "")
        print(f"       -> Total predictions: {analytics.get('total_predictions')}, "
              f"Avg age: {analytics.get('average_predicted_age')}")

    # -- 10. History Delete --
    print("\n[10] HISTORY - Delete record")
    if prediction_id:
        r = requests.delete(f'{BASE}/history/{prediction_id}/', headers=headers, timeout=10)
        check("Delete prediction returns 204", r.status_code == 204,
              f"status={r.status_code} body={r.text[:200]}")

        # Verify it's gone
        r = requests.get(f'{BASE}/history/', headers=headers, timeout=10)
        if r.status_code == 200:
            hist = r.json()
            remaining_ids = [p.get('id') for p in hist.get('predictions', [])]
            check("Deleted prediction no longer in history",
                  prediction_id not in remaining_ids,
                  f"still found id={prediction_id}")
    else:
        print("  [SKIP] No prediction_id to delete")

    # -- 11. Password change --
    print("\n[11] AUTH - Change password")
    r = requests.post(f'{BASE}/auth/change-password/', headers=headers, json={
        'current_password': password,
        'new_password': 'MongoNew456!',
    }, timeout=10)
    check("Change password returns 200", r.status_code == 200,
          f"status={r.status_code} body={r.text[:200]}")

    # Re-login with new password
    r = requests.post(f'{BASE}/auth/login/', json={
        'username': username,
        'password': 'MongoNew456!',
    }, timeout=10)
    check("Login with new password works", r.status_code == 200,
          f"status={r.status_code}")

    # -- Summary --
    print("\n" + "=" * 65)
    total = PASS + FAIL
    print(f"  Results: {PASS}/{total} passed, {FAIL} failed")
    if FAIL == 0:
        print("  ALL TESTS PASSED - MongoDB migration verified!")
    else:
        print(f"  {FAIL} test(s) FAILED")
    print("=" * 65)

    return 0 if FAIL == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
