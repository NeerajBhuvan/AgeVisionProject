"""
Age Prediction End-to-End Test
===============================
Tests the complete age prediction pipeline:
1. Single face prediction (predict_age)
2. Group/multi-face prediction (predict_group_faces)
3. Real-time camera frame prediction (predict_frame)
4. Celebrity accuracy benchmark (download + predict + metrics)

Uses MiVOLO v2 (primary) with InsightFace buffalo_l (fallback).

Usage
-----
    cd agevision_backend
    python test_age_prediction.py
"""

import os
import sys
import time
import json
import warnings

warnings.filterwarnings('ignore')

# Django setup
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'agevision_backend.settings')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import django
django.setup()

import cv2
import numpy as np
import requests


# ------------------------------------------------------------------
#  Celebrity face dataset with known ages
# ------------------------------------------------------------------
CELEBRITY_FACES = [
    {"wiki_title": "Sachin Tendulkar",     "filename": "sachin_tendulkar.jpg",   "ground_truth_age": 50, "ethnicity": "indian",         "gender": "Man"},
    {"wiki_title": "Virat Kohli",          "filename": "virat_kohli.jpg",        "ground_truth_age": 34, "ethnicity": "indian",         "gender": "Man"},
    {"wiki_title": "Shah Rukh Khan",       "filename": "shahrukh_khan.jpg",      "ground_truth_age": 57, "ethnicity": "indian",         "gender": "Man"},
    {"wiki_title": "Narendra Modi",        "filename": "narendra_modi.jpg",      "ground_truth_age": 74, "ethnicity": "indian",         "gender": "Man"},
    {"wiki_title": "Alia Bhatt",           "filename": "alia_bhatt.jpg",         "ground_truth_age": 29, "ethnicity": "indian",         "gender": "Woman"},
    {"wiki_title": "Deepika Padukone",     "filename": "deepika_padukone.jpg",   "ground_truth_age": 38, "ethnicity": "indian",         "gender": "Woman"},
    {"wiki_title": "MS Dhoni",             "filename": "ms_dhoni.jpg",           "ground_truth_age": 43, "ethnicity": "indian",         "gender": "Man"},
    {"wiki_title": "Amitabh Bachchan",     "filename": "amitabh_bachchan.jpg",   "ground_truth_age": 82, "ethnicity": "indian",         "gender": "Man"},
    {"wiki_title": "Jackie Chan",          "filename": "jackie_chan.jpg",        "ground_truth_age": 70, "ethnicity": "asian",          "gender": "Man"},
    {"wiki_title": "Fan Bingbing",         "filename": "fan_bingbing.jpg",       "ground_truth_age": 43, "ethnicity": "asian",          "gender": "Woman"},
    {"wiki_title": "Brad Pitt",            "filename": "brad_pitt.jpg",          "ground_truth_age": 61, "ethnicity": "white",          "gender": "Man"},
    {"wiki_title": "Angelina Jolie",       "filename": "angelina_jolie.jpg",     "ground_truth_age": 49, "ethnicity": "white",          "gender": "Woman"},
    {"wiki_title": "Elon Musk",            "filename": "elon_musk.jpg",          "ground_truth_age": 53, "ethnicity": "white",          "gender": "Man"},
    {"wiki_title": "Elizabeth II",         "filename": "queen_elizabeth.jpg",    "ground_truth_age": 68, "ethnicity": "white",          "gender": "Woman"},
    {"wiki_title": "Barack Obama",         "filename": "barack_obama.jpg",       "ground_truth_age": 49, "ethnicity": "black",          "gender": "Man"},
    {"wiki_title": "Nelson Mandela",       "filename": "nelson_mandela.jpg",     "ground_truth_age": 76, "ethnicity": "black",          "gender": "Man"},
    {"wiki_title": "Denzel Washington",    "filename": "denzel_washington.jpg",  "ground_truth_age": 69, "ethnicity": "black",          "gender": "Man"},
    {"wiki_title": "Mohamed Salah",        "filename": "mohamed_salah.jpg",      "ground_truth_age": 32, "ethnicity": "middle eastern", "gender": "Man"},
]


def download_faces(save_dir: str) -> list:
    """Download face images using Wikipedia pageimages API."""
    os.makedirs(save_dir, exist_ok=True)
    downloaded = []

    session = requests.Session()
    session.headers.update({
        'User-Agent': 'AgeVisionTest/1.0 (Academic Research; Python/requests)',
    })

    for entry in CELEBRITY_FACES:
        filepath = os.path.join(save_dir, entry['filename'])

        if os.path.exists(filepath) and os.path.getsize(filepath) > 5000:
            downloaded.append({**entry, "path": filepath, "source": "cached"})
            print(f"    Cached: {entry['filename']}")
            continue

        try:
            api_url = 'https://en.wikipedia.org/w/api.php'
            params = {
                'action': 'query',
                'titles': entry['wiki_title'],
                'prop': 'pageimages',
                'pithumbsize': 500,
                'format': 'json',
            }
            api_resp = session.get(api_url, params=params, timeout=15)
            api_resp.raise_for_status()
            pages = api_resp.json().get('query', {}).get('pages', {})

            thumb_url = None
            for page in pages.values():
                thumb_url = page.get('thumbnail', {}).get('source')

            if not thumb_url:
                print(f"    No image: {entry['wiki_title']}")
                continue

            img_resp = session.get(thumb_url, timeout=20)
            img_resp.raise_for_status()
            data = img_resp.content

            if len(data) > 5000:
                with open(filepath, 'wb') as f:
                    f.write(data)
                downloaded.append({**entry, "path": filepath, "source": "wikipedia"})
                print(f"    Downloaded: {entry['filename']} ({len(data) // 1024}KB)")
            else:
                print(f"    Too small: {entry['filename']} ({len(data)} bytes)")

            time.sleep(0.5)

        except Exception as e:
            print(f"    Failed: {entry['filename']} - {type(e).__name__}: {e}")

    session.close()
    return downloaded


def compute_metrics(results: list) -> dict:
    """Compute MAE, accuracy-within-N, per-ethnicity/age-group breakdowns."""
    labeled = [r for r in results if r.get('ground_truth_age') is not None]
    if not labeled:
        return {'error': 'No labeled samples'}

    errors = [abs(r['predicted_age'] - r['ground_truth_age']) for r in labeled]

    mae = float(np.mean(errors))
    median_ae = float(np.median(errors))
    rmse = float(np.sqrt(np.mean(np.array(errors) ** 2)))
    within_3 = sum(1 for e in errors if e <= 3) / len(errors) * 100
    within_5 = sum(1 for e in errors if e <= 5) / len(errors) * 100
    within_10 = sum(1 for e in errors if e <= 10) / len(errors) * 100

    ethnicities = set(r['ethnicity'] for r in labeled)
    per_ethnicity = {}
    for eth in sorted(ethnicities):
        eth_r = [r for r in labeled if r['ethnicity'] == eth]
        eth_e = [abs(r['predicted_age'] - r['ground_truth_age']) for r in eth_r]
        per_ethnicity[eth] = {
            'count': len(eth_r),
            'mae': round(float(np.mean(eth_e)), 2),
            'within_5': round(sum(1 for e in eth_e if e <= 5) / len(eth_e) * 100, 1),
            'within_10': round(sum(1 for e in eth_e if e <= 10) / len(eth_e) * 100, 1),
        }

    groups = {'18-29': (18, 29), '30-44': (30, 44), '45-59': (45, 59), '60+': (60, 120)}
    per_age = {}
    for name, (lo, hi) in groups.items():
        gr = [r for r in labeled if lo <= r['ground_truth_age'] <= hi]
        if gr:
            ge = [abs(r['predicted_age'] - r['ground_truth_age']) for r in gr]
            per_age[name] = {'count': len(gr), 'mae': round(float(np.mean(ge)), 2)}

    return {
        'total_samples': len(labeled),
        'mae': round(mae, 2),
        'median_ae': round(median_ae, 2),
        'rmse': round(rmse, 2),
        'within_3_pct': round(within_3, 1),
        'within_5_pct': round(within_5, 1),
        'within_10_pct': round(within_10, 1),
        'per_ethnicity': per_ethnicity,
        'per_age_group': per_age,
    }


def print_report(metrics: dict, results: list):
    """Print formatted accuracy report."""
    labeled = [r for r in results if r.get('ground_truth_age') is not None]

    print("\n" + "=" * 74)
    print("   AGE PREDICTION ACCURACY REPORT")
    print("   Model: MiVOLO v2 (YOLOv8 + Vision Transformer)")
    print("=" * 74)

    print(f"\n   Total labeled samples:    {metrics['total_samples']}")
    print(f"   Mean Absolute Error:      {metrics['mae']} years")
    print(f"   Median Absolute Error:    {metrics['median_ae']} years")
    print(f"   RMSE:                     {metrics['rmse']} years")
    print(f"\n   Accuracy within +/-3 yr:  {metrics['within_3_pct']}%")
    print(f"   Accuracy within +/-5 yr:  {metrics['within_5_pct']}%")
    print(f"   Accuracy within +/-10 yr: {metrics['within_10_pct']}%")

    print("\n   " + "-" * 66)
    print("   PER-ETHNICITY BREAKDOWN")
    print("   " + "-" * 66)
    print(f"   {'Ethnicity':<18} {'N':>4} {'MAE':>7} {'<=5yr':>8} {'<=10yr':>8}")
    print("   " + "-" * 66)
    for eth, d in metrics['per_ethnicity'].items():
        print(f"   {eth:<18} {d['count']:>4} {d['mae']:>7.2f} {d['within_5']:>7.1f}% {d['within_10']:>7.1f}%")

    if metrics.get('per_age_group'):
        print("\n   " + "-" * 66)
        print("   PER AGE-GROUP BREAKDOWN")
        print("   " + "-" * 66)
        print(f"   {'Group':<18} {'N':>4} {'MAE':>7}")
        print("   " + "-" * 66)
        for g, d in metrics['per_age_group'].items():
            print(f"   {g:<18} {d['count']:>4} {d['mae']:>7.2f}")

    print("\n   " + "-" * 66)
    print("   DETAILED PER-SAMPLE RESULTS")
    print("   " + "-" * 66)
    print(f"   {'Name':<28} {'GT':>4} {'Pred':>5} {'Err':>5}  {'Gender':>7} {'Time':>7}")
    print("   " + "-" * 66)
    for r in labeled:
        err = abs(r['predicted_age'] - r['ground_truth_age'])
        mark = " OK" if err <= 5 else "   " if err <= 10 else " !!"
        name = r.get('filename', '?')[:26]
        print(
            f"   {name:<28} {r['ground_truth_age']:>4} "
            f"{r['predicted_age']:>5} {err:>4}{mark} "
            f"{r.get('gender', '?'):>7} {r.get('total_time_ms', 0):>6.0f}ms"
        )

    print("\n" + "=" * 74)


# ======================================================================
#  TEST FUNCTIONS
# ======================================================================

def test_single_face(test_dir: str):
    """Test 1: Single face prediction via predict_age()."""
    from agevision_api.age_predictor import predict_age

    print("\n" + "=" * 74)
    print("   TEST 1: Single Face Prediction (predict_age)")
    print("=" * 74)

    # Use first available sample image
    sample_path = None
    for f in os.listdir(test_dir):
        if f.endswith('.jpg'):
            sample_path = os.path.join(test_dir, f)
            break

    if not sample_path:
        # Try frontend samples
        samples_dir = os.path.join('..', 'agevision-frontend', 'public', 'samples')
        for f in ['single_1.jpg', 'single_2.jpg']:
            p = os.path.join(samples_dir, f)
            if os.path.exists(p):
                sample_path = p
                break

    if not sample_path:
        print("   SKIP: No sample images found")
        return False

    print(f"   Image: {os.path.basename(sample_path)}")
    start = time.time()
    result = predict_age(sample_path)
    elapsed = (time.time() - start) * 1000

    print(f"   Predicted Age: {result['predicted_age']}")
    print(f"   Gender:        {result['gender']}")
    print(f"   Confidence:    {result['confidence']}")
    print(f"   Face Count:    {result['face_count']}")
    print(f"   Detector:      {result['detector_used']}")
    print(f"   Time:          {elapsed:.0f}ms")

    # Validate structure
    assert result['predicted_age'] > 0, "Age should be > 0"
    assert result['face_count'] >= 1, "Should detect at least 1 face"
    assert result['confidence'] > 0, "Confidence should be > 0"
    assert result['gender'] in ('Man', 'Woman', 'Unknown'), f"Bad gender: {result['gender']}"
    assert len(result['faces']) >= 1, "Should have at least 1 face in faces list"
    assert 'face_region' in result['faces'][0], "Face should have face_region"

    print("   PASSED")
    return True


def test_group_faces(samples_dir: str):
    """Test 2: Group/multi-face prediction via predict_group_faces()."""
    from agevision_api.age_predictor import predict_group_faces

    print("\n" + "=" * 74)
    print("   TEST 2: Group Face Prediction (predict_group_faces)")
    print("=" * 74)

    # Try frontend group sample
    group_path = None
    fe_samples = os.path.join('..', 'agevision-frontend', 'public', 'samples')
    for f in ['group_1.jpg', 'group_2.jpg', 'group_3.jpg']:
        p = os.path.join(fe_samples, f)
        if os.path.exists(p):
            group_path = p
            break

    if not group_path:
        print("   SKIP: No group sample images found")
        return False

    print(f"   Image: {os.path.basename(group_path)}")
    start = time.time()
    result = predict_group_faces(group_path)
    elapsed = (time.time() - start) * 1000

    print(f"   Faces Detected: {result['face_count']}")
    print(f"   Primary Age:    {result['predicted_age']}")
    print(f"   Detector:       {result['detector_used']}")
    print(f"   Time:           {elapsed:.0f}ms")

    if result['faces']:
        print(f"\n   Per-face results:")
        for face in result['faces']:
            print(f"     Face #{face['face_id']}: Age {face['predicted_age']}, "
                  f"{face['gender']}, conf={face['confidence']:.2f}, "
                  f"region=({face['face_region']['x_pct']:.1f}%, {face['face_region']['y_pct']:.1f}%)")

    # Validate
    assert result['face_count'] >= 2, f"Group should have 2+ faces, got {result['face_count']}"
    assert len(result['faces']) == result['face_count'], "faces list length should match face_count"
    for face in result['faces']:
        assert face['predicted_age'] > 0, "Each face age should be > 0"
        assert 'face_region' in face, "Each face should have face_region"
        region = face['face_region']
        assert 0 <= region['x_pct'] <= 100, f"x_pct out of range: {region['x_pct']}"
        assert 0 <= region['y_pct'] <= 100, f"y_pct out of range: {region['y_pct']}"

    print("   PASSED")
    return True


def test_camera_frame():
    """Test 3: Real-time camera frame prediction via predict_frame()."""
    from agevision_api.age_predictor import predict_frame

    print("\n" + "=" * 74)
    print("   TEST 3: Camera Frame Prediction (predict_frame)")
    print("=" * 74)

    # Find any test image and load as numpy array (simulates camera frame)
    frame_path = None
    fe_samples = os.path.join('..', 'agevision-frontend', 'public', 'samples')
    for f in ['single_1.jpg', 'single_2.jpg', 'group_1.jpg']:
        p = os.path.join(fe_samples, f)
        if os.path.exists(p):
            frame_path = p
            break

    if not frame_path:
        # Try test_images dir
        test_dir = os.path.join('media', 'test_images')
        if os.path.isdir(test_dir):
            for f in os.listdir(test_dir):
                if f.endswith('.jpg'):
                    frame_path = os.path.join(test_dir, f)
                    break

    if not frame_path:
        print("   SKIP: No images available to simulate camera frame")
        return False

    # Load as BGR numpy array (like camera would produce)
    frame = cv2.imread(frame_path)
    assert frame is not None, f"Could not read {frame_path}"

    print(f"   Source Image: {os.path.basename(frame_path)}")
    print(f"   Frame Shape:  {frame.shape} (H x W x C)")

    start = time.time()
    result = predict_frame(frame)
    elapsed = (time.time() - start) * 1000

    print(f"   Faces Detected: {result['face_count']}")
    print(f"   Primary Age:    {result['predicted_age']}")
    print(f"   Time:           {elapsed:.0f}ms")

    if result['faces']:
        for face in result['faces']:
            print(f"     Face #{face['face_id']}: Age {face['predicted_age']}, "
                  f"{face['gender']}, conf={face['confidence']:.2f}")

    # Validate
    assert result['face_count'] >= 1, "Should detect at least 1 face in frame"
    assert result['predicted_age'] > 0, "Age should be > 0"
    assert len(result['faces']) >= 1, "Should have faces list"

    # Test with empty/invalid frame
    empty_result = predict_frame(np.zeros((100, 100, 3), dtype=np.uint8))
    assert empty_result['face_count'] == 0, "Empty black frame should detect 0 faces"
    print("   Empty frame test: PASSED (0 faces as expected)")

    print("   PASSED")
    return True


def test_camera_base64_decode():
    """Test 4: Base64 frame decode (simulates camera API endpoint)."""
    import base64

    print("\n" + "=" * 74)
    print("   TEST 4: Base64 Frame Decode (Camera API simulation)")
    print("=" * 74)

    # Find a test image
    frame_path = None
    fe_samples = os.path.join('..', 'agevision-frontend', 'public', 'samples')
    for f in ['single_1.jpg', 'single_2.jpg']:
        p = os.path.join(fe_samples, f)
        if os.path.exists(p):
            frame_path = p
            break

    if not frame_path:
        print("   SKIP: No images available")
        return False

    # Read image, encode as base64 (like browser canvas.toDataURL())
    frame = cv2.imread(frame_path)
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    b64_str = base64.b64encode(buffer).decode('utf-8')
    data_url = f"data:image/jpeg;base64,{b64_str}"

    print(f"   Source: {os.path.basename(frame_path)}")
    print(f"   Base64 size: {len(b64_str) // 1024}KB")

    # Simulate what predict_camera_view does
    if ',' in data_url:
        b64_clean = data_url.split(',', 1)[1]
    else:
        b64_clean = data_url

    img_bytes = base64.b64decode(b64_clean)
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    decoded_frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    assert decoded_frame is not None, "Failed to decode base64 frame"
    assert decoded_frame.shape[0] > 0 and decoded_frame.shape[1] > 0, "Decoded frame has 0 dimensions"
    print(f"   Decoded shape: {decoded_frame.shape}")

    # Run prediction on decoded frame
    from agevision_api.age_predictor import predict_frame
    start = time.time()
    result = predict_frame(decoded_frame)
    elapsed = (time.time() - start) * 1000

    print(f"   Faces: {result['face_count']}, Age: {result['predicted_age']}, Time: {elapsed:.0f}ms")

    assert result['face_count'] >= 1, "Should detect face in decoded frame"
    print("   PASSED")
    return True


def test_backward_compat():
    """Test 5: Backward compatibility - predict_age_ensemble still works."""
    from agevision_api.age_predictor import predict_age_ensemble

    print("\n" + "=" * 74)
    print("   TEST 5: Backward Compatibility (predict_age_ensemble alias)")
    print("=" * 74)

    frame_path = None
    fe_samples = os.path.join('..', 'agevision-frontend', 'public', 'samples')
    for f in ['single_1.jpg', 'single_2.jpg']:
        p = os.path.join(fe_samples, f)
        if os.path.exists(p):
            frame_path = p
            break

    if not frame_path:
        print("   SKIP: No images available")
        return False

    result = predict_age_ensemble(frame_path, use_alignment=True, detector_mode='ensemble')

    assert result['predicted_age'] > 0, "predict_age_ensemble should return valid age"
    assert 'faces' in result, "Should have faces key"
    print(f"   predict_age_ensemble() -> age={result['predicted_age']}, faces={result['face_count']}")
    print("   PASSED")
    return True


def test_accuracy_benchmark(test_dir: str):
    """Test 6: Celebrity accuracy benchmark."""
    from agevision_api.age_predictor import predict_group_faces

    print("\n" + "=" * 74)
    print("   TEST 6: Celebrity Accuracy Benchmark")
    print("=" * 74)

    print("\n   Downloading celebrity face images...")
    samples = download_faces(test_dir)
    print(f"   Ready: {len(samples)} images")

    if not samples:
        print("   SKIP: No images could be downloaded")
        return False

    all_results = []
    for i, sample in enumerate(samples, 1):
        path = sample['path']
        if not os.path.exists(path):
            continue

        start = time.time()
        pred = predict_group_faces(path)
        elapsed = (time.time() - start) * 1000

        gt = sample['ground_truth_age']
        err = abs(pred['predicted_age'] - gt)
        tag = "OK" if err <= 5 else "FAIR" if err <= 10 else "MISS"

        print(f"   [{i:2d}/{len(samples)}] {sample['filename']:<28} "
              f"GT:{gt:>3} Pred:{pred['predicted_age']:>3} Err:{err:>3} [{tag}] {elapsed:.0f}ms")

        all_results.append({
            'filename': sample['filename'],
            'ground_truth_age': gt,
            'predicted_age': pred['predicted_age'],
            'gender': pred.get('gender', 'Unknown'),
            'ethnicity': sample['ethnicity'],
            'total_time_ms': elapsed,
        })

    metrics = compute_metrics(all_results)
    if 'error' not in metrics:
        print_report(metrics, all_results)

        # Save report
        report_path = os.path.join(test_dir, 'accuracy_report.json')
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'model': 'MiVOLO v2',
            'metrics': metrics,
            'results': all_results,
        }

        class NumpyEncoder(json.JSONEncoder):
            def default(self, o):
                if isinstance(o, (np.integer,)):
                    return int(o)
                if isinstance(o, (np.floating,)):
                    return float(o)
                if isinstance(o, np.ndarray):
                    return o.tolist()
                return super().default(o)

        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, cls=NumpyEncoder)
        print(f"\n   Report saved: {report_path}")

    return True


def main():
    print("=" * 74)
    print("   AgeVision - End-to-End Age Prediction Test")
    print("   Model: MiVOLO v2 (YOLOv8 + Vision Transformer)")
    print("=" * 74)

    test_dir = os.path.join('media', 'test_images')
    os.makedirs(test_dir, exist_ok=True)

    results = {}
    tests = [
        ("Single Face",       lambda: test_single_face(test_dir)),
        ("Group Faces",       lambda: test_group_faces(test_dir)),
        ("Camera Frame",      lambda: test_camera_frame()),
        ("Base64 Decode",     lambda: test_camera_base64_decode()),
        ("Backward Compat",   lambda: test_backward_compat()),
        ("Accuracy Benchmark", lambda: test_accuracy_benchmark(test_dir)),
    ]

    for name, test_fn in tests:
        try:
            results[name] = test_fn()
        except Exception as e:
            print(f"   FAILED: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False

    # Summary
    print("\n" + "=" * 74)
    print("   TEST SUMMARY")
    print("=" * 74)
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    for name, result in results.items():
        status = "PASSED" if result else "FAILED/SKIPPED"
        print(f"   {name:<25} {status}")
    print(f"\n   {passed}/{total} tests passed")
    print("=" * 74)


if __name__ == '__main__':
    main()
