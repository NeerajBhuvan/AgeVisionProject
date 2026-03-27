"""
End-to-End Age Progression Test
================================
Downloads a sample face image, runs the aging pipeline at multiple
target ages, then re-analyses each output with DeepFace to measure
how well the perceived age shifts toward the target.

Usage:
    python test_progression.py
"""

import os
import sys
import time
import json
import urllib.request
import django

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'agevision_backend.settings')
django.setup()

import cv2
import numpy as np
from deepface import DeepFace
from agevision_api.age_progression import AgingPipeline

MEDIA_ROOT = os.path.join(os.path.dirname(__file__), 'media')
TEST_DIR = os.path.join(MEDIA_ROOT, 'test_progression')
os.makedirs(TEST_DIR, exist_ok=True)


def generate_synthetic_face(path: str):
    """Generate a simple synthetic face image for testing when no real photo is available."""
    img = np.ones((400, 400, 3), dtype=np.uint8) * 220  # light gray bg

    # Head oval
    cv2.ellipse(img, (200, 200), (120, 160), 0, 0, 360, (195, 175, 160), -1)
    # Eyes
    cv2.ellipse(img, (160, 170), (18, 10), 0, 0, 360, (60, 40, 30), -1)
    cv2.ellipse(img, (240, 170), (18, 10), 0, 0, 360, (60, 40, 30), -1)
    cv2.circle(img, (160, 170), 6, (30, 20, 15), -1)
    cv2.circle(img, (240, 170), 6, (30, 20, 15), -1)
    # Nose
    pts = np.array([[200, 195], [188, 230], [212, 230]], np.int32)
    cv2.polylines(img, [pts], False, (150, 120, 100), 2)
    # Mouth
    cv2.ellipse(img, (200, 265), (30, 12), 0, 0, 180, (140, 80, 80), 2)
    # Eyebrows
    cv2.line(img, (138, 148), (180, 145), (80, 60, 50), 3)
    cv2.line(img, (220, 145), (262, 148), (80, 60, 50), 3)
    # Hair
    cv2.ellipse(img, (200, 120), (130, 80), 0, 180, 360, (50, 35, 25), -1)

    cv2.imwrite(path, img)
    print(f"  Generated synthetic face: {path}")


def download_sample_face(path: str) -> bool:
    """Try to download a real face image from thispersondoesnotexist or a public sample."""
    urls = [
        "https://thispersondoesnotexist.com",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Camponotus_flavomarginatus_ant.jpg/320px-Camponotus_flavomarginatus_ant.jpg",
    ]
    for url in urls[:1]:  # Try the first URL only
        try:
            print(f"  Downloading sample face from {url}...")
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            resp = urllib.request.urlopen(req, timeout=10)
            data = resp.read()
            with open(path, 'wb') as f:
                f.write(data)
            # Verify it's a valid image
            test = cv2.imread(path)
            if test is not None and test.shape[0] > 50:
                print(f"  Downloaded OK: {test.shape}")
                return True
        except Exception as e:
            print(f"  Download failed: {e}")
    return False


def detect_age(image_path: str) -> dict:
    """Run DeepFace analysis and return age + gender."""
    try:
        results = DeepFace.analyze(
            img_path=image_path,
            actions=['age', 'gender'],
            enforce_detection=False,
            silent=True,
        )
        r = results[0] if isinstance(results, list) else results
        return {
            'age': r.get('age', 0),
            'gender': r.get('dominant_gender', 'Unknown'),
        }
    except Exception as e:
        return {'age': 0, 'gender': 'Unknown', 'error': str(e)}


def run_test():
    print("=" * 60)
    print("  AgeVision — End-to-End Progression Test")
    print("=" * 60)

    # Step 1: Get a test image
    source_path = os.path.join(TEST_DIR, 'source_face.jpg')

    if not os.path.exists(source_path):
        downloaded = download_sample_face(source_path)
        if not downloaded:
            print("  Could not download a real face, using synthetic face.")
            generate_synthetic_face(source_path)

    # Step 2: Detect baseline age
    print("\n[1] Detecting baseline age with DeepFace...")
    baseline = detect_age(source_path)
    baseline_age = baseline['age']
    gender = baseline['gender']
    print(f"    Detected age : {baseline_age}")
    print(f"    Gender       : {gender}")

    if baseline_age == 0:
        print("    WARNING: DeepFace could not detect a face. Results will be approximate.")
        baseline_age = 25  # fallback

    # Step 3: Run progression at multiple target ages
    target_ages = [10, 20, 30, 40, 50, 60, 70, 80]
    pipeline = AgingPipeline()

    results = []
    print(f"\n[2] Running progression pipeline for targets: {target_ages}")
    print("-" * 60)
    print(f"  {'Target':>8} | {'Detected':>10} | {'Δ from Target':>14} | {'Direction':>10} | {'Time':>8}")
    print("-" * 60)

    for target in target_ages:
        t0 = time.time()
        try:
            result = pipeline.run(source_path, baseline_age, target)
            elapsed = (time.time() - t0) * 1000

            output_path = result['output_path']

            # Re-analyse the progressed image
            recheck = detect_age(output_path)
            detected_age = recheck['age']

            delta = detected_age - target
            # Did age move in the right direction?
            age_diff = target - baseline_age
            detected_diff = detected_age - baseline_age
            if age_diff == 0:
                direction = "SAME"
            elif (age_diff > 0 and detected_diff > 0) or (age_diff < 0 and detected_diff < 0):
                direction = "CORRECT"
            else:
                direction = "WRONG"

            results.append({
                'target': target,
                'detected': detected_age,
                'delta': delta,
                'direction': direction,
                'time_ms': round(elapsed, 1),
                'output': output_path,
                'insights': result['insights'],
            })

            print(f"  {target:>8} | {detected_age:>10} | {delta:>+14} | {direction:>10} | {elapsed:>7.0f}ms")

        except Exception as e:
            print(f"  {target:>8} | {'ERROR':>10} | {str(e)[:30]}")
            results.append({'target': target, 'error': str(e)})

    # Step 4: Compute summary metrics
    print("\n" + "=" * 60)
    print("  ACCURACY SUMMARY")
    print("=" * 60)

    valid = [r for r in results if 'detected' in r]

    if not valid:
        print("  No valid results to analyse.")
        return

    deltas = [abs(r['delta']) for r in valid]
    directions_correct = sum(1 for r in valid if r['direction'] == 'CORRECT')
    directions_same = sum(1 for r in valid if r['direction'] == 'SAME')

    mae = sum(deltas) / len(deltas)
    direction_accuracy = (directions_correct + directions_same) / len(valid) * 100
    avg_time = sum(r['time_ms'] for r in valid) / len(valid)

    # Age shift magnitude (how much the detected age actually moved per target change)
    shifts = []
    for r in valid:
        intended_shift = abs(r['target'] - baseline_age)
        actual_shift = abs(r['detected'] - baseline_age)
        if intended_shift > 0:
            shifts.append(actual_shift / intended_shift * 100)

    avg_shift_ratio = sum(shifts) / len(shifts) if shifts else 0

    print(f"""
    Baseline Age         : {baseline_age}
    Gender               : {gender}
    Targets Tested       : {len(valid)}

    --- Key Metrics ---
    MAE (Mean Abs Error) : {mae:.1f} years
    Direction Accuracy   : {direction_accuracy:.0f}% ({directions_correct}/{len(valid)} correct direction)
    Avg Shift Ratio      : {avg_shift_ratio:.1f}% (how much age actually moved vs intended)
    Avg Processing Time  : {avg_time:.0f} ms

    --- Interpretation ---""")

    if mae < 5:
        print("    MAE < 5  → Excellent age transformation accuracy")
    elif mae < 10:
        print("    MAE < 10 → Good accuracy, reasonable approximation")
    elif mae < 15:
        print("    MAE < 15 → Moderate accuracy, visible aging effects")
    else:
        print(f"    MAE = {mae:.1f} → Low accuracy — expected for rule-based (non-GAN) approach")

    print(f"""
    NOTE: This is a rule-based OpenCV pipeline, NOT a trained GAN.
    The aging effects (wrinkles, graying, texture) are cosmetic overlays.
    DeepFace re-detection reflects how much the visual appearance 
    changed, but the underlying face identity is preserved.

    For GAN-level accuracy (MAE ~3-5 years), you'd need a model 
    trained on UTKFace/FFHQ with ~50K+ paired face images.
    """)

    # Save results to JSON
    report_path = os.path.join(TEST_DIR, 'test_results.json')
    with open(report_path, 'w') as f:
        json.dump({
            'baseline_age': baseline_age,
            'gender': gender,
            'mae': round(mae, 2),
            'direction_accuracy': round(direction_accuracy, 1),
            'shift_ratio': round(avg_shift_ratio, 1),
            'avg_time_ms': round(avg_time, 1),
            'results': [{k: v for k, v in r.items() if k != 'insights'} for r in valid],
        }, f, indent=2)
    print(f"  Full results saved to: {report_path}")
    print("=" * 60)


if __name__ == '__main__':
    run_test()
