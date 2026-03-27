"""
HRFAE GAN Accuracy Test
========================
Tests the integrated HRFAE GAN model against DeepFace age detection
to measure how accurately the model transforms faces to target ages.
"""

import os
import sys
import time
import warnings

warnings.filterwarnings('ignore')

# Django setup
os.environ['DJANGO_SETTINGS_MODULE'] = 'agevision_backend.settings'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import django
django.setup()

import cv2
import numpy as np
import requests
from deepface import DeepFace
from agevision_api.gan_progression import GANProgressionPipeline


def download_test_face():
    """Download or create a test face image."""
    # First check for existing real face images in media/
    for candidate in [
        os.path.join('media', 'originals', 'test_prog1.jpeg'),
        os.path.join('media', 'originals', 'test_prog1.jpg'),
    ]:
        if os.path.isfile(candidate):
            print(f"Using existing face image: {candidate}")
            return candidate

    test_dir = os.path.join('media', 'temp')
    os.makedirs(test_dir, exist_ok=True)
    test_path = os.path.join(test_dir, 'test_real_face.jpg')

    # Try to download a real face from a public source
    face_urls = [
        'https://raw.githubusercontent.com/InterDigitalInc/HRFAE/master/test/input/test.jpg',
        'https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Camponotus_flavomarginatus_ant.jpg/256px-Camponotus_flavomarginatus_ant.jpg',
    ]
    for face_url in face_urls:
        try:
            resp = requests.get(face_url, timeout=15, verify=False)
            resp.raise_for_status()
            with open(test_path, 'wb') as f:
                f.write(resp.content)
            print(f"Downloaded test face: {len(resp.content)} bytes")
            return test_path
        except Exception:
            continue

    print("Using synthetic face (results may be less accurate)")
    # Create synthetic face
    img = np.ones((256, 256, 3), dtype=np.uint8) * 180
    cv2.circle(img, (128, 128), 90, (200, 185, 170), -1)
    cv2.circle(img, (98, 108), 12, (40, 40, 40), -1)
    cv2.circle(img, (158, 108), 12, (40, 40, 40), -1)
    cv2.ellipse(img, (128, 160), (20, 8), 0, 0, 180, (140, 90, 90), 2)
    cv2.imwrite(test_path, img)
    return test_path


def detect_age(image_path):
    """Use DeepFace to detect age in an image."""
    try:
        analysis = DeepFace.analyze(
            image_path, actions=['age'], enforce_detection=False
        )
        if isinstance(analysis, list):
            analysis = analysis[0]
        return int(analysis.get('age', -1))
    except Exception:
        return -1


def main():
    print("=" * 70)
    print("HRFAE GAN Age Progression — Accuracy Test")
    print("=" * 70)

    # Prepare test image
    test_path = download_test_face()
    baseline_age = detect_age(test_path)
    print(f"Baseline detected age: {baseline_age}")

    # Run GAN pipeline
    pipeline = GANProgressionPipeline()
    is_gan = pipeline.is_gan_available()
    print(f"GAN model available: {is_gan}")

    target_ages = [20, 30, 40, 50, 60, 70, 80]
    results = []

    print(f"\n{'Target':>8} | {'Detected':>10} | {'Error':>8} | "
          f"{'Direction':>10} | {'Time':>8} | Model")
    print("-" * 72)

    for target in target_ages:
        t0 = time.time()
        result = pipeline.run(test_path, baseline_age, target)
        elapsed = (time.time() - t0) * 1000

        detected = detect_age(result['output_path'])
        error = abs(detected - target) if detected > 0 else float('inf')

        direction_ok = (
            (target > baseline_age and detected > baseline_age) or
            (target < baseline_age and detected < baseline_age) or
            (target == baseline_age)
        )

        results.append({
            'target': target,
            'detected': detected,
            'error': error,
            'direction': direction_ok,
            'time': elapsed,
            'model': result['model_type'],
        })

        dir_str = "OK" if direction_ok else "WRONG"
        model = result['model_type']
        print(f"{target:>8} | {detected:>10} | {error:>8.1f} | "
              f"{dir_str:>10} | {elapsed:>7.0f}ms | {model}")

    # Summary
    valid = [r for r in results if r['detected'] > 0]
    if valid:
        mae = sum(r['error'] for r in valid) / len(valid)
        dir_acc = sum(1 for r in valid if r['direction']) / len(valid) * 100
        avg_time = sum(r['time'] for r in valid) / len(valid)

        print(f"\n{'=' * 70}")
        print(f"ACCURACY SUMMARY")
        print(f"{'=' * 70}")
        print(f"  MAE (Mean Absolute Error):  {mae:.1f} years")
        print(f"  Direction accuracy:          {dir_acc:.0f}%")
        print(f"  Average inference time:      {avg_time:.0f} ms")
        print(f"  Model:                       {results[0]['model']}")
        print(f"  Baseline age:                {baseline_age}")
        print(f"  Pre-trained checkpoint:      epoch 20 (FFHQ)")

        # Comparison with old OpenCV approach (MAE was 18.2)
        if mae < 18.2:
            improvement = ((18.2 - mae) / 18.2) * 100
            print(f"\n  Improvement over OpenCV:     {improvement:.0f}% (MAE 18.2 -> {mae:.1f})")
        print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
