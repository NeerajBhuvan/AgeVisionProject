"""
Age Progression Module – Integration Tests
============================================
Run with:
    cd agevision_backend
    python manage.py test age_progression.tests

Or standalone (no Django server required for unit parts):
    python -m pytest age_progression/tests.py -v

Tests:
  1. Face detection (Haar cascade)
  2. Age estimation (DeepFace)
  3. Age progression (image-processing fallback)
  4. Image processor helpers
  5. API endpoint (via DRF test client)
  6. MongoDB storage
"""

import os
import sys
import io
import json
import time
import tempfile
import unittest

import cv2
import numpy as np

# Ensure the Django project is on the path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'agevision_backend.settings')

import django
django.setup()

from django.test import TestCase, override_settings
from django.contrib.auth.models import User
from rest_framework.test import APIClient

from age_progression.utils.face_detector import FaceDetector
from age_progression.utils.age_estimator import AgeEstimator
from age_progression.utils.age_progressor import AgeProgressor
from age_progression.utils.image_processor import ImageProcessor
from age_progression.models import ProgressionDB


def _make_test_face(size: int = 300) -> np.ndarray:
    """
    Generate a synthetic face-like image (oval on a background)
    that the Haar cascade can plausibly detect.
    """
    img = np.full((size, size, 3), 200, dtype=np.uint8)  # light grey bg
    cx, cy = size // 2, size // 2
    # Skin-coloured ellipse
    cv2.ellipse(img, (cx, cy), (80, 110), 0, 0, 360, (180, 200, 230), -1)
    # Eyes
    cv2.circle(img, (cx - 30, cy - 20), 8, (50, 50, 50), -1)
    cv2.circle(img, (cx + 30, cy - 20), 8, (50, 50, 50), -1)
    # Mouth
    cv2.ellipse(img, (cx, cy + 35), (25, 10), 0, 0, 180, (80, 80, 150), 2)
    return img


def _save_temp_image(img: np.ndarray) -> str:
    """Write an image to a temp file and return its path."""
    fd, path = tempfile.mkstemp(suffix='.jpg')
    os.close(fd)
    cv2.imwrite(path, img)
    return path


# ──────────────────────────────────────────────────────────────────────
#  1. Face Detection
# ──────────────────────────────────────────────────────────────────────

class TestFaceDetector(TestCase):
    """Test face detection utilities."""

    def test_detect_faces_returns_list(self):
        img = _make_test_face()
        faces = FaceDetector.detect_faces(img)
        self.assertIsInstance(faces, list)
        # Should return at least the full-image fallback
        self.assertGreaterEqual(len(faces), 1)

    def test_detect_faces_returns_crop_and_bbox(self):
        img = _make_test_face()
        faces = FaceDetector.detect_faces(img)
        crop, bbox = faces[0]
        self.assertEqual(len(bbox), 4)
        self.assertIsInstance(crop, np.ndarray)
        self.assertEqual(len(crop.shape), 3)

    def test_detect_on_none_returns_empty(self):
        self.assertEqual(FaceDetector.detect_faces(None), [])


# ──────────────────────────────────────────────────────────────────────
#  2. Age Estimation
# ──────────────────────────────────────────────────────────────────────

class TestAgeEstimator(TestCase):
    """Test DeepFace-based age estimation."""

    def test_estimate_returns_dict(self):
        img = _make_test_face()
        path = _save_temp_image(img)
        try:
            result = AgeEstimator.estimate_age(path)
            self.assertIn('age', result)
            self.assertIn('gender', result)
            self.assertIsInstance(result['age'], int)
        finally:
            os.remove(path)

    def test_estimate_fallback_on_bad_image(self):
        """Should gracefully return defaults on unreadable input."""
        result = AgeEstimator.estimate_age('/tmp/nonexistent_image.jpg')
        self.assertIn('age', result)  # fallback returns age=25


# ──────────────────────────────────────────────────────────────────────
#  3. Age Progression
# ──────────────────────────────────────────────────────────────────────

class TestAgeProgressor(TestCase):
    """Test the aging pipeline (image-processing path)."""

    def test_progress_returns_image(self):
        face = _make_test_face(256)
        progressor = AgeProgressor()
        result = progressor.progress_age(face, 25, 70)
        self.assertIn('image', result)
        self.assertEqual(result['image'].shape, face.shape)

    def test_method_used_is_image_processing(self):
        """Without a GAN model, should fall back to image_processing."""
        face = _make_test_face(256)
        progressor = AgeProgressor()
        result = progressor.progress_age(face, 20, 60)
        self.assertIn(result['method_used'], ('image_processing', 'deepface_hybrid'))

    def test_aging_params_populated(self):
        face = _make_test_face(256)
        progressor = AgeProgressor()
        result = progressor.progress_age(face, 30, 80)
        params = result.get('aging_params', {})
        self.assertIn('wrinkle', params)
        self.assertIn('gray', params)

    def test_progression_modifies_image(self):
        """The output should differ from the input."""
        face = _make_test_face(256)
        progressor = AgeProgressor()
        result = progressor.progress_age(face, 20, 75)
        diff = cv2.absdiff(face, result['image'])
        self.assertGreater(diff.sum(), 0)


# ──────────────────────────────────────────────────────────────────────
#  4. Image Processor
# ──────────────────────────────────────────────────────────────────────

class TestImageProcessor(TestCase):
    """Test helper methods in ImageProcessor."""

    def test_preprocess_resizes(self):
        img = _make_test_face(400)
        out = ImageProcessor.preprocess_for_model(img, (256, 256))
        self.assertEqual(out.shape[:2], (256, 256))

    def test_comparison_view_shape(self):
        orig = _make_test_face(256)
        prog = _make_test_face(256)
        comp = ImageProcessor.create_comparison_view(orig, prog, 25, 70)
        self.assertEqual(len(comp.shape), 3)
        self.assertGreater(comp.shape[1], 256 * 2)  # wider than 2 images

    def test_save_and_read(self):
        img = _make_test_face(128)
        info = ImageProcessor.save_image(img, 'test_age_progression')
        self.assertTrue(os.path.isfile(info['absolute_path']))
        # Cleanup
        os.remove(info['absolute_path'])

    def test_aging_texture(self):
        img = _make_test_face(256)
        out = ImageProcessor.add_aging_texture(img, 0.5)
        self.assertEqual(out.shape, img.shape)


# ──────────────────────────────────────────────────────────────────────
#  5. API Endpoint
# ──────────────────────────────────────────────────────────────────────

class TestProgressionAPI(TestCase):
    """Test the REST API endpoint via Django's test client."""

    def setUp(self):
        self.user = User.objects.create_user(
            username='testuser_ap',
            password='TestPass123!',
        )
        self.client = APIClient()
        self.client.force_authenticate(user=self.user)

    def _upload_image(self, target_age: int = 70):
        img = _make_test_face(300)
        _, buf = cv2.imencode('.jpg', img)
        fp = io.BytesIO(buf.tobytes())
        fp.name = 'test_face.jpg'
        return self.client.post(
            '/api/age-progression/progress/',
            {'image': fp, 'target_age': target_age},
            format='multipart',
        )

    def test_endpoint_requires_auth(self):
        anon_client = APIClient()
        img = _make_test_face(100)
        _, buf = cv2.imencode('.jpg', img)
        fp = io.BytesIO(buf.tobytes())
        fp.name = 'test.jpg'
        resp = anon_client.post(
            '/api/age-progression/progress/',
            {'image': fp, 'target_age': 60},
            format='multipart',
        )
        self.assertEqual(resp.status_code, 401)

    def test_missing_image_returns_400(self):
        resp = self.client.post(
            '/api/age-progression/progress/',
            {'target_age': 60},
            format='multipart',
        )
        self.assertEqual(resp.status_code, 400)

    def test_missing_target_age_returns_400(self):
        img = _make_test_face(100)
        _, buf = cv2.imencode('.jpg', img)
        fp = io.BytesIO(buf.tobytes())
        fp.name = 'test.jpg'
        resp = self.client.post(
            '/api/age-progression/progress/',
            {'image': fp},
            format='multipart',
        )
        self.assertEqual(resp.status_code, 400)

    def test_invalid_target_age_returns_400(self):
        img = _make_test_face(100)
        _, buf = cv2.imencode('.jpg', img)
        fp = io.BytesIO(buf.tobytes())
        fp.name = 'test.jpg'
        resp = self.client.post(
            '/api/age-progression/progress/',
            {'image': fp, 'target_age': 150},
            format='multipart',
        )
        self.assertEqual(resp.status_code, 400)

    def test_successful_progression(self):
        resp = self._upload_image(target_age=90)
        # May succeed or return 400 if estimated age >= target
        self.assertIn(resp.status_code, [200, 400])
        if resp.status_code == 200:
            data = resp.json()
            self.assertTrue(data['success'])
            self.assertIn('original_image_url', data)
            self.assertIn('progressed_image_url', data)
            self.assertIn('comparison_image_url', data)

    def test_health_endpoint(self):
        resp = self.client.get('/api/age-progression/health/')
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data['success'])
        self.assertIn('capabilities', data)


# ──────────────────────────────────────────────────────────────────────
#  6. MongoDB Storage
# ──────────────────────────────────────────────────────────────────────

class TestMongoDBStorage(TestCase):
    """Test ProgressionDB CRUD operations (requires MongoDB running)."""

    _record_id = None

    def test_01_insert(self):
        """Insert a test record."""
        try:
            rid = ProgressionDB.insert_record(
                user_id='test_user_999',
                original_image_path='test/orig.jpg',
                progressed_image_path='test/prog.jpg',
                comparison_image_path='test/comp.jpg',
                current_age=25,
                target_age=70,
                processing_time=1234.5,
                method_used='image_processing',
            )
            self.assertIsInstance(rid, str)
            self.assertGreater(len(rid), 0)
            TestMongoDBStorage._record_id = rid
        except Exception as exc:
            self.skipTest(f'MongoDB not available: {exc}')

    def test_02_get(self):
        """Retrieve the inserted record."""
        if not TestMongoDBStorage._record_id:
            self.skipTest('No record to fetch (insert may have been skipped)')
        doc = ProgressionDB.get_record(TestMongoDBStorage._record_id)
        self.assertIsNotNone(doc)
        self.assertEqual(doc['current_age'], 25)
        self.assertEqual(doc['target_age'], 70)

    def test_03_user_records(self):
        """List records for a user."""
        try:
            records = ProgressionDB.get_user_records('test_user_999')
            self.assertIsInstance(records, list)
        except Exception:
            self.skipTest('MongoDB not available')

    def test_04_stats(self):
        """Check stats aggregation."""
        try:
            stats = ProgressionDB.get_stats('test_user_999')
            self.assertIn('total', stats)
        except Exception:
            self.skipTest('MongoDB not available')

    def test_05_delete(self):
        """Clean up test record."""
        if not TestMongoDBStorage._record_id:
            self.skipTest('No record to delete')
        ok = ProgressionDB.delete_record(TestMongoDBStorage._record_id)
        self.assertTrue(ok)


if __name__ == '__main__':
    unittest.main()
