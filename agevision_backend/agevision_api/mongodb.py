"""
MongoDB connection utility using PyMongo.
Django's ORM (SQLite) handles auth, admin, sessions.
PyMongo handles ALL application-specific data in MongoDB:
  - users (extended profiles + encrypted passwords)
  - password_resets (recovery tokens)
  - predictions (age prediction records)
  - progressions (age progression records)
  - user_settings (per-user preferences)
"""

import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

from pymongo import MongoClient, ASCENDING, DESCENDING
from bson import ObjectId
from django.conf import settings

from .crypto import agevision_encrypt, agevision_decrypt


def _iso_utc(dt) -> Optional[str]:
    """Serialize a datetime to ISO 8601 with explicit UTC suffix.

    PyMongo returns naive datetimes (UTC but without tzinfo).
    Without the '+00:00' / 'Z' suffix, JavaScript's ``new Date()``
    interprets the string as local time, causing time display mismatches.
    """
    if dt is None:
        return None
    s = dt.isoformat()
    # If already has timezone info, return as-is
    if s.endswith('Z') or '+' in s[19:] or s[19:].count('-') > 0:
        return s
    # Append UTC indicator
    return s + '+00:00'


class MongoDB:
    """Singleton MongoDB connection manager."""
    _client = None
    _db = None

    @classmethod
    def get_client(cls):
        if cls._client is None:
            config = settings.MONGODB_CONFIG
            cls._client = MongoClient(
                host=config['HOST'],
                port=config['PORT'],
                serverSelectionTimeoutMS=2000,
                connectTimeoutMS=2000,
            )
        return cls._client

    @classmethod
    def get_db(cls):
        if cls._db is None:
            config = settings.MONGODB_CONFIG
            cls._db = cls.get_client()[config['NAME']]
        return cls._db

    @classmethod
    def get_collection(cls, name):
        return cls.get_db()[name]

    @classmethod
    def close(cls):
        if cls._client:
            cls._client.close()
            cls._client = None
            cls._db = None

    # ── Index bootstrapping ──────────────────────────────────────────
    @classmethod
    def ensure_indexes(cls):
        """Create indexes for all collections (idempotent)."""
        # Users collection
        users = cls.get_collection('users')
        users.create_index([('django_user_id', ASCENDING)], unique=True)
        users.create_index([('username', ASCENDING)], unique=True)
        users.create_index([('email', ASCENDING)], unique=True)

        # Predictions collection
        predictions = cls.get_collection('predictions')
        predictions.create_index([('user_id', ASCENDING)])
        predictions.create_index([('created_at', DESCENDING)])

        # Progressions collection
        progressions = cls.get_collection('progressions')
        progressions.create_index([('user_id', ASCENDING)])
        progressions.create_index([('created_at', DESCENDING)])

        # User settings collection
        user_settings = cls.get_collection('user_settings')
        user_settings.create_index([('user_id', ASCENDING)], unique=True)

        # Batch predictions collection
        batch_predictions = cls.get_collection('batch_predictions')
        batch_predictions.create_index([('user_id', ASCENDING)])
        batch_predictions.create_index([('created_at', DESCENDING)])


class MongoUserManager:
    """Helper class for user CRUD operations in MongoDB."""

    COLLECTION = 'users'

    @classmethod
    def _col(cls):
        return MongoDB.get_collection(cls.COLLECTION)

    @classmethod
    def create_user(cls, *, django_user_id: int, username: str, email: str,
                    first_name: str = '', last_name: str = '',
                    raw_password: str = '') -> dict:
        """Insert a new user document and return it."""
        now = datetime.now(timezone.utc)
        doc = {
            'django_user_id': django_user_id,
            'username': username,
            'email': email,
            'first_name': first_name,
            'last_name': last_name,
            'plan': 'free',
            'avatar_url': '',
            'is_active': True,
            'created_at': now,
            'updated_at': now,
        }
        # Store encrypted password (AgeVision custom encryption)
        if raw_password:
            doc['encrypted_password'] = agevision_encrypt(raw_password, context=username)
        result = cls._col().insert_one(doc)
        doc['_id'] = str(result.inserted_id)
        return cls._serialize(doc)

    @classmethod
    def get_by_django_id(cls, django_user_id: int) -> Optional[dict]:
        """Find user by Django User pk."""
        doc = cls._col().find_one({'django_user_id': django_user_id})
        return cls._serialize(doc) if doc else None

    @classmethod
    def get_by_username(cls, username: str) -> Optional[dict]:
        doc = cls._col().find_one({'username': username})
        return cls._serialize(doc) if doc else None

    @classmethod
    def update_user(cls, django_user_id: int, **fields) -> Optional[dict]:
        """Update arbitrary fields on a user document."""
        fields['updated_at'] = datetime.now(timezone.utc)
        cls._col().update_one(
            {'django_user_id': django_user_id},
            {'$set': fields},
        )
        return cls.get_by_django_id(django_user_id)

    @classmethod
    def update_last_login(cls, django_user_id: int):
        cls._col().update_one(
            {'django_user_id': django_user_id},
            {'$set': {'last_login': datetime.now(timezone.utc)}},
        )

    @classmethod
    def delete_user(cls, django_user_id: int):
        cls._col().delete_one({'django_user_id': django_user_id})

    # ── Password helpers ─────────────────────────────────────────────
    @classmethod
    def store_encrypted_password(cls, django_user_id: int, raw_password: str, username: str):
        """Encrypt and store the password for an existing user."""
        encrypted = agevision_encrypt(raw_password, context=username)
        cls._col().update_one(
            {'django_user_id': django_user_id},
            {'$set': {
                'encrypted_password': encrypted,
                'updated_at': datetime.now(timezone.utc),
            }},
        )

    @classmethod
    def recover_password(cls, django_user_id: int) -> Optional[str]:
        """Decrypt and return the stored password for a user, or None."""
        doc = cls._col().find_one(
            {'django_user_id': django_user_id},
            {'encrypted_password': 1, 'username': 1},
        )
        if not doc or 'encrypted_password' not in doc:
            return None
        return agevision_decrypt(doc['encrypted_password'], context=doc.get('username', ''))

    @classmethod
    def get_by_email(cls, email: str) -> Optional[dict]:
        """Find user by email address."""
        doc = cls._col().find_one({'email': email})
        return cls._serialize(doc) if doc else None

    # ── Admin / platform helpers ─────────────────────────────────────
    @classmethod
    def list_all(cls, skip: int = 0, limit: int = 50,
                 search: Optional[str] = None) -> list:
        """Paginated list of all users, newest first.

        Optional case-insensitive regex search on username/email.
        """
        query: dict = {}
        if search:
            # Escape regex metacharacters for safety
            import re
            safe = re.escape(search)
            query = {
                '$or': [
                    {'username': {'$regex': safe, '$options': 'i'}},
                    {'email': {'$regex': safe, '$options': 'i'}},
                ]
            }
        cursor = (
            cls._col()
            .find(query)
            .sort('created_at', DESCENDING)
            .skip(skip)
            .limit(limit)
        )
        return [cls._serialize(doc) for doc in cursor]

    @classmethod
    def count_all(cls, search: Optional[str] = None) -> int:
        """Count all users (optionally filtered by search)."""
        query: dict = {}
        if search:
            import re
            safe = re.escape(search)
            query = {
                '$or': [
                    {'username': {'$regex': safe, '$options': 'i'}},
                    {'email': {'$regex': safe, '$options': 'i'}},
                ]
            }
        return cls._col().count_documents(query)

    @classmethod
    def set_active(cls, django_user_id: int, is_active: bool):
        """Mirror the Django is_active flag to the MongoDB user document."""
        cls._col().update_one(
            {'django_user_id': django_user_id},
            {'$set': {
                'is_active': bool(is_active),
                'updated_at': datetime.now(timezone.utc),
            }},
        )

    # ── Serialisation ────────────────────────────────────────────────
    @staticmethod
    def _serialize(doc: dict) -> Optional[dict]:
        """Convert MongoDB document to JSON-safe dict."""
        if doc is None:
            return None
        return {
            'id': doc.get('django_user_id'),
            'username': doc.get('username', ''),
            'email': doc.get('email', ''),
            'first_name': doc.get('first_name', ''),
            'last_name': doc.get('last_name', ''),
            'plan': doc.get('plan', 'free'),
            'avatar_url': doc.get('avatar_url', ''),
            'is_active': doc.get('is_active', True),
            'created_at': _iso_utc(doc.get('created_at')),
            'updated_at': _iso_utc(doc.get('updated_at')),
            'last_login': _iso_utc(doc.get('last_login')),
        }


class MongoPasswordResetManager:
    """
    Manage password-reset tokens entirely inside MongoDB.
    Each token lives in the `password_resets` collection and expires after
    15 minutes.  No email is sent — the token is returned to the client
    directly so the whole flow stays in-app.
    """

    COLLECTION = 'password_resets'
    TOKEN_LIFETIME_MINUTES = 15

    @classmethod
    def _col(cls):
        return MongoDB.get_collection(cls.COLLECTION)

    @classmethod
    def ensure_indexes(cls):
        """Create indexes (idempotent): unique token, TTL auto-cleanup."""
        col = cls._col()
        col.create_index([('token', ASCENDING)], unique=True)
        col.create_index([('django_user_id', ASCENDING)])
        # MongoDB TTL index: documents are automatically deleted once
        # `expires_at` is in the past.
        col.create_index([('expires_at', ASCENDING)], expireAfterSeconds=0)

    @classmethod
    def create_reset_token(
        cls,
        django_user_id: int,
        email: str,
        username: str,
    ) -> str:
        """
        Generate a cryptographically-secure reset token, store it in
        MongoDB, and return the token string.  Any previous unused tokens
        for this user are invalidated first.
        """
        cls.ensure_indexes()

        # Invalidate previous pending tokens for this user
        cls._col().update_many(
            {'django_user_id': django_user_id, 'is_used': False},
            {'$set': {'is_used': True}},
        )

        token = secrets.token_urlsafe(48)
        now = datetime.now(timezone.utc)
        cls._col().insert_one({
            'django_user_id': django_user_id,
            'email': email,
            'username': username,
            'token': token,
            'is_used': False,
            'created_at': now,
            'expires_at': now + timedelta(minutes=cls.TOKEN_LIFETIME_MINUTES),
        })
        return token

    @classmethod
    def verify_token(cls, token: str) -> Optional[dict]:
        """
        Look up a reset token.  Returns the MongoDB document if the
        token exists, has not been used, and has not expired.
        """
        doc = cls._col().find_one({
            'token': token,
            'is_used': False,
            'expires_at': {'$gt': datetime.now(timezone.utc)},
        })
        return doc

    @classmethod
    def invalidate_token(cls, token: str):
        """Mark a token as used so it cannot be reused."""
        cls._col().update_one(
            {'token': token},
            {'$set': {'is_used': True}},
        )


class MongoPredictionManager:
    """CRUD operations for prediction records in MongoDB."""

    COLLECTION = 'predictions'

    @classmethod
    def _col(cls):
        return MongoDB.get_collection(cls.COLLECTION)

    @staticmethod
    def _to_native(value):
        """Convert numpy types to Python native types for MongoDB storage."""
        try:
            import numpy as np
            if isinstance(value, (np.integer,)):
                return int(value)
            if isinstance(value, (np.floating,)):
                return float(value)
            if isinstance(value, np.ndarray):
                return value.tolist()
        except ImportError:
            pass
        return value

    @classmethod
    def _clean_list(cls, lst):
        """Recursively convert numpy types in a list."""
        if not lst:
            return []
        cleaned = []
        for item in lst:
            if isinstance(item, dict):
                cleaned.append({k: cls._to_native(v) for k, v in item.items()})
            elif isinstance(item, list):
                cleaned.append(cls._clean_list(item))
            else:
                cleaned.append(cls._to_native(item))
        return cleaned

    @classmethod
    def create(cls, *, user_id: int, image_path: str, predicted_age: int,
               confidence: float = 0.0, gender: str = 'Unknown',
               emotion: str = 'Unknown', race: str = 'Unknown',
               face_count: int = 1, processing_time_ms: float = 0.0,
               detector_used: str = 'opencv', ensemble_ages: list = None,
               age_std: float = 0.0) -> dict:
        """Insert a new prediction record and return it."""
        now = datetime.now(timezone.utc)
        doc = {
            'user_id': int(user_id),
            'image_path': image_path,
            'predicted_age': int(cls._to_native(predicted_age)),
            'confidence': round(float(cls._to_native(confidence)), 4),
            'gender': str(gender),
            'emotion': str(emotion),
            'race': str(race),
            'face_count': int(cls._to_native(face_count)),
            'processing_time_ms': round(float(cls._to_native(processing_time_ms)), 2),
            'detector_used': str(detector_used),
            'ensemble_ages': cls._clean_list(ensemble_ages or []),
            'age_std': round(float(cls._to_native(age_std)), 2),
            'created_at': now,
        }
        result = cls._col().insert_one(doc)
        doc['_id'] = str(result.inserted_id)
        return cls._serialize(doc)

    @classmethod
    def get_by_user(cls, user_id: int, limit: int = 100) -> list:
        """Fetch all prediction records for a user, newest first."""
        cursor = (
            cls._col()
            .find({'user_id': user_id})
            .sort('created_at', DESCENDING)
            .limit(limit)
        )
        return [cls._serialize(doc) for doc in cursor]

    @classmethod
    def get_by_id(cls, record_id: str) -> Optional[dict]:
        """Fetch a single prediction record by _id."""
        try:
            doc = cls._col().find_one({'_id': ObjectId(record_id)})
            return cls._serialize(doc) if doc else None
        except Exception:
            return None

    @classmethod
    def delete(cls, record_id: str, user_id: int) -> bool:
        """Delete a prediction record, ensuring it belongs to the user."""
        try:
            result = cls._col().delete_one({
                '_id': ObjectId(record_id),
                'user_id': user_id,
            })
            return result.deleted_count > 0
        except Exception:
            return False

    @classmethod
    def count(cls, user_id: int) -> int:
        return cls._col().count_documents({'user_id': user_id})

    @classmethod
    def count_since(cls, user_id: int, since: datetime) -> int:
        return cls._col().count_documents({
            'user_id': user_id,
            'created_at': {'$gte': since},
        })

    @classmethod
    def aggregate_stats(cls, user_id: int) -> dict:
        """Aggregate prediction statistics for analytics."""
        pipeline = [
            {'$match': {'user_id': user_id}},
            {'$group': {
                '_id': None,
                'avg_age': {'$avg': '$predicted_age'},
                'avg_confidence': {'$avg': '$confidence'},
            }},
        ]
        results = list(cls._col().aggregate(pipeline))
        if not results:
            return {'avg_age': 0, 'avg_confidence': 0}
        return {
            'avg_age': round(results[0].get('avg_age', 0), 1),
            'avg_confidence': round(results[0].get('avg_confidence', 0), 2),
        }

    @classmethod
    def gender_distribution(cls, user_id: int) -> list:
        """Get gender distribution counts."""
        pipeline = [
            {'$match': {'user_id': user_id}},
            {'$group': {'_id': '$gender', 'count': {'$sum': 1}}},
        ]
        return [{'gender': r['_id'], 'count': r['count']}
                for r in cls._col().aggregate(pipeline)]

    @classmethod
    def emotion_distribution(cls, user_id: int) -> list:
        """Get emotion distribution counts."""
        pipeline = [
            {'$match': {'user_id': user_id}},
            {'$group': {'_id': '$emotion', 'count': {'$sum': 1}}},
        ]
        return [{'emotion': r['_id'], 'count': r['count']}
                for r in cls._col().aggregate(pipeline)]

    @classmethod
    def daily_counts(cls, user_id: int, days: int = 7) -> list:
        """Get daily prediction counts for the past N days."""
        now = datetime.now(timezone.utc)
        results = []
        for i in range(days):
            day = now - timedelta(days=i)
            day_start = day.replace(hour=0, minute=0, second=0, microsecond=0)
            day_end = day_start + timedelta(days=1)
            count = cls._col().count_documents({
                'user_id': user_id,
                'created_at': {'$gte': day_start, '$lt': day_end},
            })
            results.append({
                'date': day_start.strftime('%Y-%m-%d'),
                'count': count,
            })
        return results

    # ── Per-user extended analytics ─────────────────────────────────
    @classmethod
    def detector_distribution(cls, user_id: int) -> list:
        """Group predictions by detector_used."""
        pipeline = [
            {'$match': {'user_id': user_id}},
            {'$group': {'_id': '$detector_used', 'count': {'$sum': 1}}},
        ]
        return [{'detector': r['_id'] or 'Unknown', 'count': r['count']}
                for r in cls._col().aggregate(pipeline)]

    @classmethod
    def race_distribution(cls, user_id: int) -> list:
        """Group predictions by race."""
        pipeline = [
            {'$match': {'user_id': user_id}},
            {'$group': {'_id': '$race', 'count': {'$sum': 1}}},
            {'$sort': {'count': -1}},
        ]
        return [{'race': r['_id'] or 'Unknown' if r['_id'] else 'Unknown', 'count': r['count']}
                for r in cls._col().aggregate(pipeline)]

    @classmethod
    def processing_time_stats(cls, user_id: int) -> dict:
        """Aggregate min/max/avg processing time for a user's predictions."""
        pipeline = [
            {'$match': {'user_id': user_id}},
            {'$group': {
                '_id': None,
                'avg': {'$avg': '$processing_time_ms'},
                'min': {'$min': '$processing_time_ms'},
                'max': {'$max': '$processing_time_ms'},
            }},
        ]
        results = list(cls._col().aggregate(pipeline))
        if not results:
            return {'avg': 0, 'min': 0, 'max': 0}
        r = results[0]
        return {
            'avg': round(r.get('avg') or 0, 1),
            'min': round(r.get('min') or 0, 1),
            'max': round(r.get('max') or 0, 1),
        }

    @classmethod
    def confidence_distribution(cls, user_id: int) -> list:
        """Bucket predictions by confidence ranges."""
        ranges = [
            ('0-20%', 0, 0.20),
            ('20-40%', 0.20, 0.40),
            ('40-60%', 0.40, 0.60),
            ('60-80%', 0.60, 0.80),
            ('80-100%', 0.80, 1.01),
        ]
        result = []
        col = cls._col()
        for label, lo, hi in ranges:
            count = col.count_documents({
                'user_id': user_id,
                'confidence': {'$gte': lo, '$lt': hi},
            })
            result.append({'range': label, 'count': count})
        return result

    # ── Admin / platform-wide aggregations ──────────────────────────
    @classmethod
    def platform_count(cls) -> int:
        return cls._col().count_documents({})

    @classmethod
    def platform_stats(cls) -> dict:
        """Platform-wide prediction stats (no user filter)."""
        pipeline = [
            {'$group': {
                '_id': None,
                'avg_age': {'$avg': '$predicted_age'},
                'avg_confidence': {'$avg': '$confidence'},
                'avg_processing_time_ms': {'$avg': '$processing_time_ms'},
                'total': {'$sum': 1},
            }},
        ]
        results = list(cls._col().aggregate(pipeline))
        if not results:
            return {
                'avg_age': 0,
                'avg_confidence': 0,
                'avg_processing_time_ms': 0,
                'total': 0,
            }
        r = results[0]
        return {
            'avg_age': round(r.get('avg_age') or 0, 1),
            'avg_confidence': round(r.get('avg_confidence') or 0, 2),
            'avg_processing_time_ms': round(r.get('avg_processing_time_ms') or 0, 1),
            'total': int(r.get('total') or 0),
        }

    @classmethod
    def platform_gender_distribution(cls) -> list:
        pipeline = [
            {'$group': {'_id': '$gender', 'count': {'$sum': 1}}},
        ]
        return [{'gender': r['_id'], 'count': r['count']}
                for r in cls._col().aggregate(pipeline)]

    @classmethod
    def platform_emotion_distribution(cls) -> list:
        pipeline = [
            {'$group': {'_id': '$emotion', 'count': {'$sum': 1}}},
        ]
        return [{'emotion': r['_id'], 'count': r['count']}
                for r in cls._col().aggregate(pipeline)]

    @classmethod
    def platform_detector_breakdown(cls) -> list:
        pipeline = [
            {'$group': {'_id': '$detector_used', 'count': {'$sum': 1}}},
        ]
        return [{'detector': r['_id'], 'count': r['count']}
                for r in cls._col().aggregate(pipeline)]

    @classmethod
    def platform_daily_counts(cls, days: int = 14) -> list:
        """Platform-wide daily prediction counts for the past N days."""
        now = datetime.now(timezone.utc)
        results = []
        for i in range(days):
            day = now - timedelta(days=i)
            day_start = day.replace(hour=0, minute=0, second=0, microsecond=0)
            day_end = day_start + timedelta(days=1)
            count = cls._col().count_documents({
                'created_at': {'$gte': day_start, '$lt': day_end},
            })
            results.append({
                'date': day_start.strftime('%Y-%m-%d'),
                'count': count,
            })
        results.reverse()  # oldest → newest for charting
        return results

    @staticmethod
    def _serialize(doc: dict) -> Optional[dict]:
        if doc is None:
            return None
        return {
            'id': str(doc['_id']),
            'user_id': doc.get('user_id'),
            'image_path': doc.get('image_path', ''),
            'predicted_age': doc.get('predicted_age', 0),
            'confidence': doc.get('confidence', 0.0),
            'gender': doc.get('gender', 'Unknown'),
            'emotion': doc.get('emotion', 'Unknown'),
            'race': doc.get('race', 'Unknown'),
            'face_count': doc.get('face_count', 1),
            'processing_time_ms': doc.get('processing_time_ms', 0.0),
            'detector_used': doc.get('detector_used', 'opencv'),
            'ensemble_ages': doc.get('ensemble_ages', []),
            'age_std': doc.get('age_std', 0.0),
            'created_at': _iso_utc(doc.get('created_at')),
        }


class MongoProgressionManager:
    """CRUD operations for progression records in MongoDB."""

    COLLECTION = 'progressions'

    @classmethod
    def _col(cls):
        return MongoDB.get_collection(cls.COLLECTION)

    @classmethod
    def create(cls, *, user_id: int, original_image_path: str,
               progressed_image_path: str = '', current_age: int = 0,
               target_age: int = 0, model_used: str = 'StyleGAN3',
               processing_time_ms: float = 0.0, gender: str = 'Unknown',
               pipeline_steps: list = None, aging_insights: list = None) -> dict:
        """Insert a new progression record and return it."""
        now = datetime.now(timezone.utc)
        doc = {
            'user_id': user_id,
            'original_image_path': original_image_path,
            'progressed_image_path': progressed_image_path,
            'current_age': current_age,
            'target_age': target_age,
            'model_used': model_used,
            'processing_time_ms': round(processing_time_ms, 2),
            'gender': gender,
            'pipeline_steps': pipeline_steps or [],
            'aging_insights': aging_insights or [],
            'created_at': now,
        }
        result = cls._col().insert_one(doc)
        doc['_id'] = str(result.inserted_id)
        return cls._serialize(doc)

    @classmethod
    def get_by_user(cls, user_id: int, limit: int = 100) -> list:
        """Fetch all progression records for a user, newest first."""
        cursor = (
            cls._col()
            .find({'user_id': user_id})
            .sort('created_at', DESCENDING)
            .limit(limit)
        )
        return [cls._serialize(doc) for doc in cursor]

    @classmethod
    def get_by_id(cls, record_id: str) -> Optional[dict]:
        """Fetch a single progression record by _id."""
        try:
            doc = cls._col().find_one({'_id': ObjectId(record_id)})
            return cls._serialize(doc) if doc else None
        except Exception:
            return None

    @classmethod
    def delete(cls, record_id: str, user_id: int) -> bool:
        """Delete a progression record, ensuring it belongs to the user."""
        try:
            result = cls._col().delete_one({
                '_id': ObjectId(record_id),
                'user_id': user_id,
            })
            return result.deleted_count > 0
        except Exception:
            return False

    @classmethod
    def count(cls, user_id: int) -> int:
        return cls._col().count_documents({'user_id': user_id})

    @classmethod
    def count_since(cls, user_id: int, since: datetime) -> int:
        return cls._col().count_documents({
            'user_id': user_id,
            'created_at': {'$gte': since},
        })

    @classmethod
    def update_progressed_image(cls, record_id: str, progressed_image_path: str):
        """Update the progressed image path after processing."""
        try:
            cls._col().update_one(
                {'_id': ObjectId(record_id)},
                {'$set': {'progressed_image_path': progressed_image_path}},
            )
        except Exception:
            pass

    # ── Per-user extended analytics ─────────────────────────────────
    @classmethod
    def model_distribution(cls, user_id: int) -> list:
        """Group progressions by model_used."""
        pipeline = [
            {'$match': {'user_id': user_id}},
            {'$group': {'_id': '$model_used', 'count': {'$sum': 1}}},
        ]
        return [{'model': r['_id'] or 'Unknown', 'count': r['count']}
                for r in cls._col().aggregate(pipeline)]

    @classmethod
    def processing_time_stats(cls, user_id: int) -> dict:
        """Aggregate min/max/avg processing time for a user's progressions."""
        pipeline = [
            {'$match': {'user_id': user_id}},
            {'$group': {
                '_id': None,
                'avg': {'$avg': '$processing_time_ms'},
                'min': {'$min': '$processing_time_ms'},
                'max': {'$max': '$processing_time_ms'},
            }},
        ]
        results = list(cls._col().aggregate(pipeline))
        if not results:
            return {'avg': 0, 'min': 0, 'max': 0}
        r = results[0]
        return {
            'avg': round(r.get('avg') or 0, 1),
            'min': round(r.get('min') or 0, 1),
            'max': round(r.get('max') or 0, 1),
        }

    @classmethod
    def model_performance(cls, user_id: int) -> list:
        """Per-model performance stats: count, avg processing time, avg age gap."""
        pipeline = [
            {'$match': {'user_id': user_id}},
            {'$group': {
                '_id': '$model_used',
                'count': {'$sum': 1},
                'avg_time_ms': {'$avg': '$processing_time_ms'},
                'min_time_ms': {'$min': '$processing_time_ms'},
                'max_time_ms': {'$max': '$processing_time_ms'},
                'avg_age_gap': {'$avg': {'$subtract': ['$target_age', '$current_age']}},
            }},
            {'$sort': {'count': -1}},
        ]
        results = []
        for r in cls._col().aggregate(pipeline):
            results.append({
                'model': r['_id'] or 'Unknown',
                'count': r['count'],
                'avg_time_ms': round(r.get('avg_time_ms') or 0, 1),
                'min_time_ms': round(r.get('min_time_ms') or 0, 1),
                'max_time_ms': round(r.get('max_time_ms') or 0, 1),
                'avg_age_gap': round(r.get('avg_age_gap') or 0, 1),
            })
        return results

    @classmethod
    def age_transformation_stats(cls, user_id: int) -> dict:
        """Aggregate avg current_age and avg target_age for a user."""
        pipeline = [
            {'$match': {'user_id': user_id}},
            {'$group': {
                '_id': None,
                'avg_current': {'$avg': '$current_age'},
                'avg_target': {'$avg': '$target_age'},
            }},
        ]
        results = list(cls._col().aggregate(pipeline))
        if not results:
            return {'avg_current': 0, 'avg_target': 0}
        r = results[0]
        return {
            'avg_current': round(r.get('avg_current') or 0, 1),
            'avg_target': round(r.get('avg_target') or 0, 1),
        }

    # ── Admin / platform-wide aggregations ──────────────────────────
    @classmethod
    def platform_count(cls) -> int:
        return cls._col().count_documents({})

    @classmethod
    def platform_stats(cls) -> dict:
        """Platform-wide progression stats (no user filter)."""
        pipeline = [
            {'$group': {
                '_id': None,
                'avg_processing_time_ms': {'$avg': '$processing_time_ms'},
                'avg_target_age': {'$avg': '$target_age'},
                'total': {'$sum': 1},
            }},
        ]
        results = list(cls._col().aggregate(pipeline))
        if not results:
            return {
                'avg_processing_time_ms': 0,
                'avg_target_age': 0,
                'total': 0,
            }
        r = results[0]
        return {
            'avg_processing_time_ms': round(r.get('avg_processing_time_ms') or 0, 1),
            'avg_target_age': round(r.get('avg_target_age') or 0, 1),
            'total': int(r.get('total') or 0),
        }

    @classmethod
    def platform_model_breakdown(cls) -> list:
        pipeline = [
            {'$group': {'_id': '$model_used', 'count': {'$sum': 1}}},
        ]
        return [{'model': r['_id'], 'count': r['count']}
                for r in cls._col().aggregate(pipeline)]

    @classmethod
    def platform_daily_counts(cls, days: int = 14) -> list:
        now = datetime.now(timezone.utc)
        results = []
        for i in range(days):
            day = now - timedelta(days=i)
            day_start = day.replace(hour=0, minute=0, second=0, microsecond=0)
            day_end = day_start + timedelta(days=1)
            count = cls._col().count_documents({
                'created_at': {'$gte': day_start, '$lt': day_end},
            })
            results.append({
                'date': day_start.strftime('%Y-%m-%d'),
                'count': count,
            })
        results.reverse()
        return results

    @staticmethod
    def _serialize(doc: dict) -> Optional[dict]:
        if doc is None:
            return None
        return {
            'id': str(doc['_id']),
            'user_id': doc.get('user_id'),
            'original_image_path': doc.get('original_image_path', ''),
            'progressed_image_path': doc.get('progressed_image_path', ''),
            'current_age': doc.get('current_age', 0),
            'target_age': doc.get('target_age', 0),
            'model_used': doc.get('model_used', ''),
            'processing_time_ms': doc.get('processing_time_ms', 0.0),
            'gender': doc.get('gender', 'Unknown'),
            'pipeline_steps': doc.get('pipeline_steps', []),
            'aging_insights': doc.get('aging_insights', []),
            'created_at': _iso_utc(doc.get('created_at')),
        }


class MongoUserSettingsManager:
    """CRUD operations for user settings in MongoDB."""

    COLLECTION = 'user_settings'

    DEFAULTS = {
        'theme': 'dark',
        'default_model': 'DeepFace v3',
        'notifications_enabled': True,
        'auto_detect_faces': True,
        'save_to_history': True,
        'high_accuracy_mode': False,
        'show_confidence': True,
        'language': 'English',
        'timezone': 'Asia/Kolkata',
    }

    @classmethod
    def _col(cls):
        return MongoDB.get_collection(cls.COLLECTION)

    @classmethod
    def get_or_create(cls, user_id: int) -> dict:
        """Get user settings, creating defaults if they don't exist."""
        doc = cls._col().find_one({'user_id': user_id})
        if doc:
            return cls._serialize(doc)

        # Create default settings
        now = datetime.now(timezone.utc)
        new_doc = {
            'user_id': user_id,
            **cls.DEFAULTS,
            'created_at': now,
            'updated_at': now,
        }
        result = cls._col().insert_one(new_doc)
        new_doc['_id'] = str(result.inserted_id)
        return cls._serialize(new_doc)

    @classmethod
    def update(cls, user_id: int, **fields) -> dict:
        """Update user settings. Only updates allowed fields."""
        allowed = set(cls.DEFAULTS.keys())
        updates = {k: v for k, v in fields.items() if k in allowed}
        if not updates:
            return cls.get_or_create(user_id)

        updates['updated_at'] = datetime.now(timezone.utc)

        cls._col().update_one(
            {'user_id': user_id},
            {'$set': updates},
            upsert=True,
        )
        return cls.get_or_create(user_id)

    @classmethod
    def delete(cls, user_id: int):
        cls._col().delete_one({'user_id': user_id})

    @staticmethod
    def _serialize(doc: dict) -> Optional[dict]:
        if doc is None:
            return None
        return {
            'id': str(doc['_id']),
            'user_id': doc.get('user_id'),
            'theme': doc.get('theme', 'dark'),
            'default_model': doc.get('default_model', 'DeepFace v3'),
            'notifications_enabled': doc.get('notifications_enabled', True),
            'auto_detect_faces': doc.get('auto_detect_faces', True),
            'save_to_history': doc.get('save_to_history', True),
            'high_accuracy_mode': doc.get('high_accuracy_mode', False),
            'show_confidence': doc.get('show_confidence', True),
            'language': doc.get('language', 'English'),
            'timezone': doc.get('timezone', 'Asia/Kolkata'),
            'updated_at': _iso_utc(doc.get('updated_at')),
        }


class MongoBatchPredictionManager:
    """CRUD operations for batch prediction records in MongoDB.

    A batch document aggregates the per-image results of a single
    multi-upload run. Each item in `results` is either a successful
    prediction (one entry per detected face) or an `error` payload
    when that file failed (size, decode, no face detected, etc.).
    """

    COLLECTION = 'batch_predictions'

    @classmethod
    def _col(cls):
        return MongoDB.get_collection(cls.COLLECTION)

    @classmethod
    def create(cls, *, user_id: int, total_images: int, total_faces: int,
               results: list, processing_time_ms: float = 0.0) -> dict:
        """Insert a new batch prediction record and return it."""
        now = datetime.now(timezone.utc)
        doc = {
            'user_id': int(user_id),
            'total_images': int(total_images),
            'total_faces': int(total_faces),
            'results': results or [],
            'processing_time_ms': round(float(processing_time_ms), 2),
            'created_at': now,
        }
        result = cls._col().insert_one(doc)
        doc['_id'] = str(result.inserted_id)
        return cls._serialize(doc)

    @classmethod
    def get_by_user(cls, user_id: int, limit: int = 50) -> list:
        """Fetch all batch records for a user, newest first."""
        cursor = (
            cls._col()
            .find({'user_id': user_id})
            .sort('created_at', DESCENDING)
            .limit(limit)
        )
        return [cls._serialize(doc) for doc in cursor]

    @classmethod
    def get_by_id(cls, batch_id: str, user_id: int) -> Optional[dict]:
        """Fetch a single batch record, ensuring it belongs to the user."""
        try:
            doc = cls._col().find_one({
                '_id': ObjectId(batch_id),
                'user_id': user_id,
            })
            return cls._serialize(doc) if doc else None
        except Exception:
            return None

    @classmethod
    def delete(cls, batch_id: str, user_id: int) -> bool:
        """Delete a batch record, ensuring it belongs to the user."""
        try:
            result = cls._col().delete_one({
                '_id': ObjectId(batch_id),
                'user_id': user_id,
            })
            return result.deleted_count > 0
        except Exception:
            return False

    @classmethod
    def count(cls, user_id: int) -> int:
        return cls._col().count_documents({'user_id': user_id})

    @staticmethod
    def _serialize(doc: dict) -> Optional[dict]:
        if doc is None:
            return None
        return {
            'id': str(doc['_id']),
            'user_id': doc.get('user_id'),
            'total_images': doc.get('total_images', 0),
            'total_faces': doc.get('total_faces', 0),
            'results': doc.get('results', []),
            'processing_time_ms': doc.get('processing_time_ms', 0.0),
            'created_at': _iso_utc(doc.get('created_at')),
        }
