"""
MongoDB models for the age_progression app.
=============================================
Uses PyMongo directly (not Django ORM) to store progression records
in MongoDB alongside the existing agevision_db.
"""

import logging
from datetime import datetime, timezone

from pymongo import MongoClient, ASCENDING, DESCENDING
from bson import ObjectId
from django.conf import settings

logger = logging.getLogger(__name__)


class ProgressionDB:
    """
    Manages the ``progressions`` collection in MongoDB.

    Uses the same MongoClient singleton configuration as the rest of the
    project (``settings.MONGODB_CONFIG``).
    """

    _client = None
    _db = None

    # ------------------------------------------------------------------
    #  Connection helpers
    # ------------------------------------------------------------------

    @classmethod
    def _get_client(cls):
        if cls._client is None:
            cfg = settings.MONGODB_CONFIG
            cls._client = MongoClient(
                host=cfg.get('HOST', 'localhost'),
                port=cfg.get('PORT', 27017),
                serverSelectionTimeoutMS=2000,
                connectTimeoutMS=2000,
                socketTimeoutMS=2000,
            )
        return cls._client

    @classmethod
    def _get_db(cls):
        if cls._db is None:
            cfg = settings.MONGODB_CONFIG
            db_name = cfg.get('NAME', 'agevision_db')
            cls._db = cls._get_client()[db_name]
        return cls._db

    @classmethod
    def collection(cls):
        """Return the ``progressions`` collection."""
        return cls._get_db()['progressions']

    @classmethod
    def ensure_indexes(cls):
        """Create indexes (idempotent)."""
        try:
            col = cls.collection()
            col.create_index([('user_id', ASCENDING)])
            col.create_index([('created_at', DESCENDING)])
            col.create_index([('status', ASCENDING)])
            logger.info('age_progression: MongoDB indexes ensured')
        except Exception as exc:
            logger.warning('age_progression: Could not create indexes – %s', exc)

    # ------------------------------------------------------------------
    #  CRUD operations
    # ------------------------------------------------------------------

    @classmethod
    def insert_record(
        cls,
        user_id: str,
        original_image_path: str,
        progressed_image_path: str,
        comparison_image_path: str,
        current_age: int,
        target_age: int,
        processing_time: float,
        method_used: str,
        extra: dict | None = None,
    ) -> str:
        """
        Insert a new progression record.

        Returns
        -------
        str  – the hex string of the inserted ``_id``.
        """
        doc = {
            'user_id': str(user_id),
            'original_image_path': original_image_path,
            'progressed_image_path': progressed_image_path,
            'comparison_image_path': comparison_image_path,
            'current_age': int(current_age),
            'target_age': int(target_age),
            'processing_time': round(float(processing_time), 3),
            'method_used': method_used,
            'created_at': datetime.now(timezone.utc),
            'status': 'success',
        }
        if extra:
            doc.update(extra)

        result = cls.collection().insert_one(doc)
        logger.info('Inserted progression record %s', result.inserted_id)
        return str(result.inserted_id)

    @classmethod
    def get_record(cls, record_id: str) -> dict | None:
        """Fetch a single record by its ``_id``."""
        try:
            doc = cls.collection().find_one({'_id': ObjectId(record_id)})
            if doc:
                doc['_id'] = str(doc['_id'])
            return doc
        except Exception:
            return None

    @classmethod
    def get_user_records(cls, user_id: str, limit: int = 50) -> list[dict]:
        """Fetch recent progression records for a user."""
        cursor = (
            cls.collection()
            .find({'user_id': str(user_id)})
            .sort('created_at', DESCENDING)
            .limit(limit)
        )
        records = []
        for doc in cursor:
            doc['_id'] = str(doc['_id'])
            records.append(doc)
        return records

    @classmethod
    def delete_record(cls, record_id: str) -> bool:
        """Delete a progression record by ``_id``."""
        try:
            result = cls.collection().delete_one({'_id': ObjectId(record_id)})
            return result.deleted_count > 0
        except Exception:
            return False

    @classmethod
    def get_stats(cls, user_id: str | None = None) -> dict:
        """Aggregate statistics (total count, avg processing time, etc.)."""
        match = {}
        if user_id:
            match['user_id'] = str(user_id)

        pipeline = [
            {'$match': match} if match else {'$match': {}},
            {
                '$group': {
                    '_id': None,
                    'total': {'$sum': 1},
                    'avg_processing_time': {'$avg': '$processing_time'},
                    'methods': {'$push': '$method_used'},
                }
            },
        ]
        results = list(cls.collection().aggregate(pipeline))
        if not results:
            return {'total': 0, 'avg_processing_time': 0, 'methods': {}}

        data = results[0]
        # Count method distribution
        method_counts = {}
        for m in data.get('methods', []):
            method_counts[m] = method_counts.get(m, 0) + 1

        return {
            'total': data['total'],
            'avg_processing_time': round(data.get('avg_processing_time', 0), 2),
            'methods': method_counts,
        }

    @classmethod
    def close(cls):
        if cls._client:
            cls._client.close()
            cls._client = None
            cls._db = None
