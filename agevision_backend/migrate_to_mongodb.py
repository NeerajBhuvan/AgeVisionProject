#!/usr/bin/env python
"""
Migrate all application data from SQLite to MongoDB.
=====================================================

This script reads existing data from Django's SQLite database
(PredictionRecord, ProgressionRecord, UserSettings) and inserts
it into the corresponding MongoDB collections.

Usage:
    cd agevision_backend
    python manage.py shell < migrate_to_mongodb.py

    OR

    python migrate_to_mongodb.py   (with Django settings configured)
"""

import os
import sys
import json
from datetime import timezone

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'agevision_backend.settings')

import django
django.setup()

from django.contrib.auth.models import User
from agevision_api.models import PredictionRecord, ProgressionRecord, UserSettings
from agevision_api.mongodb import (
    MongoDB,
    MongoUserManager,
    MongoPredictionManager,
    MongoProgressionManager,
    MongoUserSettingsManager,
)


def migrate_users():
    """Ensure all Django users have a corresponding MongoDB user document."""
    print("\n--- Migrating Users ---")
    users = User.objects.all()
    created = 0
    skipped = 0

    for user in users:
        existing = MongoUserManager.get_by_django_id(user.id)
        if existing:
            print(f"  [SKIP] User '{user.username}' already exists in MongoDB")
            skipped += 1
            continue

        try:
            MongoUserManager.create_user(
                django_user_id=user.id,
                username=user.username,
                email=user.email,
                first_name=user.first_name,
                last_name=user.last_name,
            )
            print(f"  [OK]   User '{user.username}' migrated to MongoDB")
            created += 1
        except Exception as e:
            print(f"  [ERR]  User '{user.username}': {e}")

    print(f"  Users: {created} created, {skipped} skipped")


def migrate_predictions():
    """Migrate PredictionRecord from SQLite to MongoDB."""
    print("\n--- Migrating Predictions ---")
    records = PredictionRecord.objects.all()
    created = 0

    for rec in records:
        try:
            image_path = str(rec.image) if rec.image else ''

            MongoPredictionManager.create(
                user_id=rec.user_id,
                image_path=image_path,
                predicted_age=rec.predicted_age,
                confidence=rec.confidence,
                gender=rec.gender,
                emotion=rec.emotion,
                race=rec.race,
                face_count=rec.face_count,
                processing_time_ms=rec.processing_time_ms,
                detector_used=rec.detector_used,
                ensemble_ages=rec.ensemble_ages or [],
                age_std=rec.age_std,
            )
            created += 1
        except Exception as e:
            print(f"  [ERR]  Prediction #{rec.id}: {e}")

    print(f"  Predictions: {created} migrated out of {records.count()}")


def migrate_progressions():
    """Migrate ProgressionRecord from SQLite to MongoDB."""
    print("\n--- Migrating Progressions ---")
    records = ProgressionRecord.objects.all()
    created = 0

    for rec in records:
        try:
            original_path = str(rec.original_image) if rec.original_image else ''
            progressed_path = str(rec.progressed_image) if rec.progressed_image else ''

            # Parse JSON fields
            try:
                pipeline_steps = json.loads(rec.pipeline_steps) if rec.pipeline_steps else []
            except (json.JSONDecodeError, TypeError):
                pipeline_steps = []

            try:
                aging_insights = json.loads(rec.aging_insights) if rec.aging_insights else []
            except (json.JSONDecodeError, TypeError):
                aging_insights = []

            MongoProgressionManager.create(
                user_id=rec.user_id,
                original_image_path=original_path,
                progressed_image_path=progressed_path,
                current_age=rec.current_age,
                target_age=rec.target_age,
                model_used=rec.model_used,
                processing_time_ms=rec.processing_time_ms,
                gender=rec.gender,
                pipeline_steps=pipeline_steps,
                aging_insights=aging_insights,
            )
            created += 1
        except Exception as e:
            print(f"  [ERR]  Progression #{rec.id}: {e}")

    print(f"  Progressions: {created} migrated out of {records.count()}")


def migrate_user_settings():
    """Migrate UserSettings from SQLite to MongoDB."""
    print("\n--- Migrating User Settings ---")
    records = UserSettings.objects.all()
    created = 0

    for rec in records:
        try:
            MongoUserSettingsManager.update(
                user_id=rec.user_id,
                theme=rec.theme,
                default_model=rec.default_model,
                notifications_enabled=rec.notifications_enabled,
                auto_detect_faces=rec.auto_detect_faces,
                save_to_history=rec.save_to_history,
                high_accuracy_mode=rec.high_accuracy_mode,
                show_confidence=rec.show_confidence,
                language=rec.language,
            )
            created += 1
        except Exception as e:
            print(f"  [ERR]  Settings for user #{rec.user_id}: {e}")

    print(f"  User Settings: {created} migrated out of {records.count()}")


def main():
    print("=" * 60)
    print("  AgeVision: SQLite -> MongoDB Data Migration")
    print("=" * 60)

    # Ensure MongoDB indexes exist
    try:
        MongoDB.ensure_indexes()
        print("\n[OK] MongoDB indexes created/verified")
    except Exception as e:
        print(f"\n[ERR] Could not connect to MongoDB: {e}")
        print("Make sure MongoDB is running on localhost:27017")
        sys.exit(1)

    migrate_users()
    migrate_predictions()
    migrate_progressions()
    migrate_user_settings()

    print("\n" + "=" * 60)
    print("  Migration complete!")
    print("=" * 60)
    print("\nAll application data is now in MongoDB (agevision_db).")
    print("Django's SQLite is kept only for auth User, admin, and sessions.")
    print()


if __name__ == '__main__':
    main()
