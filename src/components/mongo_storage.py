"""
mongo_storage.py
================
Handles automatic storage of:
  - Dataset metadata (shape, dtypes, null info, etc.)
  - AI-generated insights (from AnalysisExplainer)
  - Chart-based insights
  - User-selected target column (for EDA/ML)

All stored automatically after upload + analysis — no extra user action required.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from typing import Any

from src.logger import logging
from src.exception import CustomException
from src.core.connections import get_db

_DB_NAME   = "clarityAI_database"


def store_dataset_insights(
    filename: str,
    analysis: dict[str, Any],
    ai_insights: dict[str, Any],
    unique: dict[str, Any] | None = None,
) -> str:
    """Store dataset stats and AI insights in MongoDB."""
    try:
        db = get_db()
        collection = db["dataset_insights"]
        
        doc = {
            "filename": filename,
            "analysis": analysis,
            "ai_insights": ai_insights,
            "unique": unique or {},
            "updated_at": datetime.now(timezone.utc),
        }
        
        # Upsert the document based on filename
        result = collection.update_one(
            {"filename": filename},
            {"$set": doc},
            upsert=True
        )
        
        logging.info("Dataset insights stored for '%s'.", filename)
        return str(result.upserted_id) if result.upserted_id else "updated"
    except Exception as e:
        logging.error("Failed to store dataset insights for '%s': %s", filename, e)
        return ""


def get_dataset_insights(filename: str) -> dict[str, Any] | None:
    """Retrieve stored dataset insights from MongoDB."""
    try:
        db = get_db()
        collection = db["dataset_insights"]
        doc = collection.find_one({"filename": filename})
        if doc and "_id" in doc:
            doc["_id"] = str(doc["_id"])
        return doc
    except Exception as e:
        logging.error("Failed to retrieve dataset insights for '%s': %s", filename, e)
        return None



# ── User storage (merged from auth_storage.py) ────────────────────────────────

def create_user(email: str, password_hash: str) -> str:
    """Insert a new user. Returns inserted _id as string."""
    try:
        db = get_db()
        result = db["users"].insert_one({
            "email": email,
            "password_hash": password_hash,
            "created_at": datetime.now(timezone.utc),
        })
        logging.info("User created: '%s'", email)
        return str(result.inserted_id)
    except Exception as e:
        raise CustomException(e, sys) from e


def get_user_by_email(email: str) -> dict[str, Any] | None:
    """Retrieve user by email. Returns None if not found."""
    try:
        db = get_db()
        doc = db["users"].find_one({"email": email})
        if doc:
            doc["_id"] = str(doc["_id"])
        return doc
    except Exception as e:
        raise CustomException(e, sys) from e

