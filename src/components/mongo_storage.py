"""
mongo_storage.py
================
MongoDB persistence for dataset stats + AI insights, and for user accounts.
"""
from __future__ import annotations

import re
import sys
from datetime import datetime, timezone
from typing import Any

from src.logger import logging
from src.exception import CustomException
from src.core.connections import get_db


def store_dataset_insights(
    filename: str,
    analysis: dict[str, Any],
    ai_insights: dict[str, Any],
    unique: dict[str, Any] | None = None,
) -> str:
    """Store dataset stats and AI insights in MongoDB."""
    try:
        doc = {
            "filename": filename,
            "analysis": analysis,
            "ai_insights": ai_insights,
            "unique": unique or {},
            "updated_at": datetime.now(timezone.utc),
        }
        # Upsert the document based on filename
        result = get_db()["dataset_insights"].update_one(
            {"filename": filename}, {"$set": doc}, upsert=True
        )
        logging.info("Dataset insights stored for '%s'.", filename)
        return str(result.upserted_id) if result.upserted_id else "updated"
    except Exception as e:
        logging.error("Failed to store dataset insights for '%s': %s", filename, e)
        return ""


def get_dataset_insights(filename: str) -> dict[str, Any] | None:
    """Retrieve stored dataset insights from MongoDB."""
    try:
        doc = get_db()["dataset_insights"].find_one({"filename": filename})
        if doc and "_id" in doc:
            doc["_id"] = str(doc["_id"])
        return doc
    except Exception as e:
        logging.error("Failed to retrieve dataset insights for '%s': %s", filename, e)
        return None

def delete_dataset_insights(filename: str) -> None:
    """Remove stored stats + AI insights for a dataset."""
    try:
        get_db()["dataset_insights"].delete_many({"filename": filename})
    except Exception as e:
        raise CustomException(e, sys) from e


# ── User storage ──────────────────────────────────────────────────────────────

def create_user(email: str, password_hash: str) -> str:
    """Insert a new user. Returns inserted _id as string."""
    try:
        result = get_db()["users"].insert_one({
            "email": email,
            "password_hash": password_hash,
            "created_at": datetime.now(timezone.utc),
        })
        logging.info("User created: '%s'", email)
        return str(result.inserted_id)
    except Exception as e:
        raise CustomException(e, sys) from e


def get_user_by_email(email: str) -> dict[str, Any] | None:
    """Retrieve user by email, ignoring case (older accounts may be stored mixed-case)."""
    try:
        pattern = f"^{re.escape(email.strip())}$"
        doc = get_db()["users"].find_one({"email": {"$regex": pattern, "$options": "i"}})
        if doc:
            doc["_id"] = str(doc["_id"])
        return doc
    except Exception as e:
        raise CustomException(e, sys) from e


