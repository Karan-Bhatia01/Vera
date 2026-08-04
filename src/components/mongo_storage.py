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

from pymongo import MongoClient
from src.logger import logging
from src.exception import CustomException

_MONGO_URL = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
_DB_NAME   = "clarityAI_database"


_client: MongoClient | None = None


def get_db():
    """Return the shared database handle.

    A single MongoClient is reused across the process (it is internally
    pooled and thread-safe). Opening a new client per call leaked sockets
    and monitor threads, and — with no server-selection timeout — let
    requests hang indefinitely when Mongo was unreachable. The timeout
    makes those calls fail fast instead.
    """
    global _client
    if _client is None:
        _client = MongoClient(_MONGO_URL, serverSelectionTimeoutMS=3000)
    return _client[_DB_NAME]


def store_dataset_insights(
    filename: str,
    analysis: dict[str, Any],
    ai_insights: dict[str, Any],
    unique: dict[str, Any] | None = None,
) -> str:
    """No-op: Dataset insights are no longer stored in MongoDB."""
    logging.info("Dataset insights generated for '%s' (storage disabled).", filename)
    return ""



def get_dataset_insights(filename: str) -> dict[str, Any] | None:
    """No-op: Always returns None as dataset insights are not stored in MongoDB."""
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

