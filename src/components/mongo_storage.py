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
import json
from datetime import datetime, timezone
from typing import Any

from pymongo import MongoClient, DESCENDING
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
    """
    Persist dataset metadata + AI insights to MongoDB.
    Collection: dataset_insights

    Returns the inserted document _id as a string.
    Called automatically from the /info route.
    """
    try:
        db  = get_db()
        col = db["dataset_insights"]

        # Preserve target_column if one was already set (e.g. by
        # set_target_column earlier in the same pipeline run) — this
        # function otherwise wipes the whole document on every call.
        existing = col.find_one({"filename": filename}, {"target_column": 1})
        existing_target_column = existing.get("target_column") if existing else None

        # Remove existing record for this filename so we always have fresh data
        col.delete_many({"filename": filename})

        # Serialise analysis — convert tuples/numpy types to plain Python
        safe_analysis = json.loads(
            json.dumps(analysis, default=_json_default)
        )

        doc = {
            "filename":    filename,
            "stored_at":   datetime.now(timezone.utc),
            "analysis":    safe_analysis,
            "ai_insights": ai_insights,
            "unique":      json.loads(json.dumps(unique or {}, default=_json_default)),
            "target_column": existing_target_column,
        }

        result = col.insert_one(doc)
        logging.info(
            "Dataset insights stored for '%s' → _id=%s", filename, result.inserted_id
        )
        return str(result.inserted_id)

    except Exception as e:
        raise CustomException(e, sys) from e


def store_chart_insight(
    filename: str,
    chart_title: str,
    insight: dict[str, Any],
) -> None:
    """
    Append a single chart's AI analysis to the dataset_insights document.
    Called from /analyse_chart after each chart is analysed.
    """
    try:
        db  = get_db()
        col = db["dataset_insights"]

        col.update_one(
            {"filename": filename},
            {
                "$set":  {f"chart_insights.{_safe_key(chart_title)}": insight},
                "$push": {"chart_titles": chart_title},
            },
            upsert=True,
        )
        logging.info("Chart insight stored: '%s' / '%s'", filename, chart_title)

    except Exception as e:
        raise CustomException(e, sys) from e


def get_dataset_insights(filename: str) -> dict[str, Any] | None:
    """Retrieve stored insights for a filename. Returns None if not found."""
    try:
        db  = get_db()
        col = db["dataset_insights"]
        doc = col.find_one({"filename": filename}, sort=[("stored_at", DESCENDING)])
        if doc:
            doc["_id"] = str(doc["_id"])
        return doc
    except Exception as e:
        raise CustomException(e, sys) from e


def list_stored_datasets() -> list[dict]:
    """
    Return a list of all datasets that have stored insights.
    Each item: {filename, stored_at, shape, summary}
    """
    try:
        db  = get_db()
        col = db["dataset_insights"]
        docs = col.find(
            {},
            {"filename": 1, "stored_at": 1, "analysis.shape": 1, "ai_insights.summary": 1},
            sort=[("stored_at", DESCENDING)],
        )
        results = []
        for doc in docs:
            results.append({
                "filename":   doc.get("filename", ""),
                "stored_at":  doc.get("stored_at", "").strftime("%Y-%m-%d %H:%M") if doc.get("stored_at") else "",
                "shape":      doc.get("analysis", {}).get("shape", [0, 0]),
                "summary":    doc.get("ai_insights", {}).get("summary", ""),
            })
        return results
    except Exception as e:
        raise CustomException(e, sys) from e


def set_target_column(filename: str, target_column: str) -> bool:
    """
    Persist the user's chosen target column for a dataset, ahead of
    running EDA/ML on it. Stored on the same dataset_insights document
    that already tracks analysis/ai_insights for this filename.

    Upserts: if no dataset_insights document exists yet for this filename
    (e.g. the pipeline hasn't run analysis on it before), a minimal
    document is created so the target column isn't lost. The Info job's
    eventual store_dataset_insights() call will fill in the rest later,
    without clobbering this field (store_dataset_insights replaces the
    whole document on each run, so callers should re-apply the target
    column after analysis completes if it needs to persist past that).

    Always returns True — upsert means there's no "not found" failure case.
    """
    try:
        db  = get_db()
        col = db["dataset_insights"]

        col.update_one(
            {"filename": filename},
            {
                "$set": {"target_column": target_column},
                "$setOnInsert": {
                    "filename": filename,
                    "stored_at": datetime.now(timezone.utc),
                    "analysis": {},
                    "ai_insights": {},
                    "unique": {},
                },
            },
            upsert=True,
        )

        logging.info("Target column for '%s' set to '%s'", filename, target_column)
        return True

    except Exception as e:
        raise CustomException(e, sys) from e


def get_target_column(filename: str) -> str | None:
    """Retrieve the previously-stored target column for a dataset, or None if unset."""
    try:
        db  = get_db()
        col = db["dataset_insights"]
        doc = col.find_one({"filename": filename}, {"target_column": 1})
        return doc.get("target_column") if doc else None

    except Exception as e:
        raise CustomException(e, sys) from e


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


# ── helpers ────────────────────────────────────────────────────────────────────

def _safe_key(s: str) -> str:
    """MongoDB field keys can't contain dots — replace with underscores."""
    return s.replace(".", "_").replace("$", "_")


def _json_default(obj):
    """Fallback serialiser for numpy / tuple / other non-JSON types."""
    import numpy as np
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, tuple):
        return list(obj)
    return str(obj)