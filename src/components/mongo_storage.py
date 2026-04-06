"""
mongo_storage.py
================
Handles automatic storage of:
  - Dataset metadata (shape, dtypes, null info, etc.)
  - AI-generated insights (from AnalysisExplainer)
  - Chart-based insights

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


def get_db():
    client = MongoClient(_MONGO_URL)
    return client[_DB_NAME]


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
