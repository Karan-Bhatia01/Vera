"""
persistence.py
==============
Persist a finished ML run: each fitted model goes to GridFS as a pickle, and a
single summary document (metrics, best model, feature plan, SHAP) goes to the
`ml_results` collection.
"""

from __future__ import annotations

from typing import Any

from src.logger import logging


def save_results(
    db,
    filename: str,
    target_column: str,
    results: dict[str, Any],
    problem_type: str,
    feature_plan: dict,
    rank_metric: str,
    best_model: str,
) -> str:
    """No-op: ML model pickle files and ml_results document are no longer saved to MongoDB."""
    logging.info("ML run complete for '%s' (storage to MongoDB disabled).", filename)
    return ""
