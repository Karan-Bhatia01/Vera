"""
persistence.py
==============
Persist a finished ML run: each fitted model goes to GridFS as a pickle, and a
single summary document (metrics, best model, feature plan, SHAP) goes to the
`ml_results` collection.
"""

from __future__ import annotations

import pickle
from typing import Any

import gridfs

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
    shap_data: dict | None = None,
) -> str:
    """Store models + metrics. Returns the ml_results document id as a string."""
    fs = gridfs.GridFS(db)
    col = db["ml_results"]

    metrics_only: dict[str, Any] = {}
    model_gridfs_ids: dict[str, str] = {}

    for name, data in results.items():
        gridfs_id = fs.put(
            pickle.dumps(data["model_object"]),
            filename=f"{filename}__{name}.pkl",
            metadata={"source_file": filename, "model_name": name, "problem_type": problem_type},
        )
        model_gridfs_ids[name] = str(gridfs_id)
        metrics_only[name] = {k: v for k, v in data["metrics"].items() if k != "confusion_matrix"}
        metrics_only[name]["confusion_matrix"] = data["metrics"].get("confusion_matrix")
        metrics_only[name]["feature_importance"] = data.get("feature_importance")

    doc = {
        "filename":         filename,
        "target_column":    target_column,
        "problem_type":     problem_type,
        "feature_plan":     feature_plan,
        "metrics":          metrics_only,
        "model_gridfs_ids": model_gridfs_ids,
        "best_model":       best_model,
        "rank_metric":      rank_metric,
        "shap":             shap_data or {},
    }
    doc_id = col.insert_one(doc).inserted_id
    logging.info("Saved ML results to MongoDB. doc_id=%s", doc_id)
    return str(doc_id)
