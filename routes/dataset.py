from flask import Blueprint, jsonify, request
from services.auth_decorator import require_auth
from src.components.mongo_storage import (
    list_stored_datasets,
    get_dataset_insights as get_insights_from_db,
)
from src.components.data_ingestion import DataIngestion

dataset_bp = Blueprint("dataset", __name__)

@dataset_bp.route("/api/stored_datasets", methods=["GET"])
@require_auth
def stored_datasets():
    try:
        owner_email = request.user["email"]
        ingestion = DataIngestion()
        # Every CSV the user has uploaded — this is the source of truth, so a
        # freshly uploaded file shows up immediately, before it's analyzed.
        owned = ingestion.get_all_filenames(owner_email=owner_email)

        # Enrich with stored-insights metadata (shape/summary/date) where the
        # dataset has already been analyzed; unanalyzed files still appear.
        insights_by_name = {d["filename"]: d for d in list_stored_datasets()}

        datasets = [
            {
                "filename":  fname,
                "shape":     insights_by_name.get(fname, {}).get("shape"),
                "stored_at": insights_by_name.get(fname, {}).get("stored_at", ""),
                "summary":   insights_by_name.get(fname, {}).get("summary", ""),
                "analyzed":  fname in insights_by_name,
            }
            for fname in owned
        ]
        return jsonify({"status": "success", "datasets": datasets}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@dataset_bp.route("/api/insights/<filename>", methods=["GET"])
@require_auth
def dataset_insights(filename):
    try:
        insights = get_insights_from_db(filename)
        return jsonify({"status": "success", "insights": insights}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500