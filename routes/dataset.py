from flask import Blueprint, jsonify, request
from services.auth_decorator import require_auth
from src.components.mongo_storage import (
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
        owned = ingestion.get_all_filenames(owner_email=owner_email)

        datasets = [
            {
                "filename":  fname,
                "shape":     None,
                "stored_at": "",
                "summary":   "",
                "analyzed":  False,
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