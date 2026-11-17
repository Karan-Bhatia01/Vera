from flask import Blueprint, jsonify, request

from services.auth_decorator import require_auth
from src.components.mongo_storage import get_dataset_insights, delete_dataset_insights
from src.components.data_ingestion import DataIngestion

dataset_bp = Blueprint("dataset", __name__)


@dataset_bp.route("/api/stored_datasets", methods=["GET"])
@require_auth
def stored_datasets():
    try:
        owned = DataIngestion().get_all_filenames(owner_email=request.user["email"])
        return jsonify({"status": "success", "datasets": [{"filename": f} for f in owned]}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@dataset_bp.route("/api/insights/<filename>", methods=["GET"])
@require_auth
def dataset_insights(filename):
    try:
        return jsonify({"status": "success", "insights": get_dataset_insights(filename)}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@dataset_bp.route("/api/datasets/<filename>", methods=["DELETE"])
@require_auth
def delete_dataset(filename):
    try:
        ingestion = DataIngestion()
        if not ingestion.delete_file(request.user["email"], filename):
            return jsonify({"status": "error", "message": "Dataset not found"}), 404
        # Insights are keyed by filename only, so keep them while another user still has this file.
        if not ingestion.has_file(filename):
            delete_dataset_insights(filename)
        return jsonify({"status": "success", "message": f"'{filename}' deleted"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
